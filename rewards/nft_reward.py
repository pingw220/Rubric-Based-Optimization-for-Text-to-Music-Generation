"""Rubric reward for DiffusionNFT on ACE-Step.

The rater runs in ``diffusionnft/server.py`` on separate GPUs; ``RaterServerClient`` sends it
each generated clip through a shared directory and receives one 0-10 score per
dimension of the seven-dimension rubric (``rubrics/general.yaml``). Scores are
normalized to [0, 1] and combined with the rubric weights into DiffusionNFT's
``(audios, prompts, metadata)`` reward. This module also holds the Music
Flamingo and Qwen2-Audio raters that ``diffusionnft/server.py`` loads.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from contextlib import nullcontext
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import soundfile as sf
import torch


LOGGER = logging.getLogger(__name__)

SCORE_KEYS = (
    "prompt_alignment",
    "style_instrumentation",
    "melody",
    "rhythm",
    "structure",
    "production_quality",
    "overall_musicality",
)

DEFAULT_DIMENSION_WEIGHTS = {
    "prompt_alignment": 0.25,
    "style_instrumentation": 0.15,
    "melody": 0.10,
    "rhythm": 0.10,
    "structure": 0.15,
    "production_quality": 0.15,
    "overall_musicality": 0.10,
}


def _plain_dict(value) -> dict:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def _normalize_weights(values: dict, expected: Iterable[str], label: str) -> dict[str, float]:
    keys = tuple(expected)
    unknown = set(values) - set(keys)
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")
    result = {key: float(values.get(key, 0.0)) for key in keys}
    if any(value < 0 for value in result.values()):
        raise ValueError(f"{label} weights must be non-negative")
    total = sum(result.values())
    if total <= 0:
        raise ValueError(f"At least one {label} weight must be positive")
    return {key: value / total for key, value in result.items()}


def _extract_json_object(text: str) -> Dict[str, object]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    # Qwen2-Audio sometimes emits a JSON object whose quote characters are
    # escaped even though the surrounding response is not a JSON string, e.g.
    # {\"prompt_alignment\": 7.5, ...}. Try a conservative unescaped variant
    # after the original candidate.
    expanded_candidates = []
    for candidate in candidates:
        expanded_candidates.append(candidate)
        if '\\"' in candidate or "\\n" in candidate or "\\t" in candidate:
            try:
                unescaped = candidate.encode("utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                unescaped = candidate.replace('\\"', '"')
            if unescaped != candidate:
                expanded_candidates.append(unescaped)
    for candidate in expanded_candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(candidate)
                if isinstance(value, dict):
                    return value
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    raise ValueError(f"Judge did not return a JSON object: {text!r}")


def _normalize_scores(scores: Dict[str, object]) -> Dict[str, float]:
    # Music Flamingo may capitalize the first JSON key even when the prompt
    # spells it exactly (observed as ``Prompt_alignment`` in single-call 7D
    # mode). A unique case-only match is semantically unambiguous, unlike a
    # misspelling or abbreviation, so canonicalize only that narrow variant.
    canonical_scores = dict(scores)
    for key in SCORE_KEYS:
        if key in canonical_scores:
            continue
        case_only_matches = [
            raw_key
            for raw_key in scores
            if isinstance(raw_key, str) and raw_key.lower() == key
        ]
        if len(case_only_matches) == 1:
            canonical_scores[key] = scores[case_only_matches[0]]

    missing = [key for key in SCORE_KEYS if key not in canonical_scores]
    if missing:
        raise ValueError(f"Judge response is missing score fields: {missing}")
    normalized = {}
    for key in SCORE_KEYS:
        value = float(canonical_scores[key])
        if not np.isfinite(value):
            raise ValueError(f"Judge score {key!r} is not finite: {value}")
        normalized[key] = max(0.0, min(10.0, value)) / 10.0
    return normalized


def _bare_number(text: str):
    """Return the float if the whole response is one bare number, else None.

    Deterministic decoding means a retry reproduces the same bare number, so
    for a one-dimension request the value is accepted rather than retried.
    Joint-mode responses never take this path; there a bare number stays an
    error because the dimension it belongs to would be ambiguous.
    """
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*\.?\s*", text or "")
    if match:
        return float(match.group(1))
    # A labelled single value such as "Overall musicality: 9.5" is equally
    # unambiguous for a one-dimension request.
    match = re.fullmatch(
        r"\s*[A-Za-z][^:{}\n]{0,60}:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*10)?\s*\.?\s*",
        text or "",
    )
    return float(match.group(1)) if match else None


def _single_dimension_value(scores: Dict[str, object], key: str, response: str) -> float:
    """Extract one score from a response to a one-dimension-only query.

    A judge may misspell the requested JSON key (for example,
    ``style_instruction`` for ``style_instrumentation``). The key name is not
    semantically ambiguous when the prompt requested exactly one dimension, so
    accept a sole numeric field while continuing to reject multi-field output
    that does not contain the requested key.
    """
    if key in scores:
        value = scores[key]
    elif len(scores) == 1:
        value = next(iter(scores.values()))
    else:
        raise ValueError(f"Judge response is missing {key!r}: {response!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Judge score {key!r} is not finite: {value}")
    return value


def _torch_dtype(name: str):
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name == "auto":
        return "auto"
    try:
        return aliases[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported judge dtype: {name!r}") from exc


def _load_7d_rubric(name: str):
    from rewards.rubric import load_rubric

    rubric = load_rubric(name)
    if tuple(rubric.keys) != SCORE_KEYS:
        raise ValueError(f"{name} keys do not match the seven reward dimensions")
    return rubric


class _BaseJudge:
    def __init__(self, model_id: str, device: str, dtype: str, max_new_tokens: int):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def score(self, prompt: str, audio_path: str) -> Dict[str, float]:
        raise NotImplementedError


class Qwen2AudioJudge(_BaseJudge):
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        independent_dimensions: bool = False,
        prompt_schema: str = "general",
    ):
        super().__init__(model_id, device, dtype, max_new_tokens)
        self.independent_dimensions = independent_dimensions
        self.prompt_schema = str(prompt_schema)
        self._prompt_rubric = _load_7d_rubric(self.prompt_schema)
        self._logged_score_count = 0

    def _load(self):
        if self._model is None:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=_torch_dtype(self.dtype),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
        return self._model, self._processor

    def _generate(self, model, processor, prompt: str, audio_path: str, audio) -> str:
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        sample_rate = getattr(processor.feature_extractor, "sampling_rate", 16000)
        inputs = processor(
            text=text,
            audio=[audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        generated = generated[:, inputs.input_ids.shape[-1]:]
        response = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    @torch.inference_mode()
    def score(self, prompt: str, audio_path: str) -> Dict[str, float]:
        import librosa

        model, processor = self._load()
        sample_rate = getattr(processor.feature_extractor, "sampling_rate", 16000)
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        if not self.independent_dimensions:
            judge_prompt = (
                self._prompt_rubric.qwen_system_prompt()
                + "\n\n"
                + self._prompt_rubric.qwen_user_prompt(prompt)
            )
            response = self._generate(model, processor, judge_prompt, audio_path, audio)
            try:
                return _normalize_scores(_extract_json_object(response))
            except ValueError:
                # One retry with a minimal correction instruction, per the
                # documented parsing protocol. Greedy decoding makes an
                # identical retry pointless; the appended correction changes
                # the prompt and usually recovers refusals.
                corrected = judge_prompt + (
                    "\n\nYour previous reply was not the requested JSON object. Do not apologize or explain. Return only the requested JSON object now."
                )
                response = self._generate(model, processor, corrected, audio_path, audio)
                try:
                    return _normalize_scores(_extract_json_object(response))
                except ValueError:
                    # Hard double refusal on this clip. Fall back to scoring it
                    # dimension by dimension with the same rater and schema,
                    # so the run survives without fabricating a score. Logged
                    # for the audit trail; affects well under 0.1% of clips.
                    LOGGER.warning(
                        "Qwen2-Audio joint mode refused twice; falling back to "
                        "per-dimension scoring for this clip"
                    )
                    return self._score_dimensions(model, processor, prompt, audio_path, audio)

        return self._score_dimensions(model, processor, prompt, audio_path, audio)

    def _score_dimensions(self, model, processor, prompt, audio_path, audio) -> Dict[str, float]:
        scores = {}
        for key in SCORE_KEYS:
            judge_prompt = self._prompt_rubric.flamingo_dimension_prompt(key, prompt)
            response = self._generate(
                model,
                processor,
                judge_prompt,
                audio_path,
                audio,
            )
            try:
                parsed = _extract_json_object(response)
            except ValueError:
                bare = _bare_number(response)
                if bare is not None:
                    parsed = {key: bare}
                else:
                    corrected = judge_prompt + (
                        "\n\nYour previous reply was not the requested JSON object. Do not apologize or explain. Return only the requested JSON object now."
                    )
                    response = self._generate(
                        model, processor, corrected, audio_path, audio
                    )
                    try:
                        parsed = _extract_json_object(response)
                    except ValueError:
                        bare = _bare_number(response)
                        if bare is None:
                            raise
                        parsed = {key: bare}
            value = _single_dimension_value(parsed, key, response)
            scores[key] = max(0.0, min(10.0, value)) / 10.0
        if self._logged_score_count < 3:
            LOGGER.info(
                "Qwen2-Audio independent 7D score %d: %s",
                self._logged_score_count + 1,
                {key: round(10.0 * scores[key], 3) for key in SCORE_KEYS},
            )
            self._logged_score_count += 1
        return scores


class MusicFlamingoJudge(_BaseJudge):
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        independent_dimensions: bool = True,
        prompt_schema: str = "general",
    ):
        super().__init__(model_id, device, dtype, max_new_tokens)
        self.independent_dimensions = independent_dimensions
        self.prompt_schema = str(prompt_schema)
        self._prompt_rubric = _load_7d_rubric(self.prompt_schema)
        self._logged_score_count = 0

    def _load(self):
        if self._model is None:
            from transformers import AutoProcessor
            if "2601" in self.model_id:
                from transformers import MusicFlamingoForConditionalGeneration as model_cls
            else:
                from transformers import AudioFlamingo3ForConditionalGeneration as model_cls

            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype=_torch_dtype(self.dtype),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
        return self._model, self._processor

    def _generate(
        self,
        model,
        processor,
        prompt: str,
        audio_path: str,
        *,
        system_prompt: str = "",
    ) -> str:
        model, processor = self._load()
        user_turn = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }
        conversation = [user_turn]
        if system_prompt:
            conversation.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
            )
        try:
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        except (ValueError, KeyError, TypeError):
            if not system_prompt:
                raise
            user_turn["content"][0]["text"] = f"{system_prompt}\n\n{prompt}"
            inputs = processor.apply_chat_template(
                [user_turn],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        generated = generated[:, inputs.input_ids.shape[-1]:]
        response = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    @torch.inference_mode()
    def score(self, prompt: str, audio_path: str) -> Dict[str, float]:
        model, processor = self._load()
        if not self.independent_dimensions:
            response = self._generate(
                model,
                processor,
                self._prompt_rubric.qwen_user_prompt(prompt),
                audio_path,
                system_prompt=self._prompt_rubric.qwen_system_prompt(),
            )
            return _normalize_scores(_extract_json_object(response))

        scores = {}
        for key in SCORE_KEYS:
            judge_prompt = self._prompt_rubric.flamingo_dimension_prompt(key, prompt)
            response = self._generate(
                model,
                processor,
                judge_prompt,
                audio_path,
                system_prompt=self._prompt_rubric.flamingo_system_prompt(),
            )
            parsed = _extract_json_object(response)
            value = _single_dimension_value(parsed, key, response)
            scores[key] = max(0.0, min(10.0, value)) / 10.0
        if self._logged_score_count < 3:
            LOGGER.info(
                "Music Flamingo independent 7D score %d: %s",
                self._logged_score_count + 1,
                {key: round(10.0 * scores[key], 3) for key in SCORE_KEYS},
            )
            self._logged_score_count += 1
        return scores


class RaterServerClient(_BaseJudge):
    """Client for ``diffusionnft/server.py``: exchanges clips and scores through a shared directory."""

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        queue_dir: str,
        timeout_seconds: float = 600.0,
        poll_seconds: float = 0.2,
        queue_name: str = "rater server",
        expected_prompt_mode: str = "",
    ):
        super().__init__(model_id, device, dtype, max_new_tokens)
        if not queue_dir:
            raise ValueError(f"{queue_name} queue requires a shared queue_dir")
        self.queue_dir = Path(queue_dir).expanduser().resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.queue_name = str(queue_name)
        self.expected_prompt_mode = str(expected_prompt_mode)
        if self.timeout_seconds <= 0 or self.poll_seconds <= 0:
            raise ValueError(f"{self.queue_name} queue timeout and poll interval must be positive")
        for name in ("audio", "pending", "responses"):
            (self.queue_dir / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)

    def score(self, prompt: str, audio_path: str) -> Dict[str, float]:
        ready_path = self.queue_dir / "READY.json"
        if not ready_path.exists():
            raise RuntimeError(
                f"{self.queue_name} queue server is not ready: {ready_path}"
            )
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        actual_prompt_mode = str(ready.get("prompt_mode", ""))
        if (
            self.expected_prompt_mode
            and actual_prompt_mode != self.expected_prompt_mode
        ):
            raise RuntimeError(
                f"{self.queue_name} prompt-mode mismatch: expected "
                f"{self.expected_prompt_mode!r}, server reports "
                f"{actual_prompt_mode!r}"
            )

        request_id = f"{int(time.time() * 1000)}-{os.getpid()}-{uuid.uuid4().hex}"
        shared_audio = self.queue_dir / "audio" / f"{request_id}.wav"
        request_path = self.queue_dir / "pending" / f"{request_id}.json"
        response_path = self.queue_dir / "responses" / f"{request_id}.json"
        shutil.copyfile(audio_path, shared_audio)
        self._atomic_json(
            request_path,
            {
                "request_id": request_id,
                "model_id": self.model_id,
                "prompt": str(prompt),
                "audio_path": str(shared_audio),
                "created_at": time.time(),
            },
        )

        deadline = time.monotonic() + self.timeout_seconds
        response_read_error: Exception | None = None
        try:
            while time.monotonic() < deadline:
                if response_path.exists():
                    try:
                        response = json.loads(response_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as error:
                        # A non-atomic test server or network filesystem can make
                        # the directory entry visible just before the complete
                        # response is readable. Keep polling within the existing
                        # timeout instead of turning that short race into a
                        # failed reward request.
                        response_read_error = error
                        time.sleep(self.poll_seconds)
                        continue
                    if response.get("error"):
                        raise RuntimeError(
                            f"{self.queue_name} queue request {request_id} failed: "
                            f"{response['error']}"
                        )
                    return _normalize_scores(response["scores"])
                time.sleep(self.poll_seconds)
            detail = (
                f"; last response read failed: {response_read_error}"
                if response_read_error is not None
                else ""
            )
            raise TimeoutError(
                f"{self.queue_name} queue request {request_id} exceeded "
                f"{self.timeout_seconds:.1f}s{detail}"
            )
        finally:
            response_path.unlink(missing_ok=True)
            shared_audio.unlink(missing_ok=True)


def _build_judge(provider: str, server_dir: str, timeout_seconds: float, poll_seconds: float, prompting: str):
    if provider != "server":
        raise ValueError(f"Unsupported rubric rater provider: {provider!r} (use 'server')")
    return RaterServerClient(
        model_id="",
        device="cpu",
        dtype="",
        max_new_tokens=0,
        queue_dir=server_dir,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        expected_prompt_mode=prompting,
    )


class RubricReward:
    """Thread-safe callable implementing DiffusionNFT's reward contract."""

    def __init__(self, device, config):
        cfg = _plain_dict(config)
        self.device = str(device)
        self.sample_rate = int(cfg.get("sample_rate", 48000))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 256))
        self.dtype = str(cfg.get("dtype", "bfloat16"))
        self.retries = int(cfg.get("retries", 1))
        self.failure_policy = str(cfg.get("failure_policy", "error"))
        self.prompting = str(cfg.get("prompting", "joint"))
        if self.prompting not in {"joint", "per_dimension"}:
            raise ValueError("rubric_reward.prompting must be 'joint' or 'per_dimension'")
        if self.failure_policy not in {"error", "zero"}:
            raise ValueError("rubric_reward.failure_policy must be 'error' or 'zero'")

        providers = list(cfg.get("providers", ["server"]))
        if not providers:
            raise ValueError("rubric_reward.providers cannot be empty")
        raw_provider_weights = _plain_dict(cfg.get("provider_weights")) or {
            provider: 1.0 for provider in providers
        }
        self.provider_weights = _normalize_weights(raw_provider_weights, providers, "provider")
        raw_dimension_weights = _plain_dict(cfg.get("dimension_weights")) or DEFAULT_DIMENSION_WEIGHTS
        self.dimension_weights = _normalize_weights(raw_dimension_weights, SCORE_KEYS, "dimension")
        self.judges = {
            provider: _build_judge(
                provider,
                server_dir=str(cfg.get("server_dir", "")),
                timeout_seconds=float(cfg.get("server_timeout_seconds", 1800.0)),
                poll_seconds=float(cfg.get("server_poll_seconds", 0.2)),
                prompting=self.prompting,
            )
            for provider in providers
        }
        self._lock = threading.Lock()

    @staticmethod
    def _audio_numpy(audio) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            value = audio.detach().float().cpu().numpy()
        else:
            value = np.asarray(audio, dtype=np.float32)
        value = np.squeeze(value)
        if value.ndim == 2:
            value = value.mean(axis=0 if value.shape[0] <= value.shape[1] else 1)
        if value.ndim != 1:
            raise ValueError(f"Expected mono/stereo audio, got shape {value.shape}")
        return np.nan_to_num(np.clip(value, -1.0, 1.0)).astype(np.float32)

    def _score_with_retry(self, provider: str, prompt: str, path: str) -> Dict[str, float]:
        for attempt in range(self.retries + 1):
            try:
                device_context = (
                    torch.cuda.device(self.device)
                    if self.device.startswith("cuda")
                    else nullcontext()
                )
                with device_context:
                    return self.judges[provider].score(prompt, path)
            except Exception:
                if attempt >= self.retries:
                    if self.failure_policy == "zero":
                        LOGGER.exception("%s judge failed; assigning zero reward", provider)
                        return {key: 0.0 for key in SCORE_KEYS}
                    raise
                LOGGER.warning("%s judge failed; retrying (%d/%d)", provider, attempt + 1, self.retries)
                time.sleep(min(2 ** attempt, 5))
        raise AssertionError("unreachable")

    def __call__(self, audios, prompts, metadata=None, **_kwargs):
        if len(audios) != len(prompts):
            raise ValueError(f"Audio/prompt batch mismatch: {len(audios)} != {len(prompts)}")

        details: dict[str, list[float]] = {}
        totals: list[float] = []
        # DiffusionNFT may queue two futures against one reward instance. Model
        # generation is serialized to prevent races and transient VRAM spikes.
        with self._lock, tempfile.TemporaryDirectory(prefix="acestep_rubric_") as tmpdir:
            for index, (audio, prompt) in enumerate(zip(audios, prompts)):
                path = str(Path(tmpdir) / f"sample_{index:04d}.wav")
                sf.write(path, self._audio_numpy(audio), self.sample_rate, subtype="PCM_16")
                provider_scores = {
                    provider: self._score_with_retry(provider, str(prompt), path)
                    for provider in self.judges
                }
                combined = {
                    key: sum(
                        self.provider_weights[provider] * provider_scores[provider][key]
                        for provider in self.judges
                    )
                    for key in SCORE_KEYS
                }
                total = sum(self.dimension_weights[key] * combined[key] for key in SCORE_KEYS)
                totals.append(total)
                for key, value in combined.items():
                    details.setdefault(f"rubric_{key}", []).append(value)
                for provider, scores in provider_scores.items():
                    provider_total = sum(self.dimension_weights[key] * scores[key] for key in SCORE_KEYS)
                    details.setdefault(f"rubric_{provider}", []).append(provider_total)
                    details.setdefault(f"rubric_{provider}_equal_fields", []).append(
                        float(len({round(scores[key], 6) for key in SCORE_KEYS}) == 1)
                    )
                    for key, value in scores.items():
                        details.setdefault(f"rubric_{provider}_{key}", []).append(value)

        result = {key: np.asarray(value, dtype=np.float32) for key, value in details.items()}
        result["avg"] = np.asarray(totals, dtype=np.float32)
        return result, {
            "providers": list(self.judges),
            "provider_weights": self.provider_weights,
            "dimension_weights": self.dimension_weights,
            "prompting": self.prompting,
            "score_range": [0.0, 1.0],
        }


def rubric_score(device, config):
    return RubricReward(device, config)


# DiffusionNFT's ``config.reward_fn`` dispatcher (from the research code's
# ``rewards_audio.py``), reduced to the rubric reward.
def multi_score(device, score_dict, rubric_config=None):
    score_fns = {}
    for score_name, weight in score_dict.items():
        if score_name == "rubric":
            if rubric_config is None:
                raise ValueError("The rubric reward requires config.rubric_reward")
            score_fns[score_name] = rubric_score(device, rubric_config)
        else:
            # The research code skipped unknown names silently; this release
            # ships only the rubric reward, so fail loudly instead.
            raise ValueError(f"Unknown reward function {score_name!r}")

    def _fn(images, prompts, metadata, **kwargs):
        audios = images
        total_scores = []
        score_details = defaultdict(list)
        for score_name, weight in score_dict.items():
            scores, _ = score_fns[score_name](audios, prompts, metadata)
            for detail_name, detail_scores in scores.items():
                if detail_name != "avg":
                    score_details[detail_name] = detail_scores
            base_scores = scores["avg"]
            weighted_scores = [weight * score for score in base_scores]
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        score_details['avg'] = total_scores
        return score_details, {}
    return _fn
