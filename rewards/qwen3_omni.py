"""Qwen3-Omni rater for the rubric reward (joint and per-dimension prompting).

Used by ``diffusionnft/server.py`` (``--model qwen3_omni``). Prompts are generated from the rubric
(``rubrics/general.yaml`` by default).
"""
from __future__ import annotations

import json
import importlib.util
import threading
from pathlib import Path
from typing import Dict

from rewards.dpo_reward import extract_json_object, normalize_scores
from rewards.nft_reward import SCORE_KEYS, _load_7d_rubric, _single_dimension_value


def missing_dependency_error(provider: str, packages: list[str], extra: str | None = None) -> RuntimeError:
    package_text = ", ".join(packages)
    message = f"{provider} judge requires missing package(s): {package_text}."
    if extra:
        message = f"{message} {extra}"
    return RuntimeError(message)


def parse_torch_dtype(dtype: str | None):
    if not dtype:
        return None
    import torch

    normalized = str(dtype).lower()
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized == "auto":
        return "auto"
    if normalized not in aliases:
        raise ValueError(f"Unsupported torch dtype {dtype!r}; expected bfloat16, float16, float32, or auto")
    return aliases[normalized]


class Qwen3OmniJudge:
    """Qwen3-Omni rater, joint prompting: all rubric dimensions in one call."""

    provider = "qwen3_omni"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device_map: str | None = "auto",
        dtype: str | None = "bfloat16",
        max_new_tokens: int = 256,
        prompt_schema: str = "general",
    ) -> None:
        self.model_id = model_id
        self.name = f"{self.provider}:{self.model_id}"
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self.prompt_schema = str(prompt_schema)
        self._prompt_rubric = _load_7d_rubric(self.prompt_schema)

    def is_available(self) -> bool:
        return bool(importlib.util.find_spec("transformers") and importlib.util.find_spec("qwen_omni_utils"))

    def availability_error(self) -> str:
        return (
            "Install qwen-omni-utils and latest/source Transformers if your installed Transformers "
            "does not include Qwen3-Omni support. flash-attn is optional unless explicitly configured."
        )

    def _load(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            import qwen_omni_utils  # noqa: F401
        except ImportError as exc:
            raise missing_dependency_error(
                self.provider,
                ["qwen-omni-utils"],
                "Install qwen-omni-utils for Qwen3-Omni audio/video preprocessing. "
                "flash-attn is optional unless you explicitly configure an attention backend that requires it.",
            ) from exc
        try:
            from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise missing_dependency_error(
                self.provider,
                ["transformers"],
                "Transformers >= 5.5 required for Qwen3OmniMoeForConditionalGeneration.",
            ) from exc

        kwargs = {}
        if self.device_map:
            kwargs["device_map"] = self.device_map
        torch_dtype = parse_torch_dtype(self.dtype)
        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **kwargs,
        )
        if hasattr(self._model, "disable_talker"):
            self._model.disable_talker()
        return self._model, self._processor

    def _prompt(self, prompt: str) -> str:
        # Qwen3-Omni receives the system prompt folded into the user turn.
        return (
            f"{self._prompt_rubric.qwen_system_prompt()}\n\n"
            f"{self._prompt_rubric.qwen_user_prompt(prompt)}"
        )

    def score_audio_multi(self, prompt: str, audio_path: str) -> Dict[str, float]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        model, processor = self._load()
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as exc:
            raise missing_dependency_error(
                self.provider,
                ["qwen-omni-utils"],
                "Install qwen-omni-utils for Qwen3-Omni preprocessing.",
            ) from exc

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(path)},
                    {"type": "text", "text": self._prompt(prompt)},
                ],
            }
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(model, "device"):
            inputs = inputs.to(model.device)
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None:
            for key, value in list(inputs.items()):
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    inputs[key] = value.to(dtype=model_dtype)

        generate_kwargs = {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
        }
        try:
            generated = model.generate(**inputs, return_audio=False, **generate_kwargs)
        except TypeError:
            generated = model.generate(**inputs, **generate_kwargs)

        if isinstance(generated, tuple):
            generated = generated[0]
        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = generated[:, prompt_len:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            parsed = extract_json_object(response, source=self.name)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"{self.name} response was not valid scoring JSON: {response!r}") from exc
        return normalize_scores(parsed)


class Qwen3OmniIndependentJudge(Qwen3OmniJudge):
    """Qwen3-Omni rater, per-dimension prompting: one fresh call per dimension.

    Scores stay on the [0, 10] scale; the trainer-side client normalizes them to
    [0, 1].
    """

    def _generate_text(self, prompt_text: str, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        model, processor = self._load()
        from qwen_omni_utils import process_mm_info

        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(path)},
                {"type": "text", "text": prompt_text},
            ],
        }]
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=False,
        )
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(model, "device"):
            inputs = inputs.to(model.device)
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None:
            for key, value in list(inputs.items()):
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    inputs[key] = value.to(dtype=model_dtype)

        result: dict = {}

        def generate() -> None:
            try:
                result["out"] = model.generate(
                    **inputs,
                    return_audio=False,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                )
            except TypeError:
                result["out"] = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as exc:
                result["err"] = exc

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        worker.join(timeout=180)
        if worker.is_alive():
            raise RuntimeError(
                f"{self.name}: generate() timed out after 180s on "
                f"{path.name} for independent-dimension prompt"
            )
        if "err" in result:
            raise result["err"]

        generated = result["out"]
        if isinstance(generated, tuple):
            generated = generated[0]
        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = generated[:, prompt_len:]
        return processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def score_audio_multi(self, prompt: str, audio_path: str) -> Dict[str, float]:
        scores = {}
        for key in SCORE_KEYS:
            prompt_text = (
                f"{self._prompt_rubric.flamingo_system_prompt()}\n\n"
                f"{self._prompt_rubric.flamingo_dimension_prompt(key, prompt)}"
            )
            response = self._generate_text(
                prompt_text,
                audio_path,
            )
            parsed = extract_json_object(response, source=self.name)
            value = _single_dimension_value(parsed, key, response)
            scores[key] = max(0.0, min(10.0, value))
        return scores
