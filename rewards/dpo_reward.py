"""Seven-dimension reward used to rank the DPO candidates.

The rater (Music Flamingo, Qwen2-Audio, or Qwen3-Omni) receives the prompt in
``prompts/dpo.txt`` and replies with one JSON object of seven 0-10 scores. The
reply is parsed with ``extract_json_object``, checked and clamped to [0, 10] with
``normalize_scores``, and combined with the rubric weights by
``weighted_reward`` (0-10 scale). Candidates for the same text prompt are ranked
by this reward to form preference pairs.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Dict


SCORE_KEYS = (
    "prompt_alignment",
    "style_instrumentation",
    "melody",
    "rhythm",
    "structure",
    "production_quality",
    "overall_musicality",
)
DEFAULT_REWARD_WEIGHTS = {
    "prompt_alignment": 0.25,
    "style_instrumentation": 0.15,
    "melody": 0.10,
    "rhythm": 0.10,
    "structure": 0.15,
    "production_quality": 0.15,
    "overall_musicality": 0.10,
}
assert set(DEFAULT_REWARD_WEIGHTS) == set(SCORE_KEYS)
assert abs(sum(DEFAULT_REWARD_WEIGHTS.values()) - 1.0) < 1e-9


def extract_json_object(text: str, *, source: str = "judge") -> Dict[str, object]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try strict JSON first, then extract embedded {...}, then fall back to
    # ast.literal_eval for models that return Python dict syntax (single quotes).
    last_exc: Exception | None = None
    for candidate in _json_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError) as exc:
            last_exc = exc

    raise ValueError(
        f"Expected {source} JSON object, could not parse: {text!r}"
    ) from last_exc


def _json_candidates(text: str):
    yield text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        yield match.group(0)


def normalize_scores(scores: Dict[str, object]) -> Dict[str, float]:
    missing = [key for key in SCORE_KEYS if key not in scores]
    if missing:
        available = ", ".join(sorted(str(key) for key in scores.keys()))
        raise ValueError(
            f"Missing required score field(s): {', '.join(missing)}. "
            f"Available fields: {available}"
        )

    out: Dict[str, float] = {}
    for key in SCORE_KEYS:
        try:
            value = float(scores[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Score field {key!r} must be numeric, got {scores[key]!r}") from exc
        out[key] = max(0.0, min(10.0, value))
    return out


def weighted_reward(
    reward: Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> float:
    w = weights or DEFAULT_REWARD_WEIGHTS
    return sum(float(reward.get(k, 0.0)) * float(v) for k, v in w.items())
