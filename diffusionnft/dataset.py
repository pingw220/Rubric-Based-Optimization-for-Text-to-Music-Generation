"""DiffusionNFT data loading. Implement ``load_prompts`` for your own data.

The DiffusionNFT trainer (``diffusionnft/train.py``) reads its text prompts only
through this function; how you store them is up to you.
"""
from __future__ import annotations


def load_prompts(path: str, split: str) -> list[str]:
    """Return the text prompts of ``split`` (``"train"`` or ``"val"``).

    ``path`` is the value of the ``RUBRIC_DATASET`` environment variable.
    DiffusionNFT samples training prompts for the policy updates and scores the
    validation prompts periodically to log the reward curve. Keep the two splits
    disjoint.
    """
    raise NotImplementedError(
        "Implement load_prompts() in diffusionnft/dataset.py to read your text prompts."
    )
