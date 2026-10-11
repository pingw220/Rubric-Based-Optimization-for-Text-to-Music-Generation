"""DPO data loading. Implement ``load_preference_pairs`` for your own data.

Both DPO trainers (``dpo/musicgen/train.py``, ``dpo/acestep/train.py``) read
preference pairs only through this function; how you store and build the pairs
is up to you.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreferencePair:
    prompt: str                   # text prompt the two clips were generated from
    winner: str                   # path to the preferred audio file
    loser: str                    # path to the rejected audio file
    margin: float | None = None   # reward gap between winner and loser (used by --min_margin)


def load_preference_pairs(path: str) -> list[PreferencePair]:
    """Return the DPO preference pairs stored at ``path``.

    ``path`` is the value of ``--train_preferences`` / ``--val_preferences``
    (MusicGen) or ``--train_pairs`` (ACE-Step). Audio files can be in any format
    readable by torchaudio or soundfile; they are resampled and cropped by the
    trainers. Pairs whose ``margin`` is below ``--min_margin`` are dropped by the
    trainers, so ``margin`` is required only when you use that option.
    """
    raise NotImplementedError(
        "Implement load_preference_pairs() in dpo/dataset.py to read your preference pairs."
    )
