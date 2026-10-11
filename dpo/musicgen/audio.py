"""Preference pairs -> fixed-length winner/loser audio batches for MusicGen DPO.

Pairs come from ``load_preference_pairs`` in ``dpo/dataset.py``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from dpo.dataset import PreferencePair, load_preference_pairs


def select_pairs(pairs: Sequence[PreferencePair], *, min_margin: float = 0.0, max_pairs: int | None = None) -> List[PreferencePair]:
    """Drop pairs whose margin is below ``min_margin``; keep at most ``max_pairs``."""
    selected = [p for p in pairs if p.margin is None or float(p.margin) >= min_margin]
    return selected[:max_pairs] if max_pairs is not None else selected


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio as a [C, T] float tensor (torchaudio, falling back to soundfile)."""
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        return wav.float(), int(sr)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        wav = torch.from_numpy(data).transpose(0, 1).contiguous()
        return wav, int(sr)


def _resample_audio(wav: torch.Tensor, src_sr: int, target_sr: int) -> torch.Tensor:
    if src_sr == target_sr:
        return wav
    try:
        import torchaudio

        return torchaudio.functional.resample(wav, src_sr, target_sr)
    except Exception:
        import librosa

        channels = []
        for ch in wav.detach().cpu().numpy():
            channels.append(torch.from_numpy(librosa.resample(ch, orig_sr=src_sr, target_sr=target_sr)))
        return torch.stack(channels).to(wav.dtype)


def _convert_channels(wav: torch.Tensor, channels: int) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == channels:
        return wav
    if channels == 1:
        return wav.mean(dim=0, keepdim=True)
    if wav.shape[0] == 1:
        return wav.repeat(channels, 1)
    return wav[:channels]


def _fit_length(wav: torch.Tensor, target_frames: int, *, random_crop: bool) -> torch.Tensor:
    frames = wav.shape[-1]
    if frames == target_frames:
        return wav
    if frames < target_frames:
        return F.pad(wav, (0, target_frames - frames))
    if random_crop:
        start = random.randint(0, frames - target_frames)
    else:
        start = (frames - target_frames) // 2
    return wav[..., start : start + target_frames]


class PreferenceAudioDataset(torch.utils.data.Dataset):
    """Aligned winner/loser audio for DPO, resampled, mixed to ``channels`` and cropped/padded to ``duration``."""

    def __init__(
        self,
        pairs: Sequence[PreferencePair],
        *,
        sample_rate: int = 32000,
        duration: float = 8.0,
        channels: int = 1,
        random_crop: bool = True,
    ) -> None:
        if not pairs:
            raise ValueError("PreferenceAudioDataset requires at least one pair")
        self.pairs = list(pairs)
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.channels = int(channels)
        self.random_crop = bool(random_crop)
        self.target_frames = int(round(self.sample_rate * self.duration))

    @classmethod
    def from_path(cls, path: str | Path, **kwargs: Any) -> "PreferenceAudioDataset":
        min_margin = float(kwargs.pop("min_margin", 0.0))
        max_pairs = kwargs.pop("max_pairs", None)
        pairs = select_pairs(load_preference_pairs(str(path)), min_margin=min_margin, max_pairs=max_pairs)
        return cls(pairs, **kwargs)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_item_audio(self, path: str) -> torch.Tensor:
        wav, sr = _load_audio(path)
        wav = _resample_audio(wav, sr, self.sample_rate)
        wav = _convert_channels(wav, self.channels)
        wav = _fit_length(wav, self.target_frames, random_crop=self.random_crop)
        return wav.clamp(-1.0, 1.0).contiguous()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        return {
            "prompt": pair.prompt,
            "winner_audio": self._load_item_audio(pair.winner),
            "loser_audio": self._load_item_audio(pair.loser),
            "winner_path": pair.winner,
            "loser_path": pair.loser,
            "margin": pair.margin,
        }


def preference_collate(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    batch = list(samples)
    return {
        "prompt": [item["prompt"] for item in batch],
        "winner_audio": torch.stack([item["winner_audio"] for item in batch]),
        "loser_audio": torch.stack([item["loser_audio"] for item in batch]),
        "winner_path": [item["winner_path"] for item in batch],
        "loser_path": [item["loser_path"] for item in batch],
        "margin": [item["margin"] for item in batch],
    }
