from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    r: int = 0
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("linear", "out_proj", "in_proj", "q_proj", "k_proj", "v_proj")


class LoRALinear(nn.Module):
    """Small dependency-free LoRA wrapper for nn.Linear.

    Checkpoints are saved as merged base weights by `merged_state_dict`, so
    inference can load them into the original Audiocraft LM without LoRA code.
    """

    def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(self.r, base.in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return out + update

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight.detach() + (self.lora_b @ self.lora_a).detach() * self.scaling


def _target_match(name: str, targets: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(target and target.lower() in lowered for target in targets)


def apply_lora_to_linear_layers(module: nn.Module, cfg: LoRAConfig) -> int:
    """Replace matching Linear children with LoRA adapters and freeze the rest."""
    if cfg.r <= 0:
        return 0
    for param in module.parameters():
        param.requires_grad_(False)

    replaced = 0

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _target_match(full_name, cfg.target_modules):
                setattr(parent, child_name, LoRALinear(child, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout))
                replaced += 1
            else:
                visit(child, full_name)

    visit(module)
    return replaced


def apply_peft_lora_to_linear_layers(module: nn.Module, cfg: LoRAConfig) -> int:
    """Inject PEFT LoRA adapters into matching Linear layers.

    Walks the module tree manually (same logic as apply_lora_to_linear_layers)
    and replaces each matching nn.Linear with peft.tuners.lora.layer.Linear.
    This avoids inject_adapter_in_model, which does not support audiocraft's
    custom streaming module types (StreamingTransformerLayer, etc.).
    """
    try:
        from peft.tuners.lora.layer import Linear as PeftLoraLinear
        from peft import LoraConfig as PeftLoraConfig
    except Exception as exc:
        raise ImportError(
            "LORA_IMPL=peft requested, but `peft` is not installed in this environment. "
            "Install peft in the music-rl conda env or use LORA_IMPL=custom."
        ) from exc

    if cfg.r <= 0:
        return 0
    for param in module.parameters():
        param.requires_grad_(False)

    peft_config = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=int(cfg.alpha),
        lora_dropout=cfg.dropout,
        target_modules=[],  # not used; we walk the tree ourselves
        bias="none",
    )

    replaced = 0

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _target_match(full_name, cfg.target_modules):
                peft_linear = PeftLoraLinear(
                    base_layer=child,
                    adapter_name="default",
                    config=peft_config,
                    r=cfg.r,
                    lora_alpha=int(cfg.alpha),
                )
                peft_linear.set_adapter("default")
                setattr(parent, child_name, peft_linear)
                replaced += 1
            else:
                visit(child, full_name)

    visit(module)
    return replaced


def _peft_lora_type():
    """Return the PEFT LoRA Linear class, or None if peft is not installed."""
    try:
        from peft.tuners.lora.layer import Linear as PeftLoraLinear
        return PeftLoraLinear
    except Exception:
        return None


def merged_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a state dict loadable by the original non-LoRA module.

    Handles both custom LoRALinear and PEFT lora.layer.Linear wrappers.
    """
    PeftLoraLinear = _peft_lora_type()
    state = module.state_dict()
    out: dict[str, torch.Tensor] = {}
    lora_prefixes: set[str] = set()

    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            lora_prefixes.add(name)
            out[f"{name}.weight"] = child.merged_weight().cpu()
            if child.base.bias is not None:
                out[f"{name}.bias"] = child.base.bias.detach().cpu()
        elif PeftLoraLinear is not None and isinstance(child, PeftLoraLinear):
            lora_prefixes.add(name)
            base = child.get_base_layer()
            scaling = child.scaling.get("default", 1.0)
            A = child.lora_A["default"].weight  # (r, in_features)
            B = child.lora_B["default"].weight  # (out_features, r)
            merged_w = (base.weight + scaling * (B @ A)).detach().cpu()
            out[f"{name}.weight"] = merged_w
            if base.bias is not None:
                out[f"{name}.bias"] = base.bias.detach().cpu()

    for key, value in state.items():
        # Skip custom LoRALinear internals
        if ".lora_a" in key or ".lora_b" in key or ".base." in key:
            continue
        # Skip PEFT LoRA layer internals
        if any(
            f"{prefix}.lora_" in key or f"{prefix}.base_layer." in key
            for prefix in lora_prefixes
        ):
            continue
        if any(key == f"{prefix}.weight" or key == f"{prefix}.bias" for prefix in lora_prefixes):
            continue
        out[key] = value.detach().cpu()
    return out
