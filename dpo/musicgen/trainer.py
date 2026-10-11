"""DPO for Audiocraft MusicGen: log-probabilities, loss, and training loop.

MusicGen is an autoregressive LM over EnCodec codebooks, so DPO compares
conditional token log-probabilities (mean over valid code positions).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class MusicGenDPOConfig:
    model_name: str = "facebook/musicgen-small"
    beta: float = 0.1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    epochs: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    device: str | None = None
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    freeze_bottom_layers: int = 0
    lora_r: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported amp dtype: {name}")


def load_musicgen_for_dpo(model_name: str, device: torch.device):
    """Load Audiocraft MusicGen and place LM/compression modules on device."""
    from audiocraft.models import MusicGen

    model = MusicGen.get_pretrained(model_name, device=str(device))
    # Audiocraft may load parts of MusicGen in half precision on GPU. Keep the
    # DPO path in fp32 by default; MusicGen-small hit half/float LayerNorm
    # mismatches otherwise, even without autocast.
    model.lm.to(device).float()
    model.compression_model.to(device).float()
    model.compression_model.eval()
    model.compression_model.requires_grad_(False)
    return model


def freeze_musicgen_reference(model) -> None:
    model.lm.eval()
    model.lm.requires_grad_(False)
    model.compression_model.eval()
    model.compression_model.requires_grad_(False)


def lm_module(model):
    """Return the underlying LM, unwrapping DistributedDataParallel when used."""
    return model.lm.module if hasattr(model.lm, "module") else model.lm


def build_text_conditions(prompts: List[str]) -> list:
    from audiocraft.modules.conditioners import ConditioningAttributes

    return [ConditioningAttributes(text={"description": prompt}) for prompt in prompts]


@torch.no_grad()
def encode_audio_codes(musicgen_model, audio: torch.Tensor) -> torch.Tensor:
    """Encode waveform audio to MusicGen EnCodec tokens.

    Args:
        musicgen_model: Audiocraft MusicGen wrapper.
        audio: [B, C, T] waveform tensor at the MusicGen sample rate.

    Returns:
        Long tensor of codes [B, K, T_codes].
    """
    codes, scale = musicgen_model.compression_model.encode(audio)
    if scale is not None:
        # MusicGen's LM models the discrete codes only. Scale is for waveform
        # reconstruction, not token likelihood.
        del scale
    return codes.long()


def musicgen_sequence_logprob(musicgen_model, codes: torch.Tensor, prompts: List[str]) -> torch.Tensor:
    """Compute conditional log p(codes | prompts) for MusicGen.

    Audiocraft's `compute_predictions` returns logits aligned with target code
    positions, so no manual one-token shift is applied.
    """
    conditions = build_text_conditions(prompts)
    output = lm_module(musicgen_model).compute_predictions(codes=codes, conditions=conditions)
    logits = output.logits
    mask = output.mask
    if logits.shape[:3] != codes.shape:
        raise RuntimeError(f"Unexpected logits/codes shape mismatch: {tuple(logits.shape)} vs {tuple(codes.shape)}")
    token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, codes.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(~mask.bool(), 0.0)
    # Normalize by sequence length so margins stay scale-invariant across
    # different audio lengths and don't saturate the DPO logit clamp.
    n_tokens = mask.bool().sum(dim=(1, 2)).clamp(min=1).float()
    return token_logprobs.sum(dim=(1, 2)) / n_tokens


def dpo_loss(
    policy_w: torch.Tensor,
    policy_l: torch.Tensor,
    ref_w: torch.Tensor,
    ref_l: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_margin = policy_w - policy_l
    ref_margin = ref_w - ref_l
    logits = float(beta) * (policy_margin - ref_margin)
    logits = logits.clamp(min=-20.0, max=20.0)
    loss = -F.logsigmoid(logits).mean()
    metrics = {
        "loss": loss.detach(),
        "policy_margin": policy_margin.mean().detach(),
        "ref_margin": ref_margin.mean().detach(),
        "implicit_reward_margin": (policy_margin - ref_margin).mean().detach(),
        "dpo_accuracy_over_ref": (policy_margin > ref_margin).float().mean().detach(),
        "policy_pair_accuracy": (policy_margin > 0).float().mean().detach(),
        "policy_w_logprob": policy_w.mean().detach(),
        "policy_l_logprob": policy_l.mean().detach(),
    }
    return loss, metrics


def musicgen_dpo_step(
    policy_model,
    reference_model,
    batch: Dict[str, Any],
    *,
    beta: float,
    use_amp: bool,
    amp_dtype_value: torch.dtype,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    prompts = list(batch["prompt"])
    winner_audio = batch["winner_audio"]
    loser_audio = batch["loser_audio"]

    with torch.no_grad():
        winner_codes = encode_audio_codes(policy_model, winner_audio)
        loser_codes = encode_audio_codes(policy_model, loser_audio)

    enabled_amp = use_amp and winner_audio.is_cuda and amp_dtype_value != torch.float32
    with torch.autocast(device_type=winner_audio.device.type, dtype=amp_dtype_value, enabled=enabled_amp):
        policy_w = musicgen_sequence_logprob(policy_model, winner_codes, prompts)
        policy_l = musicgen_sequence_logprob(policy_model, loser_codes, prompts)
    ref_device = next(lm_module(reference_model).parameters()).device
    with torch.no_grad(), torch.autocast(device_type=ref_device.type, dtype=amp_dtype_value, enabled=use_amp and ref_device.type == "cuda" and amp_dtype_value != torch.float32):
        ref_w = musicgen_sequence_logprob(reference_model, winner_codes.to(ref_device, non_blocking=True), prompts)
        ref_l = musicgen_sequence_logprob(reference_model, loser_codes.to(ref_device, non_blocking=True), prompts)
    return dpo_loss(policy_w, policy_l, ref_w.to(policy_w.device), ref_l.to(policy_l.device), beta=beta)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _float_metrics(metrics: Dict[str, torch.Tensor], *, epoch: int, step: int) -> Dict[str, float]:
    row = {"epoch": float(epoch), "step": float(step)}
    row.update({key: float(value.detach().cpu()) for key, value in metrics.items()})
    return row


def reduce_metrics_if_distributed(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return metrics
    keys = sorted(metrics)
    tensor = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return {k: float(v) for k, v in zip(keys, tensor.detach().cpu().tolist())}


def average_gradients_if_distributed(module: torch.nn.Module) -> None:
    """Average gradients across torchrun ranks.

    We use manual gradient averaging instead of wrapping the LM in DDP because
    Audiocraft exposes `compute_predictions()` as the training API rather than a
    plain `forward()` call, and DDP hooks only fire through `forward()`.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return
    for param in module.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def assert_finite_trainable_parameters(module: torch.nn.Module, *, step: int) -> None:
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if not torch.isfinite(param).all():
            raise FloatingPointError(f"Non-finite trainable parameter after optimizer step {step}: {name}")


def save_musicgen_dpo_checkpoint(
    path: str | Path,
    *,
    policy_model,
    optimizer: torch.optim.Optimizer,
    cfg: MusicGenDPOConfig,
    epoch: int,
    global_step: int,
    val_metrics: Dict[str, float] | None,
) -> None:
    try:
        from dpo.musicgen.lora import merged_state_dict

        lm_state = merged_state_dict(lm_module(policy_model))
    except Exception:
        lm_state = lm_module(policy_model).state_dict()
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_step": global_step // max(1, int(cfg.gradient_accumulation_steps)),
        "model_name": cfg.model_name,
        "lm_state_dict": lm_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg.__dict__,
        "val_metrics": val_metrics,
    }
    torch.save(ckpt, path)


def _build_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    if warmup_steps <= 0:
        return None
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps))



def _freeze_bottom_layers(policy_model, n: int) -> int:
    """Freeze the bottom n transformer layers of the MusicGen LM.

    Returns the number of layers actually frozen (may be capped at total depth).
    Audiocraft's LMModel exposes its layers via .transformer.layers.
    (Copied from the research code's SFT module, which is not part of this release.)
    """
    try:
        layers = list(lm_module(policy_model).transformer.layers)
    except AttributeError:
        print("[freeze] WARNING: could not find .transformer.layers; skipping layer freeze", flush=True)
        return 0
    n_freeze = min(n, len(layers))
    for layer in layers[:n_freeze]:
        for param in layer.parameters():
            param.requires_grad_(False)
    print(f"[freeze] froze bottom {n_freeze}/{len(layers)} transformer layers; "
          f"training top {len(layers) - n_freeze}", flush=True)
    return n_freeze

def train_musicgen_dpo(
    policy_model,
    reference_model,
    train_loader: Iterable[Dict[str, Any]],
    *,
    cfg: MusicGenDPOConfig,
    val_loader: Iterable[Dict[str, Any]] | None = None,
    output_dir: str | Path = "outputs/musicgen_dpo",
    log_every: int = 10,
    resume_checkpoint: str | Path | None = None,
    wandb_run: Any | None = None,
    is_main_process: bool = True,
    train_sampler: Any | None = None,
    val_sampler: Any | None = None,
    checkpoint_callback: Callable[[Any, int, int, Path, Dict[str, float] | None], None] | None = None,
) -> list[Dict[str, float]]:
    device = resolve_device(cfg.device)
    policy_model.lm.train()
    reference_model.lm.eval()
    freeze_musicgen_reference(reference_model)

    if cfg.freeze_bottom_layers > 0:
        _freeze_bottom_layers(policy_model, cfg.freeze_bottom_layers)

    trainable_params = [p for p in policy_model.lm.parameters() if p.requires_grad]
    if is_main_process:
        n_total = sum(p.numel() for p in policy_model.lm.parameters())
        n_train = sum(p.numel() for p in trainable_params)
        print(f"[model] trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)", flush=True)
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = _build_warmup_scheduler(optimizer, cfg.warmup_steps)
    start_epoch = 0
    global_step = 0
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        lm_module(policy_model).load_state_dict(ckpt["lm_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))

    amp_value = autocast_dtype(cfg.amp_dtype)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[Dict[str, float]] = []
    grad_accum = max(1, int(cfg.gradient_accumulation_steps))
    optimizer_step = 0

    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        policy_model.lm.train()
        optimizer.zero_grad(set_to_none=True)
        accum_metrics: Dict[str, float] = {}
        accum_count = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            global_step += 1
            batch = move_batch_to_device(batch, device)
            loss, metrics = musicgen_dpo_step(
                policy_model,
                reference_model,
                batch,
                beta=cfg.beta,
                use_amp=cfg.use_amp,
                amp_dtype_value=amp_value,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DPO loss at micro_step={global_step}: {float(loss.detach().cpu())}")
            (loss / grad_accum).backward()
            metric_row = _float_metrics(metrics, epoch=epoch, step=global_step)
            for key, value in metric_row.items():
                if key in {"epoch", "step"}:
                    continue
                accum_metrics[key] = accum_metrics.get(key, 0.0) + float(value)
            accum_count += 1

            grad_norm = None
            if batch_idx % grad_accum == 0:
                average_gradients_if_distributed(lm_module(policy_model))
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.lm.parameters(), cfg.max_grad_norm)
                optimizer.step()
                assert_finite_trainable_parameters(lm_module(policy_model), step=optimizer_step + 1)
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                row = {key: value / max(1, accum_count) for key, value in accum_metrics.items()}
                row = reduce_metrics_if_distributed(row, device)
                row.update({
                    "epoch": float(epoch),
                    "micro_step": float(global_step),
                    "optimizer_step": float(optimizer_step),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                })
                if grad_norm is not None:
                    row["grad_norm"] = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
                history.append(row)
                if wandb_run is not None and is_main_process:
                    wandb_run.log(
                        {"optimizer_step": optimizer_step, **{f"train/{key}": value for key, value in row.items()}},
                        step=optimizer_step,
                    )
                if is_main_process and log_every > 0 and optimizer_step % log_every == 0:
                    print(
                        f"[train] epoch={epoch} opt_step={optimizer_step} micro_step={global_step} "
                        f"loss={row['loss']:.4f} dpo_acc={row['dpo_accuracy_over_ref']:.3f} "
                        f"policy_acc={row['policy_pair_accuracy']:.3f} "
                        f"reward_margin={row['implicit_reward_margin']:.3f}",
                        flush=True,
                    )
                accum_metrics = {}
                accum_count = 0

        if len(train_loader) % grad_accum != 0:  # type: ignore[arg-type]
            average_gradients_if_distributed(lm_module(policy_model))
            if cfg.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.lm.parameters(), cfg.max_grad_norm)
            else:
                grad_norm = None
            optimizer.step()
            assert_finite_trainable_parameters(lm_module(policy_model), step=optimizer_step + 1)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            row = {key: value / max(1, accum_count) for key, value in accum_metrics.items()}
            row = reduce_metrics_if_distributed(row, device)
            row.update({
                "epoch": float(epoch),
                "micro_step": float(global_step),
                "optimizer_step": float(optimizer_step),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })
            if grad_norm is not None:
                row["grad_norm"] = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            history.append(row)
            if wandb_run is not None and is_main_process:
                wandb_run.log(
                    {"optimizer_step": optimizer_step, **{f"train/{key}": value for key, value in row.items()}},
                    step=optimizer_step,
                )

        val_metrics = evaluate_musicgen_dpo(
            policy_model,
            reference_model,
            val_loader,
            beta=cfg.beta,
            device=device,
            use_amp=cfg.use_amp,
            amp_dtype_value=amp_value,
            sampler=val_sampler,
        ) if val_loader is not None else None
        if is_main_process:
            save_musicgen_dpo_checkpoint(
                output_path / f"checkpoint_epoch_{epoch:03d}.pt",
                policy_model=policy_model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                val_metrics=val_metrics,
            )
            save_musicgen_dpo_checkpoint(
                output_path / "checkpoint_last.pt",
                policy_model=policy_model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                val_metrics=val_metrics,
            )
            if checkpoint_callback is not None:
                checkpoint_callback(policy_model, epoch, global_step, output_path, val_metrics)
        if is_main_process and val_metrics is not None:
            if wandb_run is not None:
                wandb_run.log(
                    {"optimizer_step": optimizer_step, **{f"val/{key}": value for key, value in val_metrics.items()}},
                    step=optimizer_step,
                )
            print(
                f"[val] epoch={epoch} loss={val_metrics['loss']:.4f} "
                f"dpo_acc={val_metrics['dpo_accuracy_over_ref']:.3f} "
                f"policy_acc={val_metrics['policy_pair_accuracy']:.3f}",
                flush=True,
            )

    return history


@torch.no_grad()
def evaluate_musicgen_dpo(
    policy_model,
    reference_model,
    loader: Iterable[Dict[str, Any]] | None,
    *,
    beta: float,
    device: torch.device,
    use_amp: bool,
    amp_dtype_value: torch.dtype,
    sampler: Any | None = None,
) -> Dict[str, float] | None:
    if loader is None:
        return None
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    policy_model.lm.eval()
    rows: list[Dict[str, float]] = []
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        _, metrics = musicgen_dpo_step(
            policy_model,
            reference_model,
            batch,
            beta=beta,
            use_amp=use_amp,
            amp_dtype_value=amp_dtype_value,
        )
        rows.append(_float_metrics(metrics, epoch=0, step=step))
    if not rows:
        return None
    keys = [key for key in rows[0].keys() if key not in {"epoch", "step"}]
    averaged = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    if dist.is_available() and dist.is_initialized():
        for key, value in averaged.items():
            tensor = torch.tensor(value, device=device, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            averaged[key] = float((tensor / dist.get_world_size()).item())
    return averaged
