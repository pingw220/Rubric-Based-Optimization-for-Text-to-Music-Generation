from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dpo.musicgen.audio import PreferenceAudioDataset, preference_collate
from dpo.musicgen.trainer import (
    MusicGenDPOConfig,
    load_musicgen_for_dpo,
    lm_module,
    resolve_device,
    train_musicgen_dpo,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train facebook/musicgen-* with autoregressive DPO")
    p.add_argument("--config", default=None,
                   help="Optional YAML file whose keys are argument names. Provides defaults for any "
                        "argument; explicit CLI flags always override it.")
    p.add_argument("--train_preferences", required=True, help="training preference pairs (read by load_preference_pairs in dpo/dataset.py)")
    p.add_argument("--val_preferences", default=None, help="validation preference pairs (read by load_preference_pairs in dpo/dataset.py)")
    p.add_argument("--output_dir", default="outputs/musicgen_dpo/musicgen-small")
    p.add_argument("--model_name", default="facebook/musicgen-small")
    p.add_argument("--sample_rate", type=int, default=32000)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_train_pairs", type=int, default=None)
    p.add_argument("--max_val_pairs", type=int, default=None)
    p.add_argument("--min_margin", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=0, help="Linear LR warmup over this many optimizer steps")
    p.add_argument("--device", default=None)
    p.add_argument("--reference_device", default=None, help="Optional device for the frozen reference model, e.g. cuda:1")
    p.add_argument("--amp", action="store_true", help="Enable autocast. Disabled by default for Audiocraft MusicGen stability.")
    p.add_argument("--no_amp", action="store_true", help="Deprecated alias; AMP is already disabled unless --amp is set.")
    p.add_argument("--amp_dtype", default="bfloat16")
    p.add_argument("--freeze_bottom_layers", type=int, default=0,
                   help="Freeze the bottom N transformer layers of the policy to prevent catastrophic forgetting (0=off)")
    p.add_argument("--lora_r", type=int, default=0, help="Enable LoRA with this rank on matching Linear layers (0=off)")
    p.add_argument("--lora_impl", default="custom", choices=["custom", "peft"])
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", default="linear,out_proj,in_proj,q_proj,k_proj,v_proj",
                   help="Comma-separated substrings of Linear module names to wrap with LoRA")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume_checkpoint", default=None)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="musicgen-dpo")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_run_id", default=None, help="Optional W&B run id to resume/update")
    p.add_argument("--wandb_resume", default="allow")
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument(
        "--wandb_api_key_file",
        default=os.environ.get("WANDB_API_KEY_FILE"),
        help="Optional file containing a W&B API key; avoids storing secrets in the repo",
    )
    p.add_argument("--experiment_index", default=os.environ.get("EXPERIMENT_INDEX", ""))
    p.add_argument("--experiment_tag", default=os.environ.get("EXPERIMENT_TAG", ""))
    p.add_argument("--training_mode", default=os.environ.get("TRAINING_MODE", "dpo"))
    p.add_argument("--dry_run_data", action="store_true", help="Load one batch and exit before loading MusicGen")
    p.add_argument("--dry_run_model", action="store_true", help="Load MusicGen, encode one batch, then exit before training")
    p.add_argument(
        "--policy_init_checkpoint",
        default=None,
        help=(
            "Optional SFT or other checkpoint: loads only the LM state_dict into the "
            "policy model before DPO begins. The optimizer is NOT restored — training "
            "starts fresh. Use this for SFT → DPO pipelines. "
            "Distinct from --resume_checkpoint which resumes a DPO run including optimizer state."
        ),
    )
    p.add_argument(
        "--reference_checkpoint",
        default=None,
        help=(
            "Optional checkpoint to load LM weights into the reference model. "
            "Default (None): reference = original base MusicGen (recommended). "
            "Only set this if you explicitly want the reference to differ from base."
        ),
    )
    args = p.parse_args()
    _apply_yaml_config(args)
    return args


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def _apply_yaml_config(args: argparse.Namespace) -> None:
    """Merge a --config YAML file into args. Precedence: CLI flag > YAML > argparse default.

    YAML keys must match argument dest names (e.g. learning_rate, beta, epochs).
    Keys that are not arguments (e.g. training_type) are ignored with a notice.
    """
    if not getattr(args, "config", None):
        return
    import yaml

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"--config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # dest names the user passed explicitly on the command line (these win over YAML)
    cli_set = {
        tok.lstrip("-").split("=")[0].replace("-", "_")
        for tok in sys.argv[1:] if tok.startswith("-")
    }
    valid = set(vars(args))
    applied, ignored = [], []
    for key, value in cfg.items():
        if key not in valid:
            ignored.append(key)
            continue
        if key in cli_set:
            continue  # explicit CLI flag overrides the YAML value
        setattr(args, key, value)
        applied.append(key)
    print(f"[config] loaded {config_path} — applied {sorted(applied)}", flush=True)
    if ignored:
        print(f"[config] ignored non-argument keys: {sorted(ignored)}", flush=True)


def _build_loader(
    path: str,
    *,
    sample_rate: int,
    duration: float,
    channels: int,
    batch_size: int,
    num_workers: int,
    min_margin: float,
    max_pairs: int | None,
    random_crop: bool,
    shuffle: bool,
    distributed: bool = False,
) -> tuple[PreferenceAudioDataset, DataLoader, DistributedSampler | None]:
    ds = PreferenceAudioDataset.from_path(
        path,
        sample_rate=sample_rate,
        duration=duration,
        channels=channels,
        random_crop=random_crop,
        min_margin=min_margin,
        max_pairs=max_pairs,
    )
    sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=False) if distributed else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=preference_collate,
    )
    return ds, loader, sampler


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed MusicGen DPO currently requires CUDA GPUs")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return distributed, rank, local_rank, world_size


def _cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _init_wandb(args: argparse.Namespace, *, train_size: int, val_size: int | None):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if args.wandb_api_key_file:
        key_path = Path(args.wandb_api_key_file).expanduser()
        if key_path.exists():
            os.environ.setdefault("WANDB_API_KEY", key_path.read_text(encoding="utf-8").strip())
        else:
            raise FileNotFoundError(f"W&B API key file not found: {key_path}")
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    import wandb

    config = {
        "model_name": args.model_name,
        "experiment_index": args.experiment_index,
        "experiment_tag": args.experiment_tag,
        "training_mode": args.training_mode,
        "train_preferences": args.train_preferences,
        "val_preferences": args.val_preferences,
        "output_dir": args.output_dir,
        "train_pairs": train_size,
        "val_pairs": val_size,
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "channels": args.channels,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_train_pairs": args.max_train_pairs,
        "max_val_pairs": args.max_val_pairs,
        "min_margin": args.min_margin,
        "epochs": args.epochs,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "amp_dtype": args.amp_dtype,
        "use_amp": bool(args.amp and not args.no_amp),
        "lora_r": args.lora_r,
        "lora_impl": args.lora_impl,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": args.lora_targets,
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume=args.wandb_resume if args.wandb_run_id else None,
        config=config,
    )
    wandb.define_metric("optimizer_step")
    wandb.define_metric("train/*", step_metric="optimizer_step")
    wandb.define_metric("val/*", step_metric="optimizer_step")
    wandb.define_metric("samples/*", step_metric="optimizer_step")
    return run


def _git_commit() -> str | None:
    """Return the current git commit hash for run provenance, or None."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _save_run_config(
    args: argparse.Namespace,
    output_dir: str,
    *,
    train_size: int,
    val_size: int | None,
) -> None:
    """Dump the fully-resolved DPO run configuration to output_dir/run_config.json.

    Makes a run reproducible from its output directory alone.
    """
    cfg = {
        "training_type": "dpo",
        "experiment_index": args.experiment_index,
        "experiment_tag": args.experiment_tag,
        "training_mode": args.training_mode,
        "model_name": args.model_name,
        "train_preferences": args.train_preferences,
        "val_preferences": args.val_preferences,
        "output_dir": output_dir,
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "channels": args.channels,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "warmup_steps": args.warmup_steps,
        "min_margin": args.min_margin,
        "max_train_pairs": args.max_train_pairs,
        "max_val_pairs": args.max_val_pairs,
        "use_amp": bool(args.amp and not args.no_amp),
        "amp_dtype": args.amp_dtype,
        "freeze_bottom_layers": args.freeze_bottom_layers,
        "lora_r": args.lora_r,
        "lora_impl": args.lora_impl,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": args.lora_targets,
        "policy_init_checkpoint": args.policy_init_checkpoint,
        "reference_checkpoint": args.reference_checkpoint,
        "resume_checkpoint": args.resume_checkpoint,
        "train_pairs": train_size,
        "val_pairs": val_size,
        "git_commit": _git_commit(),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[config] saved run_config.json to {out}", flush=True)


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = _init_distributed()
    is_main_process = rank == 0
    train_path = _resolve_path(args.train_preferences)
    val_path = _resolve_path(args.val_preferences)
    assert train_path is not None

    train_ds, train_loader, train_sampler = _build_loader(
        train_path,
        sample_rate=args.sample_rate,
        duration=args.duration,
        channels=args.channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_margin=args.min_margin,
        max_pairs=args.max_train_pairs,
        random_crop=True,
        shuffle=True,
        distributed=distributed,
    )
    val_loader = None
    val_sampler = None
    if val_path is not None:
        val_ds, val_loader, val_sampler = _build_loader(
            val_path,
            sample_rate=args.sample_rate,
            duration=args.duration,
            channels=args.channels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            min_margin=args.min_margin,
            max_pairs=args.max_val_pairs,
            random_crop=False,
            shuffle=False,
            distributed=distributed,
        )
    else:
        val_ds = None

    if is_main_process:
        print(f"[dist] distributed={distributed} world_size={world_size}", flush=True)
        print(f"[data] train pairs: {len(train_ds)}", flush=True)
    if is_main_process and val_ds is not None:
        print(f"[data] val pairs  : {len(val_ds)}", flush=True)
    first_batch = next(iter(train_loader))
    if is_main_process:
        print(
            f"[data] batch winner shape: {tuple(first_batch['winner_audio'].shape)} "
            f"loser shape: {tuple(first_batch['loser_audio'].shape)}",
            flush=True,
        )
    if args.dry_run_data:
        if is_main_process:
            print("[dry_run_data] loaded one preference batch successfully; exiting before model load", flush=True)
        _cleanup_distributed(distributed)
        return

    device = torch.device(f"cuda:{local_rank}") if distributed else resolve_device(args.device)
    if is_main_process:
        print(f"[model] loading policy     : {args.model_name}", flush=True)
    policy = load_musicgen_for_dpo(args.model_name, device)
    reference_device = torch.device(args.reference_device) if args.reference_device else device
    if is_main_process:
        print(f"[model] loading reference  : {args.model_name} on {reference_device}", flush=True)
    reference = load_musicgen_for_dpo(args.model_name, reference_device)

    # Optional: load only LM weights into policy from an SFT checkpoint.
    # Keeps a fresh optimizer so this is not a resume — it initialises policy from SFT.
    if args.policy_init_checkpoint:
        ckpt_path = _resolve_path(args.policy_init_checkpoint) or args.policy_init_checkpoint
        if is_main_process:
            print(f"[model] loading policy LM weights from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        lm_module(policy).load_state_dict(ckpt["lm_state_dict"])
        if is_main_process:
            training_type = ckpt.get("training_type") or ckpt.get("training_mode", "unknown")
            print(f"[model] policy init checkpoint type={training_type} "
                  f"epoch={ckpt.get('epoch')} step={ckpt.get('global_step')}", flush=True)

    if args.lora_r > 0:
        from dpo.musicgen.lora import LoRAConfig, apply_lora_to_linear_layers, apply_peft_lora_to_linear_layers

        targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
        apply_fn = apply_peft_lora_to_linear_layers if args.lora_impl == "peft" else apply_lora_to_linear_layers
        n_lora = apply_fn(
            lm_module(policy),
            LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, target_modules=targets),
        )
        if is_main_process:
            print(f"[lora] impl={args.lora_impl} rank={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} "
                  f"wrapped_linear_layers={n_lora} targets={targets}", flush=True)

    # Optional: load LM weights into reference model (not recommended; default is base model).
    if args.reference_checkpoint:
        ref_ckpt_path = _resolve_path(args.reference_checkpoint) or args.reference_checkpoint
        if is_main_process:
            print(f"[model] loading reference LM weights from {ref_ckpt_path}", flush=True)
        ref_ckpt = torch.load(ref_ckpt_path, map_location=reference_device)
        lm_module(reference).load_state_dict(ref_ckpt["lm_state_dict"])

    if args.dry_run_model:
        from dpo.musicgen.trainer import encode_audio_codes, musicgen_sequence_logprob

        first_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in first_batch.items()}
        codes = encode_audio_codes(policy, first_batch["winner_audio"])
        logp = musicgen_sequence_logprob(policy, codes, list(first_batch["prompt"]))
        if is_main_process:
            print(f"[dry_run_model] codes shape: {tuple(codes.shape)} logp shape: {tuple(logp.shape)}", flush=True)
        _cleanup_distributed(distributed)
        return

    output_dir = _resolve_path(args.output_dir) or args.output_dir
    if is_main_process:
        _save_run_config(
            args,
            output_dir,
            train_size=len(train_ds),
            val_size=len(val_ds) if val_ds is not None else None,
        )

    cfg = MusicGenDPOConfig(
        model_name=args.model_name,
        beta=args.beta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        device=str(device),
        use_amp=bool(args.amp and not args.no_amp),
        amp_dtype=args.amp_dtype,
        freeze_bottom_layers=args.freeze_bottom_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    wandb_run = _init_wandb(args, train_size=len(train_ds), val_size=len(val_ds) if val_ds is not None else None) if is_main_process else None
    try:
        train_musicgen_dpo(
            policy,
            reference,
            train_loader,
            cfg=cfg,
            val_loader=val_loader,
            output_dir=output_dir,
            log_every=args.log_every,
            resume_checkpoint=_resolve_path(args.resume_checkpoint),
            wandb_run=wandb_run,
            is_main_process=is_main_process,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
