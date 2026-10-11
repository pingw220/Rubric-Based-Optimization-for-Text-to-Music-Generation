#!/usr/bin/env python
"""ACE-Step DPO worker, launched by dpo/acestep/train.py (requires the ACE-Step package).

Implements flow-matching DPO on the ACEStepTransformer2DModel:

* per-sample flow-matching loss (mirrors the official trainer.py run_step,
  including the SD3 preconditioning and attention-mask weighting, with
  reduction kept per-sample instead of batch mean),
* trainable policy = base transformer + LoRA adapter (diffusers/peft),
* frozen reference = the same base weights with adapters disabled
  (mathematically identical to a separate frozen copy, without 2x memory),
* shared timestep for winner/loser and shared noise for policy/reference.

Modes:
  audit  -- feasibility audit (per-sample loss shape, policy/ref margins,
            one fake optimizer step on LoRA params).
  train  -- real DPO training for --max_steps optimizer steps.

Spec JSON:
{
  "mode": "train", "repo_path": ..., "checkpoint_path": ...,
  "lora_config": {"r": 16, "lora_alpha": 32, "target_modules": [...]},
  "pairs": [{"prompt": ..., "winner": "/x.wav", "loser": "/y.wav"}],
  "probe": {"prompt": ..., "audio": "/x.wav"},
  "beta": 0.1, "lr": 1e-5, "max_steps": 10, "grad_accum": 4,
  "seed": 42, "output_dir": ..., "save_every": 0, "shift": 3.0
}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path


def _init_wandb(spec: dict, extra: dict | None = None):
    """Init a wandb run if spec["wandb"] is set (online by default). Returns the
    run or None. Auth comes from ~/.netrc; never logs secrets."""
    wcfg = spec.get("wandb")
    if not wcfg:
        return None
    try:
        import os as _os
        import wandb
        _os.environ.setdefault("WANDB_MODE", wcfg.get("mode", "online"))
        config = {k: spec.get(k) for k in ("beta", "lr", "max_steps", "grad_accum", "seed", "lora_config")}
        config.update(extra or {})
        return wandb.init(
            project=wcfg.get("project", "acestep-dpo"),
            entity=wcfg.get("entity"),
            name=wcfg.get("name"),
            group=wcfg.get("group"),
            tags=wcfg.get("tags"),
            mode=wcfg.get("mode", "online"),
            config=config,
        )
    except Exception as exc:  # never let logging break training
        print(f"[wandb] init failed ({exc}); continuing without wandb", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    repo_path = spec["repo_path"]
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    import torch
    import torch.nn.functional as F

    # torchaudio >= 2.9 delegates load()/save() to torchcodec; fall back to
    # soundfile if torchcodec or its ffmpeg libs are broken in this env.
    import torchaudio

    _orig_load = torchaudio.load

    def _load_robust(uri, *args, **kwargs):
        try:
            return _orig_load(uri, *args, **kwargs)
        except Exception:
            import soundfile as sf

            data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _load_robust

    from acestep.pipeline_ace_step import ACEStepPipeline
    from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(int(spec.get("seed", 42)))

    report: dict = {"mode": spec["mode"], "device": str(device), "steps": []}
    out_dir = Path(spec["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load base components via the official pipeline loader ----
    pipe = ACEStepPipeline(checkpoint_dir=spec["checkpoint_path"], dtype="bfloat16" if dtype == torch.bfloat16 else "float32")
    pipe.load_checkpoint(spec["checkpoint_path"])
    transformer = pipe.ace_step_transformer.to(device)
    dcae = pipe.music_dcae.to(device)
    report["base_loaded"] = True

    # ---- LoRA policy adapter (base weights stay frozen) ----
    from peft import LoraConfig

    lora_cfg_dict = dict(spec["lora_config"])
    lora_cfg = LoraConfig(**lora_cfg_dict)
    adapter_name = spec.get("adapter_name", "dpo_lora")
    transformer.add_adapter(adapter_config=lora_cfg, adapter_name=adapter_name)
    for name, param in transformer.named_parameters():
        param.requires_grad = "lora" in name
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    report["num_lora_params"] = sum(p.numel() for p in lora_params)
    report["num_total_params"] = sum(p.numel() for p in transformer.parameters())
    assert report["num_lora_params"] > 0, "no trainable LoRA params after add_adapter"

    # ---- flow-matching scheduler (matches trainer.py defaults) ----
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=float(spec.get("shift", 3.0)),
    )

    def sample_timesteps(bsz: int) -> torch.Tensor:
        # logit-normal density, as in trainer.Pipeline.get_timestep
        u = torch.sigmoid(torch.normal(mean=0.0, std=1.0, size=(bsz,), device=device))
        indices = (u * scheduler.config.num_train_timesteps).long().clamp(
            0, scheduler.config.num_train_timesteps - 1
        )
        return scheduler.timesteps.to(device)[indices]

    def get_sd3_sigmas(timesteps: torch.Tensor, n_dim: int) -> torch.Tensor:
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ---- conditioning / latent encoding ----
    @torch.no_grad()
    def encode_audio(path: str) -> torch.Tensor:
        audio, sr = dcae.load_audio(path)
        audio = audio.unsqueeze(0).to(device=device, dtype=dtype)
        latents, _ = dcae.encode(audio, sr=sr)
        return latents  # [1, 8, 16, T]

    @torch.no_grad()
    def encode_prompt(prompt: str, lyrics: str = "[instrumental]") -> dict:
        text_hidden, text_mask = pipe.get_text_embeddings([prompt])
        token_idx = pipe.tokenize_lyrics(lyrics)
        lyric_token_idx = torch.tensor([token_idx], device=device).long()
        lyric_mask = torch.ones_like(lyric_token_idx, device=device).long()
        return {
            "encoder_text_hidden_states": text_hidden.to(device=device, dtype=dtype),
            "text_attention_mask": text_mask.to(device),
            "speaker_embeds": torch.zeros(1, 512, device=device, dtype=dtype),
            "lyric_token_idx": lyric_token_idx,
            "lyric_mask": lyric_mask,
        }

    def compute_loss_per_sample(latents: torch.Tensor, cond: dict, timesteps: torch.Tensor, noise: torch.Tensor, return_debug: bool = False) -> dict:
        """Unreduced per-sample flow loss, mirroring trainer.py run_step.

        latents: [B, 8, 16, T]; timesteps: [B]; noise: same shape as latents.
        Returns {"loss": scalar, "loss_per_sample": [B], "timestep": ..., "noise": ...}.
        """
        bsz = latents.shape[0]
        attention_mask = torch.ones(bsz, latents.shape[-1], device=device, dtype=dtype)
        sigmas = get_sd3_sigmas(timesteps, latents.ndim)
        noisy = sigmas * noise + (1.0 - sigmas) * latents
        out = transformer(
            hidden_states=noisy,
            attention_mask=attention_mask,
            encoder_text_hidden_states=cond["encoder_text_hidden_states"].expand(bsz, -1, -1),
            text_attention_mask=cond["text_attention_mask"].expand(bsz, -1),
            speaker_embeds=cond["speaker_embeds"].expand(bsz, -1),
            lyric_token_idx=cond["lyric_token_idx"].expand(bsz, -1),
            lyric_mask=cond["lyric_mask"].expand(bsz, -1),
            timestep=timesteps.to(device).to(dtype),
        )
        model_pred = out.sample * (-sigmas) + noisy  # SD3 preconditioning -> x0 prediction
        mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(-1, latents.shape[1], latents.shape[2], -1)
        pred_flat = (model_pred * mask).reshape(bsz, -1)
        target_flat = (latents * mask).reshape(bsz, -1)
        loss_unreduced = F.mse_loss(pred_flat, target_flat, reduction="none")
        loss_per_sample = loss_unreduced.mean(dim=1) * mask.reshape(bsz, -1).mean(dim=1)
        result = {
            "loss": loss_per_sample.mean(),
            "loss_per_sample": loss_per_sample,
            "timestep": timesteps,
            "noise": noise,
        }
        if return_debug:
            result["debug"] = {"sigmas": sigmas.flatten().tolist(), "latent_shape": list(latents.shape)}
        return result

    def reference_loss_per_sample(latents, cond, timesteps, noise):
        transformer.disable_adapters()
        try:
            with torch.no_grad():
                return compute_loss_per_sample(latents, cond, timesteps, noise)["loss_per_sample"]
        finally:
            transformer.enable_adapters()

    beta = float(spec.get("beta", 0.1))

    if spec["mode"] == "audit":
        probe = spec["probe"]
        latents = encode_audio(probe["audio"])
        cond = encode_prompt(probe["prompt"])
        report["probe_audio"] = probe["audio"]
        report["latent_shape"] = list(latents.shape)

        # per-sample loss shape check with batch_size=2
        latents2 = latents.expand(2, -1, -1, -1).contiguous()
        t2 = sample_timesteps(2)
        n2 = torch.randn_like(latents2)
        with torch.no_grad():
            probe_out = compute_loss_per_sample(latents2, cond, t2, n2, return_debug=True)
        shape_ok = tuple(probe_out["loss_per_sample"].shape) == (2,)
        report["loss_per_sample_shape"] = list(probe_out["loss_per_sample"].shape)
        report["loss_per_sample_values"] = [float(x) for x in probe_out["loss_per_sample"].float()]
        report["per_sample_loss_ok"] = shape_ok

        # policy + reference winner/loser losses with shared timestep/noise
        t1 = sample_timesteps(1)
        noise = torch.randn_like(latents)
        loss_pw = compute_loss_per_sample(latents, cond, t1, noise)["loss_per_sample"]
        loss_pl = compute_loss_per_sample(latents, cond, t1, torch.randn_like(latents))["loss_per_sample"]
        loss_rw = reference_loss_per_sample(latents, cond, t1, noise)
        loss_rl = reference_loss_per_sample(latents, cond, t1, torch.randn_like(latents))
        policy_margin = -loss_pw + loss_pl
        ref_margin = -loss_rw + loss_rl
        dpo_loss = -F.logsigmoid(beta * (policy_margin - ref_margin)).mean()
        report["policy_loss_w"] = float(loss_pw.float().mean())
        report["policy_loss_l"] = float(loss_pl.float().mean())
        report["ref_loss_w"] = float(loss_rw.float().mean())
        report["ref_loss_l"] = float(loss_rl.float().mean())
        report["dpo_loss"] = float(dpo_loss.float())

        # one fake optimizer step on LoRA params only
        optimizer = torch.optim.AdamW(lora_params, lr=float(spec.get("lr", 1e-5)))
        dpo_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report["fake_optimizer_step_ok"] = True
        report["grad_norm"] = float(grad_norm)
        report["status"] = "FEASIBLE" if shape_ok else "NOT_FEASIBLE"
    else:
        pairs = spec["pairs"]
        assert pairs, "no DPO pairs provided"
        max_steps = int(spec.get("max_steps", 10))
        grad_accum = max(1, int(spec.get("grad_accum", 1)))
        lr = float(spec.get("lr", 1e-5))
        optimizer = torch.optim.AdamW(lora_params, lr=lr)
        log_path = out_dir / "dpo_train_log.jsonl"
        log_handle = log_path.open("a", encoding="utf-8")

        wb = _init_wandb(spec, extra={"num_lora_params": report["num_lora_params"],
                                                    "num_train_pairs": len(pairs)})

        cache: dict = {}

        def get_encoded(prompt: str, audio: str):
            key = (prompt, audio)
            if key not in cache:
                cache[key] = (encode_audio(audio), encode_prompt(prompt))
            return cache[key]

        micro = 0
        t_start = time.time()
        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            stats_acc = []
            for _ in range(grad_accum):
                pair = pairs[micro % len(pairs)]
                micro += 1
                lat_w, cond = get_encoded(pair["prompt"], pair["winner"])
                lat_l, _ = get_encoded(pair["prompt"], pair["loser"])
                # shared timestep for winner & loser; shared noise policy & ref
                t1 = sample_timesteps(1)
                noise_w = torch.randn_like(lat_w)
                noise_l = torch.randn_like(lat_l)
                loss_pw = compute_loss_per_sample(lat_w, cond, t1, noise_w)["loss_per_sample"]
                loss_pl = compute_loss_per_sample(lat_l, cond, t1, noise_l)["loss_per_sample"]
                loss_rw = reference_loss_per_sample(lat_w, cond, t1, noise_w)
                loss_rl = reference_loss_per_sample(lat_l, cond, t1, noise_l)
                policy_margin = -loss_pw + loss_pl
                ref_margin = -loss_rw + loss_rl
                logits = beta * (policy_margin - ref_margin)
                dpo_loss = -F.logsigmoid(logits).mean()
                (dpo_loss / grad_accum).backward()
                stats_acc.append(
                    {
                        "dpo_loss": float(dpo_loss.float()),
                        "policy_loss_w": float(loss_pw.float().mean()),
                        "policy_loss_l": float(loss_pl.float().mean()),
                        "ref_loss_w": float(loss_rw.float().mean()),
                        "ref_loss_l": float(loss_rl.float().mean()),
                        "policy_margin": float(policy_margin.float().mean()),
                        "reference_margin": float(ref_margin.float().mean()),
                        "preference_accuracy": float((policy_margin > ref_margin).float().mean()),
                    }
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, float(spec.get("gradient_clip_val", 1.0)))
            optimizer.step()
            mean_stats = {k: sum(s[k] for s in stats_acc) / len(stats_acc) for k in stats_acc[0]}
            entry = {
                "step": step + 1,
                **mean_stats,
                "grad_norm": float(grad_norm),
                "lr": lr,
                "elapsed_sec": round(time.time() - t_start, 1),
            }
            report["steps"].append(entry)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in entry.items() if k != "step"}, step=step + 1)
            log_handle.write(json.dumps(entry) + "\n")
            log_handle.flush()
            print(f"[dpo_worker] step {step + 1}/{max_steps} dpo_loss={entry['dpo_loss']:.4f} "
                  f"pref_acc={entry['preference_accuracy']:.2f} grad_norm={entry['grad_norm']:.3f}", flush=True)
            save_every = int(spec.get("save_every", 0))
            if save_every and (step + 1) % save_every == 0:
                ckpt_dir = out_dir / f"checkpoint_step{step + 1}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                transformer.save_lora_adapter(str(ckpt_dir), adapter_name=adapter_name)
        log_handle.close()

        final_dir = out_dir / "checkpoint_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        transformer.save_lora_adapter(str(final_dir), adapter_name=adapter_name)
        report["adapter_dir"] = str(final_dir)
        report["num_optimizer_steps"] = max_steps
        report["status"] = "TRAINED"
        if report["steps"]:
            for k in ("dpo_loss", "policy_loss_w", "policy_loss_l", "ref_loss_w", "ref_loss_l", "preference_accuracy"):
                report[f"final_{k}"] = report["steps"][-1][k]
        nonfinite = [s["step"] for s in report["steps"] if not math.isfinite(s["dpo_loss"])]
        if nonfinite:
            report["status"] = "FAILED_NONFINITE_LOSS"
            report["nonfinite_steps"] = nonfinite
        if wb is not None:
            wb.summary.update({k: report.get(k) for k in report if k.startswith("final_")})
            wb.summary["status"] = report["status"]
            wb.summary["adapter_dir"] = report.get("adapter_dir")
            report["wandb_url"] = getattr(wb, "url", None)
            wb.finish()

    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[dpo_worker] done: {report['status']}", flush=True)
    if report["status"] not in {"FEASIBLE", "TRAINED"}:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
