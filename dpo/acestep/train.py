#!/usr/bin/env python
"""ACE-Step DPO training entry point.

Reads preference pairs (``load_preference_pairs`` in ``dpo/dataset.py``), writes the
JSON spec consumed by ``dpo/acestep/worker.py`` and runs the worker,
which implements the flow-matching DPO surrogate used in the paper: per-sample
flow-matching losses for the chosen and rejected latents at a shared
logit-normal timestep, shared noise between policy and reference, a LoRA adapter
as the policy and the same transformer with the adapter disabled as the
reference.

In the research code this driver ran the worker inside a dedicated conda
environment through a cluster-specific launcher; here the worker runs with the
current interpreter, which must have ACE-Step importable (``import acestep``).

Example:
  python dpo/acestep/train.py \\
    --train_pairs <your pairs> --min_margin 4.0 \\
    --checkpoint_path checkpoints/ACE-Step-v1-3.5B \\
    --output_dir outputs/acestep_dpo/example \\
    --beta 500 --lr 1e-6 --max_steps 1500 --grad_accum 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dpo.dataset import load_preference_pairs  # noqa: E402

WORKER = PROJECT_ROOT / "dpo" / "acestep" / "worker.py"


# LoRA adapter used for every ACE-Step DPO run in the paper (rank 16, alpha 32).
DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["linear_q", "linear_k", "linear_v", "to_q", "to_k", "to_v", "to_out.0"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACE-Step flow-matching DPO trainer")
    p.add_argument("--train_pairs", required=True, help="preference pairs (read by load_preference_pairs in dpo/dataset.py)")
    p.add_argument("--min_margin", type=float, default=0.0, help="drop pairs whose margin is below this value")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint_path", required=True,
                   help="ACE-Step checkpoint directory (ACE-Step downloads ACE-Step/ACE-Step-v1-3.5B into it if empty)")
    p.add_argument("--acestep_repo", default="",
                   help="Optional path to an ACE-Step source checkout to put on sys.path")
    p.add_argument("--lora_config_path", default=None,
                   help="Optional JSON with peft LoraConfig kwargs; defaults to the paper's rank-16 adapter")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_train_pairs", type=int, default=None)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run_data", action="store_true", help="Validate the pairs file and exit")
    p.add_argument("--audit", action="store_true",
                   help="Run the worker's feasibility audit (per-sample loss shape, one fake optimizer step) and exit")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument("--wandb_project", default="acestep-dpo")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_config = (
        json.loads(Path(args.lora_config_path).read_text(encoding="utf-8"))
        if args.lora_config_path
        else dict(DEFAULT_LORA_CONFIG)
    )

    pairs = [p for p in load_preference_pairs(args.train_pairs) if p.margin is None or p.margin >= args.min_margin]
    pairs = pairs[: args.max_train_pairs] if args.max_train_pairs is not None else pairs
    if not pairs:
        raise SystemExit(f"No preference pairs left in {args.train_pairs} after --min_margin {args.min_margin}")
    missing = [p for p in pairs if not Path(p.winner).exists() or not Path(p.loser).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} pairs reference missing audio files, e.g. {missing[0]}")
    if args.dry_run_data:
        print(json.dumps({"status": "data_ok", "num_train_pairs": len(pairs), "first_pair": pairs[0].__dict__}, indent=2))
        return

    spec = {
        "mode": "audit" if args.audit else "train",
        "repo_path": str(Path(args.acestep_repo).expanduser().resolve()) if args.acestep_repo else "",
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
        "lora_config": lora_config,
        "beta": args.beta,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "save_every": args.save_every,
        "seed": args.seed,
        "output_dir": str(out_dir / "audit" if args.audit else out_dir),
        "probe": {"audio": pairs[0].winner, "prompt": pairs[0].prompt},
        "pairs": [{"prompt": p.prompt, "winner": p.winner, "loser": p.loser} for p in pairs],
    }
    if args.wandb and not args.audit:
        spec["wandb"] = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_run_name or f"acestep-{out_dir.name}-b{args.beta}-lr{args.lr}",
            "mode": args.wandb_mode,
            "tags": ["acestep", f"beta{args.beta}", f"lr{args.lr}"],
        }

    with tempfile.TemporaryDirectory(prefix="acestep_dpo_") as tmp:
        spec_path = Path(tmp) / "spec.json"
        result_path = Path(tmp) / "result.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        started = time.time()
        proc = subprocess.run(
            [sys.executable, str(WORKER), "--spec", str(spec_path), "--result", str(result_path)],
            env={**os.environ, "ACESTEP_CHECKPOINTS_DIR": spec["checkpoint_path"]},
        )
        report = {"worker_exit_code": proc.returncode, "worker_elapsed_sec": round(time.time() - started, 1)}
        if result_path.exists():
            report.update(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            report["status"] = "WORKER_CRASHED"

    report["train_pairs"] = str(args.train_pairs)
    report["num_train_pairs"] = len(pairs)
    report_path = out_dir / ("dpo_audit_report.json" if args.audit else "dpo_train_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report.get("status"),
        "num_optimizer_steps": report.get("num_optimizer_steps"),
        "final_dpo_loss": report.get("final_dpo_loss"),
        "adapter_dir": report.get("adapter_dir"),
        "report": str(report_path),
    }, indent=2))
    if report.get("status") not in {"TRAINED", "FEASIBLE"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
