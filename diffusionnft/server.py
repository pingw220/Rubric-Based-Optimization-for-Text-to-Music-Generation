#!/usr/bin/env python3
"""Rubric rater server for DiffusionNFT training.

Loads an audio-language model (the rater) on its own GPUs and scores the clips
that the trainer writes to a shared directory, using the rubric prompts
(``rubrics/general.yaml`` by default). The trainer side is
``rewards/nft_reward.py:RaterServerClient``.

Choose the rater with ``--model``:

* ``qwen3_omni``, ``music_flamingo``, ``qwen2_audio``: built-in raters
  (``--model_id`` overrides the Hugging Face checkpoint);
* ``my_module:MyRater``: your own rater class, importable from the repository
  root (run the server from there). It is constructed as ``MyRater(model_id=..., device=..., dtype=...,
  max_new_tokens=..., prompting=..., rubric=...)`` and must provide
  ``score(prompt, audio_path) -> {dimension: score in [0, 10]}`` for every
  dimension of the rubric; an optional ``load()`` is called once at startup.
  ``prompting`` is ``"joint"`` or ``"per_dimension"`` and ``rubric`` is a
  ``rewards.rubric.RewardRubric`` that generates the prompts.

Examples:

    python -m diffusionnft.server --dir /shared/rater --model qwen3_omni --prompting joint --gpus_per_worker 4
    python -m diffusionnft.server --dir /shared/rater --model music_flamingo --prompting per_dimension --num_workers 4

Each worker gets ``--gpus_per_worker`` GPUs; ``--num_workers`` workers share the
request queue. Create ``<dir>/STOP`` to shut the server down.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path


# Make the repository root importable when this file is run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUILT_IN_MODEL_IDS = {
    "qwen3_omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "music_flamingo": "nvidia/music-flamingo-hf",
    "qwen2_audio": "Qwen/Qwen2-Audio-7B-Instruct",
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


class _BuiltInRater:
    """Adapts the built-in raters to ``score(prompt, audio_path) -> 0..10 scores``."""

    def __init__(self, model, model_id, device, dtype, max_new_tokens, prompting, rubric):
        per_dimension = prompting == "per_dimension"
        if model == "qwen3_omni":
            from rewards.qwen3_omni import Qwen3OmniIndependentJudge, Qwen3OmniJudge

            judge_cls = Qwen3OmniIndependentJudge if per_dimension else Qwen3OmniJudge
            # Qwen3-Omni is sharded over every GPU visible to the worker.
            self.judge = judge_cls(
                model_id=model_id, device_map="auto", dtype=dtype,
                max_new_tokens=max_new_tokens, prompt_schema=rubric.name,
            )
            self._score = self.judge.score_audio_multi    # already on the 0-10 scale
            self._scale = 1.0
        else:
            from rewards.nft_reward import MusicFlamingoJudge, Qwen2AudioJudge

            judge_cls = MusicFlamingoJudge if model == "music_flamingo" else Qwen2AudioJudge
            self.judge = judge_cls(
                model_id=model_id, device=device, dtype=dtype, max_new_tokens=max_new_tokens,
                independent_dimensions=per_dimension, prompt_schema=rubric.name,
            )
            self._score = self.judge.score               # normalized to [0, 1]
            self._scale = 10.0

    def load(self):
        self.judge._load()

    def score(self, prompt, audio_path):
        scores = self._score(prompt=prompt, audio_path=audio_path)
        return {key: self._scale * float(value) for key, value in scores.items()}


def build_rater(args, rubric):
    if args.model in BUILT_IN_MODEL_IDS:
        return _BuiltInRater(
            args.model, args.model_id or BUILT_IN_MODEL_IDS[args.model], args.device,
            args.dtype, args.max_new_tokens, args.prompting, rubric,
        )
    if ":" not in args.model:
        raise ValueError(
            f"--model must be one of {sorted(BUILT_IN_MODEL_IDS)} or 'module:ClassName', got {args.model!r}"
        )
    module_name, class_name = args.model.split(":", 1)
    rater_cls = getattr(importlib.import_module(module_name), class_name)
    return rater_cls(
        model_id=args.model_id, device=args.device, dtype=args.dtype,
        max_new_tokens=args.max_new_tokens, prompting=args.prompting, rubric=rubric,
    )


def score_request(rater, rubric, request: dict) -> dict:
    started = time.time()
    scores = rater.score(str(request["prompt"]), str(request["audio_path"]))
    missing = [key for key in rubric.keys if key not in scores]
    if missing:
        raise ValueError(f"rater returned no score for {missing}")
    return {
        "request_id": request["request_id"],
        "scores": {key: float(scores[key]) for key in rubric.keys},
        "error": None,
        "elapsed_seconds": time.time() - started,
        "finished_at": time.time(),
    }


def worker_main(args, queue_dir: Path) -> int:
    from rewards.rubric import load_rubric

    pending = queue_dir / "pending"
    processing = queue_dir / "processing"
    responses = queue_dir / "responses"
    for path in (pending, processing, responses, queue_dir / "audio"):
        path.mkdir(parents=True, exist_ok=True)

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    tag = f"[rater worker {args.worker_id}]"
    rubric = load_rubric(args.rubric)
    print(f"{tag} loading {args.model} ({args.model_id or 'default checkpoint'})", flush=True)
    rater = build_rater(args, rubric)
    if hasattr(rater, "load"):
        rater.load()
    print(f"{tag} model loaded", flush=True)

    smoke_result = None
    if args.smoke_audio:
        smoke_result = score_request(
            rater,
            rubric,
            {
                "request_id": f"startup-smoke-worker-{args.worker_id}",
                "prompt": args.smoke_prompt,
                "audio_path": str(Path(args.smoke_audio).resolve()),
            },
        )
        print(f"{tag} startup smoke scores: {smoke_result['scores']}", flush=True)

    atomic_json(
        queue_dir / f"READY.worker{args.worker_id}.json",
        {
            "worker_id": args.worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "model": args.model,
            "model_id": args.model_id,
            "prompt_mode": args.prompting,
            "rubric_schema": rubric.name,
            "prompt_hash": rubric.prompt_hash(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "smoke": smoke_result,
            "ready_at": time.time(),
        },
    )

    audit_path = queue_dir / f"scores.worker{args.worker_id}.jsonl"
    heartbeat_path = queue_dir / f"HEARTBEAT.worker{args.worker_id}.json"
    stop_path = queue_dir / "STOP"
    while True:
        atomic_json(
            heartbeat_path,
            {
                "time": time.time(),
                "hostname": socket.gethostname(),
                "worker_id": args.worker_id,
            },
        )
        request_paths = sorted(pending.glob("*.json"))
        if (stop_requested or stop_path.exists()) and not request_paths:
            break
        if not request_paths:
            time.sleep(args.poll_seconds)
            continue

        source = request_paths[0]
        claimed = processing / source.name
        try:
            os.replace(source, claimed)
        except FileNotFoundError:
            continue

        request = json.loads(claimed.read_text(encoding="utf-8"))
        try:
            response = score_request(rater, rubric, request)
        except Exception as exc:
            response = {
                "request_id": request.get("request_id"),
                "scores": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "finished_at": time.time(),
            }
        atomic_json(responses / source.name, response)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                **response,
                "worker_id": args.worker_id,
                "prompt": request.get("prompt"),
                "audio_path": request.get("audio_path"),
                "created_at": request.get("created_at"),
            }) + "\n")
        claimed.unlink(missing_ok=True)
        print(
            f"{tag} {request.get('request_id')} "
            f"elapsed={response.get('elapsed_seconds')} error={response.get('error')}",
            flush=True,
        )

    atomic_json(
        queue_dir / f"STOPPED.worker{args.worker_id}.json",
        {"worker_id": args.worker_id, "stopped_at": time.time()},
    )
    print(f"{tag} stopped cleanly", flush=True)
    return 0


def _worker_gpus(args) -> list[str]:
    """Split the visible GPUs into ``num_workers`` groups of ``gpus_per_worker``."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        gpus = [gpu for gpu in visible.split(",") if gpu.strip()]
    else:
        gpus = [str(index) for index in range(args.num_workers * args.gpus_per_worker)]
    needed = args.num_workers * args.gpus_per_worker
    if len(gpus) < needed:
        raise ValueError(
            f"{args.num_workers} workers x {args.gpus_per_worker} GPUs need {needed} GPUs, "
            f"but CUDA_VISIBLE_DEVICES lists {len(gpus)}"
        )
    return [
        ",".join(gpus[worker * args.gpus_per_worker:(worker + 1) * args.gpus_per_worker])
        for worker in range(args.num_workers)
    ]


def coordinator_main(args, queue_dir: Path) -> int:
    for path in (
        queue_dir / "audio",
        queue_dir / "pending",
        queue_dir / "processing",
        queue_dir / "responses",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for stale in (
        queue_dir / "READY.json",
        queue_dir / "FAILED.json",
        queue_dir / "STOP",
    ):
        stale.unlink(missing_ok=True)
    for stale in queue_dir.glob("READY.worker*.json"):
        stale.unlink(missing_ok=True)

    children = []
    for worker_id, gpus in enumerate(_worker_gpus(args)):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpus
        command = [
            sys.executable,
            "-m",
            "diffusionnft.server",
            "--dir", str(queue_dir),
            "--worker_id", str(worker_id),
            "--device", "cuda:0",
            "--model", args.model,
            "--model_id", args.model_id,
            "--prompting", args.prompting,
            "--rubric", args.rubric,
            "--dtype", args.dtype,
            "--max_new_tokens", str(args.max_new_tokens),
            "--poll_seconds", str(args.poll_seconds),
        ]
        if args.smoke_audio:
            command.extend(["--smoke_audio", args.smoke_audio])
        if args.smoke_prompt:
            command.extend(["--smoke_prompt", args.smoke_prompt])
        environment["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(REPO_ROOT), os.getcwd(), environment.get("PYTHONPATH", "")) if path
        )
        children.append(subprocess.Popen(command, env=environment))

    ready_paths = [
        queue_dir / f"READY.worker{worker_id}.json"
        for worker_id in range(args.num_workers)
    ]
    try:
        while not all(path.exists() for path in ready_paths):
            failures = [
                {"worker_id": index, "exit_code": child.poll()}
                for index, child in enumerate(children)
                if child.poll() is not None
            ]
            if failures:
                atomic_json(
                    queue_dir / "FAILED.json",
                    {"error": "worker exited before readiness", "workers": failures},
                )
                raise RuntimeError(f"rater workers failed before readiness: {failures}")
            time.sleep(1)

        worker_status = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in ready_paths
        ]
        atomic_json(
            queue_dir / "READY.json",
            {
                "hostname": socket.gethostname(),
                "model": args.model,
                "model_id": args.model_id,
                "rubric_schema": args.rubric,
                "prompt_hash": worker_status[0].get("prompt_hash") if worker_status else None,
                "num_workers": args.num_workers,
                "prompt_mode": args.prompting,
                "workers": worker_status,
                "ready_at": time.time(),
            },
        )
        print(f"[rater server] {args.num_workers} worker(s) ready; serving {queue_dir}", flush=True)

        while not (queue_dir / "STOP").exists():
            failures = [
                {"worker_id": index, "exit_code": child.poll()}
                for index, child in enumerate(children)
                if child.poll() is not None
            ]
            if failures:
                atomic_json(
                    queue_dir / "FAILED.json",
                    {"error": "ready worker exited", "workers": failures},
                )
                raise RuntimeError(f"rater workers exited unexpectedly: {failures}")
            atomic_json(
                queue_dir / "HEARTBEAT.json",
                {"time": time.time(), "hostname": socket.gethostname()},
            )
            time.sleep(1)

        for child in children:
            child.wait(timeout=120)
        atomic_json(queue_dir / "STOPPED.json", {"stopped_at": time.time()})
        print("[rater server] stopped cleanly", flush=True)
        return 0
    finally:
        (queue_dir / "READY.json").unlink(missing_ok=True)
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True, help="Shared directory, also given to the trainer as RATER_SERVER_DIR")
    parser.add_argument("--model", required=True,
                        help=f"one of {sorted(BUILT_IN_MODEL_IDS)}, or module:ClassName for your own rater")
    parser.add_argument("--model_id", default="", help="Model checkpoint; built-in raters default to the paper's")
    parser.add_argument("--prompting", required=True, choices=["joint", "per_dimension"],
                        help="Must match RATER_PROMPTING on the trainer")
    parser.add_argument("--rubric", default="general", help="Rubric name under rubrics/")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpus_per_worker", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--poll_seconds", type=float, default=0.2)
    parser.add_argument("--smoke_audio", default="", help="Optional clip scored once at startup")
    parser.add_argument(
        "--smoke_prompt",
        default="A coherent instrumental music piece with a clear rhythm and balanced production.",
    )
    parser.add_argument("--worker_id", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_workers <= 0 or args.gpus_per_worker <= 0:
        raise ValueError("--num_workers and --gpus_per_worker must be positive")
    if not args.model_id and args.model in BUILT_IN_MODEL_IDS:
        args.model_id = BUILT_IN_MODEL_IDS[args.model]

    queue_dir = Path(args.dir).expanduser().resolve()
    queue_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_id >= 0:
        return worker_main(args, queue_dir)
    return coordinator_main(args, queue_dir)


if __name__ == "__main__":
    raise SystemExit(main())
