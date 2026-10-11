# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from DiffusionNFT's scripts/train_nft_sd3.py
# (https://github.com/NVlabs/DiffusionNFT) for ACE-Step music generation with
# audio-language-model rubric rewards.
"""DiffusionNFT training of ACE-Step with a 7-dimension rubric (ALM) reward.

Launch from the repository root, e.g.::

    torchrun --nproc_per_node 4 -m diffusionnft.train \
        --config diffusionnft/config.py:rubric
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
import numpy as np
from diffusionnft.stat_tracking import PerPromptStatTracker
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import tqdm as tqdm_module
import tempfile
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast
from typing import List, Optional, Union, Tuple

from diffusionnft.pipeline import ACEStepSDEPipeline
from diffusionnft.scheduler import FlowMatchEulerDiscreteSDEScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from rewards.nft_reward import multi_score
from diffusionnft.ema import EMAModuleWrapper

from acestep.apg_guidance import (
    apg_forward,
    MomentumBuffer,
    cfg_forward,
    cfg_zero_star,
    cfg_double_condition_forward,
)


def tqdm(*args, **kwargs):
    if os.getenv("DISABLE_TQDM", "0") == "1":
        kwargs["disable"] = True
    return tqdm_module.tqdm(*args, dynamic_ncols=True, **kwargs)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "diffusionnft/config.py:rubric", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    timeout_seconds = int(
        os.getenv("RUBRIC_DISTRIBUTED_TIMEOUT_SECONDS", "10800")
    )
    if timeout_seconds <= 0:
        raise ValueError("RUBRIC_DISTRIBUTED_TIMEOUT_SECONDS must be positive")
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

class PromptDataset(Dataset):
    """Text prompts of one split, from ``load_prompts`` in ``diffusionnft/dataset.py``."""

    def __init__(self, path, split="train"):
        from diffusionnft.dataset import load_prompts

        self.prompts = [str(prompt) for prompt in load_prompts(path, split)]
        if not self.prompts:
            raise ValueError(f"load_prompts({path!r}, {split!r}) returned no prompts")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(pipeline, prompts, lyrics="[inst]", duration=15.0, text_max_length=33, padding="max_length"):
    with torch.no_grad():
        def _ensure_list(x, *, length_if_none: int = 1) -> List[str]:
            if x is None:
                return [""] * length_if_none
            if isinstance(x, list):
                return x
            return [x]  
        
        p_list = _ensure_list(prompts)
        B = len(p_list)

        l_list = _ensure_list(lyrics, length_if_none=B)
        if len(l_list) == 1 and B > 1:
            l_list = l_list * B
        assert len(l_list) == B, f"lyrics length ({len(l_list)}) must match prompts ({B})"

        batch_size = B

        # prompt embedding
        # NOTE: change max_length to something else later, for now the value 33 is set because that is the longest prompt in the dataset
        encoder_text_hidden_states, text_attention_mask = pipeline.get_text_embeddings(p_list, text_max_length=text_max_length, padding=padding)

        # speaker embeddings
        speaker_embeds = torch.zeros(batch_size, 512).to(pipeline.device).to(pipeline.dtype)

        # lyrics
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(pipeline.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(pipeline.device).long()
        if len(lyrics) > 0:
            lyric_token_idx = pipeline.tokenize_lyrics(lyrics, debug=False)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = (
                torch.tensor(lyric_token_idx)
                .unsqueeze(0)
                .to(pipeline.device)
                .repeat(batch_size, 1)
            )
            lyric_mask = (
                torch.tensor(lyric_mask)
                .unsqueeze(0)
                .to(pipeline.device)
                .repeat(batch_size, 1)
            )
        # calculate the conditional and unconditional input embeddings
        frame_length = int(duration * 44100 / 512 / 8)
        # attention_mask = torch.ones(batch_size, frame_length, device=pipeline.device, dtype=pipeline.dtype)

        # conditional
        encoder_hidden_states, encoder_hidden_mask = pipeline.ace_step_transformer.encode(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embeds,
            lyric_token_idx,
            lyric_mask,
        )

        # unconditional
        encoder_hidden_states_null, _ = pipeline.ace_step_transformer.encode(
            torch.zeros_like(encoder_text_hidden_states),
            text_attention_mask,
            torch.zeros_like(speaker_embeds),
            torch.zeros_like(lyric_token_idx),
            lyric_mask,
        )

    return encoder_hidden_states, encoder_hidden_mask, encoder_hidden_states_null


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def eval_fn(
    pipeline,
    test_dataloader,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    enable_amp,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
    save_audio_dir,
):
    # NOTE: use ema parameters for evaluation, store_temp is turned on to ensure restoring live parameters
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    last_eval_pred_wavs = None
    last_eval_prompts = None
    last_eval_rewards_np = None

    for test_batch in tqdm(
            eval_loader,
            desc="Eval: ",
            disable=not is_main_process(rank),
            position=0,
        ):
        prompts, prompt_metadata = test_batch

         # pre-computing
        encoder_hidden_states, encoder_hidden_mask, encoder_hidden_states_null = compute_text_embeddings(
            pipeline,
            prompts,
            lyrics="[inst]",
            duration=config.duration,
            text_max_length=getattr(config, "text_max_length", 33),
        )

        with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
            with torch.no_grad():
                pred_wavs, _, _, _ = pipeline(
                    prompts=prompts,
                    lyrics="[inst]",
                    audio_duration=config.duration,
                    infer_step=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    scheduler_type="euler",
                    cfg_type="cfg",
                    omega_scale=0.0,
                    # manual_seeds=manual_seeds,
                    guidance_interval=1.0,
                    use_erg_tag=False,
                    use_erg_lyric=False,
                    use_erg_diffusion=False,
                    save_path=save_audio_dir if getattr(config, "save_eval_audio_locally", False) else None,
                    noise_level=0.0,
                    # pre-computing
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    encoder_hidden_states_null=encoder_hidden_states_null
                )

        rewards = executor.submit(reward_fn, pred_wavs, prompts, prompt_metadata, only_strict=False)

        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        if is_main_process(rank):
            if isinstance(pred_wavs, torch.Tensor):
                last_eval_pred_wavs = pred_wavs.detach().cpu()
            elif isinstance(pred_wavs, (list, tuple)):
                last_eval_pred_wavs = [
                    w.detach().cpu() if isinstance(w, torch.Tensor) else w for w in pred_wavs
                ]
            else:
                last_eval_pred_wavs = pred_wavs
            last_eval_prompts = list(prompts)
            last_eval_rewards_np = {k: np.asarray(v, dtype=np.float64) for k, v in rewards.items()}

        for key, value in rewards.items():
            # NOTE: gather rewards from different gpus so each gpu see the same rewards
            rewards_gather = gather_tensor_to_all(torch.as_tensor(value, device=device), world_size)
            all_rewards[key].append(rewards_gather)

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if is_main_process(rank):
        sr = getattr(config, "sample_rate", 48000)
        eval_audios = []
        if last_eval_pred_wavs is not None and last_eval_prompts is not None:
            if isinstance(last_eval_pred_wavs, torch.Tensor):
                wav_list = [last_eval_pred_wavs[i] for i in range(last_eval_pred_wavs.shape[0])]
            else:
                wav_list = list(last_eval_pred_wavs)
            num_to_log = min(15, len(wav_list), len(last_eval_prompts))
            if last_eval_rewards_np:
                num_to_log = min(
                    num_to_log,
                    min(len(v) for v in last_eval_rewards_np.values()),
                )
            for idx in range(num_to_log):
                wav = wav_list[idx]
                if isinstance(wav, torch.Tensor):
                    wav = wav.detach().cpu().float().numpy()
                if wav.ndim == 2:
                    if wav.shape[0] < wav.shape[1]:
                        wav = wav.mean(axis=0)
                    else:
                        wav = wav.mean(axis=1)
                elif wav.ndim > 2:
                    wav = np.reshape(wav, (-1,))
                wav = np.clip(wav, -1.0, 1.0).astype(np.float32)

                p = last_eval_prompts[idx]
                cap_prompt = p[:200] if isinstance(p, str) else str(p)[:200]
                if last_eval_rewards_np:
                    reward_row = {k: last_eval_rewards_np[k][idx] for k in last_eval_rewards_np}
                    tail = " | ".join(
                        f"{k}: {float(v):.2f}" for k, v in reward_row.items() if v != -10
                    )
                    caption = f"{cap_prompt} | {tail}" if tail else cap_prompt
                else:
                    caption = cap_prompt
                eval_audios.append(wandb.Audio(wav, sample_rate=sr, caption=caption))

        for key, value in all_rewards.items():
            print(key, value.shape)

        eval_means = {
            key: np.mean(value[value != -10]) for key, value in all_rewards.items()
        }
        log_payload = {
            **{f"eval_reward_{key}": value for key, value in eval_means.items()},
        }
        if eval_audios:
            log_payload["eval_audios"] = eval_audios
        wandb.log(log_payload, step=global_step)
    else:
        eval_means = {
            key: np.mean(value[value != -10]) for key, value in all_rewards.items()
        }
    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()
    return eval_means


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb_kwargs = {
            "project": os.environ.get("WANDB_PROJECT", "acestep-diffusionnft-rubric"),
            "name": config.run_name,
            "config": config.to_dict(),
            "dir": log_dir,
        }
        if os.environ.get("WANDB_ENTITY"):
            wandb_kwargs["entity"] = os.environ["WANDB_ENTITY"]
        wandb.init(**wandb_kwargs)
    logger.info(f"\n{config}")

    save_audio_dir = None
    if getattr(config, "save_eval_audio_locally", False):
        save_audio_dir = os.path.join(config.logdir, config.run_name, "eval_audio")
        # ACEStep pipeline treats save_path as a directory only if it already exists; create it on every rank before eval.
        os.makedirs(save_audio_dir, exist_ok=True)

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = ACEStepSDEPipeline(
        checkpoint_dir=config.pretrained.model,
        device_id=local_rank,
        dtype="float32",
    )

    pipeline.load_checkpoint()

    scheduler = FlowMatchEulerDiscreteSDEScheduler(num_train_timesteps=1000, shift=3.0)

    pipeline.ace_step_transformer.requires_grad_(False)
    pipeline.music_dcae.requires_grad_(False)
    pipeline.text_encoder_model.requires_grad_(False)

    tokenizer = pipeline.text_tokenizer

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    assert text_encoder_dtype == torch.float32, print("Only supports fp32 for now")

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.music_dcae.to(device, dtype=torch.float32)
    pipeline.text_encoder_model.to(device, dtype=text_encoder_dtype)    
    pipeline.ace_step_transformer.to(device)

    ##### check if lora target modules exist
    print("\n" + "="*60)
    print("Checking transformer structure...")
    print("="*60)
    
    print("\nAll module names in transformer:")
    for name, module in pipeline.ace_step_transformer.named_modules():
        if 'attn' in name and any(x in name for x in ['proj', 'to_', 'out']):
            print(f"  {name}: {type(module).__name__}")
    
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    
    print("\nChecking if target modules exist:")
    all_module_names = [name for name, _ in pipeline.ace_step_transformer.named_modules()]
    for target in target_modules:
        # 检查是否有任何模块名包含这个 target
        matches = [name for name in all_module_names if target in name]
        if matches:
            print(f"  ✓ '{target}' found in {len(matches)} modules")
        else:
            print(f"  ✗ '{target}' NOT FOUND - this is a problem!")
    
    print("="*60 + "\n")

    transformer = pipeline.ace_step_transformer.to(device)

    ###### check if lora target modules exist
    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)

        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")

    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    rubric_config = getattr(config, "rubric_reward", None)
    reward_device = device
    reward_fn = multi_score(reward_device, config.reward_fn, rubric_config=rubric_config)
    # Rubric judges are large and their generation methods are not thread-safe.
    # Reuse one locked instance rather than loading a duplicate eval copy.
    eval_reward_fn = reward_fn

    if config.prompt_fn == "rubric":
        train_dataset = PromptDataset(config.dataset, "train")
        test_dataset = PromptDataset(config.dataset, "val")

        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_audio_per_prompt,
            num_replicas=world_size,
            rank=rank,
            seed=config.seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=PromptDataset.collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=PromptDataset.collate_fn,
            shuffle=False,
            num_workers=1,
        )

    else:
        raise NotImplementedError(f"Unsupported prompt_fn: {config.prompt_fn!r}")
    
    if config.sample.num_audio_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=2)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            transformer_ddp.module.load_adapter(lora_path, adapter_name="old", is_trainable=False)
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # Extract epoch and step from checkpoint name, e.g., "checkpoint-1000" -> global_step = 1000
        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            # This trainer performs one optimizer update per outer epoch for the
            # rubric configurations. Resume the outer loop at the checkpoint
            # step so a checkpoint-125 continuation finishes at step 200 rather
            # than silently performing another 200 updates.
            first_epoch = global_step
            logger.info(f"Resumed global_step and first_epoch to {global_step}.")
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting global_step from 0."
            )
            global_step = 0

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.ace_step_transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)

            encoder_hidden_states, encoder_hidden_mask, encoder_hidden_states_null = compute_text_embeddings(
                pipeline,
                prompts,
                lyrics="[inst]",
                duration=config.duration,
                text_max_length=getattr(config, "text_max_length", 33),
            )
            prompt_ids = tokenizer(
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and config.eval_freq > 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_metrics = eval_fn(
                    pipeline,
                    test_dataloader,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    enable_amp,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                    save_audio_dir,
                )
                if global_step == 0:
                    max_equal_rate = float(
                        getattr(config, "rubric_max_initial_equal_fields_rate", 1.0)
                    )
                    rubric_cfg = getattr(config, "rubric_reward", None)
                    providers = list(getattr(rubric_cfg, "providers", [])) if rubric_cfg else []
                    for provider in providers:
                        metric_key = f"rubric_{provider}_equal_fields"
                        equal_rate = float(eval_metrics.get(metric_key, 0.0))
                        logger.info(
                            "Initial %s equal-fields rate %.6f (maximum %.6f)",
                            provider,
                            equal_rate,
                            max_equal_rate,
                        )
                        if equal_rate > max_equal_rate:
                            raise RuntimeError(
                                f"Initial {provider} all-fields-equal rate {equal_rate:.3%} "
                                f"exceeds the configured maximum {max_equal_rate:.3%}; "
                                "aborting before the first optimizer update."
                            )

            if i == 0 and epoch % config.save_freq == 0 and is_main_process(rank) and not config.debug:
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():

                    pred_wavs, latents, log_probs, _ = pipeline(
                        prompts=prompts,
                        lyrics="[inst]",
                        audio_duration=config.duration,
                        infer_step=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        scheduler_type="euler",
                        cfg_type="cfg",
                        omega_scale=0.0,
                        # manual_seeds=manual_seeds,
                        guidance_interval=1.0,
                        use_erg_tag=False,
                        use_erg_lyric=False,
                        use_erg_diffusion=False,
                        save_path=None,
                        noise_level=config.sample.noise_level,
                        # pre-computing
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_mask=encoder_hidden_mask,
                        encoder_hidden_states_null=encoder_hidden_states_null
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            # timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=config.sample.num_steps,
                device=pipeline.device,
                timesteps=None,
            )

            timesteps = timesteps.repeat(config.sample.train_batch_size, 1)

            # compute rewards asynchronously
            rewards_future = executor.submit(reward_fn, pred_wavs, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_hidden_mask": encoder_hidden_mask,
                    "encoder_hidden_states_null": encoder_hidden_states_null,
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "rewards_future": rewards_future,  # Store future
                }
            )

        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        # Logging audio (main process) — last sampling batch on this rank
        if epoch % 10 == 0 and is_main_process(rank):
            sr = getattr(config, "sample_rate", 48000)
            wavs_to_log = pred_wavs
            prompts_to_log = prompts
            rewards_to_log = collated_samples["rewards"]["avg"][-len(wavs_to_log) :].cpu()

            num_to_log = min(15, len(wavs_to_log))
            audios = []
            for idx in range(num_to_log):
                wav = wavs_to_log[idx]
                if isinstance(wav, torch.Tensor):
                    wav = wav.detach().cpu().float().numpy()
                if wav.ndim == 2:
                    if wav.shape[0] < wav.shape[1]:
                        wav = wav.mean(axis=0)
                    else:
                        wav = wav.mean(axis=1)
                elif wav.ndim > 2:
                    wav = np.reshape(wav, (-1,))
                wav = np.clip(wav, -1.0, 1.0).astype(np.float32)

                p = prompts_to_log[idx]
                cap_prompt = p[:200] if isinstance(p, str) else str(p)[:200]
                audios.append(
                    wandb.Audio(
                        wav,
                        sample_rate=sr,
                        caption=f"{cap_prompt} | avg: {rewards_to_log[idx].item():.2f}",
                    )
                )
            wandb.log({"train_audios": audios}, step=global_step)
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.text_tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            # Stat tracker update expects numpy arrays for rewards
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["encoder_hidden_states"])

                if config.sample.guidance_scale > 1.0:
                    assert False, "Not implemented for guidance scale > 1.0"
                else:
                    encoder_hidden_states = train_sample_batch["encoder_hidden_states"]
                    encoder_hidden_mask = train_sample_batch["encoder_hidden_mask"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt.clone(),
                                attention_mask=torch.ones_like(xt)[:, 0, 0, :],
                                encoder_hidden_states=encoder_hidden_states.clone(),
                                encoder_hidden_mask=encoder_hidden_mask.clone(),
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                output_length=xt.shape[-1],
                                return_dict=True,
                            ).detach()
                        transformer_ddp.module.set_adapter("default")

                        # Run reference forward before the grad-requiring forward to avoid autograd version conflicts.
                        with torch.no_grad():
                            if config.use_lora:
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt.clone(),
                                        attention_mask=torch.ones_like(xt)[:, 0, 0, :],
                                        encoder_hidden_states=encoder_hidden_states.clone(),
                                        encoder_hidden_mask=encoder_hidden_mask.clone(),
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        output_length=xt.shape[-1],
                                        return_dict=True,
                                    ).detach()
                                transformer_ddp.module.set_adapter("default")
                            else:
                                assert False

                        # prediction v (keep grad for policy loss backprop)
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            attention_mask=torch.ones_like(xt)[:, 0, 0, :],
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_mask=encoder_hidden_mask,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            output_length=xt.shape[-1],
                            return_dict=True,
                        )
                    loss_terms = {}
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # normalize advantage
                    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    loss += config.train.beta * torch.mean(kl_div_loss)
                    kl_div_loss = torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

    # The periodic checkpoint/evaluation happens before each epoch's optimizer
    # update. Persist and evaluate the actual bounded-run endpoint as well, so a
    # 200-update run produces checkpoint-200 and metrics at step 200 rather than
    # ending with checkpoint/evaluation 175/150.
    if not config.debug:
        if config.eval_freq > 0:
            eval_fn(
                pipeline,
                test_dataloader,
                config,
                device,
                rank,
                world_size,
                global_step,
                eval_reward_fn,
                executor,
                enable_amp,
                mixed_precision_dtype,
                ema,
                transformer_trainable_parameters,
                save_audio_dir,
            )
        if is_main_process(rank):
            save_ckpt(
                config.save_dir,
                transformer_ddp,
                global_step,
                rank,
                ema,
                transformer_trainable_parameters,
                config,
                optimizer,
                scaler,
            )

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
