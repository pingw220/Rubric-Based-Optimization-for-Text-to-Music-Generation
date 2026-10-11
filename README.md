# Rubric-Based Optimization for Text-to-Music Generation

Code for the paper **Rubric-Based Optimization for Text-to-Music Generation**
([arXiv:2610.03589](https://arxiv.org/abs/2610.03589)).

We turn natural-language evaluation criteria into rewards for text-to-music
generators. An open-weight audio-language model (ALM) listens to a generated clip,
reads the text prompt and a multidimensional rubric, and returns one score per
rubric dimension as a small JSON object. The scores are parsed, weighted, and used
either to rank candidates into preference pairs for DPO or as an online scalar
reward for DiffusionNFT.

## What is in this repository

- **DPO** for MusicGen-small (LoRA) and for ACE-Step (flow-matching DPO, LoRA).
- **DiffusionNFT** for ACE-Step with the rubric reward.
- **Rubrics and prompts.** The seven-dimension music-quality rubric and the tempo,
  key, and instrumentation rubrics, with the exact prompts sent to the raters.
- **Rewards.** Response parsing and weighting, and the DiffusionNFT reward, which
  is served by a rater server: Music Flamingo, Qwen2-Audio, Qwen3-Omni, or any
  audio-language model you plug in.

Generators: `facebook/musicgen-small`, ACE-Step (3.5B).
Raters: `nvidia/music-flamingo-hf`, `Qwen/Qwen2-Audio-7B-Instruct`,
`Qwen/Qwen3-Omni-30B-A3B-Instruct`, all decoded greedily.

## Release scope

This is a code release of the method. It does **not** include model checkpoints,
datasets, data splits, generated audio, rater scores, preference pairs,
evaluation code (CLAP / SongEval / Audiobox-Aesthetics scoring, listening-test
tooling, rater screening), the scripts that rate candidates and build preference
pairs, the training code for the tempo, key, and instrumentation experiments
(their rubrics and prompts are included), experiment launch scripts, or results.
The paper's numbers cannot be reproduced from this repository alone, and the
training code has not been re-run end to end in this form.

## Repository structure

```
dpo/
  dataset.py          your data loading: implement load_preference_pairs
  musicgen/
    train.py          MusicGen-small DPO entry point
    trainer.py        log-probabilities, DPO loss, training loop
    audio.py          preference pairs -> winner/loser audio batches
    lora.py           LoRA layers
  acestep/
    train.py          ACE-Step DPO entry point (reads the preference pairs)
    worker.py         flow-matching DPO surrogate and training loop
diffusionnft/
  dataset.py          your data loading: implement load_prompts
  server.py           rater server for the reward (choose the model with --model)
  train.py            DiffusionNFT trainer for ACE-Step
  config.py           training configuration (rubric)
  base_config.py      DiffusionNFT default settings
  transformer.py, pipeline.py, scheduler.py   modified ACE-Step components
  stat_tracking.py    per-prompt reward normalization
  ema.py              EMA of the policy weights
rewards/
  rubric.py           loads rubrics/*.yaml and generates the rating prompts
  dpo_reward.py       parsing and weighting of the DPO ratings
  nft_reward.py       DiffusionNFT rubric reward and its rater-server client; Music Flamingo and Qwen2-Audio raters
  qwen3_omni.py       Qwen3-Omni rater (joint and per-dimension prompting)
rubrics/
  general.yaml        seven-dimension music-quality rubric
  tempo.yaml, key.yaml, instrument.yaml
prompts/
  dpo.txt             prompt used to rate the DPO candidates
  general.txt         prompts generated from rubrics/general.yaml (DiffusionNFT reward)
  tempo.txt, key.txt, instrument.txt
  render.py           regenerates the rubric prompt files
```

## Installation

Python 3.10 and CUDA GPUs are assumed. We used PyTorch 2.6.0 (CUDA 12.6),
transformers 5.5.4, diffusers 0.37.1, peft 0.19.1, audiocraft 1.1.0, accelerate
1.6.0, ml-collections 1.1.0, librosa 0.11.0, and qwen-omni-utils 0.0.9. ACE-Step
DPO was run in a separate environment with ACE-Step's own pins (transformers
4.50.0).

```bash
pip install -r requirements.txt
# ACE-Step (needed for ACE-Step DPO and DiffusionNFT). Its setup pins
# transformers==4.50.0, which is too old for Qwen3-Omni / Music Flamingo; install it
# without dependencies to share one environment with the raters, or use a separate
# environment for the generator.
pip install --no-deps "ace-step @ git+https://github.com/ace-step/ACE-Step.git@1bee4c9f5b43e30995f8d4d33b3919197ce1bd68"
```

All commands below are run from the repository root.

## Your data

The trainers read data only through two functions that you implement for your own
data (each raises `NotImplementedError` until you do):

- `load_preference_pairs(path)` in `dpo/dataset.py` returns the DPO preference pairs, each with the
  text prompt, the paths of the preferred (`winner`) and rejected (`loser`) audio,
  and optionally the reward `margin` between them. `path` is what you pass to
  `--train_preferences` / `--val_preferences` (MusicGen) or `--train_pairs`
  (ACE-Step).
- `load_prompts(path, split)` in `diffusionnft/dataset.py` returns the text prompts of the `"train"` or `"val"`
  split for DiffusionNFT. `path` is the `RUBRIC_DATASET` environment variable.

## Rubrics and prompts

Each rubric is a YAML file listing, per dimension, a definition, audible evidence,
anchors at 0/2/5/8/10, a calibration note, confounders, and a weight.
`rewards/rubric.py` generates the rating prompts from it, in two forms:
**joint** (all dimensions in one call, one flat JSON object) and
**per-dimension** (one fresh call per dimension, one single-key JSON object).

| File | Used for |
|---|---|
| `prompts/dpo.txt` | Rating the DPO candidates (all three raters, joint) |
| `prompts/general.txt` | DiffusionNFT rubric reward on MusicCaps (joint and per-dimension) |
| `prompts/tempo.txt`, `prompts/key.txt`, `prompts/instrument.txt` | Tempo, key, and instrumentation experiments |

`<MUSIC_DESCRIPTION>` in these files stands for the text prompt of the clip being
rated. The prompts printed in the paper's appendix (system prompts, the
seven-dimension joint and per-dimension prompts, and the tempo joint and
per-dimension prompts) are identical to the ones generated here. `prompts/dpo.txt`
uses the same seven dimensions and weights as `rubrics/general.yaml` but a shorter
description of each (a one-sentence definition and score bands).

```python
from rewards.rubric import load_rubric
rubric = load_rubric("general")              # or "tempo", "key", "instrument"
rubric.weights                               # prompt_alignment 0.25, ... sums to 1
system = rubric.qwen_system_prompt()         # joint
user = rubric.qwen_user_prompt("a calm solo piano piece")
user_k = rubric.flamingo_dimension_prompt("rhythm", "a calm solo piano piece")  # per-dimension
```

(The `qwen_*` / `flamingo_*` method names are historical; every rater supports
both forms.) After editing a rubric, run `python prompts/render.py` to update the
prompt files.

## Rewards

**DPO.** A rater's reply to `prompts/dpo.txt` is parsed and weighted with
`rewards/dpo_reward.py`; candidates for the same text prompt are ranked by this
reward:

```python
from rewards.dpo_reward import extract_json_object, normalize_scores, weighted_reward
scores = normalize_scores(extract_json_object(reply))   # {dimension: 0..10}; missing fields are errors
reward = weighted_reward(scores)                         # 0..10
```

**DiffusionNFT.** `rewards/nft_reward.py` divides each dimension score by 10 and
takes the weighted sum, giving a reward in [0, 1] per clip.

## DPO preference pairs

In the paper, each training prompt has N = 6 candidates from the base generator (8 s
clips for MusicGen-small, 30 s for ACE-Step), rated with `prompts/dpo.txt` and
ranked by the weighted reward; the top two are paired with the bottom two, ties
are dropped, and pairs whose reward gap is below a margin m are discarded
(`--min_margin` in both DPO trainers). Building the pairs is up to you; the
trainers take them from `load_preference_pairs`.

## DPO: MusicGen-small

The policy is MusicGen-small plus LoRA, the reference is the frozen base model, and
a clip's log-probability is the mean token log-probability over valid EnCodec code
positions.

```bash
python dpo/musicgen/train.py \
  --train_preferences <train pairs> --val_preferences <val pairs> \
  --output_dir outputs/musicgen_dpo/example \
  --model_name facebook/musicgen-small --duration 8 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0 \
  --beta 0.5 --learning_rate 3e-7 --min_margin 2.0 \
  --epochs 2 --batch_size 1 --gradient_accumulation_steps 4 --warmup_steps 100
```

The values above are the Music Flamingo setting reported in the paper; the
Qwen2-Audio and Qwen3-Omni settings are in its hyperparameter appendix. Multi-GPU
training uses `torchrun`. `--dry_run_data` and `--dry_run_model` check the data and
model loading without training.

## DPO: ACE-Step

ACE-Step has no exact sequence likelihood, so DPO uses a flow-matching surrogate:
chosen and rejected latents are noised at a shared logit-normal timestep, policy and
reference see the same noise, and the DPO logit is
`beta * ((l_ref_w - l_theta_w) - (l_ref_l - l_theta_l))` over per-sample flow-matching
losses. The policy is a rank-16 LoRA adapter; disabling it gives the reference.

```bash
python dpo/acestep/train.py \
  --train_pairs <train pairs> --min_margin 4.0 \
  --checkpoint_path checkpoints/ACE-Step-v1-3.5B \
  --output_dir outputs/acestep_dpo/example \
  --beta 500 --lr 1e-6 --max_steps 1500 --grad_accum 4
```

`--audit` runs a feasibility check (per-sample loss shape and one optimizer step)
and exits. The adapter is saved to `<output_dir>/checkpoint_final`.

## DiffusionNFT on ACE-Step

At each update the trainer samples G clips per prompt from the current policy,
scores them with the rubric reward, normalizes rewards within each prompt group by
the batch standard deviation, and applies the DiffusionNFT loss with a KL term to
the frozen model.

The rater runs in `diffusionnft/server.py` on its own GPUs; the trainer and the server exchange
clips and scores through a shared directory. Choose the rater with `--model` and
the prompting form with `--prompting`, and give the trainer the same directory and
prompting:

```bash
# Rater server, e.g. Qwen3-Omni with joint prompting, sharded over 4 GPUs
python -m diffusionnft.server --dir /shared/rater --model qwen3_omni --prompting joint --gpus_per_worker 4

# Trainer
export RUBRIC_DATASET=<your prompts>   # passed to load_prompts in diffusionnft/dataset.py
export RATER_SERVER_DIR=/shared/rater RATER_PROMPTING=joint
export RUBRIC_NUM_EPOCHS=125 RUBRIC_N_GPUS=4
torchrun --nproc_per_node 4 -m diffusionnft.train --config diffusionnft/config.py:rubric
```

| Server option | Meaning |
|---|---|
| `--model` | `qwen3_omni`, `music_flamingo`, or `qwen2_audio` (built in), or `my_module:MyRater` for your own model |
| `--model_id` | checkpoint to load; the built-in raters default to the paper's |
| `--prompting` | `joint` or `per_dimension`; must equal `RATER_PROMPTING` (the trainer checks) |
| `--rubric` | rubric under `rubrics/` (default `general`) |
| `--num_workers`, `--gpus_per_worker` | parallel copies of the rater and the GPUs each one gets |

Qwen3-Omni (30B) is sharded over the GPUs of its worker, so give it several
(`--gpus_per_worker`); Music Flamingo and Qwen2-Audio fit on one GPU per worker.

**Your own rater.** Any audio-language model can be plugged in with
`--model my_module:MyRater` (importable from the repository root, where the server is started). The class is
constructed as `MyRater(model_id=..., device=..., dtype=..., max_new_tokens=...,
prompting=..., rubric=...)` and must provide
`score(prompt, audio_path) -> {dimension: score in [0, 10]}` for every rubric
dimension; an optional `load()` is called once at startup. `rubric` generates the
prompts: `rubric.qwen_system_prompt()` + `rubric.qwen_user_prompt(prompt)` for joint
prompting, `rubric.flamingo_system_prompt()` + `rubric.flamingo_dimension_prompt(key, prompt)`
per dimension.

Training prompts are sampled for the policy updates; validation prompts are scored
every `RUBRIC_EVAL_FREQ` updates to log the reward curve. `ACESTEP_CHECKPOINT_DIR` selects the ACE-Step checkpoint
(downloaded when unset); checkpoints and logs go to `logs/nft/acestep/`.

## Notes on the implementation

These describe the released code as it is; where it differs from the paper's
description, the code is what produced the results.

- Scores are clamped to [0, 10] after parsing.
- `rewards/nft_reward.py` accepts some non-JSON replies for one-dimension requests
  (a bare number or `"Name: 7.5"`) and canonicalizes case-only key mismatches. For
  Qwen2-Audio with joint prompting, a malformed reply is retried once, and after two
  failed attempts that clip is scored per dimension instead.
- One task instruction in `rubrics/tempo.yaml` is written as a YAML mapping; the
  rater saw its Python `str()` form, and it is kept as is so the prompt is unchanged.

## Acknowledgments

This work builds on [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT),
[Flow-GRPO](https://github.com/yifan123/flow_grpo),
[ACE-Step](https://github.com/ace-step/ACE-Step),
[AudioCraft / MusicGen](https://github.com/facebookresearch/audiocraft),
[Music Flamingo](https://huggingface.co/nvidia/music-flamingo-hf),
[Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio), and
[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni). See `THIRD_PARTY_NOTICES.md` for
the files derived from these projects.

## License

Apache License 2.0 (`LICENSE`). Third-party notices are in `THIRD_PARTY_NOTICES.md`.
Model weights used with this code have their own licenses; `nvidia/music-flamingo-hf`
is for non-commercial research use.

## Citation

```bibtex
@article{wang2026rubric,
  title   = {Rubric-Based Optimization for Text-to-Music Generation},
  author  = {Wang, Ping and Yang, Guang and Su, Shao-Rong and Wu, Junkai and Koh, Pang Wei and Smith, Noah A.},
  journal = {arXiv preprint arXiv:2610.03589},
  year    = {2026}
}
```
