"""DiffusionNFT configuration for ACE-Step with the seven-dimension rubric reward.

Trimmed from the research code's ``config/nft.py`` (itself derived from
DiffusionNFT's ``config/nft.py``). The reward comes from a rater served by
``diffusionnft/server.py`` on separate GPUs; set

* ``RATER_SERVER_DIR``: the directory given to ``python -m diffusionnft.server --dir``;
* ``RATER_PROMPTING``: ``joint`` or ``per_dimension``, matching ``--prompting`` of the server;
* ``RUBRIC_DATASET``: passed to ``load_prompts`` in ``diffusionnft/dataset.py``;
* optionally ``ACESTEP_CHECKPOINT_DIR``.
"""
import importlib.util
import os

import ml_collections

_spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base_config.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def _required_env(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"Set the {name} environment variable (see README).")
    return value


def get_config(name):
    return globals()[name]()


def _get_config_acestep(
    base_model="acestep",
    n_gpus=1,
    gradient_step_per_epoch=1,
    reward_fn={},
    name="",
    num_groups=48,
):
    config = base.get_config()
    assert base_model in ["acestep"]
    config.base_model = base_model
    if base_model == "acestep":
        # Empty -> ACE-Step downloads ACE-Step/ACE-Step-v1-3.5B into ~/.cache/ace-step/checkpoints.
        config.pretrained.model = os.environ.get("ACESTEP_CHECKPOINT_DIR", "")
        config.sample.num_steps = 40
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 1.0 # TODO: tune for correct guidance scale
        config.duration = 15
        config.train.beta = 0.001
        config.sample.noise_level = 0.7
        bsz = 9
        config.train.learning_rate = 3e-4
        config.sample.num_audio_per_prompt = 24

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_audio_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_audio_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_audio_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    config.sample.test_batch_size = 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = None

    config.run_name = f"nft_{base_model}_{name}_num_steps_{config.sample.num_steps}_guidance_scale_{config.sample.guidance_scale}_beta_{config.train.beta}_lr_{config.train.learning_rate}_num_groups_{num_groups}_group_size_{config.sample.num_audio_per_prompt}"
    config.save_dir = f"logs/nft/{base_model}/{name}_num_steps_{config.sample.num_steps}_guidance_scale_{config.sample.guidance_scale}_beta_{config.train.beta}_lr_{config.train.learning_rate}_num_groups_{num_groups}_group_size_{config.sample.num_audio_per_prompt}"
    config.reward_fn = reward_fn
    config.save_freq = 10

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.deterministic = True
    config.sample.solver = "dpm2"

    config.mixed_precision = "fp32"
    return config


def rubric():
    """Rubric-reward DiffusionNFT config; the rater is whatever ``diffusionnft/server.py`` serves."""
    name = "rubric"
    n_gpus = int(os.environ.get("RUBRIC_N_GPUS", "4"))
    group_size = int(os.environ.get("RUBRIC_GROUP_SIZE", "4"))
    num_groups = int(os.environ.get("RUBRIC_NUM_GROUPS", "8"))
    if group_size < 2:
        raise ValueError("RUBRIC_GROUP_SIZE must be >=2 so within-prompt advantages are non-zero")
    if group_size % n_gpus != 0:
        raise ValueError("RUBRIC_GROUP_SIZE must be divisible by RUBRIC_N_GPUS")

    config = _get_config_acestep(
        base_model="acestep",
        # The generic helper's original 24-sample batch solver has no solution
        # below four GPUs. Rubric settings are fully replaced immediately below,
        # so use its known-good four-GPU initialization here.
        n_gpus=4,
        gradient_step_per_epoch=1,
        reward_fn={"rubric": 1.0},
        name=name,
        num_groups=num_groups,
    )
    # Passed to load_prompts() in diffusionnft/dataset.py.
    config.dataset = _required_env("RUBRIC_DATASET")
    config.prompt_fn = "rubric"
    config.pretrained.model = os.environ.get("ACESTEP_MODEL_PATH", config.pretrained.model)
    config.sample.num_audio_per_prompt = group_size
    config.sample.train_batch_size = group_size // n_gpus
    config.sample.num_batches_per_epoch = num_groups
    config.sample.test_batch_size = 1
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = num_groups
    config.text_max_length = int(os.environ.get("RUBRIC_TEXT_MAX_LENGTH", "256"))
    prompting = os.environ.get("RATER_PROMPTING", "joint")
    if prompting not in {"joint", "per_dimension"}:
        raise ValueError("RATER_PROMPTING must be 'joint' or 'per_dimension'")
    audio_seconds = int(os.environ.get("RUBRIC_AUDIO_SECONDS", "30"))
    config.duration = audio_seconds

    # _get_config_acestep builds these strings before the rubric-specific batch
    # overrides above. Rebuild them so logs/checkpoints describe the real run.
    run_suffix = (
        f"{name}_num_steps_{config.sample.num_steps}"
        f"_guidance_scale_{config.sample.guidance_scale}"
        f"_beta_{config.train.beta}_lr_{config.train.learning_rate}"
        f"_num_groups_{num_groups}_group_size_{group_size}"
        f"_text{config.text_max_length}"
        f"_audio{audio_seconds}s"
    )
    if prompting == "per_dimension":
        run_suffix += "_per_dimension"
    run_tag = os.environ.get("RUBRIC_RUN_TAG", "").strip()
    if run_tag:
        safe_run_tag = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in run_tag
        )
        run_suffix += f"_{safe_run_tag}"
    config.run_name = f"nft_acestep_{run_suffix}"
    config.save_dir = f"logs/nft/acestep/{run_suffix}"
    config.eval_freq = int(os.environ.get("RUBRIC_EVAL_FREQ", "10"))
    config.save_freq = int(os.environ.get("RUBRIC_SAVE_FREQ", "10"))
    config.num_epochs = int(os.environ.get("RUBRIC_NUM_EPOCHS", str(config.num_epochs)))
    config.resume_from = os.environ.get("RUBRIC_RESUME_FROM", config.resume_from)
    # Keep reward training and held-out evaluation on the same audio horizon.
    # The original pilot trained on 15 s clips but selected checkpoints with
    # 30 s evaluation, which left structure and ending quality out of reward.
    config.rubric_max_initial_equal_fields_rate = float(
        os.environ.get("RUBRIC_MAX_INITIAL_EQUAL_FIELDS_RATE", "1.0")
    )

    config.rubric_reward = ml_collections.ConfigDict({
        "providers": ["server"],
        "server_dir": _required_env("RATER_SERVER_DIR"),
        "prompting": prompting,
        "server_timeout_seconds": float(os.environ.get("RATER_TIMEOUT_SECONDS", "1800")),
        "server_poll_seconds": float(os.environ.get("RATER_POLL_SECONDS", "0.2")),
        "dimension_weights": {
            "prompt_alignment": 0.25,
            "style_instrumentation": 0.15,
            "melody": 0.10,
            "rhythm": 0.10,
            "structure": 0.15,
            "production_quality": 0.15,
            "overall_musicality": 0.10,
        },
        "sample_rate": 48000,
        "retries": int(os.environ.get("RUBRIC_RETRIES", "1")),
        # Training should stop on rater failure rather than silently learn from zeros.
        "failure_policy": os.environ.get("RUBRIC_FAILURE_POLICY", "error"),
    })
    return config
