# Third-party notices

This repository is released under the Apache License 2.0 (`LICENSE`). Some files
are derived from the following projects; each such file carries a header naming
its origin and summarizing the changes.

| Project | License | Files in this repository |
|---|---|---|
| [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) (NVIDIA) | Apache-2.0 | `diffusionnft/train.py` (from `scripts/train_nft_sd3.py`), `diffusionnft/config.py` (from `config/nft.py`), `diffusionnft/base_config.py` (unchanged `config/base.py`), `diffusionnft/stat_tracking.py`, `diffusionnft/ema.py` |
| [Flow-GRPO](https://github.com/yifan123/flow_grpo) | MIT | origin of DiffusionNFT's `flow_grpo/stat_tracking.py` and `flow_grpo/ema.py` (see above) |
| [ACE-Step](https://github.com/ace-step/ACE-Step) (ACE Studio, StepFun) | Apache-2.0 | `diffusionnft/transformer.py` (from `acestep/models/ace_step_transformer.py`), `diffusionnft/pipeline.py` (from `acestep/pipeline_ace_step.py`), `diffusionnft/scheduler.py` (from `acestep/schedulers/scheduling_flow_match_euler_discrete.py`) (modified copies; all other ACE-Step code is imported from the installed package) |

Models and tools used at run time are not redistributed and keep their own
licenses, including MusicGen / AudioCraft, ACE-Step checkpoints, Qwen2-Audio,
Qwen3-Omni, and Music Flamingo (`nvidia/music-flamingo-hf` is released for
non-commercial research use).

## Flow-GRPO license

```
MIT License

Copyright (c) 2025 Jie Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
