# Third-party components

This repository contains or adapts the following third-party work. Full licence
texts are in [LICENSES/](LICENSES/).

| Component | Where it is used here | Upstream | Licence |
|---|---|---|---|
| Unitree G1 MJCF model and meshes (MuJoCo Menagerie `unitree_g1`) | `g1.xml`, `assets/*.STL` (`scene.xml` adds the platforms and box) | https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1 | BSD-3-Clause, © 2016-2023 HangZhou YuShu Technology Co., Ltd. (Unitree Robotics) — [LICENSES/mujoco_menagerie-unitree_g1-BSD-3-Clause.txt](LICENSES/mujoco_menagerie-unitree_g1-BSD-3-Clause.txt) |
| unitree_rl_gym, including the pre-trained G1 walking policy `deploy/pre_train/g1/motion.pt` | `unitree_rl_gym/` (vendored); the policy drives the legs | https://github.com/unitreerobotics/unitree_rl_gym | BSD-3-Clause, © 2016-2023 Unitree Robotics — [unitree_rl_gym/LICENSE](unitree_rl_gym/LICENSE), copy in [LICENSES/unitree_rl_gym-BSD-3-Clause.txt](LICENSES/unitree_rl_gym-BSD-3-Clause.txt). `unitree_rl_gym/legged_gym/` keeps its own [LICENSE](unitree_rl_gym/legged_gym/LICENSE). |
| ACT (Action Chunking with Transformers), tonyzhaozh/act @ `742c753c` | `g1_model/act.py`, `g1_model/act_memory.py` are a state-only port of the reference model (CVAE encoder, transformer, losses); comments cite the reference's file:line | https://github.com/tonyzhaozh/act — paper arXiv:2304.13705 | MIT, © 2023 Tony Z. Zhao — [LICENSES/ACT-MIT.txt](LICENSES/ACT-MIT.txt) |
| DETR transformer (shipped inside ACT as `detr/`) | the transformer encoder/decoder structure ported in `g1_model/act.py` | https://github.com/facebookresearch/detr | Apache-2.0, © Facebook, Inc. and its affiliates — [LICENSES/DETR-Apache-2.0.txt](LICENSES/DETR-Apache-2.0.txt). Modified: rewritten as a state-only model (no image backbone), decoder depth set to 1 (see `docs/ACT_CORRESPONDENCE.md` for every change). |

The ACT reference itself is **not** vendored: `reference/` is gitignored. To run
the side-by-side checks that compare against it, clone it yourself:

```bash
git clone https://github.com/tonyzhaozh/act reference/act
git -C reference/act checkout 742c753c0d4a5d87076c8f69e5628c79a8cc5488
```

Not distributed here: the Stereolabs ZED SDK / `pyzed` (proprietary; only
needed for live teleoperation recording, not for anything in "How to run").
