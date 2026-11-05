"""BC: behavioural cloning, the thesis baseline. Entry module only (aliases, no code).

    thesis name   BC  (CLAUDE.md §7 D8: "BC (batch 256)")
    code name     bc  (`tools/train_bc.py --model bc --chunk 1`; metadata `config.notes.model`)
    real code     g1_model/models.py: BCConfig, BCPolicy, build_bc
                  (chunked BC is the SAME class at --chunk > 1; it is a Phase-4 rung,
                  not a thesis model)

WHAT IT IS. A plain MLP: the last W_o = 12 normalized states, flattened
(12 x 48 = 576), -> Linear 512 -> ReLU -> Linear 512 -> ReLU -> Linear 22.
No dropout, no LayerNorm, no memory, no chunk, no attention: the honest floor.

INPUT   obs (B, 12, 48)  normalized states t-11 .. t; ticks before the episode
                         start are zeros (g1_model/loader.py:376-382)
OUTPUT  (B, 1, 22)       the normalized action for tick t (K = 1). The loss trains
                         16 of the 22 dims (`spec.ACTION_MASK`).

TRAINING (thesis runs: tools/train_scripted_v3_bc_b256.ps1, `bc_b256_*`).
Masked L1 (`train.masked_l1`); AdamW, lr 1e-3, weight decay 0, grad clip 1.0
(`models.LADDER_GRAD_CLIP`); batch 256; 300 epochs, a checkpoint every 30; the
deployed checkpoint is chosen by task success (tools/eval_scripted.py). At the
overfit-10 gate dropout and weight decay are 0 too (they already are).

DEPLOYMENT (tools/deploy.py `CheckpointPolicy`). Queried every tick (25 Hz); the
single predicted action is applied (K = 1, nothing to ensemble); no memory.
"""
from g1_model.models import BCConfig, BCPolicy, build_bc

#: Aliases of the real objects, under the same names in every entry module.
Config = BCConfig
Policy = BCPolicy
build = build_bc

THESIS_NAME = "BC"
CODE_NAME = "bc"
ROLE = "headline"

#: `Config(**CONFIG_KWARGS)` is the config `tools/train_bc.py` builds for the
#: thesis runs (outside the overfit-10 gate). Checked by g1_model/test_policies.py.
CONFIG_KWARGS = dict(obs_window=12, chunk_size=1, batch_size=256)

#: The model arguments of the thesis runs' `tools/train_bc.py train` command
#: (budget, data and output arguments are in the launcher). Checked by the test.
TRAIN_ARGS = "--model bc --chunk 1 --obs-window 12 --batch 256"
