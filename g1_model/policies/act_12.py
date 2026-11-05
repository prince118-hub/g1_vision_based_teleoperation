"""ACT-12: ACT with a 12-tick observation window. Ablation entry module only.

    thesis name   ACT-12  (short-memory ablation, CLAUDE.md §7 D8 / NOTES.md 2026-10-25b)
    code name     act  (`tools/train_bc.py --model act --obs-window 12`; the Run A/B "ACT")
    real code     g1_model/act.py: ACTConfig, ACTPolicy, build_act, TemporalEnsembler

WHAT IT IS. Exactly ACT (`g1_model/policies/act.py`) except the observation: the
last W_o = 12 normalized states (0.48 s at 25 Hz), flattened to 576 numbers, are
what `input_proj_robot_state` (the proprio token) and the CVAE encoder's
`encoder_joint_proj` read. The original paper uses one state (W_o = 1).

INPUT   obs (B, 12, 48)  normalized states t-11 .. t, zeros before the episode start
OUTPUT  (B, 100, 22)     normalized action chunk (training also returns mu, logvar)

TRAINING (Run A, 40 episodes: `act_n40_seed0` in tools/train_scripted_v3.ps1,
seeds 1-2 in tools/train_scripted_v3_n40.ps1; Run B, 160 episodes: `act_160_seed0`
in tools/train_scripted_v3.ps1). Masked L1 + 10 * KL, AdamW
lr 1e-5, weight decay 1e-4 (0 at the gate), grad clip 1.0, batch 8, 200k steps,
a checkpoint every 20k, chosen by task success.

DEPLOYMENT (tools/deploy.py). Queried every tick with a rolling 12-state window
(zeros at the episode start, as in training); chunks temporally ensembled
(m = 0.01, reset per episode). No memory beyond the window.
"""
from g1_model.act import (LR, WEIGHT_DECAY, ACTConfig, ACTPolicy,
                          TemporalEnsembler, build_act)
from g1_model.models import K_PROVISIONAL

#: Aliases of the real objects, under the same names in every entry module.
Config = ACTConfig
Policy = ACTPolicy
build = build_act
Ensembler = TemporalEnsembler

THESIS_NAME = "ACT-12"
CODE_NAME = "act"
ROLE = "ablation"

#: `Config(**CONFIG_KWARGS)` is the config `tools/train_bc.py` builds for the
#: ablation runs (outside the overfit-10 gate). Checked by g1_model/test_policies.py.
CONFIG_KWARGS = dict(obs_window=12, chunk_size=K_PROVISIONAL,
                     lr=LR, weight_decay=WEIGHT_DECAY, use_lstm=False)

#: The model arguments of the ablation runs' `tools/train_bc.py train` command
#: (budget, data and output arguments are in the launcher). Checked by the test.
TRAIN_ARGS = "--model act --chunk 100 --obs-window 12 --batch 8"
