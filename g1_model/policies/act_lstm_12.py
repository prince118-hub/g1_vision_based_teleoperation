"""ACT-LSTM-12: ACT-12 plus an LSTM over the same 12-tick window. Ablation entry module only.

    thesis name   ACT-LSTM-12  (short-memory ablation, CLAUDE.md §7 D8 / NOTES.md 2026-10-25b)
    code name     act_lstm  (`tools/train_bc.py --model act_lstm --obs-window 12`;
                  the Run A/B "ACT-LSTM")
    real code     g1_model/act.py: ACTConfig(use_lstm=True), ACTPolicy, build_act
                  (the SAME class as ACT; `use_lstm` adds the LSTM, Option B)

WHAT IT IS. ACT-12 (`g1_model/policies/act_12.py`) with one addition: a 2-layer
LSTM (48 -> 256, dropout 0.3 between layers) reads the SAME 12-state window,
oldest first, from ZERO memory on every call; its last layer's final hidden
state -> Linear(256 -> 512) is a third token [latent, proprio, lstm] with its own
position embedding. No (h, c) crosses calls, in training or in deployment. The
configs of ACT-12 and ACT-LSTM-12 differ in `use_lstm` alone.

INPUT   obs (B, 12, 48)  normalized states t-11 .. t, zeros before the episode start
OUTPUT  (B, 100, 22)     normalized action chunk (training also returns mu, logvar)

TRAINING (Run A, 40 episodes: `act_lstm_n40_seed0` in tools/train_scripted_v3.ps1,
seeds 1-2 in tools/train_scripted_v3_n40.ps1; Run B, 160 episodes:
`act_lstm_160_seed0` in tools/train_scripted_v3.ps1). Identical to ACT-12: masked
L1 + 10 * KL, AdamW
lr 1e-5, weight decay 1e-4 (0 at the gate), grad clip 1.0, batch 8, 200k steps,
a checkpoint every 20k, chosen by task success.

DEPLOYMENT (tools/deploy.py). As ACT-12: every tick, rolling 12-state window,
temporal ensembling (m = 0.01). Nothing carried (`CARRIES_MEMORY` is not set).
"""
from g1_model.act import (LR, WEIGHT_DECAY, ACTConfig, ACTPolicy,
                          TemporalEnsembler, build_act)
from g1_model.models import K_PROVISIONAL

#: Aliases of the real objects, under the same names in every entry module.
Config = ACTConfig
Policy = ACTPolicy
build = build_act
Ensembler = TemporalEnsembler

THESIS_NAME = "ACT-LSTM-12"
CODE_NAME = "act_lstm"
ROLE = "ablation"

#: `Config(**CONFIG_KWARGS)` is the config `tools/train_bc.py` builds for the
#: ablation runs (outside the overfit-10 gate). Checked by g1_model/test_policies.py.
CONFIG_KWARGS = dict(obs_window=12, chunk_size=K_PROVISIONAL,
                     lr=LR, weight_decay=WEIGHT_DECAY, use_lstm=True)

#: The model arguments of the ablation runs' `tools/train_bc.py train` command
#: (budget, data and output arguments are in the launcher). Checked by the test.
TRAIN_ARGS = "--model act_lstm --chunk 100 --obs-window 12 --batch 8"
