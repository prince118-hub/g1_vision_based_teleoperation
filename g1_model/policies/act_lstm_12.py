"""ACT-LSTM-12 (code name act_lstm): ACT-12 plus an LSTM over its 12-tick window. Ablation entry module.

Real code: ACTConfig(use_lstm=True), ACTPolicy, build_act in g1_model/act.py (one class with ACT).
why: see DESIGN_NOTES.md#policies-act-lstm-12
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
