"""ACT (code name act_orig): ACT on the ORIGINAL paper's input. Entry module only.

Real code: ACTConfig, ACTPolicy, build_act, TemporalEnsembler in g1_model/act.py, at W_o = 1.
why: see DESIGN_NOTES.md#policies-act
"""
from g1_model.act import (LR, WEIGHT_DECAY, ACTConfig, ACTPolicy,
                          TemporalEnsembler, build_act)
from g1_model.act_memory import ACT_ORIG, ORIGINAL_OBS_WINDOW
from g1_model.models import K_PROVISIONAL

#: Aliases of the real objects, under the same names in every entry module.
Config = ACTConfig
Policy = ACTPolicy
build = build_act
Ensembler = TemporalEnsembler

THESIS_NAME = "ACT"
CODE_NAME = ACT_ORIG
ROLE = "headline"

#: `Config(**CONFIG_KWARGS)` is the config `tools/train_bc.py` builds for the
#: thesis runs (outside the overfit-10 gate). Checked by g1_model/test_policies.py.
CONFIG_KWARGS = dict(obs_window=ORIGINAL_OBS_WINDOW, chunk_size=K_PROVISIONAL,
                     lr=LR, weight_decay=WEIGHT_DECAY, use_lstm=False)

#: The model arguments of the thesis runs' `tools/train_bc.py train` command
#: (budget, data and output arguments are in the launcher). Checked by the test.
TRAIN_ARGS = "--model act_orig --chunk 100 --obs-window 1 --batch 8"
