"""ACT-LSTM (code name act_lstm_full): ACT plus an LSTM over the WHOLE episode. Entry module only.

Real code: ACTFullHistoryConfig, ACTLSTMFullPolicy, build_act_lstm_full in g1_model/act_memory.py.
why: see DESIGN_NOTES.md#policies-act-lstm
"""
from g1_model.act import LR, WEIGHT_DECAY, TemporalEnsembler
from g1_model.act_memory import (ACT_LSTM_FULL, ORIGINAL_OBS_WINDOW,
                                 ACTFullHistoryConfig, ACTLSTMFullPolicy,
                                 build_act_lstm_full)
from g1_model.history import CarriedObs, HistoryChunkDataset, HistoryObs
from g1_model.models import K_PROVISIONAL

#: Aliases of the real objects, under the same names in every entry module.
Config = ACTFullHistoryConfig
Policy = ACTLSTMFullPolicy
build = build_act_lstm_full
Ensembler = TemporalEnsembler
Dataset = HistoryChunkDataset

THESIS_NAME = "ACT-LSTM"
CODE_NAME = ACT_LSTM_FULL
ROLE = "headline"

#: `Config(**CONFIG_KWARGS)` is the config `tools/train_bc.py` builds for the
#: thesis runs (outside the overfit-10 gate). Checked by g1_model/test_policies.py.
CONFIG_KWARGS = dict(obs_window=ORIGINAL_OBS_WINDOW, chunk_size=K_PROVISIONAL,
                     lr=LR, weight_decay=WEIGHT_DECAY, use_lstm=True)

#: The model arguments of the thesis runs' `tools/train_bc.py train` command
#: (budget, data and output arguments are in the launcher). Checked by the test.
TRAIN_ARGS = "--model act_lstm_full --chunk 100 --obs-window 1 --batch 8"
