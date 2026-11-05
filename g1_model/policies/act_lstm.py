"""ACT-LSTM: ACT plus an LSTM over the WHOLE episode so far. Entry module only.

    thesis name   ACT-LSTM  (CLAUDE.md §7 D8)
    code name     act_lstm_full  (`tools/train_bc.py --model act_lstm_full --obs-window 1`;
                  `g1_model.act_memory.ACT_LSTM_FULL`)
    real code     g1_model/act_memory.py: ACTFullHistoryConfig, ACTLSTMFullPolicy,
                  build_act_lstm_full (a subclass of act.ACTPolicy: every ACT part is
                  ACTPolicy's own code); g1_model/history.py: HistoryChunkDataset,
                  HistoryObs, CarriedObs (the episode-history input)

WHAT IT IS. ACT (`g1_model/policies/act.py`, act_orig) with ONE addition: a
2-layer LSTM (48 -> 256, dropout 0.3 between the two layers only) reads the
episode's normalized states from tick 0 to the current tick; its last layer's
hidden state -> Linear(256 -> 512) becomes a THIRD encoder token,
[latent, proprio, lstm], with its own learned position embedding. The CVAE
encoder never sees it. The configs of ACT and ACT-LSTM differ in `use_lstm` alone,
and from the same seed every ACT parameter starts bit-identical (the LSTM is
built last).

INPUT   training / scoring: HistoryObs(current (B, 1, 48), history (B, T, 48)
        zero-padded, lengths (B,)); the LSTM re-reads 0..t from ZERO memory per sample
        deployment: `step(obs (B, 1, 48), (h, c) or None)` advances the LSTM one tick
        A plain tensor is refused (it has no history).
OUTPUT  (B, 100, 22) normalized action chunk; `step` also returns the new (h, c)

TRAINING (thesis runs: tools/train_memory_design_n40.ps1, `act_lstm_full_n40_seed*`).
Identical to ACT: masked L1 + 10 * KL, AdamW lr 1e-5, weight decay 1e-4 (0 at the
gate), grad clip 1.0, batch 8, 200k steps, checkpoint every 20k, chosen by task
success. Each sample is a random (episode, t); nothing is carried between samples.
Validation is scored in batches of 32 histories (`HistoryChunkDataset`).

DEPLOYMENT (tools/deploy.py `CheckpointPolicy`, `CARRIES_MEMORY = True`). Every
tick the LSTM advances one step from the CARRIED (h, c), kept in the deploy
wrapper (never on the model) and reset per episode; the chunk is temporally
ensembled as ACT's. Carried vs re-read: equal to float rounding (test_act_memory).
Deployment may run to 900 ticks, longer than any training episode (548-692).
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
