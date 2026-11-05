"""ACT: Action Chunking with Transformers on the ORIGINAL paper's input. Entry module only.

    thesis name   ACT  (CLAUDE.md §7 D8)
    code name     act_orig  (`tools/train_bc.py --model act_orig --obs-window 1`;
                  `g1_model.act_memory.ACT_ORIG`)
    real code     g1_model/act.py: ACTConfig, ACTPolicy, build_act, TemporalEnsembler
                  (act_orig is ACTPolicy built from ACTConfig(obs_window=1) - no subclass)

WHAT IT IS. State-only ACT (Zhao et al., arXiv 2304.13705; reference code pinned
at tonyzhaozh/act@742c753c). A CVAE whose decoder is a transformer:
  CVAE encoder (TRAINING ONLY): [CLS, state, 100-action chunk] -> 4-layer
      transformer encoder -> CLS -> (mu, logvar) of a 32-D style variable z.
  policy: tokens [latent(z), proprio(state)] -> 4-layer transformer encoder ->
      1-layer decoder with 100 learned queries -> Linear -> 100 x 22 actions.
  At inference z = 0 (the prior mean) and the CVAE encoder is never run.
  Width 512, 8 heads, feed-forward 3200, dropout 0.1, post-norm, ReLU.
  Vision is dropped; see docs/ACT_FAITHFULNESS.md for every difference.

INPUT   obs (B, 1, 48)   the CURRENT normalized state only (W_o = 1), as the paper
        training adds actions (B, 100, 22) and action_mask (B, 100) for the encoder
OUTPUT  (B, 100, 22)     a chunk of K = 100 normalized actions (4.0 s at 25 Hz);
                         training also returns (mu, logvar)

TRAINING (thesis runs: tools/train_memory_design_n40.ps1, `act_orig_n40_seed*`).
Masked L1 + beta * KL, beta 10 (`act.act_loss`); AdamW, lr 1e-5, weight decay
1e-4 (0 at the overfit-10 gate), grad clip 1.0; batch 8; 200k optimizer steps, a
checkpoint every 20k, chosen by task success (tools/eval_scripted.py).

DEPLOYMENT (tools/deploy.py `CheckpointPolicy`). Queried EVERY tick; the
overlapping chunks are temporally ensembled (`act.TemporalEnsembler`, m = 0.01,
oldest prediction weighted most, reset per episode). No memory.
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
