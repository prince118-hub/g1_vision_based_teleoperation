# Human_v2 final analysis: Experiments 1 and 2

**The locked human_v2 study** (CLAUDE.md section 8, 2026-11-20): 40 human demonstrations (50 accepted, split 40/10), walking labels smoothed (`centred_ma13_v1`), raw 48-D state. **9 runs, 1800 episodes** (3 models x seeds 0-2 x 100 Exp 1 + 100 Exp 2). Thesis names per CLAUDE.md D8 (**pending adviser approval**): BC = `bc_b256`, ACT = `act_orig`, ACT-LSTM = `act_lstm_full`. Evaluation cap 1568 ticks, lock rule v3, policy walk-in.

Regenerate with `python tools/make_human_report.py` (read-only; inputs `runs/human_v2/eval`, scripted Run A in `runs/scripted_v3/eval` + `runs/scripted_v3_memory_design/eval`, teleop tables in `teleop/`). The counts loaded here were checked against `runs/human_v2/eval/summary.txt` and every run's own `metadata.json`.

**Phase definition (Charles, 2026-11-22): FIVE official phases.** 1 walk-in = the pickup base lock engaged (lock rule v3; a lock engaged while the box was not welded, before the weld). 2 grasp, 3 lift, 4 transport, 5 place: `success.py`, unchanged. **Cumulative** rate = reached phases 1..k; **conditional** = phase k given phases 1..k-1. Full success is shown beside them (it equals place on every episode). Goal lock and release are **diagnostics**, not phases (section 5).

**Tests.** Primary: the position-level paired test (per box position, seeds 0-3 reaching the phase; Wilcoxon signed-rank and sign test). Secondary: Fisher's exact test on pooled counts. **Holm family for the phases = all 5 phases x 2 experiments x 3 pairs = 30 tests** (per test type); full success is its own family of 6; the diagnostics another of 12 (exploratory).

## Key results

- **Full success, pooled over 3 seeds (Exp 1 | Exp 2):** BC 0/300 (0.0%) | 0/300 (0.0%); ACT 37/300 (12.3%) | 6/300 (2.0%); ACT-LSTM 34/300 (11.3%) | 2/300 (0.7%). Per seed, Exp 1: BC 0, 0, 0%; ACT 15, 13, 9%; ACT-LSTM 16, 8, 10%.
- **Walk-in (phase 1), pickup lock reached, Exp 1 | Exp 2:** BC 178/300 (59.3%) | 113/300 (37.7%); ACT 169/300 (56.3%) | 129/300 (43.0%); ACT-LSTM 168/300 (56.0%) | 37/300 (12.3%).
- **Largest single loss (the first phase not reached), per model and experiment:** BC Exp 1: grasp (178 of 300); BC Exp 2: walk-in (187 of 300); ACT Exp 1: walk-in (131 of 300); ACT Exp 2: walk-in (171 of 300); ACT-LSTM Exp 1: walk-in (132 of 300); ACT-LSTM Exp 2: walk-in (263 of 300).
- **Given the walk-in, grasp (phase 2 | 1), Exp 1:** BC 0/178; ACT 140/169; ACT-LSTM 133/168; **place given transport, Exp 1:** BC 0/0; ACT 37/123; ACT-LSTM 34/106.
- **ACT-LSTM vs ACT, position-level (primary):** Exp 1: Holm-significant (Wilcoxon) in no phase; Exp 2: Holm-significant (Wilcoxon) in walk-in. Full success Exp 1: 18 positions favour ACT-LSTM, 16 favour ACT, Wilcoxon Holm p 0.7201.
- **ACT-LSTM vs BC, position-level (primary):** Exp 1: Holm-significant (Wilcoxon) in grasp, lift, transport, place; Exp 2: Holm-significant (Wilcoxon) in walk-in, grasp, lift, transport. Full success Exp 1: 27 positions favour ACT-LSTM, 0 favour BC, Wilcoxon Holm p <0.0001.
- **ACT vs BC, position-level (primary):** Exp 1: Holm-significant (Wilcoxon) in grasp, lift, transport, place; Exp 2: Holm-significant (Wilcoxon) in grasp, lift, transport. Full success Exp 1: 26 positions favour ACT, 0 favour BC, Wilcoxon Holm p <0.0001.
- **Generalisation gap, full success (Exp 1 - Exp 2, points, Newcombe 95% CI):** BC 0 vs 0 (floor); ACT +10.3 [+6.4, +14.7]; ACT-LSTM +10.7 [+7.1, +14.8].
- **Walk-in gap:** BC +21.7 [+13.7, +29.2]; ACT +13.3 [+5.3, +21.1]; ACT-LSTM +43.7 [+36.6, +50.1].
- **Exploratory, lateral-adjusted gap** (Exp 1 re-weighted to the patch's lateral position; the walk-in depends strongly on side, below): walk-in BC -10.4 [-18.9, -1.3]; ACT -30.4 [-37.6, -23.1]; ACT-LSTM -4.6 [-10.4, +1.9]; full success BC +0.0 [+0.0, +0.0]; ACT -2.0 [-3.7, -0.7]; ACT-LSTM -0.7 [-1.7, +0.0].
- **Walk-in by spawn side, Exp 1 (right / centre / left):** BC 73/81 / 85/138 / 20/81; ACT 73/81 / 83/138 / 13/81; ACT-LSTM 81/81 / 84/138 / 3/81.
- **Diagnostics, of grasped boxes (Exp 1):** BC 0 grasped: goal lock 0, released 0, placed 0; ACT 140 grasped: goal lock 43, released 65, placed 37; ACT-LSTM 133 grasped: goal lock 61, released 77, placed 34. Welds with no pickup lock (the base free at the weld), Exp 1 + Exp 2: BC 0; ACT 33; ACT-LSTM 41.
- **Human vs scripted, full success Exp 1 | Exp 2 (human vs scripted Run A):** BC 0 vs 0 | 0 vs 0; ACT 37 vs 36 | 6 vs 11; ACT-LSTM 34 vs 49 | 2 vs 49 (of 300 each).

## 0. Which run is which

Model, chunk, window, batch and 40 training demos read back from each run's `metadata.json`; checkpoint chosen on 20 selection seeds (`tools/eval_human.ps1` / `eval_scripted.py`).

| data | model | code | seed | run | chosen | at selection | optimizer steps | train status | eval cap |
|---|---|---|---|---|---|---|---|---|---|
| human | ACT-LSTM | act_lstm_full | 0 | act_lstm_full_human40v2_seed0 | ckpt_step0040000.pt | 5/20 | 200,000 | hard_cap | 1568 |
| human | ACT-LSTM | act_lstm_full | 1 | act_lstm_full_human40v2_seed1 | ckpt_step0180000.pt | 3/20 | 200,000 | hard_cap | 1568 |
| human | ACT-LSTM | act_lstm_full | 2 | act_lstm_full_human40v2_seed2 | ckpt_step0060000.pt | 5/20 | 200,000 | hard_cap | 1568 |
| human | ACT | act_orig | 0 | act_orig_human40v2_seed0 | ckpt_step0080000.pt | 5/20 | 200,000 | hard_cap | 1568 |
| human | ACT | act_orig | 1 | act_orig_human40v2_seed1 | ckpt_step0060000.pt | 5/20 | 200,000 | hard_cap | 1568 |
| human | ACT | act_orig | 2 | act_orig_human40v2_seed2 | ckpt_step0060000.pt | 5/20 | 200,000 | hard_cap | 1568 |
| human | BC | bc | 0 | bc_b256_human40v2_seed0 | ckpt_epoch0090.pt | 0/20 | 35,400 | finished | 1568 |
| human | BC | bc | 1 | bc_b256_human40v2_seed1 | ckpt_epoch0240.pt | 0/20 | 35,400 | finished | 1568 |
| human | BC | bc | 2 | bc_b256_human40v2_seed2 | ckpt_epoch0090.pt | 0/20 | 35,400 | finished | 1568 |
| scripted | BC | bc | 0 | bc_b256_n40_seed0 | ckpt_epoch0270.pt | 0/20 | 25,800 | finished | 900 |
| scripted | BC | bc | 1 | bc_b256_n40_seed1 | ckpt_epoch0030.pt | 0/20 | 25,800 | finished | 900 |
| scripted | BC | bc | 2 | bc_b256_n40_seed2 | ckpt_epoch0060.pt | 0/20 | 25,800 | finished | 900 |
| scripted | ACT-LSTM | act_lstm_full | 0 | act_lstm_full_n40_seed0 | ckpt_step0020000.pt | 5/20 | 200,000 | hard_cap | 900 |
| scripted | ACT-LSTM | act_lstm_full | 1 | act_lstm_full_n40_seed1 | ckpt_step0180000.pt | 4/20 | 200,000 | hard_cap | 900 |
| scripted | ACT-LSTM | act_lstm_full | 2 | act_lstm_full_n40_seed2 | ckpt_step0160000.pt | 6/20 | 200,000 | hard_cap | 900 |
| scripted | ACT | act_orig | 0 | act_orig_n40_seed0 | ckpt_step0200000.pt | 4/20 | 200,000 | hard_cap | 900 |
| scripted | ACT | act_orig | 1 | act_orig_n40_seed1 | ckpt_step0080000.pt | 5/20 | 200,000 | hard_cap | 900 |
| scripted | ACT | act_orig | 2 | act_orig_n40_seed2 | ckpt_step0120000.pt | 5/20 | 200,000 | hard_cap | 900 |

## 1. Phase-level success (five official phases)

### 1a. Cumulative rate, pooled over 3 seeds: % [Wilson 95% CI], n = 300 per cell

| model | exp | 1 walk-in | 2 grasp | 3 lift | 4 transport | 5 place | full success |
|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 59.3 [53.7, 64.7] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] |
| BC | Exp 2 | 37.7 [32.4, 43.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] |
| ACT | Exp 1 | 56.3 [50.7, 61.8] | 46.7 [41.1, 52.3] | 46.7 [41.1, 52.3] | 41.0 [35.6, 46.6] | 12.3 [9.1, 16.5] | 12.3 [9.1, 16.5] |
| ACT | Exp 2 | 43.0 [37.5, 48.7] | 5.0 [3.1, 8.1] | 5.0 [3.1, 8.1] | 3.7 [2.1, 6.4] | 2.0 [0.9, 4.3] | 2.0 [0.9, 4.3] |
| ACT-LSTM | Exp 1 | 56.0 [50.3, 61.5] | 44.3 [38.8, 50.0] | 44.3 [38.8, 50.0] | 35.3 [30.1, 40.9] | 11.3 [8.2, 15.4] | 11.3 [8.2, 15.4] |
| ACT-LSTM | Exp 2 | 12.3 [9.1, 16.5] | 5.7 [3.6, 8.9] | 5.7 [3.6, 8.9] | 5.7 [3.6, 8.9] | 0.7 [0.2, 2.4] | 0.7 [0.2, 2.4] |

### 1b. Cumulative rate, mean +- SD over the 3 seeds (%)

| model | exp | 1 walk-in | 2 grasp | 3 lift | 4 transport | 5 place | full success |
|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 59.3 +- 13.7 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |
| BC | Exp 2 | 37.7 +- 54.2 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |
| ACT | Exp 1 | 56.3 +- 13.4 | 46.7 +- 10.0 | 46.7 +- 10.0 | 41.0 +- 8.9 | 12.3 +- 3.1 | 12.3 +- 3.1 |
| ACT | Exp 2 | 43.0 +- 37.5 | 5.0 +- 6.9 | 5.0 +- 6.9 | 3.7 +- 4.6 | 2.0 +- 2.6 | 2.0 +- 2.6 |
| ACT-LSTM | Exp 1 | 56.0 +- 1.0 | 44.3 +- 2.3 | 44.3 +- 2.3 | 35.3 +- 7.8 | 11.3 +- 4.2 | 11.3 +- 4.2 |
| ACT-LSTM | Exp 2 | 12.3 +- 12.5 | 5.7 +- 9.8 | 5.7 +- 9.8 | 5.7 +- 9.8 | 0.7 +- 1.2 | 0.7 +- 1.2 |

### 1c. Conditional rate, phase k given phases 1..k-1, pooled: k/n, % [Wilson 95% CI]

| model | exp | 1 walk-in | 2 grasp | 3 lift | 4 transport | 5 place | full success |
|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 178/300, 59.3 [53.7, 64.7] | 0/178, 0.0 [0.0, 2.1] | 0/0 | 0/0 | 0/0 | 0/0 |
| BC | Exp 2 | 113/300, 37.7 [32.4, 43.3] | 0/113, 0.0 [0.0, 3.3] | 0/0 | 0/0 | 0/0 | 0/0 |
| ACT | Exp 1 | 169/300, 56.3 [50.7, 61.8] | 140/169, 82.8 [76.4, 87.8] | 140/140, 100.0 [97.3, 100.0] | 123/140, 87.9 [81.4, 92.3] | 37/123, 30.1 [22.7, 38.7] | 37/37, 100.0 [90.6, 100.0] |
| ACT | Exp 2 | 129/300, 43.0 [37.5, 48.7] | 15/129, 11.6 [7.2, 18.3] | 15/15, 100.0 [79.6, 100.0] | 11/15, 73.3 [48.0, 89.1] | 6/11, 54.5 [28.0, 78.7] | 6/6, 100.0 [61.0, 100.0] |
| ACT-LSTM | Exp 1 | 168/300, 56.0 [50.3, 61.5] | 133/168, 79.2 [72.4, 84.6] | 133/133, 100.0 [97.2, 100.0] | 106/133, 79.7 [72.1, 85.7] | 34/106, 32.1 [24.0, 41.5] | 34/34, 100.0 [89.8, 100.0] |
| ACT-LSTM | Exp 2 | 37/300, 12.3 [9.1, 16.5] | 17/37, 45.9 [31.0, 61.6] | 17/17, 100.0 [81.6, 100.0] | 17/17, 100.0 [81.6, 100.0] | 2/17, 11.8 [3.3, 34.3] | 2/2, 100.0 [34.2, 100.0] |

Mean +- SD over seeds of the conditional rate (seeds where it is defined):

| model | exp | 1 walk-in | 2 grasp | 3 lift | 4 transport | 5 place | full success |
|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 59.3 +- 13.7 (3 seeds) | 0.0 +- 0.0 (3 seeds) | n/a (0 seeds) | n/a (0 seeds) | n/a (0 seeds) | n/a (0 seeds) |
| BC | Exp 2 | 37.7 +- 54.2 (3 seeds) | 0.0 +- 0.0 (3 seeds) | n/a (0 seeds) | n/a (0 seeds) | n/a (0 seeds) | n/a (0 seeds) |
| ACT | Exp 1 | 56.3 +- 13.4 (3 seeds) | 83.6 +- 8.4 (3 seeds) | 100.0 +- 0.0 (3 seeds) | 87.9 +- 6.7 (3 seeds) | 32.2 +- 15.0 (3 seeds) | 100.0 +- 0.0 (3 seeds) |
| ACT | Exp 2 | 43.0 +- 37.5 (3 seeds) | 39.9 +- 52.7 (3 seeds) | 100.0 +- 0.0 (3 seeds) | 89.7 +- 17.8 (3 seeds) | 51.9 +- 50.1 (3 seeds) | 100.0 +- 0.0 (2 seeds) |
| ACT-LSTM | Exp 1 | 56.0 +- 1.0 (3 seeds) | 79.2 +- 4.3 (3 seeds) | 100.0 +- 0.0 (3 seeds) | 79.3 +- 13.3 (3 seeds) | 31.7 +- 6.5 (3 seeds) | 100.0 +- 0.0 (3 seeds) |
| ACT-LSTM | Exp 2 | 12.3 +- 12.5 (3 seeds) | 34.0 +- 48.1 (2 seeds) | 100.0 +- n/a (1 seeds) | 100.0 +- n/a (1 seeds) | 11.8 +- n/a (1 seeds) | 100.0 +- n/a (1 seeds) |

### 1d. Per run: episodes reaching phases 1..k (of 100) [Wilson 95% CI, %]

| model | seed | exp | run | walk-in | grasp | lift | transport | place | full success |
|---|---|---|---|---|---|---|---|---|---|
| BC | 0 | Exp 1 | bc_b256_human40v2_seed0 | 75 [66, 82] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] |
| BC | 0 | Exp 2 | bc_b256_human40v2_seed0 | 100 [96, 100] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] |
| BC | 1 | Exp 1 | bc_b256_human40v2_seed1 | 50 [40, 60] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] |
| BC | 1 | Exp 2 | bc_b256_human40v2_seed1 | 11 [6, 19] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] |
| BC | 2 | Exp 1 | bc_b256_human40v2_seed2 | 53 [43, 62] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] |
| BC | 2 | Exp 2 | bc_b256_human40v2_seed2 | 2 [1, 7] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] |
| ACT | 0 | Exp 1 | act_orig_human40v2_seed0 | 41 [32, 51] | 37 [28, 47] | 37 [28, 47] | 31 [23, 41] | 15 [9, 23] | 15 [9, 23] |
| ACT | 0 | Exp 2 | act_orig_human40v2_seed0 | 1 [0, 5] | 1 [0, 5] | 1 [0, 5] | 1 [0, 5] | 1 [0, 5] | 1 [0, 5] |
| ACT | 1 | Exp 1 | act_orig_human40v2_seed1 | 62 [52, 71] | 46 [37, 56] | 46 [37, 56] | 44 [35, 54] | 13 [8, 21] | 13 [8, 21] |
| ACT | 1 | Exp 2 | act_orig_human40v2_seed1 | 55 [45, 64] | 1 [0, 5] | 1 [0, 5] | 1 [0, 5] | 0 [0, 4] | 0 [0, 4] |
| ACT | 2 | Exp 1 | act_orig_human40v2_seed2 | 66 [56, 75] | 57 [47, 66] | 57 [47, 66] | 48 [38, 58] | 9 [5, 16] | 9 [5, 16] |
| ACT | 2 | Exp 2 | act_orig_human40v2_seed2 | 73 [64, 81] | 13 [8, 21] | 13 [8, 21] | 9 [5, 16] | 5 [2, 11] | 5 [2, 11] |
| ACT-LSTM | 0 | Exp 1 | act_lstm_full_human40v2_seed0 | 56 [46, 65] | 47 [38, 57] | 47 [38, 57] | 44 [35, 54] | 16 [10, 24] | 16 [10, 24] |
| ACT-LSTM | 0 | Exp 2 | act_lstm_full_human40v2_seed0 | 25 [18, 34] | 17 [11, 26] | 17 [11, 26] | 17 [11, 26] | 2 [1, 7] | 2 [1, 7] |
| ACT-LSTM | 1 | Exp 1 | act_lstm_full_human40v2_seed1 | 55 [45, 64] | 43 [34, 53] | 43 [34, 53] | 33 [25, 43] | 8 [4, 15] | 8 [4, 15] |
| ACT-LSTM | 1 | Exp 2 | act_lstm_full_human40v2_seed1 | 12 [7, 20] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] |
| ACT-LSTM | 2 | Exp 1 | act_lstm_full_human40v2_seed2 | 57 [47, 66] | 43 [34, 53] | 43 [34, 53] | 29 [21, 39] | 10 [6, 17] | 10 [6, 17] |
| ACT-LSTM | 2 | Exp 2 | act_lstm_full_human40v2_seed2 | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] | 0 [0, 4] |

### 1e. The old four-phase table (grasp, lift, transport, place), for continuity

As earlier reports showed it: the rate of each phase on its own (not chained through walk-in), pooled over seeds, % (count of 300). On human data no grasp happened without a pickup lock, so these equal the cumulative rates above; on scripted Run A 2 of 1200 episodes grasped without one.

| data | model | exp | grasp | lift | transport | place | success |
|---|---|---|---|---|---|---|---|
| human | BC | Exp 1 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) |
| human | BC | Exp 2 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) |
| human | ACT | Exp 1 | 46.7 (140) | 46.7 (140) | 41.0 (123) | 12.3 (37) | 12.3 (37) |
| human | ACT | Exp 2 | 5.0 (15) | 5.0 (15) | 3.7 (11) | 2.0 (6) | 2.0 (6) |
| human | ACT-LSTM | Exp 1 | 44.3 (133) | 44.3 (133) | 35.3 (106) | 11.3 (34) | 11.3 (34) |
| human | ACT-LSTM | Exp 2 | 5.7 (17) | 5.7 (17) | 5.7 (17) | 0.7 (2) | 0.7 (2) |
| scripted | BC | Exp 1 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) |
| scripted | BC | Exp 2 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) |
| scripted | ACT | Exp 1 | 63.7 (191) | 63.3 (190) | 28.3 (85) | 12.0 (36) | 12.0 (36) |
| scripted | ACT | Exp 2 | 23.3 (70) | 23.3 (70) | 5.3 (16) | 3.7 (11) | 3.7 (11) |
| scripted | ACT-LSTM | Exp 1 | 64.0 (192) | 64.0 (192) | 31.7 (95) | 16.3 (49) | 16.3 (49) |
| scripted | ACT-LSTM | Exp 2 | 41.0 (123) | 41.0 (123) | 22.0 (66) | 16.3 (49) | 16.3 (49) |

![Cumulative phase rates](fig_phase_cumulative.png)

![Conditional phase rates](fig_phase_conditional.png)

## 2. Pairwise tests

### 2a. PRIMARY: position-level paired test

Each experiment's 100 box positions are the same for every policy. Per position, the number of a model's 3 seeds that reached phases 1..k (0-3); two models compared over positions. Wilcoxon signed-rank: normal approximation, tie and continuity corrections, ties dropped; sign test exact. Effect size: matched-pairs rank-biserial r, +1 = every untied position favours A. **Holm over the 30 phase x experiment x pair tests**, per test type. Unit of inference: box positions, for THESE 3 trained seeds per model; it does not speak to a new training seed.

| exp | phase | A vs B | mean seeds reaching it, per position (of 3) | favour A | favour B | tie | Wilcoxon p | Holm | rank-biserial r | sign p | Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp 1 | walk-in | ACT-LSTM vs ACT | 1.68 vs 1.69 | 18 | 19 | 63 (20 both 0) | 0.9178 | 1.0000 | -0.02 | 1.0000 | 1.0000 |
| Exp 1 | walk-in | ACT-LSTM vs BC | 1.68 vs 1.78 | 30 | 31 | 39 (8 both 0) | 0.2481 | 1.0000 | -0.16 | 1.0000 | 1.0000 |
| Exp 1 | walk-in | ACT vs BC | 1.69 vs 1.78 | 26 | 26 | 48 (10 both 0) | 0.3123 | 1.0000 | -0.15 | 1.0000 | 1.0000 |
| Exp 2 | walk-in | ACT-LSTM vs ACT | 0.37 vs 1.29 | 6 | 74 | 20 (11 both 0) | <0.0001 | <0.0001 | -0.87 | <0.0001 | <0.0001 |
| Exp 2 | walk-in | ACT-LSTM vs BC | 0.37 vs 1.13 | 6 | 73 | 21 (0 both 0) | <0.0001 | <0.0001 | -0.87 | <0.0001 | <0.0001 |
| Exp 2 | walk-in | ACT vs BC | 1.29 vs 1.13 | 39 | 21 | 40 (0 both 0) | 0.0501 | 0.6513 | +0.26 | 0.0273 | 0.3828 |
| Exp 1 | grasp | ACT-LSTM vs ACT | 1.33 vs 1.40 | 12 | 19 | 69 (37 both 0) | 0.3910 | 1.0000 | -0.17 | 0.2810 | 1.0000 |
| Exp 1 | grasp | ACT-LSTM vs BC | 1.33 vs 0.00 | 55 | 0 | 45 (45 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 1 | grasp | ACT vs BC | 1.40 vs 0.00 | 62 | 0 | 38 (38 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 2 | grasp | ACT-LSTM vs ACT | 0.17 vs 0.15 | 13 | 11 | 76 (73 both 0) | 0.6951 | 1.0000 | +0.08 | 0.8388 | 1.0000 |
| Exp 2 | grasp | ACT-LSTM vs BC | 0.17 vs 0.00 | 17 | 0 | 83 (83 both 0) | <0.0001 | 0.0008 | +1.00 | <0.0001 | 0.0003 |
| Exp 2 | grasp | ACT vs BC | 0.15 vs 0.00 | 14 | 0 | 86 (86 both 0) | 0.0003 | 0.0053 | +1.00 | 0.0001 | 0.0021 |
| Exp 1 | lift | ACT-LSTM vs ACT | 1.33 vs 1.40 | 12 | 19 | 69 (37 both 0) | 0.3910 | 1.0000 | -0.17 | 0.2810 | 1.0000 |
| Exp 1 | lift | ACT-LSTM vs BC | 1.33 vs 0.00 | 55 | 0 | 45 (45 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 1 | lift | ACT vs BC | 1.40 vs 0.00 | 62 | 0 | 38 (38 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 2 | lift | ACT-LSTM vs ACT | 0.17 vs 0.15 | 13 | 11 | 76 (73 both 0) | 0.6951 | 1.0000 | +0.08 | 0.8388 | 1.0000 |
| Exp 2 | lift | ACT-LSTM vs BC | 0.17 vs 0.00 | 17 | 0 | 83 (83 both 0) | <0.0001 | 0.0008 | +1.00 | <0.0001 | 0.0003 |
| Exp 2 | lift | ACT vs BC | 0.15 vs 0.00 | 14 | 0 | 86 (86 both 0) | 0.0003 | 0.0053 | +1.00 | 0.0001 | 0.0021 |
| Exp 1 | transport | ACT-LSTM vs ACT | 1.06 vs 1.23 | 13 | 26 | 61 (38 both 0) | 0.0525 | 0.6513 | -0.34 | 0.0533 | 0.6390 |
| Exp 1 | transport | ACT-LSTM vs BC | 1.06 vs 0.00 | 52 | 0 | 48 (48 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 1 | transport | ACT vs BC | 1.23 vs 0.00 | 61 | 0 | 39 (39 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 2 | transport | ACT-LSTM vs ACT | 0.17 vs 0.11 | 15 | 9 | 76 (75 both 0) | 0.2269 | 1.0000 | +0.25 | 0.3075 | 1.0000 |
| Exp 2 | transport | ACT-LSTM vs BC | 0.17 vs 0.00 | 17 | 0 | 83 (83 both 0) | <0.0001 | 0.0008 | +1.00 | <0.0001 | 0.0003 |
| Exp 2 | transport | ACT vs BC | 0.11 vs 0.00 | 10 | 0 | 90 (90 both 0) | 0.0027 | 0.0411 | +1.00 | 0.0020 | 0.0293 |
| Exp 1 | place | ACT-LSTM vs ACT | 0.34 vs 0.37 | 18 | 16 | 66 (62 both 0) | 0.7201 | 1.0000 | -0.07 | 0.8642 | 1.0000 |
| Exp 1 | place | ACT-LSTM vs BC | 0.34 vs 0.00 | 27 | 0 | 73 (73 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 1 | place | ACT vs BC | 0.37 vs 0.00 | 26 | 0 | 74 (74 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 2 | place | ACT-LSTM vs ACT | 0.02 vs 0.06 | 1 | 5 | 94 (93 both 0) | 0.1294 | 1.0000 | -0.67 | 0.2188 | 1.0000 |
| Exp 2 | place | ACT-LSTM vs BC | 0.02 vs 0.00 | 2 | 0 | 98 (98 both 0) | 0.3458 | 1.0000 | +1.00 | 0.5000 | 1.0000 |
| Exp 2 | place | ACT vs BC | 0.06 vs 0.00 | 6 | 0 | 94 (94 both 0) | 0.0197 | 0.2752 | +1.00 | 0.0312 | 0.4062 |

Full success (Holm over its own 6):

| exp | phase | A vs B | mean seeds reaching it, per position (of 3) | favour A | favour B | tie | Wilcoxon p | Holm | rank-biserial r | sign p | Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp 1 | full success | ACT-LSTM vs ACT | 0.34 vs 0.37 | 18 | 16 | 66 (62 both 0) | 0.7201 | 0.7201 | -0.07 | 0.8642 | 1.0000 |
| Exp 1 | full success | ACT-LSTM vs BC | 0.34 vs 0.00 | 27 | 0 | 73 (73 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 1 | full success | ACT vs BC | 0.37 vs 0.00 | 26 | 0 | 74 (74 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 2 | full success | ACT-LSTM vs ACT | 0.02 vs 0.06 | 1 | 5 | 94 (93 both 0) | 0.1294 | 0.3883 | -0.67 | 0.2188 | 0.6562 |
| Exp 2 | full success | ACT-LSTM vs BC | 0.02 vs 0.00 | 2 | 0 | 98 (98 both 0) | 0.3458 | 0.6916 | +1.00 | 0.5000 | 1.0000 |
| Exp 2 | full success | ACT vs BC | 0.06 vs 0.00 | 6 | 0 | 94 (94 both 0) | 0.0197 | 0.0786 | +1.00 | 0.0312 | 0.1250 |

### 2b. SECONDARY: Fisher's exact test on pooled counts (episodes treated as independent)

Two-sided; each n = 300 is 3 seeds on the same 100 positions, so this ignores the between-seed variance and the pairing. Same Holm families (30; 6).

| exp | phase | A vs B | reached | diff (pts) | odds ratio | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| Exp 1 | walk-in | ACT-LSTM vs ACT | 168/300 vs 169/300 | -0.3 | 0.987 | 1.0000 | 1.0000 |
| Exp 1 | walk-in | ACT-LSTM vs BC | 168/300 vs 178/300 | -3.3 | 0.872 | 0.4571 | 1.0000 |
| Exp 1 | walk-in | ACT vs BC | 169/300 vs 178/300 | -3.0 | 0.884 | 0.5084 | 1.0000 |
| Exp 2 | walk-in | ACT-LSTM vs ACT | 37/300 vs 129/300 | -30.7 | 0.186 | <0.0001 | <0.0001 |
| Exp 2 | walk-in | ACT-LSTM vs BC | 37/300 vs 113/300 | -25.3 | 0.233 | <0.0001 | <0.0001 |
| Exp 2 | walk-in | ACT vs BC | 129/300 vs 113/300 | +5.3 | 1.25 | 0.2119 | 1.0000 |
| Exp 1 | grasp | ACT-LSTM vs ACT | 133/300 vs 140/300 | -2.3 | 0.91 | 0.6228 | 1.0000 |
| Exp 1 | grasp | ACT-LSTM vs BC | 133/300 vs 0/300 | +44.3 | n/a | <0.0001 | <0.0001 |
| Exp 1 | grasp | ACT vs BC | 140/300 vs 0/300 | +46.7 | n/a | <0.0001 | <0.0001 |
| Exp 2 | grasp | ACT-LSTM vs ACT | 17/300 vs 15/300 | +0.7 | 1.14 | 0.8562 | 1.0000 |
| Exp 2 | grasp | ACT-LSTM vs BC | 17/300 vs 0/300 | +5.7 | n/a | <0.0001 | 0.0002 |
| Exp 2 | grasp | ACT vs BC | 15/300 vs 0/300 | +5.0 | n/a | <0.0001 | 0.0009 |
| Exp 1 | lift | ACT-LSTM vs ACT | 133/300 vs 140/300 | -2.3 | 0.91 | 0.6228 | 1.0000 |
| Exp 1 | lift | ACT-LSTM vs BC | 133/300 vs 0/300 | +44.3 | n/a | <0.0001 | <0.0001 |
| Exp 1 | lift | ACT vs BC | 140/300 vs 0/300 | +46.7 | n/a | <0.0001 | <0.0001 |
| Exp 2 | lift | ACT-LSTM vs ACT | 17/300 vs 15/300 | +0.7 | 1.14 | 0.8562 | 1.0000 |
| Exp 2 | lift | ACT-LSTM vs BC | 17/300 vs 0/300 | +5.7 | n/a | <0.0001 | 0.0002 |
| Exp 2 | lift | ACT vs BC | 15/300 vs 0/300 | +5.0 | n/a | <0.0001 | 0.0009 |
| Exp 1 | transport | ACT-LSTM vs ACT | 106/300 vs 123/300 | -5.7 | 0.786 | 0.1787 | 1.0000 |
| Exp 1 | transport | ACT-LSTM vs BC | 106/300 vs 0/300 | +35.3 | n/a | <0.0001 | <0.0001 |
| Exp 1 | transport | ACT vs BC | 123/300 vs 0/300 | +41.0 | n/a | <0.0001 | <0.0001 |
| Exp 2 | transport | ACT-LSTM vs ACT | 17/300 vs 11/300 | +2.0 | 1.58 | 0.3333 | 1.0000 |
| Exp 2 | transport | ACT-LSTM vs BC | 17/300 vs 0/300 | +5.7 | n/a | <0.0001 | 0.0002 |
| Exp 2 | transport | ACT vs BC | 11/300 vs 0/300 | +3.7 | n/a | 0.0009 | 0.0133 |
| Exp 1 | place | ACT-LSTM vs ACT | 34/300 vs 37/300 | -1.0 | 0.909 | 0.8006 | 1.0000 |
| Exp 1 | place | ACT-LSTM vs BC | 34/300 vs 0/300 | +11.3 | n/a | <0.0001 | <0.0001 |
| Exp 1 | place | ACT vs BC | 37/300 vs 0/300 | +12.3 | n/a | <0.0001 | <0.0001 |
| Exp 2 | place | ACT-LSTM vs ACT | 2/300 vs 6/300 | -1.3 | 0.329 | 0.2858 | 1.0000 |
| Exp 2 | place | ACT-LSTM vs BC | 2/300 vs 0/300 | +0.7 | n/a | 0.4992 | 1.0000 |
| Exp 2 | place | ACT vs BC | 6/300 vs 0/300 | +2.0 | n/a | 0.0305 | 0.4266 |

Full success:

| exp | phase | A vs B | reached | diff (pts) | odds ratio | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| Exp 1 | full success | ACT-LSTM vs ACT | 34/300 vs 37/300 | -1.0 | 0.909 | 0.8006 | 0.9983 |
| Exp 1 | full success | ACT-LSTM vs BC | 34/300 vs 0/300 | +11.3 | n/a | <0.0001 | <0.0001 |
| Exp 1 | full success | ACT vs BC | 37/300 vs 0/300 | +12.3 | n/a | <0.0001 | <0.0001 |
| Exp 2 | full success | ACT-LSTM vs ACT | 2/300 vs 6/300 | -1.3 | 0.329 | 0.2858 | 0.8573 |
| Exp 2 | full success | ACT-LSTM vs BC | 2/300 vs 0/300 | +0.7 | n/a | 0.4992 | 0.9983 |
| Exp 2 | full success | ACT vs BC | 6/300 vs 0/300 | +2.0 | n/a | 0.0305 | 0.1219 |

## 3. Generalisation gap dS = Exp 1 - Exp 2 (cumulative rates)

Newcombe hybrid score 95% CI (method 10, from the two Wilson intervals: valid at 0 successes); the Wald CI the scripted report used is shown beside it. Exp 1 and Exp 2 are different positions, so the gap is unpaired. SE over seeds = SD of the per-seed gaps / sqrt(3).

| model | phase | Exp 1 | Exp 2 | gap (pts) | Newcombe 95% CI | Wald 95% CI | SE over seeds | gap per seed (pts) |  |
|---|---|---|---|---|---|---|---|---|---|
| BC | walk-in | 178/300 | 113/300 | +21.7 | +13.7 to +29.2 | +13.9 to +29.5 | 23.6 | -25, +39, +51 |  |
| BC | grasp | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| BC | lift | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| BC | transport | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| BC | place | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| BC | full success | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| ACT | walk-in | 169/300 | 129/300 | +13.3 | +5.3 to +21.1 | +5.4 to +21.3 | 13.9 | +40, +7, -7 |  |
| ACT | grasp | 140/300 | 15/300 | +41.7 | +35.3 to +47.6 | +35.5 to +47.8 | 2.8 | +36, +45, +44 |  |
| ACT | lift | 140/300 | 15/300 | +41.7 | +35.3 to +47.6 | +35.5 to +47.8 | 2.8 | +36, +45, +44 |  |
| ACT | transport | 123/300 | 11/300 | +37.3 | +31.2 to +43.2 | +31.4 to +43.3 | 3.8 | +30, +43, +39 |  |
| ACT | place | 37/300 | 6/300 | +10.3 | +6.4 to +14.7 | +6.3 to +14.4 | 3.2 | +14, +13, +4 |  |
| ACT | full success | 37/300 | 6/300 | +10.3 | +6.4 to +14.7 | +6.3 to +14.4 | 3.2 | +14, +13, +4 |  |
| ACT-LSTM | walk-in | 168/300 | 37/300 | +43.7 | +36.6 to +50.1 | +36.9 to +50.4 | 7.5 | +31, +43, +57 |  |
| ACT-LSTM | grasp | 133/300 | 17/300 | +38.7 | +32.3 to +44.7 | +32.5 to +44.9 | 4.3 | +30, +43, +43 |  |
| ACT-LSTM | lift | 133/300 | 17/300 | +38.7 | +32.3 to +44.7 | +32.5 to +44.9 | 4.3 | +30, +43, +43 |  |
| ACT-LSTM | transport | 106/300 | 17/300 | +29.7 | +23.6 to +35.6 | +23.7 to +35.7 | 1.8 | +27, +33, +29 |  |
| ACT-LSTM | place | 34/300 | 2/300 | +10.7 | +7.1 to +14.8 | +7.0 to +14.4 | 1.8 | +14, +8, +10 |  |
| ACT-LSTM | full success | 34/300 | 2/300 | +10.7 | +7.1 to +14.8 | +7.0 to +14.4 | 1.8 | +14, +8, +10 |  |

### 3b. Exploratory: the gap adjusted for lateral position

The walk-in rate depends strongly on the box's lateral position (section 4b), and the held-out patch (y 0.04-0.16 m) lies in the centre-left. So part of the raw gap may be WHERE the patch is, not that its positions were unseen. Here Exp 1's rate is re-weighted to Exp 2's distribution over the 8 lateral columns of the position heatmap (columns `columns`, weights `weights` in `generalization_gap_lateral.csv`), then Exp 2's rate is subtracted. 95% CI: cluster bootstrap over box positions (2000 resamples, seed 0). The adjustment is coarse (columns 5.25 cm wide, depth not adjusted) and post hoc: read it as a check on the raw gap, not as the headline.

| data | model | phase | Exp 1 (%) | Exp 1 lateral-adjusted (%) | Exp 2 (%) | raw gap [bootstrap CI] | adjusted gap [bootstrap CI] | Exp 1 positions per column |
|---|---|---|---|---|---|---|---|---|
| human | BC | walk-in | 59.3 | 27.2 | 37.7 | +21.7 [+15.3, +28.0] | -10.4 [-18.9, -1.3] | 17,9,9,11 |
| human | BC | grasp | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| human | BC | lift | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| human | BC | transport | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| human | BC | place | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| human | BC | full success | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| human | ACT | walk-in | 56.3 | 12.6 | 43.0 | +13.3 [+3.7, +22.3] | -30.4 [-37.6, -23.1] | 17,9,9,11 |
| human | ACT | grasp | 46.7 | 5.5 | 5.0 | +41.7 [+33.3, +50.7] | +0.5 [-4.1, +5.5] | 17,9,9,11 |
| human | ACT | lift | 46.7 | 5.5 | 5.0 | +41.7 [+32.7, +50.3] | +0.5 [-3.9, +5.5] | 17,9,9,11 |
| human | ACT | transport | 41.0 | 5.2 | 3.7 | +37.3 [+29.3, +45.7] | +1.6 [-2.7, +6.3] | 17,9,9,11 |
| human | ACT | place | 12.3 | 0.0 | 2.0 | +10.3 [+6.0, +15.3] | -2.0 [-3.7, -0.7] | 17,9,9,11 |
| human | ACT | full success | 12.3 | 0.0 | 2.0 | +10.3 [+5.7, +15.3] | -2.0 [-3.7, -0.7] | 17,9,9,11 |
| human | ACT-LSTM | walk-in | 56.0 | 7.7 | 12.3 | +43.7 [+34.7, +54.0] | -4.6 [-10.4, +1.9] | 17,9,9,11 |
| human | ACT-LSTM | grasp | 44.3 | 0.9 | 5.7 | +38.7 [+29.3, +47.7] | -4.8 [-7.4, -2.2] | 17,9,9,11 |
| human | ACT-LSTM | lift | 44.3 | 0.9 | 5.7 | +38.7 [+29.3, +47.7] | -4.8 [-7.4, -2.5] | 17,9,9,11 |
| human | ACT-LSTM | transport | 35.3 | 0.9 | 5.7 | +29.7 [+22.0, +37.7] | -4.8 [-7.4, -2.2] | 17,9,9,11 |
| human | ACT-LSTM | place | 11.3 | 0.0 | 0.7 | +10.7 [+6.7, +15.0] | -0.7 [-1.7, +0.0] | 17,9,9,11 |
| human | ACT-LSTM | full success | 11.3 | 0.0 | 0.7 | +10.7 [+6.7, +15.0] | -0.7 [-1.7, +0.0] | 17,9,9,11 |
| scripted | BC | walk-in | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| scripted | BC | grasp | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| scripted | BC | lift | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| scripted | BC | transport | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| scripted | BC | place | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| scripted | BC | full success | 0.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 17,9,9,11 |
| scripted | ACT | walk-in | 73.7 | 37.8 | 36.7 | +37.0 [+26.3, +47.3] | +1.2 [-12.8, +15.9] | 17,9,9,11 |
| scripted | ACT | grasp | 63.0 | 27.4 | 23.3 | +39.7 [+29.0, +49.7] | +4.1 [-7.9, +16.6] | 17,9,9,11 |
| scripted | ACT | lift | 63.0 | 27.4 | 23.3 | +39.7 [+29.7, +51.0] | +4.1 [-7.6, +17.8] | 17,9,9,11 |
| scripted | ACT | transport | 28.3 | 5.9 | 5.3 | +23.0 [+16.7, +29.3] | +0.5 [-2.5, +3.4] | 17,9,9,11 |
| scripted | ACT | place | 12.0 | 4.1 | 3.7 | +8.3 [+4.3, +12.7] | +0.5 [-2.2, +3.0] | 17,9,9,11 |
| scripted | ACT | full success | 12.0 | 4.1 | 3.7 | +8.3 [+4.0, +12.7] | +0.5 [-2.3, +2.9] | 17,9,9,11 |
| scripted | ACT-LSTM | walk-in | 86.7 | 77.6 | 93.7 | -7.0 [-13.3, -1.3] | -16.1 [-27.6, -5.9] | 17,9,9,11 |
| scripted | ACT-LSTM | grasp | 64.0 | 37.7 | 41.0 | +23.0 [+10.7, +34.3] | -3.3 [-15.6, +9.3] | 17,9,9,11 |
| scripted | ACT-LSTM | lift | 64.0 | 37.7 | 41.0 | +23.0 [+11.7, +34.7] | -3.3 [-15.2, +8.4] | 17,9,9,11 |
| scripted | ACT-LSTM | transport | 31.7 | 16.4 | 22.0 | +9.7 [+0.7, +18.7] | -5.6 [-14.4, +3.9] | 17,9,9,11 |
| scripted | ACT-LSTM | place | 16.3 | 12.0 | 16.3 | +0.0 [-7.0, +6.7] | -4.4 [-12.3, +4.2] | 17,9,9,11 |
| scripted | ACT-LSTM | full success | 16.3 | 12.0 | 16.3 | +0.0 [-7.0, +7.0] | -4.4 [-12.2, +4.4] | 17,9,9,11 |

![Generalisation gap](fig_generalization_gap.png)

## 4. Failure phase and spawn side

### 4a. The first of the five phases not reached (pooled over seeds; counts)

`fell` = the episode fell before reaching its next phase.

| model | exp | n | full success | walk-in | grasp | lift | transport | place | fell |
|---|---|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 300 | 0 | 122 | 178 | 0 | 0 | 0 | 0 |
| BC | Exp 2 | 300 | 0 | 187 | 113 | 0 | 0 | 0 | 0 |
| ACT | Exp 1 | 300 | 37 | 131 | 29 | 0 | 17 | 86 | 0 |
| ACT | Exp 2 | 300 | 6 | 171 | 114 | 0 | 4 | 5 | 0 |
| ACT-LSTM | Exp 1 | 300 | 34 | 132 | 35 | 0 | 27 | 72 | 0 |
| ACT-LSTM | Exp 2 | 300 | 2 | 263 | 20 | 0 | 0 | 15 | 0 |

For comparison, `deploy.py`'s own `fail_reason` bucket (from the offline phase labels, `spec.SCORED_OF`; it charges a never-grasped episode to WALK_IN only when its last label is SETTLE, so it is not the same as the walk-in phase):

| model | exp | n | success | WALK_IN | GRASP | LIFT | TRANSPORT | PLACE | fell | other |
|---|---|---|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 300 | 0 | 122 | 178 | 0 | 0 | 0 | 0 | 0 |
| BC | Exp 2 | 300 | 0 | 187 | 113 | 0 | 0 | 0 | 0 | 0 |
| ACT | Exp 1 | 300 | 37 | 119 | 41 | 0 | 17 | 86 | 0 | 0 |
| ACT | Exp 2 | 300 | 6 | 150 | 135 | 0 | 4 | 5 | 0 | 0 |
| ACT-LSTM | Exp 1 | 300 | 34 | 121 | 46 | 0 | 27 | 72 | 0 | 0 |
| ACT-LSTM | Exp 2 | 300 | 2 | 233 | 50 | 0 | 0 | 15 | 0 | 0 |

![Failure phase](fig_failure_phase.png)

### 4b. By spawn side (robot's view: world y < -0.07 m right, |y| <= 0.07 centre, > 0.07 left)

Walk-in (pickup lock), grasp and full success, k/n, pooled over seeds. The held-out patch (y 0.04-0.16 m) has no right third.

Where the 40 TRAINING demonstrations were, by the same thirds (`splits_human_v2.json`, train seeds; spawn from the seed, `analyze_experiments.spawn_xy`): human right 14, centre 17, left 9; scripted Run A right 14, centre 15, left 11.

| model | exp | side | n | walk-in: k (% [CI]) | grasp k | full success k |
|---|---|---|---|---|---|---|
| BC | Exp 1 | right | 81 | 73 (90 [82, 95]) | 0 | 0 |
| BC | Exp 1 | centre | 138 | 85 (62 [53, 69]) | 0 | 0 |
| BC | Exp 1 | left | 81 | 20 (25 [17, 35]) | 0 | 0 |
| BC | Exp 2 | centre | 90 | 35 (39 [29, 49]) | 0 | 0 |
| BC | Exp 2 | left | 210 | 78 (37 [31, 44]) | 0 | 0 |
| BC | both | right | 81 | 73 (90 [82, 95]) | 0 | 0 |
| BC | both | centre | 228 | 120 (53 [46, 59]) | 0 | 0 |
| BC | both | left | 291 | 98 (34 [28, 39]) | 0 | 0 |
| ACT | Exp 1 | right | 81 | 73 (90 [82, 95]) | 62 | 25 |
| ACT | Exp 1 | centre | 138 | 83 (60 [52, 68]) | 76 | 12 |
| ACT | Exp 1 | left | 81 | 13 (16 [10, 26]) | 2 | 0 |
| ACT | Exp 2 | centre | 90 | 40 (44 [35, 55]) | 9 | 3 |
| ACT | Exp 2 | left | 210 | 89 (42 [36, 49]) | 6 | 3 |
| ACT | both | right | 81 | 73 (90 [82, 95]) | 62 | 25 |
| ACT | both | centre | 228 | 123 (54 [47, 60]) | 85 | 15 |
| ACT | both | left | 291 | 102 (35 [30, 41]) | 8 | 3 |
| ACT-LSTM | Exp 1 | right | 81 | 81 (100 [95, 100]) | 73 | 24 |
| ACT-LSTM | Exp 1 | centre | 138 | 84 (61 [53, 69]) | 60 | 10 |
| ACT-LSTM | Exp 1 | left | 81 | 3 (4 [1, 10]) | 0 | 0 |
| ACT-LSTM | Exp 2 | centre | 90 | 25 (28 [20, 38]) | 12 | 2 |
| ACT-LSTM | Exp 2 | left | 210 | 12 (6 [3, 10]) | 5 | 0 |
| ACT-LSTM | both | right | 81 | 81 (100 [95, 100]) | 73 | 24 |
| ACT-LSTM | both | centre | 228 | 109 (48 [41, 54]) | 72 | 12 |
| ACT-LSTM | both | left | 291 | 15 (5 [3, 8]) | 5 | 0 |

Exploratory: walk-in independent of side? Chi-square, Holm over the 3 models per experiment:

| exp | model | sides | chi2 | dof | p | p (Holm) | min expected |  |
|---|---|---|---|---|---|---|---|---|
| Exp 1 | BC | 3 | 72.40 | 2 | <0.0001 | <0.0001 | 32.9 |  |
| Exp 1 | ACT | 3 | 91.85 | 2 | <0.0001 | <0.0001 | 35.4 |  |
| Exp 1 | ACT-LSTM | 3 | 154.88 | 2 | <0.0001 | <0.0001 | 35.6 |  |
| Exp 2 | BC | 2 | 0.02 | 1 | 0.8760 | 1.0000 | 33.9 |  |
| Exp 2 | ACT | 2 | 0.04 | 1 | 0.8387 | 1.0000 | 38.7 |  |
| Exp 2 | ACT-LSTM | 2 | 26.36 | 1 | <0.0001 | <0.0001 | 11.1 |  |

Failure phase by side (Exp 1 + Exp 2):

| model | side | n | full success | walk-in | grasp | lift | transport | place | fell |
|---|---|---|---|---|---|---|---|---|---|
| BC | centre | 228 | 0 | 108 | 120 | 0 | 0 | 0 | 0 |
| BC | left | 291 | 0 | 193 | 98 | 0 | 0 | 0 | 0 |
| BC | right | 81 | 0 | 8 | 73 | 0 | 0 | 0 | 0 |
| ACT | centre | 228 | 15 | 105 | 38 | 0 | 6 | 64 | 0 |
| ACT | left | 291 | 3 | 189 | 94 | 0 | 2 | 3 | 0 |
| ACT | right | 81 | 25 | 8 | 11 | 0 | 13 | 24 | 0 |
| ACT-LSTM | centre | 228 | 12 | 119 | 37 | 0 | 7 | 53 | 0 |
| ACT-LSTM | left | 291 | 0 | 276 | 10 | 0 | 0 | 5 | 0 |
| ACT-LSTM | right | 81 | 24 | 0 | 8 | 0 | 20 | 29 | 0 |

![Walk-in by side](fig_lock_by_side.png)

![Failure by side](fig_failure_by_side.png)

![Success by position](fig_success_by_position.png)

## 5. Diagnostics (not official phases)

**Goal lock**: a lock engaged while the box was welded (it targets the goal). **Release**: the weld let go after engaging, wherever that was (it includes welds that failed the D19 push rule and never counted as a grasp). Of grasped boxes = of episodes that reached phase 2. k/n, % [Wilson 95% CI].

| data | model | exp | goal lock | release | grasped | of grasped: goal lock | of grasped: released | of grasped: placed | welded (any) | welded with no pickup lock |
|---|---|---|---|---|---|---|---|---|---|---|
| human | BC | Exp 1 | 0/300, 0.0 [0.0, 1.3] | 0/300, 0.0 [0.0, 1.3] | 0 | 0/0 | 0/0 | 0/0 | 0 | 0 |
| human | BC | Exp 2 | 0/300, 0.0 [0.0, 1.3] | 0/300, 0.0 [0.0, 1.3] | 0 | 0/0 | 0/0 | 0/0 | 0 | 0 |
| human | ACT | Exp 1 | 56/300, 18.7 [14.7, 23.5] | 91/300, 30.3 [25.4, 35.8] | 140 | 43/140 | 65/140 | 37/140 | 170 | 12 |
| human | ACT | Exp 2 | 26/300, 8.7 [6.0, 12.4] | 82/300, 27.3 [22.6, 32.6] | 15 | 6/15 | 13/15 | 6/15 | 129 | 21 |
| human | ACT-LSTM | Exp 1 | 71/300, 23.7 [19.2, 28.8] | 99/300, 33.0 [27.9, 38.5] | 133 | 61/133 | 77/133 | 34/133 | 162 | 11 |
| human | ACT-LSTM | Exp 2 | 17/300, 5.7 [3.6, 8.9] | 41/300, 13.7 [10.2, 18.0] | 17 | 6/17 | 5/17 | 2/17 | 59 | 30 |
| scripted | BC | Exp 1 | 1/300, 0.3 [0.1, 1.9] | 0/300, 0.0 [0.0, 1.3] | 0 | 0/0 | 0/0 | 0/0 | 40 | 0 |
| scripted | BC | Exp 2 | 0/300, 0.0 [0.0, 1.3] | 2/300, 0.7 [0.2, 2.4] | 0 | 0/0 | 0/0 | 0/0 | 12 | 0 |
| scripted | ACT | Exp 1 | 47/300, 15.7 [12.0, 20.2] | 225/300, 75.0 [69.8, 79.6] | 189 | 40/189 | 181/189 | 36/189 | 237 | 16 |
| scripted | ACT | Exp 2 | 16/300, 5.3 [3.3, 8.5] | 158/300, 52.7 [47.0, 58.2] | 70 | 10/70 | 69/70 | 11/70 | 177 | 67 |
| scripted | ACT-LSTM | Exp 1 | 62/300, 20.7 [16.5, 25.6] | 249/300, 83.0 [78.3, 86.8] | 192 | 51/192 | 188/192 | 49/192 | 268 | 8 |
| scripted | ACT-LSTM | Exp 2 | 61/300, 20.3 [16.2, 25.2] | 273/300, 91.0 [87.2, 93.7] | 123 | 43/123 | 123/123 | 49/123 | 282 | 1 |

Tests on the two diagnostics (exploratory; Holm over their 12): primary position-level, then Fisher.

| exp | phase | A vs B | mean seeds reaching it, per position (of 3) | favour A | favour B | tie | Wilcoxon p | Holm | rank-biserial r | sign p | Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp 1 | goal lock | ACT-LSTM vs ACT | 0.71 vs 0.56 | 30 | 19 | 51 (35 both 0) | 0.0694 | 0.2081 | +0.28 | 0.1524 | 0.3048 |
| Exp 1 | goal lock | ACT-LSTM vs BC | 0.71 vs 0.00 | 50 | 0 | 50 (50 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 1 | goal lock | ACT vs BC | 0.56 vs 0.00 | 43 | 0 | 57 (57 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 2 | goal lock | ACT-LSTM vs ACT | 0.17 vs 0.26 | 12 | 23 | 65 (63 both 0) | 0.2091 | 0.4183 | -0.22 | 0.0895 | 0.2686 |
| Exp 2 | goal lock | ACT-LSTM vs BC | 0.17 vs 0.00 | 14 | 0 | 86 (86 both 0) | 0.0005 | 0.0022 | +1.00 | 0.0001 | 0.0005 |
| Exp 2 | goal lock | ACT vs BC | 0.26 vs 0.00 | 25 | 0 | 75 (75 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 1 | release | ACT-LSTM vs ACT | 0.99 vs 0.91 | 30 | 24 | 46 (27 both 0) | 0.4077 | 0.4183 | +0.12 | 0.4966 | 0.4966 |
| Exp 1 | release | ACT-LSTM vs BC | 0.99 vs 0.00 | 56 | 0 | 44 (44 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 1 | release | ACT vs BC | 0.91 vs 0.00 | 59 | 0 | 41 (41 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 2 | release | ACT-LSTM vs ACT | 0.41 vs 0.82 | 12 | 46 | 42 (26 both 0) | <0.0001 | <0.0001 | -0.64 | <0.0001 | <0.0001 |
| Exp 2 | release | ACT-LSTM vs BC | 0.41 vs 0.00 | 33 | 0 | 67 (67 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |
| Exp 2 | release | ACT vs BC | 0.82 vs 0.00 | 66 | 0 | 34 (34 both 0) | <0.0001 | <0.0001 | +1.00 | <0.0001 | <0.0001 |

| exp | phase | A vs B | reached | diff (pts) | odds ratio | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| Exp 1 | goal lock | ACT-LSTM vs ACT | 71/300 vs 56/300 | +5.0 | 1.35 | 0.1616 | 0.4848 |
| Exp 1 | goal lock | ACT-LSTM vs BC | 71/300 vs 0/300 | +23.7 | n/a | <0.0001 | <0.0001 |
| Exp 1 | goal lock | ACT vs BC | 56/300 vs 0/300 | +18.7 | n/a | <0.0001 | <0.0001 |
| Exp 2 | goal lock | ACT-LSTM vs ACT | 17/300 vs 26/300 | -3.0 | 0.633 | 0.2050 | 0.4848 |
| Exp 2 | goal lock | ACT-LSTM vs BC | 17/300 vs 0/300 | +5.7 | n/a | <0.0001 | <0.0001 |
| Exp 2 | goal lock | ACT vs BC | 26/300 vs 0/300 | +8.7 | n/a | <0.0001 | <0.0001 |
| Exp 1 | release | ACT-LSTM vs ACT | 99/300 vs 91/300 | +2.7 | 1.13 | 0.5391 | 0.5391 |
| Exp 1 | release | ACT-LSTM vs BC | 99/300 vs 0/300 | +33.0 | n/a | <0.0001 | <0.0001 |
| Exp 1 | release | ACT vs BC | 91/300 vs 0/300 | +30.3 | n/a | <0.0001 | <0.0001 |
| Exp 2 | release | ACT-LSTM vs ACT | 41/300 vs 82/300 | -13.7 | 0.421 | <0.0001 | 0.0002 |
| Exp 2 | release | ACT-LSTM vs BC | 41/300 vs 0/300 | +13.7 | n/a | <0.0001 | <0.0001 |
| Exp 2 | release | ACT vs BC | 82/300 vs 0/300 | +27.3 | n/a | <0.0001 | <0.0001 |

## 6. Proximity

- **closest palm to box:** the nearer palm's closest approach to the box before the weld (m)
- **farther palm to box:** the farther palm's closest approach before the weld: what the 0.16 m weld gate reads (m)
- **final box to goal:** box to goal centre on the last tick; placement needs 0.10 m (m)

Median [Q1, Q3], pooled over seeds.

| data | model | exp | closest palm to box (m) | farther palm to box (m) | final box to goal (m) |
|---|---|---|---|---|---|
| human | BC | Exp 1 | 0.126 [0.111, 0.175] | 0.366 [0.294, 0.426] | 1.446 [1.358, 1.518] |
| human | BC | Exp 2 | 0.119 [0.100, 0.168] | 0.394 [0.292, 0.580] | 1.553 [1.403, 1.636] |
| human | ACT | Exp 1 | 0.125 [0.107, 0.155] | 0.147 [0.123, 0.230] | 1.496 [0.397, 1.674] |
| human | ACT | Exp 2 | 0.133 [0.113, 0.226] | 0.166 [0.133, 0.451] | 1.542 [0.607, 1.624] |
| human | ACT-LSTM | Exp 1 | 0.118 [0.104, 0.136] | 0.160 [0.129, 0.280] | 1.473 [0.360, 1.629] |
| human | ACT-LSTM | Exp 2 | 0.129 [0.113, 0.136] | 0.279 [0.191, 0.350] | 1.558 [1.487, 1.615] |
| scripted | BC | Exp 1 | 0.103 [0.098, 0.109] | 0.115 [0.108, 0.121] | 1.512 [1.397, 1.602] |
| scripted | BC | Exp 2 | 0.109 [0.102, 0.114] | 0.116 [0.110, 0.135] | 1.606 [1.570, 1.635] |
| scripted | ACT | Exp 1 | 0.105 [0.085, 0.116] | 0.136 [0.130, 0.154] | 0.515 [0.157, 1.445] |
| scripted | ACT | Exp 2 | 0.074 [0.068, 0.106] | 0.150 [0.141, 0.161] | 1.365 [1.072, 1.550] |
| scripted | ACT-LSTM | Exp 1 | 0.102 [0.086, 0.114] | 0.136 [0.130, 0.146] | 0.399 [0.127, 0.598] |
| scripted | ACT-LSTM | Exp 2 | 0.084 [0.078, 0.111] | 0.138 [0.127, 0.147] | 0.423 [0.160, 0.628] |

Kruskal-Wallis across the three models, then two-sided Mann-Whitney U, Holm over the 3 pairs per metric and experiment (episodes treated as independent; human data):

| metric | exp | test | models | median (m) | stat | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| closest palm to box | Exp 1 | Kruskal-Wallis | BC, ACT, ACT-LSTM | 0.126, 0.125, 0.118 | 18.75 | <0.0001 |  |
| closest palm to box | Exp 1 | Mann-Whitney U | ACT-LSTM vs ACT | 0.118, 0.125 | 3.866e+04 | 0.0028 | 0.0056 |
| closest palm to box | Exp 1 | Mann-Whitney U | ACT-LSTM vs BC | 0.118, 0.126 | 3.604e+04 | <0.0001 | <0.0001 |
| closest palm to box | Exp 1 | Mann-Whitney U | ACT vs BC | 0.125, 0.126 | 4.249e+04 | 0.2365 | 0.2365 |
| closest palm to box | Exp 2 | Kruskal-Wallis | BC, ACT, ACT-LSTM | 0.119, 0.133, 0.129 | 39.43 | <0.0001 |  |
| closest palm to box | Exp 2 | Mann-Whitney U | ACT-LSTM vs ACT | 0.129, 0.133 | 3.479e+04 | <0.0001 | <0.0001 |
| closest palm to box | Exp 2 | Mann-Whitney U | ACT-LSTM vs BC | 0.129, 0.119 | 4.685e+04 | 0.3843 | 0.3843 |
| closest palm to box | Exp 2 | Mann-Whitney U | ACT vs BC | 0.133, 0.119 | 5.76e+04 | <0.0001 | <0.0001 |
| farther palm to box | Exp 1 | Kruskal-Wallis | BC, ACT, ACT-LSTM | 0.366, 0.147, 0.160 | 252 | <0.0001 |  |
| farther palm to box | Exp 1 | Mann-Whitney U | ACT-LSTM vs ACT | 0.160, 0.147 | 4.889e+04 | 0.0670 | 0.0670 |
| farther palm to box | Exp 1 | Mann-Whitney U | ACT-LSTM vs BC | 0.160, 0.366 | 1.582e+04 | <0.0001 | <0.0001 |
| farther palm to box | Exp 1 | Mann-Whitney U | ACT vs BC | 0.147, 0.366 | 1.598e+04 | <0.0001 | <0.0001 |
| farther palm to box | Exp 2 | Kruskal-Wallis | BC, ACT, ACT-LSTM | 0.394, 0.166, 0.279 | 190.7 | <0.0001 |  |
| farther palm to box | Exp 2 | Mann-Whitney U | ACT-LSTM vs ACT | 0.279, 0.166 | 5.7e+04 | <0.0001 | <0.0001 |
| farther palm to box | Exp 2 | Mann-Whitney U | ACT-LSTM vs BC | 0.279, 0.394 | 2.14e+04 | <0.0001 | <0.0001 |
| farther palm to box | Exp 2 | Mann-Whitney U | ACT vs BC | 0.166, 0.394 | 2e+04 | <0.0001 | <0.0001 |
| final box to goal | Exp 1 | Kruskal-Wallis | BC, ACT, ACT-LSTM | 1.446, 1.496, 1.473 | 4.025 | 0.1337 |  |
| final box to goal | Exp 1 | Mann-Whitney U | ACT-LSTM vs ACT | 1.473, 1.496 | 4.068e+04 | 0.0418 | 0.1253 |
| final box to goal | Exp 1 | Mann-Whitney U | ACT-LSTM vs BC | 1.473, 1.446 | 4.286e+04 | 0.3136 | 0.6272 |
| final box to goal | Exp 1 | Mann-Whitney U | ACT vs BC | 1.496, 1.446 | 4.699e+04 | 0.3498 | 0.6272 |
| final box to goal | Exp 2 | Kruskal-Wallis | BC, ACT, ACT-LSTM | 1.553, 1.542, 1.558 | 6.094 | 0.0475 |  |
| final box to goal | Exp 2 | Mann-Whitney U | ACT-LSTM vs ACT | 1.558, 1.542 | 4.886e+04 | 0.0692 | 0.1385 |
| final box to goal | Exp 2 | Mann-Whitney U | ACT-LSTM vs BC | 1.558, 1.553 | 4.449e+04 | 0.8105 | 0.8105 |
| final box to goal | Exp 2 | Mann-Whitney U | ACT vs BC | 1.542, 1.553 | 3.988e+04 | 0.0159 | 0.0476 |

## 7. Human vs scripted Run A (same models, 40 demos, seeds 0-2)

**Not a controlled comparison:** the training data differ in source AND labels (human walking labels smoothed, `centred_ma13_v1`; scripted raw), the evaluation caps differ (human 1568 ticks = 1.3 x the longest human demo; scripted 900), and the checkpoints were selected separately. The box positions are the same, so the paired test pairs human and scripted runs of one model position by position.

### 7a. Scripted Run A, five phases, cumulative, pooled: % [Wilson 95% CI]

| model | exp | 1 walk-in | 2 grasp | 3 lift | 4 transport | 5 place | full success |
|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 100.0 [98.7, 100.0] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] |
| BC | Exp 2 | 100.0 [98.7, 100.0] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] |
| ACT | Exp 1 | 73.7 [68.4, 78.3] | 63.0 [57.4, 68.3] | 63.0 [57.4, 68.3] | 28.3 [23.5, 33.7] | 12.0 [8.8, 16.2] | 12.0 [8.8, 16.2] |
| ACT | Exp 2 | 36.7 [31.4, 42.3] | 23.3 [18.9, 28.4] | 23.3 [18.9, 28.4] | 5.3 [3.3, 8.5] | 3.7 [2.1, 6.4] | 3.7 [2.1, 6.4] |
| ACT-LSTM | Exp 1 | 86.7 [82.4, 90.1] | 64.0 [58.4, 69.2] | 64.0 [58.4, 69.2] | 31.7 [26.7, 37.1] | 16.3 [12.6, 20.9] | 16.3 [12.6, 20.9] |
| ACT-LSTM | Exp 2 | 93.7 [90.3, 95.9] | 41.0 [35.6, 46.6] | 41.0 [35.6, 46.6] | 22.0 [17.7, 27.0] | 16.3 [12.6, 20.9] | 16.3 [12.6, 20.9] |

Mean +- SD over seeds:

| model | exp | 1 walk-in | 2 grasp | 3 lift | 4 transport | 5 place | full success |
|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 100.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |
| BC | Exp 2 | 100.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |
| ACT | Exp 1 | 73.7 +- 4.2 | 63.0 +- 1.0 | 63.0 +- 1.0 | 28.3 +- 9.7 | 12.0 +- 2.6 | 12.0 +- 2.6 |
| ACT | Exp 2 | 36.7 +- 21.5 | 23.3 +- 18.9 | 23.3 +- 18.9 | 5.3 +- 8.4 | 3.7 +- 6.4 | 3.7 +- 6.4 |
| ACT-LSTM | Exp 1 | 86.7 +- 7.1 | 64.0 +- 3.6 | 64.0 +- 3.6 | 31.7 +- 4.5 | 16.3 +- 5.5 | 16.3 +- 5.5 |
| ACT-LSTM | Exp 2 | 93.7 +- 11.0 | 41.0 +- 12.5 | 41.0 +- 12.5 | 22.0 +- 9.5 | 16.3 +- 9.8 | 16.3 +- 9.8 |

Conditional:

| model | exp | 1 walk-in | 2 grasp | 3 lift | 4 transport | 5 place | full success |
|---|---|---|---|---|---|---|---|
| BC | Exp 1 | 300/300, 100.0 [98.7, 100.0] | 0/300, 0.0 [0.0, 1.3] | 0/0 | 0/0 | 0/0 | 0/0 |
| BC | Exp 2 | 300/300, 100.0 [98.7, 100.0] | 0/300, 0.0 [0.0, 1.3] | 0/0 | 0/0 | 0/0 | 0/0 |
| ACT | Exp 1 | 221/300, 73.7 [68.4, 78.3] | 189/221, 85.5 [80.3, 89.6] | 189/189, 100.0 [98.0, 100.0] | 85/189, 45.0 [38.1, 52.1] | 36/85, 42.4 [32.4, 53.0] | 36/36, 100.0 [90.4, 100.0] |
| ACT | Exp 2 | 110/300, 36.7 [31.4, 42.3] | 70/110, 63.6 [54.3, 72.0] | 70/70, 100.0 [94.8, 100.0] | 16/70, 22.9 [14.6, 34.0] | 11/16, 68.8 [44.4, 85.8] | 11/11, 100.0 [74.1, 100.0] |
| ACT-LSTM | Exp 1 | 260/300, 86.7 [82.4, 90.1] | 192/260, 73.8 [68.2, 78.8] | 192/192, 100.0 [98.0, 100.0] | 95/192, 49.5 [42.5, 56.5] | 49/95, 51.6 [41.7, 61.4] | 49/49, 100.0 [92.7, 100.0] |
| ACT-LSTM | Exp 2 | 281/300, 93.7 [90.3, 95.9] | 123/281, 43.8 [38.1, 49.6] | 123/123, 100.0 [97.0, 100.0] | 66/123, 53.7 [44.9, 62.2] | 49/66, 74.2 [62.6, 83.3] | 49/49, 100.0 [92.7, 100.0] |

### 7b. Human - scripted, per model (position-level primary; Holm over the 30 phase x experiment x model tests)

| exp | phase | model | mean per position, human vs scripted (of 3) | favour human | favour scripted | tie | Wilcoxon p | Holm | r | sign p | Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp 1 | walk-in | BC | 1.78 vs 3.00 | 0 | 75 | 25 | <0.0001 | <0.0001 | -1.00 | <0.0001 | <0.0001 |
| Exp 2 | walk-in | BC | 1.13 vs 3.00 | 0 | 100 | 0 | <0.0001 | <0.0001 | -1.00 | <0.0001 | <0.0001 |
| Exp 1 | walk-in | ACT | 1.69 vs 2.21 | 11 | 39 | 50 | <0.0001 | 0.0013 | -0.64 | <0.0001 | 0.0018 |
| Exp 2 | walk-in | ACT | 1.29 vs 1.10 | 41 | 26 | 33 | 0.2007 | 1.0000 | +0.18 | 0.0864 | 1.0000 |
| Exp 1 | walk-in | ACT-LSTM | 1.68 vs 2.60 | 4 | 46 | 50 | <0.0001 | <0.0001 | -0.90 | <0.0001 | <0.0001 |
| Exp 2 | walk-in | ACT-LSTM | 0.37 vs 2.81 | 0 | 100 | 0 | <0.0001 | <0.0001 | -1.00 | <0.0001 | <0.0001 |
| Exp 1 | grasp | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 2 | grasp | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 1 | grasp | ACT | 1.40 vs 1.89 | 15 | 39 | 46 | 0.0004 | 0.0065 | -0.54 | 0.0015 | 0.0284 |
| Exp 2 | grasp | ACT | 0.15 vs 0.70 | 4 | 41 | 55 | <0.0001 | <0.0001 | -0.87 | <0.0001 | <0.0001 |
| Exp 1 | grasp | ACT-LSTM | 1.33 vs 1.92 | 21 | 39 | 40 | 0.0001 | 0.0023 | -0.56 | 0.0273 | 0.4374 |
| Exp 2 | grasp | ACT-LSTM | 0.17 vs 1.23 | 1 | 52 | 47 | <0.0001 | <0.0001 | -0.99 | <0.0001 | <0.0001 |
| Exp 1 | lift | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 2 | lift | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 1 | lift | ACT | 1.40 vs 1.89 | 15 | 39 | 46 | 0.0004 | 0.0065 | -0.54 | 0.0015 | 0.0284 |
| Exp 2 | lift | ACT | 0.15 vs 0.70 | 4 | 41 | 55 | <0.0001 | <0.0001 | -0.87 | <0.0001 | <0.0001 |
| Exp 1 | lift | ACT-LSTM | 1.33 vs 1.92 | 21 | 39 | 40 | 0.0001 | 0.0023 | -0.56 | 0.0273 | 0.4374 |
| Exp 2 | lift | ACT-LSTM | 0.17 vs 1.23 | 1 | 52 | 47 | <0.0001 | <0.0001 | -0.99 | <0.0001 | <0.0001 |
| Exp 1 | transport | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 2 | transport | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 1 | transport | ACT | 1.23 vs 0.85 | 39 | 16 | 45 | 0.0012 | 0.0184 | +0.48 | 0.0027 | 0.0453 |
| Exp 2 | transport | ACT | 0.11 vs 0.16 | 6 | 11 | 83 | 0.3147 | 1.0000 | -0.26 | 0.3323 | 1.0000 |
| Exp 1 | transport | ACT-LSTM | 1.06 vs 0.95 | 35 | 27 | 38 | 0.4434 | 1.0000 | +0.11 | 0.3742 | 1.0000 |
| Exp 2 | transport | ACT-LSTM | 0.17 vs 0.66 | 2 | 32 | 66 | <0.0001 | <0.0001 | -0.93 | <0.0001 | <0.0001 |
| Exp 1 | place | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 2 | place | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 1 | place | ACT | 0.37 vs 0.36 | 22 | 21 | 57 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 2 | place | ACT | 0.06 vs 0.11 | 4 | 9 | 87 | 0.1779 | 1.0000 | -0.38 | 0.2668 | 1.0000 |
| Exp 1 | place | ACT-LSTM | 0.34 vs 0.49 | 20 | 29 | 51 | 0.1465 | 1.0000 | -0.23 | 0.2529 | 1.0000 |
| Exp 2 | place | ACT-LSTM | 0.02 vs 0.49 | 0 | 32 | 68 | <0.0001 | <0.0001 | -1.00 | <0.0001 | <0.0001 |
| Exp 1 | full success | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 2 | full success | BC | 0.00 vs 0.00 | 0 | 0 | 100 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 1 | full success | ACT | 0.37 vs 0.36 | 22 | 21 | 57 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 1.0000 |
| Exp 2 | full success | ACT | 0.06 vs 0.11 | 4 | 9 | 87 | 0.1779 | 0.7323 | -0.38 | 0.2668 | 1.0000 |
| Exp 1 | full success | ACT-LSTM | 0.34 vs 0.49 | 20 | 29 | 51 | 0.1465 | 0.7323 | -0.23 | 0.2529 | 1.0000 |
| Exp 2 | full success | ACT-LSTM | 0.02 vs 0.49 | 0 | 32 | 68 | <0.0001 | <0.0001 | -1.00 | <0.0001 | <0.0001 |

Secondary, Fisher on pooled counts:

| exp | phase | model | human vs scripted | diff (pts) | p | p (Holm) |
|---|---|---|---|---|---|---|
| Exp 1 | walk-in | BC | 178/300 vs 300/300 | -40.7 | <0.0001 | <0.0001 |
| Exp 1 | walk-in | ACT | 169/300 vs 221/300 | -17.3 | <0.0001 | 0.0002 |
| Exp 1 | walk-in | ACT-LSTM | 168/300 vs 260/300 | -30.7 | <0.0001 | <0.0001 |
| Exp 2 | walk-in | BC | 113/300 vs 300/300 | -62.3 | <0.0001 | <0.0001 |
| Exp 2 | walk-in | ACT | 129/300 vs 110/300 | +6.3 | 0.1333 | 1.0000 |
| Exp 2 | walk-in | ACT-LSTM | 37/300 vs 281/300 | -81.3 | <0.0001 | <0.0001 |
| Exp 1 | grasp | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 1 | grasp | ACT | 140/300 vs 189/300 | -16.3 | <0.0001 | 0.0014 |
| Exp 1 | grasp | ACT-LSTM | 133/300 vs 192/300 | -19.7 | <0.0001 | <0.0001 |
| Exp 2 | grasp | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 2 | grasp | ACT | 15/300 vs 70/300 | -18.3 | <0.0001 | <0.0001 |
| Exp 2 | grasp | ACT-LSTM | 17/300 vs 123/300 | -35.3 | <0.0001 | <0.0001 |
| Exp 1 | lift | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 1 | lift | ACT | 140/300 vs 189/300 | -16.3 | <0.0001 | 0.0014 |
| Exp 1 | lift | ACT-LSTM | 133/300 vs 192/300 | -19.7 | <0.0001 | <0.0001 |
| Exp 2 | lift | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 2 | lift | ACT | 15/300 vs 70/300 | -18.3 | <0.0001 | <0.0001 |
| Exp 2 | lift | ACT-LSTM | 17/300 vs 123/300 | -35.3 | <0.0001 | <0.0001 |
| Exp 1 | transport | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 1 | transport | ACT | 123/300 vs 85/300 | +12.7 | 0.0015 | 0.0221 |
| Exp 1 | transport | ACT-LSTM | 106/300 vs 95/300 | +3.7 | 0.3871 | 1.0000 |
| Exp 2 | transport | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 2 | transport | ACT | 11/300 vs 16/300 | -1.7 | 0.4314 | 1.0000 |
| Exp 2 | transport | ACT-LSTM | 17/300 vs 66/300 | -16.3 | <0.0001 | <0.0001 |
| Exp 1 | place | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 1 | place | ACT | 37/300 vs 36/300 | +0.3 | 1.0000 | 1.0000 |
| Exp 1 | place | ACT-LSTM | 34/300 vs 49/300 | -5.0 | 0.0974 | 1.0000 |
| Exp 2 | place | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 2 | place | ACT | 6/300 vs 11/300 | -1.7 | 0.3253 | 1.0000 |
| Exp 2 | place | ACT-LSTM | 2/300 vs 49/300 | -15.7 | <0.0001 | <0.0001 |
| Exp 1 | full success | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 1 | full success | ACT | 37/300 vs 36/300 | +0.3 | 1.0000 | 1.0000 |
| Exp 1 | full success | ACT-LSTM | 34/300 vs 49/300 | -5.0 | 0.0974 | 0.4870 |
| Exp 2 | full success | BC | 0/300 vs 0/300 | +0.0 | 1.0000 | 1.0000 |
| Exp 2 | full success | ACT | 6/300 vs 11/300 | -1.7 | 0.3253 | 1.0000 |
| Exp 2 | full success | ACT-LSTM | 2/300 vs 49/300 | -15.7 | <0.0001 | <0.0001 |

### 7c. Generalisation gap, scripted

| model | phase | Exp 1 | Exp 2 | gap (pts) | Newcombe 95% CI | Wald 95% CI | SE over seeds | gap per seed (pts) |  |
|---|---|---|---|---|---|---|---|---|---|
| BC | walk-in | 300/300 | 300/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 |  |
| BC | grasp | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| BC | lift | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| BC | transport | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| BC | place | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| BC | full success | 0/300 | 0/300 | +0.0 | -1.3 to +1.3 | +0.0 to +0.0 | 0.0 | +0, +0, +0 | floor (0 vs 0) |
| ACT | walk-in | 221/300 | 110/300 | +37.0 | +29.3 to +44.0 | +29.6 to +44.4 | 10.1 | +24, +30, +57 |  |
| ACT | grasp | 189/300 | 70/300 | +39.7 | +32.1 to +46.5 | +32.4 to +46.9 | 11.4 | +17, +49, +53 |  |
| ACT | lift | 189/300 | 70/300 | +39.7 | +32.1 to +46.5 | +32.4 to +46.9 | 11.4 | +17, +49, +53 |  |
| ACT | transport | 85/300 | 16/300 | +23.0 | +17.3 to +28.7 | +17.3 to +28.7 | 1.5 | +24, +25, +20 |  |
| ACT | place | 36/300 | 11/300 | +8.3 | +4.1 to +12.8 | +4.1 to +12.6 | 3.5 | +2, +14, +9 |  |
| ACT | full success | 36/300 | 11/300 | +8.3 | +4.1 to +12.8 | +4.1 to +12.6 | 3.5 | +2, +14, +9 |  |
| ACT-LSTM | walk-in | 260/300 | 281/300 | -7.0 | -11.9 to -2.2 | -11.7 to -2.3 | 2.9 | -7, -2, -12 |  |
| ACT-LSTM | grasp | 192/300 | 123/300 | +23.0 | +15.1 to +30.5 | +15.2 to +30.8 | 6.5 | +36, +16, +17 |  |
| ACT-LSTM | lift | 192/300 | 123/300 | +23.0 | +15.1 to +30.5 | +15.2 to +30.8 | 6.5 | +36, +16, +17 |  |
| ACT-LSTM | transport | 95/300 | 66/300 | +9.7 | +2.6 to +16.6 | +2.6 to +16.7 | 7.9 | +25, +5, -1 |  |
| ACT-LSTM | place | 49/300 | 49/300 | +0.0 | -5.9 to +5.9 | -5.9 to +5.9 | 8.6 | +17, -6, -11 |  |
| ACT-LSTM | full success | 49/300 | 49/300 | +0.0 | -5.9 to +5.9 | -5.9 to +5.9 | 8.6 | +17, -6, -11 |  |

### 7d. Proximity, human vs scripted (Mann-Whitney U, Holm over the 3 models per metric and experiment)

| metric | exp | model | median human vs scripted (m) | n | p | p (Holm) |
|---|---|---|---|---|---|---|
| closest palm to box | Exp 1 | BC | 0.126 vs 0.103 | 300, 300 | <0.0001 | <0.0001 |
| closest palm to box | Exp 1 | ACT | 0.125 vs 0.105 | 300, 300 | <0.0001 | <0.0001 |
| closest palm to box | Exp 1 | ACT-LSTM | 0.118 vs 0.102 | 300, 300 | <0.0001 | <0.0001 |
| closest palm to box | Exp 2 | BC | 0.119 vs 0.109 | 300, 300 | <0.0001 | <0.0001 |
| closest palm to box | Exp 2 | ACT | 0.133 vs 0.074 | 300, 300 | <0.0001 | <0.0001 |
| closest palm to box | Exp 2 | ACT-LSTM | 0.129 vs 0.084 | 300, 300 | <0.0001 | <0.0001 |
| farther palm to box | Exp 1 | BC | 0.366 vs 0.115 | 300, 300 | <0.0001 | <0.0001 |
| farther palm to box | Exp 1 | ACT | 0.147 vs 0.136 | 300, 300 | 0.0537 | 0.0537 |
| farther palm to box | Exp 1 | ACT-LSTM | 0.160 vs 0.136 | 300, 300 | <0.0001 | <0.0001 |
| farther palm to box | Exp 2 | BC | 0.394 vs 0.116 | 300, 300 | <0.0001 | <0.0001 |
| farther palm to box | Exp 2 | ACT | 0.166 vs 0.150 | 300, 300 | <0.0001 | <0.0001 |
| farther palm to box | Exp 2 | ACT-LSTM | 0.279 vs 0.138 | 300, 300 | <0.0001 | <0.0001 |
| final box to goal | Exp 1 | BC | 1.446 vs 1.512 | 300, 300 | <0.0001 | <0.0001 |
| final box to goal | Exp 1 | ACT | 1.496 vs 0.515 | 300, 300 | <0.0001 | <0.0001 |
| final box to goal | Exp 1 | ACT-LSTM | 1.473 vs 0.399 | 300, 300 | <0.0001 | <0.0001 |
| final box to goal | Exp 2 | BC | 1.553 vs 1.606 | 300, 300 | <0.0001 | <0.0001 |
| final box to goal | Exp 2 | ACT | 1.542 vs 1.365 | 300, 300 | <0.0001 | <0.0001 |
| final box to goal | Exp 2 | ACT-LSTM | 1.558 vs 0.423 | 300, 300 | <0.0001 | <0.0001 |

![Human vs scripted](fig_human_vs_scripted.png)

Scripted figures in the same form: ![Scripted cumulative](fig_phase_cumulative_scripted.png)

![Scripted gap](fig_generalization_gap_scripted.png)

## 8. Teleoperation: the 50 human demonstrations

From `tools/teleop_metrics.py recordings/demonstrations` (tables in `teleop/`). One operator, 11 sessions.

| attempts | accepted | acceptance [Wilson CI] | failed | discarded | of which successes | revoked | unfinished | wall s / attempt | wall s / accepted demo | tick rate (Hz) | late ticks (%) | dropout (%) | demo ticks min / median / max | demo s median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 289 | 50 | 17.3% [13.4, 22.1] | 2 | 236 | 29 | 4 | 1 | 25.9 | 149.0 | 24.93 | 1.82 | 0.00 | 526 / 715.5 / 1206 | 28.6 |

**Acceptance = accepted / attempts, every outcome in the denominator.** Most non-accepts are operator discards (236 of 289); 29 of the discards were episodes that met the success rule (including 4 accepts later revoked; the acceptance rules are CLAUDE.md section 8 2026-11-13c), so acceptance understates how often the task was completed. Session dates are the ledger's.

### Per session

| session | date | attempts | accepted | acceptance % [CI] | discarded successes | wall s / accepted demo | tick rate (Hz) | late ticks (%) | dropout (%) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-13 | 25 | 4 | 16.0 [6.4, 34.7] | 3 | 206.5 | 24.98 | 1.34 | 0.00 |
| 2 | 2026-11-13 | 25 | 4 | 16.0 [6.4, 34.7] | 3 | 176.2 | 25.00 | 0.23 | 0.00 |
| 3 | 2026-11-13 | 38 | 6 | 15.8 [7.4, 30.4] | 5 | 127.0 | 25.00 | 0.94 | 0.00 |
| 4 | 2026-11-13 | 28 | 7 | 25.0 [12.7, 43.4] | 7 | 98.4 | 24.99 | 1.99 | 0.00 |
| 5 | 2026-11-13 | 13 | 2 | 15.4 [4.3, 42.2] | 2 | 171.0 | 25.00 | 2.03 | 0.00 |
| 6 | 2026-11-13 | 54 | 8 | 14.8 [7.7, 26.6] | 4 | 172.0 | 25.00 | 2.27 | 0.00 |
| 7 | 2026-11-13 | 42 | 11 | 26.2 [15.3, 41.1] | 4 | 110.9 | 24.99 | 2.03 | 0.00 |
| 8 | 2026-11-13 | 23 | 2 | 8.7 [2.4, 26.8] | 0 | 205.5 | 24.99 | 2.24 | 0.00 |
| 9 | 2026-11-13 | 24 | 3 | 12.5 [4.3, 31.0] | 0 | 208.3 | 24.99 | 2.80 | 0.00 |
| 10 | 2026-11-13 | 9 | 2 | 22.2 [6.3, 54.7] | 0 | 135.5 | 23.29 | 3.20 | 0.00 |
| 11 | 2026-11-13 | 8 | 1 | 12.5 [2.2, 47.1] | 1 | 221.0 | 24.99 | 3.29 | 0.00 |

**Learning trend:** Spearman of acceptance vs attempt index rho = -0.025 (p 0.6776); first half 26/144 vs second half 24/145 (Fisher p 0.7580); per-session acceptance vs session index rho = -0.379 (p 0.2504).

![Teleop learning curve](fig_teleop_learning.png)

## Caveats, to carry into the write-up

- **Five phases is a 2026-11-22 decision.** The proposal and earlier reports had four (grasp, lift, transport, place); section 1e keeps the four-phase numbers. Walk-in is a navigation-to-the-box outcome, not a manipulation target, so it does not change what Objective 4 varies (box position only, Q6/TR13).
- **Two levels of inference, neither over training seeds.** Fisher treats episodes as independent; the position-level test keeps the pairing by position, but its 0-3 counts come from the same 3 seeds. With 3 seeds an exact test on seed rates cannot go below p = 0.1, so nothing here says what a new training seed would do. Read the per-seed values.
- **Wilcoxon with few untied positions.** Where only a handful of positions differ (e.g. Exp 2 place, 2-6 untied), the normal approximation is poor (scipy warns); read the exact sign test there.
- **The walk-in depends on side.** Right-side spawns lock far more often than left-side ones for every model (section 4b), and the held-out patch is centre-left. The raw Exp 1 - Exp 2 gap therefore mixes novelty with lateral position; section 3b gives an exploratory adjustment.
- **Multiplicity.** Holm is applied within each named family (30 phase tests; 6 success; 12 diagnostics; and the same for human vs scripted), not across families, test types or the proximity metrics.
- **Cumulative vs marginal.** The cumulative chain requires the walk-in; on human data this changes nothing (no grasp without a pickup lock), on scripted 2 of 1200 episodes.
- **Free-based welds.** Some welds engaged with no pickup lock (section 5): the base was not held (D12) at the weld.
- **Selection is noisy.** Checkpoints were chosen on 20 episodes; chosen steps range 40k-180k across ACT-LSTM seeds.
- **BC's budget.** BC is epoch-budgeted (35,400 optimizer steps) against 200k for ACT and ACT-LSTM.
- **ACT-LSTM training exits** with 0xC0000409 (the cuDNN exit crash, CLAUDE.md 2026-09-29); every run reached its 200k cap and saved its checkpoints first.
- **Human vs scripted is confounded** (section 7): data source, label smoothing, evaluation cap and selection all differ.
- **Spawn sides are uneven in Exp 2**: the held-out patch has no right third, so 'by side' pooled over experiments weights centre/left by Exp 2.

## Files

CSV tables beside this report: `audit_notes.csv`, `diagnostics.csv`, `episodes.csv`, `failure_by_side.csv`, `failure_by_side_exp.csv`, `failure_phase.csv`, `failure_reason_deploy.csv`, `generalization_gap.csv`, `generalization_gap_lateral.csv`, `human_vs_scripted_fisher.csv`, `human_vs_scripted_position.csv`, `human_vs_scripted_proximity.csv`, `phase_by_run.csv`, `phase_pooled.csv`, `proximity_summary.csv`, `proximity_tests.csv`, `runs.csv`, `spawn_side.csv`, `spawn_side_tests.csv`, `teleop_trend.csv`, `tests_fisher_diagnostics.csv`, `tests_fisher_phases.csv`, `tests_fisher_success.csv`, `tests_position_diagnostics.csv`, `tests_position_phases.csv`, `tests_position_success.csv`. Figures: PNG at 300 dpi and SVG with the same stem.

