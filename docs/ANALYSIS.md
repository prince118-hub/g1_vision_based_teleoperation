# Experiment 1 / 2 analysis — `tools/analyze_experiments.py`

Turns the output of `tools/eval_scripted.py` into one report folder: CSV tables, a
markdown report and PNG figures. It only **reads** the eval folder. It runs nothing and
writes nothing there, and it needs no GPU, no MuJoCo and no torch.

## Setup

```
pip install numpy scipy pandas matplotlib pytest
```

These are not in `requirements.txt` (that file is pinned to the training build). On the
pinned env (NumPy 1.26.4) use versions that still support NumPy 1.x, e.g.
`scipy<1.14 pandas<2.3 matplotlib<3.10`.

## Run

```
python tools/analyze_experiments.py                         # runs/scripted_v3/eval -> runs/scripted_v3/analysis
python tools/analyze_experiments.py --eval-root runs/scripted_v3/eval --out reports/exp12
python tools/analyze_experiments.py --eval-root A/eval B/eval --out reports/both
python tools/analyze_experiments.py --no-figures            # tables + report only
python tools/analyze_experiments.py --splits 160=data/processed/splits_scripted_v3.json n40=data/processed/splits_scripted_v3_n40.json
```

**Input.** Each eval root holds one folder per run, named `<model>_<size>_seed<k>` (`bc_160_seed0`,
`act_lstm_n40_seed2`, ...). Inside it are `exp1.jsonl` / `exp2.jsonl` (one `tools/deploy.py` record per
episode) and `selection.json`. Any folder holding an `exp1.jsonl` or `exp2.jsonl` counts as a run; the
`select/` and `logs/` folders are ignored. **Missing runs are fine.** The report lists what it found and
what is missing (a run with only Exp 1, a model with no runs, a data size with one seed), and leaves
empty any cell it cannot compute instead of making a value up.

`--splits` gives the training split files for the coverage map. The default is the scripted_v3 160 and
n40 splits. A box spawn is a pure function of its seed, so the map needs only the seeds.

## Output (`--out`, default `<eval root>/../analysis/`)

| file | content |
|---|---|
| `report.md` | everything below, labelled with model, data size, training seeds, checkpoint, lock rule at evaluation, and the lock rule the training data was collected under |
| `runs.csv`, `episodes.csv` | the runs found; every episode record used, flattened |
| `success_by_run.csv` | model × size × seed × experiment: successes/n, rate, Wilson 95% CI |
| `success_summary.csv` | mean ± SD over training seeds; pooled successes/n with Wilson 95% CI |
| `generalization_gap.csv` | Exp 1 − Exp 2 per model × size, SE (binomial), 95% CI, SE over seeds |
| `stage_rates.csv` | grasp / lift / transport / place / success, with counts and CIs |
| `failure_phase.csv` | deploy.py `fail_reason` counts: WALK_IN, GRASP, LIFT, TRANSPORT, PLACE, fell |
| `fisher.csv` | pairwise Fisher exact test on success (BC / ACT / ACT-LSTM), pooled over seeds, per experiment and data size, raw and Holm-adjusted p |
| `distance_tests.csv` | Kruskal–Wallis + pairwise Mann-Whitney U (Holm) on grasp proximity and final box-to-goal distance |
| `thresholds.csv` | final / closest box-to-goal within 5 and 10 cm; palms within the 0.16 m weld gate |
| `size_comparison.csv` | n40 vs 160 per model and experiment (when both exist), with Fisher p |
| `fig_success.png` | success bars, Wilson 95% CI, per-seed dots |
| `fig_stages.png` | stage-wise rates per experiment |
| `fig_failure_phase.png` | stacked failure phase |
| `fig_proximity.png` | boxplots of grasp proximity (0.16 m gate line) and final box-to-goal distance (5 / 10 cm lines) |
| `fig_success_by_position.png` | success rate per spawn cell, sizes × models, every episode marked, held-out patch dashed |
| `fig_training_coverage.png` | training-demo spawns per cell for each split |
| `fig_size_comparison.png` | n40 vs 160 (only when both exist) |

## What the numbers are

- **Success** is deploy.py's `success`, which is `success.evaluate` and not a fall. **Stages** are its
  `grasp`/`lift`/`transport`/`place` (each one implies the one before it).
- **Grasp proximity** is `palm_box_min_m`: the farther palm's closest approach to the box before the weld.
  The weld needs both palms within 0.16 m. **Final distance** is `final_box_goal_m`, the box-to-goal
  distance on the last tick (the same quantity as the placement error). **Closest** is `closest_box_goal_m`.
  Records from before these fields existed are dropped from those tests and counted in the `dropped`
  column.
- **Wilson** score 95% intervals. **Pooled** treats every seed's episodes as one binomial sample, which
  ignores the variation between seeds; the SD column shows that variation.
- **Gap SE (binomial)** = sqrt(p1(1−p1)/n1 + p2(1−p2)/n2). This holds because the Exp 1 and Exp 2 seed sets
  are disjoint, so the two samples are independent. **SE (seeds)** = SD of the per-seed gaps / sqrt(k), shown
  only when there are at least 2 seeds.
- **Fisher's exact test** is two-sided. **Kruskal–Wallis** is followed by two-sided **Mann-Whitney U**.
  **Holm–Bonferroni** correction is applied over the model pairs within one (data size, experiment[, metric])
  family.
- With 100 episodes per experiment and one seed, a rate near 50% has SE ≈ 5%, so differences under about
  15 points are not resolvable (CLAUDE.md §8, dataset design). Read the CIs, not the bar heights.
- **Disclose** when reading 40 vs 160: BC is epoch-budgeted, so BC-n40 ran about 4× fewer optimizer steps
  than BC-160. ACT and ACT-LSTM are step-budgeted (CLAUDE.md §8 2026-10-23).

## Tests

```
python tools/test_analyze_experiments.py        # or: pytest tools/test_analyze_experiments.py
```

The tests build synthetic eval folders with known counts: Run B's measured counts plus a 3-seed n40 run.
They check the tables entry by entry. Wilson intervals are checked against Newcombe (1998) worked examples.
Fisher p-values are checked against the hypergeometric distribution computed directly, and Kruskal–Wallis H
against the rank formula. Mann-Whitney U is checked on complete separation and Holm by hand. They also cover
a partial folder (missing runs, missing Exp 2, old records without the proximity field) and check that the
spawn mirror equals `g1_teleop/box_reset.py` where MuJoCo is installed.
