# Report extras: how to run

Four read-only tools for tables the thesis needs that no other tool writes. They only **read** recording, run
and eval folders; nothing is written into them. Outputs go under `runs/` (gitignored) unless `--out` says
otherwise. None of them imports mujoco; only the optional inference benchmark needs torch.

```
pip install numpy scipy pandas matplotlib pytest
python -m pytest tools/test_report_extras.py tools/test_teleop_metrics.py tools/test_sus_score.py -q
```

The tests use synthetic fixtures with hand-computed answers (a SUS form scoring 80.0, a Spearman ρ of 0.8,
a learning curve, dropout 40/1150 ticks, …); they need neither mujoco nor torch.

## 1. `tools/report_extras.py`: data quality, compute, offline vs real

```
python tools/report_extras.py \
    --datasets recordings/episodes recordings/demonstrations \
    --run-root runs/scripted_v3 \
    --out runs/report_extras
```

| Flag | Meaning |
|---|---|
| `--datasets DIR …` | recording folders, each holding its `ledger.jsonl` and episode `.npz` files |
| `--runs DIR …` / `--run-root DIR` | run directories; with `--run-root`, every subfolder that holds `metadata.json` or `result.json` |
| `--eval-root DIR` | `tools/eval_scripted.py` output (default `<run-root>/eval`, else `<run's parent>/eval`) |
| `--no-bench` | skip the inference benchmark |
| `--bench-device cpu\|cuda\|auto` | default `cpu`, so it never competes with a GPU job; use `cuda` for the deployment-device number, and only when nothing else is running |

Writes `data_quality.csv`, `compute.csv`, `offline_vs_real.csv`, `offline_vs_real_spearman.csv`,
`offline_vs_real.png` and `report_extras.md`.

- **(a) Data quality**, one row per dataset folder:
  - **attempts** = `issue` rows.
  - **accepted / rejected / errors** come from `EpisodeLedger.summary`, which takes the **last** decision per seed.
  - **episode length** is the min / median / max over the accepted files (`failed/` is excluded).
  - **spec version, lock rule, grip label, source and demonstrator settings / operators** are read from each file's meta. If the files disagree, the value shows as `MIXED: …`.
  - **tracking dropout** is counted only over episodes that measured it (`tracking_ok`). Scripted episodes read `n/a (not measured)`.
  - **replay check**: `tools/replay_check.py` only prints, so save its output as `<dataset>/replay_check.txt` (`python tools/replay_check.py "<dataset>/ep_*.npz" > <dataset>/replay_check.txt`). Its `REPLAY PASS/FAIL` and `MANIPULATION reproduced a/b` lines are then reported. Without that file the column reads "not run / not saved".
- **(b) Compute**, one row per run:
  - **wall time, optimizer steps, steps/s, parameters and training device** come from `result.json` and `metadata.json`. A killed run falls back to its last `metrics.jsonl` row.
  - **Peak RAM / GPU memory is not logged by `train_bc.py`.** It is read only from a `resources.json` sidecar in the run folder (`{"peak_ram_gb": 5.1, "peak_gpu_gb": 2.3}`, e.g. read off Task Manager at the end of a run) or from a `resources` dict in `result.json` / `metadata.json`. Otherwise it reads n/a.
  - **Inference**: ms per policy call for the **model forward only** (`model(obs)` on a `(1, W_o, 48)` window under `no_grad`, 20 warm-up calls + 200 timed calls). It times the selected checkpoint from `selection.json`, else `best.pt` / `last.pt`. It needs torch plus the model stack (`train_bc.py`); without them it reads `n/a` with the reason.
- **(c) Offline vs real**: every deployed selection checkpoint (aliases skipped) gives one point:
  - **x**: its validation loss, taken from the checkpoint, else from the `metrics.jsonl` row at that step / epoch.
  - **y**: its full successes on the 20 selection seeds.
  - **Spearman ρ** is computed per run and pooled per model. Agreement shows up as a **negative** ρ. When every checkpoint of a run scored the same success (e.g. BC-160, 0/20 everywhere), ρ is **undefined** and is reported that way, not as 0. The plot has one panel per model, and the selected checkpoint is ringed.

## 2. `tools/teleop_metrics.py`: operator metrics from a teleop recording folder

```
python tools/teleop_metrics.py recordings/demonstrations            # real collection
python tools/teleop_metrics.py recordings/teleop_fake                # fake-source dry runs
python tools/teleop_metrics.py recordings/demonstrations --operator charles --window 5
```

Writes `attempts.csv`, `operators.csv`, `learning_curve.png` and `teleop_metrics.md` to
`runs/teleop_metrics/<folder>/` (or `--out`).

- **Attempt**: an `issue` row, closed by the next accept / discard / error row for the same seed.
- **Outcome**: accept, failed (saved to `failed/`), discard (operator `x`, nothing written), error, or unfinished (crash).
- **Success rate** = accepted / attempted.
- **Time per attempt**: the issue → decision interval from the ledger (1 s resolution). It does not include the idle screen before ENTER. Simulated time (ticks / 25 Hz) is reported beside it.
- **Time per accepted demo** = the operator's total attempt time, failures included, divided by accepted.
- **Learning curve**: per operator, for each attempt index: success, cumulative success rate, rolling success rate over `--window` attempts, and wall time.
- **Dropout %**: tick-weighted over saved episodes (`tracking_ok`).
- **Tick rate and overruns**: from each episode's `timing` block. Headless fake runs are unpaced, so their tick rate is simulation speed, not 25 Hz.

## 3. SUS questionnaire: `docs/SUS_FORM.md` + `tools/sus_score.py`

Print `docs/SUS_FORM.md` (standard 10 items, 1–5). Transcribe each form as one CSV row:

```csv
operator,session,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
charles,2026-11-02,4,1,5,2,3,3,4,2,5,1
```

```
python tools/sus_score.py responses.csv --out runs/sus
```

Standard scoring: odd items give answer − 1, even items give 5 − answer, and the sum is multiplied by 2.5 (range 0–100). The example row scores 80.0.

- Each operator's score is the mean of their rows.
- The overall score is the mean over **operators**, so each operator counts once. The mean over all responses is printed beside it.
- A blank or out-of-range answer refuses the file, naming the row.
