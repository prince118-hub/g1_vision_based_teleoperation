# Videos of the Exp 1 / Exp 2 episodes

Four tools, run in this order. All of them read a **results folder**, the folder `tools/eval_scripted.py` calls
`--run-root`: run folders (`<root>/<run>/metadata.json` + checkpoints) with `eval/<run>/selection.json`,
`exp1.jsonl` and `exp2.jsonl` beside them. The tools do not care whether the runs were trained on scripted or
teleoperated demonstrations, so every command below works for either. Only the folder and the run names change.

| tool | what it makes | simulates? |
|---|---|---|
| `tools/pick_video_episodes.py` | `picks.json` / `.csv`: the seeds worth filming | no (eval JSONLs only) |
| `tools/video_side_by_side.py` | one seed, several models, one MP4 | yes, one `deploy.py` episode per panel |
| `tools/record_exp_videos.py` (existing) | per run: every success + 2 typical failures, `index.csv` | yes, one episode per video |
| `tools/video_highlight_reel.py` | a 1-2 min montage with title cards and captions | no (existing MP4s only) |

**Before you start**

- Evaluate first: `tools/eval_scripted.py` must have written `selection.json`, `exp1.jsonl` and `exp2.jsonl` for every run you film.
- Film on the machine that ran the evaluation (CLAUDE.md O34, Option B: results come from the NVIDIA laptop). A video
  is a re-run of the evaluated episode, and its trajectory hash is checked against the eval record. On another
  machine the hashes can differ. When that happens the tool marks the video and exits with code 1; it never hides it.
- One heavy process at a time. The simulating tools refuse to start while `train_bc.py` or `deploy.py` runs.
- `--plan` prints what `video_side_by_side.py` and `video_highlight_reel.py` would do and runs nothing.
- The videos are mp4v, as `deploy.py` writes them. If a browser or PowerPoint will not play one, add `--h264`
  (needs `ffmpeg` on PATH, or `--ffmpeg C:/path/ffmpeg.exe`).

## 1. Scripted-trained, Run B (160 demos, training seed 0)

Pick the seeds. The default contrast is ACT-LSTM vs ACT of the same size and training seed:

```
python tools/pick_video_episodes.py --results runs/scripted_v3 --runs bc_160_seed0 act_160_seed0 act_lstm_160_seed0 --out D:/thesis_videos/picks/run_b.json
```

Render the side-by-sides: ACT-LSTM wins, the reverse cases, and held-out positions solved by the most runs. Plan first:

```
python tools/video_side_by_side.py --picks D:/thesis_videos/picks/run_b.json --category contrast contrast_reverse heldout_success --out-dir D:/thesis_videos/side_by_side --plan
python tools/video_side_by_side.py --picks D:/thesis_videos/picks/run_b.json --category contrast contrast_reverse heldout_success --out-dir D:/thesis_videos/side_by_side
```

Or one seed by hand, from any experiment:

```
python tools/video_side_by_side.py --results runs/scripted_v3 --exp exp2 --seed 15 --runs bc_160_seed0 act_160_seed0 act_lstm_160_seed0 --out-dir D:/thesis_videos/side_by_side
```

Per-model clips (the existing tool):

```
python tools/record_exp_videos.py --runs bc_160_seed0 act_160_seed0 act_lstm_160_seed0
```

If `record_exp_videos.py` has already filmed a seed, add `--reuse D:/thesis_videos/scripted_exp` to
`video_side_by_side.py`. It then takes that panel from the existing video instead of re-running it. The hash is
still checked.

The reel. Plan it, read the plan, then render:

```
python tools/video_highlight_reel.py --index D:/thesis_videos/scripted_exp/index.csv --side-by-side D:/thesis_videos/side_by_side/side_by_side_index.csv --runs bc_160_seed0 act_160_seed0 act_lstm_160_seed0 --results runs/scripted_v3 --data-label "trained on 160 scripted demonstrations" --size 1920x1080 --plan --write-manifest D:/thesis_videos/reel_run_b.json
python tools/video_highlight_reel.py --manifest D:/thesis_videos/reel_run_b.json --out D:/thesis_videos/reel_run_b.mp4 --h264
```

`--size 1920x1080` is recommended whenever side-by-side clips are in the reel. A three-panel row is 1920 px wide,
and at 1280x720 its text gets small. The manifest is plain JSON: edit the captions, reorder or drop segments, set
`speed`, `start_s` or `end_s` on a clip, then render it with `--manifest`.

## 2. Scripted-trained, the D8 models (n40, training seed 0; two results folders)

D8 is still pending adviser approval (CLAUDE.md section 7). BC (b256) sits in `runs/scripted_v3`; act_orig and
act_lstm_full sit in `runs/scripted_v3_memory_design`:

```
python tools/pick_video_episodes.py --results runs/scripted_v3 runs/scripted_v3_memory_design --runs bc_b256_n40_seed0 act_orig_n40_seed0 act_lstm_full_n40_seed0 --out D:/thesis_videos/picks/d8_n40.json
python tools/video_side_by_side.py --picks D:/thesis_videos/picks/d8_n40.json --category contrast heldout_success --out-dir D:/thesis_videos/side_by_side_d8
python tools/record_exp_videos.py --runs bc_b256_n40_seed0
python tools/record_exp_videos.py --run-root runs/scripted_v3_memory_design --runs act_orig_n40_seed0 act_lstm_full_n40_seed0 --video-root D:/thesis_videos/memory_design_exp
python tools/video_highlight_reel.py --index D:/thesis_videos/scripted_exp/index.csv D:/thesis_videos/memory_design_exp/index.csv --side-by-side D:/thesis_videos/side_by_side_d8/side_by_side_index.csv --runs bc_b256_n40_seed0 act_orig_n40_seed0 act_lstm_full_n40_seed0 --results runs/scripted_v3 runs/scripted_v3_memory_design --size 1920x1080 --out D:/thesis_videos/reel_d8_n40.mp4 --h264
```

Use one `record_exp_videos.py` video root per results folder. It rebuilds `index.csv` from every run folder under
its video root and looks each one up in its OWN `--run-root`'s eval folder, so a run from the other results
folder would not be found. `--index` takes both index files, and `--runs` filters them.

## 3. Human-trained (teleoperated demonstrations)

Train and evaluate into your own results folder with the usual tools. Below it is `runs/teleop_v1`; use the
folder you actually trained into:

```
python tools/eval_scripted.py --run-root runs/teleop_v1 --runs bc_teleop150_seed0 act_teleop150_seed0 act_lstm_teleop150_seed0
```

Then the same four commands, with `--results runs/teleop_v1` (and `--run-root runs/teleop_v1` for
`record_exp_videos.py`). The run names here are examples. Name a model when its run name does not say it:
`--runs my_run=ACT-LSTM`. Labels come from `<model>_<size>_seed<k>` names (`bc`, `bc_b256`, `act`,
`act_lstm`, `act_orig`, `act_lstm_full`), then from the eval record's `kind`.

```
python tools/pick_video_episodes.py --results runs/teleop_v1 --runs bc_teleop150_seed0 act_teleop150_seed0 act_lstm_teleop150_seed0 --out D:/thesis_videos/picks/teleop.json
python tools/video_side_by_side.py --picks D:/thesis_videos/picks/teleop.json --category contrast heldout_success --out-dir D:/thesis_videos/side_by_side_teleop
python tools/record_exp_videos.py --run-root runs/teleop_v1 --runs bc_teleop150_seed0 act_teleop150_seed0 act_lstm_teleop150_seed0 --video-root D:/thesis_videos/teleop_exp
python tools/video_highlight_reel.py --index D:/thesis_videos/teleop_exp/index.csv --side-by-side D:/thesis_videos/side_by_side_teleop/side_by_side_index.csv --results runs/teleop_v1 --data-label "trained on 150 teleoperated demonstrations" --size 1920x1080 --out D:/thesis_videos/reel_teleop.mp4 --h264
```

`record_exp_videos.py` re-runs every episode under its `--lock-rule` (default v3). If the teleop evaluation used
another rule, pass that rule. Otherwise every video will fail its hash check, and the check says so.
`video_side_by_side.py` needs no flag: it reads the lock rule and every other deploy option from the eval record's header.

## What is in the files

- **Side-by-side panel.** The header shows the model label, the run's demo count and training seed, and the
  outcome: `SUCCESS`, or `FAIL: <bucket>`, where the bucket is deploy.py's `fail_reason`. Below that, per frame:
  seed, time (tick / 25 Hz), the offline phase label, and weld and base lock as the policy observes them at that
  tick. When an episode ends, its panel freezes darkened with `ENDED`. One clock drives every panel. The title bar
  shows the experiment, the seed, the box spawn, and the training region or the held-out patch. The footer shows
  how many re-runs matched their eval hash. A mismatched panel gets a red frame and `RE-RUN != EVAL RECORD`.
- **`<out-dir>/<exp>/seedNNNN/panels_<tag>.json`.** Per panel: the checkpoint, the deploy options, the stored and
  re-run outcome, both hashes, device and git commit, and whether the video frame count equals the episode's ticks.
- **`<out-dir>/side_by_side_index.csv`.** One row per video, rebuilt from every `panels_*.json`. The reel reads it.
- **`<out-dir>/_panels/<run>/<exp>/seedNNNN/`.** The re-run itself. A finished one is reused, so a restart is
  safe. To force a re-run, delete that folder.
- **`picks.json`.** The categories are `contrast`, `contrast_reverse` (the counterexamples are always picked too),
  `typical_failure` (per run, experiment and bucket, nearest the bucket median: `record_exp_videos`' rule),
  `heldout_spread`, `heldout_success` and `all_succeed`. Each pick carries its `why` and every run's outcome.
  The exact rules are in the tool's docstring.
- **Reel.** Clips are shown whole and sped up to fit `--target-s` (default 90 s; at least 1x, rounded up to 0.25x,
  at most `--max-speed` 4x). Every caption states the speed. Clips whose hash did not match are left out unless
  `--allow-mismatch` is given, and then they are captioned. `<reel>.json` records what was rendered.

## Tests

No simulation or rendering is needed. The tests use synthetic frames and a fake deploy. Where MuJoCo imports,
they also check the constants against `tools/episode_view.py`, the `--plan` command against
`eval_scripted.deploy`, and the typical-failure rule against `record_exp_videos`:

```
python tools/test_video_side_by_side.py
python tools/test_pick_video_episodes.py
python tools/test_video_highlight_reel.py
```
