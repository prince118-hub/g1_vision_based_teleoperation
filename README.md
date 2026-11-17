# State-Based Imitation Learning for Bimanual Box Pickup on the Unitree G1 (MuJoCo)

BSCS thesis, Caraga State University, 2026. A simulated Unitree G1 picks up a box
with both arms, carries it and places it on a second platform. The repository holds:

- **Demonstration collection**: a scripted demonstrator (`g1_data/scripted_demo.py`)
  and live teleoperation (ZED 2i body tracking → IK → G1 arms, keyboard walking;
  `tools/record_teleop_live.py`). The legs are driven by Unitree's pre-trained
  walking policy (`unitree_rl_gym/deploy/pre_train/g1/motion.pt`).
- **Policies**: BC, ACT and ACT-LSTM on the 48-D privileged MuJoCo state
  (`g1_model/`, trained by `tools/train_bc.py`).
- **Evaluation**: closed-loop deployment (`tools/deploy.py`) on box positions inside the
  collection region (Exp 1) and in a held-out patch (Exp 2), and the final analysis
  (`tools/make_final_report.py`, described in `docs/ANALYSIS.md`).

Methods, decisions and measurements are in `CLAUDE.md` (authoritative for methods),
`PLAN.md` and `NOTES.md`; the proposal is `docs/Thesis_Proposal.pdf`.
Third-party parts and their licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Repository layout

```
scene.xml, g1.xml, assets/   MuJoCo scene, G1 model (MuJoCo Menagerie) and meshes
unitree_rl_gym/              vendored; the G1 walking policy (motion.pt)
g1_data/                     state/action spec (spec.py), scripted demonstrator, recorder, dataset
g1_teleop/                   retargeting, IK, weld grasp, base lock, ZED source
g1_model/                    BC, ACT, ACT-LSTM, training loop, deployment guard
tools/                       entry points (record, replay, train, deploy, evaluate, report)
data/eval/                   fixed seed lists: train, selection, Exp 1, Exp 2
data/processed/              train/val splits and normalization stats per dataset
recordings/, runs/           data and run outputs - NOT in git; fetched from the GitHub Release
```

## How to run

Tested on Windows 11 with Python 3.10.11. Commands below work in PowerShell and in
bash; run them from the repository root.

### 1. Environment

```bash
git clone https://github.com/prince118-hub/g1_vision_based_teleoperation.git
cd g1_vision_based_teleoperation
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
python -m pip install --upgrade pip
```

PyTorch first, then the rest (the pins in `requirements.txt` are the versions every
reported result was produced with):

```bash
# NVIDIA GPU (CUDA 12.6 build, as used for the results):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
# or CPU only (enough for everything below except long training runs):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Check that the scene and the walking policy load:

```bash
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('scene.xml'); print('scene ok,', m.nq, 'qpos')"
python -m pytest tools/test_deploy.py -q
```

The ZED SDK (`pyzed`) is only needed for live teleoperation recording, not for
anything in this section.

### 2. Fetch the data

Data is not in git. It is attached to the GitHub Release
[`data-v1`](https://github.com/prince118-hub/g1_vision_based_teleoperation/releases/tag/data-v1)
as zips; the script checks each zip's SHA-256 against `tools/release_manifest.json` and
unpacks every file into the folder it came from.

```bash
python tools/fetch_release_data.py --list
python tools/fetch_release_data.py --only scripted_v3_dataset analysis_scripted_v3
python tools/fetch_release_data.py --only checkpoints_memory_design      # 0.98 GB
python tools/fetch_release_data.py --only checkpoints_scripted_v3        # 1.34 GB, optional
```

| Asset | Size | Unpacks to | Needed for |
|---|---|---|---|
| `scripted_v3_dataset` | 56 MB | `recordings/scripted_v3/` (200 scripted demos + ledger) | (a), (b), (c) |
| `analysis_scripted_v3` | 5 MB | `runs/scripted_v3*/eval/`, `.../analysis_final/`, each final run's `metadata.json` | (c), (d) |
| `checkpoints_memory_design` | 979 MB | selected checkpoint of the 6 ACT / ACT-LSTM runs | (c) |
| `checkpoints_scripted_v3` | 1340 MB | selected checkpoint of the 16 BC, ACT-12, ACT-LSTM-12 runs | (c), other models |
| `human_demonstrations` | 31 MB | `recordings/demonstrations/` (teleoperated pilot, no raw keypoints) | optional |

Without network access to GitHub, download the zips by hand from the Release page and
run `python tools/fetch_release_data.py --from-dir <folder with the zips>`.

### 3. Commands

**(a) Replay a scripted demonstration.** Re-runs the recorded 25 Hz actions open loop in
MuJoCo and checks the task still succeeds (weld, lift, placement). `--negative` adds the
control: the same actions against a shifted box must fail.

```bash
python tools/replay_check.py recordings/scripted_v3/ep_seed0000.npz
python tools/replay_check.py recordings/scripted_v3/ep_seed0000.npz recordings/scripted_v3/ep_seed0001.npz --negative
```

To watch the scripted demonstrator itself generate a demo in the MuJoCo viewer (needs a
display; on macOS run it with `mjpython` instead of `python`):

```bash
python tools/record_episodes.py --count 1 --seeds data/eval/train_seeds.json --auto pass --viewer --realtime --out-dir recordings/scripted_try
```

**(b) Train a small run.** A 300-step BC run on the 40-demo split (the size the
thesis models use), a plumbing check rather than a useful policy:

```bash
python tools/train_bc.py train --model bc --chunk 1 --obs-window 12 --batch 256 --max-steps 300 --window-steps 100 --budget-reason "smoke test" --episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz --out-dir runs/try_bc
```

The full training commands of every reported run are in `tools/train_scripted_v3_n40.ps1`
(BC, ACT-12, ACT-LSTM-12) and `tools/train_memory_design_n40.ps1` (ACT, ACT-LSTM);
ACT runs take 200k steps on a GPU.

**(c) Evaluate a final checkpoint on a few box positions.** ACT-LSTM, training seed 0,
its selected checkpoint, on three Exp 1 positions and one Exp 2 (held-out patch)
position; about 10 s per episode. Then compare with the recorded result for the same
seeds:

```bash
python tools/deploy.py --run runs/scripted_v3_memory_design/act_lstm_full_n40_seed0 --ckpt ckpt_step0020000.pt --seed-list 252,255,240,115 --allow-lock-rule-mismatch --out runs/try_eval/act_lstm_full_seed0.jsonl
python tools/compare_to_final.py runs/try_eval/act_lstm_full_seed0.jsonl
```

- The selected checkpoint of every run is the `chosen_ckpt` column of
  `runs/scripted_v3/analysis_final/runs.csv` (also `runs/<root>/eval/<run>/selection.json`).
- Exp 1 seeds: `data/eval/exp1_seeds.json`; Exp 2 seeds: `data/eval/exp2_seeds.json`.
  `--seeds data/eval/exp1_seeds.json` runs all 100.
- `--allow-lock-rule-mismatch` is required and intended: the scripted_v3 demos were
  recorded under base-lock rule v2 and every experiment ran under v3 (`CLAUDE.md` §8,
  2026-10-24). It is recorded in the output header.
- Add `--viewer --realtime` to watch, `--video-dir <folder>` to save MP4s.
- Bit-identical trajectories are expected only on the machine the results came from;
  on other hardware a few episodes diverge numerically (`CLAUDE.md` O34), so compare
  outcomes, not hashes.

**(d) Regenerate the final report.** Read-only on the evaluation records; rebuilds the
tables, statistical tests and figures of `runs/scripted_v3/analysis_final/report.md`
into a new folder:

```bash
python tools/make_final_report.py --out runs/report_regen
```

`runs/report_regen/report.md` should match the shipped
`runs/scripted_v3/analysis_final/report.md`; `docs/ANALYSIS.md` explains every table.

## Rebuilding the Release data (maintainers)

```bash
python tools/package_release_data.py --dry-run   # list contents and sizes
python tools/package_release_data.py             # zips -> release_build/, updates tools/release_manifest.json
```

Commit the updated manifest, then upload the zips to the `data-v1` Release.
