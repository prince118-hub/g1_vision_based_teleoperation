# Live recording checklist — Windows laptop, real ZED 2i

What `tools/record_teleop_live.py` could NOT be verified on without the camera. It was
built and tested on a Linux CPU machine with a fake source only
(`tools/test_record_teleop_live.py`). Run these on the NVIDIA laptop (the one machine
results come from, CLAUDE.md §8 2026-09-28, O34), in order, from the repo root, in the
`g1_match` environment. Write every number into `NOTES.md` under the date you ran it.

Nothing here trains anything. Everything writes under `recordings/` (gitignored); back it
up off-machine as PLAN.md Phase 6 says.

---

## 0. The path fallback did not move anything

`G1_POLICY_PATH` / `G1_SCENE_PATH` now fall back to files in the repo instead of the
hardcoded `D:\Charles_Aninon\Thesis Project\...`. If the repo IS that folder, nothing
changed. Check it:

```powershell
python -c "import walk_test as W, run_integrated_combined as R; print(W.POLICY_PATH); print(W.SCENE_PATH); print(R.SCENE_PATH)"
echo $env:G1_POLICY_PATH $env:G1_SCENE_PATH
```

**Pass:** either both env vars are set (nothing changed for you), or the printed paths
are inside this checkout. If the repo is NOT `D:\Charles_Aninon\Thesis Project`, compare
the files byte for byte (the two copies must be the same files):

```powershell
certutil -hashfile "D:\Charles_Aninon\Thesis Project\unitree_rl_gym\deploy\pre_train\g1\motion.pt" SHA256
certutil -hashfile unitree_rl_gym\deploy\pre_train\g1\motion.pt SHA256
certutil -hashfile "D:\Charles_Aninon\Thesis Project\scene.xml" SHA256
certutil -hashfile scene.xml SHA256
```

## 1. The suites on this machine

```powershell
python -m pytest g1_data g1_model g1_teleop tools -q
```

**Pass:** everything passes. In particular on THIS machine
`g1_model/test_act.py::test_use_lstm_false_is_act_exactly_as_it_was_before_the_flag` must
**run and pass**, not skip: it skips only when torch is not the pinned build, and this
laptop is the pinned build (2.11.0+cu126). A skip here means the environment drifted.
`tools/test_record_teleop_live.py::test_split_main_is_the_old_main_step_for_step` needs
git and commit `473a7d7` in the clone; it must run, not skip.

## 2. Camera, tick rate and dropout (one short session, nothing kept)

Open the camera through the recorder into a scratch folder. Stand in frame at your normal
operating distance, arms visible, for ~30 s; do not walk the robot.

```powershell
python tools/record_teleop_live.py --source zed --operator charles --out-dir recordings/live_check --seed-list 0 --count 1 --note "checklist step 2"
```

ENTER starts, wait ~30 s, `e` stops and saves (it will be saved as FAILED in
`recordings/live_check/failed/` - nothing was picked up; that is expected). Then:

```powershell
python -c "import glob,json; from g1_data.recorder import load_episode; p=sorted(glob.glob('recordings/live_check/failed/*.npz'))[-1]; a,m=load_episode(p); print(p); print(json.dumps(m['timing'],indent=1)); print('dropout_rate', m['dropout_rate'], 'degraded', m['degraded_ticks'], 'of', m['n_ticks']); c=m['camera']; print(c['sdk_version'], c['serial_number'], c['camera_fps'], json.dumps(c['body_tracking']))"
```

| check | pass criterion | why |
|---|---|---|
| `timing.tick_rate_hz` | ≥ 24.0 Hz | the loop holds 25 Hz (D4); the 2026-09-15 fix measured 128% of real time |
| `timing.overrun_ticks` | ≤ 1% of `n_ticks` | an overrun = a tick interval > 60 ms |
| `timing.max_tick_interval_ms` | note it | a single stall (window drag, antivirus) shows here |
| `dropout_rate` | ≤ 0.02 with you standing in frame | O28: a degraded tick = arms stale > 0.4 s |
| `camera.serial_number`, `sdk_version`, `body_tracking` | real values (not -1 / "fake") and `body_tracking` READ BACK (no `error` key) | the provenance every episode carries |
| `camera.camera_fps.actual` | note it | O29: One-Euro is tuned for 30 Hz; record what actually ran |

Also, during that session, check by eye and write down:
- the HUD shows `REC teleop-live seed 0 tick n/900` and the lock indicator
  (`lock rule v3 needs: fwd ... lat ... head ...`, `LOCKED` once locked) in the MINIMAL view;
- pressing `b` while recording prints `b ignored: B-prime cannot change while recording`
  and B-PRIME stays ON; the episode's `d18.toggle_blocked_attempts_s` lists the press;
- step out of frame for ~2 s: `TRACKING LOST` appears, the arms hold, and the saved
  episode's `tracking_ok` is 0 on those ticks (`degraded_ticks` > 0).

If `tick_rate_hz` fails: run `python run_integrated_combined.py keyboard 0 --profile --seconds 30`
(the throughput measurement) and compare with the 128% recorded on 2026-09-15 before
blaming the recorder.

## 3. Invariance on a short REAL take

The episode in step 2 wrote its raw camera stream to `recordings/live_check/keypoints/*_kp.npz`.
Replay it through the same loop twice, recorder on and off, and compare every physics step:

```powershell
$kp = (Get-ChildItem recordings/live_check/keypoints/*_kp.npz | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
python tools/record_teleop_live.py --source fake --replay $kp --seed-list 0 --start-standoff 0.32 --headless --script "20:e" --invariance --out-dir recordings/live_check_fake
```

**Pass:** the last line is `OBSERVATION ONLY: CONFIRMED` (both runs report the same step
count and trajectory sha256). The real take is replayed on the simulated frame clock, so
this checks the recorder on REAL keypoints (noise, NaNs, no-body frames), not the camera
timing.

## 4. One full episode, saved and loaded

Do the whole task as a demonstration on the first training seed: walk in from the start,
wait for `LOCKED`, reach, `g`, lift, carry to the goal, lower, `g` to release, withdraw
the hands. Let the recorder end it (success ends it 2.5 s after the release) or press `e`.

```powershell
python tools/record_teleop_live.py --source zed --operator charles --out-dir recordings/live_check_full --count 5 --note "checklist step 4"
```

`--count` caps ATTEMPTS: a failed or discarded attempt offers the same seed again; once one
is accepted, press `q` at the idle screen.

Then load it the way training will, and replay it the way the scripted episodes were:

```powershell
python -c "import glob; from g1_data import dataset as DS; from g1_data.recorder import load_episode; ps=glob.glob('recordings/live_check_full/*.npz'); print(ps); eps=DS.scan('recordings/live_check_full'); DS.assert_uniform_lock_rule(eps); DS.assert_uniform_grip_label(eps); DS.assert_successful(eps); a,m=load_episode(ps[0]); print(m['end_reason'], m['outcome'], m['timing']['tick_rate_hz'], m['dropout_rate'])"
python tools/replay_check.py "recordings/live_check_full/ep_seed*.npz"
type recordings\live_check_full\ledger.jsonl
```

**Pass:**
- the episode is in `recordings/live_check_full/` (not `failed/`), the ledger's last row is
  `accept` for that seed, and `end_reason` is `success`;
- `DS.assert_successful` does not raise; `outcome.placement_error` ≤ 0.10 m;
- `replay_check`: `reset fingerprint matched 1/1` and `MANIPULATION reproduced 1/1`. The
  whole task is NOT expected to reproduce open loop (the walk; NOTES 2026-09-16);
- `spec_version` g1-spec-1.2.0, `lock_rule.version` v3 (the default since 2026-10-24), `grip_label` next_weld_v1,
  `contact_contract` equal to the scripted episodes', `source` teleop_live, `operator` set.

A failed attempt is in `failed/` with a `discard` ledger row; the reason is in
`outcome.attributed_to`. Write down how many attempts the first success took - that is the
first number on teleop success rate.

## 5. What to report back

Into `NOTES.md` (dated): steps 0-4 pass/fail; tick rate, overruns, max interval, dropout
rate, the camera's actual fps; the attempts needed in step 4 and its placement error; any
crash with its traceback. If step 4 passes, the collection can start:

```powershell
python tools/record_teleop_live.py --source zed --operator charles --count 25
```

(`recordings/demonstrations/`, train seeds from `data/eval/train_seeds.json`; CLAUDE.md §8
2026-09-11: collect 25 FIRST and train BC on them.)
