# Recording protocol — one collection session, and the 25 → check → +25 plan

The procedure for recording teleoperated demonstrations with `tools/record_teleop_live.py`.
Companion documents:

- `docs/LIVE_RECORDING_CHECKLIST.md` — the one-time camera acceptance checks. **Its steps 2–5
  must pass before the first real session.** As of 2026-10-24 they are BLOCKED: no ZED 2i
  available (CLAUDE.md §2).
- `docs/OPERATOR_GUIDE.md` — keys, the floor guide, the verdict screen. It is on branch
  `operator-tools` (PR #3, open as of 2026-10-24), not on main; see §7 for what changes
  without it.
- `tools/backup_recordings.py` (§10) and `tools/pilot_check.py` (§11).

Numbers marked **(starting value)** are not measured; replace them with what session 1
measures and record the change in `NOTES.md`.

---

## 1. Before the first session (once)

- [ ] `docs/LIVE_RECORDING_CHECKLIST.md` steps 0–5 pass, numbers written into `NOTES.md`.
- [ ] Operator roles decided (§3) and the `--operator` naming fixed.
- [ ] PR #3 (operator tools: practice mode, floor guide, accept/discard screen) merged, or
      the manual discard procedure in §7 understood.
- [ ] Two backup destinations exist and are tested (§10): one on another disk of the laptop
      (e.g. `D:\thesis_backup\demonstrations`) and one OFF the laptop (USB drive or a
      cloud-synced folder). First run with `--create`.
- [ ] A restore drill done once (§10) — PLAN.md Phase 6 asks for a backup "verified restorable".

## 2. Session setup checklist

**Machine** — results come from ONE machine, the NVIDIA laptop (CLAUDE.md O34, Option B).

- [ ] NVIDIA laptop, `g1_match` environment, on mains power, sleep and screen-off disabled.
- [ ] Nothing else heavy running: no training, no evaluation (CLAUDE.md §2: one heavy process
      at a time). Close browsers / sync clients that spike CPU; postpone antivirus scans.
- [ ] `git status` clean on the commit you mean to record with (every episode stores
      `git_commit` and `git_dirty`; a dirty tree makes the commit meaningless). Write the
      short hash in the session notes.

**Camera**

- [ ] ZED 2i on a tripod, plugged DIRECTLY into a USB 3 port (no hub), lens clean.
- [ ] Same tripod position, height and aim every session: tape marks on the floor for the
      tripod feet and for where the operator stands.
- [ ] Distance: the operator's whole upper body — head to hips, both hands at full forward
      and sideways reach — stays inside the image at all times. **(starting value: the
      distance used in checklist step 2; record it in `--note` every session.)** Depth is
      the noisiest ZED axis and also the reach axis (D1), so do not change it between sessions.
- [ ] Only the body operator in view. Nobody walks behind or beside them while recording:
      the tracker picks one body per frame and a second person can take over the arms.

**Lighting**

- [ ] Even light from the front or above; no window or lamp BEHIND the operator (backlight).
- [ ] No direct sunlight on the operator or the background; the same lights on every
      session (note it — daylight changes through the day).

**Clothing (operator)**

- [ ] Fitted clothing that contrasts with the background; no loose or long sleeves covering
      the wrists, no gloves, no bags or lanyards; nothing reflective.
- [ ] Hands, wrists and elbows clearly visible — the IK targets are elbow AND wrist (D3).

**Software**

- [ ] Start the recorder and look at the idle screen: camera live, the reset scene for the
      next seed, `B-PRIME ON`.
- [ ] Session note decided, e.g. `--note "2.5 m, ceiling lights, blue long-sleeve, keys: ana"`.

## 3. Operator roles — **TO DECIDE**

> **TO DECIDE (Charles, before session 1): one person or two?**
>
> - **One person** — the body operator also presses the walking keys, `g`, `e`/`x`. Simple,
>   one demonstrator. But the arms are tracked continuously: every reach to the keyboard is
>   recorded as arm motion (walk-in, and around the `g` press next to the box).
> - **Two people** — a **body operator** (in frame; arms only) and a **keyboard driver**
>   (out of frame; walking keys, `p`, `g`, `e`/`x`, `ENTER`/`n`), with fixed spoken calls
>   ("locked", "grasp", "lift", "release"). Cleaner arm data; the demonstration becomes a
>   two-person coordination, which is a disclosure for the write-up (O25: more than one
>   demonstrator).
>
> Whichever is chosen: keep it fixed for the whole dataset, and fix the `--operator` string
> (e.g. `charles` or `charles+ana` = body+keys). It is stored in every episode.

## 4. Warm-up (every session, ~10 min, nothing kept)

1. Walk-in practice: `python walk_test.py` (operator guide §1, PR #3) — 3–5 locked tries.
2. Two or three full episodes into a scratch folder that is never trained on:
   ```powershell
   python tools/record_teleop_live.py --source zed --operator charles --out-dir recordings/warmup --count 3 --note "warm-up"
   ```
   Check the HUD tick counter runs and `TRACKING LOST` does not flash while you stand still.

## 5. Session length and breaks

- Each attempt is capped at 900 ticks (36 s) plus reset and verdict; budget **~1–1.5 min per
  attempt (starting value)**. Holding the arms up is tiring and fatigue shows as tremor.
- **Blocks of 10 attempts, then a 5-minute break; at most 3 blocks (~45–60 min) per session
  (starting values).** Stop early when two attempts in a row fail for a reason that is the
  operator's (overshoot, re-grasp, dropped box) rather than the seed's.
- Run the backup (§10) at the end of EVERY session, before closing the laptop.

## 6. Seed order

- Always the pre-partitioned stream, in order: the default `--seeds data/eval/train_seeds.json`.
  The ledger resumes at the first seed with no ACCEPT. **Never `--seed-list` for real
  collection** — hand-picking seeds biases coverage of the spawn region.
- The recorder refuses exp1 / exp2 / selection seeds and skips held-out spawns itself.
- Real episodes go to `recordings/demonstrations` (the default for `--source zed`) and nowhere
  else: that folder IS the dataset, and its `ledger.jsonl` is the record of what was consumed.
- Session command (attempts this session = `--count`):
  ```powershell
  python tools/record_teleop_live.py --source zed --operator charles --count 10 --note "<session note>"
  ```

## 7. Accept / reject rules

What the code enforces (main):

- A FAILED episode (`success.evaluate`: grasp, lift, transport, place; or a fall) goes to
  `failed/` with a `discard` ledger row; the same seed is offered again. It can never be trained on.
- `x` during an episode discards it: nothing is written, the seed is offered again.
- Ctrl+C / `q` save what was recorded; it is accepted only if it succeeded.

With PR #3 merged, every episode ends on a verdict screen and the operator presses `a`/`d`;
a successful episode you discard goes to `discarded/`. **Without PR #3, main accepts every
success automatically** — to reject one afterwards, move the file AND record it:

```powershell
move recordings\demonstrations\ep_seedNNNN_<stamp>.npz recordings\demonstrations\discarded\
python -c "from g1_data.ledger import EpisodeLedger; EpisodeLedger('recordings/demonstrations/ledger.jsonl').append('discard', seed=NNNN, reason='operator: <why>', saved='recordings/demonstrations/discarded/ep_seedNNNN_<stamp>.npz')"
```

(The ledger's LAST row per seed decides; without the row the dataset tools report the seed
as accepted but missing from disk.)

Discard a success — the criteria cannot see intent — when any of these happened:

- the box was pushed or bumped before the grasp; a hesitant or repeated grasp;
- a long stand-still (> ~3 s, starting value) in the middle of the task;
- the arms jumped (tracking glitch) or `TRACKING LOST` appeared during reach, grasp or place;
- a wrong key (a walking key while locked, `b`, an accidental `g`);
- someone entered the camera view, or the camera / tripod was touched;
- tracking dropout > 5 % of ticks or tick overruns > 2 % of ticks (operator guide; PROVISIONAL).

When in doubt, discard: a seed is cheap, a bad demonstration is copied by every policy.

## 8. Retry policy

- A failed or discarded seed is offered again at once. **At most 3 attempts per seed per
  session (starting value)**; after the third, press `n` (skip) at the idle screen and write
  the seed and the reasons in the notes.
- A skipped seed stays pending in the ledger and is offered FIRST in the next session. If it
  fails 3 times in two sessions, raise it with Charles before skipping it for good — a seed
  the operator cannot do is a coverage hole, and it must be reported, not hidden.
- Never re-record an accepted seed (the ledger will not offer it), and never delete anything
  from `failed/` or `discarded/` — they are the reject-rate evidence.

## 9. Notes

One notes file per session, written during the session, **never edited after it is backed
up** (the backup refuses to overwrite a changed file; add a new file instead):
`recordings/demonstrations/notes/session_<YYYYMMDD>_<n>.md`

```
date / start-end time:
operators (roles):                       commit (git rev-parse --short HEAD):
camera distance / lighting / clothing:   (same text as --note)
warm-up: done / issues
per attempt: seed, attempt #, outcome (accept / failed <phase> / discard <why> / skip), remarks
breaks taken:
problems (crashes with traceback, tracking, timing, anything unusual):
backup: BACKUP COMPLETE to <dest 1>, <dest 2>  (yes / no)
```

After the session, a dated `NOTES.md` entry with the numbers: attempts, accepted, failed by
phase, discarded, median tick rate and dropout (from `tools/pilot_check.py` or the episodes'
metadata), and anything that changed in the setup.

## 10. After every session: back up

```powershell
python tools/backup_recordings.py recordings/demonstrations --dest D:\thesis_backup\demonstrations --dest E:\thesis_backup\demonstrations
```

- Copies episodes, `failed/`, `discarded/`, `keypoints/`, the ledger and the notes; verifies
  every file by sha256 on every destination (also the ones already there); never overwrites a
  different file; writes `_backup_manifests/backup_<stamp>.json` (date, counts) on each
  destination; ends with what is NOT yet backed up.
- **Done = the last line says `BACKUP COMPLETE`.** `INCOMPLETE` lists each file and which
  destination lacks it: `unavailable` (drive not plugged in / not synced — plug it in and
  re-run), `CONFLICT` (a different file is already there — stop and investigate, never
  delete either copy), `ERROR` (disk problem or a file still being written).
- Check without copying: add `--status`. First backup to a new destination: add `--create`.
- **Restore drill** (once, and after any change of destination): copy a backup back into a
  scratch folder and run the checks on it:
  ```powershell
  python tools/backup_recordings.py E:\thesis_backup\demonstrations --dest recordings\restore_drill --create
  python tools/pilot_check.py --episodes recordings\restore_drill --expect 0 --skip-train
  ```
  (`INCOMPLETE` = verdict with the training skipped; any `FAIL` means the backup is not usable.)

## 11. The pilot: 25 → check → +25

CLAUDE.md §8 2026-09-11 and PLAN.md Phase 6 task 1: collect 25 FIRST.

1. **Record 25 accepted episodes** (about 2–3 sessions), backing up after each.
2. **Run the pilot check** on the NVIDIA laptop (nothing else running):
   ```powershell
   python tools/pilot_check.py
   ```
   It checks `recordings/demonstrations`: 25 accepted, one per seed, ledger = disk; every
   dataset guard (spec, contact contract, lock rule, grip label, success, held-out, real
   source, training seeds); open-loop replay of every episode; split + normalization into a
   scratch folder (`runs/pilot_check/<stamp>/`, never `data/processed`); a 300-step BC smoke
   training; tick rate and dropout. It ends `PILOT CHECK: PASS | FAIL | INCOMPLETE` with the
   reasons, and writes `pilot_report.txt` / `.json` in the scratch folder.
3. **On FAIL — stop collecting** until the cause is understood:

   | failing check | what it means | do |
   |---|---|---|
   | inventory | fewer than 25, a seed twice, ledger ≠ disk | fix the folder/ledger (§7 procedure); never delete |
   | a dataset guard | the data is not one dataset (rule, contract, spec, label) or contains a failure / eval seed | find which session caused it (ledger `session` rows); move the bad episodes out |
   | replay | the recorder did not log what drove the robot, or the reset is not deterministic | **stop**; this invalidates the data; investigate before any more sessions |
   | split + norm / BC smoke | the data does not build or does not train | read the logs in the scratch folder |
   | timing + dropout | the loop did not hold 25 Hz, or tracking dropped systematically | fix the machine (§2) or camera/lighting/clothing, then re-check with a short session |

4. **On PASS**, two things before the next 25, both for Charles:
   - **The BC-at-25 run** (PLAN.md Phase 6 task 1: "if BC saturates at 25 the task does not
     discriminate"). The pilot check's smoke test is NOT that experiment — it only shows the
     data trains. The real run (budget, evaluation seeds) is **TO DECIDE**.
   - The PLAN.md Phase 5 items the check does not cover: the overfit-10 gate on piloted data
     (O31: every Phase 4 gate is provisional until then), deploying that policy, whether the
     ACT latent stays active on piloted data (O32), and the D12 question (§8 2026-09-15).
5. **+25**: record the next 25 with the same setup, roles and notes, then re-run
   `python tools/pilot_check.py --expect 50`. Then continue towards 150, re-checking every 25.
