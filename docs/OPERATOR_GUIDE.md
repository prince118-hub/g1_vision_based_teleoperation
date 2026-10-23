# Operator guide — practise, record, accept or discard

For whoever drives the robot for the teleoperated demonstrations. The camera-side
acceptance checks (does the ZED path work at all on this laptop) are in
`docs/LIVE_RECORDING_CHECKLIST.md`; this page is how to use the tools once it does.

Everything an operator sees — scene, start pose (0.60, 0.00), walking keys and speeds,
lock rule (default **v3**), lock indicator, floor guide, seeded box spawn, base-lock
behaviour — comes from one place, `g1_teleop/operator_config.py`, for practice and for
recording alike. Practice is the real thing minus the camera.

## 1. Practise the walk-in with `walk_test.py`

```
python walk_test.py                       # first training seed, floor guide on
python walk_test.py --seed 12             # a particular TRAINING seed
python walk_test.py --guide off           # practise without the guide
```

Same window, same loop, same keys as the recorder (`run_integrated_combined.run_loop`).
There is no camera, so **the arms are not driven** — the panel says
`PRACTICE - walk in, lock`. The goal is the walk-in and the base lock.

| key | does |
|---|---|
| arrows (or `i j k l`) | walk forward/back, turn — **hold** to move, release to stop |
| `n` / `m` | strafe left / right (hold) |
| `p` | precision mode (half speed) for the final approach |
| space | stop and hold here |
| `u` | abort the lock (re-lock waits until you walk/turn out of the band) |
| `t` | new try on the **next** training seed |
| `y` | new try on the **same** seed |
| `o` | overlay minimal / full |
| `q` | quit (prints the practice summary) |

`b` is blocked: B-prime is fixed ON for collection (D18). Only seeds in
`data/eval/train_seeds.json` are accepted; exp1 / exp2 / selection seeds are refused.

**Per try** the panel and the terminal show: time to lock, lateral error (m) and heading
error (deg) at the lock, and the **overshoot** — how far the base came closer to the box
than the band's near edge before it locked (0.000 is ideal). The summary at quit gives the
median and worst of each over the locked tries. A fall ends the try and restarts the seed.

Good practice: arrive head-on, slow down (`p`) when the band turns yellow, stop inside it
and let the lock engage (it needs the band held for 5 ticks = 0.2 s). At full speed you
cross the 0.14 m band in about 0.23 s — the debounce barely fits, so walking through it is
how overshoot happens.

## 2. The floor guide — what the colours mean

A marking on the floor in front of the target (the box; the goal platform while the box is
welded), drawn from the **active** lock rule — never a copied number. Rule v3:
`fwd 0.26–0.40 m`, `|lat| <= 0.05 m`, `|heading| <= 5 deg`.

| element | meaning |
|---|---|
| **band** (filled rectangle) | where the base (pelvis) must stand. **red** = no condition met, **yellow** = one or two met, **green** = all three met — the lock engages after 0.2 s |
| near / far edges | green when the forward distance is in the band, else red |
| side edges | green when the lateral offset is within the limit, else red |
| two rays from the target | the heading wedge: green when the robot faces the target within 5 deg |
| arrow at the robot | the robot's facing; same colour as the rays |
| solid green band, **LOCKED** label and banner | the pelvis weld is engaged: legs held, walking keys do nothing |

The band is measured in the robot's **current heading**, so it swings around the target as
you turn. That is how the rule works, not a drawing artefact: square up first, then step in.
Heading and lateral are linked (heading = atan(lat / fwd)), so at the near edge the 5 deg
heading limit is tighter than the 0.05 m lateral limit — a yellow band with red rays means
"turn", not "step sideways".

Nothing is drawn once the task is done (box at the goal and released: the rule cannot lock
there). The guide is purely visual: decor geoms in the render scene only — no mass, no
collision, not in the recorded state (`tools/test_operator_tools.py` proves guide on vs off
gives identical trajectories). `--guide off` turns it off in all three tools.

## 3. Record

```
python tools/record_teleop_live.py --source zed --operator charles --count 10
```

Per seed: the **idle screen** (physics paused, camera live, the floor guide on the reset
scene) — `ENTER` start, `n` skip the seed, `q` quit. While recording: every teleop key,
plus `e` stop and save, `x` discard (nothing written), `q` quit (saves). `b` is blocked and
logged. The episode also ends by itself on a fall, on success held for 2.5 s after the
release, or at 900 ticks.

## 4. Accept or discard — the rules

After every episode the **verdict screen** shows (and the terminal prints):

- the four success criteria (`g1_data/success.py`): grasp, lift, transport, place — each
  PASS/FAIL with its number;
- no fall;
- the **failing phase** (GRASP / LIFT / TRANSPORT / PLACE / WALK_IN / FELL);
- the **placement error** (box to goal centre, limit 0.10 m), tilt, resting, released;
- **tracking dropout**: % of ticks whose arm command was not backed by a tracked frame;
- **tick overruns**: ticks that ran more than 1.5 ticks late (camera sessions only);
- a **RECOMMENDATION**, accept or discard, with every reason against accepting.

Then press **`a` = accept** or **`d` = discard**. Keys pressed during the episode are
flushed and keys are ignored for the first 0.5 s, so a camera-rotate `a`/`d` cannot answer
for you.

| rule | |
|---|---|
| **An episode that FAILS the success rule can never be accepted.** | `a` is refused; only `d` works. It is kept in `failed/` for audit; the seed is offered again. |
| A success is recommended **discard** when tracking dropout > **5 %** of ticks or overruns > **2 %** of ticks. | Soft: you may still accept. Both limits are PROVISIONAL until the camera session (checklist step 2) measures the real population. |
| A success you discard | is kept in `discarded/` (never the training folder); the seed is offered again. |
| `x` during the episode | writes nothing; the seed is offered again. |
| Ctrl+C | saves the episode but never accepts it. |

Accepted episodes go to the out-dir's top level — the only folder the dataset builder reads.
The ledger (`ledger.jsonl` in the out-dir) records, for every episode, the operator's
choice, who decided (`operator`, or `auto:pass` for headless dry runs) and the
recommendation with its reasons. `--auto pass` (take the recommendation) is for the fake
source only; it is refused with `--source zed`.

Discard anything you would not want a policy to copy even if it passed: a bump of the
box before the grasp, a hesitant re-grasp, a long stand still — the criteria cannot see
intent.

## 5. Laptop checks that need a display (not covered by the tests)

1. `python walk_test.py`: the window opens, arrows walk, the band changes red → yellow →
   green on approach, `LOCKED` appears (label in the scene and banner top-right), `t`/`y`
   start new tries and the per-try line updates.
2. The guide's colours and the `LOCKED` label render as described (the tests check the
   geometry and the scene, not pixels — there is no OpenGL on the test machine).
3. `record_teleop_live.py --source fake ...` (windowed): the idle screen shows the guide;
   after `e` the verdict screen appears, `a` on a failed episode shows the refusal, `d`
   moves on.
4. With the ZED: the same verdict screen after a real episode, with tick overruns filled in.
