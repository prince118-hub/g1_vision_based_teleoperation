# CLAUDE.md

## 1. Thesis, objectives, constraints, authority model

**Title:** A State-Based Imitation Learning Pipeline for Bimanual Box Pickup on the Unitree G1
Humanoid Robot Using ACT-LSTM in MuJoCo Simulation. BSCS, Caraga State University, April 2026.

**Objectives (FIXED — never negotiate these away):** (1) teleop demonstration-collection system:
ZED 2i captures human bimanual motion, retargeted to G1 joint commands in MuJoCo; (2) train a
hybrid ACT-LSTM policy plus a standard BC baseline for bimanual box pickup/placement; (3)
evaluate both on box positions seen during collection (phase-level success rate); (4) evaluate
both on positions NOT seen during collection (spatial generalization).

**Constraints:** simulation-only, no sim-to-real, no hardware. Policy consumes privileged MuJoCo
state, not camera input, at deployment. Fixed box size and orientation. G1 dexterous fingers not
used. RL and locomotion-learning-from-demonstration out of scope.

**Authority model — read before changing anything:** `docs/Thesis_Proposal.pdf` is authoritative
for OBJECTIVES, scope, and evaluation intent. **This file is authoritative for current METHODS
and design decisions.** The proposal's methods are a direction, not a spec; several deviations
are deliberate (§7). When code disagrees with the proposal's method, do NOT "fix" the code toward
the proposal — record the divergence and ask Charles. When something drifts from an OBJECTIVE
rather than a method, flag it loudly. That distinction is the one that matters.

## 2. CURRENT BLOCKER

Implementing the actuated palm-pad gripper (decided 2026-08-23, see §8). The G1's
`left_rubber_hand` / `right_rubber_hand` geoms are `class="visual"`
(`contype=0, conaffinity=0`) — the hands have **no collision geometry**, so proposal §3.3.6's
friction-press grasp cannot function and no gripper joints/actuators/sites exist. The fix is one
slide joint + one `<position>` actuator + one high-friction collision pad per hand, pressing the
opposite faces of the box. Until a scripted run holds the 0.4 kg box through a full 1.5 m walking
transport 10/10 times, nothing downstream (recorder, dataset, spec freeze, policies, evaluation)
can be trusted. See PLAN.md Phase 1.

## 3. Full pipeline (end to end)

```
ZED body tracking (30 Hz, BODY_38)
  -> One-Euro -> geometric scaling -> DLS IK -> arm actuators; waist pinned   [EXISTS]
  -> KeyboardCommand -> unitree_rl_gym policy -> leg torques; physics @500 Hz [EXISTS]
  -> palm-pad grasp mechanism                                                [MISSING]
  -> recorder (state 47-D, action 22-D, phase label) @25 Hz; episode reset   [MISSING]
  -> success detection + quality filter -> dataset (split, norm, chunks)     [MISSING]
  -> 3 policies: BC | plain ACT | ACT-LSTM (one class, flag)                 [MISSING]
  -> autonomous deployment (policy replaces teleop, same action interface)   [MISSING]
  -> Exp 1 in-distribution + Exp 2 held-out -> generalization gap            [MISSING]
```

## 4. Current state (verified against files)

- **Kinematic path** — `teleop.py` writes `data.qpos` directly on a *kinematic twin* `G1Robot`
  and calls `mj_forward`. `run_teleop.py` / `run_teleop_combined.py` never `mj_step`: gravity
  never acts, the box never falls. Preview tools only.
- **Actuated path** — `run_integrated_combined.py` steps physics on a *separate* model and copies
  17 twin qpos into `data.ctrl[12:29]`. The only full-stack loop. `walk_test.py` is the same
  physics path with arms pinned and no ZED.
- **IK** — DLS with nullspace seed-pull, driving **4 joints per arm** (shoulder pitch/roll/yaw,
  elbow); wrists pinned at `WRIST_NATURAL`. 6-D task: elbow AND wrist positions targeted.
- **Retargeting** — geometric scaling: unit segment directions from ZED, rescaled to G1 link
  lengths measured live, anchored at the live G1 shoulder. **Filtering** — One-Euro on keypoints,
  `arm_alpha` low-pass on joints, `depth_alpha` low-pass on X only, per-arm stillness lock,
  5-frame coast on dropout.
- **Locomotion** — `torch.jit` load of `pre_train/g1/motion.pt`; 47-D obs, 12 actions, decimation
  10 → 50 Hz, `GAIT_PERIOD=0.8`. Legs `<motor>` + hand-rolled PD (gains from `g1.yaml`);
  waist+arms `<position>` kp=500. Position-hold and heading-hold outer loops keep the robot
  stationary at zero command. Three command strategies exist; `KeyboardCommand` chosen (D6).
- **Scene** — platforms top at 0.75 m: pickup (1.5, 0), goal (1.5, −1.5); non-colliding
  `goal_marker` cylinder r=0.06; `box1` 0.18 m cube, 0.4 kg, friction 1.5, solref 0.004.
  nq=43, nv=41, nu=29, one keyframe `stand`.
- **Collision** — every leg/torso/arm/wrist link collides. **Both rubber hands do not.** No
  grippers exist: no joints, actuators, sites, or finger bodies.
- **Absent entirely:** demonstration logging, physics-model episode reset, phase labelling,
  success detection, dataset, training, policy, evaluation code.

## 5. Planned components (none built)

New learning code goes in the git repo root `g1_vision_teleop/`, as siblings to `g1_teleop/`, so
the recorder can import the existing teleop stack unchanged:

```
g1_data/   spec.py  episode.py  recorder.py  reset.py  phases.py  success.py  dataset.py
g1_policy/ base.py  bc.py  act.py (ACT + ACT-LSTM behind use_lstm flag)  ensemble.py
g1_train/  train.py  config.py  normalize.py
g1_eval/   deploy.py  experiments.py  metrics.py
configs/   task.yaml  record.yaml  train_{bc,act,act_lstm}.yaml
data/      raw/<episode_id>.npz   processed/
```

Hard requirement: **ACT and ACT-LSTM are ONE model class behind a flag.** Two implementations
make the gap uninterpretable — it would no longer isolate the LSTM.

## 6. State and action vectors

**State — 47-D (proposal Table 3.3):**
| Group | Dim | Source | Status |
|---|---|---|---|
| box pos / quat | 3 + 4 | `data.qpos[box_qadr:+7]` | resolved |
| robot base pos / quat | 3 + 4 | `data.qpos[0:7]` | resolved |
| L/R EE pos / quat | (3+4)×2 | `site_xpos`/`site_xmat` of the two pad sites | pending P1 |
| L/R gripper state | 1 + 1 | pad slide-joint qpos, normalized 0–1 | pending P1 |
| L/R arm joint angles | 7 + 7 | `data.qpos` of the 14 arm joints | resolved |
| waist joint angles | 3 | yaw/roll/pitch — **constant 0**, waist is pinned (D10) | resolved |

**Action — 22-D (proposal Table 3.4):** left arm 7 + right arm 7 + waist 3 + left gripper 1 +
right gripper 1 + walking velocity 3.
- The 17 joint-target dims map to `data.ctrl[12:29]` on the physics model. **Log `data.ctrl`,
  not twin qpos** — they are equal at write time but diverge under load.
- The 2 gripper dims are the pad actuator command per hand, 0 = retracted, 1 = pressing.
- The 3 velocity dims go to the locomotion policy, body frame (vx, vy, wz), clipped to
  (±0.80, ±0.40, ±0.80). Source is `KeyboardCommand` (D6).
- Only 8 of 14 arm dims are varied by teleop (wrists pinned) and the 3 waist dims are constant
  (D10). Constant dims are harmless for training but must be disclosed in the thesis.

## 7. Deviations from the proposal

| # | Proposal says | Code does | Why | Chapter to update |
|---|---|---|---|---|
| D1 | §3.3.3 pure rotation matrix camera→robot | rotation + `DEPTH_SCALE=0.6` on the depth axis | depth is the noisiest ZED axis and also the reach axis | 3.3.3 |
| D2 | §3.3.5 IK over all arm joints | IK drives 4 joints/arm; wrists pinned | wrist-twist artifact made grasp pose unusable | 3.3.5 |
| D3 | §3.3.5 IK targets end-effector position | IK targets elbow AND wrist (6-D task) | elbow-free IK produced mirrored/folded poses | 3.3.5 |
| D4 | §3.3.1 single 25 Hz control rate | recording + policy at 25 Hz, locomotion inner loop at 50 Hz | the two rates are deliberately decoupled; 50 Hz is the pre-trained policy's native rate and is not free to change | 3.3.1, 3.3.5 |
| D5 | §3.3.1 robot starts with all joints at zero | legs start at `DEFAULT_ANGLES` crouch | the walking policy cannot recover from straight legs — verified | 3.3.1, 3.8.1 |
| D6 | §3.3.4 pelvis-velocity walking trigger | **abandoned.** `KeyboardCommand` is the collection interface | §3.3.4 and §3.3.8 "walking in space" are mutually incompatible (TR1). Pelvis method is written up as a negative result | 3.3.4, 3.3.8 |
| D7 | §3.3.6 grippers press inward via arm IK error | actuated palm pads press; arm holds a clean pose | hands have no collision geometry, and IK-error pressing corrupts the arm joint dims it is recorded into | 3.3.6 |
| D10 | §3.4 waist joint angles as live DOF | waist pinned at 0; 3 state + 3 action dims constant | torso-yaw sign never verified, and waist motion perturbs a locomotion policy trained without it; turning is done by `wz` instead | 3.4 |
| D8 | two policies (ACT-LSTM, BC) | three (BC, plain ACT, ACT-LSTM) | isolates the LSTM's contribution rather than confounding it with chunking | 3.6, 3.7, 3.8.4, and RQ2 |
| D9 | §3.9 Ubuntu 22.04 + ROS2 Jazzy | Windows 11, no ROS | ROS adds no value for a single-process sim pipeline | 3.9 |

D6 and D7 touch **objectives**, not just methods, and both now have decided resolutions.
Everything else is a method change and is defensible as-is.

## 8. Decision log

- 2026-08-23 — CLAUDE.md and PLAN.md created as the cross-session memory for this repo.
- 2026-08-23 — Learning code lives in `g1_vision_teleop/` (the actual git repo) as siblings to
  `g1_teleop`, not the unversioned outer folder. Reason: importability + version control.
- 2026-08-23 — Recorder logs `data.ctrl[12:29]` from the physics model as the action, not twin
  `qpos`. Reason: under box load the two diverge; the policy must learn what was commanded.
- 2026-08-23 — Plain ACT added as a third condition (D8). Reason: without it, an ACT-LSTM-vs-BC
  gap cannot be attributed to the LSTM rather than to action chunking.
- 2026-08-23 (Q1) — Gripper = one slide joint + one `<position>` actuator + one high-friction
  collision pad per hand, pressing opposite box faces. Reason: gives the proposal's 0–1 command
  real physical meaning, and decouples press force from arm IK, so the arm holds a clean pose
  instead of sitting in permanent IK error that would corrupt the 14 arm dims it is recorded
  into. Chosen over a weld because a contact-and-friction grasp is defensible at panel.
- 2026-08-23 (Q2) — Pelvis-velocity trigger abandoned; `KeyboardCommand` is the demonstration
  interface. Reason: TR1. `locomotion_input.py` is **kept** — `PelvisVelocity` and its
  diagnostics are the evidence for the §3.3.4 negative result. Do not delete it.
- 2026-08-23 (Q3) — Record and run the policy at **25 Hz**; locomotion inner loop stays at 50 Hz.
  Reason: keeps the proposal's stated rate (one fewer deviation), halves sequence length, and the
  two loops have no reason to be locked together.
- 2026-08-23 (Q4) — `d_place = 0.10 m` XY from box centre to goal centre, **and** the box resting
  on the goal platform and roughly upright. Reason: 0.06 m (the marker radius) demands
  near-perfect centring and would floor place-success for all three policies, destroying
  discrimination. The resting clause stops a box at the right XY but on the *floor* from scoring.
- 2026-08-23 (Q5) — Waist pinned at 0 (D10). Reason: fewer collection failure modes; turning is
  handled by `wz`. Dims kept as constants so the vectors still match proposal Tables 3.3/3.4.
- 2026-08-23 (Q6) — Provisional: enlarge the pickup platform, split along **x** (distance from
  robot), near half = training, far half = held-out. Reason: approach distance is the axis that
  stresses the policy; a left/right split is weaker because the robot can just turn. Revisitable,
  but must close before Phase 2 does.
- (earlier, from code) — `DEPTH_SCALE` 0.3 → 0.6 once One-Euro handled noise (see TR7); IK gained
  a nullspace seed-pull to stop elbow-in/elbow-out flipping; tracking gating removed from the
  live path leaving only a NaN guard (TR6).

## 9. Tried and rejected — NEVER retry these

- **TR1. Pelvis-velocity locomotion trigger as specified in §3.3.4.** Marching in place
  oscillates the pelvis about a fixed point, so *net* velocity over any window longer than one
  step is ~zero. The pelvis is also often out of frame in an upper-body setup, and leaning to
  reach translates it forward indistinguishably from intent to walk. Diagnostics live in
  `locomotion_input.py`. §3.3.4 and §3.3.8 cannot both stand. Formally abandoned 2026-08-23.
- **TR2. Freezing leg targets to stop the in-place march.** The locomotion policy *is* the
  balance controller — the gait continuously repositions the support polygon. Freezing removes
  balance entirely; the robot topples within seconds.
- **TR3. Freezing the gait-phase input while still querying the policy.** Preserves balance but
  puts the network out of distribution: unnatural stance, unstable walk/stop transitions.
- **TR4. Slowing the gait cadence** (constant slower period, and an idle/moving blend). Both
  destabilised the gait, and measured leg cadence did not follow the commanded clock. Stepping
  rate is fixed inside the network; `GAIT_PERIOD=0.8` is not a tunable.
- **TR5. Starting legs at the teleop keyframe's straight-leg pose.** Out of distribution; the
  policy cannot recover. Must start at `DEFAULT_ANGLES`.
- **TR6. Confidence / yaw / segment-length gating on tracking frames.** Froze the robot far more
  often than it caught genuinely bad frames. Only the NaN guard survives.
- **TR7. `DEPTH_SCALE = 0.3`.** Crushed forward reach; arms drifted sideways and jittered.
- **TR8. Single-frame stillness threshold for the arm lock.** ZED noise keeps every frame above
  threshold, so the lock never re-engaged and the arm followed noise forever ("automove").
  Re-lock requires sustained stillness across `still_frames` consecutive frames.
- **TR9. One OpenCV key event per render tick.** Auto-repeat (~30 Hz) outruns the render loop
  (~25 Hz); the queue grows and input arrives seconds late. Drain to empty each tick.
- **TR10. `zed.grab()` inside the physics loop.** Blocks 35–50 ms, so the sim ran at ~40% of real
  time. Grab on a background thread; the main loop reads the newest frame.

## 10. Known open issues

- **O1.** Rubber hands have no collision → grasping impossible. Design decided (Q1), not built.
- **O2.** Gripper pads not built → 18 state and 2 action dims have a decided source but no code.
  Risk: a friction-only pinch may still slip while the biped walks. If 10/10 is not reached,
  escalate — pad friction, `solimp`, pad area, actuator kp, box mass, then weld (must disclose).
- **O3.** `pickup_half=0.13` + box half-width 0.09 = 0.22 m, but the platform half-extent is only
  0.18 m — corner spawns overhang and can topple. Resolve with the Q6 platform resize.
- **O4.** Box randomization only ever touches the *kinematic twin* (`G1Robot.__init__`); the
  stepped physics model keeps the keyframe position. **No episode currently varies box position
  at all** — Objectives 3 and 4 depend on fixing this.
- **O5.** ~~Waist never driven~~ — resolved by decision, not by code: waist stays pinned (D10).
  `set_waist_yaw()` remains dead code.
- **O6.** Dead code: `gating.py`, `GatingConfig`, `TorsoYawConfig`, `IKConfig.neutral_weight`,
  `IKConfig.target_deadzone`, `set_waist_yaw`. `RejectReason` survives only for `NAN`.
- **O7.** Stale docs: README claims torso-yaw following and active gating (neither is true);
  `run_integrated_combined.py` references a nonexistent `run_integrated_vision.py`;
  `config.py:35` says locomotion is "not yet built" (it is); `test/ik_test.py` uses platform
  (1.8, 1.5) top 0.47; `test/get_g1_links.py` points at a path that does not exist.
- **O9.** The outer working folder is not version-controlled — only `g1_vision_teleop/` is, so
  `docs/`, `test/`, this file and `PLAN.md` are untracked.
- **O10.** Rate decided (D4), but the **episode cap is open**: 500 steps at 25 Hz = 20 s, and
  walk 1.5 m + grasp + walk 1.5 m + place is likely 15–20 s at ~0.35 m/s. Set it from the Phase 4
  pilot's measured durations; raising to ~750 is a small, defensible deviation.
- **O11.** Sim-vs-wall-clock pacing exists only in the integrated entry point.
- **O12.** Q7 (does the robot walk during demonstrations at all?) is unanswered — a scope
  question, not an implementation one. See end of PLAN.md.

## 11. File and module structure

**Existing** — paths relative to the git repo `g1_vision_teleop/` unless prefixed `../`.
- `../docs/Thesis_Proposal.pdf` — 62 pages; Ch.3 methodology at PDF pages 33–56.
- `run_teleop.py` — kinematic ZED teleop, interactive viewer. `run_teleop_combined.py` — same,
  single OpenCV window, offscreen render.
- `run_integrated_combined.py` — full stack: physics + locomotion + ZED arms.
- `walk_test.py` — locomotion policy only, stepped physics, no ZED.
- `locomotion_input.py` — `PelvisVelocity`, `LeanJoystick`, `KeyboardCommand`. **Do not delete**
  — `PelvisVelocity` is the evidence for the §3.3.4 negative result (D6).
- `scene.xml` — platforms, goal marker, box. `g1.xml` — G1 29-DOF MJCF, actuators, keyframe
  `stand`, IMU sensors.
- `g1_teleop/config.py` — all tunables as dataclass configs.
- `g1_teleop/transforms.py` — camera→robot rotation, torso yaw helper.
- `g1_teleop/retargeting.py` — geometric scaling to elbow/wrist targets.
- `g1_teleop/ik.py` — damped least-squares IK with nullspace seed-pull.
- `g1_teleop/robot.py` — model wrapper, index resolution, twin box reset.
- `g1_teleop/teleop.py` — per-frame controller: NaN guard, IK, smoothing, stillness lock.
- `g1_teleop/` also: `onceuro.py` One-Euro filter, `overlay.py` debug overlay, `zed_source.py`
  ZED/BODY_38 wrapper, `gating.py` DEAD (kept only for `RejectReason`).
- `../unitree_rl_gym/deploy/` — `pre_train/g1/motion.pt` locomotion policy,
  `deploy_mujoco/configs/g1.yaml` reference gains. `../test/*.py` — stale scratch, not current.

**Planned** — see §5 for the full layout of `g1_data/`, `g1_policy/`, `g1_train/`, `g1_eval/`.

## 12. Conventions

- **ZED camera frame:** X right, Y down, Z forward. Metres. **G1 robot frame:** X forward, Y left,
  Z up. Mapping (`transforms.apply_camera_rotation`): `x_rob=-DEPTH_SCALE*z_cam`, `y_rob=+x_cam`,
  `z_rob=-y_cam`.
- **ZED BODY_38 indices:** pelvis 0, L/R shoulder 12/13, elbow 14/15, wrist 16/17.
- **Rates:** physics 500 Hz; locomotion 50 Hz (decimation 10); **recording + policy 25 Hz** (D4);
  ZED ~30 Hz; display ~25 Hz.
- **MuJoCo index layout:** `qpos` = base[0:7] + 29 joints[7:36] + box freejoint[36:43];
  `qvel` = base[0:6] + 29[6:35] + box[35:41]; `ctrl` = legs[0:12] (torque) + waist+arms[12:29]
  (position). Leg slices: qpos `[7:19]`, qvel `[6:18]`. **Gripper dims will extend all three —
  re-derive these after Phase 1, do not assume they still hold.**
- **Config:** tunables live in `g1_teleop/config.py` as frozen dataclasses aggregated by
  `TeleopConfig`. New subsystems follow that pattern. (The entry-point scripts violate it — they
  hold module-level constants.)

## 13. Verified vs assumed

**Verified by reading files or loading the model:** all four entry points; the kinematic-vs-
actuated split and the twin-qpos→ctrl copy; `N_IK_JOINTS=4` with pinned wrists; the 6-D
elbow+wrist IK task; locomotion policy path, obs layout, decimation, gains; scene geometry, box
mass/friction/solref, keyframe contents; nq/nv/nu/nkey; the full collision-geom list and the
absence of collision on both rubber hands; the total absence of grippers, recorder, dataset,
training and evaluation code; the dead code in O6; every contradiction in O7; proposal §3.3–§3.9
including Tables 3.1–3.6.

**Assumed, not verified:** that `run_integrated_combined.py` runs end to end here (never
executed — needs a ZED); that `pyzed` is installed; that the ZED is a 2i specifically (the
proposal says only "ZED stereo depth camera" and no code names a model); that box
friction/solref and the position-hold gains were tuned rather than inherited.

## 14. Maintenance protocol — instructions to future sessions

**At the end of every working session, before finishing, update this file:**
1. Rewrite §2 Current Blocker to whatever is actually blocking now, in one short paragraph. A
   fresh session must be able to start work from §2 alone without re-deriving anything.
2. Add a dated one-line entry to §8 for every decision made, with its reason.
3. Add an entry to §9 for anything tried that did not work, with why it failed.
4. Add a row to §7 for any new proposal divergence, with justification and the chapter affected.
5. Move completed items from §5 into §4, citing the file that proves it.
6. Update §10 — add new issues, delete resolved ones. Resolved issues are the *only* deletable
   thing in this file.

**Rules:**
- **Never delete a §9 entry.** That section is the most expensive knowledge here to rediscover.
  Entries stay even when they later look obvious.
- **Keep this file under 300 lines.** Past that, move detail to `NOTES.md` and leave a one-line
  pointer. Never solve overflow by cutting §9.
- **If code and this file disagree, the code is correct.** Fix the file, note the drift in §8.
- **If code and the proposal disagree, neither is automatically correct.** Do not silently
  reconcile — surface it to Charles and record the outcome in §7.
- **No routine implementation detail here.** Only decisions, state, and things a fresh session
  could not work out from the code in five minutes.
