# PLAN.md — Current state to thesis completion

Companion to `CLAUDE.md`. CLAUDE.md holds *state and decisions*; this file holds *sequence*.
Read CLAUDE.md §2 (Current Blocker) first — it is more current than this file.

**Dependency shape.** Phases 1 and 2 are hard gates: nothing downstream is safe until both close.
Phase 5 (model code) is the one large chunk of work with **no dependency on Phase 1** and should
run in parallel with Phases 1–4 to compress the calendar. Phase 6 (collection) is the long pole
in wall-clock time and cannot start until Phase 4 closes.

```
P1 grasp physics ──┐
                   ├─> P3 recorder ──> P4 pilot ──> P6 collection ──> P7 train+eval ──> P8 writing
P2 freeze specs ───┘                                                    ^
P5 model codebase ──────────────────────────────────────────────────────┘  (parallel from day 1)
```

---

## Phase 1 — Stabilize box-grasp physics — HARD GATE

**Decided (2026-08-23):** the gripper is one slide joint + one `<position>` actuator + one
high-friction collision pad per hand, pressing the opposite faces of the box. Remaining unknowns:
whether the pinned-wrist IK (D2) produces a palm orientation that can present the pads squarely
to both faces — verify early, it is the assumption most likely to be wrong — and whether box
mass/friction/solref (0.4 kg / 1.5 / 0.004) need retuning.

**Tasks:**
1. Add the pad bodies, slide joints, position actuators, and collision geoms to `g1.xml`. Fixes
   O1. **This shifts every `qpos`/`qvel`/`ctrl` index** — re-derive the slices in CLAUDE.md §12
   and update every script that hardcodes them (`LEG_QPOS`, `ARM_CTRL`, and friends).
2. Expose the grasp behind a scalar 0–1 command per hand, matching the 2 gripper action dims.
3. Define what `gripper_state` reads back as — the pad slide-joint qpos, normalized. Fixes 2
   state dims.
4. Add a gripper *site* on each pad for end-effector pose readout. Fixes 14 state dims.
5. Fix O3 (spawn range overhangs the platform) and O4 (randomization never reaches the physics
   model) while you are in here.
6. Build a scripted, ZED-free grasp-and-carry test.

**Exit criterion:** a scripted run (no human, no ZED) picks up the 0.4 kg box, carries it while
the locomotion policy walks the full 1.5 m to the goal platform, and places it — 10 out of 10
times, without dropping, without the box tunnelling, and without destabilising the walker.

**What breaks downstream if done wrong:** everything. A flaky grasp makes every demonstration
noisy in the *same* way, so all three policies learn the flakiness and the ACT-vs-ACT-LSTM
comparison ends up measuring grasp noise instead of architecture. Phase-level success rates
become uninterpretable, because grasp failures contaminate transport and place.

---

## Phase 2 — Freeze state vector, action vector, spatial split — HARD GATE

Can start as soon as Phase 1's gripper *decision* is made, before its implementation lands.

**Decided (2026-08-23):** 25 Hz recording (Q3); waist pinned, dims kept as constants (Q5);
pinned wrist dims kept so the vectors still match proposal Tables 3.3/3.4; `KeyboardCommand`
supplies the 3 velocity dims (Q2). **Still open:** the episode cap (CLAUDE.md O10) — set it from
the Phase 4 pilot's measured durations, not now.

**Tasks:**
1. Write `g1_data/spec.py`: named, ordered, versioned index maps for the 47-D state and 22-D
   action. Every other module imports from it; nothing else may hardcode an index.
2. Enlarge the pickup platform and define the training and held-out regions in explicit
   coordinates, split along x per Q6. Held-out must be genuinely disjoint, not merely resampled,
   or Objective 4 measures nothing.
3. Fix the recording rate (25 Hz) and, after Phase 4, the episode cap in `configs/task.yaml`.
4. Write the normalization contract: statistics from the training split only, applied unchanged
   at validation, test, and autonomous deployment.

**Exit criterion:** `spec.py` exists, both vectors are dimension-complete with no `TODO` entries,
and a platform diagram showing training vs held-out regions is saved for Chapter 3.

**What breaks downstream if done wrong:** a state/action change after collection invalidates
every recorded episode — the single most expensive mistake available in this project. A split
that silently overlaps destroys Objective 4: the generalization gap comes out near zero and it
looks like a positive result.

---

## Phase 3 — Data logging pipeline

**Decide before starting:** episode file format (recommend one `.npz` per episode holding
`states`, `actions`, `phase_labels`, `box_spawn_xy`, `success_flags`, `metadata`), and whether
phase labels are auto-derived or hand-marked.

**Tasks:**
1. `g1_data/recorder.py` — logs state and action at the Phase-2 rate. **Action is
   `data.ctrl[12:29]` on the physics model, plus the 2 gripper commands and the 3 velocity dims —
   not twin qpos.** Buffer in memory and write once at episode end, so a rejected episode costs
   nothing.
2. `g1_data/reset.py` — full physics-model episode reset: robot to the `stand` keyframe with legs
   at `DEFAULT_ANGLES` (never straight legs — CLAUDE.md TR5), zero velocities, box sampled from
   the training region, and teleop controller state cleared (stillness locks, depth filter,
   One-Euro history).
3. `g1_data/phases.py` — approach / grasp-lift / transport / place, auto-derived from box height,
   box-to-base distance, and gripper state. Auto-derivation is preferred: hand-labelling 150
   episodes is slow and inconsistent between sessions.
4. `g1_data/success.py` — the four criteria from proposal §3.8.2: grasp (both pads closed AND box
   above platform surface), lift (box ≥ platform + 0.05 m), transport (box above floor level
   throughout), place (`||box_final − goal_center||_xy < 0.10 m` **and** box resting on the goal
   platform, roughly upright — Q4).
5. Operator controls in the recorder: start / discard-and-retry / accept, implementing proposal
   §3.3.8's per-episode visual inspection.

**Exit criterion:** 5 hand-driven episodes recorded, reloaded from disk, and replayed open-loop
into the simulator, reproducing the original trajectory within a stated tolerance. If replay does
not reproduce, the logged action is the wrong quantity — find that out now, not after 150
episodes.

**What breaks downstream if done wrong:** if the recorder logs the wrong action quantity,
behavioural cloning trains on a signal that never actually drove the robot, and every policy
fails for a reason no amount of architecture work can fix. The replay test is the only thing that
catches this.

---

## Phase 4 — Pilot: full loop end to end — HARD GATE before scaling

**Decide before starting:** nothing new. This phase exists precisely to surface what Phases 1–3
got wrong while being wrong costs 10 episodes instead of 150.

**Tasks:**
1. Record **10 complete episodes** with the real ZED, real demonstrator, real walking, real
   grasp, real placement — box positions from the training region only.
2. Measure per-episode wall-clock time and the reject rate. This turns Phase 6's schedule into a
   number rather than a hope.
3. Build the dataset from those 10, run normalization and action-chunk construction, and train
   the BC baseline to overfit them deliberately. A model that cannot overfit 10 episodes has a
   data-format bug, not a capacity problem.
4. Deploy that overfit BC policy autonomously and confirm the deployment loop closes: policy
   reads state, emits a 22-D action, arm/waist/gripper dims reach the actuators, velocity dims
   reach the locomotion policy.
5. Confirm phase labels and success detection fire correctly on both known-good and deliberately
   sabotaged episodes.

**Exit criterion:** 10 clean episodes on disk; BC overfits them; the autonomous deployment loop
runs a full episode without crashing; the success detector agrees with human judgement on all 10.

**What breaks downstream if done wrong:** skipping this phase is how a project discovers a format
bug after 150 episodes of irreplaceable human-generated data. Everything found here is found
cheaply.

---

## Phase 5 — Shared model codebase — RUNS IN PARALLEL with Phases 1–4

No dependency on the grasp mechanism, only on the Phase-2 dimensions. Start it immediately;
it is the largest body of code in the project and the only one that can proceed while the
simulator is broken.

**Decide before starting:** observation window `W_o`, action chunk size `K`, LSTM hidden size,
LSTM layer count (proposal says 2 stacked, dropout 0.3), transformer width/depth, and the KL
weight β. The proposal explicitly defers all of these to implementation.

**Tasks:**
1. `g1_policy/base.py` — one interface all three conditions satisfy, so `g1_eval/deploy.py` never
   branches on policy type.
2. `g1_policy/bc.py` — feedforward, single-step action prediction, L1 loss only, no CVAE.
3. `g1_policy/act.py` — **one class**. Transformer encoder/decoder, action chunking, CVAE
   objective (L1 + β·KL), and a `use_lstm` flag that adds the stacked LSTM encoder whose hidden
   state is projected and concatenated as an input token to the chunk decoder. Same init, same
   optimizer, same schedule, same everything else.
4. `g1_policy/ensemble.py` — temporal ensembling of overlapping chunks at deployment (exponential
   weighting, per Zhao et al. 2023). Shared by ACT and ACT-LSTM.
5. `g1_train/train.py` — Adam, cosine annealing, early stop after 10 epochs without validation
   improvement, best-checkpoint save. Identical across all three conditions.
6. Shape-and-gradient unit tests on synthetic data at the Phase-2 dimensions. These run before
   any real data exists.

**Exit criterion:** all three conditions train to convergence on synthetic data, and toggling
`use_lstm` is the *only* difference between the ACT and ACT-LSTM runs — verified by diffing the
two config objects.

**What breaks downstream if done wrong:** if ACT and ACT-LSTM become two implementations, the gap
between them confounds the LSTM with implementation differences and RQ2/RQ3 cannot be answered.
That is the entire reason plain ACT was added (CLAUDE.md D8). Do not undermine it.

---

## Phase 6 — Demonstration collection

**Decide before starting:** final episode count (proposal says 100–150), the demonstrator
training protocol, and a sampling scheme that guarantees no two consecutive episodes share a box
position.

**Tasks:**
1. Collect 100–150 episodes from the training region, visually inspecting and re-recording
   failures per proposal §3.3.8.
2. Log every box spawn position. At the end, verify coverage of the training region is actually
   uniform — plot it. That plot is a Chapter 3 figure.
3. Verify no spawn falls inside the held-out region. A single leak invalidates Objective 4.
4. Back up `data/raw/` off-machine. This is irreplaceable human-generated data and the outer
   project folder is not even version-controlled (CLAUDE.md O9).

**Exit criterion:** 100–150 accepted episodes, coverage plot generated, zero held-out leakage,
backup verified restorable.

**What breaks downstream if done wrong:** non-uniform coverage means the in-distribution result
really measures where the demonstrator happened to stand. A held-out leak makes the
generalization gap meaningless in the direction that flatters the result — the worst kind of
error, because it looks like success.

---

## Phase 7 — Training and evaluation

**Decide before starting:** number of evaluation episodes per condition (proposal says 100 each
for Experiments 1 and 2), and whether evaluation seeds are fixed across the three policies. They
should be — identical box positions for all three, or the comparison carries an uncontrolled
variable.

**Tasks:**
1. Train all three conditions on the identical dataset, split 80/20 stratified by box position,
   with identical hyperparameters wherever shared.
2. **Experiment 1** — 100 in-distribution episodes per condition. Report four phase-level success
   rates each (grasp, lift, transport, place).
3. **Experiment 2** — 100 held-out-position episodes per condition, same four rates.
4. Compute the generalization gap per phase per policy: `Δ = SR_in − SR_out`.
5. Report three pairwise comparisons: BC vs ACT (does chunking help?), ACT vs ACT-LSTM (does the
   LSTM help — the actual research question), BC vs ACT-LSTM (the proposal's headline).
6. Record failure-mode video. The phase-level table says *where* each policy fails; video says
   *how*, and panels ask.

**Exit criterion:** a 3-policy × 4-phase × 2-experiment results table plus the gap table, every
number reproducible from a saved checkpoint and a fixed seed list.

**What breaks downstream if done wrong:** unmatched hyperparameters or unmatched evaluation seeds
across conditions make the comparison indefensible at the panel, and no amount of writing repairs
it afterwards.

---

## Phase 8 — Thesis writing

Can start during Phase 6 — collection is mostly waiting.

| Chapter | Fed by | Status |
|---|---|---|
| 1 Introduction | — | proposal text largely reusable; RQ2 must fold in the plain-ACT condition (D8) |
| 2 RRL | — | reusable; add a chunking-ablation framing |
| 3.1–3.2 Design | — | reusable |
| 3.3.1 Scene | P1, P2 | **rewrite**: init pose (D5), control rate (D4), spawn region + held-out split |
| 3.3.3 Coordinate transform | existing code | **rewrite**: `DEPTH_SCALE` is not a pure rotation (D1) |
| 3.3.4 Walking detection | P2 | **rewrite**: the literal pelvis-velocity method does not work (D6, TR1). Report it — a documented negative result on a proposed method is a genuine contribution, not a retreat |
| 3.3.5 Arm teleoperation | existing code | **rewrite**: 4-joint IK, pinned wrists, elbow+wrist 6-D task (D2, D3) |
| 3.3.6 Grasping trigger | P1 | **full rewrite**: the described mechanism is not implementable — no hand collision (D7) |
| 3.3.8 Protocol | P4, P6 | update with pilot-measured timings and the real reject rate |
| 3.4 State/Action | P2 | update dimension sources; gripper rows depend on the Phase-1 decision |
| 3.5 Preprocessing | P5 | mostly reusable |
| 3.6 Architecture | P5 | **extend**: add plain ACT as a third condition and justify it (D8) |
| 3.7 Training | P5, P7 | fill in the hyperparameters the proposal deferred |
| 3.8 Evaluation | P7 | extend to three policies; set `d_place`; reconcile the 500-step cap with the chosen rate |
| 3.9 Hardware/Software | — | **rewrite**: Windows 11, no ROS2 (D9) |
| 4 Results | P7 | new |
| 5 Conclusion | P7 | new |

**Exit criterion:** defence-ready draft in which every deviation listed in CLAUDE.md §7 has a
written justification in its corresponding chapter.

---

## Open questions

**Q1–Q6 answered 2026-08-23.** Full decisions with reasons are in CLAUDE.md §8; the short form:

- **Q1 gripper** — one slide joint + one `<position>` actuator + one high-friction collision pad
  per hand, pressing the opposite faces of the box. Gives the proposal's 0–1 gripper command real
  physical meaning, and decouples press force from arm IK so the arm holds a clean pose instead
  of sitting in permanent IK error that would corrupt the arm dims it is recorded into.
- **Q2 locomotion interface** — `KeyboardCommand`. Pelvis-velocity triggering is abandoned and
  written up as a negative result. `locomotion_input.py` stays in the repo as the evidence.
- **Q3 rate** — record and run the policy at 25 Hz; locomotion inner loop stays at 50 Hz.
- **Q4 `d_place`** — 0.10 m XY from box centre to goal centre, plus the box must be resting on
  the goal platform and roughly upright.
- **Q5 waist** — stays pinned at 0. Dims retained as constants; turning is done by `wz`.
- **Q6 split** — provisional: enlarge the pickup platform, split along x (distance from robot),
  near half training / far half held-out. Must close before Phase 2 does.

**Q7 — STILL OPEN (scope, not implementation). Does the robot walk during demonstrations at
all?** The task requires walking 1.5 m between platforms, and the box starts 1.5 m from the
robot. But TR1–TR4 document that in-place walking detection does not work and the gait is not
adjustable. `KeyboardCommand` (Q2) removes the *detection* problem, so walking is now
demonstrable — but it is still the least reliable part of the loop and the biggest risk to
Phase 1's 10/10 exit criterion. If reliable walking transport proves impractical, the fallback is
to place both platforms within arm's reach and drop locomotion — but that turns the task from
loco-manipulation into manipulation and weakens the thesis's contribution. Decide this
deliberately at the Phase 1 exit, not by attrition during collection.
