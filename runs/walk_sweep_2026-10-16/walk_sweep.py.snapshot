"""Headless locomotion speed sweep for the pre-trained G1 walking policy.

    python tools/walk_sweep.py --axis vx --speeds -1.0 -0.5 0.5 1.0 --trials 10 --out runs/walk_sweep_vx.csv
    python tools/walk_sweep.py --axis vx --speeds 0.3 0.8 --trials 2          # smoke test

Same physics and policy interface as `walk_test.py` (500 Hz physics, policy at
50 Hz, same obs layout, gains, scales and arm hold), with no viewer. Every
constant and helper is IMPORTED from `walk_test.py`, not copied, so the two
cannot drift apart. The 47-D obs assembly mirrors walk_test.py:216-234 line for
line; that block lives inside walk_test's viewer loop and is not importable.

WHAT IS DIFFERENT FROM walk_test.py, ON PURPOSE
-----------------------------------------------
  * The command goes to the policy RAW. No clip from `spec.VELOCITY_CLIP` or
    `locomotion_input._clip_cmd` - the point is to test above 0.80. The only
    limit is the refusal at |speed| > 1.2 (`MAX_ABS_SPEED`). The policy's
    training ranges are +-1.0 on all three axes
    (unitree_rl_gym/legged_gym/envs/base/legged_robot_config.py:45-48).
  * The STAND is raw zero command (no position hold). The STOP is chosen by
    --stop-mode:
      hold  (default) the live teleop's station-keeping controller, mirrored
            from run_integrated_combined.py:809-824: position hold
            (IDLE_THRESHOLD, HOLD_KP, HOLD_MAX, HOLD_DEADBAND at :118) plus
            heading hold (HOLD_YAW_KP/MAX/DEADBAND at :132-134), with
            world_to_body at :199-201. The constants are IMPORTED from that
            module, so they cannot drift from what the operator runs. The
            target is the base xy and yaw at the stop command, as the live
            loop's target follows the robot until the command goes idle
            (:825-828). g1_teleop/config.py LocomotionConfig carries the same
            four position values (:294-297) but nothing reads them, and it has
            no heading hold; walk_test.py:74-77 has the same four values and
            no heading hold either.
      raw   zero command, the policy's own behaviour, drift included.
    Raw zero does NOT settle (2026-10-16: 3/5 trials at vx 0.8 never met the
    stillness rule within 10 s), so settle time is only meaningful under hold.
  * The LSTM state is zeroed per trial (`g1_data.reset.reset_policy_state`).
    walk_test runs one episode per process and never needs to.
  * The obstacles are made non-colliding by default (`--keep-obstacles` to
    keep them). The pickup platform is a slab at x = 1.5 m, 0.73 m up, which
    the pelvis cannot pass under (scene.xml:25-31): any forward trial longer
    than ~1.3 m would measure a collision, not the gait. The box is
    gravity-compensated so it hangs in place rather than falling forever.

TRIAL
-----
  reset to keyframe 0, legs at DEFAULT_ANGLES + leg noise; yaw/xy noise; stand
  at zero command for STAND_S + U(0, GAIT_PERIOD); --move-s at the test speed;
  --stop-s at zero command. A trial ends early on a fall.

  NOISE, all from the trial seed (`--no-noise` turns all of it off):
    yaw +-5 deg, xy +-2 cm   kept, but NOT a source of variation: on a flat
                             floor with the obstacles cleared they only rotate
                             and translate the same trajectory, and every
                             measure here is body-frame or tilt (smoke test
                             2026-10-16: trials differed at 1e-5).
    leg joints +-0.02 rad    uniform, each of the 12, around DEFAULT_ANGLES.
    stand time               STAND_S + U(0, GAIT_PERIOD), so the speed step
                             arrives at a random point of the gait cycle. The
                             policy's gait phase is a STEP COUNTER, not
                             data.time: phase = (counter*SIM_DT % GAIT_PERIOD)
                             / GAIT_PERIOD (walk_test.py:221-222, GAIT_PERIOD
                             = 0.8 at walk_test.py:63), counter 0 at reset.

  Trial seeds are `--seed + trial index`, the SAME for every speed, so speeds
  are compared on identical start perturbations (common random numbers).

MEASURES
--------
  fall        base_z < 0.35 (scripted_demo.DemoConfig.fall_z,
              g1_data/scripted_demo.py:236) OR |roll| or |pitch| > 0.8 rad.
  achieved    mean over the last MEASURE_S of the move, sampled every physics
              step: vx/vy in the YAW-ALIGNED body frame (world linear velocity
              rotated by -yaw), wz as the body-frame angular z rate (the
              quantity the policy was trained to track,
              legged_robot.py:698-700). NaN if the trial fell before the move
              ended.
  tracking    achieved / commanded on the swept axis.
  settle      the base-lock predicate's stillness rule, verbatim
              (g1_data/phases.py LockConfig): net base XY travel over
              settle_window = 20 ticks (0.80 s) <= disp_max = 0.010 m
              (phases.py:137,139), held for debounce = 5 consecutive 25 Hz
              ticks (phases.py:138), evaluated at the 25 Hz tick as
              LockPredicate.update does (phases.py:241-247; window semantics
              phases.py:175-178, 212-221). Settle time is measured from the
              stop command to the tick the debounce completes. The window is
              continuous across the stop, as the predicate's is; the debounce
              run starts at the stop.
  drift       |net base XY displacement| over the last DRIFT_S of the stop,
              divided by the time, in mm/s - the mean XY velocity's magnitude,
              so the in-place sway averages out. Logged in both stop modes;
              NaN if the trial fell.
  overshoot   max base XY distance (m) from the hold target - the base xy at
              the stop command - over every physics step of the stop. Filled
              in both stop modes; up to the fall if the trial fell.
  hold cmd    hold mode only, over every 50 Hz policy tick of the stop: max
              and mean planar command norm |(vx, vy)| and |wz|, and the
              fraction of ticks with 0 < |(vx, vy)| < 0.2 - planar commands the
              policy never trained on, because training zeroes them
              (unitree_rl_gym/legged_gym/envs/base/legged_robot.py:306).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np
import mujoco
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import walk_test as WT                                   # noqa: E402
import run_integrated_combined as RIC                    # noqa: E402  live teleop hold
from g1_data.phases import LockConfig                    # noqa: E402
from g1_data.reset import reset_policy_state             # noqa: E402
from g1_data.scripted_demo import DemoConfig             # noqa: E402
from g1_data import spec                                 # noqa: E402
from g1_teleop.indices import ModelIndex                 # noqa: E402

AXES = ("vx", "vy", "wz")
STOP_MODES = ("hold", "raw")
DRIFT_S = 2.0
UNTRAINED_PLANAR = 0.2                 # legged_robot.py:306: |(vx, vy)| below this is zeroed in training
MAX_ABS_SPEED = 1.2
STAND_S = 2.0
MEASURE_S = 5.0
FALL_Z = DemoConfig().fall_z           # 0.35, the demonstrator's own criterion
FALL_TILT = 0.8                        # rad, |roll| or |pitch|
YAW_NOISE = np.radians(5.0)
XY_NOISE = 0.02
LEG_NOISE = 0.02                       # rad, per leg joint
OBSTACLE_BODIES = ("platform_pickup", "platform_goal", "box1")
LOCK = LockConfig()
TICK = spec.PHYSICS_STEPS_PER_TICK     # 20 steps = 25 Hz, the predicate's rate

CSV_FIELDS = ["axis", "speed", "trial", "trial_seed", "noise", "stand_s",
              "cmd_gait_phase", "fell", "fall_time",
              "fall_phase", "achieved", "tracking_ratio", "max_abs_roll",
              "max_abs_pitch", "stop_mode", "settled", "settle_time",
              "drift_mm_s", "stop_overshoot_m", "hold_ticks", "hold_planar_max", "hold_planar_mean",
              "hold_wz_max", "hold_wz_mean", "hold_untrained_frac"]


def hold_command(data, ix, hold_xy, hold_yaw):
    """run_integrated_combined.py:810-824, verbatim but for the names."""
    active_cmd = np.zeros(3, dtype=np.float32)
    yaw_now = RIC.yaw_from_quat(data.qpos[ix.base_quat_qpos])
    err_world = hold_xy - data.qpos[ix.base_xy_qpos]
    if np.linalg.norm(err_world) >= RIC.HOLD_DEADBAND:
        err_body = RIC.world_to_body(err_world, yaw_now)
        active_cmd[0:2] = np.clip(RIC.HOLD_KP * err_body, -RIC.HOLD_MAX, RIC.HOLD_MAX)
    yaw_err = (hold_yaw - yaw_now + np.pi) % (2 * np.pi) - np.pi
    if abs(yaw_err) >= RIC.HOLD_YAW_DEADBAND:
        active_cmd[2] = float(np.clip(RIC.HOLD_YAW_KP * yaw_err,
                                      -RIC.HOLD_YAW_MAX, RIC.HOLD_YAW_MAX))
    return active_cmd


def roll_pitch(quat):
    qw, qx, qy, qz = quat
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    return float(roll), float(pitch)


def clear_obstacles(model):
    """Platforms and box stop colliding; the box hangs in place."""
    for name in OBSTACLE_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"obstacle body not found: {name}")
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid:
                model.geom_contype[g] = 0
                model.geom_conaffinity[g] = 0
        model.body_gravcomp[bid] = 1.0


def reset_trial(model, data, ix, policy, rng, noise=True):
    """walk_test.py:127-136, plus per-trial LSTM reset and start noise.

    Returns the stand time. Draw order is fixed (yaw, xy, legs, stand) so a
    trial seed always means the same perturbation.
    """
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[ix.leg_qvel] = 0.0
    if not noise:
        data.qpos[ix.leg_qpos] = WT.DEFAULT_ANGLES
        mujoco.mj_forward(model, data)
        reset_policy_state(policy)
        return STAND_S
    dyaw = rng.uniform(-YAW_NOISE, YAW_NOISE)
    dxy = rng.uniform(-XY_NOISE, XY_NOISE, size=2)
    dleg = rng.uniform(-LEG_NOISE, LEG_NOISE, size=WT.NUM_ACTIONS)
    stand_s = STAND_S + rng.uniform(0.0, WT.GAIT_PERIOD)
    data.qpos[ix.leg_qpos] = WT.DEFAULT_ANGLES + dleg
    data.qpos[ix.base_xy_qpos] += dxy
    q_yaw = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, np.array(data.qpos[ix.base_quat_qpos]))
    data.qpos[ix.base_quat_qpos] = q
    mujoco.mj_forward(model, data)
    reset_policy_state(policy)
    return stand_s


def run_trial(model, data, ix, policy, axis, speed, move_s, stop_s, trial_seed,
              noise=True, stop_mode="hold"):
    rng = np.random.default_rng(trial_seed)
    stand_s = reset_trial(model, data, ix, policy, rng, noise=noise)

    n_stand = int(round(stand_s / WT.SIM_DT))
    n_move = int(round(move_s / WT.SIM_DT))
    n_stop = int(round(stop_s / WT.SIM_DT))
    n_total = n_stand + n_move + n_stop
    measure_from = n_stand + n_move - int(round(min(MEASURE_S, move_s) / WT.SIM_DT))
    ax = AXES.index(axis)
    move_cmd = np.zeros(3, dtype=np.float32)
    move_cmd[ax] = speed
    zero_cmd = np.zeros(3, dtype=np.float32)

    action = np.zeros(WT.NUM_ACTIONS, dtype=np.float32)
    target_leg_pos = WT.DEFAULT_ANGLES.copy()
    obs = np.zeros(WT.NUM_OBS, dtype=np.float32)

    hist = deque(maxlen=LOCK.settle_window)
    run = 0
    settle_time = None
    fell, fall_time, fall_phase = False, None, ""
    max_roll = max_pitch = 0.0
    samples = []
    n_stop_start = n_stand + n_move
    drift_from = n_total - int(round(min(DRIFT_S, stop_s) / WT.SIM_DT))
    drift_xy0 = None
    hold_xy, hold_yaw = None, None
    hold_planar, hold_wz = [], []
    overshoot = float("nan")
    cmd = zero_cmd

    for counter in range(n_total):
        phase = ("stand" if counter < n_stand else
                 "move" if counter < n_stop_start else "stop")
        if phase != "stop" or stop_mode == "raw":
            cmd = move_cmd if phase == "move" else zero_cmd
        if counter == n_stop_start:
            hold_xy = np.array(data.qpos[ix.base_xy_qpos], dtype=np.float64)
            hold_yaw = RIC.yaw_from_quat(data.qpos[ix.base_quat_qpos])
            cmd = zero_cmd          # until the first hold tick computes one
        if counter == drift_from:
            drift_xy0 = np.array(data.qpos[ix.base_xy_qpos], dtype=np.float64)

        # 25 Hz stillness tick at mj_step ENTRY, as the recorder and predicate.
        if counter % TICK == 0:
            xy = np.array(data.qpos[ix.base_xy_qpos], dtype=np.float64)
            if len(hist) < hist.maxlen:
                disp = float("inf")
            else:
                disp = float(np.linalg.norm(xy - hist[0]))
            hist.append(xy)
            if phase == "stop" and settle_time is None:
                run = run + 1 if disp <= LOCK.disp_max else 0
                if run >= LOCK.debounce:
                    settle_time = (counter - n_stand - n_move) * WT.SIM_DT

        # walk_test.py:179-188
        leg_q = data.qpos[ix.leg_qpos]
        leg_dq = data.qvel[ix.leg_qvel]
        data.ctrl[ix.leg_ctrl] = (target_leg_pos - leg_q) * WT.KPS + (0.0 - leg_dq) * WT.KDS
        data.ctrl[ix.upper_ctrl] = WT.ARM_HOLD_TARGETS
        mujoco.mj_step(model, data)
        step = counter + 1

        quat = data.qpos[ix.base_quat_qpos]
        if phase == "stop":
            d_hold = float(np.linalg.norm(data.qpos[ix.base_xy_qpos] - hold_xy))
            overshoot = d_hold if not overshoot >= d_hold else overshoot
        roll, pitch = roll_pitch(quat)
        max_roll, max_pitch = max(max_roll, abs(roll)), max(max_pitch, abs(pitch))
        if (data.qpos[ix.base_qpos][2] < FALL_Z
                or abs(roll) > FALL_TILT or abs(pitch) > FALL_TILT):
            fell, fall_time, fall_phase = True, step * WT.SIM_DT, phase
            break

        if measure_from <= counter < n_stand + n_move:
            yaw = WT.yaw_from_quat(quat)
            if ax == 2:
                samples.append(float(data.qvel[ix.base_angvel_qvel][2]))
            else:
                v_body = WT.world_to_body(data.qvel[ix.base_qvel][0:2], yaw)
                samples.append(float(v_body[ax]))

        # walk_test.py:196-234, with the raw command in place of the hold logic.
        if step % WT.CONTROL_DECIMATION == 0:
            if phase == "stop" and stop_mode == "hold":
                cmd = hold_command(data, ix, hold_xy, hold_yaw)
                hold_planar.append(float(np.linalg.norm(cmd[0:2])))
                hold_wz.append(abs(float(cmd[2])))
            qj = (data.qpos[ix.leg_qpos] - WT.DEFAULT_ANGLES) * WT.DOF_POS_SCALE
            dqj = data.qvel[ix.leg_qvel] * WT.DOF_VEL_SCALE
            omega = data.qvel[ix.base_angvel_qvel] * WT.ANG_VEL_SCALE
            grav = WT.get_gravity_orientation(quat)
            t = step * WT.SIM_DT
            ph = (t % WT.GAIT_PERIOD) / WT.GAIT_PERIOD
            n = WT.NUM_ACTIONS
            obs[:3] = omega
            obs[3:6] = grav
            obs[6:9] = cmd * WT.CMD_SCALE
            obs[9:9 + n] = qj
            obs[9 + n:9 + 2 * n] = dqj
            obs[9 + 2 * n:9 + 3 * n] = action
            obs[9 + 3 * n:9 + 3 * n + 2] = [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target_leg_pos = action * WT.ACTION_SCALE + WT.DEFAULT_ANGLES

    fell_before_move_end = fell and fall_phase in ("stand", "move")
    achieved = (float(np.mean(samples)) if samples and not fell_before_move_end
                else float("nan"))
    ratio = achieved / speed if speed != 0 else float("nan")
    # The gait phase the policy sees on the first step under the new command.
    cmd_phase = ((n_stand * WT.SIM_DT) % WT.GAIT_PERIOD) / WT.GAIT_PERIOD
    drift = float("nan")
    if not fell and drift_xy0 is not None:
        span = (n_total - drift_from) * WT.SIM_DT
        drift = 1000.0 * float(np.linalg.norm(
            data.qpos[ix.base_xy_qpos] - drift_xy0)) / span
    hp, hw = np.asarray(hold_planar), np.asarray(hold_wz)
    nan = float("nan")
    hold = dict(
        hold_ticks=len(hp),
        hold_planar_max=float(hp.max()) if hp.size else nan,
        hold_planar_mean=float(hp.mean()) if hp.size else nan,
        hold_wz_max=float(hw.max()) if hw.size else nan,
        hold_wz_mean=float(hw.mean()) if hw.size else nan,
        hold_untrained_frac=(float(np.mean((hp > 0) & (hp < UNTRAINED_PLANAR)))
                             if hp.size else nan))
    return dict(noise=int(noise), stand_s=n_stand * WT.SIM_DT,
                cmd_gait_phase=cmd_phase,
                fell=int(fell), fall_time=fall_time if fell else "",
                fall_phase=fall_phase, achieved=achieved, tracking_ratio=ratio,
                max_abs_roll=max_roll, max_abs_pitch=max_pitch,
                stop_mode=stop_mode,
                settled=int(settle_time is not None),
                settle_time=(round(settle_time, 3) if settle_time is not None
                             else "not settled"),
                drift_mm_s=drift, stop_overshoot_m=overshoot, **hold)


def summarize(axis, speed, rows):
    n = len(rows)
    falls = sum(r["fell"] for r in rows)
    ratios = [r["tracking_ratio"] for r in rows if np.isfinite(r["tracking_ratio"])]
    tilt = max(max(r["max_abs_roll"], r["max_abs_pitch"]) for r in rows)
    settles = [r["settle_time"] for r in rows if r["settled"]]
    mean_ratio = f"{np.mean(ratios):.3f}" if ratios else "n/a"
    med = f"{np.median(settles):.2f} s" if settles else "n/a"
    return (f"{axis} {speed:+.2f}: falls {falls}/{n}  mean tracking {mean_ratio}  "
            f"worst tilt {tilt:.3f} rad  median settle {med} "
            f"({len(settles)}/{n} settled)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--axis", choices=AXES, required=True)
    p.add_argument("--speeds", type=float, nargs="+", required=True,
                   help="signed commands on --axis (m/s or rad/s)")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--move-s", type=float, default=10.0)
    p.add_argument("--stop-s", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="CSV path (default: runs/walk_sweep_<axis>.csv)")
    p.add_argument("--keep-obstacles", action="store_true",
                   help="leave platforms and box colliding (forward trials will hit the slab)")
    p.add_argument("--stop-mode", choices=STOP_MODES, default="hold",
                   help="hold: live teleop station-keeping after the move; raw: zero command")
    p.add_argument("--no-noise", action="store_true",
                   help="no start noise: legs at DEFAULT_ANGLES, no yaw/xy offset, stand exactly 2.0 s")
    a = p.parse_args(argv)
    bad = [s for s in a.speeds if abs(s) > MAX_ABS_SPEED]
    if bad:
        p.error(f"refusing |speed| > {MAX_ABS_SPEED}: {bad}")
    if a.trials < 1 or a.move_s <= 0 or a.stop_s <= 0:
        p.error("--trials, --move-s and --stop-s must be positive")
    if a.out is None:
        a.out = os.path.join(ROOT, "runs", f"walk_sweep_{a.axis}.csv")
    return a


def main(argv=None):
    a = parse_args(argv)
    model = mujoco.MjModel.from_xml_path(WT.SCENE_PATH)
    model.opt.timestep = WT.SIM_DT
    if not a.keep_obstacles:
        clear_obstacles(model)
    data = mujoco.MjData(model)
    ix = ModelIndex.resolve(model)
    policy = torch.jit.load(WT.POLICY_PATH)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    summaries = []
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for speed in a.speeds:
            rows = []
            for k in range(a.trials):
                t0 = time.perf_counter()
                seed = a.seed + k
                r = run_trial(model, data, ix, policy, a.axis, speed,
                              a.move_s, a.stop_s, seed, noise=not a.no_noise,
                              stop_mode=a.stop_mode)
                row = dict(axis=a.axis, speed=speed, trial=k, trial_seed=seed, **r)
                w.writerow(row)
                f.flush()
                rows.append(r)
                print(f"  {a.axis} {speed:+.2f} trial {k}: stand={r['stand_s']:.3f}s "
                      f"phase={r['cmd_gait_phase']:.2f} fell={r['fell']} "
                      f"achieved={r['achieved']:+.3f} ratio={r['tracking_ratio']:.3f} "
                      f"settle={r['settle_time']} drift={r['drift_mm_s']:.1f}mm/s"
                      f"  ({time.perf_counter() - t0:.1f} s wall)")
            summaries.append(summarize(a.axis, speed, rows))
    print("\n=== summary ===")
    for s in summaries:
        print(s)
    print(f"CSV: {a.out}")


if __name__ == "__main__":
    main()
