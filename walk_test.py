"""Walk-in and base-lock PRACTICE (default), and the standalone walking test (--legacy).

PRACTICE MODE (default)
-----------------------
    python walk_test.py                      # first training seed, guide on
    python walk_test.py --seed 12 --guide off --lock-rule v3

The SAME configuration as the final teleop, from ONE source: it opens the
teleop session itself (`run_integrated_combined.open_session` / `run_loop`,
the loop tools/record_teleop_live.py records) with every operator-facing
setting from g1_teleop/operator_config.py - scene, start pose (0.60, 0.00),
keyboard keys and speeds, the lock rule (default v3), the lock indicator, the
floor guide (g1_teleop/floor_guide.py), the seeded box spawn and the base-lock
behaviour (predicate, pelvis weld, legs PD-held, u = abort). Only the camera is
missing: a no-camera source delivers frames with no body, so the arms are NOT
driven - this is for practising the walk-in and the lock. Training seeds only
(data/eval/train_seeds.json); exp1 / exp2 / selection seeds are refused.

  arrows (or i j k l)  walk / turn (hold)     n / m  strafe     p  precision
  space  stop          u  abort the lock      o  overlay       q  quit
  t  new try on the NEXT training seed        y  new try on the SAME seed
  b  is blocked (B-prime is fixed ON for collection, D18)

Per try it reports time to lock, lateral and heading error at the lock and
the overshoot (how far the base came closer than the band's near edge before
locking), on screen and in the terminal, with a summary at quit.

LEGACY: THE STANDALONE WALKING TEST (--legacy)
----------------------------------------------
NOT the practice configuration (its keys step an accumulating command; it has
no lock). Kept because it is the plainest demonstration of the two facts below.

Runs the Unitree pre-trained G1 walking policy on the 12 leg joints of your
full 29-DOF robot (with box + platforms), holding the arms/waist at a fixed
pose via their position actuators, under stepped physics.

Two things this script demonstrates:

1. The policy IS the balance controller. It must run every control step, even
   at zero velocity command — the small in-place march is how a biped keeps its
   centre of mass over its support polygon. Freezing the legs at a fixed stance
   removes all balance feedback and the robot falls. (Do not "fix" the stepping
   by bypassing the policy.)

2. Because this robot carries arm/torso mass the 12-DOF training model did not,
   it drifts slowly at zero command (~2 cm/s) and never corrects back. A
   position-hold outer loop closes that gap: measure how far the base has moved
   from where it should be, and feed a small corrective velocity back into the
   policy. The policy already knows how to walk — we just tell it where to go.

Controls (focus the MuJoCo window). Each press STEPS the command by 0.1 and the
current (vx, vy, wz) is printed after every key:
  arrow up/down    : forward speed +/- 0.1
  arrow left/right : turn rate +/- 0.1
  n / m            : lateral speed +/- 0.1 (strafe left / right)
  space            : stop and hold current position
  h                : re-anchor the hold target to the current position
  (close window to quit)
Limits are MAX_FORWARD / MAX_LATERAL / MAX_TURN from locomotion_input.py, so
the command stays inside spec.VELOCITY_CLIP. Every letter is also a MuJoCo
viewer shortcut (n island, m centre of mass, h convex hull), so those display
toggles flip as a side effect.

Paths: g1_teleop/operator_config.py (env G1_POLICY_PATH / G1_SCENE_PATH).
Run:   python walk_test.py --legacy
"""
import os
import time
from typing import Optional

import numpy as np
import mujoco
import mujoco.viewer
import torch

from g1_teleop import operator_config as OC
from g1_teleop.indices import ModelIndex
from locomotion_input import (
    MAX_FORWARD, MAX_LATERAL, MAX_TURN,
    KEYCODE_UP, KEYCODE_DOWN, KEYCODE_LEFT, KEYCODE_RIGHT,
    KEYCODE_STRAFE_LEFT, KEYCODE_STRAFE_RIGHT, KEYCODE_SPACE)

CMD_STEP = 0.1          # per key press, on vx, vy and wz

# ── Paths ─────────────────────────────────────────────────────────────────────
# G1_POLICY_PATH / G1_SCENE_PATH override the paths per machine; unset, both
# resolve inside this checkout. Defined ONCE, in g1_teleop/operator_config.py,
# for this file and the teleop entry points alike (they held identical copies).
POLICY_PATH, SCENE_PATH = OC.POLICY_PATH, OC.SCENE_PATH

# ── Policy config (from unitree_rl_gym deploy/deploy_mujoco/configs/g1.yaml) ──
SIM_DT = 0.002
CONTROL_DECIMATION = 10          # policy runs every 10 sim steps => 50 Hz
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS = 12
NUM_OBS = 47
GAIT_PERIOD = 0.8

# ── Command limits: MAX_FORWARD / MAX_LATERAL / MAX_TURN, imported above ──────

# ── Position hold (station keeping at zero command) ───────────────────────────
# Below IDLE_THRESHOLD the operator is not commanding motion, so instead of
# sending zero (which lets the robot drift) we send a small corrective velocity
# proportional to how far it has strayed from the hold target.
IDLE_THRESHOLD = 0.05   # command magnitude below which hold takes over
HOLD_KP = 0.8           # corrective m/s per metre of position error
HOLD_MAX = 0.25         # cap on corrective speed (keep well under MAX_FORWARD)
HOLD_DEADBAND = 0.02    # ignore errors under 2 cm to avoid twitchy corrections

# ── Layout of YOUR scene's qpos/qvel/ctrl ─────────────────────────────────────
# Resolved by joint/actuator NAME at model load — see g1_teleop/indices.py.
# These offsets used to be literals (legs qpos[7:19], waist+arms ctrl[12:29]
# and so on). They must not be: the palm-pad gripper adds a slide joint inside
# each wrist_yaw_link, which shifts right-arm indices while leaving leg indices
# alone, so a literal slice would stay correct for the legs and silently
# address the wrong joints for the arms.

# Fixed pose for waist+arms while testing walking. Matches the keyframe ctrl
# block so the arms do not jolt at startup (a jolt shoves the torso and can
# topple the walker).
ARM_HOLD_TARGETS = np.array([
    0.0, 0.0, 0.0,                        # waist yaw, roll, pitch
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,   # left arm
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,  # right arm
], dtype=np.float32)


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def yaw_from_quat(quat):
    qw, qx, qy, qz = quat
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def world_to_body(vec_world, yaw):
    """Rotate a world-frame XY vector into the robot's body frame.

    The velocity command the policy consumes is body-relative, but drift is
    measured in world coordinates. Without this rotation the correction would
    push the wrong way once the robot has turned.
    """
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * vec_world[0] - s * vec_world[1],
                     s * vec_world[0] + c * vec_world[1]])


def legacy_main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    mujoco.mj_resetDataKeyframe(model, data, 0)

    ix = ModelIndex.resolve(model)

    # The policy expects the legs to START at DEFAULT_ANGLES (a slight knee-bent
    # crouch), not the teleop keyframe's straight-leg pose. Starting elsewhere
    # puts the policy in a state it never trained on and it cannot recover.
    data.qpos[ix.leg_qpos] = DEFAULT_ANGLES
    data.qvel[ix.leg_qvel] = 0.0
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(POLICY_PATH)

    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_leg_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # operator command
    hold_target = np.array(data.qpos[ix.base_xy_qpos], dtype=np.float64)  # world XY
    counter = 0

    def key_callback(keycode):
        if keycode == KEYCODE_UP:
            cmd[0] += CMD_STEP
        elif keycode == KEYCODE_DOWN:
            cmd[0] -= CMD_STEP
        elif keycode == KEYCODE_LEFT:
            cmd[2] += CMD_STEP
        elif keycode == KEYCODE_RIGHT:
            cmd[2] -= CMD_STEP
        elif keycode == KEYCODE_STRAFE_LEFT:
            cmd[1] += CMD_STEP
        elif keycode == KEYCODE_STRAFE_RIGHT:
            cmd[1] -= CMD_STEP
        elif keycode == KEYCODE_SPACE:   # space — stop and hold here
            cmd[:] = 0.0
            hold_target[:] = data.qpos[ix.base_xy_qpos]
            print(f"holding at x={hold_target[0]:+.3f} y={hold_target[1]:+.3f}")
        elif keycode == ord('H'):   # re-anchor hold target
            hold_target[:] = data.qpos[ix.base_xy_qpos]
            print(f"hold re-anchored at x={hold_target[0]:+.3f} y={hold_target[1]:+.3f}")
        # Snap to the 0.1 grid so repeated float32 steps print and command
        # exactly what was pressed (0.1 + 0.1 + 0.1 is not 0.3 in float32).
        cmd[:] = np.round(cmd, 2)
        cmd[0] = np.clip(cmd[0], -MAX_FORWARD, MAX_FORWARD)
        cmd[1] = np.clip(cmd[1], -MAX_LATERAL, MAX_LATERAL)
        cmd[2] = np.clip(cmd[2], -MAX_TURN, MAX_TURN)
        print(f"cmd = vx {cmd[0]:+.2f} m/s   vy {cmd[1]:+.2f} m/s   wz {cmd[2]:+.2f} rad/s")

    print("=== Standalone walking test ===")
    print("Focus the MuJoCo window. Arrows: forward/turn +/-0.1. n/m: strafe +/-0.1. "
          "Space: stop+hold. H: re-anchor.")
    print("At zero command the robot marches in place — that is the policy balancing.")
    print("Position hold keeps it from drifting away while it does so.\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # PD torque control on the LEG actuators (motor type).
            leg_q = data.qpos[ix.leg_qpos]
            leg_dq = data.qvel[ix.leg_qvel]
            tau = (target_leg_pos - leg_q) * KPS + (0.0 - leg_dq) * KDS
            data.ctrl[ix.leg_ctrl] = tau

            # Hold waist+arms at a fixed pose (position actuators).
            data.ctrl[ix.upper_ctrl] = ARM_HOLD_TARGETS

            mujoco.mj_step(model, data)
            counter += 1

            if counter % 500 == 0:   # ~1 Hz drift readout
                err = np.linalg.norm(hold_target - data.qpos[ix.base_xy_qpos])
                bx, by, bz = data.qpos[ix.base_qpos][:3]
                print(f"base xyz = {bx:+.3f} {by:+.3f} "
                      f"{bz:.3f}   hold_err = {err:.3f} m")

            if counter % CONTROL_DECIMATION == 0:
                # ── Choose the command the policy actually receives ─────────
                # Idle: substitute a corrective velocity to hold station.
                # Active: pass the operator command straight through, and keep
                # the hold target following the robot so releasing the keys
                # holds wherever it ended up.
                if np.linalg.norm(cmd) < IDLE_THRESHOLD:
                    err_world = hold_target - data.qpos[ix.base_xy_qpos]
                    if np.linalg.norm(err_world) < HOLD_DEADBAND:
                        active_cmd = np.zeros(3, dtype=np.float32)
                    else:
                        yaw = yaw_from_quat(data.qpos[ix.base_quat_qpos])
                        err_body = world_to_body(err_world, yaw)
                        active_cmd = np.zeros(3, dtype=np.float32)
                        active_cmd[0:2] = np.clip(HOLD_KP * err_body,
                                                  -HOLD_MAX, HOLD_MAX)
                else:
                    active_cmd = cmd.astype(np.float32)
                    hold_target[:] = data.qpos[ix.base_xy_qpos]

                qj = (data.qpos[ix.leg_qpos] - DEFAULT_ANGLES) * DOF_POS_SCALE
                dqj = data.qvel[ix.leg_qvel] * DOF_VEL_SCALE
                omega = data.qvel[ix.base_angvel_qvel] * ANG_VEL_SCALE
                grav = get_gravity_orientation(data.qpos[ix.base_quat_qpos])

                t = counter * SIM_DT
                phase = (t % GAIT_PERIOD) / GAIT_PERIOD
                sin_p, cos_p = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = grav
                obs[6:9] = active_cmd * CMD_SCALE
                obs[9:9 + NUM_ACTIONS] = qj
                obs[9 + NUM_ACTIONS:9 + 2 * NUM_ACTIONS] = dqj
                obs[9 + 2 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS] = action
                obs[9 + 3 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS + 2] = [sin_p, cos_p]

                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
                target_leg_pos = action * ACTION_SCALE + DEFAULT_ANGLES

            viewer.sync()
            dt_left = model.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)


# ─── practice mode ────────────────────────────────────────────────────────────
class NoCameraSource:
    """The teleop loop's camera with no camera: every frame has NO body, so the
    arms hold their reset pose. The image is the left panel's notice."""

    def __init__(self, fps=None):
        self.fps = fps
        self._image = None

    def _notice(self):
        import cv2
        img = np.zeros((620, 760, 3), np.uint8)
        for i, (line, col) in enumerate((
                ("NO CAMERA - PRACTICE MODE", (255, 255, 0)),
                ("arms are not driven", (200, 200, 200)),
                ("walk to the box and get the base LOCKED", (200, 200, 200)),
                ("t next seed   y same seed   q quit", (200, 200, 200)))):
            cv2.putText(img, line, (30, 60 + 44 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2,
                        cv2.LINE_AA)
        return img

    def grab(self):
        from g1_teleop.zed_source import BodyFrame
        if self.fps:
            time.sleep(1.0 / self.fps)
        if self._image is None:
            try:
                self._image = self._notice()
            except ImportError:                    # no cv2: headless tests
                self._image = np.zeros((8, 8, 3), np.uint8)
        return BodyFrame(keypoints_3d=[], keypoints_2d=[], confidences=[], image=self._image)

    def close(self):
        pass


class PracticeStats:
    """One try's walk-in, from the lock terms at each 25 Hz tick. Pure: fed by
    `update`, read by `summary`; tested without a simulator."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.min_fwd = float("inf")
        self.band_entries = 0
        self._in_band = False
        self.lock = None                     # dict at the first lock, else None
        self.fell = False
        self.t_end = 0.0

    def in_band(self, t) -> bool:
        c = self.cfg
        return bool(c.fwd_lo <= t["fwd"] <= c.fwd_hi and abs(t["lat"]) <= c.lat_max
                    and abs(t["head"]) <= c.head_max_deg)

    def update(self, t_sim: float, terms: dict, locked: bool) -> None:
        """`terms` are the tick's LockPredicate terms; `locked` whether the
        base lock is engaged AFTER that tick's lock update."""
        self.t_end = t_sim
        if self.lock is not None or not terms:
            return
        if not terms["welded"]:
            self.min_fwd = min(self.min_fwd, terms["fwd"])
        now = self.in_band(terms)
        if now and not self._in_band:
            self.band_entries += 1
        self._in_band = now
        if locked:
            self.lock = dict(t=t_sim, fwd=terms["fwd"], lat=terms["lat"], head=terms["head"])

    def summary(self) -> dict:
        c, L = self.cfg, self.lock
        shoot = (max(0.0, c.fwd_lo - self.min_fwd) if self.min_fwd != float("inf")
                 else float("nan"))
        return dict(locked=L is not None, time_to_lock_s=L["t"] if L else float("nan"),
                    fwd_at_lock_m=L["fwd"] if L else float("nan"),
                    lat_err_at_lock_m=L["lat"] if L else float("nan"),
                    head_err_at_lock_deg=L["head"] if L else float("nan"),
                    overshoot_m=shoot, band_entries=self.band_entries, fell=self.fell,
                    try_seconds=self.t_end)


def format_stats(st: dict) -> str:
    if st["locked"]:
        s = ("locked after %.1f s   lat %+.3f m   head %+.1f deg   fwd %.3f m   overshoot %.3f m"
             % (st["time_to_lock_s"], st["lat_err_at_lock_m"], st["head_err_at_lock_deg"],
                st["fwd_at_lock_m"], st["overshoot_m"]))
    else:
        s = "NOT locked (%.1f s)   overshoot %.3f m" % (st["try_seconds"], st["overshoot_m"])
    return s + ("   FELL" if st["fell"] else "")


class PracticeHooks:
    """The practice try's view of `run_loop` (a LoopHooks): stats, keys, HUD.
    Reads only - the same contract as the live recorder's EpisodeHooks."""

    def __init__(self, S, seed: int, try_no: int, last: Optional[dict], script=None):
        self.S, self.seed, self.try_no, self.last = S, seed, try_no, last
        self.stats = PracticeStats(S.lock_rule)
        self.request = None                  # "next" | "same" | "fell"
        self.script = sorted(script or [])
        self._terms = {}
        self.fall_z = _fall_z()

    def keys_due(self, t_sim):
        due = []
        while self.script and self.script[0][0] <= t_sim + 1e-12:
            due.append(self.script.pop(0)[1])
        return due

    def on_key(self, key, S):
        if key == OC.KEY_PRACTICE_NEXT:
            self.request = "next"
            return True
        if key == OC.KEY_PRACTICE_SAME:
            self.request = "same"
            return True
        if key == ord("b"):
            print("b ignored: B-prime is fixed ON for collection (D18), and practice "
                  "runs what collection runs")
            return True
        return False

    def should_end(self, S, counter):
        tick = _TICK
        if counter % tick == 1:
            # One step after a tick boundary: that tick's lock update has run.
            self.stats.update((counter - 1) * SIM_DT, self._terms, S.base_lock.locked(S.data))
        if counter % tick == 0:
            # The terms the loop's predicate is about to test (read-only).
            self._terms = S.pred.terms(S.sp.build_state(S.model, S.data, S.ix, sync=False))
        if counter > 0 and S.data.qpos[S.ix.base_qpos][2] < self.fall_z:
            self.stats.fell, self.request = True, "fell"
        return self.request is not None

    def draw(self, img, st, S):
        import run_integrated_combined as RIC
        RIC.text(img, "PRACTICE  seed %d  try %d  t %.1f s  rule %s"
                 % (self.seed, self.try_no, S.counter * SIM_DT, S.lock_rule.version),
                 (10, RIC.PANEL_H - 196), 0.6, RIC.CYAN)
        cur = self.stats.summary()
        line = ("this try: " + format_stats(cur)) if cur["locked"] else (
            "last try: " + format_stats(self.last) if self.last else "this try: not locked yet")
        RIC.text(img, line, (10, RIC.PANEL_H - 172), 0.45,
                 RIC.GREEN if cur["locked"] else RIC.GREY)
        RIC.text(img, "t next seed   y same seed   u unlock   q quit", (10, RIC.PANEL_H - 148),
                 0.5, RIC.GREY)
        if not S.overlay_full:
            RIC.draw_lock_status(img, 10, RIC.PANEL_H - 120, st)


_TICK = 20                                   # = g1_data.spec.PHYSICS_STEPS_PER_TICK (tested)


def _fall_z() -> float:
    from g1_data.scripted_demo import DemoConfig
    return float(DemoConfig().fall_z)        # 0.35 m, deploy.py's and the recorder's


def _parse_script(s):
    out = []
    for item in (s or "").split(","):
        if item.strip():
            t, k = item.split(":")
            out.append((float(t), ord(k) if len(k) == 1 else int(k)))
    return out


def practice_parser():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy", action="store_true",
                    help="the old standalone walking test (passive viewer; NOT the "
                         "practice configuration)")
    ap.add_argument("--seed", type=int, default=None,
                    help="first seed (default: the first of data/eval/train_seeds.json); "
                         "training seeds only")
    ap.add_argument("--lock-rule", default=None,
                    help="lock rule name (g1_data.phases.LOCK_RULES; default LockConfig's)")
    ap.add_argument("--guide", choices=OC.GUIDE_CHOICES, default=OC.GUIDE_DEFAULT,
                    help="base-lock floor guide (default %(default)s)")
    ap.add_argument("--headless", action="store_true",
                    help="tests: no window, sim-clock frames; keys only from --script")
    ap.add_argument("--script", default=None, help="headless: keys on the sim clock per try")
    ap.add_argument("--tries", type=int, default=1, help="headless: tries before exiting")
    ap.add_argument("--try-seconds", type=float, default=10.0,
                    help="headless: simulated seconds per try")
    ap.add_argument("--start-standoff", type=float, default=None,
                    help="FIXTURE: start head-on this far behind the box instead of (0.60, 0.00)")
    return ap


def open_practice_session(p, first_seed: int):
    """The teleop session, exactly as tools/record_teleop_live.py opens it
    (`run_integrated_combined.build_parser` with "keyboard <seed>"), with the
    no-camera source and practice flags."""
    import run_integrated_combined as RIC
    ns = RIC.build_parser().parse_args(["keyboard", str(first_seed)])
    ns.headless = bool(p.headless)
    ns.seconds = float(p.try_seconds) if p.headless else float("inf")
    ns.guide = p.guide
    S = RIC.open_session(ns, source=NoCameraSource(fps=None if p.headless else 30.0),
                         sim_clock=bool(p.headless))
    if S is None:
        raise SystemExit("renderer could not be created")
    S.lock_rule = OC.lock_rule(p.lock_rule)
    S.practice = True
    return S


def run_practice(p) -> list:
    """The practice session. Returns one summary dict per try."""
    import run_integrated_combined as RIC
    seeds = OC.training_seeds()
    if not seeds:
        raise SystemExit("no training seeds: data/eval/train_seeds.json is missing")
    first = seeds[0] if p.seed is None else int(p.seed)
    OC.assert_training_seeds([first])
    idx = seeds.index(first)
    S = open_practice_session(p, first)
    print("PRACTICE: lock rule %s, guide %s, start %s, keyboard %s"
          % (S.lock_rule.version, p.guide, OC.START_XY, OC.keyboard_settings(S.loco)))
    history, last = [], None
    try:
        while True:
            seed = seeds[idx % len(seeds)]
            info = RIC.reset_for_episode(S, seed, start_standoff=p.start_standoff)
            if info["start"].heldout:                 # never: the train stream excludes it
                print("seed %d spawns in the held-out patch - skipped" % seed)
                idx += 1
                continue
            hooks = PracticeHooks(S, seed, len(history) + 1, last, _parse_script(p.script))
            reason = RIC.run_loop(S, hooks)
            st = dict(hooks.stats.summary(), seed=seed, try_no=len(history) + 1)
            history.append(st)
            last = st
            print("try %d  seed %d: %s" % (st["try_no"], seed, format_stats(st)))
            if reason == "quit" or (p.headless and len(history) >= p.tries):
                break
            if hooks.request == "next" or (p.headless and hooks.request is None):
                idx += 1
            elif hooks.request == "fell":
                print("  fell - the same seed again")
    except KeyboardInterrupt:
        pass
    finally:
        RIC.close_session(S)
    _print_summary(history)
    return history


def _print_summary(history) -> None:
    if not history:
        return
    locked = [h for h in history if h["locked"]]
    print("\nPRACTICE SUMMARY: %d tries, %d locked" % (len(history), len(locked)))
    if locked:
        for k, unit in (("time_to_lock_s", "s"), ("lat_err_at_lock_m", "m"),
                        ("head_err_at_lock_deg", "deg"), ("overshoot_m", "m")):
            v = np.array([h[k] for h in locked])
            print("  %-22s median %+.3f %s   worst |%.3f|" % (k, np.median(v), unit,
                                                            np.abs(v).max()))


def main(argv=None):
    p = practice_parser().parse_args(argv)
    if p.legacy:
        return legacy_main()
    run_practice(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
