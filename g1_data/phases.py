"""Event-driven base lock (D12), as an observable predicate over the spec state vector.

WHY THE LOCK CANNOT BE A PHASE TRANSITION
-----------------------------------------
The scripted demonstrator used to lock and release the base on membership of a
phase set (`LOCKED_PHASES_WALKING`). Deployment has no phase machine, and
neither does teleoperated collection - the human supplies the trajectory, not a
schedule. If the lock were keypress-triggered during collection and
predicate-triggered at deployment, the demonstrations would show lock timings
the policy never sees.

So the lock fires from quantities that are IN the state vector, on the same
principle as D11's geometric weld gate: something the policy causes and can
observe. Every term below is a `g1_data.spec` accessor plus two fixed platform
constants (Q6/TR13: the platforms never move).

MEASURED, NOT GUESSED
---------------------
Thresholds come from NOTES.md "2026-09-10 Base-lock predicate (D12)", 12
episodes of the walking demonstrator at 25 Hz, all quantities read from the
STEPPED model (TR14). The margins measured at the four transitions were:

    disp        +9.8 mm     fwd (pickup)  +66.7 mm    fwd (goal)  +19.1 mm
    lat        +37.0 mm     head          +2.71 deg   to_goal    +1.278 m
    box_lift    +6.4 mm     |box_lift|    +19.9 mm    palm_far   +77.8 mm

THE SETTLED TEST IS A GAIT-PERIOD TEST
--------------------------------------
"Is the base still?" cannot be answered over an arbitrary window. TR2: the
locomotion policy IS the balance controller and its in-place march never stops.
Measured over 0.52 s (0.65 of a gait period) the march's net displacement stayed
under 10 mm on only ~30% of settled ticks, with runs of up to 19 consecutive
ticks above it. Over one whole `GAIT_PERIOD` the limit cycle cancels and the
same test holds on 98.7%. `GAIT_PERIOD = 0.8` is fixed inside the network (TR4),
so `settle_window = 20` ticks at 25 Hz is a constant, not a tuned value.

Even then the term is not monotone, and TR8 applies (a single-frame stillness
threshold is not a stillness test): the predicate is debounced over 5 consecutive
ticks, and the lock LATCHES. LOCK is only tested while free, RELEASE only while
locked. It is an edge trigger, never a continuously re-evaluated condition.

WHY THE "TASK ALREADY DONE" GUARD IS NOT OPTIONAL
-------------------------------------------------
At the first lock the robot is settled at a standoff from a box at rest with the
hands far away - which is *also* exactly what RELEASE's placed-and-withdrawn
branch describes. And after the final release the box sits at the goal standoff
in front of a settled robot, which is what LOCK's geometry describes. Without
`to_goal` each predicate fires in the other's situation. `to_goal <= 0.10` is the
Q4 placement tolerance, i.e. "is the box where the task wants it" - the one
distinction that separates them. It costs one scene constant.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np
import mujoco

from g1_data import spec


@dataclass(frozen=True)
class PlatformGeometry:
    """The two platforms and the box, resolved from the model by name.

    Fixed by Q6/TR13 - only the box POSITION is randomised, never the platforms -
    so these are legitimate constants for a predicate to reference. Resolved
    rather than hardcoded for the reason ModelIndex exists: a scene edit must
    not silently leave a stale number behind (O3, O22).
    """

    pickup_xy: np.ndarray
    pickup_half: np.ndarray
    goal_xy: np.ndarray
    goal_half: np.ndarray
    top_z: float
    box_half: float

    @property
    def rest_z(self) -> float:
        """Box centre height when the box rests on a platform."""
        return self.top_z + self.box_half

    @classmethod
    def resolve(cls, model, box_geom: str = "box1_geom") -> "PlatformGeometry":
        def plat(geom: str):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            if gid < 0:
                raise ValueError(f"platform geom not found: {geom}")
            bid = int(model.geom_bodyid[gid])
            return (np.array(model.body_pos[bid][:2], dtype=np.float64),
                    np.array(model.geom_size[gid][:2], dtype=np.float64),
                    float(model.body_pos[bid][2] + model.geom_size[gid][2]))

        p_xy, p_half, p_top = plat("platform_pickup_geom")
        g_xy, g_half, g_top = plat("platform_goal_geom")
        if abs(p_top - g_top) > 1e-9:
            raise AssertionError(
                f"the two platform tops differ ({p_top} vs {g_top}); the lock "
                f"predicate and the clearance metric assume one resting height")
        bgid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, box_geom)
        if bgid < 0:
            raise ValueError(f"box geom not found: {box_geom}")
        return cls(pickup_xy=p_xy, pickup_half=p_half, goal_xy=g_xy,
                   goal_half=g_half, top_z=p_top,
                   box_half=float(model.geom_size[bgid][2]))

    def clearance(self, box_pos) -> float:
        """Box bottom above a platform top, or NaN when over neither.

        NaN, not a large number: between the platforms the box is over the
        floor and there is nothing to scrape, which is a different statement
        from "it is high above the platform". A minimum taken over an episode
        must skip those ticks, not be dominated by them.
        """
        b = np.asarray(box_pos, dtype=np.float64)
        for ctr, half in ((self.pickup_xy, self.pickup_half),
                          (self.goal_xy, self.goal_half)):
            if (abs(b[0] - ctr[0]) <= half[0] + self.box_half
                    and abs(b[1] - ctr[1]) <= half[1] + self.box_half):
                return float(b[2] - self.box_half - self.top_z)
        return float("nan")


@dataclass(frozen=True)
class LockConfig:
    """The lock rule. VERSIONED: every episode records the rule it was collected
    under (`lock_rule_meta`), and replay, the phase labeller and deployment use
    that episode's own rule (`lock_config_from_meta`); episodes from before the
    field existed are v1.

      v1  "stance band + stillness" - the 2026-09-10 rule: the stance band below
          AND base XY travel <= 10 mm over one gait period (0.8 s). Kept as
          LOCK_RULE_V1 for every episode recorded under it (scripted_v1).
      v2  "stance band only" - ADOPTED 2026-10-17 (CLAUDE.md section 8), the
          default: fwd in [0.26, 0.40] m, |lat| <= 0.05 m, |head| <= 5 deg for
          `debounce` consecutive ticks; disp_max = inf, so no stillness rule.
          Measured on the 20 spare seeds, real pelvis lock: 19/20 (v1 20/20),
          pickup lock at tick 32-48 (v1 62-92), box pushed before grasp <= 0.5
          mm (v1 1.7), base speed at lock <= 0.26 m/s with pitch <= 3.1 deg in
          the following second (no jolt). The PELVIS HOLD is what prevents the
          pre-grasp push, not the stillness rule: with the lock removed entirely
          the push was 52 mm median (2026-10-17 free-base test).
      v3  "stance band only, carry-release at pickup only" - 2026-10-24 (open
          item 43). v2's "welded box > lift_min above rest -> release the base"
          (carrying: the legs take over) fired at the GOAL lock too: a policy
          that arrives with the box still at carry height was locked for 5
          ticks, then released, walked off and dropped the box 0.14-0.82 m away
          (4/20 episodes each for ACT-160 200k and ACT-LSTM-160 120k). Under v3
          that release applies only to a lock engaged WITHOUT the weld - the
          pickup lock; a lock engaged while welded (the goal lock: a welded
          lock targets the goal) is released only by the placed-and-withdrawn
          test. NOT identical to v2 for every demonstration: 4 of the 200
          scripted_v3 episodes (seeds 6, 14, 146, 206) lock at the goal with
          the box 0.058-0.065 m up and v2 releases them for 5 ticks before
          re-locking; re-recorded under v3, seed 6 diverges from tick 334 and
          still succeeds (placement 0.0418 vs 0.0412 m). Seeds 0-5, 7-9 are
          bit-identical (2026-10-24). NOT the default: the demonstrator and
          teleop still build `LockConfig()` = v2.

    Each threshold has a measured margin at the transition it governs (module
    docstring). `lift_min` is the only one tied to another config value: it is
    a fraction of what the arm ACHIEVES of `DemoConfig.lift_h`, so changing the
    commanded lift means re-measuring the achieved lift and re-checking this.
    `disp` is still computed (diagnostics, the teleop overlay) under v2.
    """

    version: str = "v2"
    settle_window: int = 20        # ticks: 0.80 s = one GAIT_PERIOD (TR4)
    debounce: int = 5              # ticks: 0.20 s (TR8)
    disp_max: float = float("inf")  # v2: no stillness rule (v1: 0.010 m per gait period)
    fwd_lo: float = 0.26           # m, standoff band the arms can serve (O14)
    fwd_hi: float = 0.40
    lat_max: float = 0.05          # m, the head-on band grasping works in (O18)
    head_max_deg: float = 5.0
    lift_min: float = 0.055        # m above resting: carrying, hand to the legs
    rest_tol: float = 0.020        # m: the box is down
    withdraw: float = 0.25         # m, worst palm to box centre: hands clear
    done_radius: float = 0.10      # m, the Q4 placement tolerance
    #: v1/v2: True - the carry release (welded, box above lift_min) applies at
    #: every lock. v3: False - only at a lock engaged without the weld (pickup).
    carry_release_goal: bool = True


#: The lock rules, by name. v2 is the default (`LockConfig()`).
LOCK_RULE_V1 = LockConfig(version="v1", disp_max=0.010)
LOCK_RULE_V2 = LockConfig(version="v2")
LOCK_RULE_V3 = LockConfig(version="v3", carry_release_goal=False)
LOCK_RULES = {"v1": LOCK_RULE_V1, "v2": LOCK_RULE_V2, "v3": LOCK_RULE_V3}
LOCK_RULE_NAMES = {"v1": "stance band + stillness (10 mm / 0.8 s)",
                   "v2": "stance band only",
                   "v3": "stance band only, carry-release at pickup only"}


def lock_rule_meta(cfg: "LockConfig") -> dict:
    """Every LockConfig value, for an episode's metadata. disp_max = inf is
    written as None (JSON null = "no stillness rule"), keeping the file valid
    JSON for tools stricter than Python's."""
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    if d["disp_max"] == float("inf"):
        d["disp_max"] = None
    d["name"] = LOCK_RULE_NAMES.get(cfg.version, "custom")
    return d


def lock_config_from_meta(meta: Optional[dict]) -> "LockConfig":
    """The rule an episode was collected under. No `lock_rule` field = v1, the
    only rule that existed before the field (2026-10-17). A recorded rule whose
    values differ from its named version is refused rather than trusted."""
    meta = meta or {}
    if "lock_rule" not in meta:
        return LOCK_RULE_V1
    rule = meta["lock_rule"]
    if rule is None:
        raise ValueError("episode was collected WITHOUT a lock predicate (lock_rule "
                         "null: the retired phase-set lock) - no rule to apply")
    kw = {f.name: rule[f.name] for f in fields(LockConfig) if f.name in rule}
    if kw.get("disp_max") is None:
        kw["disp_max"] = float("inf")
    cfg = LockConfig(**kw)
    known = LOCK_RULES.get(cfg.version)
    if known is not None and cfg != known:
        raise ValueError("episode lock_rule says %r but its values differ from that "
                         "rule: %r vs %r" % (cfg.version, cfg, known))
    return cfg


class LockPredicate:
    """Two-state latch driven by the spec state. One call per 25 Hz tick.

    Usage, from the control loop::

        pred = LockPredicate(geo)
        ...
        event = pred.update(state)          # "lock" | "release" | None
        if event == "lock":   base_lock.lock(model, data)
        elif event == "release": base_lock.release(model, data, policy)

    `state` must be the spec state vector built from the STEPPED model. The
    predicate never sees a phase, a tick index or an elapsed time, by
    construction: nothing in this class has access to one.
    """

    def __init__(self, geo: PlatformGeometry, cfg: Optional[LockConfig] = None):
        self.geo = geo
        self.cfg = cfg or LockConfig()
        self.reset()

    def reset(self) -> None:
        """Per-episode state. Every field here leaks into the next episode if
        forgotten - see g1_data/reset.py's docstring for why that matters."""
        self.locked = False
        # maxlen == the window, so _hist[0] is exactly `settle_window`
        # ticks back at the moment `terms` reads it (the current tick is
        # appended after). One gait period, not one plus a tick.
        self._hist = deque(maxlen=self.cfg.settle_window)
        self._run = 0
        self.n_lock = 0
        self.n_release = 0
        #: Was the box welded when the CURRENT lock engaged? (v3: a welded lock
        #: is the goal lock, which the carry release does not apply to.)
        self.locked_welded = False

    # ---- terms ---------------------------------------------------------
    def terms(self, state) -> dict:
        """Every quantity the predicate tests, for diagnostics and logging."""
        c, g = self.cfg, self.geo
        box = spec.box_pos(state)
        base = spec.base_pos(state)
        q = spec.base_quat(state)
        yaw = float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                               1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])))
        welded = float(spec.grip_left(state)) >= 0.5
        target = g.goal_xy if welded else box[:2]
        dx, dy = target[0] - base[0], target[1] - base[1]
        cy, sy = np.cos(yaw), np.sin(yaw)
        palm_far = max(
            float(np.linalg.norm(spec.palm_left_pos(state) - box)),
            float(np.linalg.norm(spec.palm_right_pos(state) - box)))
        bearing = float(np.arctan2(dy, dx))
        return dict(
            welded=welded,
            fwd=float(cy * dx + sy * dy),
            lat=float(-sy * dx + cy * dy),
            head=float(np.degrees(np.arctan2(np.sin(bearing - yaw),
                                             np.cos(bearing - yaw)))),
            disp=self._disp(base[:2]),
            box_lift=float(box[2] - g.rest_z),
            to_goal=float(np.linalg.norm(box[:2] - g.goal_xy)),
            palm_far=palm_far,
        )

    def _disp(self, base_xy) -> float:
        """Net base travel over one gait period, or inf before the window fills.

        Infinite, not zero: a "settled" test that is trivially true on the first
        frame of an episode is a bug, and this one would fire before the robot
        had taken a single step.
        """
        if len(self._hist) < self._hist.maxlen:
            return float("inf")
        return float(np.linalg.norm(np.asarray(base_xy) - self._hist[0]))

    def lock_ok(self, t: dict) -> bool:
        c = self.cfg
        if (not t["welded"]) and t["to_goal"] <= c.done_radius:
            return False                    # the task is already finished
        return bool(t["disp"] <= c.disp_max
                    and c.fwd_lo <= t["fwd"] <= c.fwd_hi
                    and abs(t["lat"]) <= c.lat_max
                    and abs(t["head"]) <= c.head_max_deg)

    def release_ok(self, t: dict) -> bool:
        c = self.cfg
        if t["welded"]:
            if self.locked_welded and not c.carry_release_goal:
                return False                              # v3: goal lock holds
            return bool(t["box_lift"] >= c.lift_min)      # carrying: legs take over
        return bool(t["to_goal"] <= c.done_radius
                    and abs(t["box_lift"]) <= c.rest_tol
                    and t["palm_far"] >= c.withdraw)      # placed and withdrawn

    # ---- the one call the control loop makes ---------------------------
    def update(self, state) -> Optional[str]:
        """Advance one 25 Hz tick. Returns "lock", "release" or None."""
        t = self.terms(state)
        self._hist.append(np.asarray(spec.base_pos(state)[:2], dtype=np.float64))
        want = self.release_ok(t) if self.locked else self.lock_ok(t)
        self._run = self._run + 1 if want else 0
        if self._run < self.cfg.debounce:
            return None
        self._run = 0
        self.locked = not self.locked
        if self.locked:
            self.n_lock += 1
            self.locked_welded = bool(t["welded"])
            return "lock"
        self.n_release += 1
        return "release"
