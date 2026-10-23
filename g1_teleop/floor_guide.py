"""Base-lock floor guide: where to stand for the lock rule, drawn on the floor.

PURELY VISUAL. Everything here is an `MjvGeom` appended to a render scene
(`mujoco.Renderer.scene`, or a passive viewer's `user_scn`) AFTER
`mjv_updateScene`: it is not in the MjModel, so it has no body, no mass, no
collision and no contact, and it never reads or writes `MjData` beyond the
lock terms the loop already computed. It cannot reach the recorded state
(g1_data.spec.SpecLayout.build_state reads the model only). Proven by
tools/test_operator_tools.py: guide on vs off -> identical trajectories and
identical recorded states, step for step.

WHAT IT SHOWS
-------------
The lock rule (`g1_data.phases.LockConfig`, the ACTIVE rule - never a copy of
its numbers) locks the base when, for `debounce` ticks, the TARGET (the box; the
goal platform while welded) sits in the robot's own frame at

    fwd in [fwd_lo, fwd_hi]   |lat| <= lat_max   |head| <= head_max_deg

`LockPredicate.terms` measures fwd/lat/head in the frame of the robot's CURRENT
heading, so the set of base positions that satisfy fwd and lat is a rectangle
behind the target, aligned with the robot's heading - it swings around the
target as the robot turns, which is the truth of the rule, not a display
choice. Heading is a wedge: |head| <= 5 deg means the base must lie within 5
deg of the robot's backward axis through the target.

    band (fill)          red = no condition met, yellow = one or two,
                         green = all three (the lock engages after the debounce)
    near / far edges     green when fwd is in the band, else red
    side edges           green when |lat| is within the limit, else red
    wedge rays           green when |head| is within the limit, else red
    heading arrow        from the base along the robot's facing; same colour
                         as the wedge
    LOCKED               solid green band and a "LOCKED" label/banner while the
                         pelvis weld is engaged

Nothing is drawn while the task is already done (box at the goal, not welded:
the rule refuses to lock there) or before the first lock tick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

RED = (0.92, 0.16, 0.10)
YELLOW = (1.00, 0.80, 0.00)
GREEN = (0.10, 0.85, 0.25)
GREY = (0.70, 0.70, 0.70)

FLOOR_Z = 0.004          # m: just above the floor plane
BAR_W = 0.012            # m: edge / ray width
BAR_H = 0.002            # m: half-height of a floor marking
ARROW_LEN = 0.35         # m
RAY_EXTRA = 0.08         # m: wedge rays run this far past the band's far edge


@dataclass(frozen=True)
class GuideStatus:
    """Which lock conditions hold now, and the colour that follows."""
    visible: bool
    locked: bool
    fwd_ok: bool = False
    lat_ok: bool = False
    head_ok: bool = False

    @property
    def n_ok(self) -> int:
        return int(self.fwd_ok) + int(self.lat_ok) + int(self.head_ok)

    @property
    def colour(self) -> tuple:
        if self.locked or self.n_ok == 3:
            return GREEN
        return YELLOW if self.n_ok else RED

    @property
    def colour_name(self) -> str:
        return {GREEN: "green", YELLOW: "yellow", RED: "red"}[self.colour]


def status(terms: dict, cfg, locked: bool) -> GuideStatus:
    """The guide's state from `LockPredicate.terms` and the ACTIVE LockConfig."""
    if not terms:
        return GuideStatus(visible=False, locked=bool(locked))
    if not locked and (not terms["welded"]) and terms["to_goal"] <= cfg.done_radius:
        return GuideStatus(visible=False, locked=False)        # task done: no lock
    return GuideStatus(
        visible=True, locked=bool(locked),
        fwd_ok=bool(cfg.fwd_lo <= terms["fwd"] <= cfg.fwd_hi),
        lat_ok=bool(abs(terms["lat"]) <= cfg.lat_max),
        head_ok=bool(abs(terms["head"]) <= cfg.head_max_deg))


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def geometry(terms: dict, cfg, base_xy, yaw: float, locked: bool) -> List[dict]:
    """The guide as primitives in WORLD coordinates (no MuJoCo objects):

      dict(kind="bar",   a=xy, b=xy, rgb, name)     a floor strip from a to b
      dict(kind="fill",  center=xy, half=(u, v), yaw, rgb, name, label)
      dict(kind="arrow", a=xy, b=xy, rgb, name)

    Empty when the guide is not visible."""
    st = status(terms, cfg, locked)
    if not st.visible:
        return []
    R = _rot(yaw)
    base = np.asarray(base_xy, dtype=np.float64)[:2]
    target = base + R @ np.array([terms["fwd"], terms["lat"]])

    def w(u, v):                     # robot-frame offset from the target -> world
        return target + R @ np.array([u, v])

    lo, hi, lm = cfg.fwd_lo, cfg.fwd_hi, cfg.lat_max
    c_fwd = GREEN if (st.fwd_ok or locked) else RED
    c_lat = GREEN if (st.lat_ok or locked) else RED
    c_head = GREEN if (st.head_ok or locked) else RED
    out = [dict(kind="fill", center=w(-(lo + hi) / 2.0, 0.0), half=((hi - lo) / 2.0, lm),
                yaw=float(yaw), rgb=st.colour, name="band",
                label="LOCKED" if locked else "")]
    out += [dict(kind="bar", a=w(-lo, -lm), b=w(-lo, lm), rgb=c_fwd, name="near_edge"),
            dict(kind="bar", a=w(-hi, -lm), b=w(-hi, lm), rgb=c_fwd, name="far_edge"),
            dict(kind="bar", a=w(-lo, lm), b=w(-hi, lm), rgb=c_lat, name="left_edge"),
            dict(kind="bar", a=w(-lo, -lm), b=w(-hi, -lm), rgb=c_lat, name="right_edge")]
    h = np.radians(cfg.head_max_deg)
    L = hi + RAY_EXTRA
    for sgn, nm in ((+1, "wedge_left"), (-1, "wedge_right")):
        out.append(dict(kind="bar", a=target, b=w(-L * np.cos(h), sgn * L * np.sin(h)),
                        rgb=c_head, name=nm))
    out.append(dict(kind="arrow", a=base, b=base + R @ np.array([ARROW_LEN, 0.0]),
                    rgb=c_head, name="heading_arrow"))
    return out


def _add(scn, gtype, size, pos, mat, rgba, label: str = "") -> bool:
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64), np.asarray(pos, np.float64),
                        np.asarray(mat, np.float64).reshape(9), np.asarray(rgba, np.float32))
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    if label:
        g.label = label
    scn.ngeom += 1
    return True


def _zmat(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def add_to_scene(scn, prims: List[dict], alpha_fill: float = 0.35,
                 alpha_line: float = 0.9) -> int:
    """Append the primitives to an MjvScene. Returns how many geoms were added."""
    import mujoco
    n0 = scn.ngeom
    for p in prims:
        if p["kind"] == "fill":
            a = 0.6 if p["label"] else alpha_fill
            _add(scn, mujoco.mjtGeom.mjGEOM_BOX, (p["half"][0], p["half"][1], BAR_H / 2),
                 (p["center"][0], p["center"][1], FLOOR_Z - BAR_H / 2), _zmat(p["yaw"]),
                 (*p["rgb"], a), p["label"])
        elif p["kind"] == "bar":
            a, b = np.asarray(p["a"]), np.asarray(p["b"])
            d = b - a
            length = float(np.linalg.norm(d))
            if length < 1e-9:
                continue
            mid = (a + b) / 2.0
            _add(scn, mujoco.mjtGeom.mjGEOM_BOX, (length / 2, BAR_W / 2, BAR_H),
                 (mid[0], mid[1], FLOOR_Z), _zmat(float(np.arctan2(d[1], d[0]))),
                 (*p["rgb"], alpha_line))
        elif p["kind"] == "arrow":
            if scn.ngeom >= scn.maxgeom:
                continue
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.eye(3).reshape(9), np.asarray((*p["rgb"], alpha_line), np.float32))
            z = FLOOR_Z + 0.02
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02,
                                 np.array([p["a"][0], p["a"][1], z]),
                                 np.array([p["b"][0], p["b"][1], z]))
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            scn.ngeom += 1
    return scn.ngeom - n0


def _yaw(quat) -> float:
    qw, qx, qy, qz = quat
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


class FloorGuide:
    """The guide as the teleop loop and walk_test use it. Stateless apart from
    `last` (what the most recent draw showed, for the HUD and the tests)."""

    def __init__(self):
        self.last: Optional[GuideStatus] = None
        self.last_ngeom = 0

    def draw(self, scn, terms: dict, cfg, data, ix, locked: bool) -> int:
        """Append the guide to `scn` (call after mjv_updateScene / update_scene).
        Reads the base pose from `data`; writes nothing to it."""
        base = np.asarray(data.qpos[ix.base_qpos][:2], dtype=np.float64)
        yaw = _yaw(np.asarray(data.qpos[ix.base_quat_qpos], dtype=np.float64))
        self.last = status(terms, cfg, locked)
        self.last_ngeom = add_to_scene(scn, geometry(terms, cfg, base, yaw, locked))
        return self.last_ngeom

    def banner(self, img, x: int, y: int) -> None:
        """A large "LOCKED" on the simulation panel while the lock is engaged."""
        if self.last is None or not self.last.locked:
            return
        import cv2
        for col, th in (((0, 0, 0), 6), ((0, 220, 0), 2)):
            cv2.putText(img, "LOCKED", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, th, cv2.LINE_AA)
