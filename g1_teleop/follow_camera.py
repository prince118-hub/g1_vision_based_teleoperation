"""Follow camera for the MuJoCo view: the view stays on the robot's work area.

RENDER-ONLY. A `FollowCamera` reads the base position and heading it is handed
and writes four fields of an `MjvCamera` (lookat, azimuth, elevation, distance).
It never touches `MjModel` or `MjData`, so it cannot reach the physics, the
recorded state, the spec or the reset fingerprint. Proven by
tools/test_operator_tools.py (follow vs fixed: identical trajectories and
identical recorded states, step for step).

WHAT IT FRAMES
--------------
The camera aims at a point AHEAD of the pelvis at box/hand height - where the
work happens - not at the pelvis, and it turns with the robot's heading:

    aim       pelvis xy + AIM_FORWARD_M along the heading, z = AIM_HEIGHT_M
    behind    (default) behind the robot, raised 55 deg, 25 deg to its left:
              over the left shoulder, so the torso does not cover the box.
              55 and not 35 deg: rendered at the grasp of a scripted episode,
              at 35-50 deg the RIGHT hand is hidden behind the box it holds
              (2026-11-13, recordings/view_check)
    front-side, side, top   the other presets (key `c` cycles them)

Heading and position are low-pass filtered (the gait sways the pelvis and
wobbles its yaw several times a second; an unfiltered follow shakes the view).
The filter state is display state only.

The operator's a/d (rotate), w/s (tilt), +/- (zoom) keys are OFFSETS on top of
the preset, so the view keeps following after an adjustment; `r` returns to the
default view (preset `behind`, no offsets). The preset and the offsets in use
are stored per episode (`operator_ui.view`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: The aim point, in the robot's heading frame.
AIM_FORWARD_M = 0.45     # ahead of the pelvis
AIM_HEIGHT_M = 0.85      # world z: box / hand height at the platforms


@dataclass(frozen=True)
class ViewPreset:
    """A view relative to the robot's heading. `azimuth_deg` is added to the
    heading to give MuJoCo's camera azimuth (the direction the camera LOOKS
    along): 0 = from directly behind, negative = the camera moves to the
    robot's left. `elevation_deg` is MuJoCo's (negative = looking down)."""
    name: str
    azimuth_deg: float
    elevation_deg: float
    distance_m: float


#: Cycle order. `behind` is the default view.
PRESETS: Tuple[ViewPreset, ...] = (
    ViewPreset("behind", -25.0, -55.0, 1.50),
    ViewPreset("front-side", -140.0, -25.0, 1.90),
    ViewPreset("side", -90.0, -15.0, 1.90),
    ViewPreset("top", 0.0, -89.0, 2.40),
)
PRESET_NAMES = tuple(p.name for p in PRESETS)
DEFAULT_PRESET = PRESETS[0].name

#: Key steps (the loop's a/d, w/s, +/- keys) and limits.
AZIMUTH_STEP_DEG = 5.0
ELEVATION_STEP_DEG = 5.0
DISTANCE_STEP_M = 0.15
ELEVATION_LIMIT_DEG = 89.0
DISTANCE_MIN_M = 0.5
DISTANCE_MAX_M = 6.0

#: Low-pass factor per display frame (25 Hz nominal): ~0.25 s to settle.
SMOOTHING = 0.15

MODES = ("follow", "fixed")
MODE_DEFAULT = "follow"

#: Keys this adds to the loop (run_integrated_combined.handle_key). `r` was
#: already the loop's "reset view" key and keeps that meaning.
KEY_CYCLE_PRESET = ord("c")
KEY_RESET_VIEW = ord("r")


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class FollowCamera:
    """Preset + manual offsets + a smoothed base pose -> an MjvCamera."""

    def __init__(self, preset: str = DEFAULT_PRESET):
        self.index = PRESET_NAMES.index(preset)
        self.d_azimuth = 0.0
        self.d_elevation = 0.0
        self.d_distance = 0.0
        self._xy: Optional[Tuple[float, float]] = None      # smoothed base xy
        self._yaw_deg: Optional[float] = None               # smoothed heading
        #: (sim s, what, preset, offsets) for every change since `begin_episode`.
        self.changes: List[dict] = []

    # ── state ────────────────────────────────────────────────────────────
    @property
    def preset(self) -> ViewPreset:
        return PRESETS[self.index]

    def offsets(self) -> Dict[str, float]:
        return dict(azimuth_deg=float(self.d_azimuth), elevation_deg=float(self.d_elevation),
                    distance_m=float(self.d_distance))

    def has_offsets(self) -> bool:
        return bool(self.d_azimuth or self.d_elevation or self.d_distance)

    def view(self) -> Dict[str, float]:
        """The view in force: preset plus offsets, clamped."""
        p = self.preset
        return dict(
            azimuth_deg=_wrap_deg(p.azimuth_deg + self.d_azimuth),
            elevation_deg=max(-ELEVATION_LIMIT_DEG,
                              min(ELEVATION_LIMIT_DEG, p.elevation_deg + self.d_elevation)),
            distance_m=max(DISTANCE_MIN_M, min(DISTANCE_MAX_M, p.distance_m + self.d_distance)))

    def label(self) -> str:
        """HUD / header text: 'behind' or 'behind  az -10 el +5 dist +0.15'."""
        s = self.preset.name
        if self.has_offsets():
            s += "  az %+.0f el %+.0f dist %+.2f" % (self.d_azimuth, self.d_elevation,
                                                    self.d_distance)
        return s

    # ── operator input ───────────────────────────────────────────────────
    def _note(self, t_s: float, what: str) -> None:
        self.changes.append(dict(t_s=round(float(t_s), 3), what=what,
                                 preset=self.preset.name, offsets=self.offsets()))

    def rotate(self, steps: int, t_s: float = 0.0) -> None:
        self.d_azimuth = _wrap_deg(self.d_azimuth + steps * AZIMUTH_STEP_DEG)
        self._note(t_s, "rotate")

    def tilt(self, steps: int, t_s: float = 0.0) -> None:
        """+1 = the loop's `w` (camera elevation up, as before), -1 = `s`."""
        p = self.preset
        e = p.elevation_deg + self.d_elevation + steps * ELEVATION_STEP_DEG
        e = max(-ELEVATION_LIMIT_DEG, min(ELEVATION_LIMIT_DEG, e))
        self.d_elevation = round(e - p.elevation_deg, 6)
        self._note(t_s, "tilt")

    def zoom(self, steps: int, t_s: float = 0.0) -> None:
        """+1 = closer (the loop's `+`), -1 = further (`-`)."""
        p = self.preset
        d = p.distance_m + self.d_distance - steps * DISTANCE_STEP_M
        d = max(DISTANCE_MIN_M, min(DISTANCE_MAX_M, d))
        self.d_distance = round(d - p.distance_m, 6)
        self._note(t_s, "zoom")

    def reset_view(self, t_s: float = 0.0) -> None:
        """The default view: preset `behind`, no offsets."""
        self.index = PRESET_NAMES.index(DEFAULT_PRESET)
        self.d_azimuth = self.d_elevation = self.d_distance = 0.0
        self._note(t_s, "reset")

    def cycle(self, t_s: float = 0.0) -> str:
        """Next preset; the manual offsets are cleared (they belong to a view)."""
        self.index = (self.index + 1) % len(PRESETS)
        self.d_azimuth = self.d_elevation = self.d_distance = 0.0
        self._note(t_s, "preset")
        return self.preset.name

    # ── per episode ──────────────────────────────────────────────────────
    def begin_episode(self) -> None:
        """A new episode: the change log starts empty and the filter snaps to
        the reset pose. The preset and offsets the operator chose are kept."""
        self.changes = []
        self.snap()

    def snap(self) -> None:
        """Forget the smoothed pose: the next `apply` jumps to the robot."""
        self._xy, self._yaw_deg = None, None

    def meta(self) -> dict:
        """What an episode stores (`operator_ui.view`)."""
        return dict(mode="follow", preset=self.preset.name, offsets=self.offsets(),
                    view=self.view(), default_preset=DEFAULT_PRESET,
                    aim=dict(forward_m=AIM_FORWARD_M, height_m=AIM_HEIGHT_M),
                    changes=list(self.changes))

    # ── the render side ──────────────────────────────────────────────────
    def apply(self, cam, base_xy, yaw_rad: float, smoothing: float = SMOOTHING) -> None:
        """Point `cam` (an MjvCamera, free type) for a base at `base_xy`
        heading `yaw_rad`. Writes `cam` only."""
        x, y = float(base_xy[0]), float(base_xy[1])
        yaw = math.degrees(float(yaw_rad))
        if self._xy is None or self._yaw_deg is None:
            self._xy, self._yaw_deg = (x, y), yaw
        else:
            a = float(smoothing)
            self._xy = (self._xy[0] + a * (x - self._xy[0]), self._xy[1] + a * (y - self._xy[1]))
            self._yaw_deg = _wrap_deg(self._yaw_deg + a * _wrap_deg(yaw - self._yaw_deg))
        h = math.radians(self._yaw_deg)
        v = self.view()
        cam.lookat[0] = self._xy[0] + AIM_FORWARD_M * math.cos(h)
        cam.lookat[1] = self._xy[1] + AIM_FORWARD_M * math.sin(h)
        cam.lookat[2] = AIM_HEIGHT_M
        cam.azimuth = _wrap_deg(self._yaw_deg + v["azimuth_deg"])
        cam.elevation = v["elevation_deg"]
        cam.distance = v["distance_m"]


def fixed_meta(cam) -> dict:
    """`operator_ui.view` for the old fixed camera (`--camera fixed`)."""
    return dict(mode="fixed", preset=None,
                view=dict(azimuth_deg=float(cam.azimuth), elevation_deg=float(cam.elevation),
                          distance_m=float(cam.distance),
                          lookat=[float(v) for v in cam.lookat]))
