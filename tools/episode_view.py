"""Watch and film a scripted episode: passive viewer, real-time pacing, MP4.

Imported by tools/record_episodes.py (--viewer, --realtime, --video-dir).

OBSERVATION ONLY, same mechanism as `g1_data.recorder.ScriptedRecorder`: a
wrapper around `mujoco.mj_step` that acts only on steps taken from inside
`scripted_demo.run_episode`. `scripted_demo.py` is not modified.

NESTING ORDER MATTERS. Enter this BEFORE the ScriptedRecorder:

    with EpisodeView(...) as view, ScriptedRecorder() as rec:
        SD.run_episode(...)

The recorder identifies demonstrator steps by its IMMEDIATE caller being
`run_episode` (recorder.py:436-442), so it must be the outermost wrapper. This
one walks up the stack to find `run_episode` instead, so it works underneath.

WHAT THIS MAY AND MAY NOT TOUCH
-------------------------------
  * Rendering reads `data` (`Renderer.update_scene`); pacing only sleeps.
  * The passive viewer's `sync()` WRITES mouse perturbations into
    `data.xfrc_applied`. Any non-zero value aborts the episode with
    `ViewerPerturbation` before the step that would use it. The UI panels are
    hidden (`show_left_ui/right_ui=False`) so no slider can edit ctrl or
    options, and a jump in `data.time` (a viewer-side reset) also aborts.
  * Proof that a watched run is the same episode as a headless one: compare
    the recorded arrays' sha256 (record_episodes.py `--invariance` style); the
    2026-10-17 check is in the commit that added this file.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Optional

import cv2
import mujoco
import numpy as np

from g1_data import spec

TICK = spec.PHYSICS_STEPS_PER_TICK          # 20 physics steps = 25 Hz
VIDEO_W, VIDEO_H, VIDEO_FPS = 1280, 720, 25  # one frame per recorded tick
# One fixed view showing the start point (0.6, 0), the pickup platform (1.5, 0)
# and the goal platform (1.5, -1.5), from the +y side so the hands at the box
# are not hidden behind it. Chosen from rendered candidates, 2026-10-17.
CAMERA = dict(lookat=(1.05, -0.65, 0.3), distance=4.3, elevation=-45.0,
              azimuth=240.0)


class ViewerPerturbation(RuntimeError):
    """The viewer changed the physics (mouse perturbation or reset)."""


class _VideoSink:
    """Overlay + encode on a background thread.

    Measured 2026-10-17 (RTX 3050 laptop): render+encode in the physics loop
    cost ~15 ms per tick, which with viewer.sync (~7.5 ms) put a 28 s episode at
    39 s of wall time - too slow for --realtime. Only `Renderer.render()` reads
    MjData, so only it stays in the loop; the frame it returns is a fresh array,
    so the thread never sees the simulation state.
    """

    def __init__(self, path: str):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      VIDEO_FPS, (VIDEO_W, VIDEO_H))
        if not self.writer.isOpened():
            raise RuntimeError("could not open video writer at %s" % path)
        self.q: "queue.Queue" = queue.Queue(maxsize=256)
        self.error = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def put(self, rgb: np.ndarray, text: str) -> None:
        if self.error is not None:
            raise RuntimeError("video writer failed: %r" % self.error)
        self.q.put((rgb, text))

    def _run(self) -> None:
        try:
            while True:
                item = self.q.get()
                if item is None:
                    return
                rgb, text = item
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(img, (0, 0), (VIDEO_W, 44), (0, 0, 0), -1)
                cv2.putText(img, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2, cv2.LINE_AA)
                self.writer.write(img)
        except Exception as e:                              # noqa: BLE001
            self.error = e

    def release(self) -> None:
        self.q.put(None)
        self.t.join()
        self.writer.release()
        if self.error is not None:
            raise RuntimeError("video writer failed: %r" % self.error)


def _free_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAMERA["lookat"]
    cam.distance = CAMERA["distance"]
    cam.elevation = CAMERA["elevation"]
    cam.azimuth = CAMERA["azimuth"]
    return cam


def _run_episode_frame(depth: int = 4):
    f = sys._getframe(2)
    for _ in range(depth):
        if f is None:
            return None
        if f.f_code.co_name == "run_episode":
            return f
        f = f.f_back
    return None


class EpisodeView:
    """Context manager around ONE `run_episode` call."""

    def __init__(self, seed: int, viewer: bool = False, realtime: bool = False,
                 video_dir: Optional[str] = None):
        self.seed = int(seed)
        self.want_viewer = viewer
        self.realtime = realtime
        self.video_dir = video_dir
        self.video_path = None          # final path, set by finalize()
        self.n_frames = 0
        self._partial = None
        self._started = False
        self._renderer = self._writer = self._viewer = None

    # ---- attachment ----------------------------------------------------
    def __enter__(self) -> "EpisodeView":
        self._real_step = mujoco.mj_step
        view = self

        def step_wrapper(m, d, *a, **k):
            f = _run_episode_frame()
            if f is None:
                return view._real_step(m, d, *a, **k)
            # run_episode calls mj_step exactly once per loop index
            # (scripted_demo.py:1097), so the index is counted here and the
            # frame's locals - a full dict copy - are read only on tick steps.
            if not view._started:
                L = f.f_locals
                i = int(L["i"])
                view._start(m, d, L, i)
                view._n = 0
            i = view._i0 + view._n
            view._n += 1
            view._guard(d, i)
            if i % TICK == 0:
                L = f.f_locals
                if int(L["i"]) != i:
                    raise RuntimeError("episode_view lost count: run_episode i=%d, "
                                       "counted %d" % (int(L["i"]), i))
                if view._writer is not None:
                    view._frame(m, d, L, i)
            out = view._real_step(m, d, *a, **k)
            if (i + 1) % TICK == 0:
                if view._viewer is not None and view._viewer.is_running():
                    view._viewer.sync()
                if view.realtime:
                    lag = view._wall0 + (i + 1 - view._i0) * view._dt - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
            return out

        mujoco.mj_step = step_wrapper
        return self

    def __exit__(self, *exc) -> None:
        mujoco.mj_step = self._real_step
        if self._writer is not None:
            self._writer.release()
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()

    # ---- per episode ---------------------------------------------------
    def _start(self, m, d, L, i) -> None:
        self._started = True
        self._dt = float(m.opt.timestep)
        self._i0, self._t0 = i, float(d.time)
        self._wall0 = time.perf_counter()
        self._weld_eq = L["weld"].eq_id
        self._lock_eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "base_lock")
        self._cam = _free_camera()
        if self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)
            self._partial = os.path.join(self.video_dir,
                                         "ep_seed%04d.partial.mp4" % self.seed)
            self._renderer = mujoco.Renderer(m, height=VIDEO_H, width=VIDEO_W)
            self._writer = _VideoSink(self._partial)
        if self.want_viewer:
            from mujoco import viewer as mj_viewer
            self._viewer = mj_viewer.launch_passive(
                m, d, show_left_ui=False, show_right_ui=False)
            with self._viewer.lock():
                self._viewer.cam.lookat[:] = CAMERA["lookat"]
                self._viewer.cam.distance = CAMERA["distance"]
                self._viewer.cam.elevation = CAMERA["elevation"]
                self._viewer.cam.azimuth = CAMERA["azimuth"]

    def _guard(self, d, i) -> None:
        """Before the step: the viewer must not have changed the physics."""
        if np.any(d.xfrc_applied != 0.0):
            raise ViewerPerturbation(
                "xfrc_applied non-zero at tick %d (mouse perturbation in the "
                "viewer) - episode failed" % (i // TICK))
        expect = self._t0 + (i - self._i0) * self._dt
        if abs(float(d.time) - expect) > 1e-6:
            raise ViewerPerturbation(
                "data.time %.6f, expected %.6f at tick %d (viewer reset?) - "
                "episode failed" % (float(d.time), expect, i // TICK))

    def _frame(self, m, d, L, i) -> None:
        self._renderer.update_scene(d, camera=self._cam)
        rgb = self._renderer.render()           # a new array each call
        phase = L.get("phase")
        phase = getattr(phase, "name", str(phase))
        weld = "ON" if d.eq_active[self._weld_eq] else "off"
        lock = "ON" if (self._lock_eq >= 0 and d.eq_active[self._lock_eq]) else "off"
        text = ("seed %d   tick %d   phase %s   weld %s   base lock %s"
                % (self.seed, i // TICK, phase, weld, lock))
        self._writer.put(rgb, text)
        self.n_frames += 1

    # ---- after the decision ------------------------------------------
    def finalize(self, accepted: bool) -> Optional[str]:
        """Name the video by outcome: ep_seedNNNN.mp4 or ep_seedNNNN_FAIL.mp4."""
        if not self._partial or not os.path.exists(self._partial):
            return None
        name = "ep_seed%04d%s.mp4" % (self.seed, "" if accepted else "_FAIL")
        self.video_path = os.path.join(self.video_dir, name)
        os.replace(self._partial, self.video_path)
        return self.video_path
