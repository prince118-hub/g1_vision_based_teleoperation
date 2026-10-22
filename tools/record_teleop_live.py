"""Live teleoperated demonstration recorder: ZED 2i -> the teleop loop -> one .npz per episode.

    # the real collection (Windows laptop, ZED 2i plugged in):
    python tools/record_teleop_live.py --source zed --operator charles --count 10

    # dry run on any machine, no camera, no pyzed (a take from tools/record_keypoints.py,
    # or a synthetic one from --make-fake-take):
    python tools/record_teleop_live.py --source fake --replay recordings/keypoints/reach_01_x.npz
    python tools/record_teleop_live.py --make-fake-take recordings/keypoints/fake_grasp_s0.npz \\
        --seed-list 0 --start-standoff 0.32
    python tools/record_teleop_live.py --source fake --replay recordings/keypoints/fake_grasp_s0.npz \\
        --seed-list 0 --start-standoff 0.32 --headless --script 9.6:g,17:e

THE LOOP IS run_integrated_combined.run_loop, UNCHANGED
-------------------------------------------------------
ZED (BODY_38, grabbed on a background thread, TR10) -> One-Euro -> scaling -> IK
arms (waist pinned); keyboard -> LSTM legs; 500 Hz physics; the lock predicate
and the weld gate at 25 Hz - exactly the loop `run_integrated_combined.py`
runs, opened once per session (`open_session`: one camera, one model, one
policy) and run once per episode (`run_loop`). `--source fake` swaps ONLY the
camera: everything after `.grab()` is the same code. Fake mode never imports
pyzed.

Recording is OBSERVATION ONLY: `TeleopRecorder` wraps `mj_step` and reads the
loop's frame, as it does for the headless fixture. The episode hooks
(`LoopHooks`) only read state, end the loop BEFORE a step, block the `b` key while
recording and draw the HUD. `--invariance` and tools/test_record_teleop_live.py
measure record vs no-record step by step.

ONE EPISODE PER SEED
--------------------
Seeds come from the pre-partitioned stream (`--seeds`, default
data/eval/train_seeds.json) minus those the out-dir's ledger has accepted. Per
episode: the shared `reset_episode` (seeded box, LSTM/filters/keyboard reset,
contact contract checked) and the reset fingerprint, then the base is placed at
the deploy/scripted start (0.60, 0.00) - or `--start-standoff` head-on, a fixture.

  idle (physics paused)  ENTER start   n skip this seed   b B-prime   q quit
  recording              e stop+save   x discard (writes nothing)   q quit (saves)
                         and every key of run_integrated_combined.py, except
                         b (D18), which is BLOCKED while recording and logged.

TERMINATION = deploy.py's: a fall (pelvis z < 0.35 m, checked every physics
step) ends at once; success ends it early only once the weld has engaged AND
released and `success.evaluate` keeps passing for 2.5 s; 900 ticks is the cap.
`e` (or q) ends it where it is. Then `success.evaluate` scores it:
  succeeded -> <out-dir>/ep_seedNNNN_<stamp>.npz, ledger ACCEPT
  failed    -> <out-dir>/failed/..., ledger DISCARD (re-issuable); the loaders
               refuse it anyway (dataset.assert_successful)
  discarded -> nothing written, ledger DISCARD
Raw keypoints go to <out-dir>/keypoints/<episode>_kp.npz in the
tools/record_keypoints.py format (replayable with --source fake); per tick
`frame_seq` indexes them. Out-dirs are under recordings/ (gitignored).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import threading
import time
import types
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_integrated_combined as RIC                                # noqa: E402
import deploy as D                                                  # noqa: E402
from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data import success as SU                                   # noqa: E402
from g1_data.ledger import EpisodeLedger, seed_stream               # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402
from g1_data.phases import LOCK_RULES, LockConfig, lock_config_from_meta  # noqa: E402
from g1_data.recorder import (NS_COLLECTION, NS_TELEOP_FAKE,       # noqa: E402
                              SOURCE_TELEOP_LIVE, SOURCE_TELEOP_LIVE_FAKE,
                              TeleopRecorder, assert_namespace, label_of,
                              ledger_for)
from g1_data.reset import reset_episode, state_fingerprint          # noqa: E402
from g1_teleop import contact_contract as CC                        # noqa: E402
from g1_teleop import synthetic_source as SS                        # noqa: E402

#: Termination, shared with deploy.py so the two can never disagree.
MAX_TICKS = D.MAX_TICKS                    # 900
FALL_Z = D.FALL_Z                          # 0.35 m
SUCCESS_HOLD_TICKS = D.SUCCESS_HOLD_TICKS  # 63 ticks = 2.5 s
START_XY = D.START_XY                      # (0.60, 0.00)
TICK = spec.PHYSICS_STEPS_PER_TICK
TICK_S = TICK * RIC.SIM_DT                 # 0.04 s
#: A tick interval longer than this (wall clock) counts as an overrun: the loop
#: fell more than half a tick behind the 25 Hz schedule.
OVERRUN_S = 1.5 * TICK_S
KEY_ENTER = (10, 13)
KP_DIR, FAILED_DIR = "keypoints", "failed"


# ─── the camera side ─────────────────────────────────────────────────────────
class ReplaySource:
    """`--source fake`: a recorded keypoint take played into the pipeline.

    `SyntheticSource.from_recording(mode="zed")` is exactly the frame sequence
    `ZEDSource.grab` handed the controller live. `fps` paces it to wall clock
    for the threaded grabber; past the end it reports a failed grab (None),
    sleeping one frame so the grabber thread does not spin on the GIL."""

    def __init__(self, path: str, fps: Optional[float]):
        self.path = path
        self.rec = SS.load_recording(path)
        self.inner = SS.SyntheticSource.from_recording(self.rec, mode="zed")
        self.inner.fps = fps
        self.fps = fps
        self._lock = threading.Lock()

    def grab(self):
        with self._lock:
            f = self.inner.grab()
        if f is None and self.fps:
            time.sleep(1.0 / self.fps)
        return f

    def rewind(self) -> None:
        with self._lock:
            self.inner.i = 0
            self.inner._t0 = None

    def close(self) -> None:
        pass


class AuditTap:
    """Wraps the source; while armed, keeps every `grab()` it passes on.

    Rows are in the tools/record_keypoints.py format so the episode's own camera
    stream can be replayed with `--source fake`. ZEDSource.grab collapses
    not_new / no_body / rejected-body into one EMPTY frame, so a tap row can only
    say ok, no_body (empty) or grab_failed (None); `n_bodies` is -1 (unknown).
    """

    def __init__(self, source):
        self.source = source
        self._lock = threading.Lock()
        self._rows = None
        self.seq_at_arm = 0

    def grab(self):
        f = self.source.grab()
        t = time.monotonic_ns()
        with self._lock:
            if self._rows is not None:
                self._rows.append((t, f))
        return f

    def arm(self, seq_now: int) -> None:
        with self._lock:
            self._rows, self.seq_at_arm = [], int(seq_now)

    def disarm(self):
        with self._lock:
            rows, self._rows = self._rows, None
        return rows or []

    def close(self) -> None:
        self.source.close()

    @staticmethod
    def to_recording(rows, meta) -> SS.Recording:
        n = len(rows)
        kp = np.full((n, SS.N_KEYPOINTS, 3), np.nan)
        cf = np.full((n, SS.N_KEYPOINTS), np.nan)
        status = np.zeros(n, np.int8)
        for i, (_, f) in enumerate(rows):
            if f is None:
                status[i] = SS.STATUS_GRAB_FAILED
            elif not f.keypoints_3d:
                status[i] = SS.STATUS_NO_BODY
            else:
                status[i] = SS.STATUS_OK
                kp[i] = np.asarray(f.keypoints_3d, dtype=np.float64).reshape(SS.N_KEYPOINTS, 3)
                cf[i] = np.asarray(f.confidences, dtype=np.float64).reshape(SS.N_KEYPOINTS)
        return SS.Recording(
            keypoints=kp, confidence=cf, status=status,
            n_bodies=np.full(n, -1, np.int16), body_id=np.full(n, -1, np.int32),
            frame_index=np.arange(n, dtype=np.int64), ts_image_ns=np.zeros(n, np.int64),
            ts_host_ns=np.array([r[0] for r in rows], np.int64),
            dropped_total=np.zeros(n, np.int64), meta=meta)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def take_fps(rec: SS.Recording, default: float = 30.0) -> float:
    """The frame rate the take was RECORDED at, from its host timestamps."""
    ok = rec.status != SS.STATUS_GRAB_FAILED
    ts = rec.ts_host_ns[ok]
    if len(ts) > 2:
        dt = np.median(np.diff(ts)) / 1e9
        if dt > 0:
            return float(1.0 / dt)
    return default


def open_camera(a):
    """(source, camera metadata). The metadata is READ BACK from the camera in
    zed mode (tools/record_keypoints.session_meta); fake mode writes fake values
    plus the replayed take's identity and the take's own camera metadata."""
    if a.source == "zed":
        import record_keypoints as RK
        from g1_teleop.config import ZEDConfig
        from g1_teleop.zed_source import ZEDSource
        print("opening ZED through ZEDSource (body-tracking model load can take a while)...")
        zed = ZEDSource(ZEDConfig())
        if not zed.camera.is_opened():
            raise SystemExit("ZED did not open - ZEDSource does not check camera.open(); "
                             "check the cable/SDK")
        meta = RK.session_meta(zed, types.SimpleNamespace(
            label="teleop_live", note=a.note, headless=False, still_s=0.0))
        meta["fake"] = False
        return zed, meta
    rec_fps = take_fps(SS.load_recording(a.replay))
    src = ReplaySource(a.replay, fps=None if a.headless else (a.fake_fps or rec_fps))
    take = src.rec.meta
    meta = dict(
        fake=True, sdk_version="none (fake source)", camera_model="fake",
        serial_number=-1, firmware_version=-1,
        body_tracking=dict(fake=True, note="no camera: keypoints replayed from a file"),
        replay=dict(path=repo_relpath(a.replay), sha256=_sha256(a.replay),
                    n_rows=len(src.rec), recorded_fps=rec_fps,
                    paced_fps=src.fps, take_label=take.get("label"),
                    take_sdk_version=take.get("sdk_version"),
                    take_serial_number=take.get("serial_number"),
                    take_body_tracking=take.get("body_tracking"),
                    synthetic=bool(take.get("synthetic", False))))
    return src, meta


# ─── the recorder ────────────────────────────────────────────────────────────
class LiveRecorder(TeleopRecorder):
    """TeleopRecorder plus two per-tick audit columns read from the same frame:
    `frame_seq` (the loop's `last_seq`: which camera frame the arms came from)
    and `tick_wall_ns` (host monotonic clock at the tick)."""

    def __init__(self):
        super().__init__()
        self.frame_seq: List[int] = []
        self.tick_wall_ns: List[int] = []

    def _observe_teleop(self, m, d, L: dict) -> None:
        super()._observe_teleop(m, d, L)
        if self.buf is not None and len(self.buf) > len(self.frame_seq):
            self.frame_seq.append(int(L.get("last_seq", -1)))
            self.tick_wall_ns.append(time.monotonic_ns())


class EpisodeHooks(RIC.LoopHooks):
    """The episode's view of `run_loop`: termination, keys, HUD. Reads only."""

    def __init__(self, sess, rec, seed: int, label: str, max_ticks: int,
                 script=None, probe=None, early_success: bool = True):
        self.sess, self.rec, self.seed, self.label = sess, rec, seed, label
        self.max_ticks = int(max_ticks)
        self.script = sorted(script or [])          # [(t_sim, key)]
        self.probe = probe                          # test hook: probe(S, counter)
        self.early_success = early_success
        self.request = None                         # "stop" | "discard" | "quit"
        self.end_reason = None
        self.fell, self.fall_tick, self.success_tick, self.pass_run = False, -1, -1, 0
        self.bprime_attempts = []                   # sim seconds of each blocked 'b'
        self.meta_for_eval = None                   # set by the session (box_spawn_xy)

    # keys --------------------------------------------------------------
    def keys_due(self, t_sim):
        due = []
        while self.script and self.script[0][0] <= t_sim + 1e-12:
            due.append(self.script.pop(0)[1])
        return due

    def on_key(self, key, S):
        if key == ord("b"):
            # D18 (item 23): the contact model may not change inside an episode -
            # the recorded physics would be two physics. Logged, not applied.
            t = S.counter * RIC.SIM_DT
            self.bprime_attempts.append(round(t, 3))
            S.events.append((t, "B-prime toggle BLOCKED (episode recording)"))
            print("[%.2f s] b ignored: B-prime cannot change while recording (D18); "
                  "attempt logged" % t)
            return True
        if key == ord("e"):
            self.request = "stop"
            return True
        if key == ord("x"):
            self.request = "discard"
            return True
        if key == ord("q"):
            self.request = "quit"
            return True
        return False

    # termination (deploy.py's rules) ------------------------------------
    def should_end(self, S, counter):
        if self.probe is not None:
            self.probe(S, counter)
        if counter > 0 and S.data.qpos[S.ix.base_qpos][2] < FALL_Z:
            self.fell, self.fall_tick, self.end_reason = True, len(self.rec.buf or ()) - 1, "fell"
            return True
        if self.request is not None:
            self.end_reason = {"stop": "operator_stop", "discard": "operator_discard",
                               "quit": "operator_quit"}[self.request]
            return True
        if counter == 0 or counter % TICK:
            return False
        n = counter // TICK                 # ticks taken so far (0 .. n-1)
        if (self.early_success and n and self.rec.weld_release_tick >= 0
                and not S.weld.engaged(S.data)):
            r = SU.evaluate(dict(states=np.asarray(self.rec.buf.states),
                                 qvel=np.asarray(self.rec.buf.qvel)),
                            self.meta_for_eval, geo=self.sess.geo)
            self.pass_run = self.pass_run + 1 if r["episode_success"] else 0
            if self.pass_run >= SUCCESS_HOLD_TICKS:
                self.end_reason, self.success_tick = "success", n - 1
                return True
        if n >= self.max_ticks:
            self.end_reason = "time_limit"
            return True
        return False

    # HUD ------------------------------------------------------------------
    def draw(self, img, st, S):
        n = len(self.rec.buf) if self.rec.buf is not None else 0
        # Bottom-left, under the MINIMAL panel's four lines and above the hints.
        RIC.text(img, "REC  %s  seed %d  tick %d/%d" % (self.label, self.seed, n, self.max_ticks),
                 (10, RIC.PANEL_H - 172), 0.6, RIC.RED)
        RIC.text(img, "e stop+save   x discard   q quit (saves)", (10, RIC.PANEL_H - 148),
                 0.5, RIC.GREY)
        if not S.overlay_full:
            # the lock indicator ("lock rule v3 needs: ..." / "LOCKED"); the FULL
            # view already draws it as its first two lines
            RIC.draw_lock_status(img, 10, RIC.PANEL_H - 120, st)


# ─── one session ─────────────────────────────────────────────────────────────
def parse_script(s: Optional[str]):
    """'9.6:g,17:e' -> [(9.6, ord('g')), (17.0, ord('e'))] (headless fixtures)."""
    out = []
    for item in (s or "").split(","):
        if item.strip():
            t, k = item.split(":")
            out.append((float(t), ord(k) if len(k) == 1 else int(k)))
    return out


def ric_args(a, first_seed: int):
    """The run_integrated_combined arguments the session is opened with:
    keyboard locomotion, B-prime ON, no synthetic branch (the source is given)."""
    ns = RIC.build_parser().parse_args(["keyboard", str(first_seed)])
    ns.headless = bool(a.headless)
    ns.seconds = float("inf")
    return ns


def open_live_session(a, seeds):
    src, cam_meta = open_camera(a)
    tap = AuditTap(src)
    S = RIC.open_session(ric_args(a, seeds[0]), source=tap, sim_clock=bool(a.headless))
    if S is None:
        raise SystemExit("renderer could not be created")
    S.lock_rule = LOCK_RULES[a.lock_rule]
    S.tap, S.camera_meta, S.source_obj = tap, cam_meta, src
    S.twin_qpos0 = S.twin.data.qpos.copy()
    S.geo = SU.geometry(S.model)
    return S


def reset_for_episode(S, a, seed: int) -> dict:
    """The shared reset, then the start placement. Returns what the metadata needs."""
    # The twin's ARM pose is warm-start state for the IK; restore it so episode N
    # does not start its first solve from episode N-1's last pose.
    S.twin.data.qpos[:] = S.twin_qpos0
    S.twin.data.qvel[:] = 0.0
    mujoco.mj_forward(S.twin.model, S.twin.data)
    start = reset_episode(S.model, S.data, S.ix, S.cfg, seed, controller=S.controller,
                          loco_input=S.loco, twin=S.twin, policy=S.policy)
    fp = state_fingerprint(S.data, S.ix)
    if a.start_standoff is not None:
        xy = (float(start.box_pos[0]) - a.start_standoff, float(start.box_pos[1]))
    else:
        xy = (float(a.start_xy[0]), float(a.start_xy[1]))
    S.data.qpos[S.ix.base_qpos][0] = xy[0]
    S.data.qpos[S.ix.base_qpos][1] = xy[1]
    mujoco.mj_forward(S.model, S.data)
    # The camera side starts the episode fresh too: a fake take from frame 0,
    # and no frame from before the start is applied.
    if isinstance(getattr(S, "source_obj", None), ReplaySource):
        S.source_obj.rewind()
    g = S.grabber
    lock = getattr(g, "_lock", None)
    with (lock if lock is not None else nullcontext()):
        g._frame = None
        if isinstance(g, RIC.SimClockGrabber):
            g._seq = 0
    return dict(start=start, fingerprint=fp, start_xy=xy)


def _timing(tick_ns, paced: bool) -> dict:
    t = np.asarray(tick_ns, dtype=np.int64)
    if len(t) < 2:
        return dict(paced=paced, tick_rate_hz=None, overrun_ticks=0,
                    overrun_threshold_ms=1000 * OVERRUN_S)
    dt = np.diff(t) / 1e9
    return dict(paced=paced, tick_rate_hz=float((len(t) - 1) / max(dt.sum(), 1e-12)),
                overrun_ticks=int((dt > OVERRUN_S).sum()),
                overrun_threshold_ms=1000 * OVERRUN_S,
                max_tick_interval_ms=float(1000 * dt.max()),
                p95_tick_interval_ms=float(1000 * np.percentile(dt, 95)),
                note="tick interval on the host clock; an overrun is an interval "
                     "> 1.5 ticks (the loop fell half a tick behind 25 Hz)")


def run_episode(S, a, seed: int, label: str, record: bool = True, probe=None):
    """Reset, run the loop once, and return (hooks, recorder, reset info).
    `record=False` runs the same loop with no recorder attached (--invariance);
    early success then cannot be judged, so only fall / cap / keys end it."""
    info = reset_for_episode(S, a, seed)
    rec = LiveRecorder() if record else types.SimpleNamespace(buf=None, weld_release_tick=-1)
    hooks = EpisodeHooks(S, rec, seed, label, a.max_ticks, script=parse_script(a.script),
                         probe=probe, early_success=record)
    hooks.meta_for_eval = dict(box_spawn_xy=[float(info["start"].box_pos[0]),
                                             float(info["start"].box_pos[1])])
    S.tap.arm(getattr(S.grabber, "_seq", 0))
    contract_before = CC.contract_of(S.model)
    try:
        with (rec if record else nullcontext()):
            try:
                reason = RIC.run_loop(S, hooks)
            except KeyboardInterrupt:
                # Ctrl+C is a quit, and a quit SAVES what was recorded (flagged
                # failed unless it already succeeded) rather than losing it.
                print("\ninterrupted - ending the episode and saving it")
                reason, hooks.end_reason = "hook", "interrupted"
    finally:
        info["kp_rows"] = S.tap.disarm()
    if hooks.end_reason is None:
        hooks.end_reason = {"quit": "operator_quit", "seconds": "time_limit"}.get(reason, reason)
    # D18: the blocked key means this can only fail if something else toggled it.
    if CC.contract_of(S.model) != contract_before:
        raise AssertionError("the contact contract changed during the episode (D18)")
    return hooks, rec, info


def episode_arrays(rec):
    """The arrays exactly as `EpisodeBuffer.save` will store them (float32), so
    the verdict here is the verdict the loaders will compute."""
    return dict(states=np.asarray(rec.buf.states, dtype=np.float32),
                qvel=np.asarray(rec.buf.qvel, dtype=np.float32))


def episode_meta(S, a, seed, hooks, rec, info, verdict, kp_path) -> dict:
    start = info["start"]
    tracking = np.asarray(rec.buf.tracking, dtype=bool)
    src = SOURCE_TELEOP_LIVE if a.source == "zed" else SOURCE_TELEOP_LIVE_FAKE
    pd_ = verdict["place_detail"]
    return dict(
        seed=int(seed), box_spawn_xy=[float(start.box_pos[0]), float(start.box_pos[1])],
        heldout=bool(start.heldout),
        standoff_cmd=float(a.start_standoff) if a.start_standoff is not None else float("nan"),
        lateral_cmd=float("nan"), start_base_xy=list(info["start_xy"]),
        source=src, operator=a.operator, note=a.note,
        camera=S.camera_meta,
        reset_fingerprint=info["fingerprint"],
        contact_contract=CC.contract_of(S.model),
        d18=dict(hand_pickup_exclusion=bool(CC.hand_pickup_exclusion_active(S.model)),
                 toggle_blocked_attempts_s=list(hooks.bprime_attempts)),
        lock_rule=rec.lock_rule, lock_rule_id=(rec.lock_rule or {}).get("version"),
        weld_engage_tick=rec.weld_engage_tick, weld_release_tick=rec.weld_release_tick,
        lock_engage_tick=rec.lock_engage_tick, lock_release_tick=rec.lock_release_tick,
        box_pre_grasp_disturb_mm=float(rec.box_pre_grasp_disturb_mm),
        end_reason=hooks.end_reason, failed=not verdict["episode_success"],
        outcome=dict(ok=bool(verdict["episode_success"]), attributed_to=verdict["attributed_to"],
                     grasp=bool(verdict["grasp"]), lift=bool(verdict["lift"]),
                     transport=bool(verdict["transport"]), place=bool(verdict["place"]),
                     placement_error=float(pd_["placement_error_m"]),
                     tilt_deg=float(pd_["tilt_deg"]), released=bool(pd_["released"]),
                     fell=bool(hooks.fell), fall_tick=int(hooks.fall_tick),
                     success_tick=int(hooks.success_tick)),
        termination=dict(rule="deploy.py", max_ticks=int(a.max_ticks), fall_z=FALL_Z,
                         success_hold_ticks=SUCCESS_HOLD_TICKS),
        degraded_ticks=int((~tracking).sum()),
        dropout_rate=float((~tracking).mean()) if len(tracking) else 0.0,
        timing=dict(_timing(rec.tick_wall_ns, paced=not a.headless),
                    sim_seconds=len(rec.buf) * TICK_S),
        keypoints_file=repo_relpath(kp_path) if kp_path else None,
        keypoints_seq_at_arm=int(S.tap.seq_at_arm),
        ik_max_iter=int(S.cfg.ik.max_iter),
        note_phase_labels="teleop: phase_labels are UNKNOWN by construction - derive offline")


def save_episode(S, a, out_dir, seed, hooks, rec, info, verdict):
    """Write the episode (and its raw keypoints). Failed -> the failed/ folder."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = "ep_seed%04d_%s" % (seed, stamp)
    base = out_dir if verdict["episode_success"] else os.path.join(out_dir, FAILED_DIR)
    path = os.path.join(base, stem + ".npz")
    kp_path = os.path.join(out_dir, KP_DIR, stem + "_kp.npz")
    meta = episode_meta(S, a, seed, hooks, rec, info, verdict, kp_path)
    os.makedirs(os.path.dirname(kp_path), exist_ok=True)
    SS.save_recording(kp_path, AuditTap.to_recording(info["kp_rows"], dict(
        label="teleop_live episode %s" % stem, episode=repo_relpath(path),
        seq_at_arm=int(S.tap.seq_at_arm), camera=S.camera_meta,
        status_note="ok / no_body (EMPTY frame: ZEDSource does not say why) / "
                    "grab_failed; frame_seq s in the episode = the "
                    "(s - seq_at_arm)-th delivered row")))
    rec.buf.save(path, meta, extra=dict(
        frame_seq=np.asarray(rec.frame_seq, dtype=np.int64),
        tick_wall_ns=np.asarray(rec.tick_wall_ns, dtype=np.int64)))
    return path, kp_path, meta


def checks_of(verdict, meta) -> dict:
    return {"grasp": bool(verdict["grasp"]), "lift": bool(verdict["lift"]),
            "transport": bool(verdict["transport"]), "place": bool(verdict["place"]),
            "no fall": not meta["outcome"]["fell"],
            "tracking never lost": meta["degraded_ticks"] == 0}


def finish_episode(S, a, ledger, out_dir, seed, label, hooks, rec, info) -> str:
    """Evaluate, then save / discard and write the ledger row. Returns the
    decision: "accept", "failed" or "discard"."""
    n = len(rec.buf) if rec.buf is not None else 0
    if hooks.end_reason == "operator_discard" or n == 0:
        why = "operator discard" if n else "no ticks recorded"
        ledger.append("discard", seed=seed, reason=why, label=label, saved=None,
                      end_reason=hooks.end_reason, ticks=n)
        print("  DISCARDED (%s) - nothing written" % why)
        return "discard"
    verdict = SU.evaluate(episode_arrays(rec),
                          dict(box_spawn_xy=hooks.meta_for_eval["box_spawn_xy"]), geo=S.geo)
    if hooks.fell:
        verdict["episode_success"] = False
        verdict["attributed_to"] = verdict["attributed_to"] or "FELL"
    path, kp_path, meta = save_episode(S, a, out_dir, seed, hooks, rec, info, verdict)
    checks = checks_of(verdict, meta)
    if verdict["episode_success"]:
        ledger.append("accept", seed=seed, path=repo_relpath(path), heldout=meta["heldout"],
                      label=label, checks=checks, keypoints=repo_relpath(kp_path),
                      placement_error=meta["outcome"]["placement_error"],
                      end_reason=hooks.end_reason, ticks=n)
        decision = "accept"
    else:
        ledger.append("discard", seed=seed, reason="failed: %s (%s)"
                      % (verdict["attributed_to"] or "unknown", hooks.end_reason),
                      label=label, saved=repo_relpath(path), keypoints=repo_relpath(kp_path),
                      checks=checks, end_reason=hooks.end_reason, ticks=n,
                      placement_error=meta["outcome"]["placement_error"])
        decision = "failed"
    print("  %s -> %s (%d ticks, %s, placement %.3f m, degraded %d)"
          % ("ACCEPTED" if decision == "accept" else "FAILED, kept for audit",
             repo_relpath(path), n, hooks.end_reason, meta["outcome"]["placement_error"],
             meta["degraded_ticks"]))
    return decision


def assert_out_dir_compatible(out_dir: str, rule: LockConfig) -> None:
    """One out-dir = one spec version, one contact contract, one lock rule -
    including the failed/ folder, so a session cannot start a mix that the
    dataset builder would refuse later."""
    eps = DS.scan(out_dir, require_uniform=False) + DS.scan(
        os.path.join(out_dir, FAILED_DIR), require_uniform=False)
    if not eps:
        return
    DS.assert_uniform(eps)
    DS.assert_uniform_lock_rule(eps)
    have = lock_config_from_meta(eps[0].meta)
    if have != rule:
        raise DS.DatasetError(
            "%s already holds episodes recorded under lock rule %s; this session "
            "asks for %s. A dataset mixes lock rules only by mistake (build_dataset "
            "refuses it): use another --out-dir." % (repo_relpath(out_dir), have.version,
                                                     rule.version))


def assert_seeds_are_training_seeds(seeds) -> None:
    """Never record a demonstration on an evaluation or selection seed."""
    held = {}
    for name in ("exp1_seeds.json", "exp2_seeds.json", "select_seeds.json"):
        p = os.path.join(DS.EVAL, name)
        if os.path.exists(p):
            for s in DS.load_seed_file(p):
                held[int(s)] = name
    bad = sorted(s for s in seeds if int(s) in held)
    if bad:
        raise SystemExit("refusing: seed(s) %s belong to %s - evaluation/selection seeds "
                         "are never demonstrated" % (bad[:10], sorted({held[s] for s in bad})))


# ─── idle screen (interactive only): physics paused between episodes ─────────
def idle(S, seed: int, k: int, total: int, label: str, last: str) -> str:
    """Show the camera and the reset scene for the NEXT seed; wait for a key.
    Returns "start", "skip" or "quit"."""
    import cv2
    if not S.window_open:
        cv2.namedWindow(RIC.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RIC.WINDOW, RIC.ZED_PANEL_W + RIC.MJ_PANEL_W, RIC.PANEL_H)
        S.window_open = True
    while True:
        frame, _ = S.grabber.latest()
        left = (RIC.fit_to_box(frame.image, RIC.ZED_PANEL_W, RIC.PANEL_H) if frame is not None
                else np.zeros((RIC.PANEL_H, RIC.ZED_PANEL_W, 3), np.uint8))
        S.renderer.update_scene(S.data, camera=S.cam, scene_option=S.scene_option)
        right = RIC.fit_to_box(cv2.cvtColor(S.renderer.render(), cv2.COLOR_RGB2BGR),
                               RIC.MJ_PANEL_W, RIC.PANEL_H)
        RIC.text(right, "NEXT  %s  seed %d   (%d of %d)" % (label, seed, k, total), (16, 46), 0.9,
                 RIC.CYAN)
        RIC.text(right, "ENTER start   n skip seed   q quit", (16, 86), 0.6, RIC.GREY)
        RIC.text(right, "B-PRIME ON (fixed for collection, D18)", (16, 120), 0.6, RIC.GREEN)
        if last:
            RIC.text(right, "last: %s" % last, (16, 154), 0.55, RIC.GREY)
        RIC.text(left, "camera live - physics paused", (16, 40), 0.7, RIC.GREY)
        cv2.imshow(RIC.WINDOW, np.hstack([left, right]))
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        low = key & 0xFF
        if low in KEY_ENTER:
            return "start"
        if low == ord("n"):
            return "skip"
        if low == ord("q"):
            return "quit"
        if low == ord("b"):
            # Every episode must start under the D18 contract (reset_episode
            # raises otherwise), so there is nothing to toggle it FOR here; A/B
            # work on the contact model belongs in run_integrated_combined.py.
            print("b ignored: B-prime is fixed ON for collection (D18)")


def cmd_record(a) -> int:
    rule = LOCK_RULES[a.lock_rule]
    src_name = SOURCE_TELEOP_LIVE if a.source == "zed" else SOURCE_TELEOP_LIVE_FAKE
    label = label_of(src_name, where="record_teleop_live")
    out_dir = os.path.abspath(a.out_dir or (NS_COLLECTION if a.source == "zed" else NS_TELEOP_FAKE))
    assert_namespace(out_dir, src_name, where="record_teleop_live --out-dir")
    assert_out_dir_compatible(out_dir, rule)
    ledger = EpisodeLedger(ledger_for(out_dir))
    stream = ([int(s) for s in a.seed_list.split(",") if s.strip()] if a.seed_list
              else seed_stream(a.seeds, 0))
    if a.source == "zed":
        assert_seeds_are_training_seeds(stream)
    pending = ledger.pending(stream)
    if not pending:
        print("nothing pending: every seed in the stream already has an ACCEPT")
        return 0
    S = open_live_session(a, pending)
    ledger.append("session", label=label, source=src_name, operator=a.operator,
                  seeds=pending[:a.count], spec_version=spec.SPEC_VERSION,
                  lock_rule=rule.version, out_dir=repo_relpath(out_dir),
                  camera=dict((k, S.camera_meta.get(k)) for k in
                              ("fake", "sdk_version", "serial_number", "camera_model")))
    print("session %r (operator %s, lock rule %s): %d seed(s) pending, %d episode(s) "
          "this session -> %s" % (label, a.operator, rule.version, len(pending), a.count,
                                  repo_relpath(out_dir)))
    tally = dict(accept=0, failed=0, discard=0)
    queue, attempts, last = list(pending), 0, ""
    try:
        while queue and attempts < a.count:
            seed = queue[0]
            info0 = reset_for_episode(S, a, seed)
            if info0["start"].heldout:
                ledger.append("error", seed=seed, reason="held-out spawn: never demonstrated")
                print("seed %d spawns in the HELD-OUT patch - skipped" % seed)
                queue.pop(0)
                continue
            if not a.headless:
                choice = idle(S, seed, attempts + 1, a.count, label, last)
                if choice == "quit":
                    break
                if choice == "skip":
                    queue.pop(0)
                    continue
            attempts += 1
            ledger.append("issue", seed=seed, label=label)
            print("\n=== episode %d/%d: seed %d ===" % (attempts, a.count, seed))
            hooks, rec, info = run_episode(S, a, seed, label)
            decision = finish_episode(S, a, ledger, out_dir, seed, label, hooks, rec, info)
            tally[decision] += 1
            last = "seed %d %s (%s)" % (seed, decision.upper(), hooks.end_reason)
            # Interactive: a failed or discarded seed is offered again (proposal
            # 3.3.8: failures are re-recorded); `n` skips it. Headless: move on.
            if decision == "accept" or a.headless:
                queue.pop(0)
            if hooks.end_reason in ("operator_quit", "interrupted"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        RIC.close_session(S)
    print("\naccepted %d, failed %d, discarded %d; ledger %s"
          % (tally["accept"], tally["failed"], tally["discard"], repo_relpath(ledger.path)))
    print("\n".join("%-18s %s" % (k, v) for k, v in ledger.summary().items()))
    return 0


def cmd_invariance(a) -> int:
    """Run the first seed twice, recorder attached and not, and compare qpos and
    qvel at EVERY physics step. Needs --headless (the sim clock): a threaded
    grabber is wall-clock timed and never repeats exactly."""
    if not a.headless:
        raise SystemExit("--invariance needs --headless (deterministic sim-clock source)")
    seeds = ([int(s) for s in a.seed_list.split(",")] if a.seed_list else seed_stream(a.seeds, 0))
    S = open_live_session(a, seeds)
    try:
        traces = []
        for record in (False, True):
            h = hashlib.sha256()
            steps = [0]

            def probe(S_, counter, h=h, steps=steps):
                h.update(np.asarray(S_.data.qpos).tobytes())
                h.update(np.asarray(S_.data.qvel).tobytes())
                steps[0] += 1
            hooks, rec, _ = run_episode(S, a, seeds[0], "invariance", record=record, probe=probe)
            traces.append((h.hexdigest(), steps[0], hooks.end_reason))
            print("record=%s: %d steps, end %s, trajectory sha256 %s"
                  % (record, steps[0], hooks.end_reason, h.hexdigest()[:16]))
    finally:
        RIC.close_session(S)
    same = traces[0][:2] == traces[1][:2]
    print("OBSERVATION ONLY: %s" % ("CONFIRMED" if same else "VIOLATED"))
    return 0 if same else 1


def make_fake_take(a) -> int:
    """Write a SYNTHETIC grasp-and-lift take for `--source fake`, built for one
    seed's spawn and start pose exactly as the --synthetic grasp fixture builds
    its frames, on the simulated frame clock (FRAME_STEPS). The grasp command is
    the operator's key; its time is in the take's meta (`grasp_on_s`)."""
    import torch  # noqa: F401  (open_session loads the policy)
    seed = int((a.seed_list or "0").split(",")[0])
    ns = RIC.build_parser().parse_args(["keyboard", str(seed), "--synthetic", "none", "--headless"])
    S = RIC.open_session(ns)
    S.twin_qpos0 = S.twin.data.qpos.copy()
    try:
        info = reset_for_episode(S, a, seed)
        twin, model, data = S.twin, S.model, S.data
        mujoco.mj_forward(twin.model, twin.data)
        tp = twin.data.xpos[mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].copy()
        shoulders = {"left": twin.left_shoulder_world() - tp, "right": twin.right_shoulder_world() - tp}
        home = {"left": twin.data.xpos[twin.left_wrist_body] - tp,
                "right": twin.data.xpos[twin.right_wrist_body] - tp}
        lengths = {"left": (twin.upper_arm_left, twin.forearm_left),
                   "right": (twin.upper_arm_right, twin.forearm_right)}
        box_rel = (data.xpos[S.box_body_id] - data.xpos[S.pel_bid])
        frame_hz = 1.0 / (RIC.FRAME_STEPS * RIC.SIM_DT)
        frames, grasp_on_s, lift_s = SS.grasp_lift_frames(shoulders, lengths, home, box_rel, frame_hz)
    finally:
        RIC.close_session(S)
    gaps = [tuple(int(x) for x in g.split("+")) for g in (a.no_body or "").split(",") if g.strip()]
    if gaps:
        frames = SS.with_no_body(frames, gaps)
    n = len(frames)
    status = np.array([SS.STATUS_OK if f else SS.STATUS_NO_BODY for f in frames], np.int8)
    kp = np.array([np.asarray(f, float) if f else np.full((SS.N_KEYPOINTS, 3), np.nan)
                   for f in frames]).reshape(n, SS.N_KEYPOINTS, 3)
    conf = np.where(status[:, None] == SS.STATUS_OK, 100.0, np.nan) * np.ones((n, SS.N_KEYPOINTS))
    ts = (np.arange(n) * 1e9 / frame_hz).astype(np.int64)
    rec = SS.Recording(keypoints=kp, confidence=conf, status=status,
                       n_bodies=(status == SS.STATUS_OK).astype(np.int16),
                       body_id=np.where(status == SS.STATUS_OK, 0, -1).astype(np.int32),
                       frame_index=np.arange(n, dtype=np.int64), ts_image_ns=ts, ts_host_ns=ts,
                       dropped_total=np.zeros(n, np.int64),
                       meta=dict(label="synthetic grasp-and-lift", synthetic=True, seed=seed,
                                 start_standoff=a.start_standoff, start_xy=list(info["start_xy"]),
                                 frame_hz=frame_hz, grasp_on_s=float(grasp_on_s),
                                 lift_s=float(lift_s), no_body_gaps=gaps,
                                 box_rel_pelvis=[float(v) for v in box_rel]))
    os.makedirs(os.path.dirname(os.path.abspath(a.make_fake_take)), exist_ok=True)
    SS.save_recording(a.make_fake_take, rec)
    print("wrote %s: %d frames at %.2f Hz; press g at %.2f s (lift at %.2f s)"
          % (a.make_fake_take, n, frame_hz, grasp_on_s, lift_s))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("zed", "fake"), default=None,
                    help="zed: the ZED 2i (pyzed). fake: replay --replay; never imports pyzed")
    ap.add_argument("--replay", default=None, help="--source fake: keypoint take (.npz, "
                    "tools/record_keypoints.py format)")
    ap.add_argument("--fake-fps", type=float, default=None,
                    help="--source fake, interactive: pace the take at this rate "
                         "(default: the rate it was recorded at)")
    ap.add_argument("--operator", default=None, help="operator id, stored per episode "
                    "(required with --source zed)")
    ap.add_argument("--note", default="", help="free text stored per episode (lighting, ...)")
    ap.add_argument("--seeds", default=os.path.join(DS.EVAL, "train_seeds.json"),
                    help="JSON seed stream (default data/eval/train_seeds.json)")
    ap.add_argument("--seed-list", default=None, help="comma-separated seeds instead of --seeds")
    ap.add_argument("--count", type=int, default=1, help="episodes to attempt this session")
    ap.add_argument("--out-dir", default=None,
                    help="episode folder AND its ledger (default recordings/demonstrations "
                         "for zed, recordings/teleop_fake for fake)")
    ap.add_argument("--lock-rule", choices=sorted(LOCK_RULES), default=LockConfig().version,
                    help="lock rule (g1_data.phases.LOCK_RULES; default: LockConfig's, "
                         "currently %s). Stored per episode." % LockConfig().version)
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--start-xy", type=lambda s: tuple(float(v) for v in s.split(",")),
                    default=START_XY, help="base start x,y (default the deploy/scripted start)")
    ap.add_argument("--start-standoff", type=float, default=None,
                    help="FIXTURE: start head-on this far behind the box instead of --start-xy")
    ap.add_argument("--headless", action="store_true",
                    help="--source fake only: no window, sim-clock frames, episodes start "
                         "at once; keys only from --script")
    ap.add_argument("--script", default=None,
                    help="headless: keys on the simulated clock, e.g. '9.6:g,17:e'")
    ap.add_argument("--invariance", action="store_true",
                    help="run the first seed with and without the recorder and compare "
                         "every physics step (needs --source fake --headless)")
    ap.add_argument("--make-fake-take", default=None, metavar="OUT.npz",
                    help="write a synthetic grasp-and-lift take for --seed-list's first seed")
    ap.add_argument("--no-body", default=None,
                    help="--make-fake-take: no-body gaps 'start+length,...' in frames (O28)")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.make_fake_take:
        return make_fake_take(a)
    if a.source is None:
        ap.error("--source zed | fake is required")
    if a.source == "fake" and not a.replay:
        ap.error("--source fake needs --replay <keypoints.npz>")
    if a.source == "zed":
        if a.headless:
            ap.error("--headless is for --source fake (a live operator needs the window)")
        if not a.operator:
            ap.error("--source zed needs --operator <id>")
    a.operator = a.operator or "fake"
    if a.invariance:
        return cmd_invariance(a)
    return cmd_record(a)


if __name__ == "__main__":
    raise SystemExit(main())
