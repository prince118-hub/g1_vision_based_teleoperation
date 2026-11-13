"""Tests for the operator tools: the accept/reject step of the live recorder
(TASK A), the base-lock floor guide (TASK B) and walk_test.py's practice mode
(TASK C). CPU only, no ZED, no display, no OpenGL.

    python tools/test_operator_tools.py

Fixtures are those of tools/test_record_teleop_live.py: the synthetic
grasp-and-lift take for seed 0 at a head-on 0.32 m start, replayed on the
simulated clock. It grasps and lifts and never places, so its natural verdict
is FAILED (PLACE); a successful verdict is forced where a test needs one.

No OpenGL here: the guide is drawn into a REAL `MjvScene` filled by
`mjv_updateScene` (what `mujoco.Renderer.update_scene` does) through a
stand-in renderer whose `render()` returns a black image. What is NOT tested,
because it needs a display: the pixels (colours, the LOCKED label), the
verdict screen's keys through `cv2.waitKeyEx`, the practice window. Those are
in docs/OPERATOR_GUIDE.md's laptop checks.
"""
from __future__ import annotations

import glob
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import mujoco                                                       # noqa: E402

import record_teleop_live as RT                                     # noqa: E402
from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data.ledger import EpisodeLedger                            # noqa: E402
from g1_data.phases import LOCK_RULE_V1, LockConfig                 # noqa: E402
from g1_data.recorder import load_episode                           # noqa: E402
from g1_teleop import floor_guide as FG                             # noqa: E402
from g1_teleop import operator_config as OC                         # noqa: E402

SEED, STANDOFF = 0, 0.32
_CACHE: dict = {}


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _tmp() -> str:
    if "tmp" not in _CACHE:
        _CACHE["tmp"] = tempfile.mkdtemp(prefix="operator_tools_test_")
    return _CACHE["tmp"]


def _take():
    if "take" not in _CACHE:
        path = os.path.join(_tmp(), "take.npz")
        assert RT.main(["--make-fake-take", path, "--seed-list", str(SEED),
                        "--start-standoff", str(STANDOFF)]) == 0
        from g1_teleop.synthetic_source import load_recording
        _CACHE["take"] = (path, load_recording(path).meta)
    return _CACHE["take"]


def _argv(out_dir, script, extra=()):
    return ["--source", "fake", "--replay", _take()[0], "--seed-list", str(SEED),
            "--start-standoff", str(STANDOFF), "--headless", "--operator", "tester",
            "--out-dir", out_dir, "--script", script] + list(extra)


def _grasp_script(stop_s=14.0) -> str:
    return "%.3f:g,%.3f:e" % (_take()[1]["grasp_on_s"] + 0.1, stop_s)


class _Forced:
    """Force `success.evaluate` to report a success (the fixture never places)."""

    def __enter__(self):
        self.real = RT.SU.evaluate

        def forced(arrays, meta, cfg=None, geo=None):
            r = self.real(arrays, meta, cfg, geo)
            r.update(episode_success=True, place=True, attributed_to="")
            return r
        RT.SU.evaluate = forced
        return self

    def __exit__(self, *e):
        RT.SU.evaluate = self.real


class _Answers:
    """Script the headless a/d prompt (RT.INPUT); records every prompt."""

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __enter__(self):
        self.real = RT.INPUT

        def fake(prompt):
            self.prompts.append(prompt)
            return self.answers.pop(0)
        RT.INPUT = fake
        return self

    def __exit__(self, *e):
        RT.INPUT = self.real


def _rows(out):
    return EpisodeLedger(os.path.join(out, "ledger.jsonl")).rows()


# ═══ TASK A - accept / reject per human episode ══════════════════════════════
def test_a_failed_episode_recommends_discard_with_the_failing_phase():
    """The natural fixture verdict (FAILED at PLACE): recommendation DISCARD
    with the reason, saved to failed/, the ledger row carries choice +
    recommendation; headless defaults to --auto pass."""
    out = os.path.join(_tmp(), "a_failed")
    assert RT.main(_argv(out, _grasp_script())) == 0
    path = glob.glob(os.path.join(out, "failed", "*.npz"))[0]
    _, meta = load_episode(path)
    assert meta["accepted"] is False and meta["failed"] is True
    assert meta["operator_choice"] == "discard" and meta["decided_by"] == "auto:pass"
    rec = meta["recommendation"]
    assert rec["recommend"] == "discard" and any("PLACE" in r for r in rec["reasons"]), rec
    d = _rows(out)[-1]
    assert d["event"] == "discard" and d["reason"].startswith("failed: PLACE")
    assert d["operator_choice"] == "discard" and d["recommendation"]["recommend"] == "discard"
    assert meta["operator_ui"]["guide"] == "on"
    # the follow camera at its default: preset behind, no offsets, nothing changed
    v = meta["operator_ui"]["view"]
    assert (v["mode"], v["preset"], v["changes"]) == ("follow", "behind", [])
    assert v["offsets"] == dict(azimuth_deg=0.0, elevation_deg=0.0, distance_m=0.0)
    assert set(meta["operator_ui"]) == {"guide", "view"}


def test_a_a_failed_episode_can_never_be_accepted():
    """--auto off: the operator presses a twice (refused both times), then d."""
    out = os.path.join(_tmp(), "a_refuse")
    with _Answers("a", "accept", "d") as ans:
        assert RT.main(_argv(out, "2.0:e", extra=["--auto", "off"])) == 0
    assert len(ans.prompts) == 3 and "accept not possible" in ans.prompts[0]
    assert not glob.glob(os.path.join(out, "*.npz"))
    assert len(glob.glob(os.path.join(out, "failed", "*.npz"))) == 1
    d = _rows(out)[-1]
    assert d["event"] == "discard" and d["operator_choice"] == "discard"
    assert d["decided_by"] == "operator"
    # and decide() itself refuses, whoever asks
    rep = dict(can_accept=False, recommend="discard")
    a = type("A", (), dict(auto="pass", headless=True))()
    e = _raises(AssertionError, RT.decide, None, a, dict(rep, recommend="accept"), "operator_stop")
    assert "cannot be accepted" in str(e)


def test_a_operator_accepts_a_successful_episode():
    out = os.path.join(_tmp(), "a_accept")
    with _Forced(), _Answers("a") as ans:
        assert RT.main(_argv(out, "2.0:e", extra=["--auto", "off"])) == 0
    assert "[a]ccept / [d]iscard" in ans.prompts[0]
    files = glob.glob(os.path.join(out, "*.npz"))
    assert len(files) == 1
    _, meta = load_episode(files[0])
    assert meta["accepted"] is True and meta["operator_choice"] == "accept"
    assert meta["recommendation"] == {"recommend": "accept", "reasons": []}
    r = _rows(out)[-1]
    assert r["event"] == "accept" and r["operator_choice"] == "accept"
    assert r["decided_by"] == "operator" and r["recommendation"]["recommend"] == "accept"
    assert EpisodeLedger(os.path.join(out, "ledger.jsonl")).pending([SEED]) == []


def test_a_operator_discards_a_successful_episode():
    """A success the operator discards: kept in discarded/ (never the train
    folder), a re-issuable DISCARD row recording that the recommendation was
    accept; the out-dir check scans discarded/ too."""
    out = os.path.join(_tmp(), "a_discard_ok")
    with _Forced(), _Answers("d"):
        assert RT.main(_argv(out, "2.0:e", extra=["--auto", "off"])) == 0
    assert not glob.glob(os.path.join(out, "*.npz"))
    kept = glob.glob(os.path.join(out, "discarded", "*.npz"))
    assert len(kept) == 1
    _, meta = load_episode(kept[0])
    assert meta["failed"] is False and meta["accepted"] is False
    r = _rows(out)[-1]
    assert r["event"] == "discard" and "recommended accept" in r["reason"]
    assert r["operator_choice"] == "discard" and r["recommendation"]["recommend"] == "accept"
    assert EpisodeLedger(os.path.join(out, "ledger.jsonl")).pending([SEED]) == [SEED]
    assert DS.scan(out) == []                       # the loaders never see it
    e = _raises(DS.DatasetError, RT.assert_out_dir_compatible, out, LOCK_RULE_V1)
    assert "lock rule" in str(e)


def _report(verdict_success=True, dropout=0.0, overruns=0, paced=False, n=400):
    ep = _CACHE.get("ep_arrays")
    if ep is None:
        out = os.path.join(_tmp(), "a_failed")
        if not glob.glob(os.path.join(out, "failed", "*.npz")):
            test_a_failed_episode_recommends_discard_with_the_failing_phase()
        arrays, meta = load_episode(glob.glob(os.path.join(out, "failed", "*.npz"))[0])
        ep = _CACHE["ep_arrays"] = (arrays, meta)
    arrays, meta = ep
    v = RT.SU.evaluate(arrays, meta)
    if verdict_success:
        v.update(episode_success=True, place=True, attributed_to="")
    m = dict(meta, dropout_rate=dropout, degraded_ticks=int(dropout * n), n_ticks=n,
             timing=dict(paced=paced, tick_rate_hz=24.9 if paced else None,
                         overrun_ticks=overruns))
    return RT.episode_report(v, m)


def test_a_recommendation_reasons_dropout_and_overruns():
    """Soft reasons: a success with 20% dropout or 5% late ticks is recommended
    DISCARD but may still be accepted; headless overruns are n/a."""
    ok = _report()
    assert ok["recommend"] == "accept" and ok["can_accept"] and not ok["reasons"]
    assert [r for r in ok["rows"] if r[0] == "tick overruns"][0][1] is None
    drop = _report(dropout=0.20)
    assert drop["recommend"] == "discard" and drop["can_accept"]
    assert any("tracking dropout 20.0%" in r for r in drop["reasons"]), drop["reasons"]
    late = _report(paced=True, overruns=20, n=400)
    assert late["recommend"] == "discard" and any("overruns" in r for r in late["reasons"])
    assert _report(paced=True, overruns=2, n=400)["recommend"] == "accept"
    bad = _report(verdict_success=False)
    assert not bad["can_accept"] and bad["failing_phase"] == "PLACE"
    lines = [t for t, _ in RT.report_lines(bad)]
    assert any(t.startswith("place") and "FAIL" in t for t in lines)
    assert any("placement" in t for t in lines) and any("tracking dropout" in t for t in lines)
    assert "accept NOT possible" in lines[-1]
    img = RT.verdict_image(bad)
    assert img.shape == (RT.RIC.PANEL_H, RT.RIC.ZED_PANEL_W, 3) and img.any()


def test_a_keys_and_refusals():
    assert RT.choice_from_key(ord("a"), True) == "accept"
    assert RT.choice_from_key(ord("a"), False) == "refused"
    assert RT.choice_from_key(ord("d"), False) == "discard"
    assert RT.choice_from_key(ord("D"), True) == "discard"
    assert RT.choice_from_key(-1, True) is None and RT.choice_from_key(ord("e"), True) is None
    a = type("A", (), dict(auto="off", headless=True))()
    assert RT.decide(None, a, dict(can_accept=True, recommend="accept"),
                     "interrupted")[0] == "discard"           # Ctrl+C never accepts
    _raises(SystemExit, RT.main, ["--source", "zed", "--operator", "x", "--auto", "pass"])


# ═══ TASK B - the floor guide ═════════════════════════════════════════════════
def _terms(fwd, lat, welded=False, to_goal=1.0):
    return dict(fwd=fwd, lat=lat, head=float(np.degrees(np.arctan2(lat, fwd))),
                welded=welded, to_goal=to_goal, disp=float("inf"), box_lift=0.0, palm_far=1.0)


def test_b_status_colours_follow_the_active_rule():
    c = LockConfig()
    assert FG.status(_terms(0.32, 0.0), c, False).colour_name == "green"
    s = FG.status(_terms(0.32, 0.045), c, False)        # lat ok, head 8 deg > 5
    assert (s.fwd_ok, s.lat_ok, s.head_ok, s.colour_name) == (True, True, False, "yellow")
    assert FG.status(_terms(0.60, 0.20), c, False).colour_name == "red"
    assert FG.status(_terms(0.60, 0.20), c, True).colour_name == "green"   # LOCKED
    assert not FG.status(_terms(0.32, 0.0, to_goal=0.05), c, False).visible  # task done
    assert not FG.status({}, c, False).visible
    # read from the active LockConfig, not hardcoded
    far = LockConfig(version="custom", fwd_lo=0.50, fwd_hi=0.70)
    assert FG.status(_terms(0.60, 0.0), far, False).colour_name == "green"
    assert not FG.status(_terms(0.32, 0.0), far, False).fwd_ok


def _in_rect(p, prim):
    c, (hu, hv), yaw = prim["center"], prim["half"], prim["yaw"]
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    u, v = R @ (np.asarray(p) - c)
    return abs(u) <= hu + 1e-9 and abs(v) <= hv + 1e-9


def test_b_band_and_wedge_are_the_rule_on_the_floor():
    """For random robot poses around the box: the base lies inside the drawn
    band iff fwd and lat hold, and inside the drawn wedge iff heading holds -
    the marking IS the predicate's geometry, for any heading."""
    rng = np.random.default_rng(0)
    c = LockConfig()
    box = np.array([1.5, 0.05])
    n_in = 0
    for _ in range(400):
        yaw = rng.uniform(-np.pi, np.pi)
        base = box + rng.uniform(-0.5, 0.5, 2)
        d = box - base
        fwd = np.cos(yaw) * d[0] + np.sin(yaw) * d[1]
        lat = -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]
        head = np.degrees(np.arctan2(np.sin(np.arctan2(d[1], d[0]) - yaw),
                                     np.cos(np.arctan2(d[1], d[0]) - yaw)))
        t = dict(_terms(fwd, lat), head=head)
        prims = {p["name"]: p for p in FG.geometry(t, c, base, yaw, False)}
        # the target recovered from the terms is the box
        assert np.allclose(prims["wedge_left"]["a"], box)
        st = FG.status(t, c, False)
        assert _in_rect(base, prims["band"]) == (st.fwd_ok and st.lat_ok)
        n_in += st.fwd_ok and st.lat_ok
        # inside the wedge: within head_max of the backward axis through the box
        back = np.array([-np.cos(yaw), -np.sin(yaw)])
        rel = base - box
        ang = np.degrees(np.arccos(np.clip(rel @ back / np.linalg.norm(rel), -1, 1)))
        assert abs(ang - abs(head)) < 1e-6          # the wedge angle IS |head|
        assert (ang <= c.head_max_deg) == st.head_ok
        col = {p["name"]: p["rgb"] for p in prims.values()}
        assert col["near_edge"] == (FG.GREEN if st.fwd_ok else FG.RED)
        assert col["left_edge"] == (FG.GREEN if st.lat_ok else FG.RED)
        assert col["heading_arrow"] == col["wedge_left"] == (FG.GREEN if st.head_ok else FG.RED)
    assert n_in > 0, "no sample landed in the band"


def _model():
    if "model" not in _CACHE:
        from g1_teleop.config import TeleopConfig
        from g1_teleop.contact_contract import load_model
        m = load_model(TeleopConfig(), OC.SCENE_PATH)
        _CACHE["model"] = m
    return _CACHE["model"]


def test_b_guide_is_decor_in_the_render_scene_only():
    """Drawn into a real MjvScene after mjv_updateScene: 8 DECOR geoms
    (band, 4 edges, 2 wedge rays, arrow), a LOCKED label when locked, and the
    model and data are byte-for-byte untouched."""
    from g1_data.phases import LockPredicate, PlatformGeometry
    from g1_teleop.indices import ModelIndex
    m = _model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ix = ModelIndex.resolve(m)
    sp = spec.SpecLayout.resolve(m, ix)
    pred = LockPredicate(PlatformGeometry.resolve(m))
    terms = pred.terms(sp.build_state(m, d, ix, sync=False))
    scn = mujoco.MjvScene(m, maxgeom=10000)
    opt, pert, cam = mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera()
    before_d = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.qacc_warmstart.copy())
    m_sig = (m.ngeom, m.nbody, float(m.body_mass.sum()), m.geom_contype.copy().tobytes())
    g = FG.FloorGuide()
    for locked in (False, True):
        mujoco.mjv_updateScene(m, d, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        n0 = scn.ngeom
        assert g.draw(scn, terms, pred.cfg, d, ix, locked) == 8 == scn.ngeom - n0
        cats = {int(scn.geoms[i].category) for i in range(n0, scn.ngeom)}
        assert cats == {int(mujoco.mjtCatBit.mjCAT_DECOR)}
        labels = [scn.geoms[i].label for i in range(n0, scn.ngeom)]
        assert ("LOCKED" in labels) == locked
        assert g.last.locked == locked
    after_d = (d.qpos, d.qvel, d.ctrl, d.qacc_warmstart)
    assert all(np.array_equal(x, y) for x, y in zip(before_d, after_d))
    assert m_sig == (m.ngeom, m.nbody, float(m.body_mass.sum()), m.geom_contype.tobytes())
    # a full scene is never overrun
    small = mujoco.MjvScene(m, maxgeom=3)
    assert g.draw(small, terms, pred.cfg, d, ix, False) == 3


class _CpuRenderer:
    """Stand-in for mujoco.Renderer with no OpenGL: the same MjvScene filled by
    the same mjv_updateScene; render() returns a black image."""
    draws = 0

    def __init__(self, model, height=240, width=320):
        self.model, self.h, self.w = model, height, width
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self._opt, self._pert = mujoco.MjvOption(), mujoco.MjvPerturb()

    def update_scene(self, data, camera=None, scene_option=None):
        mujoco.mjv_updateScene(self.model, data, scene_option or self._opt, self._pert,
                               camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)

    def render(self):
        return np.zeros((self.h, self.w, 3), np.uint8)

    def close(self):
        pass


def _guided_run(guide: str, leak: bool = False, camera: str = "follow", cams=None):
    """run_integrated_combined on the --synthetic grasp fixture in --profile mode
    (renders + draws every 25 Hz frame, no window) with the TeleopRecorder
    attached: (sha of qpos+qvel after every step, steps, recorded states,
    guide statuses seen)."""
    import run_integrated_combined as RIC
    from g1_data.recorder import TeleopRecorder
    real_step, real_r, real_draw = mujoco.mj_step, mujoco.Renderer, FG.FloorGuide.draw
    h, n, seen = hashlib.sha256(), [0], []

    def draw(self, scn, terms, cfg, data, ix, locked):
        r = real_draw(self, scn, terms, cfg, data, ix, locked)
        seen.append((self.last.locked, self.last.colour_name, r))
        if leak and len(seen) == 30:
            data.qvel[0] += 1e-9                      # negative control
        return r

    def step(m, d, *a, **k):
        r = real_step(m, d, *a, **k)
        h.update(np.asarray(d.qpos).tobytes())
        h.update(np.asarray(d.qvel).tobytes())
        n[0] += 1
        return r
    mujoco.Renderer, FG.FloorGuide.draw = _CpuRenderer, draw
    S = RIC.open_session(RIC.build_parser().parse_args(
        ["keyboard", "0", "--synthetic", "grasp", "--profile", "--start-standoff", "0.32",
         "--seconds", "14", "--guide", guide, "--camera", camera]))
    if cams is not None:
        # record what the render path hands mjv_updateScene, frame by frame
        real_update = _CpuRenderer.update_scene

        def update(self, data, camera=None, scene_option=None):
            cams.append((float(camera.azimuth), float(camera.elevation), float(camera.distance),
                         tuple(float(v) for v in camera.lookat)))
            return real_update(self, data, camera=camera, scene_option=scene_option)
        S.renderer.update_scene = update.__get__(S.renderer)
    try:
        mujoco.mj_step = step
        with TeleopRecorder() as rec:
            RIC.run_loop(S)
    finally:
        mujoco.mj_step, mujoco.Renderer, FG.FloorGuide.draw = real_step, real_r, real_draw
        real_close = RIC.cv2.destroyAllWindows     # --profile is not headless; no GUI here
        RIC.cv2.destroyAllWindows = lambda: None
        try:
            RIC.close_session(S)
        finally:
            RIC.cv2.destroyAllWindows = real_close
    return h.hexdigest(), n[0], np.asarray(rec.buf.states), seen


def test_b_guide_on_vs_off_identical_trajectories_and_recorded_states():
    """THE invariance proof: guide on vs off on the grasp fixture through the
    lock, the weld, the lift and the carry release - every physics step's qpos
    and qvel identical, and the recorded 48-D states identical. The guide drew
    both the LOCKED and the free (goal-targeted) variants on the way."""
    h_on, n_on, s_on, seen = _guided_run("on")
    h_off, n_off, s_off, seen_off = _guided_run("off")
    assert n_on == n_off == 7000, (n_on, n_off)
    assert not seen_off and len(seen) > 300
    locked = [x[0] for x in seen]
    first, last = locked.index(True), len(locked) - 1 - locked[::-1].index(True)
    assert first > 0 and last < len(locked) - 1, "must draw free -> LOCKED -> free (carry release)"
    assert all(x[2] == 8 for x in seen if x[0]), "LOCKED draws must be complete"
    assert h_on == h_off, "the floor guide changed the trajectory"
    assert s_on.shape == s_off.shape and np.array_equal(s_on, s_off)
    _CACHE["h_on"] = h_on


def test_b_invariance_check_catches_a_guide_that_writes():
    """Negative control: a guide nudging one velocity by 1e-9 once must change
    the hash, or the proof above proves nothing."""
    h_on = _CACHE.get("h_on") or _guided_run("on")[0]
    assert _guided_run("on", leak=True)[0] != h_on


def test_b_guide_flag_defaults_on_everywhere():
    import run_integrated_combined as RIC
    import walk_test as W
    assert OC.GUIDE_DEFAULT == "on"
    assert RIC.build_parser().parse_args([]).guide == "on"
    assert RT.build_parser().parse_args([]).guide == "on"
    assert W.practice_parser().parse_args([]).guide == "on"
    assert RIC.build_parser().parse_args(["--guide", "off"]).guide == "off"


# ═══ TASK C - walk_test practice mode ═════════════════════════════════════════
def test_c_practice_stats_from_a_walk_in():
    """Approach from 0.6 m, overshoot to 0.20 m (0.06 past the near edge), back
    into the band, lock: time, errors at the lock, overshoot, band entries."""
    import walk_test as W
    c = LockConfig()
    ps = W.PracticeStats(c)
    seq = ([(0.6 - 0.04 * k, 0.01) for k in range(11)]          # 0.60 -> 0.20
           + [(0.20 + 0.02 * k, 0.01) for k in range(1, 6)])     # back to 0.30
    for k, (f, l) in enumerate(seq):
        ps.update(0.04 * k, _terms(f, l), locked=(k == len(seq) - 1))
    s = ps.summary()
    assert s["locked"] and abs(s["time_to_lock_s"] - 0.04 * (len(seq) - 1)) < 1e-12
    assert abs(s["overshoot_m"] - 0.06) < 1e-9 and s["band_entries"] == 2
    assert abs(s["lat_err_at_lock_m"] - 0.01) < 1e-12 and abs(s["fwd_at_lock_m"] - 0.30) < 1e-9
    assert abs(s["head_err_at_lock_deg"] - np.degrees(np.arctan2(0.01, 0.30))) < 1e-9
    # nothing after the lock counts
    ps.update(1.0, _terms(0.1, 0.3), locked=True)
    assert ps.summary()["overshoot_m"] == s["overshoot_m"]
    assert "locked after" in W.format_stats(s)
    none = W.PracticeStats(c).summary()
    assert not none["locked"] and np.isnan(none["time_to_lock_s"])


def test_c_practice_headless_locks_and_t_moves_to_the_next_training_seed():
    import walk_test as W
    p = W.practice_parser().parse_args(["--headless", "--start-standoff", str(STANDOFF),
                                        "--tries", "2", "--try-seconds", "2", "--script", "1.0:t"])
    hist = W.run_practice(p)
    train = OC.training_seeds()
    assert [h["seed"] for h in hist] == train[:2]
    for h in hist:
        assert h["locked"] and 0.1 < h["time_to_lock_s"] < 0.5, h
        assert abs(h["lat_err_at_lock_m"]) < OC.default_lock_rule().lat_max
        assert abs(h["head_err_at_lock_deg"]) < OC.default_lock_rule().head_max_deg
        assert h["overshoot_m"] == 0.0 and not h["fell"]
    p = W.practice_parser().parse_args(["--headless", "--start-standoff", str(STANDOFF),
                                        "--tries", "2", "--try-seconds", "1", "--script", "0.5:y"])
    assert [h["seed"] for h in W.run_practice(p)] == [train[0]] * 2


def test_c_practice_refuses_evaluation_and_non_training_seeds():
    import walk_test as W
    for name in OC.HELD_FILES:
        held = OC._load(name)
        if held:
            e = _raises(SystemExit, W.main, ["--headless", "--seed", str(held[0])])
            assert name in str(e)
    outside = max(OC.training_seeds() + sum((OC._load(n) for n in OC.HELD_FILES), [])) + 1
    e = _raises(SystemExit, W.main, ["--headless", "--seed", str(outside)])
    assert "train_seeds.json" in str(e)
    assert 679 in OC._load("select_seeds.json")
    _raises(SystemExit, OC.assert_training_seeds, [679])


def _session_fingerprint(S, info) -> dict:
    from g1_teleop.contact_contract import contract_of
    ns = {k: v for k, v in vars(S.args).items() if k not in ("headless", "seconds")}
    return dict(args=ns, lock_rule=S.lock_rule, keyboard=OC.keyboard_settings(S.loco),
                loco=type(S.loco).__name__, cfg=S.cfg, contract=contract_of(S.model),
                guide=S.guide is not None, model=(S.model.nq, S.model.nv, S.model.nu,
                                                  S.model.ngeom, float(S.model.body_mass.sum())),
                plat=(tuple(S.plat_geo.pickup_xy), tuple(S.plat_geo.goal_xy)),
                fingerprint=info["fingerprint"], start_xy=tuple(info["start_xy"]),
                box=tuple(np.round(info["start"].box_pos, 12)),
                qpos=S.data.qpos.copy().tobytes(), lock_cfg=S.base_lock.cfg,
                cam=(tuple(S.cam.lookat), S.cam.distance, S.cam.azimuth, S.cam.elevation))


def test_c_walk_test_and_teleop_read_identical_config():
    """walk_test's practice session and the live recorder's session, opened by
    their own code for the same training seed and reset by the one shared
    reset: identical arguments, lock rule, keyboard, TeleopConfig, contact
    contract, guide, model, start pose, box spawn and reset state."""
    import run_integrated_combined as RIC
    import deploy as D
    import walk_test as W
    seed = OC.training_seeds()[3]
    p = W.practice_parser().parse_args(["--headless", "--seed", str(seed)])
    S1 = W.open_practice_session(p, seed)
    try:
        f1 = _session_fingerprint(S1, RIC.reset_for_episode(S1, seed))
        assert S1.practice is True
    finally:
        RIC.close_session(S1)
    a = RT.build_parser().parse_args(["--source", "fake", "--replay", _take()[0],
                                      "--seed-list", str(seed), "--headless"])
    a.auto, a.operator = "pass", "tester"
    S2 = RT.open_live_session(a, [seed])
    try:
        f2 = _session_fingerprint(S2, RT.reset_for_episode(S2, a, seed))
        assert S2.practice is False
    finally:
        RIC.close_session(S2)
    diff = [k for k in f1 if f1[k] != f2[k]]
    assert not diff, diff
    assert f1["start_xy"] == OC.START_XY == RT.START_XY == tuple(D.START_XY)
    assert f1["lock_rule"] == LockConfig() and f1["lock_rule"].version == "v3"
    assert W.POLICY_PATH == RIC.POLICY_PATH == OC.POLICY_PATH
    assert W.SCENE_PATH == RIC.SCENE_PATH == OC.SCENE_PATH
    assert W._TICK == spec.PHYSICS_STEPS_PER_TICK
    assert W._fall_z() == RT.FALL_Z
    assert f1["keyboard"] == dict(forward=0.6, backward=0.4, strafe=0.4, turn=0.6,
                                  release_sec=0.25, precision_scale=0.5)


def test_c_practice_keys_do_not_collide_with_the_loop():
    """t / y are not keys of run_integrated_combined's loop, a / d are only
    read on the verdict screen (the loop is not running then)."""
    src = open(os.path.join(ROOT, "run_integrated_combined.py"), encoding="utf-8").read()
    for k in (OC.KEY_PRACTICE_NEXT, OC.KEY_PRACTICE_SAME):
        assert 'ord("%s")' % chr(k) not in src, chr(k)


# ═══ TASK D - the follow camera (g1_teleop/follow_camera.py) ════════════════
def _cam():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    return cam


def _cam_pos(cam):
    """Where MuJoCo puts a free camera: lookat minus distance along the view."""
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(cam.lookat) - cam.distance * fwd


def test_d_default_view_is_over_the_left_shoulder_and_aims_ahead_of_the_pelvis():
    from g1_teleop import follow_camera as FC
    assert FC.PRESET_NAMES == ("behind", "front-side", "side", "top")
    assert FC.DEFAULT_PRESET == "behind" and FC.MODE_DEFAULT == "follow"
    assert 0.4 <= FC.AIM_FORWARD_M <= 0.5 and FC.AIM_HEIGHT_M == 0.85
    d = FC.PRESETS[0]
    assert 1.3 <= d.distance_m <= 1.6
    for yaw_deg, base in ((0.0, (0.6, 0.0)), (90.0, (1.2, -0.3)), (-170.0, (2.0, 1.0))):
        yaw = np.radians(yaw_deg)
        f = FC.FollowCamera()
        cam = _cam()
        f.apply(cam, base, yaw)
        head = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        # aims AHEAD of the pelvis at hand height, not at the pelvis
        assert np.allclose(cam.lookat[:2], np.asarray(base) + FC.AIM_FORWARD_M * head)
        assert cam.lookat[2] == FC.AIM_HEIGHT_M
        rel = _cam_pos(cam)[:2] - np.asarray(base)
        assert rel @ head < -0.05, "the camera must be BEHIND the robot"
        assert rel @ left > 0.2, "and to its LEFT (over the left shoulder)"
        assert _cam_pos(cam)[2] > 1.4, "and above its head"
        assert abs(FC._wrap_deg(cam.azimuth - (yaw_deg + d.azimuth_deg))) < 1e-9
        assert (cam.elevation, cam.distance) == (d.elevation_deg, d.distance_m)


def test_d_keys_are_offsets_and_the_view_still_follows():
    """a/d/w/s/+/- change OFFSETS on the preset: after an adjustment the camera
    still turns and moves with the robot. r = default view, c = next preset
    (offsets cleared). Limits hold."""
    import types
    import run_integrated_combined as RIC
    from g1_teleop import follow_camera as FC
    f = FC.FollowCamera()
    S = types.SimpleNamespace(cam=_cam(), follow=f)
    for k in "aawww++":
        assert RIC.camera_key(S, ord(k), 1.0)
    assert f.offsets() == dict(azimuth_deg=-10.0, elevation_deg=15.0,
                               distance_m=-2 * FC.DISTANCE_STEP_M)
    assert "az -10 el +15" in f.label() and f.label().startswith("behind")
    p = f.preset
    for yaw_deg, base in ((0.0, (0.0, 0.0)), (120.0, (3.0, 1.0))):
        f.snap()
        f.apply(S.cam, base, np.radians(yaw_deg))
        assert abs(FC._wrap_deg(S.cam.azimuth - (yaw_deg + p.azimuth_deg - 10.0))) < 1e-9
        assert S.cam.elevation == p.elevation_deg + 15.0
        assert abs(S.cam.distance - (p.distance_m - 2 * FC.DISTANCE_STEP_M)) < 1e-12
        head = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])
        assert np.allclose(S.cam.lookat[:2], np.asarray(base) + FC.AIM_FORWARD_M * head)
    # limits: tilt and zoom clamp, and the offset reported is the one in force
    for _ in range(60):
        f.tilt(-1)
        f.zoom(+1)
    assert f.view()["elevation_deg"] == -FC.ELEVATION_LIMIT_DEG
    assert f.view()["distance_m"] == FC.DISTANCE_MIN_M
    assert p.elevation_deg + f.d_elevation == -FC.ELEVATION_LIMIT_DEG
    # c cycles in order and clears the offsets; r is the default view from anywhere
    names = []
    for _ in range(5):
        RIC.camera_key(S, FC.KEY_CYCLE_PRESET, 2.0)
        names.append(f.preset.name)
        assert not f.has_offsets()
    assert names == ["front-side", "side", "top", "behind", "front-side"]
    f.rotate(+3)
    RIC.camera_key(S, FC.KEY_RESET_VIEW, 3.0)
    assert f.preset.name == "behind" and not f.has_offsets() and f.label() == "behind"
    assert f.changes[-1]["what"] == "reset" and f.changes[-1]["t_s"] == 3.0
    # other keys are not view keys; the fixed camera keeps its old keys
    assert not RIC.camera_key(S, ord("g")) and not RIC.camera_key(S, ord("t"))
    fixed = types.SimpleNamespace(cam=_cam(), follow=None)
    fixed.cam.azimuth, fixed.cam.elevation, fixed.cam.distance = 0.0, -20.0, 4.5
    for k in "dw-":
        assert RIC.camera_key(fixed, ord(k))
    assert (fixed.cam.azimuth, fixed.cam.elevation, fixed.cam.distance) == (5.0, -15.0, 4.8)
    assert RIC.view_label(fixed) == "fixed" and RIC.view_meta(fixed)["mode"] == "fixed"


def test_d_smoothing_is_bounded_and_wraps_and_a_new_episode_snaps():
    from g1_teleop import follow_camera as FC
    f, cam = FC.FollowCamera(), _cam()
    f.apply(cam, (0.0, 0.0), np.radians(179.0))
    f.apply(cam, (0.0, 0.0), np.radians(-179.0))           # +2 deg through the wrap
    assert abs(FC._wrap_deg(f._yaw_deg - 179.0) - 2.0 * FC.SMOOTHING) < 1e-9
    for _ in range(200):
        f.apply(cam, (1.0, 2.0), np.radians(-179.0))
    assert abs(FC._wrap_deg(f._yaw_deg + 179.0)) < 1e-6 and np.allclose(f._xy, (1.0, 2.0))
    f.rotate(+1, 4.0)
    f.begin_episode()                                       # reset: jump, log cleared, view kept
    assert f.changes == [] and f.d_azimuth == FC.AZIMUTH_STEP_DEG
    f.apply(cam, (5.0, 5.0), 0.0)
    assert f._xy == (5.0, 5.0) and f._yaw_deg == 0.0


def test_d_follow_camera_touches_the_view_camera_only():
    """`apply` is handed numbers and an MjvCamera - never the model or the
    data - and the session helper reads data without writing it."""
    import inspect
    import types
    import run_integrated_combined as RIC
    from g1_teleop import follow_camera as FC
    from g1_teleop.indices import ModelIndex
    assert list(inspect.signature(FC.FollowCamera.apply).parameters) == [
        "self", "cam", "base_xy", "yaw_rad", "smoothing"]
    src = open(FC.__file__, encoding="utf-8").read()
    assert "import mujoco" not in src and "MjData" not in src.split('"""', 2)[2]
    m = _model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    before = [np.array(getattr(d, k)).copy() for k in ("qpos", "qvel", "ctrl", "qacc_warmstart",
                                                        "xpos", "act", "time")]
    S = types.SimpleNamespace(cam=_cam(), follow=FC.FollowCamera(), data=d,
                              ix=ModelIndex.resolve(m))
    for _ in range(3):
        RIC.aim_camera(S)
    RIC.aim_camera(S, snap=True)
    after = [np.array(getattr(d, k)) for k in ("qpos", "qvel", "ctrl", "qacc_warmstart",
                                                "xpos", "act", "time")]
    assert all(np.array_equal(x, y) for x, y in zip(before, after))
    assert S.cam.lookat[2] == FC.AIM_HEIGHT_M


def test_d_follow_vs_fixed_identical_trajectories_and_recorded_states():
    """THE invariance proof for the view: follow camera vs the old fixed camera
    on the grasp fixture, rendered every display frame - every physics step's
    qpos and qvel identical and the recorded 48-D states identical, while the
    two cameras really did differ (the follow camera moved with the robot)."""
    cams_follow, cams_fixed = [], []
    h_follow, n1, s1, _ = _guided_run("on", camera="follow", cams=cams_follow)
    h_fixed, n2, s2, _ = _guided_run("on", camera="fixed", cams=cams_fixed)
    assert n1 == n2 == 7000
    assert h_follow == h_fixed, "the follow camera changed the trajectory"
    assert s1.shape == s2.shape and np.array_equal(s1, s2)
    assert len(cams_follow) == len(cams_fixed) > 300
    assert len(set(cams_fixed)) == 1, "the fixed camera must not move"
    assert len({c[3] for c in cams_follow}) > 50, "the follow camera must move with the robot"
    assert all(c[1] == -55.0 and c[2] == 1.5 for c in cams_follow)
    if _CACHE.get("h_on"):
        assert h_follow == _CACHE["h_on"]


def test_d_view_is_stored_per_episode_with_the_changes_made_in_it():
    """Headless, the view keys pressed on the sim clock: the episode stores the
    preset, the offsets in force at its end and each change with its time; the
    ledger's session row names the view; the recorded states do not depend on
    the view keys."""
    out = os.path.join(_tmp(), "d_view")
    assert RT.main(_argv(out, "1.0:c,2.0:w,2.5:+,3.0:a," + _grasp_script())) == 0
    path = glob.glob(os.path.join(out, "failed", "*.npz"))[0]
    arrays, meta = load_episode(path)
    v = meta["operator_ui"]["view"]
    assert (v["mode"], v["preset"], v["default_preset"]) == ("follow", "front-side", "behind")
    assert v["offsets"] == dict(azimuth_deg=-5.0, elevation_deg=5.0, distance_m=-0.15)
    assert [(c["t_s"], c["what"]) for c in v["changes"]] == [
        (1.0, "preset"), (2.0, "tilt"), (2.5, "zoom"), (3.0, "rotate")]
    assert v["view"]["elevation_deg"] == -20.0 and abs(v["view"]["distance_m"] - 1.75) < 1e-9
    sess = [r for r in _rows(out) if r["event"] == "session"][-1]
    assert sess["view"] == "behind"
    # the same episode without the view keys: identical recorded arrays
    out0 = os.path.join(_tmp(), "d_view0")
    assert RT.main(_argv(out0, _grasp_script())) == 0
    a0, m0 = load_episode(glob.glob(os.path.join(out0, "failed", "*.npz"))[0])
    for k in ("states", "actions", "qpos", "qvel"):
        assert np.array_equal(arrays[k], a0[k]), k
    assert m0["reset_fingerprint"] == meta["reset_fingerprint"]
    assert m0["operator_ui"]["view"]["changes"] == []
    # --camera fixed is stored as such
    outf = os.path.join(_tmp(), "d_fixed")
    assert RT.main(_argv(outf, _grasp_script(), extra=["--camera", "fixed"])) == 0
    _, mf = load_episode(glob.glob(os.path.join(outf, "failed", "*.npz"))[0])
    assert mf["operator_ui"]["view"]["mode"] == "fixed"


def test_d_view_keys_do_not_collide_and_the_flag_defaults_to_follow():
    """c is new; r was already the loop's reset-view key. Neither is a walking
    key, a recorder key, a practice key or a verdict key."""
    import run_integrated_combined as RIC
    import walk_test as W
    from g1_teleop import follow_camera as FC
    from locomotion_input import KEYCODE_STRAFE_LEFT, KEYCODE_STRAFE_RIGHT
    taken = {ord(ch) for ch in "ijklnmpgexqbytuo "} | {10, 13, OC.KEY_ACCEPT, OC.KEY_DISCARD,
                                                        OC.KEY_PRACTICE_NEXT, OC.KEY_PRACTICE_SAME,
                                                        KEYCODE_STRAFE_LEFT, KEYCODE_STRAFE_RIGHT}
    assert FC.KEY_CYCLE_PRESET == ord("c") and FC.KEY_RESET_VIEW == ord("r")
    assert not {FC.KEY_CYCLE_PRESET, FC.KEY_RESET_VIEW} & taken
    for ev in (2490368, 2621440, 2424832, 2555904):          # arrow codes: low byte 0
        assert ev & 0xFF not in (FC.KEY_CYCLE_PRESET, FC.KEY_RESET_VIEW)
    assert RIC.build_parser().parse_args([]).camera == "follow"
    assert RT.build_parser().parse_args([]).camera == "follow"
    assert W.practice_parser().parse_args([]).camera == "follow"
    assert RT.build_parser().parse_args(["--camera", "fixed"]).camera == "fixed"


def test_zzz_cleanup():
    if "tmp" in _CACHE:
        shutil.rmtree(_CACHE.pop("tmp"), ignore_errors=True)
    _CACHE.clear()


def _tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _tests():
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(_tests()) - failed}/{len(_tests())} passed")
    raise SystemExit(1 if failed else 0)
