"""Tests for tools/record_teleop_live.py - the live demonstration recorder - on a
FAKE camera. CPU only, no ZED, no pyzed.

    python tools/test_record_teleop_live.py

The fake take is the synthetic grasp-and-lift built for seed 0 at a head-on
0.32 m start (`--make-fake-take`), replayed on the simulated frame clock
(`--headless`), with the grasp key pressed on the take's own schedule. It
grasps and lifts and never places, so every recorded episode here is a FAILED
episode by `success.evaluate` and lands in failed/ - which is itself one of the
things tested. Everything is written under a temp dir.

What is NOT tested here, because it needs the camera: the threaded grabber's
real frame rate, ZED dropout statistics, wall-clock tick rate and overruns
under a live operator, and the camera metadata read back from the SDK. Those
are docs/LIVE_RECORDING_CHECKLIST.md.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import record_teleop_live as RT                                     # noqa: E402
from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data.ledger import EpisodeLedger                            # noqa: E402
from g1_data.recorder import load_episode, tracking_ok_of           # noqa: E402

SEED = 0
STANDOFF = 0.32
STOP_S = 14.0              # after the lift (11.5 s) and the lock release
#: The commit the session/loop split was made on top of: the OLD main() lives
#: there, and the refactor is checked against it (TR32).
PRE_SPLIT_COMMIT = "473a7d7"
_CACHE: dict = {}


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _tmp() -> str:
    if "tmp" not in _CACHE:
        _CACHE["tmp"] = tempfile.mkdtemp(prefix="live_rec_test_")
    return _CACHE["tmp"]


def _take(no_body: str = "") -> tuple:
    """(path, meta) of the fake take, cached per gap spec."""
    key = "take" + no_body
    if key not in _CACHE:
        path = os.path.join(_tmp(), "take%s.npz" % (no_body.replace("+", "_").replace(",", "-")))
        argv = ["--make-fake-take", path, "--seed-list", str(SEED),
                "--start-standoff", str(STANDOFF)]
        if no_body:
            argv += ["--no-body", no_body]
        assert RT.main(argv) == 0
        from g1_teleop.synthetic_source import load_recording
        _CACHE[key] = (path, load_recording(path).meta)
    return _CACHE[key]


def _argv(out_dir, script, take=None, extra=()):
    take = take or _take()[0]
    return ["--source", "fake", "--replay", take, "--seed-list", str(SEED),
            "--start-standoff", str(STANDOFF), "--headless", "--operator", "tester",
            "--out-dir", out_dir, "--script", script] + list(extra)


def _grasp_script(stop_s=STOP_S) -> str:
    g = _take()[1]["grasp_on_s"]
    return "%.3f:g,%.3f:e" % (g + 0.1, stop_s)


def _episode():
    """The main recorded episode: (out_dir, episode path, arrays, meta)."""
    if "ep" not in _CACHE:
        out = os.path.join(_tmp(), "main")
        assert RT.main(_argv(out, _grasp_script())) == 0
        files = glob.glob(os.path.join(out, "failed", "*.npz"))
        assert len(files) == 1, files
        arrays, meta = load_episode(files[0])
        _CACHE["ep"] = (out, files[0], arrays, meta)
    return _CACHE["ep"]


def _abs(p: str) -> str:
    """A ledger / metadata path (repo-relative, or absolute off the repo's
    drive - g1_data.paths.repo_relpath) back to an absolute path."""
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def _npz_under(d):
    return sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))


# ─── the refactor and observation-only ──────────────────────────────────────
def _trace_of(main_fn, argv) -> tuple:
    """sha256 over qpos after EVERY mj_step of a run, and the step count."""
    import mujoco
    real = mujoco.mj_step
    h, n = hashlib.sha256(), [0]

    def step(m, d, *a, **k):
        r = real(m, d, *a, **k)
        h.update(np.asarray(d.qpos).tobytes())
        h.update(np.asarray(d.qvel).tobytes())
        n[0] += 1
        return r
    mujoco.mj_step = step
    old_argv = sys.argv
    sys.argv = argv
    try:
        try:
            main_fn()
        except SystemExit:
            pass
    finally:
        sys.argv = old_argv
        mujoco.mj_step = real
    return h.hexdigest(), n[0]


def test_split_main_is_the_old_main_step_for_step():
    """run_integrated_combined.main() after the open_session/run_loop/close_session
    split against main() as it was BEFORE the split, taken from git (TR32: the
    reference code on one side), on the --synthetic grasp fixture through the
    weld: every physics step's qpos and qvel identical."""
    try:
        src = subprocess.run(["git", "show", "%s:run_integrated_combined.py" % PRE_SPLIT_COMMIT],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    except Exception as e:                                  # noqa: BLE001
        print("      (skipped: git unavailable: %r)" % e)
        return
    if src.returncode != 0:
        print("      (skipped: commit %s not in this clone)" % PRE_SPLIT_COMMIT)
        return
    path = os.path.join(_tmp(), "ric_pre_split.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.stdout)
    env_keys = ("G1_SCENE_PATH", "G1_POLICY_PATH")
    old_env = {k: os.environ.get(k) for k in env_keys}
    import run_integrated_combined as RIC
    os.environ["G1_SCENE_PATH"], os.environ["G1_POLICY_PATH"] = RIC.SCENE_PATH, RIC.POLICY_PATH
    try:
        spec_ = importlib.util.spec_from_file_location("ric_pre_split", path)
        old = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(old)
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    argv = ["run_integrated_combined.py", "keyboard", "0", "--synthetic", "grasp",
            "--headless", "--start-standoff", "0.32", "--seconds", "11"]
    h_old, n_old = _trace_of(old.main, argv)
    h_new, n_new = _trace_of(RIC.main, argv)
    assert n_old == n_new == 5500, (n_old, n_new)
    assert h_old == h_new, "the split loop diverged from the pre-split main()"


def test_recording_changes_nothing_step_for_step():
    """--invariance: the same seed and fake take with and without the recorder,
    qpos and qvel compared at every physics step (through the weld)."""
    out = os.path.join(_tmp(), "invariance")
    assert RT.main(_argv(out, _grasp_script(12.0), extra=["--invariance"])) == 0
    assert not _npz_under(out), "--invariance must write no episode"


def test_invariance_check_catches_a_recorder_that_writes():
    """Negative control: a recorder that nudges one velocity by 1e-9 on one tick
    must make --invariance fail, or the check above proves nothing."""
    out = os.path.join(_tmp(), "invariance_neg")
    real = RT.LiveRecorder._observe_teleop

    def leaky(self, m, d, L):
        real(self, m, d, L)
        if self.buf is not None and len(self.buf) == 50 and not getattr(self, "_leaked", False):
            self._leaked = True
            d.qvel[0] += 1e-9
    RT.LiveRecorder._observe_teleop = leaky
    try:
        assert RT.main(_argv(out, "4.0:e", extra=["--invariance"])) == 1
    finally:
        RT.LiveRecorder._observe_teleop = real


# ─── the saved file ──────────────────────────────────────────────────────────
def test_failed_episode_is_saved_flagged_and_refused_by_the_loaders():
    out, path, arrays, meta = _episode()
    assert meta["failed"] is True and meta["outcome"]["ok"] is False
    assert meta["outcome"]["grasp"] and meta["outcome"]["lift"], meta["outcome"]
    assert os.path.dirname(path) == os.path.join(out, "failed")
    assert not glob.glob(os.path.join(out, "*.npz")), "a failure must not sit in the train folder"
    eps = DS.scan(os.path.dirname(path))
    ok, why = DS.episode_success(eps[0])
    assert not ok and "success criteria failed" in why, why
    e = _raises(DS.DatasetError, DS.assert_successful, eps)
    assert "did not succeed" in str(e)


def test_saved_episode_passes_spec_validation_and_the_training_loader():
    """`load_episode` (spec version, per-tick lengths incl. the new audit arrays)
    and every dataset/loader check. The loader refuses this episode ONLY because
    it failed (asserted); with that one refusal stubbed it builds windows and
    chunks from the file, so the format is the training format."""
    from g1_data import recorder as REC
    from g1_model import loader as LD
    _, path, arrays, meta = _episode()
    assert meta["spec_version"] == spec.SPEC_VERSION
    n = REC.assert_tick_lengths(arrays, meta)
    assert arrays["states"].shape == (n, spec.STATE_DIM)
    assert arrays["actions"].shape == (n, spec.ACTION_DIM)
    for k in ("frame_seq", "tick_wall_ns", "tracking_ok", "grip_cmd"):
        assert arrays[k].shape[0] == n, k
    spec.assert_velocity_within_clip(np.asarray(arrays["actions"][:, spec.VEL_A], np.float64))
    assert (arrays["phase_labels"] == int(spec.Phase.UNKNOWN)).all()
    # D17 and D22 hold on a teleop file too
    locked = arrays["states"][:, spec.BASE_LOCKED] >= 0.5
    assert locked.any() and (arrays["actions"][locked][:, spec.VEL_A] == 0).all()
    assert np.array_equal(arrays["actions"][:, spec.GRIP_L_A],
                          spec.next_weld_label(arrays["states"][:, spec.GRIP_L]))
    eps = DS.scan(os.path.dirname(path))
    DS.assert_uniform(eps)
    DS.assert_uniform_lock_rule(eps)
    DS.assert_uniform_grip_label(eps)
    DS.assert_no_leak(eps)
    info = REC.source_info(meta["source"])
    assert info["real"] is False and meta["source"] == REC.SOURCE_TELEOP_LIVE_FAKE

    st, ac, _ = DS.fit_norm_stats(eps, split="train")
    cfg = LD.LoaderConfig(chunk_size=10, obs_window=12, tracking=LD.TrackingPolicy())
    e = _raises(DS.DatasetError, LD.ChunkDataset, eps, cfg, st, ac, None)
    assert "did not succeed" in str(e), e
    real = DS.assert_successful
    DS.assert_successful = lambda eps_: None
    try:
        ds = LD.ChunkDataset(eps, cfg, st, ac, None)
    finally:
        DS.assert_successful = real
    assert len(ds) > 0
    item = ds[0]
    shapes = {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")}
    assert any(s[-1] == spec.STATE_DIM for s in shapes.values()), shapes
    assert any(s[-1] == spec.ACTION_DIM for s in shapes.values()), shapes
    assert ds.real_demonstration == [False] and ds.tracking_is_measured == [True]


def test_metadata_is_complete():
    _, path, arrays, meta = _episode()
    want = ("seed", "box_spawn_xy", "heldout", "spec_version", "lock_rule", "lock_rule_id",
            "grip_label", "contact_contract", "d18", "source", "operator", "camera",
            "git_commit", "git_dirty", "timing", "reset_fingerprint", "n_ticks",
            "weld_engage_tick", "lock_engage_tick", "end_reason", "outcome", "failed",
            "degraded_ticks", "dropout_rate", "keypoints_file", "termination")
    missing = [k for k in want if k not in meta]
    assert not missing, missing
    assert meta["spec_version"] == "g1-spec-1.2.0"
    assert meta["grip_label"] == spec.GRIP_LABEL == "next_weld_v1"
    assert meta["lock_rule"]["version"] == meta["lock_rule_id"] == "v2"
    assert meta["source"] == "teleop_live_fake" and meta["operator"] == "tester"
    assert meta["d18"]["hand_pickup_exclusion"] is True
    from g1_teleop.contact_contract import expected_contract
    from g1_teleop.config import TeleopConfig
    assert meta["contact_contract"] == expected_contract(TeleopConfig().contact)
    cam = meta["camera"]
    assert cam["fake"] is True and cam["serial_number"] == -1 and "sdk_version" in cam
    assert "body_tracking" in cam and cam["replay"]["sha256"]
    t = meta["timing"]
    assert t["paced"] is False and t["tick_rate_hz"] > 0 and "overrun_ticks" in t
    assert meta["end_reason"] == "operator_stop"
    assert meta["termination"]["max_ticks"] == 900
    assert meta["weld_engage_tick"] > meta["lock_engage_tick"] >= 0
    assert len(meta["reset_fingerprint"]) == 32
    kp = _abs(meta["keypoints_file"])
    from g1_teleop.synthetic_source import load_recording
    rec = load_recording(kp)
    assert len(rec) > 0 and rec.meta["seq_at_arm"] == 0


def test_raw_keypoints_replay_to_the_frames_the_arms_used():
    """The episode's own keypoint file is the take it was driven by: frame_seq s
    is the s-th delivered row, and those rows equal the take's frames."""
    _, _, arrays, meta = _episode()
    from g1_teleop.synthetic_source import load_recording
    kp = load_recording(_abs(meta["keypoints_file"]))
    take = load_recording(_take()[0])
    seq = arrays["frame_seq"]
    assert (np.diff(seq) >= 0).all() and seq.max() > 0
    from g1_teleop.synthetic_source import STATUS_GRAB_FAILED
    delivered = np.flatnonzero(kp.status != STATUS_GRAB_FAILED)
    s = int(seq.max())
    row = delivered[s - 1]
    assert np.array_equal(kp.keypoints[row], take.keypoints[s - 1], equal_nan=True)


def test_ledger_rows_for_a_failed_and_a_discarded_episode():
    out, path, _, meta = _episode()
    rows = EpisodeLedger(os.path.join(out, "ledger.jsonl")).rows()
    assert [r["event"] for r in rows] == ["session", "issue", "discard"], rows
    s, d = rows[0], rows[2]
    assert s["source"] == "teleop_live_fake" and s["lock_rule"] == "v2" and s["operator"] == "tester"
    assert d["seed"] == SEED and d["reason"].startswith("failed")
    assert _abs(d["saved"]) == os.path.normpath(path)
    assert os.path.exists(_abs(d["keypoints"]))
    assert d["checks"]["grasp"] is True and d["checks"]["place"] is False
    led = EpisodeLedger(os.path.join(out, "ledger.jsonl"))
    assert led.pending([SEED]) == [SEED], "a failed seed stays pending (re-recordable)"


def test_discard_writes_nothing():
    out = os.path.join(_tmp(), "discard")
    assert RT.main(_argv(out, "2.0:x")) == 0
    assert not _npz_under(out), _npz_under(out)
    rows = EpisodeLedger(os.path.join(out, "ledger.jsonl")).rows()
    assert rows[-1]["event"] == "discard" and rows[-1]["reason"] == "operator discard"
    assert rows[-1]["saved"] is None


def test_accept_path_writes_to_the_train_folder_and_the_ledger():
    """A successful verdict (forced: the fixture never places) goes to the
    top-level folder with an ACCEPT row, and the seed is no longer pending."""
    out = os.path.join(_tmp(), "accept")
    real = RT.SU.evaluate

    def forced(arrays, meta, cfg=None, geo=None):
        r = real(arrays, meta, cfg, geo)
        r.update(episode_success=True, place=True, attributed_to="")
        return r
    RT.SU.evaluate = forced
    try:
        assert RT.main(_argv(out, "2.0:e")) == 0
    finally:
        RT.SU.evaluate = real
    files = glob.glob(os.path.join(out, "*.npz"))
    assert len(files) == 1 and not glob.glob(os.path.join(out, "failed", "*.npz"))
    led = EpisodeLedger(os.path.join(out, "ledger.jsonl"))
    r = led.rows()[-1]
    assert r["event"] == "accept" and r["seed"] == SEED and r["keypoints"]
    assert led.pending([SEED]) == []
    # and a second session on that folder has nothing left to do
    assert RT.main(_argv(out, "2.0:e")) == 0
    assert len(glob.glob(os.path.join(out, "*.npz"))) == 1


# ─── tracking, D18, lock rules, pyzed ────────────────────────────────────────
def test_dropout_ticks_are_marked_and_the_arms_hold():
    """A 50-frame NO-BODY gap (1.70 s from frame 60) in the take: the ticks past
    NO_BODY_HOLD_S inside it are tracking_ok False, the arm command is frozen,
    and the raw keypoint file shows the gap as no_body rows."""
    import run_integrated_combined as RIC
    take, _ = _take("60+50")
    out = os.path.join(_tmp(), "dropout")
    assert RT.main(_argv(out, "5.0:e", take=take)) == 0
    path = glob.glob(os.path.join(out, "failed", "*.npz"))[0]
    arrays, meta = load_episode(path)
    ok = tracking_ok_of(arrays)
    hz = 1.0 / (RIC.FRAME_STEPS * RIC.SIM_DT)
    lo, hi = int(60 / hz * 25), int(110 / hz * 25)
    bound = int(RIC.NO_BODY_HOLD_S * 25)
    assert not ok[lo + bound + 2:hi - 1].any()
    assert ok[8:lo - 2].all()
    assert meta["degraded_ticks"] == int((~ok).sum()) > 0
    d = np.abs(np.diff(arrays["actions"][lo + 2:hi - 1, :14], axis=0)).max()
    assert d == 0, d
    from g1_teleop.synthetic_source import load_recording, STATUS_NO_BODY
    kp = load_recording(_abs(meta["keypoints_file"]))
    assert int((kp.status == STATUS_NO_BODY).sum()) == 50


def test_bprime_key_is_blocked_while_recording_and_logged():
    out = os.path.join(_tmp(), "bprime")
    assert RT.main(_argv(out, "1.0:b,1.5:b,2.0:e")) == 0
    arrays, meta = load_episode(glob.glob(os.path.join(out, "failed", "*.npz"))[0])
    assert meta["d18"]["toggle_blocked_attempts_s"] == [1.0, 1.5]
    assert meta["d18"]["hand_pickup_exclusion"] is True
    from g1_teleop.contact_contract import expected_contract
    from g1_teleop.config import TeleopConfig
    assert meta["contact_contract"] == expected_contract(TeleopConfig().contact)


def test_mixed_lock_rules_are_refused():
    """The lock rule comes from the CLI and is stored; a session refuses an
    out-dir holding another rule, and the dataset refuses a mix."""
    out = os.path.join(_tmp(), "rule_v1")
    assert RT.main(_argv(out, "1.0:e", extra=["--lock-rule", "v1"])) == 0
    v1 = glob.glob(os.path.join(out, "failed", "*.npz"))[0]
    assert DS.read_meta(v1)["lock_rule"]["version"] == "v1"
    e = _raises(DS.DatasetError, RT.main, _argv(out, "1.0:e", extra=["--lock-rule", "v2"]))
    assert "lock rule" in str(e)
    _, v2, _, _ = _episode()
    e = _raises(DS.DatasetError, DS.assert_uniform_lock_rule,
                [DS.Episode(v1, DS.read_meta(v1)), DS.Episode(v2, DS.read_meta(v2))])
    assert "mixes lock rules" in str(e)


def test_fake_source_never_imports_pyzed():
    """`sys.modules[name] = None` makes any import of it raise, so this holds even
    on the laptop where pyzed IS installed."""
    saved = {k: sys.modules.get(k) for k in ("pyzed", "pyzed.sl")}
    for k in saved:
        sys.modules[k] = None
    try:
        out = os.path.join(_tmp(), "nopyzed")
        assert RT.main(_argv(out, "1.0:e")) == 0
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert glob.glob(os.path.join(out, "failed", "*.npz"))


def test_recorded_fake_episode_replays_like_a_scripted_one():
    """tools/replay_check.replay, unchanged: reset fingerprint, then the recorded
    actions open loop; the manipulation (weld tick, box while locked) reproduces."""
    import replay_check as RC
    _, path, _, meta = _episode()
    r = RC.replay(path, verbose=False)
    assert r["reset_fingerprint_ok"], "reset fingerprint mismatch"
    assert r["weld_engaged"] and r["weld_tick_delta"] <= RC.TICK_TOL, r["weld_tick_delta"]
    assert r["manipulation_ok"], (r["weld_tick_delta"], r["locked"])


def test_cli_refusals():
    _raises(SystemExit, RT.main, ["--source", "zed"])                       # no operator
    _raises(SystemExit, RT.main, ["--source", "zed", "--operator", "x", "--headless"])
    _raises(SystemExit, RT.main, ["--source", "fake"])                      # no --replay
    _raises(SystemExit, RT.assert_seeds_are_training_seeds, [0, 679])       # a select seed
    RT.assert_seeds_are_training_seeds([0, 1, 2])
    from g1_data.recorder import NamespaceMismatch, NS_COLLECTION
    # a fake session may not write into the real collection namespace
    _raises(NamespaceMismatch, RT.main, _argv(NS_COLLECTION, "1.0:e"))


def test_zzz_cleanup():
    """Last (sorted): remove the temp tree."""
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
