"""Tests for tools/video_side_by_side.py and tools/video_common.py on SYNTHETIC
frames: no simulation, no rendering, no checkpoint. The deploy re-run is
replaced by a fake that writes a synthetic MP4 and a deploy-shaped record, so
everything from the results folder to the index is exercised.

    python tools/test_video_side_by_side.py          (or pytest)

Where MuJoCo imports (the laptop), two more checks run against the real code:
the video constants equal tools/episode_view.py's, and the command printed by
--plan is the one tools/eval_scripted.deploy runs.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import video_common as VC                                           # noqa: E402
import video_side_by_side as SBS                                    # noqa: E402

SMALL = (320, 180)          # synthetic source size: fast to encode
BAR = 11                    # the 44 px bar at 1/4 scale
COLORS = [(40, 90, 140), (90, 140, 40), (140, 40, 90)]


def _skip(msg: str) -> None:
    try:
        import pytest
    except ImportError:
        print("      skip: %s" % msg)
        return
    pytest.skip(msg)


def _mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


def synth_video(path: str, n: int, color, size=SMALL, bar: int = BAR) -> str:
    """n frames: a solid colour, a black top bar (deploy's overlay) and a white
    square whose x position encodes the frame index (x = 4 + 2 k)."""
    w, h = size
    with VC.Mp4Writer(path, w, h) as wr:
        for k in range(n):
            f = np.full((h, w, 3), color, np.uint8)
            f[:bar] = 0
            x = 4 + 2 * k
            f[h // 2 - 8:h // 2 + 8, x:x + 16] = 255
            wr.write(f)
    return path


def row(seed=7, ok=True, fr="", ticks=30, sha="h", **kw) -> dict:
    r = dict(seed=seed, success=ok, fail_reason=fr, ticks=ticks, kind="act",
             end_reason="success" if ok else "time_limit", box_xy=[1.5, 0.02], heldout=False,
             placement_error_m=0.03, final_box_goal_m=0.03, phase_reached="VERIFY" if ok else "SETTLE",
             weld_engage_tick=10, weld_release_tick=20 if ok else -1, lock_ticks=[5], release_ticks=[],
             phase_boundaries=dict(SETTLE=0, REACH=5, GRASP=11, LIFT=14, MOVE=16, LOWER=18,
                                   RELEASE=21, VERIFY=ticks),
             trajectory_sha256=sha,
             header=dict(ckpt=dict(file="ckpt_step0100000.pt", step=100000), K=100,
                         lock_rule=dict(version="v3"), training_data_lock_rule="v2",
                         lock_rule_mismatch_allowed=True, temporal_ensemble=True,
                         ensemble_horizon=100, walkin="policy", max_ticks=900, device="cuda",
                         train_seeds_evaluated=[], git_commit="abc"))
    r.update(kw)
    return r


def _mean(img, y0, y1, x0, x1):
    return img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def _frames(path):
    return list(VC.iter_frames(path))


# ─── per-tick state ──────────────────────────────────────────────────────────
def test_phase_at_follows_the_boundaries():
    b = dict(SETTLE=0, REACH=5, GRASP=11, LIFT=14, MOVE=16, LOWER=18, RELEASE=21, VERIFY=30)
    got = [SBS.phase_at(b, t, 30) for t in (0, 4, 5, 10, 11, 15, 17, 20, 21, 29)]
    assert got == ["SETTLE", "SETTLE", "REACH", "REACH", "GRASP", "LIFT", "MOVE", "LOWER",
                   "RELEASE", "RELEASE"], got
    # a collapsed segment (REACH == GRASP) shows the later phase, as label_episode
    # writes it; a phase never reached sits at the episode length and never shows
    c = dict(SETTLE=0, REACH=8, GRASP=8, LIFT=30, MOVE=30, LOWER=30, RELEASE=30, VERIFY=30)
    assert [SBS.phase_at(c, t, 30) for t in (7, 8, 29)] == ["SETTLE", "GRASP", "GRASP"]
    assert SBS.phase_at(None, 3, 30) == "?"


def test_weld_and_lock_as_the_policy_observes_them():
    r = row(weld_engage_tick=10, weld_release_tick=20, lock_ticks=[5, 25], release_ticks=[15])
    assert [SBS.weld_on(r, t) for t in (10, 11, 20, 21)] == [False, True, True, False]
    assert [SBS.lock_on(r, t) for t in (4, 5, 14, 15, 24, 25)] == [False, True, True, False, False, True]
    never = row(weld_engage_tick=-1, weld_release_tick=-1, lock_ticks=[], release_ticks=[])
    assert not any(SBS.weld_on(never, t) or SBS.lock_on(never, t) for t in range(30))
    held = row(weld_engage_tick=10, weld_release_tick=-1)
    assert SBS.weld_on(held, 899)


# ─── re-running the evaluation ───────────────────────────────────────────────
def test_deploy_options_repeat_the_evaluation():
    o = SBS.deploy_options(row()["header"])
    assert o == dict(ckpt="ckpt_step0100000.pt", lock_rule="v3", allow_mismatch=True,
                     extra=["--max-ticks", "900"]), o
    h = dict(row()["header"], temporal_ensemble=False, lock_rule_mismatch_allowed=False,
             walkin="scripted", max_ticks=None)
    assert SBS.deploy_options(h)["extra"] == ["--no-ensemble", "--walkin", "scripted"]
    assert SBS.deploy_options(h)["allow_mismatch"] is False
    h = dict(row()["header"], ensemble_horizon=10)
    assert "--ensemble-horizon" in SBS.deploy_options(h)["extra"]
    old = {k: v for k, v in row()["header"].items() if k != "lock_rule_mismatch_allowed"}
    assert SBS.deploy_options(old)["allow_mismatch"] is True          # v3 eval of v2 data
    assert SBS.deploy_options(dict(old, training_data_lock_rule="v3"))["allow_mismatch"] is False
    bc = dict(row()["header"], K=1, temporal_ensemble=False, ensemble_horizon=None)
    assert SBS.deploy_options(bc)["extra"] == ["--max-ticks", "900"]   # K = 1: nothing to ensemble
    for bad in (dict(row()["header"], ckpt={}), dict(row()["header"], lock_rule={})):
        try:
            SBS.deploy_options(bad)
        except SystemExit:
            continue
        raise AssertionError("an unreproducible header was accepted")


def test_plan_command_is_what_eval_scripted_deploy_runs():
    if not _mujoco():
        return _skip("needs MuJoCo (eval_scripted imports g1_data)")
    import eval_scripted as E
    tmp = tempfile.mkdtemp()
    seen = {}

    def fake_call(cmd, stdout=None, stderr=None, cwd=None):
        seen["cmd"] = cmd
        out = cmd[cmd.index("--out") + 1]
        with open(out, "w") as fh:
            fh.write(json.dumps(dict(row(), video="v.mp4")) + "\n")
        return 0
    real = E.subprocess.call
    E.subprocess.call = fake_call
    try:
        opts = SBS.deploy_options(row()["header"])
        SBS.run_deploy("RUN", opts, 7, tmp)
    finally:
        E.subprocess.call = real
        shutil.rmtree(tmp, ignore_errors=True)
    planned = SBS.deploy_command("RUN", opts, 7, tmp)
    ran = seen["cmd"][2:]                                 # drop python -u
    ran[ran.index("--out") + 1] = ran[ran.index("--out") + 1][:-len(".part")]
    assert os.path.basename(ran[0]) == "deploy.py" and ran[1:] != []
    assert sorted(ran[1:]) == sorted(planned[2:]), (ran, planned)


# ─── geometry and composition ────────────────────────────────────────────────
def test_layout_one_row_up_to_three_then_two_rows():
    g3 = SBS.layout(3, None, 640)
    assert (g3["cols"], g3["rows"], g3["W"]) == (3, 1, 1920)
    assert g3["img_h"] == 338 and g3["W"] % 2 == 0 and g3["H"] % 2 == 0   # 640 * 676 / 1280
    g4 = SBS.layout(4, None, 640)
    assert (g4["cols"], g4["rows"]) == (2, 2)
    cells = {c for c in g4["cells"]}
    assert len(cells) == 4
    assert SBS.layout(5, 5, 320)["cols"] == 5


def _compose(tmp, lengths, oks, hashes, speed=1.0, tail_s=0.2):
    srcs = []
    for i, (n, ok, hm) in enumerate(zip(lengths, oks, hashes)):
        p = synth_video(os.path.join(tmp, "src%d.mp4" % i), n, COLORS[i])
        srcs.append((VC.iter_frames(p), row(ok=ok, fr="" if ok else "GRASP", ticks=n),
                     "model %d" % i, "40 demos, train seed 0", hm))
    out = os.path.join(tmp, "sbs.mp4")
    info = SBS.compose(srcs, out, "title", "footer", panel_w=320, speed=speed, tail_s=tail_s,
                       crop_top=BAR, src_size=SMALL)
    return out, info


def test_compose_one_clock_freezes_ended_panels_and_marks_a_mismatch():
    tmp = tempfile.mkdtemp()
    try:
        out, info = _compose(tmp, (20, 30, 25), (False, True, True), (True, True, False))
        tail = int(round(0.2 * VC.TICK_HZ))
        assert info["panel_frames"] == [20, 30, 25]
        assert info["frames"] == 30 + tail, info
        frames = _frames(out)
        assert len(frames) == info["frames"]
        g = info["geometry"]
        assert frames[0].shape == (g["H"], g["W"], 3)
        y0 = SBS.TITLE_H + g["header_h"]

        def pic(f, i):                                   # panel i's picture, square-free corner
            x0 = i * g["panel_w"]
            return _mean(f, y0 + 4, y0 + 30, x0 + 8, x0 + 60)
        for i in range(3):                               # colours land in their own panels
            assert np.abs(pic(frames[2], i) - COLORS[i]).max() < 25, (i, pic(frames[2], i))
        # panel 0 ended at frame 20: darkened from then on, live before
        assert pic(frames[25], 0).mean() < 0.6 * pic(frames[10], 0).mean()
        assert np.abs(pic(frames[25], 1) - COLORS[1]).max() < 25        # panel 1 still live

        def square_x(f, i):                              # the moving square's left edge
            x0 = i * g["panel_w"]
            band = f[y0 + g["img_h"] // 2, x0:x0 + g["panel_w"]].astype(int)
            hot = np.flatnonzero(band.min(axis=1) > 120)
            return int(hot[0]) if len(hot) else -1
        assert square_x(frames[25], 0) == square_x(frames[29], 0) > square_x(frames[10], 0)
        assert square_x(frames[29], 1) > square_x(frames[10], 1)
        # outcome badges: red for the failure, green for the successes
        hb = SBS.TITLE_H + 3

        def badge(f, i):                                 # the badge's right padding, no text
            x1 = i * g["panel_w"] + g["panel_w"] - 4 - 1
            return _mean(f, hb + 2, hb + 10, x1 - 4, x1)
        assert badge(frames[5], 0)[2] > 150 > badge(frames[5], 0)[1]          # BGR: red
        assert badge(frames[5], 1)[1] > 120 > badge(frames[5], 1)[2]          # green
        # the mismatched panel has a red frame, the matched ones do not
        x2 = 2 * g["panel_w"]
        left = _mean(frames[5], y0 + 40, y0 + 60, x2, x2 + 2)
        assert left[2] > 150 and left[0] < 120, left
        assert np.abs(_mean(frames[5], y0 + 40, y0 + 60, g["panel_w"], g["panel_w"] + 2)
                      - COLORS[1]).max() < 30
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_speed_skips_source_frames():
    tmp = tempfile.mkdtemp()
    try:
        _, info = _compose(tmp, (20, 31), (True, True), (True, True), speed=2.0, tail_s=0.0)
        assert info["frames"] == 16, info                   # source frames 0, 2, ..., 30
        assert info["panel_frames"] == [20, 31]
        try:
            _compose(tmp, (5,), (True,), (True,), speed=0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("speed 0 accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_writer_leaves_no_finished_looking_file_on_error():
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "x.mp4")
        try:
            with VC.Mp4Writer(p, 64, 64) as w:
                w.write(np.zeros((64, 64, 3), np.uint8))
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert os.listdir(tmp) == [], os.listdir(tmp)
        try:
            VC.Mp4Writer(p, 63, 64)
        except ValueError:
            pass
        else:
            raise AssertionError("odd size accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── results folders and labels ──────────────────────────────────────────────
def make_results(root: str, runs, seeds=(7, 8), exp="exp1", hashes=None, meta=True) -> str:
    """<root>/<run>/metadata.json + <root>/eval/<run>/{exp}.jsonl, selection.json."""
    for i, run in enumerate(runs):
        os.makedirs(os.path.join(root, "eval", run), exist_ok=True)
        if meta:
            os.makedirs(os.path.join(root, run), exist_ok=True)
            json.dump(dict(data=dict(seeds=list(range(40))),
                           config=dict(notes=dict(model=VC.parse_run(run)["model"] or "act",
                                                  data_paths=dict(episodes="recordings/teleop_v1")))),
                      open(os.path.join(root, run, "metadata.json"), "w"))
        with open(os.path.join(root, "eval", run, "%s.jsonl" % exp), "w") as fh:
            for s in seeds:
                sha = (hashes or {}).get((run, s), "%s-%d" % (run, s))
                fh.write(json.dumps(row(seed=s, ok=(i + s) % 2 == 0, fr="" if (i + s) % 2 == 0 else "GRASP",
                                        sha=sha, ticks=20 + 3 * i)) + "\n")
        json.dump(dict(chosen="ckpt_step0100000.pt"),
                  open(os.path.join(root, "eval", run, "selection.json"), "w"))
    return root


def test_results_folder_or_eval_folder_and_duplicate_runs():
    tmp = tempfile.mkdtemp()
    try:
        a = make_results(os.path.join(tmp, "a"), ["act_n40_seed0"])
        r1, r2 = VC.open_results(a), VC.open_results(os.path.join(a, "eval"))
        assert r1 == r2 and r1.root == os.path.abspath(a)
        assert list(VC.find_runs([r1])) == ["act_n40_seed0"]
        b = make_results(os.path.join(tmp, "b"), ["act_n40_seed0"])
        try:
            VC.find_runs([r1, VC.open_results(b)])
        except SystemExit:
            pass
        else:
            raise AssertionError("a run in two results folders was accepted")
        try:
            VC.open_results(tmp)
        except SystemExit:
            pass
        else:
            raise AssertionError("a folder without an evaluation was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_labels_from_run_names_and_metadata():
    assert VC.run_label("act_lstm_160_seed0") == "ACT-LSTM (W_o 12)"
    assert VC.run_label("bc_b256_n40_seed1", kind="bc") == "BC (b256)"     # name beats kind
    assert VC.run_label("act_lstm_full_n40_seed2") == "ACT-LSTM (episode)"
    assert VC.run_label("act_orig_n40_seed0") == "ACT (W_o 1)"
    assert VC.run_label("mystery", kind="act") == "ACT (W_o 12)"
    assert VC.run_label("act_160_seed0", override="ACT") == "ACT"
    assert VC.run_detail("act_n40_seed2") == "40 demos, train seed 2"
    assert VC.run_detail("act_lstm_teleop_seed0") == "teleop, train seed 0"
    assert VC.run_detail("act_n40_seed2", dict(data=dict(seeds=list(range(37))))) == \
        "37 demos, train seed 2"                                   # metadata wins
    assert VC.parse_run_spec("act_160_seed0=ACT-12") == ("act_160_seed0", "ACT-12")
    assert VC.outcome_text(dict(success=False, fail_reason="fell")) == "FAIL: fell"
    assert VC.ascii_text("a × b — c") == "a x b - c"


def test_constants_match_episode_view():
    if not _mujoco():
        return _skip("needs MuJoCo (episode_view imports it)")
    import inspect
    import episode_view as EV
    assert (EV.VIDEO_FPS, EV.VIDEO_W, EV.VIDEO_H) == (VC.TICK_HZ, VC.SRC_W, VC.SRC_H)
    assert "(VIDEO_W, %d)" % VC.OVERLAY_BAR_PX in inspect.getsource(EV._VideoSink._run)


# ─── the CLI, end to end, with a fake deploy ─────────────────────────────────
class FakeDeploy:
    """Stands in for run_deploy: writes deploy's episodes.jsonl and a synthetic
    video into the panel folder. `rehash` = {(run, seed): sha} to simulate a
    re-run that does not reproduce the stored episode."""

    def __init__(self, results_root: str, rehash=None):
        self.root, self.rehash, self.calls = results_root, rehash or {}, []

    def __call__(self, run_dir, opts, seed, pdir):
        run = os.path.basename(run_dir)
        self.calls.append((run, seed, opts["ckpt"]))
        stored = VC.load_exp(VC.open_results(self.root), run, "exp1")[seed]
        os.makedirs(pdir, exist_ok=True)
        vid = synth_video(os.path.join(pdir, "%s_seed%04d.mp4" % (run, seed)), stored["ticks"],
                          COLORS[len(self.calls) % 3], size=(VC.SRC_W // 4, VC.SRC_H // 4))
        r = dict(stored, video=vid,
                 trajectory_sha256=self.rehash.get((run, seed), stored["trajectory_sha256"]))
        with open(os.path.join(pdir, "episodes.jsonl"), "w") as fh:
            fh.write(json.dumps(r) + "\n")
        return r


def _run_main(argv, fake):
    real_d, real_b = SBS.run_deploy, SBS.heavy_busy
    SBS.run_deploy, SBS.heavy_busy = fake, (lambda: [])
    try:
        return SBS.main(argv)
    finally:
        SBS.run_deploy, SBS.heavy_busy = real_d, real_b


RUNS = ["bc_n40_seed0", "act_n40_seed0", "act_lstm_n40_seed0"]


def test_main_end_to_end_hash_checked_and_restart_safe():
    tmp = tempfile.mkdtemp()
    try:
        res = make_results(os.path.join(tmp, "runs_teleop"), RUNS)
        out = os.path.join(tmp, "videos")
        fake = FakeDeploy(res)
        args = ["--results", res, "--exp", "exp1", "--seed", "7", "8", "--runs"] + RUNS + [
            "--out-dir", out, "--panel-width", "160", "--tail-s", "0"]
        assert _run_main(args, fake) == 0
        assert len(fake.calls) == 6 and {c[2] for c in fake.calls} == {"ckpt_step0100000.pt"}
        idx = list(csv.DictReader(open(os.path.join(out, "side_by_side_index.csv"))))
        assert [(r["exp"], r["seed"], r["hash_matches"], r["n_panels"]) for r in idx] == [
            ("exp1", "7", "3", "3"), ("exp1", "8", "3", "3")], idx
        assert idx[0]["labels"] == "BC;ACT (W_o 12);ACT-LSTM (W_o 12)"
        pj = os.path.join(out, "exp1", "seed0007", "panels_%s.json" % SBS.default_tag(RUNS))
        rec = json.load(open(pj))
        assert [p["source"] for p in rec["panels"]] == ["deployed"] * 3
        assert all(p["frames_match_ticks"] for p in rec["panels"])
        assert rec["frames"] == max(p["video_frames"] for p in rec["panels"])   # tail 0
        assert len(_frames(rec["video"])) == rec["frames"]
        # restart: every panel is cached, nothing is deployed again
        assert _run_main(args, fake) == 0 and len(fake.calls) == 6
        rec2 = json.load(open(pj))
        assert [p["source"] for p in rec2["panels"]] == ["cached"] * 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_main_reports_a_hash_mismatch_with_exit_code_1():
    tmp = tempfile.mkdtemp()
    try:
        res = make_results(os.path.join(tmp, "r"), RUNS)
        out = os.path.join(tmp, "videos")
        fake = FakeDeploy(res, rehash={("act_n40_seed0", 7): "different"})
        code = _run_main(["--results", res, "--exp", "exp1", "--seed", "7", "--runs"] + RUNS +
                         ["--out-dir", out, "--panel-width", "160", "--tail-s", "0"], fake)
        assert code == 1
        rec = json.load(open(glob_one(out, "panels_*.json")))
        assert [p["hash_match"] for p in rec["panels"]] == [True, False, True]
        assert rec["panels"][1]["stored"]["trajectory_sha256"] != rec["panels"][1]["rerun"]["trajectory_sha256"]
        idx = list(csv.DictReader(open(os.path.join(out, "side_by_side_index.csv"))))
        assert idx[0]["hash_matches"] == "2"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def glob_one(root, pattern):
    import glob
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    assert len(hits) == 1, hits
    return hits[0]


def test_reuse_record_exp_videos_output_and_refuse_unevaluated_seeds():
    tmp = tempfile.mkdtemp()
    try:
        res = make_results(os.path.join(tmp, "r"), RUNS[:2])
        reuse = os.path.join(tmp, "scripted_exp")
        d = os.path.join(reuse, "act_n40_seed0", "exp1")
        os.makedirs(d)
        stored = VC.load_exp(VC.open_results(res), "act_n40_seed0", "exp1")[7]
        vid = synth_video(os.path.join(d, "a.mp4"), stored["ticks"], COLORS[0],
                          size=(VC.SRC_W // 4, VC.SRC_H // 4))
        with open(os.path.join(d, "episodes.jsonl"), "w") as fh:
            fh.write(json.dumps(dict(stored, video=vid)) + "\n")
        fake = FakeDeploy(res)
        out = os.path.join(tmp, "v")
        assert _run_main(["--results", res, "--exp", "exp1", "--seed", "7", "--runs"] + RUNS[:2] +
                         ["--out-dir", out, "--reuse", reuse, "--panel-width", "160"], fake) == 0
        assert [c[0] for c in fake.calls] == ["bc_n40_seed0"]          # act came from --reuse
        rec = json.load(open(glob_one(out, "panels_*.json")))
        assert [p["source"] for p in rec["panels"]] == ["deployed", "reused"]
        try:
            _run_main(["--results", res, "--exp", "exp1", "--seed", "99", "--runs"] + RUNS[:2] +
                      ["--out-dir", out], fake)
        except SystemExit as e:
            assert "no exp1 record for seed 99" in str(e)
        else:
            raise AssertionError("an unevaluated seed was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_plan_runs_nothing_and_picks_drive_the_jobs():
    tmp = tempfile.mkdtemp()
    try:
        res = make_results(os.path.join(tmp, "r"), RUNS)
        picks = os.path.join(tmp, "picks.json")
        json.dump(dict(results=[res], runs=RUNS, picks=[
            dict(category="contrast", exp="exp1", seed=7, why="w1"),
            dict(category="typical_failure", exp="exp1", seed=8, why="w2"),
            dict(category="all_succeed", exp="exp1", seed=7, why="dup")]), open(picks, "w"))
        fake = FakeDeploy(res)
        out = os.path.join(tmp, "v")
        assert _run_main(["--picks", picks, "--out-dir", out, "--plan"], fake) == 0
        assert fake.calls == [] and not os.path.exists(os.path.join(out, "_panels"))

        class A:
            pass
        a = A()
        a.runs, a.picks, a.category, a.exp, a.limit, a.seed = None, picks, ["contrast"], None, 0, None
        jobs = SBS.jobs_from_args(a)
        assert [(j["exp"], j["seed"], j["category"]) for j in jobs] == [("exp1", 7, "contrast")]
        a.category = None
        assert [j["seed"] for j in SBS.jobs_from_args(a)] == [7, 8]   # seed 7 only once
        assert _run_main(["--picks", picks, "--category", "contrast", "--out-dir", out,
                          "--panel-width", "160"], fake) == 0
        rec = json.load(open(glob_one(out, "panels_*.json")))
        assert (rec["category"], rec["why"]) == ("contrast", "w1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
            print("FAIL  %s\n        %s: %s" % (name, type(e).__name__, e))
        else:
            print("ok    %s" % name)
    print("\n%d/%d passed" % (len(_tests()) - failed, len(_tests())))
    raise SystemExit(1 if failed else 0)
