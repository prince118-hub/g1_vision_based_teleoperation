"""Tests for tools/video_highlight_reel.py on SYNTHETIC clips and index files
in the formats tools/record_exp_videos.py and tools/video_side_by_side.py write.

    python tools/test_video_highlight_reel.py          (or pytest)
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
import video_highlight_reel as R                                    # noqa: E402

#: tools/record_exp_videos.INDEX_COLS (copied: importing it needs MuJoCo; tested equal there)
INDEX_COLS = ["run", "exp", "seed", "role", "success", "fail_reason", "phase_reached",
              "placement_error_m", "ticks", "ckpt", "sha_stored", "sha_rerun", "hash_match",
              "video"]


def _skip(msg: str) -> None:
    try:
        import pytest
    except ImportError:
        print("      skip: %s" % msg)
        return
    pytest.skip(msg)


def clip(path, n, value, size=(160, 90)):
    with VC.Mp4Writer(path, *size) as w:
        for k in range(n):
            w.write(np.full((size[1], size[0], 3), value, np.uint8))
    return path


def idx_row(run, exp, seed, role, ok, fr, pe, ticks, video, match=True):
    return dict(run=run, exp=exp, seed=seed, role=role, success=ok, fail_reason=fr,
                phase_reached="VERIFY" if ok else "SETTLE", placement_error_m=pe, ticks=ticks,
                ckpt="best.pt", sha_stored="a", sha_rerun="a" if match else "b",
                hash_match=match, video=video)


def write_index(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS)
        w.writeheader()
        w.writerows(rows)
    return path


def fixture(tmp):
    """act: two successes (seed 2 placed better), failures GRASP 30/40 and
    PLACE 10/40; act_lstm: one success with a hash MISMATCH, one GRASP failure;
    bc: failures only. Clip lengths = ticks."""
    v = lambda name, n: clip(os.path.join(tmp, name + ".mp4"), n, 120)  # noqa: E731
    rows = [
        idx_row("act_160_seed0", "exp1", 1, "success", True, "", 0.050, 40, v("a1", 40)),
        idx_row("act_160_seed0", "exp1", 2, "success", True, "", 0.020, 50, v("a2", 50)),
        idx_row("act_160_seed0", "exp1", 3, "failure-GRASP (30/40 failures)", False, "GRASP", 0.6, 60, v("a3", 60)),
        idx_row("act_160_seed0", "exp2", 4, "failure-PLACE (10/40 failures)", False, "PLACE", 0.2, 60, v("a4", 60)),
        idx_row("act_lstm_160_seed0", "exp1", 5, "success", True, "", 0.010, 40, v("l5", 40), match=False),
        idx_row("act_lstm_160_seed0", "exp1", 6, "failure-GRASP (5/9 failures)", False, "GRASP", 0.7, 45, v("l6", 45)),
        idx_row("bc_160_seed0", "exp1", 7, "failure-WALK_IN (60/100 failures)", False, "WALK_IN", 1.2, 900,
                os.path.join(tmp, "missing.mp4")),
        idx_row("bc_160_seed0", "exp2", 8, "failure-GRASP (40/100 failures)", False, "GRASP", 0.9, 30, v("b8", 30)),
    ]
    sbs = []
    for i, (cat, hm) in enumerate((("heldout_spread", 3), ("contrast", 3), ("contrast", 3), ("all_succeed", 2))):
        sbs.append(dict(exp="exp2", seed=100 + i, box_x=1.5, box_y=0.0, heldout=True, tag="t",
                        category=cat, why="why %d" % i, n_panels=3, hash_matches=hm,
                        runs="bc_160_seed0;act_160_seed0;act_lstm_160_seed0",
                        labels="BC;ACT;ACT-LSTM", outcomes="x", frames=50,
                        video=clip(os.path.join(tmp, "s%d.mp4" % i), 50, 90, size=(240, 60))))
    return rows, sbs


def _segs(man, kind=None):
    return [s for s in man["segments"] if kind is None or s["type"] == kind]


def test_auto_plan_best_success_typical_failure_and_hash_filter():
    tmp = tempfile.mkdtemp()
    try:
        rows, sbs = fixture(tmp)
        man = R.auto_manifest(rows, sbs, [], {}, ("exp1", "exp2"))
        titles = [s["title"] for s in _segs(man, "card")]
        # model order bc, act, act_lstm (MODEL_ORDER), between the title and end cards
        assert titles[1:4] == ["BC", "ACT (W_o 12)", "ACT-LSTM (W_o 12)"], titles
        clips = [(s["caption"].split(" - ")[0], os.path.basename(s["video"])) for s in _segs(man, "clip")]
        # BC: no success; its WALK_IN clip is missing on disk, so the GRASP one
        # ACT: best success = seed 2 (lower placement error), commonest failure GRASP
        # ACT-LSTM: its success is a hash mismatch -> left out
        assert clips[:4] == [("BC", "b8.mp4"), ("ACT (W_o 12)", "a2.mp4"),
                             ("ACT (W_o 12)", "a3.mp4"), ("ACT-LSTM (W_o 12)", "l6.mp4")], clips
        assert "no successful episode" in _segs(man, "card")[1]["lines"][0]
        assert "typical failure: GRASP (30 of 40 failures)" in _segs(man, "clip")[2]["caption"]
        assert "placement error 2.0 cm" in _segs(man, "clip")[1]["caption"]
        # side by side: one per category in preference order, the mismatch left out
        sb = [os.path.basename(s["video"]) for s in _segs(man, "clip")[4:]]
        assert sb == ["s1.mp4", "s0.mp4"], sb                 # contrast, then heldout_spread
        loose = R.auto_manifest(rows, sbs, [], {}, ("exp1",), allow_mismatch=True, sbs_max=0)
        cap = [s["caption"] for s in _segs(loose, "clip") if s["video"].endswith("l5.mp4")]
        assert cap and "RE-RUN != EVAL RECORD" in cap[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runs_labels_and_end_card_tallies():
    tmp = tempfile.mkdtemp()
    try:
        rows, sbs = fixture(tmp)
        man = R.auto_manifest(rows, sbs, ["act_lstm_160_seed0", "act_160_seed0"],
                              {"act_lstm_160_seed0": "ACT-LSTM"}, ("exp1", "exp2"),
                              tally={"act_160_seed0": {"exp1": (14, 100), "exp2": (1, 100)}})
        titles = [s["title"] for s in _segs(man, "card")]
        assert titles[1:3] == ["ACT-LSTM", "ACT (W_o 12)"], titles   # the --runs order
        assert all("bc_160" not in s["detail"] for s in _segs(man, "clip"))
        assert not any(s["video"].endswith(("s0.mp4", "s1.mp4")) for s in _segs(man, "clip"))  # BC panel
        end = _segs(man, "card")[-1]
        assert "ACT (W_o 12) (act_160_seed0): Exp 1 14/100   Exp 2 1/100" in end["lines"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_timing_fits_the_target_and_states_the_speed():
    man = dict(fps=25, segments=[dict(type="card", title="t", seconds=10.0)] +
               [dict(type="clip", video="x", frames=n, caption="c", detail="d") for n in (500, 900, 100)])
    out = R.plan_timing(man, target_s=40.0, max_speed=3.0)
    sp = [s["speed"] for s in _segs(out, "clip")]
    # budget (40 - 10) / 3 = 10 s: 20 s -> 2x, 36 s -> 3.6x capped at 3, 4 s -> 1x
    assert sp == [2.0, 3.0, 1.0], sp
    assert [s["seconds"] for s in _segs(out, "clip")] == [10.0, 12.0, 4.0]
    assert out["planned_seconds"] == 36.0
    assert R.segment_caption(_segs(out, "clip")[0]) == "d | 2x speed"
    assert R.segment_caption(_segs(out, "clip")[2]) == "d"
    keep = R.plan_timing(dict(fps=25, segments=[dict(type="clip", video="x", frames=100, speed=1.5,
                                                     start_s=1.0, end_s=3.0)]), 5.0)
    assert keep["segments"][0]["speed"] == 1.5 and keep["segments"][0]["seconds"] == 1.36  # 50 frames / 1.5


def test_clip_source_frames_speed_and_window():
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "c.mp4")
        with VC.Mp4Writer(p, 32, 32) as w:
            for k in range(20):
                w.write(np.full((32, 32, 3), 10 * k, np.uint8))
        got = [int(round(f.mean() / 10.0)) for f in R.clip_source_frames(p, 0, None, 2.0)]
        assert got == list(range(0, 20, 2)), got
        got = [int(round(f.mean() / 10.0)) for f in R.clip_source_frames(p, 4, 10, 1.0)]
        assert got == [4, 5, 6, 7, 8, 9], got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_faded_streams_with_fades_at_both_ends():
    frames = [np.full((4, 4, 3), 200, np.uint8) for _ in range(10)]
    out = [f.mean() for f in R.faded(iter(frames), 3)]
    assert len(out) == 10
    assert out[0] < out[1] < out[2] == 200 == out[6] and out[9] < out[8] < out[7] == 200, out
    assert [f.mean() for f in R.faded(iter(frames), 0)] == [200.0] * 10


def test_render_and_sidecar_through_main():
    tmp = tempfile.mkdtemp()
    try:
        rows, sbs = fixture(tmp)
        ix = write_index(os.path.join(tmp, "index.csv"), rows)
        sx = os.path.join(tmp, "side_by_side_index.csv")
        with open(sx, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sbs[0]))
            w.writeheader()
            w.writerows(sbs)
        man_path = os.path.join(tmp, "plan.json")
        assert R.main(["--index", ix, "--side-by-side", sx, "--plan", "--write-manifest", man_path,
                       "--size", "320x180", "--target-s", "20"]) == 0
        man = json.load(open(man_path))
        assert man["size"] == [320, 180] and _segs(man, "clip")
        out = os.path.join(tmp, "reel.mp4")
        assert R.main(["--manifest", man_path, "--out", out]) == 0
        side = json.load(open(os.path.join(tmp, "reel.json")))      # <reel>.json sidecar
        assert side["segments"] == man["segments"]
        frames = list(VC.iter_frames(out))
        assert len(frames) == side["rendered"]["frames"] and frames[0].shape == (180, 320, 3)
        assert abs(side["rendered"]["seconds"] - man["planned_seconds"]) < 0.5, (
            side["rendered"]["seconds"], man["planned_seconds"])
        # 10 frames into the first clip (after the title and the first model card):
        # the caption band darkens the bottom of the clip's flat grey
        k = sum(int(round(float(s["seconds"]) * 25)) for s in man["segments"][:2]) + 10
        assert man["segments"][2]["type"] == "clip"
        f = frames[k]
        assert f[-10:].mean() < 0.6 * f[90, 160].mean(), (f[-10:].mean(), f[90, 160])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_index_columns_are_record_exp_videos():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return _skip("needs MuJoCo (record_exp_videos imports eval_scripted)")
    import record_exp_videos as V
    assert V.INDEX_COLS == INDEX_COLS


def test_h264_transcode_keeps_every_frame():
    if not shutil.which("ffmpeg"):
        return _skip("needs ffmpeg on PATH")
    tmp = tempfile.mkdtemp()
    try:
        p = clip(os.path.join(tmp, "c.mp4"), 12, 100, size=(64, 48))
        VC.transcode_h264(p)
        assert VC.probe(p)["frames"] == 12 and os.listdir(tmp) == ["c.mp4"]
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
