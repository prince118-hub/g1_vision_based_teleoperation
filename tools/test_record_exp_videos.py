"""Tests for tools/record_exp_videos.py - episode choice and the index (no simulation).

    python tools/test_record_exp_videos.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import record_exp_videos as V                                       # noqa: E402


def _r(seed, ok, fr="", d=1.0, sha=None):
    return dict(seed=seed, success=ok, fail_reason=fr, final_box_goal_m=d,
                placement_error_m=d, phase_reached="X", ticks=900,
                trajectory_sha256=sha or "s%d" % seed, header=dict(ckpt=dict(file="best.pt")))


def test_two_buckets_give_one_typical_episode_each():
    rows = [_r(0, True)] + [_r(s, False, "GRASP", d) for s, d in ((1, 1.0), (2, 1.5), (3, 2.0))] + \
           [_r(s, False, "TRANSPORT", d) for s, d in ((4, 0.4), (5, 0.6))]
    got = V.typical_failures(rows, 2)
    assert [(r["seed"], role.split()[0]) for r, role in got] == [(2, "failure-GRASP"),
                                                                   (4, "failure-TRANSPORT")], got


def test_one_bucket_gives_two_distinct_episodes_nearest_its_median():
    rows = [_r(s, False, "WALK_IN", d) for s, d in ((1, 1.0), (2, 1.4), (3, 1.5), (4, 3.0))]
    got = [r["seed"] for r, _ in V.typical_failures(rows, 2)]
    assert got == [2, 3], got                          # median 1.45: 1.4 and 1.5
    assert V.typical_failures([_r(0, True)], 2) == []
    assert len(V.typical_failures(rows[:1], 2)) == 1   # never the same episode twice


def test_index_marks_a_hash_mismatch():
    tmp = tempfile.mkdtemp()
    try:
        ev, vr = os.path.join(tmp, "eval"), os.path.join(tmp, "videos")
        os.makedirs(os.path.join(ev, "act_n40_seed0"))
        os.makedirs(os.path.join(vr, "act_n40_seed0", "exp1"))
        stored = [_r(1, True), _r(2, False, "GRASP")]
        with open(os.path.join(ev, "act_n40_seed0", "exp1.jsonl"), "w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in stored)
        rerun = [dict(_r(1, True), video="a.mp4"), dict(_r(2, False, "GRASP", sha="other"), video="b.mp4")]
        d = os.path.join(vr, "act_n40_seed0", "exp1")
        with open(os.path.join(d, "episodes.jsonl"), "w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in rerun)
        json.dump({"1": "success", "2": "failure-GRASP"}, open(os.path.join(d, "roles.json"), "w"))
        idx = V.write_index(vr, ev)
        assert [(x["seed"], x["hash_match"], x["role"]) for x in idx] == [
            (1, True, "success"), (2, False, "failure-GRASP")]
        assert os.path.exists(os.path.join(vr, "index.csv"))
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
