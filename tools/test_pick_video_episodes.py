"""Tests for tools/pick_video_episodes.py on SYNTHETIC eval folders whose
outcomes are set by hand, so every pick is known in advance.

    python tools/test_pick_video_episodes.py          (or pytest)

Where MuJoCo imports, the typical-failure rule is also checked against
tools/record_exp_videos.typical_failures itself.
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

import pick_video_episodes as P                                     # noqa: E402
import video_common as VC                                           # noqa: E402

RUNS = ["bc_160_seed0", "act_160_seed0", "act_lstm_160_seed0"]
EXP1 = list(range(0, 12))
EXP2 = list(range(100, 112))


def _skip(msg: str) -> None:
    try:
        import pytest
    except ImportError:
        print("      skip: %s" % msg)
        return
    pytest.skip(msg)


def _row(seed, ok, fr="", dist=0.5, ticks=600, exp="exp1", kind="act"):
    held = exp == "exp2"
    # exp2 spawns on a 4 x 3 grid in the patch; exp1 anywhere
    i = seed % 100
    box = [1.45 + 0.01 * (i % 4), -0.02 + 0.02 * (i // 4)] if held else [1.4 + 0.01 * i, 0.0]
    return dict(seed=seed, success=ok, fail_reason="" if ok else fr, ticks=ticks, kind=kind,
                box_xy=box, heldout=held, placement_error_m=0.02 + 0.001 * i if ok else dist,
                final_box_goal_m=0.02 + 0.001 * i if ok else dist)


#: outcome per (run, seed): True = success, else the failure bucket
OUT = {
    "bc_160_seed0": {s: ("GRASP" if s % 3 else "WALK_IN") for s in EXP1 + EXP2},
    "act_160_seed0": {**{s: ("GRASP" if s in (2, 3, 4) else "PLACE" if s in (5, 6) else True)
                         for s in EXP1},
                      **{s: ("GRASP" if s < 108 else True) for s in EXP2}},
    "act_lstm_160_seed0": {**{s: ("TRANSPORT" if s in (0, 7) else True) for s in EXP1},
                           **{s: (True if s in (100, 101, 102, 108) else "GRASP") for s in EXP2}},
}


def make(root, runs=RUNS, drop=None):
    for n in runs:
        d = os.path.join(root, "eval", n)
        os.makedirs(d, exist_ok=True)
        for exp, seeds in (("exp1", EXP1), ("exp2", EXP2)):
            with open(os.path.join(d, "%s.jsonl" % exp), "w") as fh:
                for s in seeds:
                    if drop and (n, exp, s) in drop:
                        continue
                    o = OUT[n][s]
                    fh.write(json.dumps(_row(s, o is True, "" if o is True else o,
                                             dist=0.3 + 0.01 * s % 7, ticks=500 + s % 9,
                                             exp=exp, kind=VC.parse_run(n)["model"])) + "\n")
    return root


def _picks(doc, cat, exp=None):
    return [p for p in doc["picks"] if p["category"] == cat and (exp is None or p["exp"] == exp)]


def _run(tmp, extra=()):
    res = make(os.path.join(tmp, "runs_x"))
    out = os.path.join(tmp, "picks", "p.json")
    assert P.main(["--results", res, "--out", out, "--per-category", "3"] + list(extra)) == 0
    return json.load(open(out)), out


def test_contrast_both_ways_round_robin_over_buckets():
    tmp = tempfile.mkdtemp()
    try:
        doc, _ = _run(tmp)
        assert doc["pairs"] == [["act_lstm_160_seed0", "act_160_seed0"]]   # auto: same size + seed
        # exp1: ACT-LSTM wins where ACT fails = {2,3,4} GRASP, {5,6} PLACE (0, 7 lost by ACT-LSTM)
        c = doc["counts"]["contrast"]["exp1"]["act_lstm_160_seed0>act_160_seed0"]
        r = doc["counts"]["contrast_reverse"]["exp1"]["act_lstm_160_seed0>act_160_seed0"]
        assert (c, r) == (5, 2), (c, r)
        got = [(p["seed"], p["outcomes"]["act_160_seed0"]["fail_reason"])
               for p in _picks(doc, "contrast", "exp1")]
        # commonest bucket first, then alternate; within a bucket ACT-LSTM's placement error
        assert got == [(2, "GRASP"), (5, "PLACE"), (3, "GRASP")], got
        assert [p["seed"] for p in _picks(doc, "contrast_reverse", "exp1")] == [0, 7]
        why = _picks(doc, "contrast", "exp1")[0]["why"]
        assert "5 of 12 seeds this way, 2 the other way" in why, why
        for p in _picks(doc, "contrast"):
            o = p["outcomes"]
            assert o["act_lstm_160_seed0"]["success"] and not o["act_160_seed0"]["success"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_typical_failure_one_per_bucket_nearest_the_median():
    rows = [_row(s, False, "GRASP", d) for s, d in ((1, 1.0), (2, 1.4), (3, 1.5), (4, 3.0))] + \
           [_row(5, False, "PLACE", 0.2), _row(6, True)]
    got = [(r["seed"], b, c, n) for r, b, c, n in P.typical_failures(rows)]
    assert got == [(2, "GRASP", 4, 5), (5, "PLACE", 1, 5)], got   # median 1.45: 1.4 and 1.5 tie -> seed 2
    assert P.typical_failures([_row(1, True)]) == []


def test_typical_failure_is_record_exp_videos_rule():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return _skip("needs MuJoCo (record_exp_videos imports eval_scripted)")
    import record_exp_videos as V
    rng = np.random.default_rng(0)
    for trial in range(20):
        rows = [_row(s, bool(rng.random() < 0.3), str(rng.choice(["GRASP", "PLACE", "WALK_IN"])),
                     float(np.round(rng.uniform(0, 2), 2))) for s in range(int(rng.integers(3, 30)))]
        mine = [r["seed"] for r, *_ in P.typical_failures(rows)]
        nb = len({r["fail_reason"] for r in rows if not r["success"]})
        theirs = [r["seed"] for r, _ in V.typical_failures(rows, nb)]
        assert sorted(mine) == sorted(theirs), (trial, mine, theirs)


def test_heldout_spread_and_success():
    tmp = tempfile.mkdtemp()
    try:
        doc, _ = _run(tmp)
        spread = _picks(doc, "heldout_spread")
        assert len(spread) == 3 and all(p["exp"] == "exp2" and p["heldout"] for p in spread)
        assert len({p["seed"] for p in spread}) == 3
        xy = np.array([_row(s, True, exp="exp2")["box_xy"] for s in EXP2])
        first = EXP2[int(np.argmin(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))]
        assert spread[0]["seed"] == first
        # exp2: ACT solves 108-111, ACT-LSTM 100-102 and 108 -> 108 is solved by two
        succ = _picks(doc, "heldout_success")
        assert succ[0]["seed"] == 108 and "2/3 runs" in succ[0]["why"], succ[0]
        assert [p["seed"] for p in succ][1:] == [100, 101]            # lower placement error first
        assert doc["counts"]["heldout_success"]["exp2"] == 7
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_farthest_point_spreads_and_never_repeats():
    xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.5, 0.5]])
    got = P.farthest_point(xy, 5)
    assert got[0] == 4 and len(set(got)) == 5, got
    assert set(got[1:4]) <= {0, 1, 2, 3}
    assert P.farthest_point(xy, 0) == [] and len(P.farthest_point(xy, 99)) == 6


def test_all_succeed_and_no_pairs_without_a_matching_act():
    tmp = tempfile.mkdtemp()
    try:
        res = make(os.path.join(tmp, "r"))
        data, warn = P.load([VC.open_results(res)], ["act_160_seed0", "act_lstm_160_seed0"])
        out = P.pick_all(data, list(data), [], 3)
        ok = [p["seed"] for p in out["picks"] if p["category"] == "all_succeed" and p["exp"] == "exp1"]
        assert ok == [9, 1, 10], ok     # both succeed on 1, 8-11; ticks 500 + s % 9, then seed
        assert P.auto_pairs(["act_lstm_160_seed0", "act_n40_seed0"]) == []        # size differs
        assert P.auto_pairs(["act_lstm_full_n40_seed1", "act_orig_n40_seed1", "act_orig_n40_seed0"]) == \
            [("act_lstm_full_n40_seed1", "act_orig_n40_seed1")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_different_seed_sets_use_the_common_seeds_and_say_so():
    tmp = tempfile.mkdtemp()
    try:
        res = make(os.path.join(tmp, "r"), drop={("bc_160_seed0", "exp1", 2)})
        data, warn = P.load([VC.open_results(res)], None)
        assert list(data) == RUNS                                     # model order bc, act, act_lstm
        assert len(warn) == 1 and "11 common seeds" in warn[0], warn
        assert 2 not in P.common_seeds(data, "exp1", RUNS)
        try:
            P.load([VC.open_results(res)], ["nope_160_seed0"])
        except SystemExit:
            pass
        else:
            raise AssertionError("an unknown run was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_main_writes_json_and_csv_that_the_side_by_side_tool_reads():
    tmp = tempfile.mkdtemp()
    try:
        doc, out = _run(tmp, ["--runs", "bc_160_seed0=BC-8", "act_160_seed0", "act_lstm_160_seed0",
                              "--contrast", "act_lstm_160_seed0:act_160_seed0"])
        assert doc["labels"]["bc_160_seed0"] == "BC-8" and doc["runs"] == RUNS
        rows = list(csv.DictReader(open(os.path.splitext(out)[0] + ".csv")))
        assert len(rows) == len(doc["picks"]) and set(RUNS) <= set(rows[0])
        # BC never succeeds, so no seed is solved by all three
        assert {p["category"] for p in doc["picks"]} == set(P.CATEGORIES) - {"all_succeed"}
        assert doc["counts"]["all_succeed"] == {"exp1": 0, "exp2": 0}
        import video_side_by_side as SBS

        class A:
            pass
        a = A()
        a.runs, a.picks, a.category, a.exp, a.limit, a.seed = None, out, ["contrast"], ["exp1"], 2, None
        jobs = SBS.jobs_from_args(a)
        assert [(j["exp"], j["seed"]) for j in jobs] == [("exp1", 2), ("exp1", 5)]
        assert jobs[0]["runs"] == [("bc_160_seed0", "BC-8"), ("act_160_seed0", "ACT (W_o 12)"),
                                   ("act_lstm_160_seed0", "ACT-LSTM (W_o 12)")]   # the picks' labels
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
