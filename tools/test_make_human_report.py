"""Tests for the definitions tools/make_human_report.py adds (no eval data needed)."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_human_report as H                                       # noqa: E402


def test_lock_before_weld_is_the_pickup_lock():
    f = H.lock_flags([100], weld_engage=150, weld_release=-1)
    assert f["walk_in"] and not f["goal_lock"] and f["pickup_lock_tick"] == 100 and not f["release"]


def test_lock_on_the_weld_tick_is_still_unwelded():
    # the lock decision at tick L reads the state at the START of L; a weld engaged during L's physics is not in it
    f = H.lock_flags([150], weld_engage=150, weld_release=-1)
    assert f["walk_in"] and not f["goal_lock"]


def test_lock_while_welded_is_the_goal_lock():
    f = H.lock_flags([100, 400], weld_engage=150, weld_release=600)
    assert f["walk_in"] and f["goal_lock"] and f["goal_lock_tick"] == 400 and f["release"]


def test_weld_without_a_pickup_lock():
    f = H.lock_flags([300], weld_engage=150, weld_release=-1)          # free-based weld, then a goal lock
    assert not f["walk_in"] and f["goal_lock"]


def test_lock_after_the_release_is_neither():
    f = H.lock_flags([700], weld_engage=150, weld_release=600)         # box dropped, robot locks at it again
    assert not f["walk_in"] and not f["goal_lock"] and f["release"]


def test_no_lock():
    f = H.lock_flags([], weld_engage=-1, weld_release=-1)
    assert not f["walk_in"] and not f["goal_lock"] and not f["release"] and f["n_locks"] == 0


@pytest.mark.parametrize("y,side", [(-0.21, "right"), (-0.0701, "right"), (-0.069, "centre"), (0.0, "centre"),
                                    (0.0701, "left"), (0.21, "left")])
def test_side_thirds(y, side):
    reg = dict(x=(1.44, 1.56), y=(-0.21, 0.21))
    assert H.side_of(y, reg) == side


def test_newcombe_matches_the_published_example():
    # Newcombe (1998) Table II, example (a): 56/70 vs 48/80 -> 0.0524 to 0.3339 (method 10)
    lo, hi = H.newcombe(56, 70, 48, 80)
    assert math.isclose(lo, 0.0524, abs_tol=5e-4) and math.isclose(hi, 0.3339, abs_tol=5e-4)


def test_newcombe_at_zero_is_not_degenerate():
    lo, hi = H.newcombe(0, 300, 0, 300)
    assert lo < 0 < hi


def test_cumulative_chain_requires_every_earlier_phase():
    E = pd.DataFrame(dict(walk_in=[True, False, True], grasp=[True, True, True], lift=[True, True, False],
                          transport=[True, True, False], place=[True, True, False], success=[True, True, False],
                          fell=[False, False, False]))
    C = H.add_cumulative(E)
    assert C.cum_grasp.tolist() == [True, False, True]
    assert C.cum_success.tolist() == [True, False, False]
    assert [H.first_failure(r) for _, r in C.iterrows()] == ["success", "WALK_IN", "LIFT"]
