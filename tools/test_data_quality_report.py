"""Pure helpers of tools/data_quality_report.py (no data, no runs)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import data_quality_report as DQ                                    # noqa: E402

REG = dict(x=(1.44, 1.56), y=(-0.21, 0.21), heldout_x=(1.47, 1.53), heldout_y=(0.04, 0.16))


def test_describe_uses_sample_sd_and_skips_non_finite():
    d = DQ.describe([1.0, 2.0, 3.0, float("nan"), None])
    assert d["n"] == 3 and d["mean"] == 2.0 and d["sd"] == pytest.approx(1.0)
    assert (d["min"], d["max"], d["median"]) == (1.0, 3.0, 2.0)
    assert DQ.describe([])["mean"] is None


def test_key_pattern_threshold_and_changes():
    v = np.array([[0, 0, 0], [0.6, 0, 0], [0.6, 0, 0], [0.05, 0, 0], [0, 0, -0.6], [-0.4, 0, -0.6]])
    p = DQ.key_pattern(v)
    assert p.tolist() == [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, -1], [-1, 0, -1]]
    assert DQ.key_changes(p) == 4
    assert DQ.on_runs(p) == [2, 1, 1]


def test_on_runs_splits_adjacent_different_patterns():
    p = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]], dtype=np.int8)
    assert DQ.on_runs(p) == [2, 1, 1]


def test_exact_key_pattern_ignores_hold_values():
    v = np.array([[0.6, 0.017, 0.0], [-0.2, 0.0, 0.25], [0.3, 0.0, -0.6], [-0.0544, 0.0, 0.0]])
    assert DQ.exact_key_pattern(v).tolist() == [[1, 0, 0], [-1, 0, 0], [1, 0, -1], [0, 0, 0]]


def test_command_shares_masks():
    v = np.array([[0.6, 0, 0], [-0.4, 0, 0.6], [0, 0.4, 0], [0, 0, 0]])
    m = DQ.command_shares(v)
    assert m["forward"].tolist() == [True, False, False, False]
    assert m["backward"].tolist() == [False, True, False, False]
    assert m["turn"].tolist() == [False, True, False, False]
    assert m["strafe"].tolist() == [False, False, True, False]
    assert m["idle"].tolist() == [False, False, False, True]


def test_spawn_cell_and_side_match_the_split_binning():
    assert DQ.spawn_cell((1.44, -0.21), REG) == (0, 0)
    assert DQ.spawn_cell((1.56, 0.21), REG) == (2, 2)
    assert DQ.spawn_cell((1.50, 0.0), REG) == (1, 1)
    assert DQ.side_of(-0.08, REG) == "right" and DQ.side_of(0.0, REG) == "centre" and DQ.side_of(0.08, REG) == "left"


def test_bucket_ticks_follows_scored_of():
    from g1_data import spec
    P = spec.Phase
    labels = [P.SETTLE] * 3 + [P.REACH] * 2 + [P.GRASP] + [P.LIFT] + [P.MOVE] * 4 + [P.LOWER, P.RELEASE, P.VERIFY]
    assert DQ.bucket_ticks([int(x) for x in labels]) == dict(walk_in=3, grasp=3, lift=1, transport=4, place=3)
    with pytest.raises(ValueError):
        DQ.bucket_ticks([int(P.UNKNOWN)])
