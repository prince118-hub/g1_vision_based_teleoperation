"""tools/sus_score.py: hand-checked SUS scores and the refusals.

    python -m pytest tools/test_sus_score.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sus_score as SUS                                             # noqa: E402

HEADER = "operator,session,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\n"


def _csv(tmp_path, body, name="r.csv"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return str(p)


def test_hand_checked_score():
    # docs/SUS_FORM.md worked example:
    # odd items 4,5,3,4,5 -> 3+4+2+3+4 = 16; even items 1,2,3,2,1 -> 4+3+2+3+4 = 16;
    # 2.5 * 32 = 80.0
    assert SUS.sus_score([4, 1, 5, 2, 3, 3, 4, 2, 5, 1]) == 80.0


def test_extremes_and_midpoint():
    assert SUS.sus_score([5, 1] * 5) == 100.0       # best possible
    assert SUS.sus_score([1, 5] * 5) == 0.0         # worst possible
    assert SUS.sus_score([3] * 10) == 50.0          # all neutral
    assert SUS.sus_score([4, 2] * 5) == 75.0        # 3*5 + 3*5 = 30 -> 75
    assert SUS.sus_score([5] * 10) == 50.0          # agreeing with everything is not "good"


def test_per_operator_and_means(tmp_path):
    p = _csv(tmp_path, "alice,s1,4,1,5,2,3,3,4,2,5,1\n"      # 80
                       "alice,s2,3,3,3,3,3,3,3,3,3,3\n"      # 50
                       "bob,,5,1,5,1,5,1,5,1,5,1\n")         # 100
    resp = SUS.read_responses(p)
    assert [r["sus"] for r in resp] == [80.0, 50.0, 100.0]
    s = SUS.summarize(resp)
    ops = {o["operator"]: o for o in s["operators"]}
    assert ops["alice"]["sus_mean"] == 65.0 and ops["alice"]["n_responses"] == 2
    assert ops["bob"]["sus_mean"] == 100.0 and ops["bob"]["sus_sd"] is None
    assert s["mean_over_operators"] == 82.5           # (65 + 100) / 2: each operator once
    assert s["mean_over_responses"] == pytest.approx(230.0 / 3)
    assert s["n_operators"] == 2 and s["n_responses"] == 3


def test_session_column_optional(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("operator,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\nc,4,1,5,2,3,3,4,2,5,1\n",
                 encoding="utf-8")
    assert SUS.read_responses(str(p))[0]["sus"] == 80.0


@pytest.mark.parametrize("row", ["a,,4,1,5,2,3,3,4,2,5,\n",      # blank answer
                                 "a,,4,1,5,2,3,3,4,2,5,6\n",     # out of range
                                 "a,,4,1,5,2,3,3,4,2,5,0\n",
                                 "a,,4,1,5,2,3,3,4,2,5,2.5\n",   # not an integer
                                 ",,4,1,5,2,3,3,4,2,5,1\n"])     # no operator
def test_refuses_unscoreable_rows(tmp_path, row):
    with pytest.raises(SUS.SUSError):
        SUS.read_responses(_csv(tmp_path, row))


def test_refuses_missing_column(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("operator,q1,q2\nc,1,2\n", encoding="utf-8")
    with pytest.raises(SUS.SUSError, match="q3"):
        SUS.read_responses(str(p))


def test_cli_writes_outputs(tmp_path, capsys):
    p = _csv(tmp_path, "alice,s1,4,1,5,2,3,3,4,2,5,1\n")
    out = tmp_path / "out"
    assert SUS.main([p, "--out", str(out)]) == 0
    assert "80.0" in capsys.readouterr().out
    for f in ("sus_responses.csv", "sus_operators.csv", "sus.md"):
        assert (out / f).exists()
    assert SUS.main([_csv(tmp_path, "a,,9,1,5,2,3,3,4,2,5,1\n", "bad.csv")]) == 2
