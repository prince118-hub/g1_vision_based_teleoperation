"""tools/teleop_metrics.py on a synthetic fake-source recording folder with known answers.

    python -m pytest tools/test_teleop_metrics.py -q

The fixture is written in the format tools/record_teleop_live.py writes - the
same ledger events and fields (session operator/source, issue, accept `path`,
discard `saved` / `reason` / `ticks`, the held-out `error`) and episode files
with `meta` (source teleop_live_fake, timing) and `tracking_ok`. It is built by
hand rather than recorded, because recording needs mujoco and the policy; the
times are fixed so every expected number below is hand-computed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import teleop_metrics as TM                                         # noqa: E402

SRC = "teleop_live_fake"


def _episode(path, n, degraded, rate, overruns, op):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = np.ones(n, np.uint8)
    ok[:degraded] = 0
    meta = dict(spec_version="g1-spec-1.2.0", n_ticks=n, source=SRC, operator=op,
                lock_rule=dict(version="v3"), grip_label="next_weld_v1",
                timing=dict(paced=False, tick_rate_hz=rate, overrun_ticks=overruns))
    np.savez_compressed(path, states=np.zeros((n, 48), np.float32), tracking_ok=ok,
                        meta=np.array(json.dumps(meta)))


def _rows(folder):
    def at(hms):
        return "2026-11-02T%s" % hms
    sess = lambda op: dict(event="session", t=at("09:59:00"), label="teleop-live-fake",
                           source=SRC, operator=op, seeds=[0, 1], lock_rule="v3")
    rows = [
        sess("alice"),
        dict(event="issue", t=at("10:00:00"), seed=0, label="teleop-live-fake"),
        dict(event="discard", t=at("10:00:20"), seed=0, reason="failed: GRASP (operator_stop)",
             saved="recordings\\fx\\failed\\ep_seed0000_a.npz", ticks=250,
             end_reason="operator_stop"),
        dict(event="issue", t=at("10:01:00"), seed=0, label="teleop-live-fake"),
        dict(event="discard", t=at("10:01:05"), seed=0, reason="operator discard", saved=None,
             ticks=50, end_reason="operator_discard"),
        dict(event="issue", t=at("10:02:00"), seed=0, label="teleop-live-fake"),
        dict(event="accept", t=at("10:02:30"), seed=0, path="recordings/fx/ep_seed0000_b.npz",
             ticks=500, end_reason="success", heldout=False),
        # a held-out spawn is skipped BEFORE an issue: not an attempt
        dict(event="error", t=at("10:02:40"), seed=7, reason="held-out spawn: never demonstrated"),
        dict(event="issue", t=at("10:03:00"), seed=1, label="teleop-live-fake"),
        dict(event="accept", t=at("10:03:25"), seed=1, path="recordings/fx/ep_seed0001_c.npz",
             ticks=400, end_reason="success", heldout=False),
        sess("bob"),
        dict(event="issue", t=at("11:00:00"), seed=2, label="teleop-live-fake"),
        dict(event="accept", t=at("11:00:40"), seed=2, path="recordings/fx/ep_seed0002_d.npz",
             ticks=600, end_reason="success", heldout=False),
        dict(event="issue", t=at("11:01:00"), seed=3, label="teleop-live-fake"),   # crash
    ]
    with open(os.path.join(folder, "ledger.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.write('{"event": "acc')                   # torn final line: skipped by the ledger
    _episode(os.path.join(folder, "failed", "ep_seed0000_a.npz"), 250, 25, 25.0, 1, "alice")
    _episode(os.path.join(folder, "ep_seed0000_b.npz"), 500, 0, 24.0, 2, "alice")
    _episode(os.path.join(folder, "ep_seed0001_c.npz"), 400, 15, 23.0, 0, "alice")
    _episode(os.path.join(folder, "ep_seed0002_d.npz"), 600, 6, 25.0, 0, "bob")
    # a raw keypoint take is never an episode
    _episode(os.path.join(folder, "keypoints", "ep_seed0000_b_kp.npz"), 3, 0, 0.0, 0, "x")


@pytest.fixture
def result(tmp_path):
    folder = tmp_path / "fx"
    folder.mkdir()
    _rows(str(folder))
    return TM.analyze(str(folder), out=str(tmp_path / "out"), window=2)


def test_attempts_and_outcomes(result):
    at = result["attempts"]
    assert [(a["operator"], a["seed"], a["outcome"]) for a in at] == [
        ("alice", 0, "failed"), ("alice", 0, "discard"), ("alice", 0, "accept"),
        ("alice", 1, "accept"), ("bob", 2, "accept"), ("bob", 3, "unfinished")]
    assert [a["wall_s"] for a in at] == [20.0, 5.0, 30.0, 25.0, 40.0, None]
    assert [a["sim_s"] for a in at][:5] == pytest.approx([10.0, 2.0, 20.0, 16.0, 24.0])


def test_learning_curve(result):
    alice = [a for a in result["attempts"] if a["operator"] == "alice"]
    assert [a["k"] for a in alice] == [1, 2, 3, 4]
    assert [a["cum_success_rate"] for a in alice] == pytest.approx([0, 0, 1 / 3, 0.5])
    assert [a["rolling_success_rate"] for a in alice] == pytest.approx([0, 0, 0.5, 1.0])


def test_operator_summary(result):
    ops = {r["operator"]: r for r in result["operators"]}
    a = ops["alice"]
    assert (a["attempts"], a["accepted"], a["failed"], a["discarded"]) == (4, 2, 1, 1)
    assert a["success_rate"] == 0.5
    assert a["wall_s_per_attempt"] == 20.0               # (20 + 5 + 30 + 25) / 4
    assert a["wall_s_per_accepted_demo"] == 40.0         # 80 s of attempts / 2 accepted
    assert a["wall_s_per_accepted_attempt"] == 27.5      # (30 + 25) / 2
    assert a["sim_s_per_attempt"] == pytest.approx(12.0)
    assert a["dropout_pct"] == pytest.approx(100 * 40 / 1150)   # (25 + 0 + 15) / (250+500+400)
    assert a["saved_episodes_measured"] == 3             # the operator discard wrote nothing
    assert a["tick_rate_hz_mean"] == pytest.approx(24.0)
    assert a["overrun_ticks_total"] == 3
    b = ops["bob"]
    assert (b["attempts"], b["accepted"], b["unfinished"]) == (2, 1, 1)
    assert b["success_rate"] == 0.5
    allr = ops["ALL"]
    assert (allr["attempts"], allr["accepted"]) == (6, 3)
    assert allr["success_rate"] == 0.5
    assert allr["dropout_pct"] == pytest.approx(100 * 46 / 1750)
    assert allr["sources"] == SRC


def test_outputs_written(result):
    for f in ("attempts.csv", "operators.csv", "learning_curve.png", "teleop_metrics.md"):
        assert os.path.getsize(os.path.join(result["out"], f)) > 0


def test_operator_filter(tmp_path):
    folder = tmp_path / "fx"
    folder.mkdir()
    _rows(str(folder))
    r = TM.analyze(str(folder), out=str(tmp_path / "o2"), operator="bob")
    assert [o["operator"] for o in r["operators"]] == ["bob"]


def test_missing_ledger_refused(tmp_path):
    with pytest.raises(SystemExit):
        TM.analyze(str(tmp_path), out=str(tmp_path / "o"))
