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


# ─── long episodes: demo lengths and time-limit endings (2026-11-05) ────────
def _long_rows(folder):
    """carol records at a 3000-tick cap: one accepted demo the cap ended (task
    completed), one failure the cap ended (new `time_limit` field), one failure
    ended by the cap in a ledger written BEFORE the field existed (end_reason
    only), and two normal accepted demos."""
    def at(hms):
        return "2026-11-05T%s" % hms
    rows = [
        dict(event="session", t=at("09:00:00"), source=SRC, operator="carol", seeds=[0, 1, 2, 3],
             lock_rule="v3", max_ticks=3000),
        dict(event="issue", t=at("09:01:00"), seed=0),
        dict(event="accept", t=at("09:03:10"), seed=0, path="ep_seed0000_a.npz", ticks=3000,
             end_reason="time_limit", time_limit=True, max_ticks=3000, heldout=False),
        dict(event="issue", t=at("09:04:00"), seed=1),
        dict(event="discard", t=at("09:06:10"), seed=1, reason="failed: PLACE (time_limit)",
             saved="failed/ep_seed0001_b.npz", ticks=3000, end_reason="time_limit",
             time_limit=True, max_ticks=3000),
        dict(event="issue", t=at("09:07:00"), seed=1),
        dict(event="discard", t=at("09:09:10"), seed=1, reason="failed: GRASP (time_limit)",
             saved="failed/ep_seed0001_c.npz", ticks=3000, end_reason="time_limit"),
        dict(event="issue", t=at("09:10:00"), seed=2),
        dict(event="accept", t=at("09:11:00"), seed=2, path="ep_seed0002_d.npz", ticks=1500,
             end_reason="success", time_limit=False, max_ticks=3000, heldout=False),
        dict(event="issue", t=at("09:12:00"), seed=3),
        dict(event="accept", t=at("09:13:30"), seed=3, path="ep_seed0003_e.npz", ticks=2000,
             end_reason="success", time_limit=False, max_ticks=3000, heldout=False),
    ]
    with open(os.path.join(folder, "ledger.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    for name, n in (("ep_seed0000_a", 3000), ("failed/ep_seed0001_b", 3000),
                    ("failed/ep_seed0001_c", 3000), ("ep_seed0002_d", 1500),
                    ("ep_seed0003_e", 2000)):
        _episode(os.path.join(folder, name + ".npz"), n, 0, 25.0, 0, "carol")


def test_demo_lengths_and_time_limit_endings(tmp_path):
    folder = tmp_path / "long"
    folder.mkdir()
    _long_rows(str(folder))
    r = TM.analyze(str(folder), out=str(tmp_path / "out"))
    at = r["attempts"]
    assert [a["time_limit"] for a in at] == [True, True, True, False, False]
    assert [a["max_ticks"] for a in at] == [3000] * 5          # the old row: the session's cap
    row = r["operators"][0]
    assert (row["demo_ticks_min"], row["demo_ticks_median"], row["demo_ticks_max"]) == \
        (1500, 2000.0, 3000)
    assert row["demo_s_median"] == pytest.approx(80.0)
    assert (row["time_limit_endings"], row["time_limit_accepted"]) == (3, 1)
    assert row["max_ticks"] == "3000"
    md = open(os.path.join(r["out"], "teleop_metrics.md"), encoding="utf-8").read()
    assert "time_limit_endings" in md and "demo_ticks_median" in md


def test_old_ledgers_without_the_cap_still_report(result):
    """The first fixture predates `time_limit` / `max_ticks`: none hit the cap,
    the cap is n/a, demo lengths come from the accept rows' ticks."""
    ops = {r["operator"]: r for r in result["operators"]}
    a = ops["alice"]
    assert (a["demo_ticks_min"], a["demo_ticks_median"], a["demo_ticks_max"]) == (400, 450.0, 500)
    assert a["time_limit_endings"] == 0 and a["max_ticks"] == "n/a"
    assert ops["ALL"]["demo_ticks_max"] == 600


def test_discarded_successes_are_discards_and_a_revoked_accept_is_not_accepted(tmp_path):
    """A successful episode the operator discards is `discard` (never `failed`,
    which is the success rule's verdict), and an accept taken back by
    tools/revoke_episode.py no longer counts as accepted: the numbers equal the
    ledger's own."""
    from g1_data.ledger import EpisodeLedger
    folder = tmp_path / "rv"
    folder.mkdir()
    at = lambda hms: "2026-11-13T%s" % hms
    rows = [
        dict(event="session", t=at("13:00:00"), label="teleop-live", source="teleop_live",
             operator="charles", seeds=[16, 19], lock_rule="v3"),
        dict(event="issue", t=at("13:13:23"), seed=16),
        dict(event="discard", t=at("13:14:00"), seed=16, ticks=656, end_reason="success",
             reason="operator discard of a successful episode (recommended accept)",
             saved="recordings\\rv\\discarded\\ep_seed0016_a.npz"),
        dict(event="issue", t=at("13:15:00"), seed=16),
        dict(event="discard", t=at("13:15:20"), seed=16, ticks=400, end_reason="operator_stop",
             reason="failed: GRASP (operator_stop)", saved="recordings\\rv\\failed\\ep_seed0016_f.npz"),
        dict(event="issue", t=at("13:15:48"), seed=16),
        dict(event="accept", t=at("13:16:19"), seed=16, ticks=613, end_reason="success",
             path="recordings\\rv\\ep_seed0016_b.npz", heldout=False),
        dict(event="issue", t=at("13:17:33"), seed=19),
        dict(event="accept", t=at("13:18:07"), seed=19, ticks=633, end_reason="success",
             path="recordings\\rv\\ep_seed0019_c.npz", heldout=False),
        dict(event="discard", t=at("13:21:19"), seed=19, ticks=633, end_reason="success",
             reason="acceptance revoked: operator mis-key",
             saved="recordings\\rv\\discarded\\ep_seed0019_c.npz",
             revokes="recordings\\rv\\ep_seed0019_c.npz"),
    ]
    with open(os.path.join(str(folder), "ledger.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    _episode(os.path.join(str(folder), "discarded", "ep_seed0016_a.npz"), 656, 0, 25.0, 1, "charles")
    _episode(os.path.join(str(folder), "failed", "ep_seed0016_f.npz"), 400, 0, 25.0, 0, "charles")
    _episode(os.path.join(str(folder), "ep_seed0016_b.npz"), 613, 0, 25.0, 5, "charles")
    _episode(os.path.join(str(folder), "discarded", "ep_seed0019_c.npz"), 633, 0, 25.0, 14, "charles")
    res = TM.analyze(str(folder), out=str(tmp_path / "out"), window=2)
    assert [(a["seed"], a["outcome"], a["revoked"], a["discarded_success"]) for a in res["attempts"]] == [
        (16, "discard", False, True), (16, "failed", False, False),
        (16, "accept", False, False), (19, "discard", True, True)]
    op = {r["operator"]: r for r in res["operators"]}["charles"]
    assert (op["attempts"], op["accepted"], op["failed"], op["discarded"],
            op["discarded_successes"], op["revoked"]) == (4, 1, 1, 2, 2, 1)
    assert op["accepted"] == EpisodeLedger(os.path.join(str(folder), "ledger.jsonl")).summary()["accepted"]
    assert op["success_rate"] == 0.25
    # the revoked episode is still measured, from discarded/
    revoked = res["attempts"][-1]
    assert revoked["file"] == "ep_seed0019_c.npz" and revoked["overrun_ticks"] == 14
    assert revoked["success"] == 0 and revoked["reason"].startswith("acceptance revoked")
    assert op["saved_episodes_measured"] == 4
    assert op["demo_ticks_max"] == 613                    # demo lengths: accepted only
