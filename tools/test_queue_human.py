"""Tests for `human_runs.queue` - order, skipping, stop rules, waiting. Nothing is trained:
`train`, `evaluate`, the status readers and the heavy-process probe are replaced.

    python -m pytest tools/test_queue_human.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import human_runs as H                                              # noqa: E402


class Fake:
    """Disk state per run name; records every call."""

    def __init__(self, monkeypatch, tmp_path, train_state=None, eval_state=None,
                 heavy=None, train_rc=None, eval_rc=None, refuse=()):
        self.t = dict(train_state or {})
        self.e = dict(eval_state or {})
        self.calls = []
        self.heavy = list(heavy or [])
        self.train_rc, self.eval_rc, self.refuse = dict(train_rc or {}), dict(eval_rc or {}), set(refuse)
        monkeypatch.setattr(H, "train_status", lambda root, n: dict(state=self.t.get(n, "not_started")))
        monkeypatch.setattr(H, "eval_status", lambda root, n: dict(state=self.e.get(n, "not_started")))
        monkeypatch.setattr(H, "heavy_processes", self._heavy)
        monkeypatch.setattr(H, "train", self._train)
        monkeypatch.setattr(H, "evaluate", self._eval)
        monkeypatch.setattr(H, "_append_summary", lambda root, line: None)
        monkeypatch.setattr(H.time, "sleep", lambda s: self.calls.append(("sleep", s)))

    def _heavy(self, exclude_pid=None):
        return [self.heavy.pop(0)] if self.heavy else []

    def _train(self, m, s, restart=False, root=None):
        n = H.run_name(m, s)
        self.calls.append(("train", n, restart))
        if n in self.refuse and not restart:
            raise SystemExit("REFUSED: %s was INTERRUPTED" % n)
        rc = self.train_rc.get(n, 0)
        if rc == 0:
            self.t[n] = "done"
        return rc

    def _eval(self, m, s, root=None, extra=None):
        n = H.run_name(m, s)
        self.calls.append(("eval", n))
        rc = self.eval_rc.get(n, 0)
        if rc == 0:
            self.e[n] = "done"
        return rc


@pytest.fixture(autouse=True)
def _v3():
    H.use_dataset("human_v3")
    yield
    H.use_dataset("human_v1")


def _steps(f):
    return [c[:2] for c in f.calls if c[0] in ("train", "eval")]


def test_order_is_seed_major_train_then_eval(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path)
    assert H.queue(list(H.MODELS), [0, 1], root="x") == 0
    want = []
    for s in (0, 1):
        for m in H.MODELS:
            n = H.run_name(m, s)
            want += [("train", n), ("eval", n)]
    assert _steps(f) == want
    assert H.run_name("bc", 0) == "bc_b256_human40v3_seed0"


def test_rerun_skips_finished_steps(monkeypatch, tmp_path):
    a, b = H.run_name("bc", 0), H.run_name("act_orig", 0)
    f = Fake(monkeypatch, tmp_path, train_state={a: "done", b: "done"}, eval_state={a: "done"})
    assert H.queue(["bc", "act_orig"], [0], root="x") == 0
    assert _steps(f) == [("eval", b)]


def test_interrupted_training_stops_the_queue_and_restart_retrains_it(monkeypatch, tmp_path):
    a, b = H.run_name("bc", 0), H.run_name("act_orig", 0)
    f = Fake(monkeypatch, tmp_path, train_state={b: "incomplete"}, refuse=[b])
    assert H.queue(["bc", "act_orig", "act_lstm_full"], [0], root="x") == 1
    assert _steps(f) == [("train", a), ("eval", a), ("train", b)]        # stopped at b
    f.calls.clear()
    assert H.queue(["bc", "act_orig", "act_lstm_full"], [0], restart=True, root="x") == 0
    assert [c for c in f.calls if c[0] == "train"][0] == ("train", b, True)
    assert ("eval", a) not in _steps(f)                                  # a was finished


def test_a_failed_step_stops_the_queue(monkeypatch, tmp_path):
    a = H.run_name("bc", 0)
    f = Fake(monkeypatch, tmp_path, eval_rc={a: 1})
    assert H.queue(["bc", "act_orig"], [0], root="x") == 1
    assert _steps(f) == [("train", a), ("eval", a)]
    f2 = Fake(monkeypatch, tmp_path, train_rc={a: 1})
    assert H.queue(["bc", "act_orig"], [0], root="x") == 1
    assert _steps(f2) == [("train", a)]


def test_waits_while_another_heavy_process_runs(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, heavy=["pid 1 train_bc.py", "pid 1 train_bc.py"])
    assert H.queue(["bc"], [0], root="x", poll_s=7) == 0
    assert f.calls[:3] == [("sleep", 7), ("sleep", 7), ("train", H.run_name("bc", 0), False)]


def test_no_wait_refuses(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, heavy=["pid 1 deploy.py"] * 3)
    with pytest.raises(SystemExit, match="REFUSED"):
        H.queue(["bc"], [0], root="x", wait=False)
    assert _steps(f) == []


def test_plan_runs_nothing(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path)
    assert H.queue(list(H.MODELS), [0, 1, 2], dry=True, root="x") == 0
    assert _steps(f) == []
