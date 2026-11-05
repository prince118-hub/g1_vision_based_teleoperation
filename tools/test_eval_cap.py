"""Tests for the evaluation-cap rule (g1_data/eval_cap.py, tools/eval_cap.py) and its
plumbing into tools/deploy.py and tools/eval_scripted.py. CPU only.

    python tools/test_eval_cap.py

The dataset fixture is written in the format the recorders write (episode .npz
with `meta`, a ledger with session / issue / accept / discard rows; accepted
episodes at the top level, failed/ and discarded/ below it), built by hand so
every expected cap is hand-computed. The deploy end-to-end test reuses
tools/test_memory_design.py's untrained small act_lstm_full run.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from g1_data import eval_cap as EC                                  # noqa: E402
from g1_data import spec                                            # noqa: E402

_TMP = []


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _tmp() -> str:
    d = tempfile.mkdtemp(prefix="eval_cap_test_")
    _TMP.append(d)
    return d


def _episode(path, seed, n, source="teleop_live_fake"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(seed=seed, n_ticks=n, source=source, box_spawn_xy=[0.9, 0.0],
                spec_version=spec.SPEC_VERSION, end_reason="success")
    np.savez_compressed(path, states=np.zeros((n, spec.STATE_DIM), np.float32),
                        meta=np.array(json.dumps(meta)))


def _folder(lengths=((0, 8), (1, 10), (2, 9)), failed=((3, 50),), ledger=True,
            source="teleop_live_fake") -> str:
    """Accepted episodes at the top (seed, ticks), a LONGER failed one below and
    a longer discarded one: neither may count."""
    d = _tmp()
    rows = [dict(event="session", t="2026-11-05T10:00:00", source=source, max_ticks=3000)]
    for seed, n in lengths:
        _episode(os.path.join(d, "ep_seed%04d_x.npz" % seed), seed, n, source)
        rows += [dict(event="issue", seed=seed), dict(event="accept", seed=seed, ticks=n,
                                                      path="ep_seed%04d_x.npz" % seed)]
    for seed, n in failed:
        _episode(os.path.join(d, "failed", "ep_seed%04d_f.npz" % seed), seed, n, source)
        rows += [dict(event="issue", seed=seed), dict(event="discard", seed=seed, ticks=n,
                                                      reason="failed: GRASP (time_limit)")]
    _episode(os.path.join(d, "discarded", "ep_seed0009_d.npz"), 9, 70, source)
    if ledger:
        with open(os.path.join(d, "ledger.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    return d


# ─── the rule ────────────────────────────────────────────────────────────────
def test_cap_is_ceil_1p3_in_exact_integers():
    import math
    # integer arithmetic; it agrees with the float formula wherever that is right
    assert all(EC.cap_for_length(n) == math.ceil(1.3 * n) for n in range(1, 5001))
    assert EC.cap_for_length(700) == 910
    cases = {1: 2, 10: 13, 11: 15, 100: 130, 692: 900, 693: 901, 900: 1170,
             2000: 2600, 2308: 3001, 3000: 3900}
    for n, cap in cases.items():
        assert EC.cap_for_length(n) == cap, (n, EC.cap_for_length(n), cap)
        assert cap * 10 >= 13 * n > (cap - 1) * 10           # the smallest integer >= 1.3 n
    _raises(EC.CapError, EC.cap_for_length, 0)


def test_scripted_cap_is_900_everywhere():
    """Scripted keeps 900: one number in deploy, the recorder, eval_scripted and
    the rule module - and the rule itself gives 900 for scripted_v3's recorded
    longest episode (692 ticks, NOTES 2026-10-17)."""
    import deploy as D
    import eval_scripted as E
    import record_teleop_live as RT
    assert D.MAX_TICKS == RT.MAX_TICKS == E.DEFAULT_MAX_TICKS == EC.SCRIPTED_MAX_TICKS == 900
    assert EC.cap_for_length(692) == 900


def test_cap_of_a_folder_counts_accepted_episodes_only():
    c = EC.eval_cap(_folder())
    assert (c["longest_ticks"], c["longest_seed"], c["max_ticks"]) == (10, 1, 13)
    assert c["n_accepted"] == 3 and c["ledger_checked"] is True
    assert c["lengths_ticks"] == dict(min=8, median=9.0, max=10)
    assert c["max_s"] == 0.52 and c["longest_s"] == 0.4
    assert c["rule"] == EC.RULE and c["factor"] == 1.3 and c["scripted"] is False
    assert json.loads(json.dumps(c)) == c                    # recorded verbatim in results
    # 3000-tick demonstrations (120 s): the cap is 3900 ticks = 156 s
    c = EC.eval_cap(_folder(lengths=((0, 2500), (1, 3000)), failed=()))
    assert (c["max_ticks"], c["max_s"]) == (3900, 156.0)


def test_cap_refuses_when_the_ledger_and_the_folder_disagree():
    d = _folder()
    with open(os.path.join(d, "ledger.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(event="accept", seed=7, path="gone.npz")) + "\n")
    e = _raises(EC.CapError, EC.eval_cap, d)
    assert "seeds [7]" in str(e), e
    d = _folder()
    _episode(os.path.join(d, "ep_seed0005_x.npz"), 5, 12)         # a file with no ACCEPT
    assert "seeds [5]" in str(_raises(EC.CapError, EC.eval_cap, d))
    # no ledger: the top level is taken as accepted, and the record says so
    c = EC.eval_cap(_folder(ledger=False))
    assert c["ledger_checked"] is False and c["max_ticks"] == 13
    _raises(EC.CapError, EC.eval_cap, _tmp())                       # empty
    _raises(EC.CapError, EC.eval_cap, os.path.join(_tmp(), "nope"))


def test_scripted_folder_is_reported_as_keeping_900():
    c = EC.eval_cap(_folder(lengths=((0, 600), (1, 692)), failed=(),
                            source="scripted_demo.run_episode"))
    assert c["scripted"] is True and c["max_ticks"] == 900
    assert any("keeps 900" in L for L in EC.report_lines(c))


def test_cli_prints_the_cap_and_writes_the_record():
    import eval_cap as CLI
    d = _folder()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert CLI.main([d, "--ticks"]) == 0
    assert buf.getvalue().strip() == "13"
    out = os.path.join(_tmp(), "cap.json")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert CLI.main([d, "--json", out]) == 0
    assert "13 ticks" in buf.getvalue() and json.load(open(out))["max_ticks"] == 13
    _raises(SystemExit, CLI.main, [_tmp()])


# ─── deploy.py ───────────────────────────────────────────────────────────────
def test_deploy_resolves_the_cap():
    import deploy as D
    assert D.resolve_max_ticks(None) == (900, None)                  # the default, unchanged
    assert D.resolve_max_ticks(1200) == (1200, None)
    d = _folder()
    n, rule = D.resolve_max_ticks(None, d)
    assert n == 13 and rule["max_ticks"] == 13 and rule["longest_ticks"] == 10
    assert D.resolve_max_ticks(13, d)[0] == 13                        # agreeing: fine
    e = _raises(SystemExit, D.resolve_max_ticks, 900, d)               # disagreeing: refused
    assert "disagrees" in str(e)
    _raises(SystemExit, D.resolve_max_ticks, 0)
    _raises(SystemExit, D.resolve_max_ticks, None, _tmp())


def test_deploy_main_records_the_cap_in_the_header():
    """End to end on the untrained small act_lstm_full run: the default header is
    as before (max_ticks 900, no rule); --max-ticks-from runs to the rule's cap
    (13 ticks; nothing grasps, so end_reason time_limit) and records it."""
    import deploy as D
    import test_memory_design as TM
    run = TM._run_new("act_lstm_full")
    d = _folder()
    out = os.path.join(_tmp(), "rule.jsonl")
    with contextlib.redirect_stdout(io.StringIO()):
        assert D.main(["--run", run, "--ckpt", "ckpt.pt", "--seed-list", "700",
                       "--max-ticks-from", d, "--out", out]) == 0
    r = json.loads(open(out, encoding="utf-8").readline())
    h = r["header"]
    assert h["max_ticks"] == 13 and h["max_ticks_rule"]["max_ticks"] == 13
    assert h["max_ticks_rule"]["folder"] and h["max_ticks_rule"]["longest_ticks"] == 10
    assert r["ticks"] == 13 and r["end_reason"] == "time_limit", (r["ticks"], r["end_reason"])
    # explicit and default: no rule in the header
    out = os.path.join(_tmp(), "explicit.jsonl")
    with contextlib.redirect_stdout(io.StringIO()):
        assert D.main(["--run", run, "--ckpt", "ckpt.pt", "--seed-list", "700",
                       "--max-ticks", "5", "--out", out]) == 0
    h = json.loads(open(out, encoding="utf-8").readline())["header"]
    assert h["max_ticks"] == 5 and "max_ticks_rule" not in h
    _raises(SystemExit, D.main, ["--run", run, "--ckpt", "ckpt.pt", "--seed-list", "700",
                                 "--max-ticks", "900", "--max-ticks-from", d,
                                 "--out", os.path.join(_tmp(), "x.jsonl")])


# ─── eval_scripted.py ────────────────────────────────────────────────────────
def test_eval_scripted_default_cap_calls_deploy_exactly_as_before():
    import eval_scripted as E
    c = E.resolve_cap()
    assert c.is_default() and c.deploy_args() == [] and c.kw() == {}
    assert c.record() == dict(max_ticks=900)
    c = E.resolve_cap(1200)
    assert c.deploy_args() == ["--max-ticks", "1200"] and c.kw() == dict(cap=c)
    d = _folder()
    c = E.resolve_cap(None, d)
    assert c.deploy_args() == ["--max-ticks", "13", "--max-ticks-from", os.path.abspath(d)]
    assert c.record()["max_ticks_rule"]["longest_ticks"] == 10
    _raises(SystemExit, E.resolve_cap, 900, d)
    _raises(SystemExit, E.resolve_cap, 0)


def _stub_deploy(E, calls):
    def fake(cmd, stdout=None, stderr=None, cwd=None):
        calls.append(cmd)
        part = cmd[cmd.index("--out") + 1]
        n = int(cmd[cmd.index("--max-ticks") + 1]) if "--max-ticks" in cmd else 900
        with open(part, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(dict(seed=i, success=False, header=dict(max_ticks=n)))
                          + "\n" for i in range(3))
        return 0
    return fake


def test_eval_scripted_passes_the_cap_to_deploy_and_refuses_a_mix():
    import eval_scripted as E
    tmp = _tmp()
    real, calls = E.subprocess.call, []
    E.subprocess.call = _stub_deploy(E, calls)
    seeds = ["--seed-list", "1,2,3"]
    try:
        # default: deploy's command line carries no cap at all
        E.deploy("run", "best.pt", seeds, 3, os.path.join(tmp, "a", "exp1.jsonl"), "v3",
                 os.path.join(tmp, "a", "logs"))
        assert "--max-ticks" not in calls[-1] and "--max-ticks-from" not in calls[-1]
        # the rule: both passed, deploy recomputes and must agree
        d = _folder()
        cap = E.resolve_cap(None, d)
        rows = E.deploy("run", "best.pt", seeds, 3, os.path.join(tmp, "b", "exp1.jsonl"), "v3",
                        os.path.join(tmp, "b", "logs"), cap=cap)
        cmd = calls[-1]
        assert cmd[cmd.index("--max-ticks") + 1] == "13"
        assert cmd[cmd.index("--max-ticks-from") + 1] == os.path.abspath(d)
        assert rows[0]["header"]["max_ticks"] == 13
        # a finished JSONL is reused only under its own cap
        n = len(calls)
        assert len(E.deploy("run", "best.pt", seeds, 3, os.path.join(tmp, "b", "exp1.jsonl"),
                            "v3", os.path.join(tmp, "b", "logs"), cap=cap)) == 3
        assert len(calls) == n                                   # not re-run
        e = _raises(SystemExit, E.deploy, "run", "best.pt", seeds, 3,
                    os.path.join(tmp, "a", "exp1.jsonl"), "v3", os.path.join(tmp, "a", "logs"),
                    cap=cap)
        assert "max_ticks 900" in str(e) and "asks for 13" in str(e), e
    finally:
        E.subprocess.call = real


def test_eval_scripted_selection_records_the_cap_and_refuses_another():
    import eval_scripted as E
    run = os.path.join(E.ROOT, "runs", "_test_eval_cap_%d" % os.getpid())
    os.makedirs(run, exist_ok=True)
    json.dump(dict(config=dict(epochs=None)), open(os.path.join(run, "metadata.json"), "w"))
    table = {"a.pt": [dict(success=True, grasp=True, lift=True, transport=True, place=True,
                           placement_error_m=0.1,
                           header=dict(ckpt=dict(file="a.pt", step=10, epoch=1)))] * 20}
    saved = E.candidates, E.deploy
    E.candidates = lambda d: [(f, None) for f in table]
    seen = []

    def fake(run_dir, ckpt, *a, **k):
        seen.append(k.get("cap"))
        return table[ckpt]
    E.deploy = fake
    try:
        cap = E.resolve_cap(None, _folder())
        ev = _tmp()
        with contextlib.redirect_stdout(io.StringIO()):
            sel = E.select(run, ev, "v3", cap=cap)
        assert seen == [cap] and sel["max_ticks"] == 13
        assert sel["max_ticks_rule"]["longest_ticks"] == 10
        on_disk = json.load(open(os.path.join(ev, "selection.json"), encoding="utf-8"))
        assert on_disk["max_ticks"] == 13
        with contextlib.redirect_stdout(io.StringIO()):
            assert E.select(run, ev, "v3", cap=cap)["chosen"] == "a.pt"      # reused
            e = _raises(SystemExit, E.select, run, ev, "v3")                  # 900 now
        assert "max_ticks 13" in str(e)
        # a selection.json from before the option (no max_ticks) is a 900 one
        legacy = _tmp()
        json.dump(dict(chosen="a.pt"), open(os.path.join(legacy, "selection.json"), "w"))
        with contextlib.redirect_stdout(io.StringIO()):
            assert E.select(run, legacy, "v3")["chosen"] == "a.pt"
        _raises(SystemExit, E.select, run, legacy, "v3", cap=cap)
        # the default passes no cap keyword at all (stand-ins keep working)
        seen.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            E.select(run, _tmp(), "v3")
        assert seen == [None]
    finally:
        E.candidates, E.deploy = saved
        shutil.rmtree(run, ignore_errors=True)


def test_zzz_cleanup():
    while _TMP:
        shutil.rmtree(_TMP.pop(), ignore_errors=True)


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
            import traceback
            traceback.print_exc()
            print("FAIL  %s\n        %s: %s" % (name, type(e).__name__, e))
        else:
            print("ok    %s" % name)
    print("\n%d/%d passed" % (len(_tests()) - failed, len(_tests())))
    raise SystemExit(1 if failed else 0)
