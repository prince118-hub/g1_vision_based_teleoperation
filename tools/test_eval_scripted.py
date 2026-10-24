"""Tests for tools/eval_scripted.py - checkpoint selection (no simulation).

    python tools/test_eval_scripted.py

deploy() is replaced by a stub that returns hand-built episode rows, so the
selection rule itself is what is tested.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import eval_scripted as E                                           # noqa: E402


def _rows(ckpt, step, epoch, success=0, grasp=0, place_err=1.5, n=20):
    hdr = dict(ckpt=dict(file=ckpt, step=step, epoch=epoch))
    out = []
    for i in range(n):
        out.append(dict(success=i < success, grasp=i < grasp, lift=i < grasp,
                        transport=i < success, place=i < success,
                        placement_error_m=place_err, header=hdr))
    return out


def _select(epochs, table):
    """Run E.select on a fake run dir with stubbed candidates and deploy."""
    run = os.path.join(E.ROOT, "runs", "_test_eval_scripted_%d" % os.getpid())
    ev = tempfile.mkdtemp()
    os.makedirs(run, exist_ok=True)
    json.dump(dict(config=dict(epochs=epochs)), open(os.path.join(run, "metadata.json"), "w"))
    saved = E.candidates, E.deploy
    E.candidates = lambda d: [(f, None) for f in table]
    E.deploy = lambda run_dir, ckpt, *a, **k: table[ckpt]
    try:
        return E.select(run, ev, "v3")
    finally:
        E.candidates, E.deploy = saved
        shutil.rmtree(run, ignore_errors=True)
        shutil.rmtree(ev, ignore_errors=True)


def test_success_then_last_stage_first_then_placement():
    t = {"a.pt": _rows("a.pt", 1000, 1, success=2),
         "b.pt": _rows("b.pt", 2000, 2, success=3),
         "c.pt": _rows("c.pt", 3000, 3, success=3, place_err=1.0)}
    assert _select(None, t)["chosen"] == "c.pt"
    t = {"a.pt": _rows("a.pt", 1000, 1, grasp=5, place_err=0.1),
         "b.pt": _rows("b.pt", 2000, 2, grasp=6, place_err=1.9)}
    assert _select(None, t)["chosen"] == "b.pt"


def test_bc_tie_break_is_by_epoch_not_a_mix_of_steps_and_epochs():
    """Review 2026-10-24: BC's periodic checkpoints carry a step count and its
    best.pt does not, so "step, else epoch" read best.pt (epoch 294) as LESS
    training than epoch 30 (331,440 steps) and won a full tie with it."""
    t = {"ckpt_epoch0030.pt": _rows("ckpt_epoch0030.pt", 331440, 30),
         "best.pt": _rows("best.pt", None, 294)}
    sel = _select(300, t)
    assert sel["chosen"] == "ckpt_epoch0030.pt", sel["chosen"]
    assert {c["progress_unit"] for c in sel["candidates"]} == {"epoch"}
    # a step-budgeted run ties on steps
    t = {"x.pt": _rows("x.pt", 40000, 3), "y.pt": _rows("y.pt", 20000, 5)}
    assert _select(None, t)["chosen"] == "y.pt"


def test_progress_refuses_a_checkpoint_without_the_unit():
    try:
        E._progress(_rows("best.pt", None, 294), by_epoch=False)
    except SystemExit as e:
        assert "step" in str(e)
    else:
        raise AssertionError("a step-budgeted checkpoint with no step must be refused")


def test_selection_paths_are_repo_relative():
    sel = _select(None, {"a.pt": _rows("a.pt", 1, 1)})
    for k in ("run", "seed_set"):
        assert not os.path.isabs(sel[k]) and ".." not in sel[k], sel[k]
    assert sel["seed_set"].replace("\\", "/") == "data/eval/select_seeds.json"


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
