"""tools/train_policy.py is tools/train_bc.py under another name: identical runs, bit for bit.

    python tools/test_train_policy.py        (or: python -m pytest tools/test_train_policy.py)

  * the wrapper hands train_bc.main() exactly the arguments it was given, and
    restores sys.argv for an in-process caller;
  * IN PROCESS, every model `--model` takes (bc, act, act_lstm, act_orig,
    act_lstm_full), at test width: a 4-step `train` run through train_bc.main()
    and the same run through train_policy.main() give the same exit code, the same
    files, every checkpoint and state.pt identical (every tensor to the bit,
    optimizer moments and RNG streams included), and the same metadata.json,
    result.json, metrics.jsonl and console output - except the fields that hold
    wall-clock times (`VOLATILE`), which also differ between two train_bc.py runs;
  * AS PROGRAMS (python tools/train_bc.py vs python tools/train_policy.py, separate
    processes, train_bc.py running as __main__): a full-size BC run, the same
    comparison; a refused configuration (act_orig at the default W_o = 12) and an
    argparse error give the same exit code and message; --help is train_bc.py's,
    with only the program name changed.

The fixture is three tiny real-format episodes (`g1_model.test_train._tiny`). Its
episode metadata comes from data/synthetic when something is staged there, and
otherwise from ONE scripted episode recorded in memory (`test_deploy._recorded`),
so this test never skips. Everything is written to temp dirs.
"""
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402

#: Fields that hold wall-clock times or dates. They differ between ANY two runs,
#: two train_bc.py runs included; everything else must be identical.
VOLATILE = frozenset({"started_utc", "written_utc", "window_seconds", "elapsed_seconds",
                      "wall_seconds", "total_wall_seconds", "steps_per_second",
                      "segment_steps_per_second", "epoch_seconds"})
MODELS = ("bc", "act", "act_lstm", "act_orig", "act_lstm_full")
_CACHE: dict = {}


def _tmpdir() -> str:
    d = tempfile.mkdtemp(prefix="train_policy_test_")
    atexit.register(shutil.rmtree, d, True)
    return d


# ─── fixture ─────────────────────────────────────────────────────────────────
def _data():
    """(episodes dir, splits, norm): three tiny real-format episodes, seeds 0-1
    train and 2 val, stats fitted on the train split. Cached per process."""
    if "data" in _CACHE:
        return _CACHE["data"]
    from g1_model.test_train import _tiny
    got = _tiny(n_eps=3, n_ticks=30, masked_consistent=True)
    if got is None:
        # nothing staged in data/synthetic: take the episode metadata from one
        # scripted episode recorded in memory, through _tiny itself
        import test_deploy as TD
        path, _, _ = TD._recorded()
        src = _tmpdir()
        shutil.copy(path, src)
        saved = DS.SYNTHETIC
        DS.SYNTHETIC = src
        try:
            got = _tiny(n_eps=3, n_ticks=30, masked_consistent=True)
        finally:
            DS.SYNTHETIC = saved
    ep_dir, _ = got
    atexit.register(shutil.rmtree, ep_dir, True)
    tmp = _tmpdir()
    eps = [e for e in DS.scan(ep_dir) if e.seed in (0, 1)]
    st, ac, rep = DS.fit_norm_stats(eps, "train", allow_synthetic=True)
    norm = DS.save_norm_stats(st, ac, path=os.path.join(tmp, "norm.npz"), source=rep["source"],
                              n_episodes=2, seeds=rep["seeds"], split="train")
    splits = os.path.join(tmp, "splits.json")
    with open(splits, "w", encoding="utf-8") as fh:
        json.dump(dict(train=[0, 1], val=[2], spec_version=spec.SPEC_VERSION), fh)
    _CACHE["data"] = (ep_dir, splits, norm)
    return _CACHE["data"]


def _train_args(kind, out, *extra):
    ep, splits, norm = _data()
    w = ("--obs-window", 1) if kind in ("act_orig", "act_lstm_full") else ()
    return ["train", "--model", kind, "--chunk", 1 if kind == "bc" else 4, "--batch", 4,
            "--max-steps", 4, "--window-steps", 2, "--checkpoint-every", 2,
            "--budget-reason", "test", "--episodes", ep, "--splits", splits,
            "--norm", norm, "--out-dir", out, *w, *extra]


@contextlib.contextmanager
def _small_models():
    """train_bc's OWN configs at test width (as tools/test_memory_design.py does):
    only the transformer's width/depth or BC's hidden sizes change, the config TYPE
    and every other field stay, so both entry points run the real wiring."""
    import train_bc as TB
    from g1_model.act import ACTConfig
    real = TB._model_config

    def small(kind, K, W, batch, gate):
        c = real(kind, K, W, batch, gate)
        if isinstance(c, ACTConfig):
            return dataclasses.replace(c, hidden_dim=64, dim_feedforward=128, nheads=4,
                                       enc_layers=1)
        return dataclasses.replace(c, hidden=(64, 64))
    TB._model_config = small
    try:
        yield
    finally:
        TB._model_config = real


def _in_process(entry, args):
    """Run `entry` ("train_bc" or "train_policy") in this process: (exit code,
    stdout, stderr). train_bc.main() reads sys.argv; train_policy.main() is given
    the arguments, which is the wrapper's own in-process interface."""
    import train_bc as TB
    import train_policy as TP
    saved_argv, saved_data = list(sys.argv), dict(TB.DATA)
    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if entry == "train_bc":
                sys.argv = ["train_bc.py"] + [str(a) for a in args]
                TB.main()
            else:
                TP.main(args)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else str(e.code))
    finally:
        sys.argv = saved_argv
        TB.DATA.clear()
        TB.DATA.update(saved_data)
    return code, out.getvalue(), err.getvalue()


def _cli(script, args, timeout=900):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + [str(a) for a in args],
                       cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


# ─── comparison ──────────────────────────────────────────────────────────────
def _norm_text(s: str, outs) -> str:
    for o in outs:
        for form in (o, repo_relpath(o), os.path.abspath(o)):
            s = s.replace(form, "<OUT>")
    s = s.replace("train_policy.py", "<PROG>").replace("train_bc.py", "<PROG>")
    s = re.sub(r"\d+(?:\.\d+)?\s*(?:steps/s|s)\b", "<TIME>", s)
    # argparse wraps its usage lines by the program name's length, which differs
    # (Python 3.10 can also break before a flag's metavar or the subcommands' "...")
    return re.sub(r"[ \t]*\n[ \t]+(?=[\[\-({A-Z.])", " ", s)


def _diff(a, b, outs, path="", acc=None):
    """Paths at which `a` and `b` differ, recursively, VOLATILE keys skipped,
    tensors and arrays compared to the bit."""
    import torch
    acc = [] if acc is None else acc
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            acc.append("%s: keys %s" % (path, sorted(set(a) ^ set(b), key=str)))
        for k in set(a) & set(b):
            if k not in VOLATILE:
                _diff(a[k], b[k], outs, "%s/%s" % (path, k), acc)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            acc.append("%s: %s len %d vs %s len %d" % (path, type(a).__name__, len(a),
                                                     type(b).__name__, len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diff(x, y, outs, "%s[%d]" % (path, i), acc)
    elif torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b)):
            acc.append(path)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and np.array_equal(a, b)):
            acc.append(path)
    elif isinstance(a, str) and isinstance(b, str):
        if _norm_text(a, outs) != _norm_text(b, outs):
            acc.append("%s: %r vs %r" % (path, a[:80], b[:80]))
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    elif type(a) is not type(b) or a != b:
        acc.append("%s: %r vs %r" % (path, a, b))
    return acc


def _load(path):
    import torch
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(L) for L in fh if L.strip()]
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "rb") as fh:
        return fh.read()


def _same_run(a, b):
    """Every file of run `a` equals run `b`'s (VOLATILE fields aside); returns the
    files compared."""
    fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
    assert fa == fb, (fa, fb)
    assert {"best.pt", "last.pt", "state.pt", "metadata.json", "result.json",
            "metrics.jsonl"} <= set(fa), fa
    for f in fa:
        d = _diff(_load(os.path.join(a, f)), _load(os.path.join(b, f)), (a, b), f)
        assert not d, d[:10]
    return fa


def _same_console(x, y, outs):
    assert x[0] == y[0], (x[0], y[0], x[2][-2000:], y[2][-2000:])
    for i, what in ((1, "stdout"), (2, "stderr")):
        u, v = _norm_text(x[i], outs), _norm_text(y[i], outs)
        assert u == v, (what, [(p, q) for p, q in zip(u.splitlines(), v.splitlines())
                               if p != q][:5])


# ─── tests ───────────────────────────────────────────────────────────────────
def test_wrapper_passes_its_arguments_through_unchanged():
    import train_bc as TB
    import train_policy as TP
    seen = []
    real = TB.main
    TB.main = lambda: seen.append(list(sys.argv))
    saved = list(sys.argv)
    try:
        args = ["train", "--model", "act_lstm_full", "--obs-window", 1, "--budget-reason",
                "two words", "--episodes", "a b/c"]
        TP.main(args)
        assert seen[-1][1:] == [str(a) for a in args], seen[-1]
        assert sys.argv == saved, "sys.argv restored for an in-process caller"
        TP.main()                                       # no argv: the process's own
        assert seen[-1] == saved, seen[-1]
    finally:
        TB.main = real
    assert TP.train_bc is TB, "the wrapper calls THE train_bc module, not a copy"


def test_every_model_trains_identically_through_both_entry_points():
    for kind in MODELS:
        a, b = os.path.join(_tmpdir(), "bc_entry"), os.path.join(_tmpdir(), "policy_entry")
        with _small_models():
            x = _in_process("train_bc", _train_args(kind, a))
            y = _in_process("train_policy", _train_args(kind, b))
        assert x[0] == 0, (kind, x[0], x[1][-3000:], x[2][-3000:])
        _same_console(x, y, (a, b))
        files = _same_run(a, b)
        meta = _load(os.path.join(a, "metadata.json"))
        assert meta["config"]["notes"]["model"] == kind, kind
        assert "ckpt_step0000004.pt" in files, (kind, files)


def test_as_programs_a_full_size_bc_run_is_identical():
    a, b = os.path.join(_tmpdir(), "bc_entry"), os.path.join(_tmpdir(), "policy_entry")
    x = _cli("train_bc.py", _train_args("bc", a, "--obs-window", 12))
    y = _cli("train_policy.py", _train_args("bc", b, "--obs-window", 12))
    assert x[0] == 0, (x[0], x[1][-3000:], x[2][-3000:])
    _same_console(x, y, (a, b))
    _same_run(a, b)
    meta = _load(os.path.join(a, "metadata.json"))
    assert meta["model"]["cls"] == "BCPolicy" and meta["loader"]["obs_window"] == 12


def test_as_programs_refusals_errors_and_help_are_train_bcs():
    ep, splits, norm = _data()
    refused = ["train", "--model", "act_orig", "--chunk", 4, "--max-steps", 2,
               "--budget-reason", "test", "--episodes", ep, "--splits", splits, "--norm", norm]
    a, b = os.path.join(_tmpdir(), "x"), os.path.join(_tmpdir(), "y")
    x = _cli("train_bc.py", refused + ["--out-dir", a])
    y = _cli("train_policy.py", refused + ["--out-dir", b])
    assert x[0] == 1 and "pass --obs-window 1" in x[2], x
    _same_console(x, y, (a, b))
    no_reason = ["train", "--model", "bc", "--max-steps", 2]  # --budget-reason missing
    x, y = _cli("train_bc.py", no_reason), _cli("train_policy.py", no_reason)
    assert x[0] == 2 and "--budget-reason" in x[2], x
    _same_console(x, y, ())
    x, y = _cli("train_bc.py", ["--help"]), _cli("train_policy.py", ["--help"])
    assert x[0] == 0 and "THE GATE CRITERION" in x[1], x[1][:300]
    _same_console(x, y, ())
    x, y = _cli("train_bc.py", ["train", "--help"]), _cli("train_policy.py", ["train", "--help"])
    assert x[0] == 0 and "act_lstm_full" in x[1]
    _same_console(x, y, ())


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
