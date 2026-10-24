"""Regression proof for the memory-design change: OLD code against NEW code, same machine.

    python tools/check_memory_design_regression.py                       # base origin/main
    python tools/check_memory_design_regression.py --base main \
        --real runs/scripted_v3/act_lstm_n40_seed0:ckpt_step0200000.pt \
        --real runs/scripted_v3/bc_n40_seed0:best.pt

TR32: an equivalence claim runs the OLD code on one side. `--base` is checked out
into a temporary git worktree and ONE probe script (written below, identical for
both sides) runs once with that tree on sys.path and once with this checkout's:

  (b) MODELS. For every existing runner configuration - BC (K = 1 and K = 100)
      and ACT, ACT-LSTM (K = 100), each at the gate and outside it, W_o = 12,
      batch 8 - `train_bc._model_config` + `_make_model` (seed 0): the config,
      the class, hparams, every parameter and buffer (sha256 by name), the
      optimizer declaration, the regularization statement, and the output on a
      fixed input (sha256 of the bytes). Must be IDENTICAL.
  (c) CHECKPOINTS. The OLD code writes a checkpoint of each of bc, act and
      act_lstm with its own `train.save_checkpoint`, in a run directory with norm
      stats (non-trivial mean/std) and metadata, and records
      `deploy.CheckpointPolicy` actions over 60 ticks of fixed states. The NEW code
      loads THOSE files through its own CheckpointPolicy and must produce the same
      actions to the bit (np.array_equal), temporal ensembling included.
  --real RUN:CKPT, repeatable: the same check on a real trained run (both sides
      load it; relative data paths resolve against THIS checkout, --data-root).

Exit 0 only if everything is identical. A JSON report goes to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = r'''
import argparse, dataclasses, hashlib, json, math, os, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--make", action="store_true")
ap.add_argument("--load", default=None)
ap.add_argument("--real", action="append", default=[])
ap.add_argument("--ticks", type=int, default=60)
a = ap.parse_args()
sys.path[:0] = [a.repo, os.path.join(a.repo, "tools")]
import torch
import train_bc as TB
import deploy as D
from g1_data import dataset as DS, spec
from g1_model import train as T
assert os.path.realpath(os.path.dirname(TB.__file__)) == os.path.realpath(os.path.join(a.repo, "tools")), TB.__file__
os.makedirs(a.out, exist_ok=True)
CASES = [("bc", 1, False), ("bc", 1, True), ("bc", 100, False), ("act", 100, False),
         ("act", 100, True), ("act_lstm", 100, False), ("act_lstm", 100, True)]
W, BATCH = 12, 8

def jsonable(x):
    return json.loads(json.dumps(x, default=str))

def sha_state(model):
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest(), len(sd)

fp = {}
for kind, K, gate in CASES:
    mcfg = TB._model_config(kind, K, W, BATCH, gate=gate)
    tcfg = TB._train_config(mcfg, 1, BATCH, seed=0)
    model = TB._make_model(mcfg, tcfg).eval()
    state, n_tensors = sha_state(model)
    g = torch.Generator().manual_seed(123)
    obs = torch.randn(4, W, spec.STATE_DIM, generator=g)
    with torch.no_grad():
        out = model(obs)
    fp["%s_K%d_%s" % (kind, K, "gate" if gate else "train")] = dict(
        cls=type(model).__name__, config=jsonable(dataclasses.asdict(mcfg)),
        config_type=type(mcfg).__name__, hparams=jsonable(getattr(model, "hparams", {})),
        optimizer=jsonable(mcfg.optimizer_config()),
        regularization=TB.regularization_statement(mcfg),
        n_params=int(sum(p.numel() for p in model.parameters())), n_tensors=n_tensors,
        state_sha256=state,
        out_sha256=hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest(),
        out_fsum=math.fsum(out.double().flatten().tolist()))
json.dump(fp, open(os.path.join(a.out, "fingerprints.json"), "w"), indent=1)

def states_for(st, n, seed):
    rng = np.random.default_rng(seed)
    return st.mean + st.std * rng.standard_normal((n, spec.STATE_DIM))

def actions(run, ckpt, seed):
    pol = D.CheckpointPolicy(run, ckpt, device=torch.device("cpu"))
    pol.reset()
    S = states_for(pol.st, a.ticks, seed)
    return np.stack([pol.act(S[t], t) for t in range(a.ticks)]), pol

KINDS = (("bc", 1), ("act", 100), ("act_lstm", 100))
if a.make:
    rng = np.random.default_rng(7)
    st_std = 0.5 + rng.random(spec.STATE_DIM)
    st_mean = 0.1 * rng.standard_normal(spec.STATE_DIM)
    st_mean[~spec.STATE_MASK], st_std[~spec.STATE_MASK] = 0.0, 1.0
    ac_std = 0.5 + rng.random(spec.ACTION_DIM)
    ac_mean = 0.1 * rng.standard_normal(spec.ACTION_DIM)
    ac_mean[~spec.ACTION_MASK], ac_std[~spec.ACTION_MASK] = 0.0, 1.0
    st = spec.NormStats(kind="state", mean=st_mean, std=st_std)
    ac = spec.NormStats(kind="action", mean=ac_mean, std=ac_std)
    runs = os.path.join(a.out, "runs")
    norm = DS.save_norm_stats(st, ac, path=os.path.join(runs, "norm.npz"), source="scripted",
                              n_episodes=1, seeds=[0], split="train")
    splits = os.path.join(runs, "splits.json")
    json.dump(dict(train=[0], val=[], spec_version=spec.SPEC_VERSION), open(splits, "w"))
    for kind, K in KINDS:
        mcfg = TB._model_config(kind, K, W, BATCH, gate=False)
        tcfg = TB._train_config(mcfg, 1, BATCH, seed=0,
                                notes=dict(model=kind, chunk_size=K,
                                           data_paths=dict(episodes=runs, splits=splits,
                                                           norm=norm)))
        model = TB._make_model(mcfg, tcfg)
        run = os.path.join(runs, kind)
        os.makedirs(run, exist_ok=True)
        meta = dict(config=dataclasses.asdict(tcfg), data=dict(seeds=[0]),
                    git_commit=T._git_commit())
        json.dump(meta, open(os.path.join(run, "metadata.json"), "w"), default=str)
        T.save_checkpoint(os.path.join(run, "ckpt_step0000000.pt"), model, tcfg, meta,
                          epoch=0, train_loss=0.0, val_loss=None, step=0)

if a.load:
    for kind, _ in KINDS:
        A, pol = actions(os.path.join(a.load, kind), "ckpt_step0000000.pt", seed=11)
        np.save(os.path.join(a.out, "actions_%s.npy" % kind), A)
D.ROOT = a.data_root                     # relative data paths of a real run
for i, spec_ in enumerate(a.real):
    run, ckpt = spec_.rsplit(":", 1)
    A, pol = actions(run, ckpt, seed=100 + i)
    np.save(os.path.join(a.out, "real_%d.npy" % i), A)
print("PROBE_DONE")
'''


def _git(*args, cwd=ROOT):
    return subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


def _probe(script, repo, out, extra):
    cmd = [sys.executable, script, "--repo", repo, "--out", out, "--data-root", ROOT] + extra
    r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if "PROBE_DONE" not in r.stdout:
        raise SystemExit("probe failed in %s (exit %s):\n%s\n%s"
                         % (repo, r.returncode, r.stdout[-2000:], r.stderr[-4000:]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="the OLD code (any git revision)")
    ap.add_argument("--real", action="append", default=[], metavar="RUN:CKPT",
                    help="a real trained run and checkpoint, loaded by both sides")
    ap.add_argument("--out", default=None, help="JSON report")
    ap.add_argument("--keep", action="store_true", help="keep the temporary worktree")
    a = ap.parse_args(argv)
    base_rev = _git("rev-parse", a.base)
    head_rev = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    tmp = tempfile.mkdtemp(prefix="memdesign_regress_")
    base = os.path.join(tmp, "base")
    _git("worktree", "add", "--detach", base, base_rev)
    real = [os.path.abspath(r.rsplit(":", 1)[0]) + ":" + r.rsplit(":", 1)[1] for r in a.real]
    try:
        script = os.path.join(tmp, "probe.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(PROBE)
        old, new = os.path.join(tmp, "old"), os.path.join(tmp, "new")
        print("OLD code: %s (%s)" % (a.base, base_rev[:12]))
        _probe(script, base, old, ["--make", "--load", os.path.join(old, "runs")]
               + sum([["--real", r] for r in real], []))
        print("NEW code: this checkout (%s%s)" % (head_rev[:12], ", with uncommitted changes"
                                                  if dirty else ""))
        _probe(script, ROOT, new, ["--load", os.path.join(old, "runs")]
               + sum([["--real", r] for r in real], []))

        fo = json.load(open(os.path.join(old, "fingerprints.json"), encoding="utf-8"))
        fn = json.load(open(os.path.join(new, "fingerprints.json"), encoding="utf-8"))
        rows, ok = [], True
        for k in fo:
            same = fo[k] == fn.get(k)
            ok &= same
            diff = [f for f in fo[k] if fo[k][f] != (fn.get(k) or {}).get(f)]
            rows.append(dict(check="model %s" % k, identical=same, differing_fields=diff,
                             n_params=fo[k]["n_params"], state_sha256=fo[k]["state_sha256"][:16],
                             out_sha256=fo[k]["out_sha256"][:16]))
        for kind in ("bc", "act", "act_lstm"):
            A = np.load(os.path.join(old, "actions_%s.npy" % kind))
            B = np.load(os.path.join(new, "actions_%s.npy" % kind))
            same = A.shape == B.shape and np.array_equal(A, B)
            ok &= same
            rows.append(dict(check="old-format %s checkpoint -> actions" % kind, identical=same,
                             ticks=int(A.shape[0]),
                             max_abs_diff=float(np.abs(A - B).max()) if A.shape == B.shape else None))
        for i, r in enumerate(real):
            A = np.load(os.path.join(old, "real_%d.npy" % i))
            B = np.load(os.path.join(new, "real_%d.npy" % i))
            same = A.shape == B.shape and np.array_equal(A, B)
            ok &= same
            rows.append(dict(check="real %s -> actions" % r, identical=same, ticks=int(A.shape[0]),
                             max_abs_diff=float(np.abs(A - B).max()) if A.shape == B.shape else None))
        for r in rows:
            print("  %-4s %s%s" % ("OK" if r["identical"] else "DIFF", r["check"],
                                   "" if r["identical"] else "  %r" % r))
        print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
        if a.out:
            import torch
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(dict(base=a.base, base_commit=base_rev, head_commit=head_rev,
                               head_dirty=dirty, torch=torch.__version__, rows=rows,
                               all_identical=bool(ok)), fh, indent=1)
            print("wrote %s" % a.out)
        return 0 if ok else 1
    finally:
        if not a.keep:
            subprocess.run(["git", "worktree", "remove", "--force", base], cwd=ROOT,
                           capture_output=True)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
