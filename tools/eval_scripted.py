"""Checkpoint selection, then Experiments 1 and 2, for trained runs - ONE deploy at a time.

    python tools/eval_scripted.py --runs bc_160_seed0 act_160_seed0 act_lstm_160_seed0
    python tools/eval_scripted.py --wait-for-training --runs bc_n40_seed0 ...

For each run in runs/scripted_v3/ (--run-root), in the order given:

  1. SELECTION. Every saved checkpoint (ckpt_*.pt, best.pt, last.pt; a file whose
     weights are identical to an earlier candidate's - last.pt is the final
     periodic checkpoint - is dropped as an alias) is deployed on the 20
     selection seeds (tools/deploy.py --selection, data/eval/select_seeds.json,
     disjoint from train, exp1 and exp2). The chosen checkpoint is the
     lexicographic best of
         (full successes, place, transport, lift, grasp, -mean placement error)
     - full success first, then the stage-wise counts from the LAST stage back
     (each stage implies the ones before it), then placement proximity (mean
     success.py placement error over the 20 episodes, metres). A tie on all of
     these goes to the checkpoint with the FEWEST training steps (epochs for BC).
     The choice and every candidate's tally go to <eval>/<run>/selection.json.
  2. EXP 1: the chosen checkpoint on data/eval/exp1_seeds.json (100 fresh seeds
     from the training region) -> <eval>/<run>/exp1.jsonl
  3. EXP 2: the same checkpoint on data/eval/exp2_seeds.json (100 held-out-patch
     seeds) -> <eval>/<run>/exp2.jsonl

Every JSONL is deploy.py's own per-episode record (success, grasp/lift/
transport/place, placement_error_m and the other distances, fail_reason = the
ScoredPhase the failure is charged to, phase_reached, lock/weld ticks, header).

Lock rule: --lock-rule (default v3, CLAUDE.md section 8 2026-10-24) for every
deploy, selection included.

RESTART-SAFE: a JSONL is written as <name>.jsonl.part and renamed only when it
holds one parseable line per seed; a finished JSONL is never re-run, a stale
.part is deleted and re-run. selection.json, once written, fixes the choice.
Completion is judged by the line count, NOT the exit code: the cuDNN LSTM
crashes Python AT EXIT (0xC0000409) after a clean finish (section 8,
2026-09-29). A run without a finished result.json ("finished" / "hard_cap")
is skipped and reported.

ONE HEAVY PROCESS AT A TIME. --wait-for-training first waits, every --poll s,
until (a) no train_bc.py, deploy.py, train_scripted_v3_n40.ps1 or other
eval_scripted.py process is running, seen idle on two checks 60 s apart, and
(b) every requested run has a finished result.json OR the training launcher's
summary (runs/scripted_v3/summary_n40.txt) has an "ended" line after its last
"started" line - so starting this before the training launcher never races it.
Without the flag it refuses to start while a train_bc.py or deploy.py runs.

Per-run summary lines are appended to <eval>/summary.txt.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_data import dataset as DS                                   # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402

EXP = (("exp1", os.path.join(ROOT, "data", "eval", "exp1_seeds.json")),
       ("exp2", os.path.join(ROOT, "data", "eval", "exp2_seeds.json")))
DONE_STATUS = ("finished", "hard_cap")
STAGES = ("place", "transport", "lift", "grasp")        # last stage first
SELECTION_RULE = ("max (successes, place, transport, lift, grasp, -mean placement_error_m) "
                  "over the 20 selection seeds; tie -> least training: optimizer steps for a "
                  "step-budgeted run, epochs for an epoch-budgeted one (BC)")
HEAVY = re.compile(r"train_bc\.py|deploy\.py|train_scripted_v3_n40\.ps1|eval_scripted\.py")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def log(msg: str) -> None:
    print("[%s] %s" % (now(), msg), flush=True)


# ─── processes ───────────────────────────────────────────────────────────────
def _heavy_others(pattern=HEAVY):
    """Command lines of OTHER processes matching `pattern` (never this one or
    its own children/parents)."""
    import psutil
    me = psutil.Process()
    mine = {me.pid} | {p.pid for p in me.children(recursive=True)} | {p.pid for p in me.parents()}
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in mine:
                continue
            exe = (p.info["name"] or "").lower()
            args = p.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        # Only an interpreter RUNNING one of the scripts counts - not a shell or
        # editor whose command line merely mentions the name.
        if not (exe.startswith("python") or exe.startswith("powershell") or exe.startswith("pwsh")):
            continue
        if any(pattern.fullmatch(os.path.basename(x)) for x in args[1:]):
            out.append(" ".join(args))
    return out


def _launcher_ended(summary: str) -> bool:
    if not os.path.exists(summary):
        return False
    s = e = -1
    for i, line in enumerate(open(summary, encoding="utf-8-sig", errors="replace")):
        if "training started" in line:
            s = i
        if "training ended" in line:
            e = i
    return s >= 0 and e > s


def wait_for_training(run_dirs, summary: str, poll: int) -> None:
    idle = 0
    while True:
        busy = _heavy_others()
        complete = all(run_status(d) in DONE_STATUS for d in run_dirs)
        if busy or not (complete or _launcher_ended(summary)):
            idle = 0
            what = ", ".join(sorted({os.path.basename(m.group(0)) for c in busy
                                     for m in [HEAVY.search(c)] if m})) or "training not started/finished"
            log("waiting: %s - next check in %d s" % (what, poll))
            time.sleep(poll)
            continue
        idle += 1
        if idle >= 2:
            return
        log("idle - confirming in 60 s")
        time.sleep(60)


# ─── runs and checkpoints ────────────────────────────────────────────────────
def run_status(run_dir: str) -> str:
    p = os.path.join(run_dir, "result.json")
    if not os.path.exists(p):
        return "no result.json"
    try:
        return str(json.load(open(p, encoding="utf-8")).get("status"))
    except Exception:                                    # noqa: BLE001
        return "unreadable result.json"


def _weights_hash(path: str) -> str:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for k, v in sorted(ck["state_dict"].items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def candidates(run_dir: str):
    """[(file, alias_of_or_None)] - periodic checkpoints in order, then best, last."""
    periodic = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
    names = periodic + [f for f in ("best.pt", "last.pt") if os.path.exists(os.path.join(run_dir, f))]
    seen, out = {}, []
    for f in names:
        h = _weights_hash(os.path.join(run_dir, f))
        out.append((f, seen.get(h)))
        seen.setdefault(h, f)
    return out


# ─── one deploy ──────────────────────────────────────────────────────────────
def _lines(path: str):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def deploy(run_dir: str, ckpt: str, seed_args, n_seeds: int, out: str, lock_rule: str,
           logdir: str, allow_mismatch: bool = False):
    """Run tools/deploy.py into `out` (via .part); return its rows."""
    if os.path.exists(out):
        rows = _lines(out)
        if len(rows) == n_seeds:
            return rows
        raise SystemExit("%s exists with %d rows, expected %d - inspect it by hand"
                         % (out, len(rows), n_seeds))
    part = out + ".part"
    if os.path.exists(part):
        os.remove(part)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    lg = os.path.join(logdir, os.path.basename(out).replace(".jsonl", ".log"))
    cmd = [sys.executable, "-u", os.path.join(ROOT, "tools", "deploy.py"), "--run", run_dir,
           "--ckpt", ckpt, "--lock-rule", lock_rule, "--out", part] + list(seed_args) + (
               ["--allow-lock-rule-mismatch"] if allow_mismatch else [])
    t0 = time.time()
    with open(lg, "w", encoding="utf-8") as fh:
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    rows = _lines(part) if os.path.exists(part) else []
    if len(rows) != n_seeds:
        raise SystemExit("deploy of %s/%s wrote %d/%d episodes (exit %s) - see %s"
                         % (os.path.basename(run_dir), ckpt, len(rows), n_seeds, code, lg))
    os.replace(part, out)
    log("  %-24s %2d/%d success  (%.1f min, exit %s)"
        % (os.path.basename(out), sum(r["success"] for r in rows), n_seeds,
           (time.time() - t0) / 60, code))
    return rows


def tally(rows) -> dict:
    pe = [float(r["placement_error_m"]) for r in rows
          if r.get("placement_error_m") is not None and math.isfinite(float(r["placement_error_m"]))]
    t = dict(n=len(rows), success=sum(bool(r["success"]) for r in rows))
    for s in reversed(STAGES):
        t[s] = sum(bool(r[s]) for r in rows)
    t["mean_placement_error_m"] = round(sum(pe) / len(pe), 4) if pe else None
    return t


def epoch_budgeted(run_dir: str) -> bool:
    meta = json.load(open(os.path.join(run_dir, "metadata.json"), encoding="utf-8"))
    return meta["config"].get("epochs") is not None


def _progress(rows, by_epoch: bool) -> float:
    """Training done at this checkpoint, in ONE unit per run (review,
    2026-10-24): BC's periodic checkpoints carry a step count and its best.pt
    does not, so "step, else epoch" compared 3.3M steps with 294 epochs.
    Epoch-budgeted runs (BC) use epochs throughout; step-budgeted ones steps."""
    ck = rows[0]["header"]["ckpt"]
    v = ck.get("epoch") if by_epoch else ck.get("step")
    if v is None:
        raise SystemExit("checkpoint %s records no %s" % (ck.get("file"),
                                                           "epoch" if by_epoch else "step"))
    return float(v)


def select(run_dir: str, ev: str, lock_rule: str, allow_mismatch: bool = False) -> dict:
    path = os.path.join(ev, "selection.json")
    if os.path.exists(path):
        sel = json.load(open(path, encoding="utf-8"))
        log("  selection.json exists: %s" % sel["chosen"])
        return sel
    n_sel = len(DS.load_seed_file(DS.SELECT_SEEDS))
    by_epoch = epoch_budgeted(run_dir)
    cands = []
    for f, alias in candidates(run_dir):
        if alias:
            log("  %s: identical weights to %s - alias, not deployed" % (f, alias))
            cands.append(dict(ckpt=f, alias_of=alias))
            continue
        rows = deploy(run_dir, f, ["--selection"], n_sel,
                      os.path.join(ev, "select", f.replace(".pt", ".jsonl")),
                      lock_rule, os.path.join(ev, "logs"), allow_mismatch)
        cands.append(dict(ckpt=f, alias_of=None, progress=_progress(rows, by_epoch),
                          progress_unit="epoch" if by_epoch else "step",
                          ckpt_info=rows[0]["header"]["ckpt"], **tally(rows)))
    real = [c for c in cands if not c["alias_of"]]

    def key(c):
        pe = c["mean_placement_error_m"]
        return (c["success"],) + tuple(c[s] for s in STAGES) + (
            -(pe if pe is not None else float("inf")), -c["progress"])
    best = max(real, key=key)
    sel = dict(run=repo_relpath(run_dir), chosen=best["ckpt"],
               chosen_tally={k: best[k] for k in ("n", "success", "grasp", "lift", "transport",
                                                   "place", "mean_placement_error_m")},
               rule=SELECTION_RULE, lock_rule=lock_rule,
               seed_set=repo_relpath(DS.SELECT_SEEDS), candidates=cands,
               written=now())
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1)
    log("  SELECTED %s (%d/%d success)" % (best["ckpt"], best["success"], best["n"]))
    return sel


def evaluate(run_dir: str, eval_root: str, lock_rule: str, allow_mismatch: bool = False) -> dict:
    name = os.path.basename(os.path.normpath(run_dir))
    ev = os.path.join(eval_root, name)
    sel = select(run_dir, ev, lock_rule, allow_mismatch)
    res = dict(run=name, chosen=sel["chosen"], select=sel["chosen_tally"])
    for tag, seeds in EXP:
        rows = deploy(run_dir, sel["chosen"], ["--seeds", seeds], len(DS.load_seed_file(seeds)),
                      os.path.join(ev, "%s.jsonl" % tag), lock_rule, os.path.join(ev, "logs"),
                      allow_mismatch)
        res[tag] = tally(rows)
    return res


def line(r: dict) -> str:
    def f(t):
        return "%3d/%d (g %d l %d t %d p %d, place err %.3f m)" % (
            t["success"], t["n"], t["grasp"], t["lift"], t["transport"], t["place"],
            t["mean_placement_error_m"] if t["mean_placement_error_m"] is not None else float("nan"))
    return "%-22s %-20s sel %d/%d | exp1 %s | exp2 %s" % (
        r["run"], r["chosen"], r["select"]["success"], r["select"]["n"], f(r["exp1"]), f(r["exp2"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folder names under --run-root")
    ap.add_argument("--run-root", default=os.path.join(ROOT, "runs", "scripted_v3"))
    ap.add_argument("--eval-root", default=None, help="default <run-root>/eval")
    ap.add_argument("--lock-rule", default="v3")
    ap.add_argument("--allow-lock-rule-mismatch", action="store_true",
                    help="passed to deploy.py: evaluate under --lock-rule although the "
                         "training data used another (scripted_v3 data is v2; the "
                         "experiments run v3 by decision, CLAUDE.md section 8 2026-10-24)")
    ap.add_argument("--wait-for-training", action="store_true")
    ap.add_argument("--training-summary", default=None,
                    help="default <run-root>/summary_n40.txt")
    ap.add_argument("--poll", type=int, default=300)
    a = ap.parse_args(argv)
    eval_root = a.eval_root or os.path.join(a.run_root, "eval")
    run_dirs = [os.path.join(a.run_root, r) for r in a.runs]
    if a.wait_for_training:
        wait_for_training(run_dirs, a.training_summary or os.path.join(a.run_root, "summary_n40.txt"),
                          a.poll)
    else:
        busy = [c for c in _heavy_others(re.compile(r"train_bc\.py|deploy\.py"))]
        if busy:
            raise SystemExit("refusing: another heavy process is running (one at a time):\n  "
                             + "\n  ".join(busy))
    os.makedirs(eval_root, exist_ok=True)
    summary = os.path.join(eval_root, "summary.txt")
    with open(summary, "a", encoding="utf-8") as fh:
        fh.write("eval started %s  lock rule %s  runs %s\n" % (now(), a.lock_rule, " ".join(a.runs)))
    results, skipped = [], []
    for d in run_dirs:
        st = run_status(d)
        if st not in DONE_STATUS:
            log("SKIP %s: training status %r" % (os.path.basename(d), st))
            skipped.append((os.path.basename(d), st))
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write("%-22s SKIPPED: training status %s\n" % (os.path.basename(d), st))
            continue
        log("=== %s" % os.path.basename(d))
        r = evaluate(d, eval_root, a.lock_rule, a.allow_lock_rule_mismatch)
        results.append(r)
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(line(r) + "\n")
    with open(summary, "a", encoding="utf-8") as fh:
        fh.write("eval ended %s\n" % now())
    print("")
    for r in results:
        print(line(r))
    for n, st in skipped:
        print("%-22s SKIPPED (%s)" % (n, st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
