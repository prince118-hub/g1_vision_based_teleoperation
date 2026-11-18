"""Human-data runs, ONE STEP AT A TIME: train one (model, seed), evaluate one, or show status.

    python tools/human_runs.py train  --model act_lstm_full --seed 0
    python tools/human_runs.py train  --model act_lstm_full --seed 0 --restart   # after an interruption
    python tools/human_runs.py eval   --model act_lstm_full --seed 0
    python tools/human_runs.py status
    python tools/human_runs.py dryrun                  # every path, short; measures the time

(PowerShell wrappers: tools\\run_human.ps1, tools\\eval_human.ps1, tools\\status_human.ps1.)

WHAT IS RUN
-----------
Data: the 40 human training demos of data/processed/splits_human_v1.json (40/10,
stratified over the scripted_v3 3x3 spawn bins), normalization fitted on those 40
only (norm_stats_human_v1.npz), episodes in recordings/demonstrations.
Runs go to runs/human_v1/<run>/, evaluations to runs/human_v1/eval/<run>/.

Settings are the SCRIPTED MAIN RUNS', unchanged (only the data differs):
    bc            BC, K=1, W_o=12, batch 256, 300 epochs, checkpoint every 30 epochs
                  (tools/train_scripted_v3_bc_b256.ps1)
    act_orig      ACT, K=100, W_o=1, batch 8, 200k steps, window 1000, checkpoint every 20k
    act_lstm_full ACT-LSTM (whole-episode LSTM), same as act_orig
                  (tools/train_memory_design_n40.ps1)
Seeds 0, 1, 2 (train_bc.py --seed: initial weights, data order, dropout).

Evaluation = tools/eval_scripted.py for that one run: checkpoint selection on the 20
selection seeds, then Exp 1 and Exp 2 (100 seeds each) on the chosen checkpoint,
lock rule v3 (the human data's own rule, so no mismatch flag), episode cap from
`--max-ticks-from recordings/demonstrations` = ceil(1.3 x longest accepted human demo)
(g1_data/eval_cap.py), recorded in every JSONL header.

NO MID-RUN RESUME (training)
----------------------------
A run is DONE when its result.json says "finished" or "hard_cap" - judged from the
file, never the exit code (the cuDNN LSTM crashes Python AT EXIT after a clean finish,
CLAUDE.md section 8 2026-09-29). A run folder that exists without that is INTERRUPTED:
`train` REFUSES it and says so. `train --restart` moves the incomplete folder (and any
evaluation of it) to runs/human_v1/_failed/<run>_<stamp>/ - kept, never deleted - and
trains from step 0. train_bc.py's --resume-from is never used, so every run is one
uninterrupted process from the same seed.

Evaluation is different by design: eval_scripted.py writes each JSONL as .part and
renames it only when complete, so an interrupted evaluation redoes the unfinished FILE
from its first seed and keeps finished ones. Each deploy episode is a deterministic
function of (checkpoint, seed, cap), so this changes no result.

ONE HEAVY PROCESS AT A TIME: train and eval REFUSE to start while any train_bc.py,
deploy.py or eval_scripted.py process runs (they never wait silently).

Every finished command writes runs/human_v1/summaries/<run>.train.txt or .eval.txt
(JSON) and appends one line to runs/human_v1/summary_human.txt.

DATASETS (--data, default human_v1)
-----------------------------------
human_v1  the recorded command as the walking label (recordings/demonstrations,
          *_human_v1 split/norm, runs/human_v1, run names *_human40_seed<k>).
human_v2  a TEST (2026-11-19, not adopted): the SAME 40/10 seeds, walking dims 19-21
          relabelled by tools/convert_velocity_label.py (centred moving average,
          spec.VELOCITY_LABEL_SMOOTH) into data/processed/episodes_human_v2, its own
          *_human_v2 split/norm, runs/human_v2, run names *_human40v2_seed<k> - so it
          can never write into human_v1. Same models, settings and evaluation; the cap
          still comes from recordings/demonstrations (identical episode lengths).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EPISODES = "recordings/demonstrations"
SPLITS = "data/processed/splits_human_v1.json"
NORM = "data/processed/norm_stats_human_v1.npz"
RUN_ROOT = "runs/human_v1"
DRY_ROOT = "runs/human_v1_dryrun"
SEEDS = (0, 1, 2)
DONE_STATUSES = ("finished", "hard_cap")
WHY = "human_v1 (40 teleop demos, spec 1.2.0, lock rule v3): settings of the scripted main runs"
#: The evaluation cap's source for EVERY dataset: the accepted human recordings (a
#: relabelled copy has the same episode lengths; only the raw folder has the ledger).
CAP_FROM = "recordings/demonstrations"
#: Run-name suffix after the stem ("" for human_v1, so its names are unchanged).
STEM_SUFFIX = ""
#: --data: what each dataset name points at. human_v1 is the module default above.
DATASETS: Dict[str, Dict[str, str]] = {
    "human_v1": dict(episodes=EPISODES, splits=SPLITS, norm=NORM, root=RUN_ROOT, suffix="", why=WHY),
    "human_v2": dict(episodes="data/processed/episodes_human_v2",
                     splits="data/processed/splits_human_v2.json",
                     norm="data/processed/norm_stats_human_v2.npz",
                     root="runs/human_v2", suffix="v2",
                     why="human_v2 (the human_v1 40/10 seeds; walking labels 19-21 = centred moving "
                         "average, velocity_label centred_ma13_v1; a TEST): settings of the scripted main runs"),
}
DATASET = "human_v1"

#: model -> (run-name stem, train_bc.py args without --seed, budget args, budget reason)
MODELS: Dict[str, Dict] = {
    "bc": dict(stem="bc_b256_human40",
               args=["--model", "bc", "--chunk", "1", "--obs-window", "12", "--batch", "256"],
               budget=["--epochs", "300", "--checkpoint-every", "30"],
               reason=WHY + ", 300 epochs, checkpoint every 30, selected later by task success"),
    "act_orig": dict(stem="act_orig_human40",
                     args=["--model", "act_orig", "--chunk", "100", "--obs-window", "1", "--batch", "8"],
                     budget=["--max-steps", "200000", "--window-steps", "1000", "--checkpoint-every", "20000"],
                     reason=WHY + ", 200k steps, checkpoint every 20k, no stop rule, selected later by task success"),
    "act_lstm_full": dict(stem="act_lstm_full_human40",
                          args=["--model", "act_lstm_full", "--chunk", "100", "--obs-window", "1", "--batch", "8"],
                          budget=["--max-steps", "200000", "--window-steps", "1000", "--checkpoint-every", "20000"],
                          reason=WHY + ", 200k steps, checkpoint every 20k, no stop rule, selected later by task success"),
}
#: Dry run: short budgets with periodic checkpoints, so selection has several candidates.
DRY_BUDGET = {"bc": ["--epochs", "3", "--checkpoint-every", "1"],
              "act_orig": ["--max-steps", "1000", "--window-steps", "100", "--checkpoint-every", "500"],
              "act_lstm_full": ["--max-steps", "1000", "--window-steps", "100", "--checkpoint-every", "500"]}
HEAVY_SCRIPTS = ("train_bc.py", "deploy.py", "eval_scripted.py")


def use_dataset(name: str) -> None:
    """Point EPISODES/SPLITS/NORM/RUN_ROOT and the run names at one dataset (CLI --data)."""
    global EPISODES, SPLITS, NORM, RUN_ROOT, STEM_SUFFIX, WHY, DATASET
    d = DATASETS[name]
    EPISODES, SPLITS, NORM, RUN_ROOT = d["episodes"], d["splits"], d["norm"], d["root"]
    STEM_SUFFIX, WHY, DATASET = d["suffix"], d["why"], name


def _reason(model: str) -> str:
    """The budget reason: the model's, with the current dataset's description in front."""
    r, v1 = MODELS[model]["reason"], DATASETS["human_v1"]["why"]
    return WHY + r[len(v1):] if r.startswith(v1) else r


def run_name(model: str, seed: int) -> str:
    return "%s%s_seed%d" % (MODELS[model]["stem"], STEM_SUFFIX, seed)


def all_runs():
    return [(m, s) for m in MODELS for s in SEEDS]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ─── state on disk ───────────────────────────────────────────────────────────
def _abs(*parts) -> str:
    return os.path.join(ROOT, *parts)


def _rel(path: str, start: str = ROOT) -> str:
    """Repo-relative for messages; the path itself when that is impossible (another drive)."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def train_status(root: str, name: str) -> Dict:
    """state: done | not_started | incomplete, from result.json only."""
    d = _abs(root, name)
    res = os.path.join(d, "result.json")
    if os.path.exists(res):
        try:
            r = json.load(open(res, encoding="utf-8"))
        except (OSError, ValueError):
            r = {}
        if r.get("status") in DONE_STATUSES:
            return dict(state="done", status=r["status"], steps=r.get("optimizer_steps"),
                        hours=round(float(r.get("total_wall_seconds") or r.get("wall_seconds") or 0) / 3600, 2))
        return dict(state="incomplete", status=r.get("status") or "result.json unreadable")
    if os.path.isdir(d) and os.listdir(d):
        return dict(state="incomplete", status="no result.json")
    return dict(state="not_started")


def _seed_count(path: str) -> int:
    d = json.load(open(_abs(path), encoding="utf-8"))
    return len(d["seeds"] if isinstance(d, dict) else d)


def _jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def eval_status(root: str, name: str) -> Dict:
    """state: done | not_started | incomplete."""
    d = _abs(root, "eval", name)
    n1, n2 = _seed_count("data/eval/exp1_seeds.json"), _seed_count("data/eval/exp2_seeds.json")
    sel = os.path.join(d, "selection.json")
    e1, e2 = _jsonl(os.path.join(d, "exp1.jsonl")), _jsonl(os.path.join(d, "exp2.jsonl"))
    if os.path.exists(sel) and len(e1) == n1 and len(e2) == n2:
        s = json.load(open(sel, encoding="utf-8"))
        return dict(state="done", chosen=s.get("chosen"),
                    exp1="%d/%d" % (sum(bool(x["success"]) for x in e1), n1),
                    exp2="%d/%d" % (sum(bool(x["success"]) for x in e2), n2),
                    max_ticks=(e1[0].get("header") or {}).get("max_ticks") if e1 else None)
    if not os.path.isdir(d) or not os.listdir(d):
        return dict(state="not_started")
    have = [("selection" if os.path.exists(sel) else None),
            ("exp1 %d/%d" % (len(e1), n1)), ("exp2 %d/%d" % (len(e2), n2))]
    return dict(state="incomplete", have=", ".join(h for h in have if h))


def heavy_processes(exclude_pid: Optional[int] = None) -> List[str]:
    import psutil
    out = []
    me = {os.getpid(), exclude_pid}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in me or not (p.info["name"] or "").lower().startswith("python"):
                continue
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.Error, OSError):
            continue
        hit = [s for s in HEAVY_SCRIPTS if s in cmd]
        if hit:
            out.append("pid %d %s" % (p.info["pid"], cmd[:160]))
    return out


def _refuse_if_heavy():
    h = heavy_processes()
    if h:
        raise SystemExit("REFUSED: another heavy process is running (one at a time):\n  "
                         + "\n  ".join(h) + "\nWait for it to finish, then run this again.")


def _gpu_used_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:                                    # noqa: BLE001 - no GPU tool is fine
        return -1


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _append_summary(root: str, line: str):
    os.makedirs(_abs(root), exist_ok=True)
    with open(_abs(root, "summary_human.txt"), "a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")
    print(line)


def _write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def _check_data():
    for p in (SPLITS, NORM):
        if not os.path.exists(_abs(p)):
            raise SystemExit("missing %s - build the human split and norm first (tools/build_dataset.py)" % p)
    if not os.path.isdir(_abs(EPISODES)):
        raise SystemExit("missing episode folder %s" % EPISODES)


def _run_logged(cmd: List[str], stdout_path: str, stderr_path: str):
    """Run cmd; peak RAM (child tree) and GPU sampled every 5 s. Ctrl+C stops the child."""
    import psutil
    t0 = time.time()
    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=err)
        peak_ram, peak_gpu = 0, _gpu_used_mib()
        interrupted = False
        try:
            while p.poll() is None:
                try:
                    pp = psutil.Process(p.pid)
                    rss = sum(x.memory_info().rss for x in [pp] + pp.children(recursive=True))
                    peak_ram = max(peak_ram, rss)
                except psutil.Error:
                    pass
                peak_gpu = max(peak_gpu, _gpu_used_mib())
                time.sleep(5)
        except KeyboardInterrupt:
            interrupted = True
            p.terminate()
        code = p.wait()
    return dict(exit_code=code, wall_seconds=round(time.time() - t0, 1), peak_ram_gb=round(peak_ram / 1e9, 2),
                peak_gpu_mib=peak_gpu, interrupted_by_user=interrupted)


# ─── train ───────────────────────────────────────────────────────────────────
def train(model: str, seed: int, restart: bool = False, root: str = RUN_ROOT,
          budget: Optional[List[str]] = None, reason: Optional[str] = None) -> int:
    _check_data()
    name = run_name(model, seed)
    st = train_status(root, name)
    if st["state"] == "done":
        print("%s: already DONE (status %s, %s steps, %.2f h) - nothing to do."
              % (name, st["status"], st["steps"], st["hours"]))
        return 0
    if st["state"] == "incomplete":
        if not restart:
            raise SystemExit(
                "REFUSED: %s was INTERRUPTED (%s). It will not be continued - a resumed run would not be\n"
                "identical to an uninterrupted one. Restart it FROM SCRATCH with:\n"
                "    powershell -ExecutionPolicy Bypass -File tools\\run_human.ps1 -Model %s -Seed %d%s -Restart\n"
                "(the incomplete folder is moved to %s/_failed/, not deleted)."
                % (name, st["status"], model, seed,
                   "" if DATASET == "human_v1" else " -Data " + DATASET, root))
    _refuse_if_heavy()
    run_dir = _abs(root, name)
    if st["state"] == "incomplete":
        dest = _abs(root, "_failed", "%s_%s" % (name, stamp()))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(run_dir, dest)
        for extra, suffix in ((_abs(root, name + ".stderr.log"), ".stderr.log"),
                              (_abs(root, "eval", name), ".eval")):
            if os.path.exists(extra):
                shutil.move(extra, dest + suffix)
        _append_summary(root, "%s %-28s RESTART: incomplete folder (%s) moved to %s"
                        % (now(), name, st["status"], _rel(dest, ROOT)))
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("WARNING: uncommitted changes to tracked files - this run's git_commit will not identify the "
              "code exactly:\n" + dirty)
    os.makedirs(run_dir, exist_ok=True)
    spec = MODELS[model]
    cmd = [sys.executable, "-u", os.path.join("tools", "train_bc.py"), "train", *spec["args"],
           "--seed", str(seed), *(budget or spec["budget"]), "--budget-reason", reason or _reason(model),
           "--episodes", EPISODES, "--splits", SPLITS, "--norm", NORM, "--out-dir", run_dir]
    commit = _git("rev-parse", "--short", "HEAD")
    _append_summary(root, "%s %-28s TRAIN started  commit %s" % (now(), name, commit))
    print("log: %s" % _rel(os.path.join(run_dir, "train.log"), ROOT))
    m = _run_logged(cmd, os.path.join(run_dir, "train.log"), _abs(root, name + ".stderr.log"))
    after = train_status(root, name)
    rec = dict(run=name, model=model, seed=seed, kind="train", ended=now(), git_commit=commit,
               git_dirty=bool(dirty), command=" ".join(cmd[1:]), state=after["state"],
               status=after.get("status"), optimizer_steps=after.get("steps"), **m)
    _write_json(_abs(root, "summaries", name + ".train.txt"), rec)
    _append_summary(root, "%s %-28s TRAIN %-10s status=%s steps=%s %.2f h exit=%s peak RAM %s GB peak GPU %s MiB"
                    % (now(), name, after["state"].upper(), after.get("status"), after.get("steps"),
                       m["wall_seconds"] / 3600, m["exit_code"], m["peak_ram_gb"], m["peak_gpu_mib"]))
    if after["state"] != "done":
        print("\n%s did NOT finish (see %s and %s.stderr.log). Restart it from scratch with -Restart."
              % (name, _rel(os.path.join(run_dir, "train.log"), ROOT), os.path.join(root, name)))
        return 1
    if m["exit_code"] != 0:
        print("(nonzero exit %s after a finished result.json: the known cuDNN LSTM crash at exit - the run "
              "is complete)" % m["exit_code"])
    return 0


# ─── eval ────────────────────────────────────────────────────────────────────
def evaluate(model: str, seed: int, root: str = RUN_ROOT, extra: Optional[List[str]] = None) -> int:
    name = run_name(model, seed)
    ts = train_status(root, name)
    if ts["state"] != "done":
        raise SystemExit("REFUSED: training of %s is %s - evaluate only a finished run."
                         % (name, ts["state"].replace("_", " ")))
    es = eval_status(root, name)
    if es["state"] == "done" and not extra:
        print("%s: evaluation already DONE (chosen %s, Exp 1 %s, Exp 2 %s)."
              % (name, es["chosen"], es["exp1"], es["exp2"]))
        return 0
    _refuse_if_heavy()
    if es["state"] == "incomplete":
        print("%s: evaluation incomplete (%s) - finished files are kept, the rest is redone from its first "
              "seed (deterministic per episode)." % (name, es["have"]))
    cap = extra or ["--max-ticks-from", CAP_FROM]
    cmd = [sys.executable, "-u", os.path.join("tools", "eval_scripted.py"), "--run-root", root,
           "--runs", name, "--lock-rule", "v3", *cap]
    ev_dir = _abs(root, "eval", name)
    os.makedirs(ev_dir, exist_ok=True)
    log = os.path.join(ev_dir, "eval_%s.log" % stamp())
    _append_summary(root, "%s %-28s EVAL started  log %s" % (now(), name, _rel(log, ROOT)))
    m = _run_logged(cmd, log, log + ".stderr")
    after = eval_status(root, name)
    rec = dict(run=name, model=model, seed=seed, kind="eval", ended=now(), command=" ".join(cmd[1:]),
               git_commit=_git("rev-parse", "--short", "HEAD"), **after, **m)
    _write_json(_abs(root, "summaries", name + ".eval.txt"), rec)
    _append_summary(root, "%s %-28s EVAL %-10s %s %.2f h"
                    % (now(), name, after["state"].upper(),
                       ("chosen %s  Exp1 %s  Exp2 %s  cap %s" % (after["chosen"], after["exp1"], after["exp2"],
                                                               after["max_ticks"]))
                       if after["state"] == "done" else after.get("have", ""), m["wall_seconds"] / 3600))
    if after["state"] != "done":
        print("evaluation not complete - see %s; run the same command again to finish it." % _rel(log, ROOT))
        return 1
    return 0


# ─── status ──────────────────────────────────────────────────────────────────
def status(root: str = RUN_ROOT) -> int:
    running = heavy_processes()
    print("Human-data runs under %s   (data %s, %s)" % (root, SPLITS, NORM))
    print("%-30s %-34s %s" % ("run", "training", "evaluation"))
    for model, seed in all_runs():
        name = run_name(model, seed)
        live = [r for r in running if name in r]
        t, e = train_status(root, name), eval_status(root, name)
        if live and any("train_bc.py" in r for r in live):
            tcol = "RUNNING now"
        elif t["state"] == "done":
            tcol = "done (%s, %s steps, %.1f h)" % (t["status"], t["steps"], t["hours"])
        elif t["state"] == "incomplete":
            tcol = "INCOMPLETE - restart with -Restart"
        else:
            tcol = "not started"
        if live and any("eval_scripted.py" in r or "deploy.py" in r for r in live):
            ecol = "RUNNING now"
        elif e["state"] == "done":
            ecol = "done: %s, Exp1 %s, Exp2 %s" % (e["chosen"], e["exp1"], e["exp2"])
        elif e["state"] == "incomplete":
            ecol = "incomplete (%s) - rerun eval" % e["have"]
        else:
            ecol = "not started" + ("" if t["state"] == "done" else " (needs training)")
        print("%-30s %-34s %s" % (name, tcol, ecol))
    other = [r for r in running if not any(run_name(m, s) in r for m, s in all_runs())]
    if other:
        print("\nother heavy process(es) running:\n  " + "\n  ".join(other))
    return 0


# ─── dry run ─────────────────────────────────────────────────────────────────
def dryrun(models: List[str]) -> int:
    """Seed 0 of each model, short budgets, into runs/human_v1_dryrun/: training, the full
    eval_scripted path (selection + Exp 1 + Exp 2 at a 25-tick cap), and one deploy at the
    REAL human cap to time a full-length episode. Prints measured times and estimates."""
    from g1_data import eval_cap as EC
    cap = EC.eval_cap(_abs(EPISODES))["max_ticks"]
    n_sel, n_exp = _seed_count("data/eval/select_seeds.json"), (_seed_count("data/eval/exp1_seeds.json")
                                                                + _seed_count("data/eval/exp2_seeds.json"))
    report = dict(cap=cap, models={})
    for model in models:
        name = run_name(model, 0)
        for p in (_abs(DRY_ROOT, name), _abs(DRY_ROOT, "eval", name), _abs(DRY_ROOT, name + ".stderr.log")):
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        rc = train(model, 0, root=DRY_ROOT, budget=DRY_BUDGET[model], reason="DRY RUN: timing and plumbing, deleted after")
        if rc != 0:
            report["models"][model] = dict(error="dry-run training failed")
            continue
        res = json.load(open(_abs(DRY_ROOT, name, "result.json"), encoding="utf-8"))
        tr = json.load(open(_abs(DRY_ROOT, "summaries", name + ".train.txt"), encoding="utf-8"))
        rc = evaluate(model, 0, root=DRY_ROOT, extra=["--max-ticks", "25"])
        # one full-cap episode: an untrained policy fails, so it runs to the cap (the worst case)
        out = _abs(DRY_ROOT, "eval", name, "fullcap_timing.jsonl")
        if os.path.exists(out):
            os.remove(out)
        t0 = time.time()
        subprocess.run([sys.executable, "-u", os.path.join("tools", "deploy.py"), "--run",
                        os.path.join(DRY_ROOT, name), "--ckpt", "best.pt", "--seed-list", "679",
                        "--max-ticks-from", EPISODES, "--out", out], cwd=ROOT)
        ep = _jsonl(out)
        full_s = ep[0]["wall_seconds"] if ep else None
        report["models"][model] = dict(
            train_steps=res.get("optimizer_steps"), steps_per_second=res.get("steps_per_second"),
            train_wall_s=tr["wall_seconds"], peak_ram_gb=tr["peak_ram_gb"], peak_gpu_mib=tr["peak_gpu_mib"],
            eval_plumbing="ok" if rc == 0 else "FAILED",
            fullcap_episode_ticks=ep[0]["ticks"] if ep else None, fullcap_episode_s=full_s,
            deploy_process_s=round(time.time() - t0, 1))
    _write_json(_abs(DRY_ROOT, "dryrun_report.json"), report)
    print("\nDRY RUN REPORT -> %s" % os.path.join(DRY_ROOT, "dryrun_report.json"))
    print(json.dumps(report, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("train", "eval"):
        p = sub.add_parser(c)
        p.add_argument("--data", default="human_v1", choices=sorted(DATASETS))
        p.add_argument("--model", required=True, choices=sorted(MODELS))
        p.add_argument("--seed", required=True, type=int, choices=SEEDS)
        if c == "train":
            p.add_argument("--restart", action="store_true",
                           help="move an INTERRUPTED run to _failed/ and train it again from scratch")
    p = sub.add_parser("status")
    p.add_argument("--data", default="human_v1", choices=sorted(DATASETS))
    p = sub.add_parser("dryrun")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=sorted(MODELS))
    a = ap.parse_args(argv)
    use_dataset(getattr(a, "data", "human_v1"))
    if a.cmd == "train":
        return train(a.model, a.seed, restart=a.restart, root=RUN_ROOT)
    if a.cmd == "eval":
        return evaluate(a.model, a.seed, root=RUN_ROOT)
    if a.cmd == "status":
        return status(root=RUN_ROOT)
    return dryrun(a.models)


if __name__ == "__main__":
    sys.exit(main())
