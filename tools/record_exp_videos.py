"""Documentation videos of the scripted Exp 1 / Exp 2 episodes, by bit-identical re-run.

    python tools/record_exp_videos.py                       # Run A + Run B, default paths
    python tools/record_exp_videos.py --runs act_n40_seed0 --failures-per 2

For every run in <eval-root> named <bc|act|act_lstm>_<size>_seed<k> (the BC
controls bc_b256_* are not included unless named in --runs) and each experiment:

  * EVERY successful episode, and
  * --failures-per (default 2) TYPICAL failures: one from each of the two most
    common failure buckets (deploy.py `fail_reason`), or two from the only one;
    within a bucket, the episode(s) whose final box-goal distance is nearest the
    bucket's median (ties: lower seed),

are re-run through tools/deploy.py on the run's selected checkpoint with
--video-dir, under the evaluation's lock rule. Closed-loop deployment is
deterministic and rendering only reads `data` (tools/episode_view.py), so each
re-run must reproduce the stored episode BIT-FOR-BIT: its trajectory_sha256 is
compared with the eval record and the result goes into the index. A mismatch is
reported (exit code 1) and marked, never hidden.

Output: <video-root>/<run>/<exp>/<run>_<ckpt>_seedNNNN[_FAIL].mp4, the re-run's
deploy rows in episodes.jsonl beside them, and <video-root>/index.csv (one row
per video: run, exp, seed, role, outcome, failure bucket, placement error, hash
check, path) - rebuilt from every episodes.jsonl present, so a restart is safe
(finished folders are skipped through tools/eval_scripted.deploy). ONE deploy at
a time; refuses to start while a train_bc.py or deploy.py runs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import eval_scripted as E                                           # noqa: E402

RUN_A_B = re.compile(r"^(act_lstm|act|bc)_(n40|160)_seed\d+$")
INDEX_COLS = ["run", "exp", "seed", "role", "success", "fail_reason", "phase_reached",
              "placement_error_m", "ticks", "ckpt", "sha_stored", "sha_rerun", "hash_match",
              "video"]


def typical_failures(rows, k: int = 2):
    """[(row, role)] - see the module docstring."""
    fails = [r for r in rows if not r["success"]]
    if not fails or k <= 0:
        return []
    count = Counter(r.get("fail_reason") or "?" for r in fails)
    buckets = sorted(count, key=lambda b: (-count[b], b))
    alloc = {b: 1 for b in buckets[:k]}                 # one per top bucket ...
    alloc[buckets[0]] += k - len(alloc)                 # ... the rest from the commonest
    out = []
    for b, n in alloc.items():
        grp = [r for r in fails if (r.get("fail_reason") or "?") == b]
        d = sorted(float(r.get("final_box_goal_m", r["placement_error_m"])) for r in grp)
        med = d[len(d) // 2] if len(d) % 2 else 0.5 * (d[len(d) // 2 - 1] + d[len(d) // 2])
        grp.sort(key=lambda r: (abs(float(r.get("final_box_goal_m", r["placement_error_m"])) - med),
                                r["seed"]))
        out += [(r, "failure-%s (%d/%d failures)" % (b, count[b], len(fails))) for r in grp[:n]]
    return out


def write_index(video_root: str, eval_root: str) -> list:
    """Rebuild index.csv from every <run>/<exp>/episodes.jsonl present."""
    rows = []
    for p in sorted(glob.glob(os.path.join(video_root, "*", "*", "episodes.jsonl"))):
        exp_dir = os.path.dirname(p)
        run, exp = os.path.basename(os.path.dirname(exp_dir)), os.path.basename(exp_dir)
        stored = {r["seed"]: r for r in E._lines(os.path.join(eval_root, run, "%s.jsonl" % exp))}
        roles = json.load(open(os.path.join(exp_dir, "roles.json"), encoding="utf-8"))
        for r in E._lines(p):
            s = stored[r["seed"]]
            rows.append(dict(run=run, exp=exp, seed=r["seed"], role=roles[str(r["seed"])],
                             success=r["success"], fail_reason=r["fail_reason"],
                             phase_reached=r["phase_reached"],
                             placement_error_m=round(r["placement_error_m"], 4), ticks=r["ticks"],
                             ckpt=r["header"]["ckpt"]["file"], sha_stored=s["trajectory_sha256"],
                             sha_rerun=r["trajectory_sha256"],
                             hash_match=s["trajectory_sha256"] == r["trajectory_sha256"],
                             video=r.get("video", "")))
    with open(os.path.join(video_root, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=os.path.join(E.ROOT, "runs", "scripted_v3"))
    ap.add_argument("--eval-root", default=None, help="default <run-root>/eval")
    ap.add_argument("--video-root", default=r"D:\thesis_videos\scripted_exp")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--failures-per", type=int, default=2)
    ap.add_argument("--lock-rule", default="v3")
    a = ap.parse_args(argv)
    eval_root = a.eval_root or os.path.join(a.run_root, "eval")
    busy = E._heavy_others(re.compile(r"train_bc\.py|deploy\.py"))
    if busy:
        raise SystemExit("refusing: another heavy process is running:\n  " + "\n  ".join(busy))
    runs = a.runs or sorted(d for d in os.listdir(eval_root)
                            if RUN_A_B.match(d) and os.path.exists(os.path.join(eval_root, d, "selection.json")))
    for run in runs:
        sel = json.load(open(os.path.join(eval_root, run, "selection.json"), encoding="utf-8"))
        for tag, _ in E.EXP:
            src = os.path.join(eval_root, run, "%s.jsonl" % tag)
            if not os.path.exists(src):
                continue
            rows = E._lines(src)
            pick = [(r, "success") for r in rows if r["success"]] + typical_failures(rows, a.failures_per)
            if not pick:
                continue
            out_dir = os.path.join(a.video_root, run, tag)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "roles.json"), "w", encoding="utf-8") as fh:
                json.dump({str(r["seed"]): role for r, role in pick}, fh, indent=1)
            seeds = [r["seed"] for r, _ in pick]
            E.deploy(os.path.join(a.run_root, run), sel["chosen"],
                     ["--seed-list", ",".join(str(s) for s in seeds), "--video-dir", out_dir],
                     len(seeds), os.path.join(out_dir, "episodes.jsonl"), a.lock_rule,
                     os.path.join(out_dir, "logs"), allow_mismatch=True)
            idx = write_index(a.video_root, eval_root)
            here = [x for x in idx if x["run"] == run and x["exp"] == tag]
            E.log("%-20s %s  %d videos (%d successes), %d hash mismatches"
                  % (run, tag, len(here), sum(x["success"] for x in here),
                     sum(not x["hash_match"] for x in here)))
    idx = write_index(a.video_root, eval_root)
    bad = [x for x in idx if not x["hash_match"]]
    print("%d videos, %d hash mismatches -> %s" % (len(idx), len(bad),
                                                   os.path.join(a.video_root, "index.csv")))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
