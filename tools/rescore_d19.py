"""Exact D19 status of the Exp 1/2 episodes that welded but did not score a grasp.

    python tools/rescore_d19.py                 # every run with an eval folder

success.py scores a grasp only if, at the engage tick, the box is still on the
platform AND within 50 mm of its spawn (D19). The Exp JSONLs written before
2026-10-24 record grasp=False but not WHICH condition failed, and the box pose at
the engage tick is not in them. The pre-weld push diagnostic is NOT a substitute:
a box can be pushed past 50 mm and come back before the weld (367 episodes that
scored grasp=True did exactly that).

So each such episode is re-run through tools/deploy.py, which since 2026-10-24
writes success.py's own `grasp_detail`. Closed-loop deployment is deterministic,
and the re-run must reproduce the stored trajectory BIT-FOR-BIT
(`trajectory_sha256`); any episode that does not is reported and NOT used.

Writes runs/scripted_v3/eval_d19/<run>/<exp>.jsonl (deploy's per-episode rows),
restart-safe through tools/eval_scripted.deploy (.part files, line counts). ONE
deploy at a time; refuses to start while a train_bc.py or deploy.py runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import eval_scripted as E                                           # noqa: E402


def welded_no_grasp(rows):
    return [r for r in rows if r["weld_engage_tick"] >= 0 and not r["grasp"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=os.path.join(E.ROOT, "runs", "scripted_v3"))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--lock-rule", default="v3")
    a = ap.parse_args(argv)
    busy = E._heavy_others(re.compile(r"train_bc\.py|deploy\.py"))
    if busy:
        raise SystemExit("refusing: another heavy process is running:\n  " + "\n  ".join(busy))
    eval_root = os.path.join(a.run_root, "eval")
    out_root = os.path.join(a.run_root, "eval_d19")
    runs = a.runs or sorted(d for d in os.listdir(eval_root)
                            if os.path.exists(os.path.join(eval_root, d, "selection.json")))
    bad = []
    for run in runs:
        sel = json.load(open(os.path.join(eval_root, run, "selection.json"), encoding="utf-8"))
        for tag, _ in E.EXP:
            src = os.path.join(eval_root, run, "%s.jsonl" % tag)
            if not os.path.exists(src):
                continue
            want = welded_no_grasp(E._lines(src))
            if not want:
                continue
            seeds = [r["seed"] for r in want]
            rows = E.deploy(os.path.join(a.run_root, run), sel["chosen"],
                            ["--seed-list", ",".join(str(s) for s in seeds)], len(seeds),
                            os.path.join(out_root, run, "%s.jsonl" % tag), a.lock_rule,
                            os.path.join(out_root, run, "logs"), allow_mismatch=True)
            old = {r["seed"]: r for r in want}
            for r in rows:
                if r["trajectory_sha256"] != old[r["seed"]]["trajectory_sha256"]:
                    bad.append((run, tag, r["seed"]))
            E.log("%-20s %s  %3d re-run, %d not bit-identical"
                  % (run, tag, len(rows), sum(1 for b in bad if b[:2] == (run, tag))))
    if bad:
        print("NOT BIT-IDENTICAL (excluded from the D19 count): %s" % bad)
        return 1
    print("all re-runs reproduce their stored trajectories bit-for-bit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
