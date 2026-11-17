"""Compare a new tools/deploy.py JSONL with the recorded final Exp 1 / Exp 2 episodes.

    python tools/compare_to_final.py runs/try_eval/act_lstm_full_seed0.jsonl

For each episode in the new file it finds the same run and seed in
runs/scripted_v3*/eval/<run>/exp1.jsonl or exp2.jsonl (from the
analysis_scripted_v3 release asset) and prints both outcomes side by side, plus
whether the state trajectory hash is identical.

Read the hash column with CLAUDE.md O34 in mind: the simulation is not bit-for-bit
reproducible across machines, even with the pinned versions (26/40 scripted seeds
agreed to <= 0.0003 mm on a second laptop, 14 diverged by up to 10.6 mm). A
different hash on another machine is expected; the outcome columns are what to
compare. Every reported result came from one machine (section 8, 2026-09-28).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def recorded(run: str):
    out = {}
    for exp in ("exp1", "exp2"):
        for f in glob.glob(os.path.join(ROOT, "runs", "scripted_v3*", "eval", run, exp + ".jsonl")):
            for d in _load(f):
                out[int(d["seed"])] = (exp, d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="output of tools/deploy.py --out")
    a = ap.parse_args(argv)
    new = _load(a.jsonl)
    if not new:
        raise SystemExit("%s has no episodes" % a.jsonl)
    run, ckpt = new[0]["policy"].split("/", 1)
    rec = recorded(run)
    if not rec:
        raise SystemExit("no recorded Exp 1/2 episodes for %s - fetch analysis_scripted_v3 first" % run)
    print("run %s, checkpoint %s" % (run, ckpt))
    print("%5s %4s | %-9s %-7s %9s | %-9s %-7s %9s | %s"
          % ("seed", "exp", "new", "phase", "place_err", "recorded", "phase", "place_err", "same trajectory"))
    agree = same = 0
    for d in new:
        s = int(d["seed"])
        if s not in rec:
            print("%5d  --  no recorded episode for this seed" % s)
            continue
        exp, r = rec[s]
        if r["policy"] != d["policy"]:
            print("%5d  recorded with %s, not %s" % (s, r["policy"], d["policy"]))
            continue
        ok = lambda x: "SUCCESS" if x["success"] else "fail"
        h = d["trajectory_sha256"] == r["trajectory_sha256"]
        agree += d["success"] == r["success"]
        same += h
        print("%5d %4s | %-9s %-7s %8.3fm | %-9s %-7s %8.3fm | %s"
              % (s, exp, ok(d), d.get("fail_reason") or "-", d["placement_error_m"],
                 ok(r), r.get("fail_reason") or "-", r["placement_error_m"], "yes" if h else "no"))
    print("outcome agrees on %d/%d, identical trajectory on %d/%d" % (agree, len(new), same, len(new)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
