"""Evaluation episode cap for a dataset folder: ceil(1.3 x the longest ACCEPTED demonstration).

    python tools/eval_cap.py recordings/demonstrations
    python tools/eval_cap.py recordings/demonstrations --ticks          # prints only N
    python tools/eval_cap.py recordings/demonstrations --json cap.json

The rule, what counts as accepted and why scripted evaluation keeps 900:
g1_data/eval_cap.py. To evaluate under it, give the FOLDER (the cap is then
recomputed and its provenance recorded in every result):

    python tools/deploy.py --run <run> --seeds data/eval/exp1_seeds.json \\
        --max-ticks-from recordings/demonstrations --out exp1.jsonl
    python tools/eval_scripted.py --run-root <root> --runs <run> \\
        --max-ticks-from recordings/demonstrations

Read only: episode metadata and the ledger; no torch, no simulation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_data import eval_cap as EC                                  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="dataset folder (accepted episodes at its top level)")
    ap.add_argument("--ticks", action="store_true", help="print only the cap, in ticks")
    ap.add_argument("--json", default=None, help="write the full cap record here")
    a = ap.parse_args(argv)
    try:
        c = EC.eval_cap(a.folder)
    except EC.CapError as e:
        raise SystemExit("eval_cap: %s" % e)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=1)
    if a.ticks:
        print(c["max_ticks"])
    else:
        print("\n".join(EC.report_lines(c)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
