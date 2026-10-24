"""Offline correction of the Exp / selection JSONLs written before the M1 fix.

    python tools/fix_walkin_attribution.py [--eval-root runs/scripted_v3/eval]

M1 (review, 2026-10-24): tools/deploy.py scored each episode BEFORE deriving its
phase labels, so success.attribute never saw them and could not charge a walk-in
failure to WALK_IN - every never-grasped episode was charged to GRASP. Fixed in
deploy.py; this rewrites the stored rows to what the fixed deploy writes:

    success False, fail_reason "GRASP", weld_engage_tick < 0 (never welded),
    phase_reached "SETTLE" (the last label)                  -> fail_reason "WALK_IN"

That is success.attribute's own rule (labels[-1] == SETTLE with no engage); the
re-run check (2026-10-24, act_160_seed0 Exp 1 seeds 240-242) reproduced the
stored trajectories bit-for-bit and changed nothing but this field. Corrected
rows carry `fail_reason_original` and `corrections: ["M1"]`. Nothing else in a
row changes: the four criteria never read the labels.

Also (same review): selection.json `progress` of an EPOCH-budgeted run (BC) mixed
steps (periodic checkpoints) with epochs (best.pt). Rewritten in epochs, and the
choice re-derived to confirm it is unchanged - the tie-break never decided a
selection (every chosen checkpoint was the unique best before it).

ORIGINALS ARE KEPT: on the first run every file it touches is copied to
<eval-root>_pre_m1/ (same relative paths), and every later run reads from there,
so the correction is idempotent and never compounds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import eval_scripted as E                                           # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402


def correct_row(r: dict) -> dict:
    if (not r["success"] and r.get("fail_reason") == "GRASP"
            and r["weld_engage_tick"] < 0 and r.get("phase_reached") == "SETTLE"):
        r = dict(r, fail_reason="WALK_IN", fail_reason_original="GRASP",
                 corrections=list(r.get("corrections", [])) + ["M1"])
    return r


def _backup(path: str, root: str, backup_root: str) -> str:
    dst = os.path.join(backup_root, repo_relpath(path, root))
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(path, dst)
    return dst


def _write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def fix_jsonl(path, root, backup_root):
    src = _backup(path, root, backup_root)
    rows = [correct_row(r) for r in E._lines(src)]
    _write(path, "".join(json.dumps(r) + "\n" for r in rows))
    return sum("M1" in r.get("corrections", []) for r in rows), len(rows)


def fix_selection(path, root, backup_root):
    src = _backup(path, root, backup_root)
    sel = json.load(open(src, encoding="utf-8"))
    run_dir = os.path.join(E.ROOT, sel["run"])
    if not E.epoch_budgeted(run_dir):
        return False
    for c in sel["candidates"]:
        if not c.get("alias_of"):
            c["progress"] = float(c["ckpt_info"]["epoch"])
            c["progress_unit"] = "epoch"
    real = [c for c in sel["candidates"] if not c.get("alias_of")]
    key = lambda c: ((c["success"],) + tuple(c[s] for s in E.STAGES)
                     + (-c["mean_placement_error_m"], -c["progress"]))
    if max(real, key=key)["ckpt"] != sel["chosen"]:
        raise SystemExit("%s: the epoch tie-break changes the choice - re-run its Exp 1/2" % path)
    sel["rule"] = E.SELECTION_RULE
    sel["corrections"] = list(sel.get("corrections", [])) + [
        "2026-10-24: progress rewritten in epochs (was steps for periodic checkpoints, "
        "epochs for best.pt); choice re-derived, unchanged"]
    _write(path, json.dumps(sel, indent=1))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", default=os.path.join(E.ROOT, "runs", "scripted_v3", "eval"))
    a = ap.parse_args(argv)
    root = os.path.normpath(a.eval_root)
    backup_root = root + "_pre_m1"
    tot = n = 0
    for p in sorted(glob.glob(os.path.join(root, "*", "exp*.jsonl"))
                    + glob.glob(os.path.join(root, "*", "select", "*.jsonl"))):
        k, m = fix_jsonl(p, root, backup_root)
        tot, n = tot + k, n + m
        if k:
            print("%-60s %4d / %d -> WALK_IN" % (repo_relpath(p, root), k, m))
    sels = [p for p in sorted(glob.glob(os.path.join(root, "*", "selection.json")))
            if fix_selection(p, root, backup_root)]
    print("M1: %d of %d rows re-attributed GRASP -> WALK_IN; originals in %s"
          % (tot, n, repo_relpath(backup_root)))
    print("selection.json progress in epochs: %s" % [os.path.basename(os.path.dirname(p)) for p in sels])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
