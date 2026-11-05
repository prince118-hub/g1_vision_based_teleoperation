"""Choose the box positions (seeds) worth filming, from the eval JSONLs alone.

    python tools/pick_video_episodes.py --results runs/scripted_v3 \
        --runs bc_160_seed0 act_160_seed0 act_lstm_160_seed0 --out picks/scripted_160.json
    python tools/pick_video_episodes.py --results runs/scripted_v3 runs/scripted_v3_memory_design \
        --runs bc_b256_n40_seed0 act_orig_n40_seed0 act_lstm_full_n40_seed0 --out picks/d8_n40.json

READ-ONLY. Input: <results>/eval/<run>/exp1.jsonl and exp2.jsonl, the
tools/deploy.py records that tools/eval_scripted.py writes. A results folder can
hold scripted-trained or teleop-trained runs; nothing here depends on which.
Nothing is re-run. No MuJoCo or torch needed. Evaluation seeds are FIXED
across policies (CLAUDE.md section 8 2026-09-11), so one seed is one box
position for every run. Only seeds every listed run was evaluated on are used,
and a difference between the runs' seed sets is reported.

CATEGORIES (each at most --per-category picks, per experiment or per pair as stated):
  contrast          BETTER succeeds and WORSE fails on the same seed. Per pair and
                    experiment, the picks go round-robin over WORSE's failure buckets
                    (commonest first), so the picks show different ways of failing.
                    Within a bucket: BETTER's placement error, then the seed. --contrast
                    BETTER:WORSE (repeatable). The default pairs each ACT-LSTM run with
                    the ACT run of the same size and training seed: act_lstm vs act
                    (W_o 12), act_lstm_full vs act_orig.
  contrast_reverse  the same pair the other way round, WORSE succeeds and BETTER fails.
                    The counterexamples are picked too, and both counts go into `why`.
  typical_failure   per run, experiment and failure bucket (deploy.py fail_reason): the
                    episode whose final box-to-goal distance is nearest the bucket's
                    median, ties to the lower seed. This is tools/record_exp_videos.py's
                    rule applied to every bucket (tested equal). Not capped.
  heldout_spread    Exp 2 positions spread over the held-out patch: farthest-point
                    sampling on the box spawn, starting at the seed nearest the spawns'
                    centroid.
  heldout_success   Exp 2 positions solved by the most runs. Ties go to the lower mean
                    placement error of the successes, then the seed. At least 1 success.
  all_succeed       per experiment, seeds every run solves (fewest ticks of the slowest
                    run first). A fair "all of them can do it" comparison.

OUTPUT: --out picks.json, which holds the results folders, runs in order, rules,
counts and picks. Each pick has category, exp, seed, box_xy, heldout, why, and
every run's outcome. picks.csv sits beside it, and the suggested
tools/video_side_by_side.py commands are printed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import video_common as VC                                           # noqa: E402

CATEGORIES = ("contrast", "contrast_reverse", "typical_failure", "heldout_spread",
              "heldout_success", "all_succeed")
#: better model -> the model it is contrasted with (same size and training seed)
AUTO_PAIRS = {"act_lstm": "act", "act_lstm_full": "act_orig"}


def _dist(r: dict) -> float:
    v = r.get("final_box_goal_m", r.get("placement_error_m"))
    return float(v) if v is not None else float("inf")


def _pe(r: dict) -> float:
    v = r.get("placement_error_m")
    return float(v) if v is not None else float("inf")


def outcome(r: dict) -> dict:
    return dict(success=bool(r["success"]), fail_reason=r.get("fail_reason") or "",
                ticks=r.get("ticks"), placement_error_m=r.get("placement_error_m"),
                final_box_goal_m=r.get("final_box_goal_m"))


# ─── loading ─────────────────────────────────────────────────────────────────
def load(results: Sequence[VC.Results], runs: Optional[Sequence[str]]):
    """({run: {exp: {seed: row}}}, [warnings])"""
    where = VC.find_runs(results)
    names = list(runs) if runs else sorted(where, key=lambda n: (VC.model_rank(VC.model_key(n)), n))
    missing = [n for n in names if n not in where]
    if missing:
        raise SystemExit("not found under the results folders: %s (found: %s)"
                         % (", ".join(missing), ", ".join(sorted(where)) or "none"))
    data = {n: {e: VC.load_exp(where[n], n, e) for e in VC.EXPS} for n in names}
    warn = []
    for e in VC.EXPS:
        sets = {n: set(data[n][e]) for n in names if data[n][e]}
        if sets and len({frozenset(s) for s in sets.values()}) > 1:
            common = set.intersection(*sets.values())
            warn.append("%s: the runs were evaluated on different seed sets (%s); using the %d "
                        "common seeds" % (e, ", ".join("%s %d" % (n, len(s)) for n, s in sets.items()),
                                          len(common)))
    return data, warn


def common_seeds(data: dict, exp: str, runs: Sequence[str]) -> List[int]:
    sets = [set(data[n][exp]) for n in runs]
    if not sets or any(not s for s in sets):
        return []
    return sorted(set.intersection(*sets))


def auto_pairs(runs: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for b in runs:
        pb = VC.parse_run(b)
        want = AUTO_PAIRS.get(pb["model"])
        for w in runs:
            pw = VC.parse_run(w)
            if want and pw["model"] == want and pw["size"] == pb["size"] \
                    and pw["train_seed"] == pb["train_seed"]:
                out.append((b, w))
    return out


# ─── the rules ───────────────────────────────────────────────────────────────
def nearest_median(rows: Sequence[dict]) -> dict:
    """The row whose final box-goal distance is nearest the median (ties: lower
    seed) - tools/record_exp_videos.typical_failures' rule within one bucket."""
    d = sorted(_dist(r) for r in rows)
    med = d[len(d) // 2] if len(d) % 2 else 0.5 * (d[len(d) // 2 - 1] + d[len(d) // 2])
    return min(rows, key=lambda r: (abs(_dist(r) - med), r["seed"]))


def typical_failures(rows: Sequence[dict]) -> List[Tuple[dict, str, int, int]]:
    """[(row, bucket, bucket count, failures)] one per failure bucket, commonest first."""
    fails = [r for r in rows if not r["success"]]
    count = Counter(r.get("fail_reason") or "?" for r in fails)
    out = []
    for b in sorted(count, key=lambda b: (-count[b], b)):
        grp = [r for r in fails if (r.get("fail_reason") or "?") == b]
        out.append((nearest_median(grp), b, count[b], len(fails)))
    return out


def contrast(data: dict, exp: str, seeds: Sequence[int], better: str, worse: str, n: int):
    """(wins picked, reverse picked, n wins, n reverse) - see the module docstring."""
    B, Wr = data[better][exp], data[worse][exp]
    wins = [s for s in seeds if B[s]["success"] and not Wr[s]["success"]]
    rev = [s for s in seeds if Wr[s]["success"] and not B[s]["success"]]

    def pick(cands, winner, loser):
        by = {}
        for s in cands:
            by.setdefault(loser[s].get("fail_reason") or "?", []).append(s)
        order = sorted(by, key=lambda b: (-len(by[b]), b))
        for b in order:
            by[b].sort(key=lambda s: (_pe(winner[s]), s))
        out = []
        while len(out) < n and any(by[b] for b in order):
            for b in order:
                if by[b] and len(out) < n:
                    out.append(by[b].pop(0))
        return out
    return pick(wins, B, Wr), pick(rev, Wr, B), len(wins), len(rev)


def farthest_point(xy: np.ndarray, n: int) -> List[int]:
    """Indices of n points spread over xy: start nearest the centroid, then
    each next point the farthest from those already chosen (ties: lower index)."""
    if len(xy) == 0 or n <= 0:
        return []
    first = int(np.argmin(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))
    chosen = [first]
    d = np.linalg.norm(xy - xy[first], axis=1)
    d[first] = -1.0
    while len(chosen) < min(n, len(xy)):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(xy - xy[k], axis=1))
        d[chosen] = -1.0                       # never the same point twice (duplicate spawns)
    return chosen


# ─── picking ─────────────────────────────────────────────────────────────────
def pick_all(data: dict, runs: Sequence[str], pairs: Sequence[Tuple[str, str]],
             per_category: int, labels: Optional[Dict[str, str]] = None) -> dict:
    labels = labels or {}

    def lab(n):
        kind = next((r.get("kind") for e in VC.EXPS for r in data[n][e].values()), None)
        return VC.run_label(n, kind, labels.get(n))

    def base(cat, exp, seed, why, extra=None):
        any_row = data[runs[0]][exp][seed]
        p = dict(category=cat, exp=exp, seed=int(seed), box_xy=any_row.get("box_xy"),
                 heldout=any_row.get("heldout"), why=why,
                 outcomes={n: outcome(data[n][exp][seed]) for n in runs})
        p.update(extra or {})
        return p

    picks, counts = [], {c: {} for c in CATEGORIES}
    for exp in VC.EXPS:
        seeds = common_seeds(data, exp, runs)
        if not seeds:
            continue
        el = VC.EXP_LABEL[exp].split(" (")[0]
        for b, w in pairs:
            won, lost, nw, nl = contrast(data, exp, seeds, b, w, per_category)
            key = "%s>%s" % (b, w)
            counts["contrast"].setdefault(exp, {})[key] = nw
            counts["contrast_reverse"].setdefault(exp, {})[key] = nl
            tail = "%s: %d of %d seeds this way, %d the other way" % (el, nw, len(seeds), nl)
            for s in won:
                picks.append(base("contrast", exp, s, "%s succeeds, %s fails (%s); %s" % (
                    lab(b), lab(w), data[w][exp][s].get("fail_reason") or "?", tail),
                    dict(pair=[b, w])))
            tail_r = "%s: %d of %d seeds this way, %d the other way" % (el, nl, len(seeds), nw)
            for s in lost:
                picks.append(base("contrast_reverse", exp, s, "%s succeeds, %s fails (%s); %s" % (
                    lab(w), lab(b), data[b][exp][s].get("fail_reason") or "?", tail_r),
                    dict(pair=[b, w])))
        for n in runs:
            rows = [data[n][exp][s] for s in seeds]
            tf = typical_failures(rows)
            counts["typical_failure"].setdefault(exp, {})[n] = {b: c for _, b, c, _ in tf}
            for r, bucket, c, nf in tf:
                picks.append(base("typical_failure", exp, r["seed"],
                                  "typical %s failure of %s: %d of its %d %s failures; final "
                                  "box-goal distance %.3f m, nearest the bucket median"
                                  % (bucket, lab(n), c, nf, el, _dist(r)),
                                  dict(run=n, bucket=bucket)))
        ok_all = [s for s in seeds if all(data[n][exp][s]["success"] for n in runs)]
        counts["all_succeed"][exp] = len(ok_all)
        ok_all.sort(key=lambda s: (max(int(data[n][exp][s].get("ticks") or 0) for n in runs), s))
        for s in ok_all[:per_category]:
            picks.append(base("all_succeed", exp, s, "all %d runs succeed (%d such %s seeds)"
                              % (len(runs), counts["all_succeed"][exp], el)))
        if exp != "exp2":
            continue
        held = [s for s in seeds if data[runs[0]][exp][s].get("heldout")]
        xy = np.array([data[runs[0]][exp][s]["box_xy"] for s in held], dtype=float).reshape(-1, 2)
        counts["heldout_spread"][exp] = len(held)
        for i, k in enumerate(farthest_point(xy, per_category)):
            s = held[k]
            ns = sum(data[n][exp][s]["success"] for n in runs)
            picks.append(base("heldout_spread", exp, s,
                              "held-out position %d of %d spread over the patch; %d/%d runs succeed"
                              % (i + 1, min(per_category, len(held)), ns, len(runs))))

        def solved(s):
            return [n for n in runs if data[n][exp][s]["success"]]
        cands = [s for s in held if solved(s)]
        counts["heldout_success"][exp] = len(cands)
        cands.sort(key=lambda s: (-len(solved(s)),
                                  float(np.mean([_pe(data[n][exp][s]) for n in solved(s)])), s))
        for s in cands[:per_category]:
            picks.append(base("heldout_success", exp, s, "held-out position solved by %d/%d runs (%s)"
                              % (len(solved(s)), len(runs), ", ".join(lab(n) for n in solved(s)))))
    return dict(picks=picks, counts=counts, labels={n: lab(n) for n in runs})


def write_csv(path: str, picks: Sequence[dict], runs: Sequence[str]) -> None:
    cols = ["category", "exp", "seed", "box_x", "box_y", "heldout", "why"] + list(runs)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in picks:
            row = dict(category=p["category"], exp=p["exp"], seed=p["seed"],
                       box_x=(p["box_xy"] or [None])[0], box_y=(p["box_xy"] or [None, None])[1],
                       heldout=p["heldout"], why=p["why"])
            for n in runs:
                o = p["outcomes"][n]
                row[n] = "SUCCESS" if o["success"] else "FAIL:%s" % o["fail_reason"]
            w.writerow(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True,
                    help="results folder(s) (eval_scripted --run-root) or eval folder(s)")
    ap.add_argument("--runs", nargs="+", default=None, metavar="RUN[=LABEL]",
                    help="runs to compare, in panel order (default: every run found)")
    ap.add_argument("--contrast", action="append", default=None, metavar="BETTER:WORSE",
                    help="contrast pair (repeatable); default: ACT-LSTM vs ACT, same size+seed")
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--out", required=True, help="picks JSON (a CSV is written beside it)")
    a = ap.parse_args(argv)
    specs = [VC.parse_run_spec(s) for s in (a.runs or [])]
    labels = {n: lb for n, lb in specs if lb}
    res = [VC.open_results(r) for r in a.results]
    data, warn = load(res, [n for n, _ in specs] or None)
    runs = list(data)
    if a.contrast:
        pairs = []
        for c in a.contrast:
            b, _, w = c.partition(":")
            if b not in data or w not in data:
                raise SystemExit("--contrast %s: both runs must be in --runs" % c)
            pairs.append((b, w))
    else:
        pairs = auto_pairs(runs)
        if not pairs:
            warn.append("no ACT-LSTM / ACT pair of the same size and seed among the runs - no "
                        "contrast picks (name one with --contrast BETTER:WORSE)")
    out = pick_all(data, runs, pairs, a.per_category, labels)
    doc = dict(results=[os.path.abspath(r) for r in a.results], runs=runs, labels=out["labels"],
               pairs=[list(p) for p in pairs], per_category=a.per_category,
               rules={c: d for c, d in zip(CATEGORIES, (
                   "BETTER succeeds and WORSE fails; round-robin over WORSE's failure buckets",
                   "WORSE succeeds and BETTER fails (counterexamples)",
                   "per run x exp x bucket: final box-goal distance nearest the bucket median",
                   "Exp 2: farthest-point spread over the box spawns, from the centroid",
                   "Exp 2: most runs succeeding, then lower mean placement error",
                   "every run succeeds; fewest ticks of the slowest run first"))},
               counts=out["counts"], warnings=warn, picks=out["picks"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    csv_path = os.path.splitext(a.out)[0] + ".csv"
    write_csv(csv_path, out["picks"], runs)
    for w_ in warn:
        print("WARNING " + w_)
    by = Counter((p["category"], p["exp"]) for p in out["picks"])
    for (c, e), k in sorted(by.items()):
        print("%-17s %s  %2d pick(s)" % (c, e, k))
    for p in out["picks"]:
        print("  %-17s %s seed %5d  %s" % (p["category"], p["exp"], p["seed"], p["why"]))
    print("\n-> %s, %s" % (VC.rel(a.out), VC.rel(csv_path)))
    print("render, e.g.:\n  python tools/video_side_by_side.py --picks %s --category contrast "
          "heldout_success --out-dir <video folder>" % VC.rel(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
