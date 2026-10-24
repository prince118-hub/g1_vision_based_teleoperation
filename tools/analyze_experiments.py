"""Experiment 1 / 2 analysis: tables, statistics and figures from tools/eval_scripted.py output.

    python tools/analyze_experiments.py                                # runs/scripted_v3/eval
    python tools/analyze_experiments.py --eval-root runs/scripted_v3/eval --out reports/exp12
    python tools/analyze_experiments.py --eval-root A/eval B/eval      # several eval folders

READ-ONLY on the evaluation. Input is the eval folder(s) eval_scripted.py writes,
<eval>/<run>/exp1.jsonl and exp2.jsonl (one tools/deploy.py record per episode)
plus <eval>/<run>/selection.json when present; run folders are named
<model>_<size>_seed<k> (bc_160_seed0, act_lstm_n40_seed2, ...). Nothing is
re-run, nothing in the eval folder is written. Needs numpy, scipy, pandas,
matplotlib - NOT mujoco or torch (the box-spawn sampler is mirrored below and
checked against g1_teleop/box_reset.py by the tests where mujoco exists).

OUTPUT: one folder (--out) with
  report.md                        everything below, labelled (model, data size,
                                   seeds, lock rule, checkpoint)
  runs.csv                         one row per run found, with what is missing
  episodes.csv                     every episode record used (flattened)
  success_by_run.csv               model x size x seed x experiment, Wilson 95% CI
  success_summary.csv              mean +- SD over seeds, pooled Wilson 95% CI
  generalization_gap.csv           Exp 1 - Exp 2 per model x size, with SEs
  stage_rates.csv                  grasp / lift / transport / place / success
  failure_phase.csv                failure bucket counts (deploy.py fail_reason)
  fisher.csv                       pairwise Fisher exact on successes, pooled
                                   over seeds, per experiment and data size
  distance_tests.csv               Kruskal-Wallis + pairwise Mann-Whitney U (Holm)
                                   on grasp proximity and final box-to-goal distance
  thresholds.csv                   final / closest box-to-goal within 5 / 10 cm
  size_comparison.csv              n40 vs 160 when both exist
  fig_*.png                        success bars, stage bars, failure stacks,
                                   proximity boxplots, position heatmaps,
                                   training-demo coverage, n40 vs 160

STATISTICAL CHOICES (stated in the report too)
  * Wilson score interval, 95%, per run and pooled over seeds.
  * Mean +- SD over training seeds uses the per-seed rates (SD ddof=1; blank
    with one seed). Pooling for the CI and the tests treats episodes from all
    seeds as one binomial sample - it ignores between-seed variance, which the
    SD column shows.
  * Generalization gap = Exp 1 rate - Exp 2 rate. SE_binomial from the pooled
    rates, sqrt(p1(1-p1)/n1 + p2(1-p2)/n2) (the two seed sets are disjoint, so
    the samples are independent); SE_seeds = SD of the per-seed gaps / sqrt(k)
    when k >= 2 seeds.
  * Fisher's exact test, two-sided, on the pooled 2x2 success table for each
    model pair, within one experiment and one data size; Holm-adjusted over
    the three pairs of that family.
  * Kruskal-Wallis over the models, then pairwise two-sided Mann-Whitney U,
    Holm-adjusted over the pairs, per experiment, data size and metric.
    Grasp proximity = palm_box_min_m (the FARTHER palm's closest approach to
    the box before the weld; the weld gate needs both within 0.16 m); final
    distance = final_box_goal_m (= success.py's placement error). Episodes
    without the field (older deploy records) are dropped and counted.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_paths():
    """g1_data/paths.py by file: importing the g1_data package loads MuJoCo,
    and this tool needs none (same reason BoxConfig is loaded by file)."""
    spec_ = importlib.util.spec_from_file_location(
        "_g1_paths", os.path.join(ROOT, "g1_data", "paths.py"))
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


repo_relpath = _load_paths().repo_relpath
DEFAULT_EVAL = os.path.join(ROOT, "runs", "scripted_v3", "eval")
DEFAULT_SPLITS = (("160", os.path.join(ROOT, "data", "processed", "splits_scripted_v3.json")),
                  ("n40", os.path.join(ROOT, "data", "processed", "splits_scripted_v3_n40.json")))

MODELS = ("bc", "act", "act_lstm")
#: CONTROL runs (review M3, 2026-10-25): BC at its OWN default batch 256 instead
#: of the ladder's batch 8. Reported in the per-model tables and against its base
#: model in `control_comparison`, but never inside the three-model pairwise tests,
#: so those tests and their Holm families are exactly what they were.
CONTROLS = {"bc_b256": "bc"}
MODEL_LABEL = {"bc": "BC", "act": "ACT", "act_lstm": "ACT-LSTM", "bc_b256": "BC (b256)"}
EXPS = ("exp1", "exp2")
EXP_LABEL = {"exp1": "Exp 1 (in-distribution)", "exp2": "Exp 2 (held-out patch)"}
STAGES = ("grasp", "lift", "transport", "place")
FAIL_ORDER = ("WALK_IN", "GRASP", "LIFT", "TRANSPORT", "PLACE", "fell", "other")
RUN_RE = re.compile(r"^(?P<model>act_lstm|act|bc_b256|bc)_(?P<size>n?\d+)_seed(?P<seed>\d+)$")
GRASP_GATE_M = 0.16            # g1_teleop/grasp.py palm radius (deploy.py's docstring)
THRESHOLDS_M = (0.05, 0.10)
Z95 = 1.959963984540054


# ─── the box spawn region (mirrors g1_teleop.box_reset; no mujoco) ───────────
def box_config():
    """g1_teleop/config.py's BoxConfig, loaded by FILE so the g1_teleop package
    (and with it mujoco) is not imported. That file has no third-party imports."""
    spec_ = importlib.util.spec_from_file_location(
        "_g1_teleop_config", os.path.join(ROOT, "g1_teleop", "config.py"))
    mod = importlib.util.module_from_spec(spec_)
    sys.modules.setdefault("_g1_teleop_config", mod)
    spec_.loader.exec_module(mod)
    return mod.BoxConfig()


def spawn_xy(seed: int, cfg=None) -> tuple:
    """box_reset.sample_box_pose's x, y for `seed` (tested equal to it)."""
    cfg = cfg or box_config()
    rng = np.random.default_rng(int(seed))
    (cx, cy), (hx, hy) = cfg.pickup_center, cfg.pickup_half
    return float(rng.uniform(cx - hx, cx + hx)), float(rng.uniform(cy - hy, cy + hy))


def region(cfg=None) -> dict:
    cfg = cfg or box_config()
    (cx, cy), (hx, hy) = cfg.pickup_center, cfg.pickup_half
    return dict(x=(cx - hx, cx + hx), y=(cy - hy, cy + hy),
                heldout_x=tuple(cfg.heldout_x), heldout_y=tuple(cfg.heldout_y))


# ─── statistics ──────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = Z95) -> tuple:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # k = 0 / k = n: the bound is exactly 0 / 1; float rounding must not move it
    # past the point estimate.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the input order (NaN passes through)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.flatnonzero(~np.isnan(p))
    out = np.full(len(p), np.nan)
    m = len(ok)
    run = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok], kind="stable")]):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out.tolist()


def fisher(k1: int, n1: int, k2: int, n2: int) -> tuple:
    """(odds ratio, two-sided p) for successes k1/n1 vs k2/n2."""
    from scipy.stats import fisher_exact
    res = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")
    return float(res[0]), float(res[1])


# ─── loading ─────────────────────────────────────────────────────────────────
def parse_run(name: str) -> dict:
    m = RUN_RE.match(name)
    if not m:
        return dict(model=None, size=None, train_seed=None)
    size = m.group("size")
    return dict(model=m.group("model"), size=size, train_seed=int(m.group("seed")))


def size_n(size: Optional[str]) -> Optional[int]:
    if size is None:
        return None
    return int(size[1:]) if size.startswith("n") else int(size)


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit("%s line %d is not JSON: %s" % (path, i + 1, e))
    return rows


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _rule(v) -> Optional[str]:
    if isinstance(v, dict):
        return v.get("version")
    return v if isinstance(v, str) else None


def find_runs(roots: Sequence[str]) -> List[str]:
    """Run folders: any directory holding exp1.jsonl or exp2.jsonl (a root may
    itself be one). select/ and logs/ subfolders are not runs."""
    out = []
    for r in roots:
        r = os.path.abspath(r)
        cands = [r] + sorted(d for d in glob.glob(os.path.join(r, "*")) if os.path.isdir(d))
        for d in cands:
            if any(os.path.exists(os.path.join(d, "%s.jsonl" % e)) for e in EXPS) and d not in out:
                out.append(d)
    return out


def _grasp_cols(gd, source="eval") -> dict:
    """success.py's grasp_detail (deploy.py writes it since 2026-10-24), flattened.
    None where the record predates it - see attach_grasp_detail."""
    if not gd or "box_near_spawn" not in gd:
        return dict(grasp_near_spawn=None, grasp_box_above_platform=None,
                    box_moved_from_spawn_m=float("nan"), grasp_detail_source=None)
    return dict(grasp_near_spawn=bool(gd["box_near_spawn"]),
                grasp_box_above_platform=bool(gd["box_above_platform"]),
                box_moved_from_spawn_m=_num(gd.get("box_moved_from_spawn_m")),
                grasp_detail_source=source)


def attach_grasp_detail(E, d19_roots: Sequence[str]):
    """Fill grasp_detail for older records from tools/rescore_d19.py re-runs
    (<d19 root>/<run>/<exp>.jsonl). A re-run is used ONLY if it reproduced the
    stored episode bit-for-bit (same trajectory_sha256); otherwise it is counted
    and ignored. Returns (re-runs used, re-runs rejected)."""
    used = rejected = 0
    idx = {(r.run, r.exp, int(r.seed)): i for i, r in E[["run", "exp", "seed"]].iterrows()}
    for root in d19_roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            for exp in EXPS:
                p = os.path.join(d, "%s.jsonl" % exp)
                if not os.path.exists(p):
                    continue
                for r in _read_jsonl(p):
                    i = idx.get((os.path.basename(d), exp, int(r["seed"])))
                    if i is None:
                        continue
                    if r.get("trajectory_sha256") != E.at[i, "trajectory_sha256"]:
                        rejected += 1
                        continue
                    if E.at[i, "grasp_detail_source"] is None:
                        for k, v in _grasp_cols(r.get("grasp_detail"), "rerun").items():
                            E.at[i, k] = v
                        used += 1
    return used, rejected


def d19_sensitivity(E):
    """D19 (grasp = box still within 50 mm of its spawn at the engage tick):
    how many episodes WELDED but scored no grasp, how many of those the push
    rule failed, and how many of the D19-failed episodes still ended with the
    box placed (final box-goal <= 0.10 m), resting, released (the weld let go),
    upright (tilt <= 15 deg) - and all four, i.e. every final-tick place
    condition. Lift/transport are not reconstructable from the record, so
    "all four" is an UPPER BOUND on the successes D19 removed."""
    import pandas as pd
    rows = []
    for (model, size, exp), g in E.groupby(["model", "size", "exp"], dropna=False):
        w = g[(g.weld_engage_tick >= 0) & ~g.grasp]
        known = w[w.grasp_detail_source.notna()]
        d19 = known[known.grasp_near_spawn.astype(object) == False]          # noqa: E712
        placed = d19.placement_error_m <= 0.10
        resting = d19.resting.astype(bool)
        released = d19.weld_release_tick >= 0
        upright = d19.tilt_deg <= 15.0
        on_platform = d19.grasp_box_above_platform.astype(object) == True      # noqa: E712
        not_on_platform_only = ((known.grasp_near_spawn.astype(object) == True)          # noqa: E712
                                & (known.grasp_box_above_platform.astype(object) == False))  # noqa: E712
        rows.append(dict(
            model=model, size=size, size_n=size_n(size), exp=exp, n=len(g),
            welded=int((g.weld_engage_tick >= 0).sum()), grasp=int(g.grasp.sum()),
            welded_no_grasp=len(w), detail_unknown=len(w) - len(known),
            d19=len(d19), d19_only=int(on_platform.sum()),
            not_on_platform_only=int(not_on_platform_only.sum()),
            d19_placed=int(placed.sum()), d19_resting=int(resting.sum()),
            d19_released=int(released.sum()), d19_upright=int(upright.sum()),
            d19_all_place_conditions=int((placed & resting & released & upright).sum()),
            success=int(g.success.sum()),
            median_push_m=float(d19.box_moved_from_spawn_m.median()) if len(d19) else float("nan")))
    return _order(pd.DataFrame(rows))


def load(roots: Sequence[str]):
    """(episodes DataFrame, runs DataFrame). One episode row per JSONL line."""
    import pandas as pd
    eps, runs = [], []
    for d in find_runs(roots):
        name = os.path.basename(d)
        info = parse_run(name)
        sel_path = os.path.join(d, "selection.json")
        sel = json.load(open(sel_path, encoding="utf-8")) if os.path.exists(sel_path) else {}
        run_row = dict(run=name, path=repo_relpath(d),
                       **info, size_n=size_n(info["size"]), chosen_ckpt=sel.get("chosen"),
                       selection_success=(sel.get("chosen_tally") or {}).get("success"),
                       selection_n=(sel.get("chosen_tally") or {}).get("n"))
        for exp in EXPS:
            p = os.path.join(d, "%s.jsonl" % exp)
            if not os.path.exists(p):
                run_row["%s_n" % exp] = 0
                continue
            rows = _read_jsonl(p)
            run_row["%s_n" % exp] = len(rows)
            for r in rows:
                h = r.get("header") or {}
                model = info["model"] or h.get("kind")
                ck = h.get("ckpt") or {}
                box = r.get("box_xy") or [float("nan"), float("nan")]
                eps.append(dict(
                    run=name, model=model, size=info["size"], size_n=size_n(info["size"]),
                    train_seed=info["train_seed"], exp=exp, seed=int(r["seed"]),
                    success=bool(r["success"]), **{s: bool(r.get(s, False)) for s in STAGES},
                    fail_reason=(r.get("fail_reason") or ""), phase_reached=r.get("phase_reached", ""),
                    box_x=_num(box[0]), box_y=_num(box[1]), heldout=bool(r.get("heldout", False)),
                    palm_box_min_m=_num(r.get("palm_box_min_m")),
                    palm_box_nearest_min_m=_num(r.get("palm_box_nearest_min_m")),
                    final_box_goal_m=_num(r.get("final_box_goal_m", r.get("placement_error_m"))),
                    closest_box_goal_m=_num(r.get("closest_box_goal_m")),
                    placement_error_m=_num(r.get("placement_error_m")),
                    lock_rule=_rule(r.get("lock_rule")) or _rule(h.get("lock_rule")),
                    train_lock_rule=h.get("training_data_lock_rule"),
                    ckpt=ck.get("file"), ckpt_step=ck.get("step"), ckpt_epoch=ck.get("epoch"),
                    walkin=r.get("walkin", h.get("walkin")),
                    weld_engage_tick=int(r.get("weld_engage_tick", -1)),
                    weld_release_tick=int(r.get("weld_release_tick", -1)),
                    resting=bool(r.get("resting", False)), tilt_deg=_num(r.get("tilt_deg")),
                    trajectory_sha256=r.get("trajectory_sha256"),
                    **_grasp_cols(r.get("grasp_detail"))))
            hdr = (rows[0].get("header") or {}) if rows else {}
            run_row.setdefault("lock_rule", _rule(hdr.get("lock_rule")))
            run_row.setdefault("train_lock_rule", hdr.get("training_data_lock_rule"))
            run_row.setdefault("ckpt", (hdr.get("ckpt") or {}).get("file"))
            if info["model"] is None:
                run_row["model"] = hdr.get("kind")
        runs.append(run_row)
    cols = ["run", "model", "size", "size_n", "train_seed", "exp", "seed", "success", *STAGES,
            "fail_reason", "phase_reached", "box_x", "box_y", "heldout", "palm_box_min_m",
            "palm_box_nearest_min_m", "final_box_goal_m", "closest_box_goal_m",
            "placement_error_m", "lock_rule", "train_lock_rule", "ckpt", "ckpt_step",
            "ckpt_epoch", "walkin", "weld_engage_tick", "weld_release_tick", "resting",
            "tilt_deg", "trajectory_sha256", "grasp_near_spawn", "grasp_box_above_platform",
            "box_moved_from_spawn_m", "grasp_detail_source"]
    E = pd.DataFrame(eps, columns=cols)
    R = pd.DataFrame(runs)
    return E, R


def _order(df, col="model"):
    import pandas as pd
    key = {m: i for i, m in enumerate(MODELS + tuple(CONTROLS))}
    df = df.copy()
    df["_o"] = df[col].map(lambda m: key.get(m, 99))
    sort = ["_o"] + [c for c in ("size_n", "train_seed", "exp") if c in df.columns]
    return df.sort_values(sort, na_position="last").drop(columns="_o").reset_index(drop=True) \
        if len(df) else pd.DataFrame(columns=[c for c in df.columns if c != "_o"])


def label(model, size) -> str:
    return "%s-%s" % (MODEL_LABEL.get(model, str(model)), size if size else "?")


# ─── tables ──────────────────────────────────────────────────────────────────
def success_by_run(E):
    import pandas as pd
    rows = []
    for (run, model, size, ts, exp), g in E.groupby(["run", "model", "size", "train_seed", "exp"],
                                                    dropna=False):
        k, n = int(g["success"].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(run=run, model=model, size=size, train_seed=ts, exp=exp, n=n,
                         successes=k, rate=k / n if n else float("nan"), ci_lo=lo, ci_hi=hi))
    return _order(pd.DataFrame(rows, columns=["run", "model", "size", "train_seed", "exp", "n",
                                              "successes", "rate", "ci_lo", "ci_hi"]))


def success_summary(by_run):
    import pandas as pd
    rows = []
    for (model, size, exp), g in by_run.groupby(["model", "size", "exp"], dropna=False):
        k, n = int(g["successes"].sum()), int(g["n"].sum())
        lo, hi = wilson(k, n)
        rates = g["rate"].to_numpy(float)
        rows.append(dict(model=model, size=size, size_n=size_n(size), exp=exp,
                         n_seeds=len(g), seeds=",".join(str(int(s)) for s in sorted(g["train_seed"].dropna())),
                         mean_rate=float(rates.mean()),
                         sd_rate=float(rates.std(ddof=1)) if len(rates) > 1 else float("nan"),
                         pooled_successes=k, pooled_n=n, pooled_rate=k / n if n else float("nan"),
                         ci_lo=lo, ci_hi=hi))
    return _order(pd.DataFrame(rows, columns=["model", "size", "size_n", "exp", "n_seeds", "seeds",
                                              "mean_rate", "sd_rate", "pooled_successes", "pooled_n",
                                              "pooled_rate", "ci_lo", "ci_hi"]))


def generalization_gap(by_run):
    import pandas as pd
    rows = []
    for (model, size), g in by_run.groupby(["model", "size"], dropna=False):
        e1, e2 = g[g.exp == "exp1"], g[g.exp == "exp2"]
        row = dict(model=model, size=size, size_n=size_n(size))
        if not len(e1) or not len(e2):
            row.update(note="needs both experiments (have %s)" % ",".join(sorted(g.exp.unique())))
            rows.append(row)
            continue
        k1, n1, k2, n2 = int(e1.successes.sum()), int(e1.n.sum()), int(e2.successes.sum()), int(e2.n.sum())
        p1, p2 = k1 / n1, k2 / n2
        se_b = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        paired = e1.set_index("train_seed")["rate"].sub(e2.set_index("train_seed")["rate"]).dropna()
        se_s = float(paired.std(ddof=1) / math.sqrt(len(paired))) if len(paired) > 1 else float("nan")
        row.update(exp1_rate=p1, exp2_rate=p2, gap=p1 - p2, se_binomial=se_b,
                   ci_lo=p1 - p2 - Z95 * se_b, ci_hi=p1 - p2 + Z95 * se_b,
                   n_seeds_paired=len(paired),
                   gap_seed_mean=float(paired.mean()) if len(paired) else float("nan"),
                   se_seeds=se_s, note="")
        rows.append(row)
    return _order(pd.DataFrame(rows, columns=["model", "size", "size_n", "exp1_rate", "exp2_rate", "gap",
                                              "se_binomial", "ci_lo", "ci_hi", "n_seeds_paired",
                                              "gap_seed_mean", "se_seeds", "note"]))


def stage_rates(E):
    import pandas as pd
    rows = []
    for (model, size, exp), g in E.groupby(["model", "size", "exp"], dropna=False):
        n = len(g)
        row = dict(model=model, size=size, size_n=size_n(size), exp=exp, n=n,
                   n_seeds=g.train_seed.nunique())
        for s in STAGES + ("success",):
            k = int(g[s].sum())
            lo, hi = wilson(k, n)
            row.update({s: k / n, s + "_k": k, s + "_ci_lo": lo, s + "_ci_hi": hi})
        rows.append(row)
    return _order(pd.DataFrame(rows))


def _bucket(fr: str) -> str:
    if not fr:
        return "success"
    return fr if fr in FAIL_ORDER else "other"


def failure_phase(E):
    import pandas as pd
    rows = []
    for (model, size, exp), g in E.groupby(["model", "size", "exp"], dropna=False):
        b = g.fail_reason.map(_bucket).value_counts()
        row = dict(model=model, size=size, size_n=size_n(size), exp=exp, n=len(g),
                   success=int(b.get("success", 0)))
        row.update({k: int(b.get(k, 0)) for k in FAIL_ORDER})
        rows.append(row)
    return _order(pd.DataFrame(rows))


def fisher_table(E):
    import pandas as pd
    rows = []
    for (size, exp), g in E.groupby(["size", "exp"], dropna=False):
        present = [m for m in MODELS if m in set(g.model)]
        fam = []
        for a, b in combinations(present, 2):
            ga, gb = g[g.model == a], g[g.model == b]
            k1, n1, k2, n2 = int(ga.success.sum()), len(ga), int(gb.success.sum()), len(gb)
            orr, p = fisher(k1, n1, k2, n2)
            fam.append(dict(size=size, exp=exp, model_a=a, model_b=b, a_successes=k1, a_n=n1,
                            b_successes=k2, b_n=n2, a_rate=k1 / n1, b_rate=k2 / n2,
                            odds_ratio=orr, p=p))
        for r, adj in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = adj
        rows += fam
    return pd.DataFrame(rows, columns=["size", "exp", "model_a", "model_b", "a_successes", "a_n",
                                       "b_successes", "b_n", "a_rate", "b_rate", "odds_ratio",
                                       "p", "p_holm"])


DIST_METRICS = (("palm_box_min_m", "grasp proximity (farther palm to box, before the weld)"),
                ("final_box_goal_m", "final box-to-goal distance"))


def distance_tests(E):
    """Kruskal-Wallis over the models present, then pairwise Mann-Whitney U."""
    import pandas as pd
    from scipy.stats import kruskal, mannwhitneyu
    rows = []
    for metric, what in DIST_METRICS:
        for (size, exp), g in E.groupby(["size", "exp"], dropna=False):
            present = [m for m in MODELS if m in set(g.model)]
            samples = {m: g.loc[g.model == m, metric].dropna().to_numpy(float) for m in present}
            dropped = {m: int(g.loc[g.model == m, metric].isna().sum()) for m in present}
            usable = [m for m in present if len(samples[m])]
            base = dict(metric=metric, size=size, exp=exp)
            if len(present) >= 2:
                # Reported even when fewer than two models HAVE the field, so the
                # dropped counts (older deploy records) are visible, with p blank.
                H = pk = float("nan")
                if len(usable) >= 2:
                    try:
                        H, pk = kruskal(*[samples[m] for m in usable])
                    except ValueError:                # all values identical
                        pass
                rows.append(dict(base, test="kruskal-wallis", models=",".join(present),
                                 stat=float(H), p=float(pk), p_holm=float("nan"),
                                 n=",".join(str(len(samples[m])) for m in present),
                                 dropped=",".join(str(dropped[m]) for m in present),
                                 median=",".join("%.4f" % np.median(samples[m]) if len(samples[m])
                                                 else "n/a" for m in present)))
            pairs = []
            for a, b in combinations(usable, 2):
                U, pu = mannwhitneyu(samples[a], samples[b], alternative="two-sided")
                pairs.append(dict(base, test="mann-whitney-u", models="%s,%s" % (a, b),
                                  stat=float(U), p=float(pu),
                                  n="%d,%d" % (len(samples[a]), len(samples[b])),
                                  dropped="%d,%d" % (dropped[a], dropped[b]),
                                  median="%.4f,%.4f" % (np.median(samples[a]), np.median(samples[b]))))
            for r, adj in zip(pairs, holm([r["p"] for r in pairs])):
                r["p_holm"] = adj
            rows += pairs
    return pd.DataFrame(rows, columns=["metric", "size", "exp", "test", "models", "stat", "p",
                                       "p_holm", "n", "dropped", "median"])


def thresholds(E):
    import pandas as pd
    rows = []
    for (model, size, exp), g in E.groupby(["model", "size", "exp"], dropna=False):
        row = dict(model=model, size=size, size_n=size_n(size), exp=exp, n=len(g))
        for metric in ("final_box_goal_m", "closest_box_goal_m"):
            v = g[metric].dropna().to_numpy(float)
            for t in THRESHOLDS_M:
                key = "%s_within_%dcm" % (metric.replace("_m", ""), round(100 * t))
                row[key] = float((v <= t).mean()) if len(v) else float("nan")
                row[key + "_k"] = int((v <= t).sum())
            row[metric.replace("_m", "") + "_n"] = len(v)
        v = g["palm_box_min_m"].dropna().to_numpy(float)
        row["palm_within_grasp_gate"] = float((v <= GRASP_GATE_M).mean()) if len(v) else float("nan")
        rows.append(row)
    return _order(pd.DataFrame(rows))


def size_comparison(summary):
    import pandas as pd
    rows = []
    sizes = sorted({s for s in summary["size"].dropna()}, key=lambda s: size_n(s))
    if len(sizes) < 2:
        return pd.DataFrame(columns=["model", "exp"])
    for (model, exp), g in summary.groupby(["model", "exp"]):
        if g["size"].nunique() < 2:
            continue
        row = dict(model=model, exp=exp)
        for _, r in g.iterrows():
            s = r["size"]
            row.update({"rate_%s" % s: r.pooled_rate, "k_%s" % s: r.pooled_successes,
                        "n_%s" % s: r.pooled_n, "seeds_%s" % s: r.n_seeds})
        small, big = sizes[0], sizes[-1]
        if ("k_%s" % small) in row and ("k_%s" % big) in row:
            orr, p = fisher(row["k_%s" % big], row["n_%s" % big], row["k_%s" % small], row["n_%s" % small])
            row.update(diff=row["rate_%s" % big] - row["rate_%s" % small], fisher_p=p)
        rows.append(row)
    return _order(pd.DataFrame(rows))


def control_comparison(E):
    """Each CONTROL against its base model (BC batch 256 vs the ladder's BC batch
    8), per data size and experiment, pooled over training seeds: successes,
    stage counts, Fisher's exact test on success (two-sided, no Holm - one test
    per cell, not a family)."""
    import pandas as pd
    rows = []
    for ctrl, base in CONTROLS.items():
        for (size, exp), g in E[E.model.isin([ctrl, base])].groupby(["size", "exp"], dropna=False):
            a, b = g[g.model == base], g[g.model == ctrl]
            if not len(a) or not len(b):
                continue
            row = dict(control=ctrl, base=base, size=size, size_n=size_n(size), exp=exp,
                       base_seeds=a.train_seed.nunique(), control_seeds=b.train_seed.nunique())
            for tag, d in (("base", a), ("control", b)):
                row.update({"%s_n" % tag: len(d), "%s_success" % tag: int(d.success.sum())})
                row.update({"%s_%s" % (tag, st): int(d[st].sum()) for st in STAGES})
            _, row["fisher_p"] = fisher(row["control_success"], row["control_n"],
                                        row["base_success"], row["base_n"])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["size_n", "exp"]).reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["control", "base", "size", "exp"])


def coverage(splits=DEFAULT_SPLITS) -> dict:
    """{size: (xs, ys)} - the training demonstrations' box spawns, from each
    split file's train seeds (spawn = a pure function of the seed)."""
    cfg = box_config()
    out = {}
    for size, path in splits:
        if os.path.exists(path):
            seeds = json.load(open(path, encoding="utf-8"))["train"]
            xy = np.array([spawn_xy(s, cfg) for s in seeds])
            out[size] = (xy[:, 0], xy[:, 1])
    return out


# ─── figures ─────────────────────────────────────────────────────────────────
COLOURS = {"bc": "#8c8c8c", "act": "#1f77b4", "act_lstm": "#d62728"}
EXP_HATCH = {"exp1": "", "exp2": "//"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _groups(summary):
    """[(model, size)] present, in model then size order."""
    seen = []
    for _, r in summary.iterrows():
        if (r.model, r["size"]) not in seen:
            seen.append((r.model, r["size"]))
    return seen


def fig_success(summary, out, title_extra="", by_run=None):
    plt = _plt()
    groups = _groups(summary)
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(groups) + 2), 4))
    w = 0.38
    for i, (m, s) in enumerate(groups):
        for j, exp in enumerate(EXPS):
            r = summary[(summary.model == m) & (summary["size"] == s) & (summary.exp == exp)]
            if not len(r):
                continue
            r = r.iloc[0]
            x = i + (j - 0.5) * w
            ax.bar(x, r.pooled_rate, w, color=COLOURS.get(m, "k"), hatch=EXP_HATCH[exp],
                   edgecolor="black", label=EXP_LABEL[exp] if i == 0 else None)
            ax.errorbar(x, r.pooled_rate, yerr=[[r.pooled_rate - r.ci_lo], [r.ci_hi - r.pooled_rate]],
                        color="black", capsize=3, lw=1)
            if by_run is not None and r.n_seeds > 1:
                pts = by_run[(by_run.model == m) & (by_run["size"] == s) & (by_run.exp == exp)].rate
                ax.scatter(np.full(len(pts), x), pts, s=12, c="white", edgecolors="black", zorder=3,
                           label="per training seed" if (i == 0 and j == 0) else None)
            ax.text(x, r.ci_hi + 0.02, "%d/%d" % (r.pooled_successes, r.pooled_n),
                    ha="center", fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["%s\n(%d seed%s)" % (label(m, s), _nseeds(summary, m, s),
                                              "" if _nseeds(summary, m, s) == 1 else "s")
                        for m, s in groups], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("success rate (pooled over seeds)")
    ax.set_title("Task success, Wilson 95% CI" + title_extra, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p = os.path.join(out, "fig_success.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def _nseeds(summary, m, s):
    r = summary[(summary.model == m) & (summary["size"] == s)]
    return int(r.n_seeds.max()) if len(r) else 0


def fig_stages(stages, out, title_extra=""):
    plt = _plt()
    groups = _groups(stages)
    fig, axes = plt.subplots(1, len(EXPS), figsize=(11, 3.8), sharey=True)
    for ax, exp in zip(axes, EXPS):
        w = 0.8 / max(len(groups), 1)
        for i, (m, s) in enumerate(groups):
            r = stages[(stages.model == m) & (stages["size"] == s) & (stages.exp == exp)]
            if not len(r):
                continue
            r = r.iloc[0]
            xs = np.arange(len(STAGES) + 1) + (i - (len(groups) - 1) / 2) * w
            vals = [r[k] for k in STAGES + ("success",)]
            ax.bar(xs, vals, w, color=COLOURS.get(m, "k"), alpha=0.55 + 0.45 * (size_n(s) or 0) / 200,
                   edgecolor="black", label=label(m, s))
        ax.set_xticks(range(len(STAGES) + 1))
        ax.set_xticklabels(list(STAGES) + ["success"], fontsize=8)
        ax.set_title(EXP_LABEL[exp] + title_extra, fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("rate (pooled over seeds)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out, "fig_stages.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


FAIL_COLOURS = {"success": "#2ca02c", "WALK_IN": "#9467bd", "GRASP": "#ff7f0e", "LIFT": "#bcbd22",
                "TRANSPORT": "#17becf", "PLACE": "#e377c2", "fell": "#7f7f7f", "other": "#000000"}


def fig_failures(fail, out, title_extra=""):
    plt = _plt()
    rows = [(r.model, r["size"], r.exp) for _, r in fail.iterrows()]
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(rows) + 2), 4))
    bottom = np.zeros(len(rows))
    for key in ("success",) + FAIL_ORDER:
        vals = np.array([fail.iloc[i][key] / fail.iloc[i].n for i in range(len(rows))])
        if not vals.any():
            continue
        ax.bar(range(len(rows)), vals, bottom=bottom, color=FAIL_COLOURS[key], edgecolor="black",
               label=key if key != "success" else "success (no failure)")
        bottom += vals
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["%s\n%s" % (label(m, s), e) for m, s, e in rows], fontsize=7)
    ax.set_ylabel("fraction of episodes")
    ax.set_ylim(0, 1.0)
    ax.set_title("Where episodes end: failure phase (deploy.py fail_reason)" + title_extra, fontsize=10)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    p = os.path.join(out, "fig_failure_phase.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_proximity(E, out, title_extra=""):
    plt = _plt()
    groups = []
    for m in MODELS:
        for s in sorted(E[E.model == m]["size"].dropna().unique(), key=size_n):
            groups.append((m, s))
    fig, axes = plt.subplots(len(DIST_METRICS), len(EXPS), figsize=(11, 7), squeeze=False)
    for i, (metric, what) in enumerate(DIST_METRICS):
        for j, exp in enumerate(EXPS):
            ax = axes[i][j]
            data, labels_, cols = [], [], []
            for m, s in groups:
                v = E[(E.model == m) & (E["size"] == s) & (E.exp == exp)][metric].dropna().to_numpy(float)
                if len(v):
                    data.append(v)
                    labels_.append(label(m, s))
                    cols.append(COLOURS.get(m, "k"))
            if data:
                bp = ax.boxplot(data, patch_artist=True, showfliers=True)
                for patch, c in zip(bp["boxes"], cols):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
                ax.set_xticks(range(1, len(labels_) + 1))
                ax.set_xticklabels(labels_, fontsize=7)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            if metric == "palm_box_min_m":
                ax.axhline(GRASP_GATE_M, color="green", ls="--", lw=1, label="weld gate 0.16 m")
            else:
                for t in THRESHOLDS_M:
                    ax.axhline(t, color="green", ls=":", lw=1)
            ax.set_ylabel("m")
            ax.set_title("%s - %s%s" % (what, EXP_LABEL[exp], title_extra), fontsize=8)
    fig.tight_layout()
    p = os.path.join(out, "fig_proximity.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def _patch(ax, reg):
    import matplotlib.patches as mpatches
    (x0, x1), (y0, y1) = reg["heldout_x"], reg["heldout_y"]
    ax.add_patch(mpatches.Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ec="black", lw=2,
                                    ls="--", label="held-out patch"))


def _grid(reg, nx=4, ny=8):
    return np.linspace(*reg["x"], nx + 1), np.linspace(*reg["y"], ny + 1)


def fig_position(E, out, reg, title_extra=""):
    """Rows = data sizes, columns = models: success rate per spawn cell (Exp 1
    and Exp 2 pooled), every episode as a marker; y (lateral) horizontal, x
    (depth) vertical, depth stretched for legibility."""
    plt = _plt()
    sizes = sorted(E["size"].dropna().unique(), key=size_n)
    models = [m for m in MODELS if m in set(E.model)]
    if not sizes or not models:
        return None
    xe, ye = _grid(reg)
    fig, axes = plt.subplots(len(sizes), len(models), figsize=(4.2 * len(models) + 1.2, 2.8 * len(sizes)),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.45)
    im = None
    for r_, s in enumerate(sizes):
        for c_, m in enumerate(models):
            ax = axes[r_][c_]
            g = E[(E.model == m) & (E["size"] == s)].dropna(subset=["box_x", "box_y"])
            if not len(g):
                ax.text(0.5, 0.5, "%s: no run" % label(m, s), ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            tot, _, _ = np.histogram2d(g.box_x, g.box_y, bins=[xe, ye])
            ok, _, _ = np.histogram2d(g.box_x[g.success], g.box_y[g.success], bins=[xe, ye])
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(tot > 0, ok / tot, np.nan)
            im = ax.pcolormesh(ye, xe, rate, cmap="RdYlGn", vmin=0, vmax=1, shading="flat")
            ax.scatter(g.box_y[~g.success], g.box_x[~g.success], s=7, c="black", marker="x", lw=0.6,
                       label="failure")
            ax.scatter(g.box_y[g.success], g.box_x[g.success], s=10, facecolors="none",
                       edgecolors="blue", lw=0.7, label="success")
            _patch(ax, reg)
            ax.set_title("%s - %d seed%s, %d episodes%s" % (
                label(m, s), g.train_seed.nunique(), "" if g.train_seed.nunique() == 1 else "s",
                len(g), title_extra), fontsize=8)
            ax.set_xlim(*reg["y"])
            ax.set_ylim(*reg["x"])
            if r_ == len(sizes) - 1:
                ax.set_xlabel("box y (m)")
            if c_ == 0:
                ax.set_ylabel("box x (m)")
    axes[0][-1].legend(fontsize=6, loc="upper right", framealpha=0.8)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="success rate per cell")
    fig.suptitle("Success by box spawn (Exp 1 + Exp 2 pooled; dashed = held-out patch; "
                 "depth axis stretched)", fontsize=9)
    p = os.path.join(out, "fig_success_by_position.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_coverage(cov, out, reg):
    if not cov:
        return None
    plt = _plt()
    xe, ye = _grid(reg)
    fig, axes = plt.subplots(1, len(cov), figsize=(4.2 * len(cov) + 1.2, 2.8), squeeze=False)
    im = None
    for ax, (size, (xs, ys)) in zip(axes[0], sorted(cov.items(), key=lambda kv: size_n(kv[0]))):
        h, _, _ = np.histogram2d(xs, ys, bins=[xe, ye])
        im = ax.pcolormesh(ye, xe, h, cmap="Blues", shading="flat")
        ax.scatter(ys, xs, s=6, c="black")
        _patch(ax, reg)
        ax.set_title("training demos, %s split (%d)" % (size, len(xs)), fontsize=8)
        ax.set_xlabel("box y (m)")
        ax.set_xlim(*reg["y"])
        ax.set_ylim(*reg["x"])
    axes[0][0].set_ylabel("box x (m)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="demonstrations per cell")
    p = os.path.join(out, "fig_training_coverage.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_sizes(summary, out):
    sizes = sorted(summary["size"].dropna().unique(), key=size_n)
    if len(sizes) < 2:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, len(EXPS), figsize=(10, 3.8), sharey=True)
    w = 0.8 / len(sizes)
    for ax, exp in zip(axes, EXPS):
        for j, s in enumerate(sizes):
            for i, m in enumerate(MODELS):
                r = summary[(summary.model == m) & (summary["size"] == s) & (summary.exp == exp)]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = i + (j - (len(sizes) - 1) / 2) * w
                ax.bar(x, r.pooled_rate, w, color=COLOURS[m], alpha=0.45 + 0.55 * j / max(len(sizes) - 1, 1),
                       edgecolor="black")
                ax.errorbar(x, r.pooled_rate, yerr=[[r.pooled_rate - r.ci_lo], [r.ci_hi - r.pooled_rate]],
                            color="black", capsize=3, lw=1)
                ax.text(x, -0.07, s, ha="center", fontsize=7)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
        ax.tick_params(axis="x", pad=14)
        ax.set_title("%s: data size %s" % (EXP_LABEL[exp], " vs ".join(sizes)), fontsize=9)
        ax.set_ylim(-0.1, 1.05)
    axes[0].set_ylabel("success rate (pooled, Wilson 95% CI)")
    fig.tight_layout()
    p = os.path.join(out, "fig_size_comparison.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ─── report ──────────────────────────────────────────────────────────────────
def _pct(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.1f%%" % (100 * x)


def _p(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return "%.2g" % x if x < 0.001 else "%.4f" % x


def _md(df, cols, fmt=None) -> str:
    fmt = fmt or {}
    if not len(df):
        return "_(none)_\n"
    out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c, key in cols:
            v = key(r) if callable(key) else r.get(key)
            cells.append(fmt.get(c, str)(v) if v is not None else "")
        out.append("| " + " | ".join(str(x) for x in cells) + " |")
    return "\n".join(out) + "\n"


def write_report(out, roots, E, R, T, figs, notes) -> str:
    L = ["# Experiment 1 / 2 analysis", ""]
    L.append("Input: %s. Generated by `tools/analyze_experiments.py` (read-only on the eval output)."
             % ", ".join("`%s`" % repo_relpath(r) for r in roots))
    L.append("")
    rules = sorted({str(x) for x in E.lock_rule.dropna().unique()})
    trules = sorted({str(x) for x in E.train_lock_rule.dropna().unique()})
    L.append("**Lock rule at evaluation:** %s. **Training-data lock rule:** %s. Episodes: %d."
             % (", ".join(rules) or "unknown", ", ".join(trules) or "unknown", len(E)))
    L += ["", "## Runs found", ""]
    L.append(_md(R, [("run", "run"), ("model", lambda r: MODEL_LABEL.get(r.model, r.model)),
                     ("data size", "size"), ("train seed", "train_seed"),
                     ("checkpoint", lambda r: r.get("chosen_ckpt") or r.get("ckpt")),
                     ("selection", lambda r: "%s/%s" % (r.get("selection_success"), r.get("selection_n"))
                      if r.get("selection_n") else ""),
                     ("lock rule", "lock_rule"), ("Exp 1 eps", "exp1_n"), ("Exp 2 eps", "exp2_n")]))
    for n in notes:
        L.append("- " + n)
    L += ["", "## 1. Task success", "", "### Per run (Wilson 95% CI)", ""]
    L.append(_md(T["by_run"], [("model", lambda r: MODEL_LABEL.get(r.model, r.model)), ("size", "size"),
                               ("seed", "train_seed"), ("exp", "exp"),
                               ("success", lambda r: "%d/%d" % (r.successes, r.n)),
                               ("rate", lambda r: _pct(r.rate)),
                               ("95% CI", lambda r: "%s - %s" % (_pct(r.ci_lo), _pct(r.ci_hi)))]))
    L += ["### Over training seeds", "",
          "Mean +- SD over the per-seed rates; pooled = all seeds' episodes as one sample.", ""]
    L.append(_md(T["summary"], [("model", lambda r: MODEL_LABEL.get(r.model, r.model)), ("size", "size"),
                                ("exp", "exp"), ("seeds", "seeds"),
                                ("mean +- SD", lambda r: "%s +- %s" % (_pct(r.mean_rate), _pct(r.sd_rate) or "n/a")),
                                ("pooled", lambda r: "%d/%d = %s" % (r.pooled_successes, r.pooled_n,
                                                                     _pct(r.pooled_rate))),
                                ("95% CI", lambda r: "%s - %s" % (_pct(r.ci_lo), _pct(r.ci_hi)))]))
    L += ["### Generalization gap (Exp 1 - Exp 2)", ""]
    L.append(_md(T["gap"], [("model", lambda r: MODEL_LABEL.get(r.model, r.model)), ("size", "size"),
                            ("Exp 1", lambda r: _pct(r.get("exp1_rate"))),
                            ("Exp 2", lambda r: _pct(r.get("exp2_rate"))),
                            ("gap", lambda r: _pct(r.get("gap"))),
                            ("SE (binomial)", lambda r: _pct(r.get("se_binomial"))),
                            ("95% CI", lambda r: "%s - %s" % (_pct(r.get("ci_lo")), _pct(r.get("ci_hi")))
                             if not _isnan(r.get("gap")) else ""),
                            ("SE (seeds)", lambda r: _pct(r.get("se_seeds")) or "n/a"),
                            ("note", lambda r: r.get("note") or "")]))
    L += ["## 2. Stage-wise rates", ""]
    L.append(_md(T["stages"], [("model", lambda r: MODEL_LABEL.get(r.model, r.model)), ("size", "size"),
                               ("exp", "exp"), ("n", "n")] +
                 [(s, (lambda s_: lambda r: "%s (%d)" % (_pct(r[s_]), r[s_ + "_k"]))(s))
                  for s in STAGES + ("success",)]))
    L += ["### Failure phase (counts)", ""]
    L.append(_md(T["fail"], [("model", lambda r: MODEL_LABEL.get(r.model, r.model)), ("size", "size"),
                             ("exp", "exp"), ("n", "n"), ("success", "success")] +
                 [(k, k) for k in FAIL_ORDER]))
    L += ["## 3. Statistics", "", "### Fisher's exact test on success (pooled over seeds; Holm over the pairs)", ""]
    L.append(_md(T["fisher"], [("size", "size"), ("exp", "exp"),
                               ("pair", lambda r: "%s vs %s" % (MODEL_LABEL[r.model_a], MODEL_LABEL[r.model_b])),
                               ("successes", lambda r: "%d/%d vs %d/%d" % (r.a_successes, r.a_n,
                                                                         r.b_successes, r.b_n)),
                               ("odds ratio", lambda r: "%.3g" % r.odds_ratio), ("p", lambda r: _p(r.p)),
                               ("p (Holm)", lambda r: _p(r.p_holm))]))
    L += ["### Distances: Kruskal-Wallis, then Mann-Whitney U (Holm over the pairs)", ""]
    L.append(_md(T["dist"], [("metric", "metric"), ("size", "size"), ("exp", "exp"), ("test", "test"),
                             ("models", "models"), ("n", "n"), ("dropped (no field)", "dropped"),
                             ("median (m)", "median"), ("stat", lambda r: "%.4g" % r.stat),
                             ("p", lambda r: _p(r.p)), ("p (Holm)", lambda r: _p(r.p_holm))]))
    L += ["### Threshold rates", ""]
    L.append(_md(T["thr"], [("model", lambda r: MODEL_LABEL.get(r.model, r.model)), ("size", "size"),
                            ("exp", "exp"), ("n", "n"),
                            ("final <= 5 cm", lambda r: _pct(r.get("final_box_goal_within_5cm"))),
                            ("final <= 10 cm", lambda r: _pct(r.get("final_box_goal_within_10cm"))),
                            ("closest <= 5 cm", lambda r: _pct(r.get("closest_box_goal_within_5cm"))),
                            ("closest <= 10 cm", lambda r: _pct(r.get("closest_box_goal_within_10cm"))),
                            ("palms within weld gate", lambda r: _pct(r.get("palm_within_grasp_gate")))]))
    if len(T["sizes"]):
        L += ["### Data size", ""]
        cols = [c for c in T["sizes"].columns if c not in ("model", "exp")]
        L.append(_md(T["sizes"], [("model", lambda r: MODEL_LABEL.get(r.model, r.model)), ("exp", "exp")] +
                     [(c, (lambda c_: lambda r: _pct(r[c_]) if c_.startswith(("rate", "diff"))
                           else (_p(r[c_]) if c_ == "fisher_p" else r[c_]))(c)) for c in cols]))
    L += ["### D19 sensitivity (grasp needs the box within 50 mm of its spawn at the weld)", "",
          "welded = the weld engaged; welded, no grasp = it engaged but `grasp` is False; D19 = the "
          "push rule failed (`box_near_spawn` False; D19 only = the box was still on the platform); "
          "of the D19 episodes, how many ended placed (final box-goal <= 0.10 m), resting, released, "
          "upright, and all four (every final-tick place condition: an UPPER BOUND on the successes "
          "D19 removed - lift/transport are not in the record). unknown = no grasp_detail for it.", ""]
    L.append(_md(T["d19"], [("model", lambda r: MODEL_LABEL.get(r.model, r.model)), ("size", "size"),
                            ("exp", "exp"), ("n", "n"), ("welded", "welded"), ("grasp", "grasp"),
                            ("welded, no grasp", "welded_no_grasp"), ("D19", "d19"),
                            ("D19 only", "d19_only"), ("unknown", "detail_unknown"),
                            ("placed", "d19_placed"), ("resting", "d19_resting"),
                            ("released", "d19_released"), ("upright", "d19_upright"),
                            ("all four", "d19_all_place_conditions"), ("success", "success")]))
    if len(T["control"]):
        L += ["### BC control: own default batch 256 vs the ladder's batch 8 (review M3)", "",
              "Pooled over training seeds; Fisher's exact test on success, two-sided. The "
              "control is not part of the three-model pairwise tests above.", ""]
        L.append(_md(T["control"], [
            ("size", "size"), ("exp", "exp"),
            ("BC b8", lambda r: "%d/%d (g %d l %d t %d p %d)" % (
                r.base_success, r.base_n, r.base_grasp, r.base_lift, r.base_transport, r.base_place)),
            ("BC b256", lambda r: "%d/%d (g %d l %d t %d p %d)" % (
                r.control_success, r.control_n, r.control_grasp, r.control_lift,
                r.control_transport, r.control_place)),
            ("seeds b8/b256", lambda r: "%d/%d" % (r.base_seeds, r.control_seeds)),
            ("Fisher p", lambda r: _p(r.fisher_p))]))
    L += ["## 4. Figures", ""]
    for name, path in figs:
        if path:
            L.append("**%s**\n\n![%s](%s)\n" % (name, name, os.path.basename(path)))
    L += ["## Method notes", "",
          "- Wilson score 95% intervals. Pooling over seeds treats all episodes as one binomial "
          "sample (ignores between-seed variance; see the SD column).",
          "- Gap SE (binomial) = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2) on the pooled rates (disjoint seed "
          "sets, independent samples); SE (seeds) = SD of per-seed gaps / sqrt(k), k >= 2.",
          "- Fisher's exact test two-sided; Kruskal-Wallis then two-sided Mann-Whitney U; Holm-Bonferroni "
          "over the pairs of one (data size, experiment[, metric]) family.",
          "- Grasp proximity = `palm_box_min_m` (farther palm's closest approach before the weld; the gate "
          "is 0.16 m). Final distance = `final_box_goal_m` (= placement error). Failure phase = deploy.py "
          "`fail_reason` (the ScoredPhase charged, or `fell`).",
          "- Heatmaps: x = depth (towards the platform), y = lateral; 4 x 8 cells over the spawn region; "
          "dashed rectangle = the Objective 4 held-out patch. Training coverage is from the split files' "
          "train seeds (box spawn is a pure function of the seed)."]
    path = os.path.join(out, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def default_d19_roots(roots: Sequence[str]) -> List[str]:
    """<eval root>_d19 siblings (tools/rescore_d19.py writes <run-root>/eval_d19)."""
    out = []
    for r in roots:
        d = os.path.join(os.path.dirname(os.path.abspath(r)),
                         os.path.basename(os.path.normpath(r)) + "_d19")
        if os.path.isdir(d):
            out.append(d)
    return out


def analyze(roots: Sequence[str], out: str, splits=DEFAULT_SPLITS, figures: bool = True,
            d19_roots: Optional[Sequence[str]] = None) -> dict:
    """Run everything; returns the tables (DataFrames) and the figure paths."""
    os.makedirs(out, exist_ok=True)
    E, R = load(roots)
    if not len(E):
        raise SystemExit("no exp1.jsonl / exp2.jsonl found under %s" % ", ".join(roots))
    notes = []
    d19_roots = default_d19_roots(roots) if d19_roots is None else list(d19_roots)
    used, rejected = attach_grasp_detail(E, d19_roots)
    if d19_roots:
        notes.append("grasp_detail from re-runs (%s): %d used, %d rejected (trajectory differs)"
                     % (", ".join(repo_relpath(d) for d in d19_roots), used, rejected))
    for _, r in R.iterrows():
        miss = [e for e in EXPS if not r.get("%s_n" % e)]
        if miss:
            notes.append("%s: no %s results" % (r.run, " / ".join(miss)))
        if r.model is None or r.get("size") is None:
            notes.append("%s: folder name is not <model>_<size>_seed<k>; model from the header" % r.run)
    for m in MODELS:
        if m not in set(E.model):
            notes.append("%s: no runs found" % MODEL_LABEL[m])
    by_run = success_by_run(E)
    T = dict(by_run=by_run, summary=success_summary(by_run), gap=generalization_gap(by_run),
             stages=stage_rates(E), fail=failure_phase(E), fisher=fisher_table(E),
             dist=distance_tests(E), thr=thresholds(E), d19=d19_sensitivity(E),
             control=control_comparison(E))
    T["sizes"] = size_comparison(T["summary"])
    E.to_csv(os.path.join(out, "episodes.csv"), index=False)
    R.to_csv(os.path.join(out, "runs.csv"), index=False)
    for key, name in (("by_run", "success_by_run"), ("summary", "success_summary"),
                      ("gap", "generalization_gap"), ("stages", "stage_rates"),
                      ("fail", "failure_phase"), ("fisher", "fisher"), ("dist", "distance_tests"),
                      ("thr", "thresholds"), ("sizes", "size_comparison"),
                      ("d19", "d19_sensitivity"), ("control", "bc_control")):
        T[key].to_csv(os.path.join(out, name + ".csv"), index=False)
    figs = []
    if figures:
        reg = region()
        rules = ", ".join(sorted({str(x) for x in E.lock_rule.dropna().unique()})) or "?"
        tx = " - lock rule %s" % rules
        figs = [("Success with 95% CI", fig_success(T["summary"], out, tx, by_run)),
                ("Stage-wise rates", fig_stages(T["stages"], out, tx)),
                ("Failure phase", fig_failures(T["fail"], out, tx)),
                ("Grasp proximity and final box-to-goal distance", fig_proximity(E, out)),
                ("Success by box position (Exp 1 + Exp 2 pooled)", fig_position(E, out, reg)),
                ("Training-demo coverage", fig_coverage(coverage(splits), out, reg)),
                ("n40 vs 160", fig_sizes(T["summary"], out))]
    report = write_report(out, roots, E, R, T, figs, notes)
    return dict(tables=T, episodes=E, runs=R, figures=dict(figs), report=report, notes=notes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", nargs="+", default=[DEFAULT_EVAL],
                    help="eval folder(s) from tools/eval_scripted.py (default runs/scripted_v3/eval)")
    ap.add_argument("--out", default=None,
                    help="report folder (default <first eval root>/../analysis)")
    ap.add_argument("--splits", nargs="*", default=None, metavar="SIZE=PATH",
                    help="training split files for the coverage map (default the scripted_v3 "
                         "160 and n40 splits)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--d19-root", nargs="*", default=None,
                    help="tools/rescore_d19.py output with grasp_detail for older records "
                         "(default: <eval root>_d19 when it exists)")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.eval_root[0])), "analysis")
    splits = DEFAULT_SPLITS if a.splits is None else tuple(tuple(s.split("=", 1)) for s in a.splits)
    res = analyze(a.eval_root, out, splits=splits, figures=not a.no_figures, d19_roots=a.d19_root)
    print("analysed %d episodes from %d runs -> %s" % (len(res["episodes"]), len(res["runs"]), out))
    for n in res["notes"]:
        print("  note: " + n)
    s = res["tables"]["summary"]
    for _, r in s.iterrows():
        print("  %-14s %-5s %3d/%-4d %5.1f%%  [%4.1f, %4.1f]  (%d seed%s)"
              % (label(r.model, r["size"]), r.exp, r.pooled_successes, r.pooled_n, 100 * r.pooled_rate,
                 100 * r.ci_lo, 100 * r.ci_hi, r.n_seeds, "" if r.n_seeds == 1 else "s"))
    print("report: %s" % res["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
