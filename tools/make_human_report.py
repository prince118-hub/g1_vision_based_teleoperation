"""The final human_v2 analysis, as one report folder, under thesis names.

    python tools/make_human_report.py            # runs/human_v2/analysis_final/
    python tools/make_human_report.py --out reports/human --no-strict

READ-ONLY on the evaluation: no training, no new evaluation. Inputs:
  runs/human_v2/eval/<run>/exp{1,2}.jsonl      the locked human_v2 study (CLAUDE.md section 8, 2026-11-20):
                                               BC = bc_b256, ACT = act_orig, ACT-LSTM = act_lstm_full, seeds 0-2
  runs/scripted_v3{,_memory_design}/eval        scripted Run A (n40, seeds 0-2), the same three models
  runs/human_v2/analysis_final/teleop/          tools/teleop_metrics.py on recordings/demonstrations
Loading, Wilson, Holm and Fisher come from tools/analyze_experiments.py; the paired position test, the
Mann-Whitney family and the proximity summary from tools/make_final_report.py.

PHASES (Charles, 2026-11-22). FIVE official phases, in order:
  1 walk-in    the pickup base lock engaged: a lock (lock rule v3) that engaged while the box was NOT welded,
               before the weld. A lock that engages while welded targets the GOAL (g1_data/phases.py terms():
               `target = goal_xy if welded`), so "welded at the lock tick" is what separates the two locks.
               Welded at lock tick L = weld_engage_tick < L and (weld_release_tick < 0 or >= L): the lock
               decision at L reads the state at the start of tick L, which shows a weld engaged in an
               earlier tick's physics.
  2 grasp, 3 lift, 4 transport, 5 place        success.py, unchanged (grasp includes the D19 push rule).
  Cumulative rate of phase k = episodes that passed phases 1..k. Conditional rate = phase k given 1..k-1.
  Full task success is reported beside the phases (it equals place on every episode here).
DIAGNOSTICS, not phases: goal lock (a lock engaged while welded), release (the weld let go after
engaging, wherever that was), and, of the grasped boxes, how many reached the goal lock, were released
and were placed.

TESTS. PRIMARY: the position-level paired test (make_final_report.paired_test): per box position (the
same 100 per experiment for every policy), the count of seeds 0-3 that reached the phase; Wilcoxon
signed-rank and sign test, two-sided. SECONDARY: Fisher's exact test on pooled counts (episodes treated
as independent). Holm families: (a) the 5 phases x 2 experiments x 3 pairs = 30 tests, per test type;
(b) full success, 2 x 3 = 6; (c) diagnostics, exploratory, 2 x 2 x 3 = 12. Human vs scripted uses the
same families with the 3 models in place of the 3 pairs.

SPAWN SIDE. The lateral thirds of the spawn region, the same cells as the 3x3 split bins
(g1_data/dataset._spawn_bins): world y < -0.07 m right, -0.07..0.07 centre, > 0.07 left. The robot
starts at the origin facing +x (g1.xml pelvis, identity orientation), so world +y is its left.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import analyze_experiments as AE                                    # noqa: E402
import make_final_report as MFR                                     # noqa: E402
import report_io as RIO                                             # noqa: E402

HUMAN_ROOTS = (os.path.join(ROOT, "runs", "human_v2", "eval"),)
SCRIPTED_ROOTS = MFR.ROOTS
SCRIPTED_D19 = MFR.D19_ROOTS
OUT = os.path.join(ROOT, "runs", "human_v2", "analysis_final")
HUMAN_RE = re.compile(r"^(?P<code>bc_b256|act_orig|act_lstm_full)_human40v2_seed(?P<seed>\d+)$")
CODE = {"bc_b256": "BC", "act_orig": "ACT", "act_lstm_full": "ACT-LSTM"}
MODELS = ("BC", "ACT", "ACT-LSTM")
PAIRS = (("ACT-LSTM", "ACT"), ("ACT-LSTM", "BC"), ("ACT", "BC"))
PHASES = ("walk_in", "grasp", "lift", "transport", "place")
OLD_PHASES = ("grasp", "lift", "transport", "place")
CHAIN = PHASES + ("success",)
DIAG = ("goal_lock", "release")
LABEL = {"walk_in": "walk-in", "grasp": "grasp", "lift": "lift", "transport": "transport", "place": "place",
         "success": "full success", "goal_lock": "goal lock", "release": "release"}
EXPS = AE.EXPS
EXP_NAME = MFR.EXP_NAME
SIDES = ("right", "centre", "left")
HUMAN_MAX_TICKS = 1568
N_RUNS, N_EPS = 9, 1800
INDEP = "episodes treated as independent"
SUMMARY_RE = re.compile(r"^(?P<run>\S+) +(?P<ck>\S+) +sel +\d+/\d+ \| exp1 +(?P<s1>\d+)/100 \(g (?P<g1>\d+) l (?P<l1>\d+) "
                        r"t (?P<t1>\d+) p (?P<p1>\d+),.*\| exp2 +(?P<s2>\d+)/100 \(g (?P<g2>\d+) l (?P<l2>\d+) t (?P<t2>\d+) "
                        r"p (?P<p2>\d+),")


# ─── definitions ──────────────────────────────────────────────────────────────
def lock_flags(lock_ticks: Sequence[int], weld_engage: int, weld_release: int) -> dict:
    """Pickup lock (walk-in), goal lock and release from one deploy record (see the module docstring)."""
    we, wr = int(weld_engage), int(weld_release)

    def welded(t):
        return we >= 0 and we < t and (wr < 0 or wr >= t)
    pick = [int(t) for t in lock_ticks if not welded(t) and (we < 0 or t <= we)]
    goal = [int(t) for t in lock_ticks if welded(t)]
    return dict(walk_in=bool(pick), pickup_lock_tick=pick[0] if pick else -1, goal_lock=bool(goal),
                goal_lock_tick=goal[0] if goal else -1, release=wr >= 0, n_locks=len(lock_ticks))


def side_of(y: float, reg: Optional[dict] = None) -> str:
    """The lateral third of the spawn region: the 3x3 split bins' y index (0 = most negative y = right)."""
    reg = reg or AE.region()
    lo, hi = reg["y"]
    return SIDES[int(np.clip(int((y - lo) / (hi - lo) * 3), 0, 2))]


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple:
    """Newcombe's hybrid score interval (method 10) for p1 - p2, from the two Wilson intervals."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = AE.wilson(k1, n1)
    l2, u2 = AE.wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def add_cumulative(E):
    """cum_<phase>: passed phases 1..k; cum_success: passed all five and succeeded."""
    E = E.copy()
    run = np.ones(len(E), dtype=bool)
    for p in CHAIN:
        run = run & E[p].to_numpy(bool)
        E["cum_" + p] = run
    return E


def first_failure(r) -> str:
    """The first of the five phases an episode did not reach; 'fell' when it fell first; 'success'."""
    if r.cum_success:
        return "success"
    for p in PHASES:
        if not r["cum_" + p]:
            return "fell" if r.fell else p.upper()
    return "fell" if r.fell else "PLACE"


# ─── loading ──────────────────────────────────────────────────────────────────
def _extra(run_dirs: Dict[str, str]) -> dict:
    """Fields AE.load does not keep, keyed by (run, exp, seed)."""
    out = {}
    for run, d in run_dirs.items():
        for exp in EXPS:
            p = os.path.join(d, "%s.jsonl" % exp)
            if not os.path.exists(p):
                continue
            for r in AE._read_jsonl(p):
                h = r.get("header") or {}
                f = lock_flags(r.get("lock_ticks") or [], r.get("weld_engage_tick", -1), r.get("weld_release_tick", -1))
                f.update(fell=bool(r.get("fell", False)), ticks=int(r.get("ticks", 0)), end_reason=r.get("end_reason"),
                         max_ticks=h.get("max_ticks"), norm=h.get("norm"),
                         pushed_before_weld=(r.get("diagnostics") or {}).get("pushed_before_weld"))
                out[(run, exp, int(r["seed"]))] = f
    return out


def load_set(roots: Sequence[str], data: str, d19_roots: Sequence[str] = ()):
    """(episodes, runs, run folders) for the three main models of one data set."""
    E, R = AE.load(roots)
    if d19_roots:
        AE.attach_grasp_detail(E, list(d19_roots))
    dirs = {os.path.basename(d): d for d in AE.find_runs(roots)}
    if data == "human":
        for frame in (E, R):
            m = [HUMAN_RE.match(n) for n in frame.run]
            bad = sorted({n for n, x in zip(frame.run, m) if x is None})
            if bad:
                raise SystemExit("not a human_v2 main run: %s" % bad)
            frame["model"] = [CODE[x.group("code")] for x in m]
            frame["train_seed"] = [int(x.group("seed")) for x in m]
            frame["size"], frame["size_n"] = "n40", 40
    else:
        E = E[E.model.isin(MODELS) & (E["size"] == "n40")].reset_index(drop=True)
        R = R[R.model.isin(MODELS) & (R["size"] == "n40")].reset_index(drop=True)
        dirs = {k: v for k, v in dirs.items() if k in set(R.run)}
    extra = _extra(dirs)
    keys = list(zip(E.run, E.exp, E.seed.astype(int)))
    for col in ("walk_in", "pickup_lock_tick", "goal_lock", "goal_lock_tick", "release", "n_locks", "fell", "ticks",
                "end_reason", "max_ticks", "norm", "pushed_before_weld"):
        E[col] = [extra[k][col] for k in keys]
    E["data"] = data
    E["side"] = [side_of(y) for y in E.box_y]
    E = add_cumulative(E)
    E["first_failure"] = [first_failure(r) for _, r in E.iterrows()]
    R["data"] = data
    return E, R, dirs


# ─── checks: the numbers are what they claim to be ────────────────────────────
def audit(E, R, data: str, strict: bool = True):
    """Exactly 9 runs x 200 episodes, the same seed sets, and each run is what its thesis name says."""
    problems = []
    if not (len(R) == N_RUNS and len(E) == N_EPS):
        problems.append("expected %d runs / %d episodes, found %d / %d" % (N_RUNS, N_EPS, len(R), len(E)))
    want = {(m, s) for m in MODELS for s in (0, 1, 2)}
    if set(zip(R.model, R.train_seed)) != want:
        problems.append("runs are not 3 models x seeds 0-2")
    seeds = {e: E[E.exp == e].groupby("run").seed.apply(lambda s: tuple(sorted(s))) for e in EXPS}
    checks = {"one Exp 1 seed set": seeds["exp1"].nunique() == 1, "one Exp 2 seed set": seeds["exp2"].nunique() == 1,
              "100 positions per experiment": all(len(v.iloc[0]) == 100 for v in seeds.values()),
              "Exp 1 and Exp 2 disjoint": not set(seeds["exp1"].iloc[0]) & set(seeds["exp2"].iloc[0]),
              "Exp 1 outside the patch": not E[E.exp == "exp1"].heldout.any(),
              "Exp 2 inside the patch": E[E.exp == "exp2"].heldout.all(),
              "lock rule v3": set(E.lock_rule) == {"v3"}, "policy walk-in": set(E.walkin) == {"policy"},
              "success equals place": bool((E.success == E.place).all())}
    for column, _, _ in MFR.PROXIMITY:
        checks["%s on every episode" % column] = bool(E[column].notna().all())
    if data == "human":
        checks["eval cap 1568"] = set(E.max_ticks) == {HUMAN_MAX_TICKS}
        checks["human_v2 norm"] = all(str(n).replace("\\", "/").endswith("norm_stats_human_v2.npz") for n in set(E.norm))
    problems += [k for k, ok in checks.items() if not ok]
    rows = []
    for _, r in R.iterrows():
        base = os.path.dirname(os.path.dirname(os.path.join(ROOT, r.path)))
        with open(os.path.join(base, r.run, "metadata.json"), encoding="utf-8") as fh:
            text = fh.read()
        m = json.loads(text)
        got = (m["config"]["notes"]["model"], int(m["config"]["notes"]["chunk_size"]),
               int(m["loader"]["obs_window"]), int(m["config"]["batch_size"]))
        if got != MFR.EXPECT[r.model] or len(m["data"]["seeds"]) != 40:
            problems.append("%s is not %s: %s, %d demos" % (r.run, r.model, got, len(m["data"]["seeds"])))
        if data == "human" and not ("centred_ma13_v1" in text and m["data"].get("real_demonstrations") is True):
            problems.append("%s is not trained on human_v2 (centred_ma13_v1, real demonstrations)" % r.run)
        ck = set(E[E.run == r.run].ckpt)
        if ck != {r.chosen_ckpt}:
            problems.append("%s: episodes from %s, selection chose %s" % (r.run, ck, r.chosen_ckpt))
        with open(os.path.join(base, r.run, "result.json"), encoding="utf-8") as fh:
            res = json.load(fh)
        caps = set(E[E.run == r.run].max_ticks)
        rows.append(dict(train_steps=res.get("optimizer_steps"), train_status=res.get("status"),
                         eval_max_ticks=",".join(str(c) for c in sorted(caps, key=str))))
    if data == "human":
        problems += summary_crosscheck(E)
    if problems and strict:
        raise SystemExit("%s set fails: %s" % (data, "; ".join(problems)))
    return pd.concat([R.reset_index(drop=True), pd.DataFrame(rows)], axis=1), problems


def summary_crosscheck(E) -> List[str]:
    """The counts loaded here equal the ones eval_human printed into eval/summary.txt at the time."""
    path = os.path.join(HUMAN_ROOTS[0], "summary.txt")
    bad, seen = [], set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = SUMMARY_RE.match(line.strip())
            if not m:
                continue
            run = m.group("run")
            seen.add(run)
            for j, exp in ((1, "exp1"), (2, "exp2")):
                g = E[(E.run == run) & (E.exp == exp)]
                got = [int(g.success.sum()), int(g.grasp.sum()), int(g.lift.sum()), int(g.transport.sum()), int(g.place.sum())]
                want = [int(m.group(k + str(j))) for k in ("s", "g", "l", "t", "p")]
                if got != want:
                    bad.append("%s %s: loaded %s, summary.txt %s" % (run, exp, got, want))
    if seen != set(E.run):
        bad.append("summary.txt lists %s" % sorted(seen))
    return bad


# ─── phase tables ─────────────────────────────────────────────────────────────
def _phase_rows(g, keys: dict) -> List[dict]:
    rows, prev = [], len(g)
    for p in CHAIN:
        k, n = int(g["cum_" + p].sum()), len(g)
        lo, hi = AE.wilson(k, n)
        clo, chi = AE.wilson(k, prev) if prev else (float("nan"), float("nan"))
        rows.append(dict(keys, phase=p, official=p in PHASES, n=n, k=k, rate=k / n, ci_lo=lo, ci_hi=hi,
                         marginal_k=int(g[p].sum()), cond_n=prev, cond_k=k,
                         cond_rate=k / prev if prev else float("nan"), cond_ci_lo=clo, cond_ci_hi=chi))
        prev = k
    for p in DIAG:
        k, n = int(g[p].sum()), len(g)
        lo, hi = AE.wilson(k, n)
        rows.append(dict(keys, phase=p, official=False, n=n, k=k, rate=k / n, ci_lo=lo, ci_hi=hi, marginal_k=k,
                         cond_n=float("nan"), cond_k=float("nan"), cond_rate=float("nan"), cond_ci_lo=float("nan"),
                         cond_ci_hi=float("nan")))
    return rows


def phase_by_run(E):
    rows = []
    for (data, model, seed, run, exp), g in E.groupby(["data", "model", "train_seed", "run", "exp"]):
        rows += _phase_rows(g, dict(data=data, model=model, train_seed=int(seed), run=run, exp=exp))
    return _sort(pd.DataFrame(rows))


def phase_pooled(E, by_run):
    rows = []
    for (data, model, exp), g in E.groupby(["data", "model", "exp"]):
        rows += _phase_rows(g, dict(data=data, model=model, exp=exp, n_seeds=g.train_seed.nunique()))
    P = pd.DataFrame(rows)
    stats = []
    for _, r in P.iterrows():
        s = by_run[(by_run.data == r.data) & (by_run.model == r.model) & (by_run.exp == r.exp) & (by_run.phase == r.phase)]
        rates, cond = s.rate.to_numpy(float), s.cond_rate.to_numpy(float)
        cond = cond[~np.isnan(cond)]
        stats.append(dict(mean_rate=rates.mean(), sd_rate=rates.std(ddof=1) if len(rates) > 1 else float("nan"),
                          seed_pct=", ".join("%d" % round(100 * v) for v in rates),
                          cond_mean=cond.mean() if len(cond) else float("nan"),
                          cond_sd=cond.std(ddof=1) if len(cond) > 1 else float("nan"), cond_seeds_defined=len(cond)))
    return _sort(pd.concat([P, pd.DataFrame(stats)], axis=1))


def _sort(df):
    order = {m: i for i, m in enumerate(MODELS)}
    ph = {p: i for i, p in enumerate(CHAIN + DIAG)}
    d = df.copy()
    keys = []
    for col, mp in (("data", {"human": 0, "scripted": 1}), ("model", order), ("train_seed", None), ("exp", None),
                    ("phase", ph)):
        if col in d.columns:
            if mp is None:
                keys.append(col)
            else:
                d["_" + col] = d[col].map(mp)
                keys.append("_" + col)
    d = d.sort_values(keys).reset_index(drop=True)
    return d.drop(columns=[k for k in keys if k.startswith("_")])


# ─── tests ────────────────────────────────────────────────────────────────────
def position_matrix(E, col: str, by: str = "model"):
    """{(exp, value of `by`): counts per position, 0-3}, positions in seed order; checks 3 seeds each."""
    out = {}
    for exp, g in E.groupby("exp"):
        wide = g.pivot_table(index="seed", columns=by, values=col, aggfunc="sum").astype(int)
        n = g.pivot_table(index="seed", columns=by, values="train_seed", aggfunc="count")
        if not (n == 3).all().all():
            raise SystemExit("position counts: not 3 episodes per position for %s" % col)
        for c in wide.columns:
            out[(exp, c)] = wide[c]
    return out


def _holm(rows: List[dict], keys: Sequence[str]):
    for key in keys:
        for r, adj in zip(rows, AE.holm([r[key] for r in rows])):
            r[key + "_holm"] = adj
    return rows


def paired_family(E, cols: Sequence[str], family: str, cumulative: bool = True):
    """Primary test: per box position, counts over 3 seeds; Holm over EVERY row of the family."""
    rows = []
    for col in cols:
        c = ("cum_" + col) if cumulative else col
        mat = position_matrix(E, c)
        for exp in EXPS:
            for a, b in PAIRS:
                xa, xb = mat[(exp, a)], mat[(exp, b)]
                rows.append(dict(family=family, exp=EXP_NAME[exp], phase=col, A=a, B=b, mean_A=float(xa.mean()),
                                 mean_B=float(xb.mean()), tie_both_zero=int(((xa == 0) & (xb == 0)).sum()),
                                 **MFR.paired_test(xa.to_numpy(), xb.loc[xa.index].to_numpy())))
    for key in ("wilcoxon_p", "sign_p"):
        for r, adj in zip(rows, AE.holm([r[key] for r in rows])):
            r[key.replace("_p", "_p_holm")] = adj
    return rows


def fisher_family(E, cols: Sequence[str], family: str, cumulative: bool = True):
    """Secondary test: Fisher's exact on the pooled 2x2, Holm over every row of the family."""
    rows = []
    for col in cols:
        c = ("cum_" + col) if cumulative else col
        for exp in EXPS:
            g = E[E.exp == exp]
            for a, b in PAIRS:
                ga, gb = g[g.model == a][c], g[g.model == b][c]
                k1, n1, k2, n2 = int(ga.sum()), len(ga), int(gb.sum()), len(gb)
                orr, p = AE.fisher(k1, n1, k2, n2)
                rows.append(dict(family=family, assumption=INDEP, exp=EXP_NAME[exp], phase=col, A=a, B=b, A_k=k1, A_n=n1,
                                 B_k=k2, B_n=n2, diff=k1 / n1 - k2 / n2, odds_ratio=orr, p=p))
    return _holm(rows, ["p"])


def gap_table(by_run, pooled):
    rows = []
    for (data, model, phase), g in pooled.groupby(["data", "model", "phase"]):
        e1, e2 = g[g.exp == "exp1"].iloc[0], g[g.exp == "exp2"].iloc[0]
        k1, n1, k2, n2 = int(e1.k), int(e1.n), int(e2.k), int(e2.n)
        p1, p2 = k1 / n1, k2 / n2
        lo, hi = newcombe(k1, n1, k2, n2)
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        s = by_run[(by_run.data == data) & (by_run.model == model) & (by_run.phase == phase)]
        per = (s[s.exp == "exp1"].set_index("train_seed").rate - s[s.exp == "exp2"].set_index("train_seed").rate).sort_index()
        rows.append(dict(data=data, model=model, phase=phase, official=phase in PHASES, exp1_k=k1, exp1_n=n1, exp2_k=k2,
                         exp2_n=n2, exp1_rate=p1, exp2_rate=p2, gap=p1 - p2, newcombe_lo=lo, newcombe_hi=hi,
                         wald_lo=p1 - p2 - AE.Z95 * se, wald_hi=p1 - p2 + AE.Z95 * se, gap_seed_mean=float(per.mean()),
                         se_seeds=float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else float("nan"),
                         gap_per_seed=", ".join("%+.0f" % (100 * v) for v in per), floor=(k1 == 0 and k2 == 0)))
    return _sort(pd.DataFrame(rows))


N_BOOT = 2000
SPLITS = {"human": os.path.join(ROOT, "data", "processed", "splits_human_v2.json"),
          "scripted": os.path.join(ROOT, "data", "processed", "splits_scripted_v3_n40.json")}


def training_sides() -> dict:
    """Training demos per spawn side, from each data set's split file."""
    out = {}
    for data, path in SPLITS.items():
        with open(path, encoding="utf-8") as fh:
            train = json.load(fh)["train"]
        sides = [side_of(AE.spawn_xy(int(s))[1]) for s in train]
        out[data] = {s: sides.count(s) for s in SIDES}
        if len(train) != 40:
            raise SystemExit("%s: %d training seeds, expected 40" % (path, len(train)))
    return out


def lateral_adjusted_gap(E, n_boot: int = N_BOOT, seed: int = 0):
    """EXPLORATORY. The held-out patch sits in the centre-left of the region, and the walk-in rate depends
    strongly on the lateral position. So: Exp 1's rate re-weighted to Exp 2's lateral distribution over the
    8 lateral columns of the position heatmap (AE._grid), minus Exp 2's rate. 95% CI from a cluster
    bootstrap over box positions (each position = its 3 seeds' episodes), resampled within each column for
    Exp 1 and over all positions for Exp 2; the raw gap gets the same bootstrap for comparison."""
    rng = np.random.default_rng(seed)
    _, ye = AE._grid(AE.region())
    rows = []
    for (data, model), g in E.groupby(["data", "model"]):
        e1, e2 = g[g.exp == "exp1"], g[g.exp == "exp2"]
        for p in CHAIN:
            c = "cum_" + p
            pos1 = e1.groupby("seed").agg(k=(c, "sum"), n=(c, "size"), y=("box_y", "first"))
            pos2 = e2.groupby("seed").agg(k=(c, "sum"), n=(c, "size"), y=("box_y", "first"))
            b1 = np.clip(np.digitize(pos1.y, ye) - 1, 0, len(ye) - 2)
            b2 = np.clip(np.digitize(pos2.y, ye) - 1, 0, len(ye) - 2)
            bins = sorted(set(b2))
            w = np.array([(b2 == b).mean() for b in bins])
            missing = [int(b) for b in bins if not (b1 == b).any()]
            k1, n1 = pos1.k.to_numpy(float), pos1.n.to_numpy(float)
            k2, n2 = pos2.k.to_numpy(float), pos2.n.to_numpy(float)

            def adj(i1_by_bin, i2):
                p1 = sum(wb * k1[ib].sum() / n1[ib].sum() for wb, ib in zip(w, i1_by_bin))
                return p1, k2[i2].sum() / n2[i2].sum()
            idx1 = [np.flatnonzero(b1 == b) for b in bins]
            if missing:
                rows.append(dict(data=data, model=model, phase=p, note="no Exp 1 position in lateral column(s) %s" % missing))
                continue
            p1a, p2 = adj(idx1, np.arange(len(k2)))
            raw1 = k1.sum() / n1.sum()
            boot_a, boot_r = np.empty(n_boot), np.empty(n_boot)
            all1 = np.arange(len(k1))
            for i in range(n_boot):
                r1 = [rng.choice(ib, len(ib)) for ib in idx1]
                r2 = rng.integers(0, len(k2), len(k2))
                a1, a2 = adj(r1, r2)
                boot_a[i] = a1 - a2
                rr = rng.choice(all1, len(all1))
                boot_r[i] = k1[rr].sum() / n1[rr].sum() - a2
            lo, hi = np.percentile(boot_a, [2.5, 97.5])
            rlo, rhi = np.percentile(boot_r, [2.5, 97.5])
            rows.append(dict(data=data, model=model, phase=p, official=p in PHASES, exp1_rate=raw1,
                             exp1_rate_lateral_adjusted=p1a, exp2_rate=p2, gap_raw=raw1 - p2, gap_raw_lo=rlo, gap_raw_hi=rhi,
                             gap_adjusted=p1a - p2, gap_adjusted_lo=lo, gap_adjusted_hi=hi,
                             columns=",".join(str(int(b)) for b in bins), weights=",".join("%.3f" % x for x in w),
                             exp1_positions_per_column=",".join(str(len(ib)) for ib in idx1), n_boot=n_boot, note=""))
    return _sort(pd.DataFrame(rows))


# ─── failure phase, spawn side, diagnostics ───────────────────────────────────
FAIL_KEYS = ("success", "WALK_IN", "GRASP", "LIFT", "TRANSPORT", "PLACE", "fell")


def failure_table(E, by: Sequence[str]):
    rows = []
    for key, g in E.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        b = g.first_failure.value_counts()
        rows.append(dict(zip(by, key), n=len(g), **{k: int(b.get(k, 0)) for k in FAIL_KEYS}))
    return _sort(pd.DataFrame(rows))


def side_table(E):
    rows = []
    for exp_key, sub in (("exp1", E[E.exp == "exp1"]), ("exp2", E[E.exp == "exp2"]), ("both", E)):
        for (data, model, side), g in sub.groupby(["data", "model", "side"]):
            row = dict(data=data, model=model, exp=exp_key, side=side, n=len(g))
            for c in ("cum_walk_in", "cum_grasp", "cum_success"):
                k = int(g[c].sum())
                lo, hi = AE.wilson(k, len(g))
                row.update({c[4:]: k / len(g), c[4:] + "_k": k, c[4:] + "_lo": lo, c[4:] + "_hi": hi})
            rows.append(row)
    d = pd.DataFrame(rows)
    d["_s"] = d.side.map({s: i for i, s in enumerate(SIDES)})
    d["_e"] = d.exp.map({"exp1": 0, "exp2": 1, "both": 2})
    d["_m"] = d.model.map({m: i for i, m in enumerate(MODELS)})
    return d.sort_values(["data", "_m", "_e", "_s"]).drop(columns=["_s", "_e", "_m"]).reset_index(drop=True)


def side_tests(E):
    """Exploratory: is the walk-in rate independent of spawn side? Chi-square, 2 x sides, per model and
    experiment; Holm over the models within an experiment."""
    rows = []
    for (data, exp), sub in E.groupby(["data", "exp"]):
        fam = []
        for m in MODELS:
            g = sub[sub.model == m]
            tab = pd.crosstab(g.cum_walk_in, g.side).reindex(index=[True, False], fill_value=0)
            tab = tab[[s for s in SIDES if s in tab.columns]]
            if (tab.sum(axis=1) == 0).any() or tab.shape[1] < 2:
                fam.append(dict(data=data, exp=EXP_NAME[exp], model=m, sides=tab.shape[1], chi2=float("nan"), dof=0,
                                p=float("nan"), min_expected=float("nan"), note="constant: no test"))
                continue
            chi2, p, dof, ex = chi2_contingency(tab.to_numpy())
            fam.append(dict(data=data, exp=EXP_NAME[exp], model=m, sides=tab.shape[1], chi2=float(chi2), dof=int(dof),
                            p=float(p), min_expected=float(ex.min()), note=""))
        rows += _holm(fam, ["p"])
    return pd.DataFrame(rows)


def diagnostics(E):
    rows = []
    for (data, model, exp), g in E.groupby(["data", "model", "exp"]):
        gr = g[g.cum_grasp]
        welded = g.weld_engage_tick >= 0
        row = dict(data=data, model=model, exp=exp, n=len(g), grasped=len(gr), welded=int(welded.sum()),
                   welded_without_pickup_lock=int((welded & ~g.walk_in).sum()))
        for name, k, n in (("goal_lock", int(g.goal_lock.sum()), len(g)), ("release", int(g.release.sum()), len(g)),
                           ("grasped_goal_lock", int(gr.goal_lock.sum()), len(gr)),
                           ("grasped_released", int(gr.release.sum()), len(gr)),
                           ("grasped_placed", int(gr.cum_place.sum()), len(gr))):
            lo, hi = AE.wilson(k, n) if n else (float("nan"), float("nan"))
            row.update({name + "_k": k, name + "_n": n, name: k / n if n else float("nan"), name + "_lo": lo,
                        name + "_hi": hi})
        rows.append(row)
    return _sort(pd.DataFrame(rows))


# ─── human vs scripted ────────────────────────────────────────────────────────
def hs_fisher(EA, cols, family, cumulative=True):
    rows = []
    for col in cols:
        c = ("cum_" + col) if cumulative else col
        for exp in EXPS:
            for m in MODELS:
                h = EA[(EA.data == "human") & (EA.model == m) & (EA.exp == exp)][c]
                s = EA[(EA.data == "scripted") & (EA.model == m) & (EA.exp == exp)][c]
                orr, p = AE.fisher(int(h.sum()), len(h), int(s.sum()), len(s))
                rows.append(dict(family=family, assumption=INDEP, exp=EXP_NAME[exp], phase=col, model=m, human_k=int(h.sum()),
                                 human_n=len(h), scripted_k=int(s.sum()), scripted_n=len(s),
                                 diff=h.mean() - s.mean(), odds_ratio=orr, p=p))
    return _holm(rows, ["p"])


def hs_paired(EA, cols, family, cumulative=True):
    """Human vs scripted for one model, paired by box position (both sets ran the same seeds)."""
    rows = []
    for col in cols:
        c = ("cum_" + col) if cumulative else col
        for m in MODELS:
            mat = position_matrix(EA[EA.model == m], c, by="data")
            for exp in EXPS:
                h, s = mat[(exp, "human")], mat[(exp, "scripted")]
                rows.append(dict(family=family, exp=EXP_NAME[exp], phase=col, model=m, mean_human=float(h.mean()),
                                 mean_scripted=float(s.mean()), **MFR.paired_test(h.to_numpy(), s.loc[h.index].to_numpy())))
    for key in ("wilcoxon_p", "sign_p"):
        for r, adj in zip(rows, AE.holm([r[key] for r in rows])):
            r[key.replace("_p", "_p_holm")] = adj
    return rows


def hs_proximity(EA):
    rows = []
    for metric, short, _ in MFR.PROXIMITY:
        for exp in EXPS:
            fam = []
            for m in MODELS:
                h = EA[(EA.data == "human") & (EA.model == m) & (EA.exp == exp)][metric].dropna().to_numpy(float)
                s = EA[(EA.data == "scripted") & (EA.model == m) & (EA.exp == exp)][metric].dropna().to_numpy(float)
                U, p = mannwhitneyu(h, s, alternative="two-sided")
                fam.append(dict(assumption=INDEP, metric=short, exp=EXP_NAME[exp], model=m, median_human=float(np.median(h)),
                                median_scripted=float(np.median(s)), n="%d, %d" % (len(h), len(s)), U=float(U), p=float(p)))
            rows += _holm(fam, ["p"])
    return pd.DataFrame(rows)


# ─── teleoperation ────────────────────────────────────────────────────────────
def teleop_tables(folder: str):
    """The 50 human demos: tools/teleop_metrics.py's tables, plus a learning-trend test on its attempts."""
    att = pd.read_csv(os.path.join(folder, "attempts.csv"))
    ses = pd.read_csv(os.path.join(folder, "sessions.csv"))
    ops = pd.read_csv(os.path.join(folder, "operators.csv"))
    att["session_i"] = att.session.map(lambda s: int(str(s).split()[0]))
    ses_rows = ses[ses.session != "ALL"].copy()
    ses_rows["session_i"] = ses_rows.session.map(lambda s: int(str(s).split()[0]))
    ses_rows["date"] = ses_rows.session.map(lambda s: str(s).split()[1][:10])
    ci = [AE.wilson(int(a), int(n)) for a, n in zip(ses_rows.accepted, ses_rows.attempts)]
    ses_rows["ci_lo"], ses_rows["ci_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    rho, p = spearmanr(att.k, att.success)
    half = len(att) // 2
    a1, a2 = att.iloc[:half], att.iloc[half:]
    orr, pf = AE.fisher(int(a2.success.sum()), len(a2), int(a1.success.sum()), len(a1))
    rho_s, p_s = spearmanr(ses_rows.session_i, ses_rows.success_rate)
    trend = dict(spearman_attempt_rho=float(rho), spearman_attempt_p=float(p), first_half="%d/%d" % (a1.success.sum(), len(a1)),
                 second_half="%d/%d" % (a2.success.sum(), len(a2)), halves_fisher_p=pf, halves_odds_ratio=orr,
                 spearman_session_rho=float(rho_s), spearman_session_p=float(p_s))
    allrow = ops[ops.operator != "ALL"].iloc[0] if (ops.operator != "ALL").sum() == 1 else ops[ops.operator == "ALL"].iloc[0]
    return dict(attempts=att, sessions=ses_rows, all=ses[ses.session == "ALL"].iloc[0], operator=allrow, trend=trend)


# ─── figures (300 dpi PNG + SVG; colour follows the model; values printed) ────
MODEL_COL = dict(zip(MODELS, RIO.SERIES))
FAIL_COL = MFR.FAIL_COL
EXP_TITLE = {"exp1": "Exp 1: positions in the collection region", "exp2": "Exp 2: the held-out patch"}


def _plt():
    plt = RIO.pyplot()
    plt.rcParams["svg.fonttype"] = "none"
    return plt


def _save(fig, out, stem):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, "%s.%s" % (stem, ext)), dpi=300, bbox_inches="tight")
    RIO.pyplot().close(fig)
    return stem + ".png"


def _bars(ax, pooled, by_run, exp, data, conditional=False):
    xs = np.arange(len(CHAIN))
    w = 0.26
    for j, m in enumerate(MODELS):
        for i, p in enumerate(CHAIN):
            r = pooled[(pooled.data == data) & (pooled.model == m) & (pooled.exp == exp) & (pooled.phase == p)].iloc[0]
            v, lo, hi = (r.cond_rate, r.cond_ci_lo, r.cond_ci_hi) if conditional else (r.rate, r.ci_lo, r.ci_hi)
            x = xs[i] + (j - 1) * (w + 0.02)
            if v != v:                                              # 0 of 0: undefined, drawn as nothing
                ax.text(x, 2, "n/a", ha="center", fontsize=5.5, color=RIO.INK_2, rotation=90)
                continue
            ax.bar(x, 100 * v, w, color=MODEL_COL[m], zorder=2, label=m if i == 0 else None)
            ax.errorbar(x, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], color=RIO.INK, capsize=1.8, lw=0.8, zorder=3)
            top = hi
            if not conditional:
                pts = by_run[(by_run.data == data) & (by_run.model == m) & (by_run.exp == exp) & (by_run.phase == p)].rate
                ax.scatter(np.full(len(pts), x), 100 * pts, s=9, facecolors=RIO.SURFACE, edgecolors=RIO.INK,
                           linewidths=0.6, zorder=4)
                top = max(hi, float(pts.max()))
            txt = "%.0f" % (100 * v) if not conditional else "%d/%d" % (r.cond_k, r.cond_n)
            ax.text(x, 100 * top + 2.5, txt, ha="center", va="bottom", fontsize=5.5, color=RIO.INK_2,
                    rotation=90 if conditional else 0)
    ax.axvspan(len(PHASES) - 0.5, len(CHAIN) - 0.5, color=RIO.GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([("%d %s" % (i + 1, LABEL[p])) if p in PHASES else LABEL[p] for i, p in enumerate(CHAIN)], fontsize=8)
    ax.set_ylim(0, 118 if conditional else 112)
    ax.grid(axis="x", visible=False)


def fig_phases(pooled, by_run, out, data="human", conditional=False):
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharey=True)
    for ax, exp in zip(axes, EXPS):
        _bars(ax, pooled, by_run, exp, data, conditional)
        ax.set_title(EXP_TITLE[exp], fontsize=10, loc="left")
        ax.set_ylabel(("phase k given phase k-1 (%)" if conditional else "reached phase (%)"), fontsize=9)
    axes[0].legend(fontsize=8.5, loc="upper right", ncol=3)
    what = ("Conditional phase rates: phase k given phases 1..k-1, Wilson 95% CI; label = k/n"
            if conditional else "Cumulative phase rates: reached phases 1..k, pooled over 3 seeds, Wilson 95% CI; dots = one seed")
    fig.suptitle("%s (%s data, 40 demos). Shaded: full success, beside the five phases." % (what, data), x=0.06, ha="left",
                 fontsize=9.5)
    fig.tight_layout()
    return _save(fig, out, "fig_phase_%s%s" % ("conditional" if conditional else "cumulative",
                                               "" if data == "human" else "_" + data))


def fig_gap(gap, out, data="human", lat=None):
    plt = _plt()
    g = gap[gap.data == data]
    if lat is not None:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.12))
    else:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax2 = None
    ys = np.arange(len(CHAIN))[::-1]
    if ax2 is not None:
        lg = lat[lat.data == data]
        for j, m in enumerate(MODELS):
            for y, p in zip(ys, CHAIN):
                r = lg[(lg.model == m) & (lg.phase == p)].iloc[0]
                yy = y + (1 - j) * 0.24
                ax2.plot([100 * r.gap_adjusted_lo, 100 * r.gap_adjusted_hi], [yy, yy], color=MODEL_COL[m], lw=1.6, zorder=2)
                ax2.scatter([100 * r.gap_adjusted], [yy], s=40, marker="D", color=MODEL_COL[m], edgecolors=RIO.SURFACE,
                            linewidths=1.2, zorder=3)
                ax2.text(1.01, yy, "%+.1f" % (100 * r.gap_adjusted), va="center", fontsize=6.5, color=RIO.INK_2,
                         transform=ax2.get_yaxis_transform())
        ax2.axvline(0, color=RIO.INK, lw=1)
        ax2.grid(axis="y", visible=False)
        ax2.set_xlabel("Exp 1 re-weighted to Exp 2's lateral mix, minus Exp 2, points (position bootstrap 95% CI)",
                       fontsize=8.5)
        ax2.set_title("Exploratory: lateral-adjusted gap", fontsize=9.5, loc="left")
    for j, m in enumerate(MODELS):
        for y, p in zip(ys, CHAIN):
            r = g[(g.model == m) & (g.phase == p)].iloc[0]
            yy = y + (1 - j) * 0.24
            ax.plot([100 * r.newcombe_lo, 100 * r.newcombe_hi], [yy, yy], color=MODEL_COL[m], lw=1.6, zorder=2)
            ax.scatter([100 * r.gap], [yy], s=40, color=RIO.SURFACE if r.floor else MODEL_COL[m], edgecolors=MODEL_COL[m],
                       linewidths=1.5, zorder=3, label=m if p == CHAIN[0] else None)
            ax.text(1.01, yy, "0 vs 0" if r.floor else "%+.1f" % (100 * r.gap), va="center", fontsize=6.5,
                    color=RIO.INK_2, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=RIO.INK, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([("%d %s" % (i + 1, LABEL[p])) if p in PHASES else LABEL[p] for i, p in enumerate(CHAIN)], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("generalisation gap: Exp 1 minus Exp 2, cumulative rate, points (Newcombe 95% CI)", fontsize=9)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Raw gap", fontsize=9.5, loc="left")
    fig.suptitle("Generalisation gap per phase (%s data). Above 0: worse on the held-out patch. Hollow: 0 in both "
                 "experiments, a floor, not a gap." % data, x=0.06, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig_generalization_gap" + ("" if data == "human" else "_" + data))


def fig_failures(fail, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    items = [(m, e) for m in MODELS for e in EXPS]
    ys = np.array([-(i + 0.5 * MODELS.index(m)) for i, (m, _e) in enumerate(items)])   # a gap between models
    for y, (m, e) in zip(ys, items):
        r = fail[(fail.data == "human") & (fail.model == m) & (fail.exp == e)].iloc[0]
        left = 0.0
        for k in FAIL_KEYS:
            v = 100.0 * r[k] / r.n
            if v <= 0:
                continue
            ax.barh(y, v, 0.72, left=left, color=FAIL_COL[k], edgecolor=RIO.SURFACE, linewidth=1.5, zorder=2)
            if v >= 4:
                ax.text(left + v / 2, y, "%.0f" % v, ha="center", va="center", fontsize=7.5,
                        color="white" if k in ("success", "PLACE", "fell") else RIO.INK)
            left += v
    ax.set_yticks(ys)
    ax.set_yticklabels(["%s, %s" % (m, EXP_NAME[e]) for m, e in items], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("episodes (%), pooled over 3 seeds", fontsize=9)
    ax.grid(axis="y", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, color=FAIL_COL[k]) for k in FAIL_KEYS]
    ax.legend(h, ["full success"] + ["stopped before %s" % k.lower().replace("_", "-") for k in FAIL_KEYS[1:-1]] + ["fell"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8)
    ax.set_title("The first of the five phases each episode did not reach (human data; labelled when >= 4%)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, out, "fig_failure_phase")


def fig_failures_by_side(fail_side, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    ys = np.arange(len(SIDES))[::-1]
    for ax, m in zip(axes, MODELS):
        for y, s in zip(ys, SIDES):
            r = fail_side[(fail_side.data == "human") & (fail_side.model == m) & (fail_side.side == s)].iloc[0]
            left = 0.0
            for k in FAIL_KEYS:
                v = 100.0 * r[k] / r.n
                if v <= 0:
                    continue
                ax.barh(y, v, 0.7, left=left, color=FAIL_COL[k], edgecolor=RIO.SURFACE, linewidth=1.5, zorder=2)
                if v >= 6:
                    ax.text(left + v / 2, y, "%.0f" % v, ha="center", va="center", fontsize=7,
                            color="white" if k in ("success", "PLACE", "fell") else RIO.INK)
                left += v
            ax.text(101, y, "n=%d" % r.n, va="center", fontsize=7, color=RIO.INK_2)
        ax.set_title(m, fontsize=10, loc="left")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("episodes (%), Exp 1 + Exp 2", fontsize=8.5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(["%s spawn" % s for s in SIDES], fontsize=9)
    h = [plt.Rectangle((0, 0), 1, 1, color=FAIL_COL[k]) for k in FAIL_KEYS]
    fig.legend(h, ["full success"] + ["stopped before %s" % k.lower().replace("_", "-") for k in FAIL_KEYS[1:-1]] + ["fell"],
               loc="lower center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("First phase not reached, by spawn side (robot's view; human data, pooled over seeds and experiments)",
                 x=0.06, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig_failure_by_side")


def fig_lock_by_side(sides, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    xs = np.arange(len(SIDES))
    w = 0.26
    for ax, ek in zip(axes, ("exp1", "exp2", "both")):
        g = sides[(sides.data == "human") & (sides.exp == ek)]
        for j, m in enumerate(MODELS):
            for i, s in enumerate(SIDES):
                r = g[(g.model == m) & (g.side == s)]
                x = xs[i] + (j - 1) * (w + 0.02)
                if not len(r):
                    if j == 1:
                        ax.text(xs[i], 3, "no positions", ha="center", fontsize=7, color=RIO.INK_2)
                    continue
                r = r.iloc[0]
                ax.bar(x, 100 * r.walk_in, w, color=MODEL_COL[m], zorder=2, label=m if i == 1 and ek == "exp1" else None)
                ax.errorbar(x, 100 * r.walk_in, yerr=[[100 * (r.walk_in - r.walk_in_lo)], [100 * (r.walk_in_hi - r.walk_in)]],
                            color=RIO.INK, capsize=1.8, lw=0.8, zorder=3)
                ax.text(x, 100 * r.walk_in_hi + 2, "%d/%d" % (r.walk_in_k, r.n), ha="center", fontsize=5.5, color=RIO.INK_2,
                        rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels(SIDES)
        ax.set_xlim(-0.55, len(SIDES) - 0.45)
        ax.set_ylim(0, 125)
        ax.grid(axis="x", visible=False)
        ax.set_title({"exp1": "Exp 1", "exp2": "Exp 2 (the patch has no right third)", "both": "Exp 1 + Exp 2"}[ek],
                     fontsize=9.5, loc="left")
    axes[0].set_ylabel("walk-in: pickup lock reached (%)", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left", ncol=3)
    fig.suptitle("Walk-in (pickup lock) rate by spawn side, robot's view; Wilson 95% CI; pooled over 3 seeds (human data)",
                 x=0.06, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig_lock_by_side")


def fig_position(E, out):
    import matplotlib.colors as mc
    plt = _plt()
    reg = AE.region()
    xe, ye = AE._grid(reg)
    cmap = mc.LinearSegmentedColormap.from_list("rate", ["#f4f3ef", "#a8cdf2", "#2a78d6", "#123f7a"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.6))
    fig.subplots_adjust(hspace=0.45)
    im = None
    for row, (col, title) in zip(axes, (("cum_success", "full success"), ("cum_walk_in", "walk-in (pickup lock)"))):
        for ax, m in zip(row, MODELS):
            g = E[(E.data == "human") & (E.model == m)]
            tot, _, _ = np.histogram2d(g.box_x, g.box_y, bins=[xe, ye])
            ok, _, _ = np.histogram2d(g.box_x[g[col]], g.box_y[g[col]], bins=[xe, ye])
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(tot > 0, ok / tot, np.nan)
            im = ax.pcolormesh(ye, xe, rate, cmap=cmap, vmin=0, vmax=1, shading="flat", edgecolors=RIO.SURFACE, linewidth=0.8)
            for a in range(len(xe) - 1):
                for b in range(len(ye) - 1):
                    if tot[a, b]:
                        ax.text((ye[b] + ye[b + 1]) / 2, (xe[a] + xe[a + 1]) / 2, "%d/%d" % (ok[a, b], tot[a, b]), ha="center",
                                va="center", fontsize=5.5, color="white" if rate[a, b] > 0.45 else RIO.INK)
            AE._patch(ax, reg)
            for yb in np.linspace(*reg["y"], 4)[1:-1]:
                ax.axvline(yb, color=RIO.INK_2, lw=0.6, ls=":")
            ax.set_xlim(*reg["y"])
            ax.set_ylim(*reg["x"])
            ax.invert_xaxis()                                       # robot's view: its left (+y) on the left
            ax.grid(False)
            ax.tick_params(labelsize=7)
            ax.set_title("%s: %s" % (m, title), fontsize=9, loc="left")
        row[0].set_ylabel("box x, depth (m)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("box y (m); left of the robot <- -> right", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.015, label="rate in the cell")
    fig.suptitle("By box position, Exp 1 + Exp 2, 3 seeds pooled (human data). Cell: k/episodes. Dashed: held-out patch; "
                 "dotted: the side thirds.", x=0.06, ha="left", fontsize=10)
    return _save(fig, out, "fig_success_by_position")


def fig_human_vs_scripted(pooled, out):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.6), sharey=True)
    xs = np.arange(len(CHAIN))
    w = 0.38
    for row, exp in zip(axes, EXPS):
        for ax, m in zip(row, MODELS):
            for j, data in enumerate(("human", "scripted")):
                g = pooled[(pooled.data == data) & (pooled.model == m) & (pooled.exp == exp)].set_index("phase").loc[list(CHAIN)]
                x = xs + (j - 0.5) * (w + 0.03)
                kw = dict(color=MODEL_COL[m]) if data == "human" else dict(facecolor=RIO.SURFACE, edgecolor=MODEL_COL[m],
                                                                           hatch="////", linewidth=0.8)
                ax.bar(x, 100 * g.rate, w, zorder=2, label=data, **kw)
                ax.errorbar(x, 100 * g.rate, yerr=[100 * (g.rate - g.ci_lo), 100 * (g.ci_hi - g.rate)], fmt="none",
                            ecolor=RIO.INK, capsize=1.5, lw=0.7, zorder=3)
                for xx, v, hi in zip(x, g.rate, g.ci_hi):
                    ax.text(xx, 100 * hi + 2, "%.0f" % (100 * v), ha="center", fontsize=5.5, color=RIO.INK_2)
            ax.set_xticks(xs)
            ax.set_xticklabels([LABEL[p].replace("full ", "") for p in CHAIN], fontsize=7, rotation=35, ha="right")
            ax.set_ylim(0, 115)
            ax.grid(axis="x", visible=False)
            ax.set_title("%s, %s" % (m, EXP_NAME[exp]), fontsize=9.5, loc="left")
            ax.legend(fontsize=7.5, loc="upper right")
        row[0].set_ylabel("reached phase (%)", fontsize=9)
    fig.suptitle("Human (solid) vs scripted Run A (hatched), 40 demos each, 3 seeds pooled; cumulative phase rates, Wilson 95% CI",
                 x=0.06, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig_human_vs_scripted")


def fig_teleop(T, out):
    plt = _plt()
    att, ses = T["attempts"], T["sessions"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    roll = att.success.rolling(20, min_periods=20).mean()
    ax.plot(att.k, 100 * att.cum_success_rate, color=RIO.SERIES[0], lw=2, label="cumulative acceptance")
    ax.plot(att.k, 100 * roll, color=RIO.SERIES[1], lw=1.6, label="rolling, last 20 attempts")
    for k in att.groupby("session_i").k.min().iloc[1:]:
        ax.axvline(k - 0.5, color=RIO.GRID, lw=1, zorder=0)
    ax.scatter(att.k[att.success == 1], np.full(int(att.success.sum()), -3), marker="|", s=30, color=RIO.INK_2,
               label="an accepted demo")
    ax.set_xlabel("attempt (vertical lines: session starts)", fontsize=9)
    ax.set_ylabel("accepted (%)", fontsize=9)
    ax.set_ylim(-6, 60)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Acceptance over %d attempts" % len(att), fontsize=10, loc="left")
    ax = axes[1]
    x = ses.session_i.to_numpy()
    r = ses.success_rate.to_numpy(float)
    ax.bar(x, 100 * r, 0.7, color=RIO.SERIES[0], zorder=2)
    ax.errorbar(x, 100 * r, yerr=[100 * (r - ses.ci_lo), 100 * (ses.ci_hi - r)], fmt="none", ecolor=RIO.INK, capsize=2,
                lw=0.8, zorder=3)
    for xx, a, n, hi in zip(x, ses.accepted, ses.attempts, ses.ci_hi):
        ax.text(xx, 100 * hi + 2, "%d/%d" % (a, n), ha="center", fontsize=6.5, color=RIO.INK_2, rotation=90)
    ax.set_xticks(x)
    ax.set_xlabel("session", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.set_title("Per session, Wilson 95% CI", fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, out, "fig_teleop_learning")


# ─── report ───────────────────────────────────────────────────────────────────
md, pv, pct = MFR.md, MFR.pv, MFR.pct


def ci(v, lo, hi, nd=1):
    if v != v:
        return "n/a"
    return "%.*f [%.*f, %.*f]" % (nd, 100 * v, nd, 100 * lo, nd, 100 * hi)


def msd(m, s):
    return "n/a" if m != m else ("%.1f +- %s" % (100 * m, "n/a" if s != s else "%.1f" % (100 * s)))


def _pp(pooled, data, model, exp, phase):
    return pooled[(pooled.data == data) & (pooled.model == model) & (pooled.exp == exp) & (pooled.phase == phase)].iloc[0]


def phase_matrix(pooled, data, cell, phases=CHAIN):
    rows = []
    for m in MODELS:
        for e in EXPS:
            rows.append([m, EXP_NAME[e]] + [cell(_pp(pooled, data, m, e, p)) for p in phases])
    return md(rows, ["model", "exp"] + [("%d %s" % (i + 1, LABEL[p])) if p in PHASES else LABEL[p] for i, p in enumerate(phases)])


def run_matrix(by_run, data):
    rows = []
    for (m, s, e), g in by_run[by_run.data == data].groupby(["model", "train_seed", "exp"], sort=False):
        g = g.set_index("phase")
        rows.append([m, int(s), EXP_NAME[e], g.loc["walk_in", "run"]] +
                    ["%d %s" % (g.loc[p, "k"], "[%.0f, %.0f]" % (100 * g.loc[p, "ci_lo"], 100 * g.loc[p, "ci_hi"])) for p in CHAIN])
    return md(rows, ["model", "seed", "exp", "run"] + [LABEL[p] for p in CHAIN])


def paired_rows(P):
    return md([[r.exp, LABEL[r.phase], "%s vs %s" % (r.A, r.B), "%.2f vs %.2f" % (r.mean_A, r.mean_B), r.favour_a, r.favour_b,
                "%d (%d both 0)" % (r.tie, r.tie_both_zero), pv(r.wilcoxon_p), pv(r.wilcoxon_p_holm), "%+.2f" % r.rank_biserial,
                pv(r.sign_p), pv(r.sign_p_holm)] for _, r in P.iterrows()],
              ["exp", "phase", "A vs B", "mean seeds reaching it, per position (of 3)", "favour A", "favour B", "tie",
               "Wilcoxon p", "Holm", "rank-biserial r", "sign p", "Holm"])


def fisher_rows(F):
    return md([[r.exp, LABEL[r.phase], "%s vs %s" % (r.A, r.B), "%d/%d vs %d/%d" % (r.A_k, r.A_n, r.B_k, r.B_n),
                "%+.1f" % (100 * r["diff"]), "n/a" if (r.odds_ratio != r.odds_ratio or math.isinf(r.odds_ratio))
                else "%.3g" % r.odds_ratio, pv(r.p), pv(r.p_holm)] for _, r in F.iterrows()],
              ["exp", "phase", "A vs B", "reached", "diff (pts)", "odds ratio", "p", "p (Holm)"])


def gap_rows(G, data):
    g = G[G.data == data]
    return md([[r.model, LABEL[r.phase], "%d/%d" % (r.exp1_k, r.exp1_n), "%d/%d" % (r.exp2_k, r.exp2_n),
                "%+.1f" % (100 * r.gap), "%+.1f to %+.1f" % (100 * r.newcombe_lo, 100 * r.newcombe_hi),
                "%+.1f to %+.1f" % (100 * r.wald_lo, 100 * r.wald_hi),
                "n/a" if r.se_seeds != r.se_seeds else "%.1f" % (100 * r.se_seeds), r.gap_per_seed,
                "floor (0 vs 0)" if r.floor else ""] for _, r in g.iterrows() if r.phase in CHAIN],
              ["model", "phase", "Exp 1", "Exp 2", "gap (pts)", "Newcombe 95% CI", "Wald 95% CI", "SE over seeds",
               "gap per seed (pts)", ""])


def key_results(T) -> List[str]:
    """The headline numbers, computed from the tables, not typed."""
    pooled, P, G, D, S = T["pooled"], T["paired"], T["gap"], T["diag"], T["sides"]

    def r(m, e, p, data="human"):
        return _pp(pooled, data, m, e, p)

    def prow(a, b, e, p):
        return P[(P.A == a) & (P.B == b) & (P.exp == EXP_NAME[e]) & (P.phase == p)].iloc[0]

    def cell(m, e, p, data="human"):
        x = r(m, e, p, data)
        return "%d/%d (%.1f%%)" % (x.k, x.n, 100 * x.rate)

    L = []
    L.append("- **Full success, pooled over 3 seeds (Exp 1 | Exp 2):** " + "; ".join(
        "%s %s | %s" % (m, cell(m, "exp1", "success"), cell(m, "exp2", "success")) for m in MODELS) +
        ". Per seed, Exp 1: " + "; ".join("%s %s%%" % (m, r(m, "exp1", "success").seed_pct) for m in MODELS) + ".")
    L.append("- **Walk-in (phase 1), pickup lock reached, Exp 1 | Exp 2:** " + "; ".join(
        "%s %s | %s" % (m, cell(m, "exp1", "walk_in"), cell(m, "exp2", "walk_in")) for m in MODELS) + ".")
    F = T["fail"][T["fail"].data == "human"]
    loss = []
    for _, x in F.iterrows():
        k = max(("WALK_IN",) + tuple(p.upper() for p in PHASES[1:]), key=lambda c: x[c])
        loss.append("%s %s: %s (%d of %d)" % (x.model, EXP_NAME[x.exp], k.lower().replace("_", "-"), x[k], x.n))
    L.append("- **Largest single loss (the first phase not reached), per model and experiment:** " + "; ".join(loss) + ".")
    L.append("- **Given the walk-in, grasp (phase 2 | 1), Exp 1:** " + "; ".join(
        "%s %d/%d" % (m, r(m, "exp1", "grasp").cond_k, r(m, "exp1", "grasp").cond_n) for m in MODELS) +
        "; **place given transport, Exp 1:** " + "; ".join(
        "%s %d/%d" % (m, r(m, "exp1", "place").cond_k, r(m, "exp1", "place").cond_n) for m in MODELS) + ".")
    for a, b in PAIRS:
        parts = []
        for e in EXPS:
            sig = [LABEL[p] for p in PHASES if prow(a, b, e, p).wilcoxon_p_holm < 0.05]
            parts.append("%s: Holm-significant (Wilcoxon) in %s" % (EXP_NAME[e], ", ".join(sig) if sig else "no phase"))
        PS = T["paired_success"]
        s1 = PS[(PS.A == a) & (PS.B == b) & (PS.exp == "Exp 1")].iloc[0]
        L.append("- **%s vs %s, position-level (primary):** %s. Full success Exp 1: %d positions favour %s, %d favour %s, "
                 "Wilcoxon Holm p %s." % (a, b, "; ".join(parts), s1.favour_a, a, s1.favour_b, b, pv(s1.wilcoxon_p_holm)))
    gs = G[(G.data == "human") & (G.phase == "success")].set_index("model")
    L.append("- **Generalisation gap, full success (Exp 1 - Exp 2, points, Newcombe 95% CI):** " + "; ".join(
        "%s %s" % (m, "0 vs 0 (floor)" if gs.loc[m, "floor"] else "%+.1f [%+.1f, %+.1f]" % (
            100 * gs.loc[m, "gap"], 100 * gs.loc[m, "newcombe_lo"], 100 * gs.loc[m, "newcombe_hi"])) for m in MODELS) + ".")
    gw = G[(G.data == "human") & (G.phase == "walk_in")].set_index("model")
    L.append("- **Walk-in gap:** " + "; ".join("%s %+.1f [%+.1f, %+.1f]" % (m, 100 * gw.loc[m, "gap"], 100 * gw.loc[m, "newcombe_lo"],
                                                                           100 * gw.loc[m, "newcombe_hi"]) for m in MODELS) + ".")
    LA = T["lateral"]
    la = LA[LA.data == "human"]

    def lat(m, p):
        x = la[(la.model == m) & (la.phase == p)].iloc[0]
        return "%+.1f [%+.1f, %+.1f]" % (100 * x.gap_adjusted, 100 * x.gap_adjusted_lo, 100 * x.gap_adjusted_hi)
    L.append("- **Exploratory, lateral-adjusted gap** (Exp 1 re-weighted to the patch's lateral position; the walk-in depends "
             "strongly on side, below): walk-in " + "; ".join("%s %s" % (m, lat(m, "walk_in")) for m in MODELS) +
             "; full success " + "; ".join("%s %s" % (m, lat(m, "success")) for m in MODELS) + ".")
    sb = S[(S.data == "human") & (S.exp == "exp1")]
    L.append("- **Walk-in by spawn side, Exp 1 (right / centre / left):** " + "; ".join(
        "%s %s" % (m, " / ".join("%d/%d" % (x.walk_in_k, x.n) for _, x in sb[sb.model == m].iterrows())) for m in MODELS) + ".")
    dd = D[D.data == "human"]
    L.append("- **Diagnostics, of grasped boxes (Exp 1):** " + "; ".join(
        "%s %d grasped: goal lock %d, released %d, placed %d" % (
            m, x.grasped, x.grasped_goal_lock_k, x.grasped_released_k, x.grasped_placed_k)
        for m in MODELS for _, x in dd[(dd.model == m) & (dd.exp == "exp1")].iterrows()) +
        ". Welds with no pickup lock (the base free at the weld), Exp 1 + Exp 2: " + "; ".join(
        "%s %d" % (m, int(dd[dd.model == m].welded_without_pickup_lock.sum())) for m in MODELS) + ".")
    L.append("- **Human vs scripted, full success Exp 1 | Exp 2 (human vs scripted Run A):** " + "; ".join(
        "%s %d vs %d | %d vs %d" % (m, r(m, "exp1", "success").k, r(m, "exp1", "success", "scripted").k,
                                    r(m, "exp2", "success").k, r(m, "exp2", "success", "scripted").k) for m in MODELS) +
        " (of 300 each).")
    return L


def write_report(T, figs, out) -> str:
    R, pooled, by_run = T["runs"], T["pooled"], T["by_run"]
    tele = T["teleop"]
    L = ["# Human_v2 final analysis: Experiments 1 and 2", ""]
    L.append("**The locked human_v2 study** (CLAUDE.md section 8, 2026-11-20): 40 human demonstrations (50 accepted, split 40/10), "
             "walking labels smoothed (`centred_ma13_v1`), raw 48-D state. **9 runs, 1800 episodes** (3 models x seeds 0-2 x "
             "100 Exp 1 + 100 Exp 2). Thesis names per CLAUDE.md D8 (**pending adviser approval**): BC = `bc_b256`, "
             "ACT = `act_orig`, ACT-LSTM = `act_lstm_full`. Evaluation cap 1568 ticks, lock rule v3, policy walk-in.")
    L += ["", "Regenerate with `python tools/make_human_report.py` (read-only; inputs `runs/human_v2/eval`, scripted Run A in "
          "`runs/scripted_v3/eval` + `runs/scripted_v3_memory_design/eval`, teleop tables in `teleop/`). The counts loaded "
          "here were checked against `runs/human_v2/eval/summary.txt` and every run's own `metadata.json`.", ""]
    L += ["**Phase definition (Charles, 2026-11-22): FIVE official phases.** 1 walk-in = the pickup base lock engaged (lock rule "
          "v3; a lock engaged while the box was not welded, before the weld). 2 grasp, 3 lift, 4 transport, 5 place: `success.py`, "
          "unchanged. **Cumulative** rate = reached phases 1..k; **conditional** = phase k given phases 1..k-1. Full success "
          "is shown beside them (it equals place on every episode). Goal lock and release are **diagnostics**, not phases "
          "(section 5).", "",
          "**Tests.** Primary: the position-level paired test (per box position, seeds 0-3 reaching the phase; Wilcoxon "
          "signed-rank and sign test). Secondary: Fisher's exact test on pooled counts. **Holm family for the phases = all "
          "5 phases x 2 experiments x 3 pairs = 30 tests** (per test type); full success is its own family of 6; the "
          "diagnostics another of 12 (exploratory).", ""]
    L += ["## Key results", ""] + key_results(T) + [""]
    L += ["## 0. Which run is which", "", "Model, chunk, window, batch and 40 training demos read back from each run's "
          "`metadata.json`; checkpoint chosen on 20 selection seeds (`tools/eval_human.ps1` / `eval_scripted.py`).", ""]
    L.append(md([[r.data, r.model, MFR.EXPECT[r.model][0], int(r.train_seed), r.run, r.chosen_ckpt,
                  "%d/%d" % (r.selection_success, r.selection_n), "{:,}".format(int(r.train_steps)), r.train_status,
                  r.eval_max_ticks] for _, r in T["runs"].iterrows()],
                ["data", "model", "code", "seed", "run", "chosen", "at selection", "optimizer steps", "train status", "eval cap"]))
    L += ["## 1. Phase-level success (five official phases)", "",
          "### 1a. Cumulative rate, pooled over 3 seeds: % [Wilson 95% CI], n = 300 per cell", ""]
    L.append(phase_matrix(pooled, "human", lambda x: ci(x.rate, x.ci_lo, x.ci_hi)))
    L += ["### 1b. Cumulative rate, mean +- SD over the 3 seeds (%)", ""]
    L.append(phase_matrix(pooled, "human", lambda x: msd(x.mean_rate, x.sd_rate)))
    L += ["### 1c. Conditional rate, phase k given phases 1..k-1, pooled: k/n, % [Wilson 95% CI]", ""]
    L.append(phase_matrix(pooled, "human", lambda x: "%d/%d, %s" % (x.cond_k, x.cond_n, ci(x.cond_rate, x.cond_ci_lo, x.cond_ci_hi))
                          if x.cond_n else "0/0"))
    L += ["Mean +- SD over seeds of the conditional rate (seeds where it is defined):", ""]
    L.append(phase_matrix(pooled, "human", lambda x: "%s (%d seeds)" % (msd(x.cond_mean, x.cond_sd), x.cond_seeds_defined)))
    L += ["### 1d. Per run: episodes reaching phases 1..k (of 100) [Wilson 95% CI, %]", ""]
    L.append(run_matrix(by_run, "human"))
    L += ["### 1e. The old four-phase table (grasp, lift, transport, place), for continuity", "",
          "As earlier reports showed it: the rate of each phase on its own (not chained through walk-in), pooled over seeds, "
          "% (count of 300). On human data no grasp happened without a pickup lock, so these equal the cumulative rates above; "
          "on scripted Run A 2 of 1200 episodes grasped without one.", ""]
    rows = []
    for data in ("human", "scripted"):
        for m in MODELS:
            for e in EXPS:
                rows.append([data, m, EXP_NAME[e]] + ["%s (%d)" % (pct(_pp(pooled, data, m, e, p).marginal_k / 300),
                                                                  _pp(pooled, data, m, e, p).marginal_k)
                                                     for p in OLD_PHASES + ("success",)])
    L.append(md(rows, ["data", "model", "exp", "grasp", "lift", "transport", "place", "success"]))
    L += ["![Cumulative phase rates](%s)" % figs["cum"], "", "![Conditional phase rates](%s)" % figs["cond"], ""]
    L += ["## 2. Pairwise tests", "", "### 2a. PRIMARY: position-level paired test", "",
          "Each experiment's 100 box positions are the same for every policy. Per position, the number of a model's 3 seeds "
          "that reached phases 1..k (0-3); two models compared over positions. Wilcoxon signed-rank: normal approximation, tie "
          "and continuity corrections, ties dropped; sign test exact. Effect size: matched-pairs rank-biserial r, +1 = every "
          "untied position favours A. **Holm over the 30 phase x experiment x pair tests**, per test type. Unit of inference: "
          "box positions, for THESE 3 trained seeds per model; it does not speak to a new training seed.", ""]
    L.append(paired_rows(T["paired"]))
    L += ["Full success (Holm over its own 6):", ""]
    L.append(paired_rows(T["paired_success"]))
    L += ["### 2b. SECONDARY: Fisher's exact test on pooled counts (%s)" % INDEP, "",
          "Two-sided; each n = 300 is 3 seeds on the same 100 positions, so this ignores the between-seed variance and the "
          "pairing. Same Holm families (30; 6).", ""]
    L.append(fisher_rows(T["fisher"]))
    L += ["Full success:", ""]
    L.append(fisher_rows(T["fisher_success"]))
    L += ["## 3. Generalisation gap dS = Exp 1 - Exp 2 (cumulative rates)", "",
          "Newcombe hybrid score 95% CI (method 10, from the two Wilson intervals: valid at 0 successes); the Wald CI the "
          "scripted report used is shown beside it. Exp 1 and Exp 2 are different positions, so the gap is unpaired. "
          "SE over seeds = SD of the per-seed gaps / sqrt(3).", ""]
    L.append(gap_rows(T["gap"], "human"))
    L += ["### 3b. Exploratory: the gap adjusted for lateral position", "",
          "The walk-in rate depends strongly on the box's lateral position (section 4b), and the held-out patch (y 0.04-0.16 m) "
          "lies in the centre-left. So part of the raw gap may be WHERE the patch is, not that its positions were unseen. Here "
          "Exp 1's rate is re-weighted to Exp 2's distribution over the 8 lateral columns of the position heatmap (columns "
          "`columns`, weights `weights` in `generalization_gap_lateral.csv`), then Exp 2's rate is subtracted. 95%% CI: cluster "
          "bootstrap over box positions (%d resamples, seed 0). The adjustment is coarse (columns 5.25 cm wide, depth not "
          "adjusted) and post hoc: read it as a check on the raw gap, not as the headline." % N_BOOT, ""]
    LA = T["lateral"]
    L.append(md([[r.data, r.model, LABEL[r.phase], pct(r.exp1_rate), pct(r.exp1_rate_lateral_adjusted), pct(r.exp2_rate),
                  "%+.1f [%+.1f, %+.1f]" % (100 * r.gap_raw, 100 * r.gap_raw_lo, 100 * r.gap_raw_hi),
                  "%+.1f [%+.1f, %+.1f]" % (100 * r.gap_adjusted, 100 * r.gap_adjusted_lo, 100 * r.gap_adjusted_hi),
                  r.exp1_positions_per_column] for _, r in LA.iterrows()],
                ["data", "model", "phase", "Exp 1 (%)", "Exp 1 lateral-adjusted (%)", "Exp 2 (%)", "raw gap [bootstrap CI]",
                 "adjusted gap [bootstrap CI]", "Exp 1 positions per column"]))
    L += ["![Generalisation gap](%s)" % figs["gap"], ""]
    L += ["## 4. Failure phase and spawn side", "",
          "### 4a. The first of the five phases not reached (pooled over seeds; counts)", "",
          "`fell` = the episode fell before reaching its next phase.", ""]
    F = T["fail"]
    L.append(md([[r.model, EXP_NAME[r.exp], r.n, r.success, r.WALK_IN, r.GRASP, r.LIFT, r.TRANSPORT, r.PLACE, r.fell]
                 for _, r in F[F.data == "human"].iterrows()],
                ["model", "exp", "n", "full success", "walk-in", "grasp", "lift", "transport", "place", "fell"]))
    L += ["For comparison, `deploy.py`'s own `fail_reason` bucket (from the offline phase labels, `spec.SCORED_OF`; it charges a "
          "never-grasped episode to WALK_IN only when its last label is SETTLE, so it is not the same as the walk-in phase):", ""]
    FR = T["fail_reason"]
    L.append(md([[r.model, EXP_NAME[r.exp], r.n, r.success, r.WALK_IN, r.GRASP, r.LIFT, r.TRANSPORT, r.PLACE, r.fell, r.other]
                 for _, r in FR[FR.data == "human"].iterrows()],
                ["model", "exp", "n", "success", "WALK_IN", "GRASP", "LIFT", "TRANSPORT", "PLACE", "fell", "other"]))
    L += ["![Failure phase](%s)" % figs["fail"], ""]
    L += ["### 4b. By spawn side (robot's view: world y < -0.07 m right, |y| <= 0.07 centre, > 0.07 left)", "",
          "Walk-in (pickup lock), grasp and full success, k/n, pooled over seeds. The held-out patch (y 0.04-0.16 m) has no "
          "right third.", ""]
    tr_sides = T["train_sides"]
    L += ["Where the 40 TRAINING demonstrations were, by the same thirds (`%s`, train seeds; spawn from the seed, "
          "`analyze_experiments.spawn_xy`): human %s; scripted Run A %s." % (
              os.path.basename(SPLITS["human"]), ", ".join("%s %d" % (s, tr_sides["human"][s]) for s in SIDES),
              ", ".join("%s %d" % (s, tr_sides["scripted"][s]) for s in SIDES)), ""]
    S = T["sides"]
    L.append(md([[r.model, {"exp1": "Exp 1", "exp2": "Exp 2", "both": "both"}[r.exp], r.side, r.n,
                  "%d (%s)" % (r.walk_in_k, ci(r.walk_in, r.walk_in_lo, r.walk_in_hi, 0)), "%d" % r.grasp_k, "%d" % r.success_k]
                 for _, r in S[S.data == "human"].iterrows()],
                ["model", "exp", "side", "n", "walk-in: k (% [CI])", "grasp k", "full success k"]))
    L += ["Exploratory: walk-in independent of side? Chi-square, Holm over the 3 models per experiment:", ""]
    st = T["side_tests"]
    L.append(md([[r.exp, r.model, r.sides, "" if r.chi2 != r.chi2 else "%.2f" % r.chi2, r.dof, pv(r.p),
                  pv(r.get("p_holm", float("nan"))), "" if r.min_expected != r.min_expected else "%.1f" % r.min_expected, r.note]
                 for _, r in st[st.data == "human"].iterrows()],
                ["exp", "model", "sides", "chi2", "dof", "p", "p (Holm)", "min expected", ""]))
    L += ["Failure phase by side (Exp 1 + Exp 2):", ""]
    FS = T["fail_side"]
    L.append(md([[r.model, r.side, r.n, r.success, r.WALK_IN, r.GRASP, r.LIFT, r.TRANSPORT, r.PLACE, r.fell]
                 for _, r in FS[FS.data == "human"].iterrows()],
                ["model", "side", "n", "full success", "walk-in", "grasp", "lift", "transport", "place", "fell"]))
    L += ["![Walk-in by side](%s)" % figs["side"], "", "![Failure by side](%s)" % figs["fail_side"], "",
          "![Success by position](%s)" % figs["position"], ""]
    L += ["## 5. Diagnostics (not official phases)", "",
          "**Goal lock**: a lock engaged while the box was welded (it targets the goal). **Release**: the weld let go after "
          "engaging, wherever that was (it includes welds that failed the D19 push rule and never counted as a grasp). "
          "Of grasped boxes = of episodes that reached phase 2. k/n, % [Wilson 95% CI].", ""]
    D = T["diag"]
    L.append(md([[r.data, r.model, EXP_NAME[r.exp], "%d/%d, %s" % (r.goal_lock_k, r.n, ci(r.goal_lock, r.goal_lock_lo, r.goal_lock_hi)),
                  "%d/%d, %s" % (r.release_k, r.n, ci(r.release, r.release_lo, r.release_hi)), r.grasped,
                  "%d/%d" % (r.grasped_goal_lock_k, r.grasped), "%d/%d" % (r.grasped_released_k, r.grasped),
                  "%d/%d" % (r.grasped_placed_k, r.grasped), r.welded, r.welded_without_pickup_lock]
                 for _, r in D.iterrows()],
                ["data", "model", "exp", "goal lock", "release", "grasped", "of grasped: goal lock", "of grasped: released",
                 "of grasped: placed", "welded (any)", "welded with no pickup lock"]))
    L += ["Tests on the two diagnostics (exploratory; Holm over their 12): primary position-level, then Fisher.", ""]
    L.append(paired_rows(T["paired_diag"]))
    L.append(fisher_rows(T["fisher_diag"]))
    L += ["## 6. Proximity", ""]
    L += ["- **%s:** %s" % (short, what) for _, short, what in MFR.PROXIMITY]
    L += ["", "Median [Q1, Q3], pooled over seeds.", ""]
    PS = T["prox"]
    rows = []
    for data in ("human", "scripted"):
        for m in MODELS:
            for e in EXPS:
                g = PS[(PS.data == data) & (PS.model == m) & (PS.exp == e)].set_index("metric")
                rows.append([data, m, EXP_NAME[e]] + ["%.3f [%.3f, %.3f]" % (g.loc[k, "median"], g.loc[k, "q1"], g.loc[k, "q3"])
                                                      for _, k, _ in MFR.PROXIMITY])
    L.append(md(rows, ["data", "model", "exp"] + ["%s (m)" % k for _, k, _ in MFR.PROXIMITY]))
    L += ["Kruskal-Wallis across the three models, then two-sided Mann-Whitney U, Holm over the 3 pairs per metric and "
          "experiment (%s; human data):" % INDEP, ""]
    L.append(md([[r.metric, r.exp, r.test, r.pair, r["median"], "%.4g" % r.stat, pv(r.p), pv(r.p_holm)]
                 for _, r in T["dist"].iterrows()], ["metric", "exp", "test", "models", "median (m)", "stat", "p", "p (Holm)"]))
    L += ["## 7. Human vs scripted Run A (same models, 40 demos, seeds 0-2)", "",
          "**Not a controlled comparison:** the training data differ in source AND labels (human walking labels smoothed, "
          "`centred_ma13_v1`; scripted raw), the evaluation caps differ (human 1568 ticks = 1.3 x the longest human demo; "
          "scripted 900), and the checkpoints were selected separately. The box positions are the same, so the paired test "
          "pairs human and scripted runs of one model position by position.", "",
          "### 7a. Scripted Run A, five phases, cumulative, pooled: % [Wilson 95% CI]", ""]
    L.append(phase_matrix(pooled, "scripted", lambda x: ci(x.rate, x.ci_lo, x.ci_hi)))
    L += ["Mean +- SD over seeds:", ""]
    L.append(phase_matrix(pooled, "scripted", lambda x: msd(x.mean_rate, x.sd_rate)))
    L += ["Conditional:", ""]
    L.append(phase_matrix(pooled, "scripted", lambda x: "%d/%d, %s" % (x.cond_k, x.cond_n, ci(x.cond_rate, x.cond_ci_lo, x.cond_ci_hi))
                          if x.cond_n else "0/0"))
    L += ["### 7b. Human - scripted, per model (position-level primary; Holm over the 30 phase x experiment x model tests)", ""]
    HP = T["hs_paired"]
    L.append(md([[r.exp, LABEL[r.phase], r.model, "%.2f vs %.2f" % (r.mean_human, r.mean_scripted), r.favour_a, r.favour_b, r.tie,
                  pv(r.wilcoxon_p), pv(r.wilcoxon_p_holm), "%+.2f" % r.rank_biserial, pv(r.sign_p), pv(r.sign_p_holm)]
                 for _, r in HP.iterrows()],
                ["exp", "phase", "model", "mean per position, human vs scripted (of 3)", "favour human", "favour scripted", "tie",
                 "Wilcoxon p", "Holm", "r", "sign p", "Holm"]))
    L += ["Secondary, Fisher on pooled counts:", ""]
    HF = T["hs_fisher"]
    L.append(md([[r.exp, LABEL[r.phase], r.model, "%d/%d vs %d/%d" % (r.human_k, r.human_n, r.scripted_k, r.scripted_n),
                  "%+.1f" % (100 * r["diff"]), pv(r.p), pv(r.p_holm)] for _, r in HF.iterrows()],
                ["exp", "phase", "model", "human vs scripted", "diff (pts)", "p", "p (Holm)"]))
    L += ["### 7c. Generalisation gap, scripted", ""]
    L.append(gap_rows(T["gap"], "scripted"))
    L += ["### 7d. Proximity, human vs scripted (Mann-Whitney U, Holm over the 3 models per metric and experiment)", ""]
    L.append(md([[r.metric, r.exp, r.model, "%.3f vs %.3f" % (r.median_human, r.median_scripted), r.n, pv(r.p), pv(r.p_holm)]
                 for _, r in T["hs_prox"].iterrows()], ["metric", "exp", "model", "median human vs scripted (m)", "n", "p", "p (Holm)"]))
    L += ["![Human vs scripted](%s)" % figs["hs"], "", "Scripted figures in the same form: ![Scripted cumulative](%s)" % figs["cum_s"],
          "", "![Scripted gap](%s)" % figs["gap_s"], ""]
    a, tr = tele["all"], tele["trend"]
    L += ["## 8. Teleoperation: the 50 human demonstrations", "",
          "From `tools/teleop_metrics.py recordings/demonstrations` (tables in `teleop/`). One operator, 11 sessions.", ""]
    L.append(md([[int(a.attempts), int(a.accepted), "%.1f%% [%s]" % (100 * a.success_rate, "%.1f, %.1f" % tuple(
        100 * v for v in AE.wilson(int(a.accepted), int(a.attempts)))), int(a.failed), int(a.discarded),
        int(a.discarded_successes), int(a.revoked), int(a.unfinished), "%.1f" % a.wall_s_per_attempt,
        "%.1f" % a.wall_s_per_accepted_demo, "%.2f" % a.tick_rate_hz_mean, "%.2f" % a.late_tick_pct, "%.2f" % a.dropout_pct,
        "%d / %.1f / %d" % (a.demo_ticks_min, a.demo_ticks_median, a.demo_ticks_max), "%.1f" % a.demo_s_median]],
        ["attempts", "accepted", "acceptance [Wilson CI]", "failed", "discarded", "of which successes", "revoked", "unfinished",
         "wall s / attempt", "wall s / accepted demo", "tick rate (Hz)", "late ticks (%)", "dropout (%)",
         "demo ticks min / median / max", "demo s median"]))
    L += ["**Acceptance = accepted / attempts, every outcome in the denominator.** Most non-accepts are operator discards "
          "(%d of %d); %d of the discards were episodes that met the success rule (including %d accepts later revoked; the "
          "acceptance rules are CLAUDE.md section 8 2026-11-13c), so acceptance understates how often the task was completed. "
          "Session dates are the ledger's." % (int(a.discarded), int(a.attempts), int(a.discarded_successes), int(a.revoked)), "",
          "### Per session", ""]
    L.append(md([[int(r.session_i), r.date, int(r.attempts), int(r.accepted), "%s" % ci(r.success_rate, r.ci_lo, r.ci_hi),
                  int(r.discarded_successes), "%.1f" % r.wall_s_per_accepted_demo if r.wall_s_per_accepted_demo == r.wall_s_per_accepted_demo
                  else "n/a", "%.2f" % r.tick_rate_hz_mean, "%.2f" % r.late_tick_pct if r.late_tick_pct == r.late_tick_pct else "n/a",
                  "%.2f" % r.dropout_pct if r.dropout_pct == r.dropout_pct else "n/a"] for _, r in tele["sessions"].iterrows()],
                ["session", "date", "attempts", "accepted", "acceptance % [CI]", "discarded successes", "wall s / accepted demo",
                 "tick rate (Hz)", "late ticks (%)", "dropout (%)"]))
    L += ["**Learning trend:** Spearman of acceptance vs attempt index rho = %+.3f (p %s); first half %s vs second half %s "
          "(Fisher p %s); per-session acceptance vs session index rho = %+.3f (p %s)." % (
              tr["spearman_attempt_rho"], pv(tr["spearman_attempt_p"]), tr["first_half"], tr["second_half"],
              pv(tr["halves_fisher_p"]), tr["spearman_session_rho"], pv(tr["spearman_session_p"])), "",
          "![Teleop learning curve](%s)" % figs["teleop"], ""]
    L += ["## Caveats, to carry into the write-up", "",
          "- **Five phases is a 2026-11-22 decision.** The proposal and earlier reports had four (grasp, lift, transport, place); "
          "section 1e keeps the four-phase numbers. Walk-in is a navigation-to-the-box outcome, not a manipulation target, so it "
          "does not change what Objective 4 varies (box position only, Q6/TR13).",
          "- **Two levels of inference, neither over training seeds.** Fisher treats episodes as independent; the position-level "
          "test keeps the pairing by position, but its 0-3 counts come from the same 3 seeds. With 3 seeds an exact test on seed "
          "rates cannot go below p = 0.1, so nothing here says what a new training seed would do. Read the per-seed values.",
          "- **Wilcoxon with few untied positions.** Where only a handful of positions differ (e.g. Exp 2 place, 2-6 untied), "
          "the normal approximation is poor (scipy warns); read the exact sign test there.",
          "- **The walk-in depends on side.** Right-side spawns lock far more often than left-side ones for every model "
          "(section 4b), and the held-out patch is centre-left. The raw Exp 1 - Exp 2 gap therefore mixes novelty with lateral "
          "position; section 3b gives an exploratory adjustment.",
          "- **Multiplicity.** Holm is applied within each named family (30 phase tests; 6 success; 12 diagnostics; and the "
          "same for human vs scripted), not across families, test types or the proximity metrics.",
          "- **Cumulative vs marginal.** The cumulative chain requires the walk-in; on human data this changes nothing (no grasp "
          "without a pickup lock), on scripted 2 of 1200 episodes.",
          "- **Free-based welds.** Some welds engaged with no pickup lock (section 5): the base was not held (D12) at the weld.",
          "- **Selection is noisy.** Checkpoints were chosen on 20 episodes; chosen steps range 40k-180k across ACT-LSTM seeds.",
          "- **BC's budget.** BC is epoch-budgeted (35,400 optimizer steps) against 200k for ACT and ACT-LSTM.",
          "- **ACT-LSTM training exits** with 0xC0000409 (the cuDNN exit crash, CLAUDE.md 2026-09-29); every run reached its "
          "200k cap and saved its checkpoints first.",
          "- **Human vs scripted is confounded** (section 7): data source, label smoothing, evaluation cap and selection all differ.",
          "- **Spawn sides are uneven in Exp 2**: the held-out patch has no right third, so 'by side' pooled over experiments "
          "weights centre/left by Exp 2."]
    L += ["", "## Files", "", "CSV tables beside this report: `" + "`, `".join(sorted(T["files"])) + "`. Figures: PNG at 300 dpi "
          "and SVG with the same stem.", ""]
    text = "\n".join(L) + "\n"
    for junk in ("dtype:", "<bound method", "| nan |", "nan%"):
        if junk in text:
            raise AssertionError("report cell contains %r" % junk)
    path = os.path.join(out, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ─── build ────────────────────────────────────────────────────────────────────
def build(out: str = OUT, strict: bool = True) -> dict:
    os.makedirs(out, exist_ok=True)
    EH, RH, _ = load_set(HUMAN_ROOTS, "human")
    ES, RS, _ = load_set(SCRIPTED_ROOTS, "scripted", SCRIPTED_D19)
    RH, ph = audit(EH, RH, "human", strict)
    RS, ps = audit(ES, RS, "scripted", strict)
    seeds_same = all(set(EH[EH.exp == e].seed) == set(ES[ES.exp == e].seed) for e in EXPS)
    if not seeds_same:
        raise SystemExit("human and scripted runs did not evaluate the same positions")
    EA = pd.concat([EH, ES], ignore_index=True)
    by_run = phase_by_run(EA)
    pooled = phase_pooled(EA, by_run)
    T = dict(runs=pd.concat([RH, RS], ignore_index=True), by_run=by_run, pooled=pooled, gap=gap_table(by_run, pooled),
             lateral=lateral_adjusted_gap(EA), train_sides=training_sides(),
             paired=pd.DataFrame(paired_family(EH, PHASES, "phases: 5 x 2 exps x 3 pairs")),
             paired_success=pd.DataFrame(paired_family(EH, ("success",), "full success: 2 exps x 3 pairs")),
             paired_diag=pd.DataFrame(paired_family(EH, DIAG, "diagnostics, exploratory", cumulative=False)),
             fisher=pd.DataFrame(fisher_family(EH, PHASES, "phases: 5 x 2 exps x 3 pairs")),
             fisher_success=pd.DataFrame(fisher_family(EH, ("success",), "full success: 2 exps x 3 pairs")),
             fisher_diag=pd.DataFrame(fisher_family(EH, DIAG, "diagnostics, exploratory", cumulative=False)),
             fail=failure_table(EA, ["data", "model", "exp"]), fail_side=failure_table(EA, ["data", "model", "side"]),
             fail_side_exp=failure_table(EA, ["data", "model", "exp", "side"]),
             fail_reason=pd.concat([AE.failure_phase(EH).assign(data="human"), AE.failure_phase(ES).assign(data="scripted")],
                                   ignore_index=True),
             sides=side_table(EA), side_tests=side_tests(EA), diag=diagnostics(EA),
             prox=pd.concat([MFR.proximity_summary(EH).assign(data="human"), MFR.proximity_summary(ES).assign(data="scripted")],
                            ignore_index=True),
             dist=pd.DataFrame(MFR.distance_family(EH, "n40", list(PAIRS), "primary", list(MODELS))),
             hs_paired=pd.DataFrame(hs_paired(EA, PHASES, "phases: 5 x 2 exps x 3 models") +
                                    hs_paired(EA, ("success",), "full success: 2 exps x 3 models")),
             hs_fisher=pd.DataFrame(hs_fisher(EA, PHASES, "phases: 5 x 2 exps x 3 models") +
                                    hs_fisher(EA, ("success",), "full success: 2 exps x 3 models")),
             hs_prox=hs_proximity(EA), teleop=teleop_tables(os.path.join(out, "teleop")),
             audit_notes=pd.DataFrame(dict(data=["human"] * len(ph) + ["scripted"] * len(ps), problem=ph + ps)))
    files = {}
    for key, name in (("runs", "runs"), ("by_run", "phase_by_run"), ("pooled", "phase_pooled"), ("gap", "generalization_gap"),
                      ("lateral", "generalization_gap_lateral"),
                      ("paired", "tests_position_phases"), ("paired_success", "tests_position_success"),
                      ("paired_diag", "tests_position_diagnostics"), ("fisher", "tests_fisher_phases"),
                      ("fisher_success", "tests_fisher_success"), ("fisher_diag", "tests_fisher_diagnostics"),
                      ("fail", "failure_phase"), ("fail_side", "failure_by_side"), ("fail_side_exp", "failure_by_side_exp"),
                      ("fail_reason", "failure_reason_deploy"), ("sides", "spawn_side"), ("side_tests", "spawn_side_tests"),
                      ("diag", "diagnostics"), ("prox", "proximity_summary"), ("dist", "proximity_tests"),
                      ("hs_paired", "human_vs_scripted_position"), ("hs_fisher", "human_vs_scripted_fisher"),
                      ("hs_prox", "human_vs_scripted_proximity"), ("audit_notes", "audit_notes")):
        T[key].to_csv(os.path.join(out, name + ".csv"), index=False)
        files[name + ".csv"] = True
    pd.DataFrame([T["teleop"]["trend"]]).to_csv(os.path.join(out, "teleop_trend.csv"), index=False)
    files["teleop_trend.csv"] = True
    EA.drop(columns=["trajectory_sha256"]).to_csv(os.path.join(out, "episodes.csv"), index=False)
    files["episodes.csv"] = True
    T["files"] = files
    figs = dict(cum=fig_phases(pooled, by_run, out), cond=fig_phases(pooled, by_run, out, conditional=True),
                cum_s=fig_phases(pooled, by_run, out, data="scripted"), gap=fig_gap(T["gap"], out, lat=T["lateral"]),
                gap_s=fig_gap(T["gap"], out, "scripted", lat=T["lateral"]), fail=fig_failures(T["fail"], out),
                fail_side=fig_failures_by_side(T["fail_side"], out), side=fig_lock_by_side(T["sides"], out),
                position=fig_position(EA, out), hs=fig_human_vs_scripted(pooled, out), teleop=fig_teleop(T["teleop"], out))
    report = write_report(T, figs, out)
    return dict(tables=T, episodes=EA, figures=figs, report=report)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-strict", action="store_true", help="report audit problems instead of stopping")
    a = ap.parse_args(argv)
    res = build(a.out, strict=not a.no_strict)
    print("episodes %d, report %s" % (len(res["episodes"]), res["report"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
