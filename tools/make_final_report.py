"""The final scripted analysis, as one report folder, under thesis names.

    python tools/make_final_report.py          # runs/scripted_v3/analysis_final/
    python tools/make_final_report.py --out reports/final --no-strict

READ-ONLY on the evaluation: no training and no new evaluation runs. Loading, Wilson, Holm, Fisher and
the per-model tables come from tools/analyze_experiments.py. This tool adds the comparisons the thesis
asks for, the position-level paired test and the figures. docs/ANALYSIS.md describes the output.

THE POSITION-LEVEL TEST. Every policy runs the same box positions. For each position, count successes
over a model's 3 training seeds (0-3), then compare two models over the positions with a Wilcoxon
signed-rank test and a sign test. Unlike the pooled Fisher test, this keeps the pairing by position and
does not treat one position's 3 episodes as independent. The positions are the sampling unit, so it
speaks to new box positions for THESE trained models, not to a new training seed.

--no-strict skips the checks that the input is exactly the final set (22 runs, each config read back
from its own metadata.json).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kruskal, mannwhitneyu, rankdata, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import analyze_experiments as AE                                    # noqa: E402
import report_io as RIO                                             # noqa: E402

ROOTS = (os.path.join(ROOT, "runs", "scripted_v3", "eval"),
         os.path.join(ROOT, "runs", "scripted_v3_memory_design", "eval"))
D19_ROOTS = (os.path.join(ROOT, "runs", "scripted_v3", "eval_d19"),)
OUT = os.path.join(ROOT, "runs", "scripted_v3", "analysis_final")

#: What each thesis name must be, read back from the run's own metadata.json: (code name, K, W_o, batch).
EXPECT = {"BC": ("bc", 1, 12, 256), "ACT": ("act_orig", 100, 1, 8), "ACT-LSTM": ("act_lstm_full", 100, 1, 8),
          "ACT-12": ("act", 100, 12, 8), "ACT-LSTM-12": ("act_lstm", 100, 12, 8), "BC (batch 8)": ("bc", 1, 12, 8)}
#: (model, size, tier), in report order.
GROUPS = [("BC", "n40", "main"), ("ACT", "n40", "main"), ("ACT-LSTM", "n40", "main"),
          ("ACT-12", "n40", "ablation"), ("ACT-LSTM-12", "n40", "ablation"),
          ("BC (batch 8)", "n40", "appendix"),
          ("BC", "160", "extra"), ("ACT-12", "160", "extra"), ("ACT-LSTM-12", "160", "extra"),
          ("BC (batch 8)", "160", "extra")]
TIER = {(m, s): t for m, s, t in GROUPS}
RANK = {(m, s): i for i, (m, s, _) in enumerate(GROUPS)}
#: Compared position by position, on the 40-demo runs. Holm over these three, per experiment and test.
POSITION_PAIRS = [("ACT-LSTM", "ACT"), ("ACT-LSTM", "ACT-LSTM-12"), ("ACT", "BC")]
#: The pooled comparisons, on the 40-demo runs. Holm over these four, per experiment and metric.
PRIMARY = [("ACT-LSTM", "ACT"), ("ACT-LSTM", "BC"), ("ACT", "BC"), ("ACT-LSTM", "ACT-LSTM-12")]
#: Context, labelled exploratory; a separate Holm family.
CONTEXT = [("ACT", "ACT-12"), ("ACT-LSTM-12", "ACT-12"), ("BC", "BC (batch 8)")]
EXTRA_160 = [("ACT-LSTM-12", "ACT-12"), ("ACT-LSTM-12", "BC"), ("ACT-12", "BC")]
EXP_NAME = {"exp1": "Exp 1", "exp2": "Exp 2"}
#: Proximity metrics: (column, short name, what it is).
PROXIMITY = (("palm_box_nearest_min_m", "closest palm to box",
              "the nearer palm's closest approach to the box before the weld (m)"),
             ("palm_box_min_m", "farther palm to box",
              "the farther palm's closest approach before the weld: what the 0.16 m weld gate reads (m)"),
             ("final_box_goal_m", "final box to goal", "box to goal centre on the last tick; placement needs 0.10 m (m)"))
INDEPENDENT = "episodes treated as independent"


def arrange(df):
    """Report order: main, ablation, appendix, then the 160-demo runs."""
    d = df.copy()
    d["_r"] = [RANK.get((m, s), 99) for m, s in zip(d["model"], d["size"])]
    keys = ["_r"] + [c for c in ("train_seed", "exp") if c in d.columns]
    d = d.sort_values(keys).drop(columns="_r").reset_index(drop=True)
    d.insert(0, "demos", d["size"].map(AE.size_n))
    d.insert(1, "tier", [TIER.get((m, s), "") for m, s in zip(d["model"], d["size"])])
    return d


def name_of(model, size):
    return model if size == "n40" else "%s (160)" % model


# ─── checks: the numbers are what they claim to be ────────────────────────────
def audit(E, R):
    """The input is exactly the final set, and each run is what its thesis name says."""
    if not (len(R) == 22 and len(E) == 22 * 200):
        raise SystemExit("expected 22 runs and 4400 episodes, found %d and %d" % (len(R), len(E)))
    seeds = {e: E[E.exp == e].groupby("run").seed.apply(lambda s: tuple(sorted(s))) for e in AE.EXPS}
    checks = {"one Exp 1 seed set": seeds["exp1"].nunique() == 1, "one Exp 2 seed set": seeds["exp2"].nunique() == 1,
              "Exp 1 and Exp 2 disjoint": not set(seeds["exp1"].iloc[0]) & set(seeds["exp2"].iloc[0]),
              "Exp 1 outside the patch": not E[E.exp == "exp1"].heldout.any(),
              "Exp 2 inside the patch": E[E.exp == "exp2"].heldout.all(),
              "lock rule v3": set(E.lock_rule) == {"v3"}, "policy walk-in": set(E.walkin) == {"policy"}}
    for column, _, _ in PROXIMITY:
        checks["%s on every episode" % column] = bool(E[column].notna().all())
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("the final set fails: %s" % ", ".join(failed))
    rows = []
    for _, r in R.iterrows():
        base = os.path.dirname(os.path.dirname(os.path.join(ROOT, r.path)))
        with open(os.path.join(base, r.run, "metadata.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        got = (m["config"]["notes"]["model"], int(m["config"]["notes"]["chunk_size"]),
               int(m["loader"]["obs_window"]), int(m["config"]["batch_size"]))
        if got != EXPECT[r.model] or len(m["data"]["seeds"]) != AE.size_n(r["size"]):
            raise SystemExit("%s is not %s: %s, %d demos" % (r.run, r.model, got, len(m["data"]["seeds"])))
        with open(os.path.join(base, r.run, "result.json"), encoding="utf-8") as fh:
            rows.append(dict(train_steps=json.load(fh).get("optimizer_steps")))
    return pd.concat([R.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ─── the position-level paired test ───────────────────────────────────────────
def position_counts(E, size: str = "n40"):
    """One row per (experiment, box position): each model's successes over its training seeds."""
    rows = []
    for exp, g in E[E["size"] == size].groupby("exp"):
        wide = g.pivot_table(index="seed", columns="model", values="success", aggfunc="sum")
        wide = wide.astype(int)
        where = g.groupby("seed")[["box_x", "box_y"]].first()
        wide.insert(0, "exp", EXP_NAME[exp])
        rows.append(where.join(wide).reset_index())
    out = pd.concat(rows, ignore_index=True).rename(columns={"seed": "position_seed"})
    return out


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Wilcoxon signed-rank and sign test on paired counts, two-sided.

    Ties (a == b) are dropped from both tests and counted. Wilcoxon uses the normal approximation with
    the tie correction and a continuity correction (exact is not available with tied ranks). The effect
    size is the matched-pairs rank-biserial correlation, (W+ - W-) / (W+ + W-): +1 means every untied
    position favours A, -1 every one favours B."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nz = d[d != 0]
    ranks = rankdata(np.abs(nz))
    w_plus, w_minus = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
    n_a, n_b = int((d > 0).sum()), int((d < 0).sum())
    if len(nz):
        p_w = float(wilcoxon(nz, zero_method="wilcox", correction=True, alternative="two-sided",
                             method="approx").pvalue)
        p_s = float(binomtest(n_a, n_a + n_b, 0.5, alternative="two-sided").pvalue)
    else:
        p_w = p_s = 1.0
    return dict(n_positions=len(d), favour_a=n_a, favour_b=n_b, tie=int((d == 0).sum()), w_plus=w_plus, w_minus=w_minus,
                wilcoxon_p=p_w, sign_p=p_s, rank_biserial=(w_plus - w_minus) / (w_plus + w_minus) if len(nz) else 0.0,
                mean_diff=float(d.mean()))


def position_tests(counts, pairs: Sequence[tuple] = tuple(POSITION_PAIRS)):
    """The paired test for each pair and experiment, Holm over the pairs within one experiment and test."""
    rows = []
    for exp in EXP_NAME.values():
        g = counts[counts.exp == exp]
        if not len(g):
            continue
        fam = []
        for a, b in pairs:
            r = dict(exp=exp, A=a, B=b, mean_A=float(g[a].mean()), mean_B=float(g[b].mean()),
                     tie_both_zero=int(((g[a] == 0) & (g[b] == 0)).sum()), **paired_test(g[a], g[b]))
            fam.append(r)
        for key in ("wilcoxon", "sign"):
            for r, adj in zip(fam, AE.holm([r[key + "_p"] for r in fam])):
                r[key + "_p_holm"] = adj
        rows += fam
    return pd.DataFrame(rows)


# ─── pooled tests (episodes treated as independent) ───────────────────────────
def _seed_rates(g):
    return ", ".join("%d" % round(100 * v) for v in g.groupby("train_seed").success.mean())


def fisher_family(E, size, pairs, family):
    out = []
    for exp in AE.EXPS:
        g = E[(E["size"] == size) & (E.exp == exp)]
        fam = []
        for a, b in pairs:
            ga, gb = g[g.model == a], g[g.model == b]
            k1, n1, k2, n2 = int(ga.success.sum()), len(ga), int(gb.success.sum()), len(gb)
            orr, p = AE.fisher(k1, n1, k2, n2)
            fam.append(dict(family=family, assumption=INDEPENDENT, demos=AE.size_n(size), exp=EXP_NAME[exp], A=a, B=b,
                            A_success=k1, A_n=n1, B_success=k2, B_n=n2, A_rate=k1 / n1, B_rate=k2 / n2,
                            diff=k1 / n1 - k2 / n2, odds_ratio=orr, p=p, A_seed_pct=_seed_rates(ga), B_seed_pct=_seed_rates(gb)))
        for r, adj in zip(fam, AE.holm([r["p"] for r in fam])):
            r["p_holm"] = adj
        out += fam
    return out


def distance_family(E, size, pairs, family, kw_models=None):
    out = []
    for metric, short, _ in PROXIMITY:
        for exp in AE.EXPS:
            g = E[(E["size"] == size) & (E.exp == exp)]

            def sample(m):
                return g.loc[g.model == m, metric].dropna().to_numpy(float)
            base = dict(family=family, assumption=INDEPENDENT, metric=short, demos=AE.size_n(size), exp=EXP_NAME[exp])
            if kw_models:
                H, p = kruskal(*[sample(m) for m in kw_models])
                out.append(dict(base, test="Kruskal-Wallis", pair=", ".join(kw_models), stat=float(H), p=float(p),
                                p_holm=float("nan"), median=", ".join("%.3f" % np.median(sample(m)) for m in kw_models),
                                n=", ".join(str(len(sample(m))) for m in kw_models)))
            fam = []
            for a, b in pairs:
                U, p = mannwhitneyu(sample(a), sample(b), alternative="two-sided")
                fam.append(dict(base, test="Mann-Whitney U", pair="%s vs %s" % (a, b), stat=float(U), p=float(p),
                                median="%.3f, %.3f" % (np.median(sample(a)), np.median(sample(b))),
                                n="%d, %d" % (len(sample(a)), len(sample(b)))))
            for r, adj in zip(fam, AE.holm([r["p"] for r in fam])):
                r["p_holm"] = adj
            out += fam
    return out


def proximity_summary(E):
    """Median and quartiles, per model, experiment and metric, pooled over training seeds."""
    rows = []
    for (m, s, _) in GROUPS:
        for exp in AE.EXPS:
            g = E[(E.model == m) & (E["size"] == s) & (E.exp == exp)]
            for metric, short, _what in PROXIMITY:
                v = g[metric].dropna().to_numpy(float)
                if len(v):
                    q1, med, q3 = np.percentile(v, [25, 50, 75])
                    rows.append(dict(model=m, size=s, exp=exp, metric=short, n=len(v), missing=len(g) - len(v),
                                     median=float(med), q1=float(q1), q3=float(q3)))
    return pd.DataFrame(rows)


def seed_gaps(by_run, m, s):
    """Exp 1 minus Exp 2 for each training seed, in points."""
    g = by_run[(by_run.model == m) & (by_run["size"] == s)]
    e1, e2 = g[g.exp == "exp1"].set_index("train_seed").rate, g[g.exp == "exp2"].set_index("train_seed").rate
    return ", ".join("%+.0f" % (100 * v) for v in (e1 - e2).sort_index())


# ─── figures (one fixed colour per job; identity is always also in text) ──────
EXP_COL = {"exp1": RIO.SERIES[0], "exp2": RIO.SERIES[1]}
EXP_LEGEND = {"exp1": "Exp 1 (training region)", "exp2": "Exp 2 (held-out patch)"}
FAIL_COL = dict(zip(("success", "WALK_IN", "GRASP", "LIFT", "TRANSPORT", "PLACE", "fell"),
                    ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")))
TIER_TITLE = {"main": "main", "ablation": "ablation", "appendix": "appendix", "extra": "extra: 160 demos, seed 0 only"}
LAYOUT = [[GROUPS[0], GROUPS[1], GROUPS[2], None], [GROUPS[3], GROUPS[4], GROUPS[5], None], GROUPS[6:]]


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches="tight")
    RIO.pyplot().close(fig)
    return name


def _positions():
    """x position per group, with a gap between tiers, and where each tier spans."""
    xs, spans, x, last = [], {}, 0.0, None
    for _m, _s, t in GROUPS:
        if last is not None and t != last:
            x += 0.7
        xs.append(x)
        spans.setdefault(t, [x, x])[1] = x
        x += 1.0
        last = t
    return np.array(xs), spans


def fig_success(summary, by_run, out):
    plt = RIO.pyplot()
    xs, spans = _positions()
    fig, ax = plt.subplots(figsize=(15, 4.8))
    w, top = 0.38, 0.0
    for x, (m, s, _) in zip(xs, GROUPS):
        for j, exp in enumerate(AE.EXPS):
            r = summary[(summary.model == m) & (summary["size"] == s) & (summary.exp == exp)].iloc[0]
            px = x + (j - 0.5) * (w + 0.03)
            ax.bar(px, 100 * r.pooled_rate, w, color=EXP_COL[exp], zorder=2)
            ax.errorbar(px, 100 * r.pooled_rate, yerr=[[100 * (r.pooled_rate - r.ci_lo)], [100 * (r.ci_hi - r.pooled_rate)]],
                        color=RIO.INK, capsize=2.5, lw=1, zorder=3)
            pts = by_run[(by_run.model == m) & (by_run["size"] == s) & (by_run.exp == exp)].rate
            if len(pts) > 1:
                ax.scatter(np.full(len(pts), px), 100 * pts, s=20, facecolors=RIO.SURFACE, edgecolors=RIO.INK,
                           linewidths=0.8, zorder=4)
            ax.text(px, 100 * max(r.ci_hi, pts.max()) + 0.9, "%d/%d" % (r.pooled_successes, r.pooled_n), ha="center",
                    fontsize=6.5, color=RIO.INK_2)
            top = max(top, 100 * max(r.ci_hi, pts.max()))
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace(" (", "\n(") for m, _, _ in GROUPS], fontsize=9)
    ax.set_ylim(0, max(top, 1.0) * 1.16)
    ax.set_ylabel("task success (%)")
    ax.grid(axis="x", visible=False)
    for t, (a, b) in spans.items():
        ax.annotate(TIER_TITLE[t], xy=((a + b) / 2, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=9, color=RIO.INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=EXP_COL[e]) for e in AE.EXPS]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc=RIO.SURFACE, mec=RIO.INK, ms=5))
    ax.legend(handles, [EXP_LEGEND[e] for e in AE.EXPS] + ["one training seed"], fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncol=3)
    fig.suptitle("Task success, pooled over training seeds, with Wilson 95% CI", x=0.08, ha="left", fontsize=11, y=1.0)
    return _save(fig, out, "fig_success.png")


def fig_stages(stages, out):
    plt = RIO.pyplot()
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.2), sharey=True)
    names = list(AE.STAGES) + ["success"]
    for row, items in zip(axes, LAYOUT):
        for ax, item in zip(row, items):
            if item is None:
                ax.axis("off")
                continue
            m, s, _t = item
            for j, exp in enumerate(AE.EXPS):
                r = stages[(stages.model == m) & (stages["size"] == s) & (stages.exp == exp)].iloc[0]
                vals = np.array([100 * r[k] for k in names])
                xs = np.arange(5) + (j - 0.5) * 0.4
                ax.bar(xs, vals, 0.38, color=EXP_COL[exp], label=EXP_LEGEND[exp], zorder=2)
                for x, v in zip(xs, vals):
                    ax.text(x, v + 1.5, "%.0f" % v, ha="center", fontsize=6.5, color=RIO.INK_2)
            ax.set_xticks(range(5))
            ax.set_xticklabels(names, fontsize=7.5)
            ax.set_ylim(0, 112)
            ax.grid(axis="x", visible=False)
            ax.set_title("%s%s" % (m, " (160 demos, seed 0)" if s == "160" else ""), fontsize=9.5, loc="left")
    for row in axes:
        row[0].set_ylabel("episodes (%)", fontsize=9)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", bbox_to_anchor=(0.985, 0.92), fontsize=9)
    fig.suptitle("How far episodes get: each stage implies the one before it", x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig_stages.png")


def fig_failures(fail, out):
    plt = RIO.pyplot()
    keys = ["success", "WALK_IN", "GRASP", "LIFT", "TRANSPORT", "PLACE", "fell"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    ys = np.arange(len(GROUPS))[::-1]
    for ax, exp in zip(axes, AE.EXPS):
        for y, (m, s, _) in zip(ys, GROUPS):
            r = fail[(fail.model == m) & (fail["size"] == s) & (fail.exp == exp)].iloc[0]
            left = 0.0
            for k in keys:
                v = 100.0 * r[k] / r.n
                if v <= 0:
                    continue
                ax.barh(y, v, 0.72, left=left, color=FAIL_COL[k], edgecolor=RIO.SURFACE, linewidth=1.5, zorder=2)
                if v >= 7:
                    ax.text(left + v / 2, y, "%.0f" % v, ha="center", va="center", fontsize=7.5,
                            color="white" if k in ("success", "PLACE", "fell") else RIO.INK)
                left += v
        ax.set_xlim(0, 100)
        ax.set_title(EXP_LEGEND[exp], fontsize=10, loc="left")
        ax.set_xlabel("episodes (%)", fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([m if s == "n40" else "%s, 160 demos" % m for m, s, _ in GROUPS], fontsize=9)
    h = [plt.Rectangle((0, 0), 1, 1, color=FAIL_COL[k]) for k in keys]
    fig.legend(h, ["success"] + ["failed in " + k for k in keys[1:-1]] + ["fell"], loc="lower center", ncol=7, fontsize=8.5,
               bbox_to_anchor=(0.53, -0.05))
    fig.suptitle("Where each episode ended (labelled when at least 7%)", x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig_failure_phase.png")


def fig_proximity(E, out):
    plt = RIO.pyplot()
    fig, axes = plt.subplots(len(PROXIMITY), 2, figsize=(13, 11))
    xs, spans = _positions()
    lines = {"palm_box_min_m": (AE.GRASP_GATE_M, " weld gate\n 0.16 m"), "final_box_goal_m": (0.10, " place radius\n 0.10 m")}
    for i, (metric, short, _what) in enumerate(PROXIMITY):
        for j, exp in enumerate(AE.EXPS):
            ax = axes[i][j]
            data = [E[(E.model == m) & (E["size"] == s) & (E.exp == exp)][metric].dropna().to_numpy(float)
                    for m, s, _ in GROUPS]
            bp = ax.boxplot(data, positions=xs, widths=0.62, patch_artist=True,
                            flierprops=dict(marker="o", ms=2.2, mfc=RIO.INK_2, mec="none", alpha=0.45),
                            medianprops=dict(color=RIO.INK, lw=1.8), whiskerprops=dict(color=RIO.INK_2),
                            capprops=dict(color=RIO.INK_2), boxprops=dict(edgecolor=RIO.INK_2, lw=0.8))
            for patch in bp["boxes"]:
                patch.set_facecolor(EXP_COL[exp])
                patch.set_alpha(0.55)
            ax.set_title("%s (m), %s" % (short.capitalize(), EXP_NAME[exp]), fontsize=9.5, loc="left")
            ax.grid(axis="x", visible=False)
            if metric in lines:
                y, text = lines[metric]
                ax.axhline(y, color=RIO.INK, ls="--", lw=1)
                ax.text(xs[-1] + 0.55, y, text, va="center", fontsize=7.5, color=RIO.INK)
            ax.set_xticks(xs)
            ax.set_xticklabels([m for m, _, _ in GROUPS], fontsize=8, rotation=30, ha="right")
            a, b = spans["extra"]
            ax.axvspan(a - 0.5, b + 0.5, color=RIO.GRID, alpha=0.35, lw=0, zorder=0)
    fig.suptitle("Distances before the weld and at the end. Shaded columns: the 160-demo seed-0 runs. Each box pools its "
                 "training seeds.", x=0.06, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig_proximity.png")


def fig_position(E, out):
    import matplotlib.colors as mc
    plt = RIO.pyplot()
    reg = AE.region()
    xe, ye = AE._grid(reg)
    cmap = mc.LinearSegmentedColormap.from_list("rate", ["#f4f3ef", "#a8cdf2", "#2a78d6", "#123f7a"])
    fig, axes = plt.subplots(3, 4, figsize=(15, 7.6))
    fig.subplots_adjust(hspace=0.42)
    im = None
    for row, items in zip(axes, LAYOUT):
        for ax, item in zip(row, items):
            if item is None:
                ax.axis("off")
                continue
            m, s, _t = item
            g = E[(E.model == m) & (E["size"] == s)]
            tot, _, _ = np.histogram2d(g.box_x, g.box_y, bins=[xe, ye])
            ok, _, _ = np.histogram2d(g.box_x[g.success], g.box_y[g.success], bins=[xe, ye])
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(tot > 0, ok / tot, np.nan)
            im = ax.pcolormesh(ye, xe, rate, cmap=cmap, vmin=0, vmax=1, shading="flat", edgecolors=RIO.SURFACE, linewidth=0.8)
            for a in range(len(xe) - 1):
                for b in range(len(ye) - 1):
                    if tot[a, b]:
                        ax.text((ye[b] + ye[b + 1]) / 2, (xe[a] + xe[a + 1]) / 2, "%d/%d" % (ok[a, b], tot[a, b]), ha="center",
                                va="center", fontsize=5.8, color="white" if rate[a, b] > 0.45 else RIO.INK)
            AE._patch(ax, reg)
            ax.set_xlim(*reg["y"])
            ax.set_ylim(*reg["x"])
            ax.grid(False)
            ax.tick_params(labelsize=7)
            ax.set_title("%s%s, %d seed%s" % (m, " (160 demos)" if s == "160" else "", g.train_seed.nunique(),
                                              "" if g.train_seed.nunique() == 1 else "s"), fontsize=9, loc="left")
    for row in axes:
        row[0].set_ylabel("box x, depth (m)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("box y, lateral (m)", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.015, label="success rate in the cell")
    fig.suptitle("Success by box position, Exp 1 + Exp 2 pooled. Each cell: successes/episodes. Dashed: the held-out patch.",
                 x=0.06, ha="left", fontsize=11)
    return _save(fig, out, "fig_success_by_position.png")


def fig_gap(gap, by_run, out):
    plt = RIO.pyplot()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ys = np.arange(len(GROUPS))[::-1]
    for y, (m, s, _) in zip(ys, GROUPS):
        r = gap[(gap.model == m) & (gap["size"] == s)].iloc[0]
        g, lo, hi = 100 * r.gap, 100 * r.ci_lo, 100 * r.ci_hi
        ax.plot([lo, hi], [y, y], color=RIO.INK_2, lw=1.6, zorder=2)
        ax.scatter([g], [y], s=52, color=EXP_COL["exp1"], edgecolors=RIO.SURFACE, linewidths=1.5, zorder=3)
        text = "0 successes in both: a floor, not a gap" if not (r.exp1_rate or r.exp2_rate) else \
            "%+.1f  (%.1f vs %.1f)%s" % (g, 100 * r.exp1_rate, 100 * r.exp2_rate,
                                         "; per seed %s" % seed_gaps(by_run, m, s) if s == "n40" else "")
        ax.text(hi + 0.5, y, text, va="center", fontsize=8, color=RIO.INK_2)
    ax.axvline(0, color=RIO.INK, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([m if s == "n40" else "%s, 160 demos" % m for m, s, _ in GROUPS], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("generalization gap: Exp 1 rate minus Exp 2 rate, points, with binomial 95% CI")
    ax.set_xlim(right=ax.get_xlim()[1] + 13)
    ax.set_title("Above 0: worse on positions never seen in collection", fontsize=10.5, loc="left")
    fig.tight_layout()
    return _save(fig, out, "fig_generalization_gap.png")


# ─── the report ───────────────────────────────────────────────────────────────
def pct(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.1f" % (100 * x)


def pv(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return "<0.0001" if x < 1e-4 else "%.4f" % x


def md(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def summary_table(s):
    rows = []
    for (m, z, t) in GROUPS:
        for exp in AE.EXPS:
            r = s[(s.model == m) & (s["size"] == z) & (s.exp == exp)].iloc[0]
            sd = "n/a" if r.n_seeds < 2 else "%.1f" % (100 * r.sd_rate)
            rows.append([t, name_of(m, z), EXP_NAME[exp], r.n_seeds, "%.1f +- %s" % (100 * r.mean_rate, sd),
                         "%d/%d" % (r.pooled_successes, r.pooled_n), pct(r.pooled_rate),
                         "%s - %s" % (pct(r.ci_lo), pct(r.ci_hi))])
    return md(rows, ["tier", "model", "exp", "seeds", "mean +- SD (%)", "pooled", "pooled (%)", "Wilson 95% CI (%)"])


def position_table(P):
    return md([[r.exp, "%s vs %s" % (r.A, r.B), "%.2f vs %.2f" % (r.mean_A, r.mean_B), r.favour_a, r.favour_b,
                "%d (%d both 0)" % (r.tie, r.tie_both_zero), pv(r.wilcoxon_p), pv(r.wilcoxon_p_holm), "%+.2f" % r.rank_biserial,
                "%+.2f" % r.mean_diff, pv(r.sign_p), pv(r.sign_p_holm)] for _, r in P.iterrows()],
              ["exp", "A vs B", "mean successes per position (of 3)", "favour A", "favour B", "tie", "Wilcoxon p", "Holm",
               "rank-biserial r", "mean diff (of 3)", "sign test p", "Holm"])


def fisher_rows(F):
    return md([[r.family, r.demos, r.exp, "%s vs %s" % (r.A, r.B), "%d/%d vs %d/%d" % (r.A_success, r.A_n, r.B_success, r.B_n),
                "%+.1f" % (100 * r["diff"]), "n/a" if math.isnan(r.odds_ratio) else "%.3g" % r.odds_ratio,
                pv(r.p), pv(r.p_holm), "%s vs %s" % (r.A_seed_pct, r.B_seed_pct)]
               for _, r in F.iterrows()],
              ["family", "demos", "exp", "pair", "successes", "diff (pts)", "odds ratio", "p", "p (Holm)", "per-seed % (A vs B)"])


def proximity_table(S):
    rows = []
    shorts = [short for _, short, _ in PROXIMITY]
    for (m, z, t) in GROUPS:
        for exp in AE.EXPS:
            g = S[(S.model == m) & (S["size"] == z) & (S.exp == exp)].set_index("metric")
            rows.append([t, name_of(m, z), EXP_NAME[exp]] +
                        ["%.3f [%.3f, %.3f]" % (g.loc[k, "median"], g.loc[k, "q1"], g.loc[k, "q3"]) if k in g.index else ""
                         for k in shorts])
    return md(rows, ["tier", "model", "exp"] + ["%s (m)" % k for k in shorts])


def key_results(T, by_run):
    """The headline reading, computed from the tables rather than typed."""
    s = T["summary"]
    P = T["position"]
    f = T["fisher"]
    d = T["d19"]
    th = T["thr"]
    fl = T["fail"]

    def pair(a, b, e):
        return f[(f.family == "primary") & (f.A == a) & (f.B == b) & (f.exp == e)].iloc[0]

    def posn(a, b, e):
        return P[(P.A == a) & (P.B == b) & (P.exp == e)].iloc[0]

    def pos_text(a, b):
        parts = []
        for e in EXP_NAME.values():
            r = posn(a, b, e)
            parts.append("%s: %d favour %s, %d favour %s, %d tie; Wilcoxon p %s (Holm %s), r = %+.2f; sign test p %s"
                         % (e, r.favour_a, a, r.favour_b, b, r.tie, pv(r.wilcoxon_p), pv(r.wilcoxon_p_holm), r.rank_biserial, pv(r.sign_p)))
        return ". ".join(parts)

    def dr(m, e):
        return d[(d.model == m) & (d["size"] == "n40") & (d.exp == e)].iloc[0]

    def tr(m, e):
        return th[(th.model == m) & (th["size"] == "n40") & (th.exp == e)].iloc[0]

    def fw(m):
        return int(fl[(fl.model == m) & (fl["size"] == "n40") & (fl.exp == "exp2")].iloc[0].WALK_IN)

    x, xb = dr("ACT-LSTM", "exp2"), dr("BC", "exp1")
    a1, a2 = pair("ACT-LSTM", "ACT", "Exp 1"), pair("ACT-LSTM", "ACT", "Exp 2")
    return [
        "- **Absolute rates are low.** The best pooled rate is %s%% and no training seed exceeds %.0f%%."
        % (pct(s.pooled_rate.max()), 100 * by_run.rate.max()),
        "- **ACT-LSTM vs ACT, position by position.** %s." % pos_text("ACT-LSTM", "ACT"),
        "- **ACT-LSTM vs ACT-LSTM-12, position by position.** %s." % pos_text("ACT-LSTM", "ACT-LSTM-12"),
        "- **ACT vs BC, position by position.** %s." % pos_text("ACT", "BC"),
        "- **Pooled, %s:** ACT-LSTM vs ACT %d vs %d of 300 in Exp 1 (Holm p %s), %d vs %d in Exp 2 (Holm p %s), with per-seed rates "
        "%s vs %s in Exp 2." % (INDEPENDENT, a1.A_success, a1.B_success, pv(a1.p_holm), a2.A_success, a2.B_success, pv(a2.p_holm),
                                a2.A_seed_pct, a2.B_seed_pct),
        "- **ACT-LSTM's zero gap is not a stable zero:** per-seed gaps are %s points." % seed_gaps(by_run, "ACT-LSTM", "n40"),
        "- **BC never succeeds, at either batch size or data size.** At 40 demos its palms reach the weld gate in %s%% of Exp 1 "
        "episodes, the weld engages in %d of %d, and %d of those fail the push rule (D19)."
        % (pct(tr("BC", "exp1").palm_within_grasp_gate), xb.welded, xb.n, xb.d19),
        "- **How they fail on the patch.** ACT-LSTM welds in %d of %d Exp 2 episodes, but %d of those fail D19 (the box had moved over "
        "50 mm by the weld). ACT and ACT-12 fail the walk-in in %d and %d of 300, against %d for ACT-LSTM."
        % (x.welded, x.n, x.d19, fw("ACT"), fw("ACT-12"), fw("ACT-LSTM"))]


def write_report(E, R, T, figs, by_run, out):
    L = ["# Final scripted model set: Experiments 1 and 2", ""]
    L.append("**%d runs, %d episodes.** Every result is on SCRIPTED demonstrations: no human data exists yet. "
             "The thesis names follow CLAUDE.md D8, which is **pending adviser approval**." % (len(R), len(E)))
    L += ["", "Input: `runs/scripted_v3/eval`, `runs/scripted_v3_memory_design/eval`. Regenerate with "
          "`python tools/make_final_report.py`.", ""]
    L += ["## Key results", ""] + key_results(T, by_run) + [""]
    L += ["## Which run is which", "", "Each run's model, chunk, window, batch and number of training demos were read back from its "
          "own `metadata.json`.", ""]
    L.append(md([[t, m, EXPECT[m][0], EXPECT[m][2], EXPECT[m][1], EXPECT[m][3], AE.size_n(z),
                  R[(R.model == m) & (R["size"] == z)].shape[0]] for m, z, t in GROUPS],
                ["tier", "thesis name", "code name", "W_o", "K", "batch", "demos", "training seeds"]))
    L += ["BC's two run sets share the code name `bc`: D8's BC is batch 256 (`bc_b256_*` folders), BC (batch 8) is `bc_*`.", "",
          "### Chosen checkpoint per run", "", "By task success on the 20 selection seeds (`tools/eval_scripted.py`).", ""]
    order = R.assign(_r=[RANK[(m, s)] for m, s in zip(R.model, R["size"])]).sort_values(["_r", "train_seed"])
    L.append(md([[name_of(r.model, r["size"]), int(r.train_seed), r.chosen_ckpt, "%d/%d" % (r.selection_success, r.selection_n),
                  "{:,}".format(int(r.train_steps)) if "train_steps" in r and r.train_steps == r.train_steps else "n/a"]
                 for _, r in order.iterrows()], ["model", "seed", "chosen checkpoint", "at selection", "optimizer steps trained"]))
    L += ["## 1. Full task success", "", summary_table(T["summary"]), "### Per training seed", ""]
    L.append(md([[TIER[(r.model, r["size"])], name_of(r.model, r["size"]), int(r.train_seed), EXP_NAME[r.exp],
                  "%d/%d" % (r.successes, r.n), pct(r.rate), "%s - %s" % (pct(r.ci_lo), pct(r.ci_hi))]
                 for _, r in T["by_run"].iterrows()], ["tier", "model", "seed", "exp", "success", "%", "Wilson 95% CI (%)"]))
    L += ["![Task success](%s)" % figs["success"], ""]
    L += ["## 2. Position-level paired test", "",
          "For each of the box positions (the same for every policy), successes over a model's 3 training seeds, 0 to 3, on the 40-demo "
          "runs. Two-sided Wilcoxon signed-rank (normal approximation with tie and continuity corrections; ties dropped) and exact sign "
          "test, over the positions. Holm over the 3 comparisons within each experiment and test. Effect size: matched-pairs rank-biserial "
          "correlation, (W+ - W-) / (W+ + W-), from -1 to +1; positive favours A. **Unit of inference: box positions, for these 3 trained "
          "seeds per model. It does not speak to a new training seed.**", ""]
    L.append(position_table(T["position"]))
    L += ["`position_counts.csv` holds the counts, position by position.", ""]
    L += ["## 3. Pooled Fisher's exact test (%s)" % INDEPENDENT, "",
          "Two-sided, on the table pooled over training seeds, with **episodes treated as independent**: each n = 300 is 3 training seeds "
          "on the same 100 box positions, so this ignores both the between-seed variance and the pairing by position. Section 2 is the test "
          "that keeps the pairing. **Primary:** Holm over those four within each experiment. **Exploratory:** context pairs, Holm within "
          "their own family.", ""]
    L.append(fisher_rows(T["fisher"]))
    L += ["## 4. Stage-wise (phase) success", "", "Percent of episodes (count); each stage implies the one before it.", ""]
    L.append(md([[TIER[(r.model, r["size"])], name_of(r.model, r["size"]), EXP_NAME[r.exp], r.n] +
                 ["%s (%d)" % (pct(r[k]), r[k + "_k"]) for k in list(AE.STAGES) + ["success"]] for _, r in T["stages"].iterrows()],
                ["tier", "model", "exp", "n", "grasp", "lift", "transport", "place", "success"]))
    L += ["![Stage-wise rates](%s)" % figs["stages"], ""]
    L += ["## 5. Failure phase", "", "Counts. The bucket an episode's failure is charged to (`spec.SCORED_OF`).", ""]
    L.append(md([[TIER[(r.model, r["size"])], name_of(r.model, r["size"]), EXP_NAME[r.exp], r.n, r.success, r.WALK_IN, r.GRASP, r.LIFT,
                  r.TRANSPORT, r.PLACE, r.fell] for _, r in T["fail"].iterrows()],
                ["tier", "model", "exp", "n", "success", "WALK_IN", "GRASP", "LIFT", "TRANSPORT", "PLACE", "fell"]))
    L += ["![Failure phase](%s)" % figs["failures"], ""]
    L += ["### Why grasps fail: the push rule (D19)", "", "A grasp needs the box within 50 mm of its spawn when the weld engages. welded = "
          "the weld engaged; failed D19 = it engaged after the box had been pushed further than that; all four = of those, how many still "
          "ended placed, resting, released and upright, an UPPER BOUND on the successes D19 removed (lift and transport are not in the "
          "record).", ""]
    L.append(md([[TIER[(r.model, r["size"])], name_of(r.model, r["size"]), EXP_NAME[r.exp], r.n, r.welded, r.grasp, r.d19,
                  r.d19_all_place_conditions, r.success] for _, r in T["d19"].iterrows()],
                ["tier", "model", "exp", "n", "welded", "grasp", "failed D19", "all four", "success"]))
    L += ["## 6. Proximity", ""]
    L += ["- " + "**%s:** %s" % (short, what) for _, short, what in PROXIMITY]
    L += ["", "Median [first quartile, third quartile], pooled over training seeds.", ""]
    L.append(proximity_table(T["proximity"]))
    L += ["A smaller closest-palm distance is not better by itself: the weld needs BOTH palms within 0.16 m, and one palm can be close "
          "while the other is far. Read it with the farther-palm column.", ""]
    L += ["![Proximity](%s)" % figs["proximity"], "", "### Tests (%s)" % INDEPENDENT, "",
          "Kruskal-Wallis across the three main models, then two-sided Mann-Whitney U with Holm over the family, per metric and experiment, "
          "on episodes pooled over seeds.", ""]
    L.append(md([[r.family, r.metric, r.demos, r.exp, r.test, r.pair, r["median"], "%.4g" % r.stat, pv(r.p), pv(r.p_holm)]
                 for _, r in T["dist"].iterrows()], ["family", "metric", "demos", "exp", "test", "models", "median (m)", "stat", "p", "p (Holm)"]))
    L += ["### Threshold rates", ""]
    L.append(md([[TIER[(r.model, r["size"])], name_of(r.model, r["size"]), EXP_NAME[r.exp], r.n, pct(r.palm_within_grasp_gate),
                  pct(r.final_box_goal_within_10cm), pct(r.final_box_goal_within_5cm), pct(r.closest_box_goal_within_10cm),
                  pct(r.closest_box_goal_within_5cm)] for _, r in T["thr"].iterrows()],
                ["tier", "model", "exp", "n", "both palms in gate (%)", "final <= 10 cm (%)", "final <= 5 cm (%)",
                 "closest box-goal <= 10 cm (%)", "closest box-goal <= 5 cm (%)"]))
    L += ["## 7. Success by box position", "", "![Success by box position](%s)" % figs["position"], ""]
    if figs.get("coverage"):
        L += ["Where the training demonstrations were:", "", "![Training-demo coverage](%s)" % figs["coverage"], ""]
    L += ["## 8. Generalization gap (Exp 1 - Exp 2)", ""]
    L.append(md([[TIER[(r.model, r["size"])], name_of(r.model, r["size"]), pct(r.exp1_rate), pct(r.exp2_rate), "%+.1f" % (100 * r.gap),
                  pct(r.se_binomial), "%+.1f to %+.1f" % (100 * r.ci_lo, 100 * r.ci_hi),
                  "n/a" if r.n_seeds_paired < 2 else "%.1f" % (100 * r.se_seeds), seed_gaps(by_run, r.model, r["size"])]
                 for _, r in T["gap"].iterrows()],
                ["tier", "model", "Exp 1 (%)", "Exp 2 (%)", "gap (pts)", "SE binomial", "95% CI (pts)", "SE over seeds", "gap per seed (pts)"]))
    L += ["A model with 0 successes in both experiments has a gap of 0 by floor, not by generalizing.", "",
          "![Generalization gap](%s)" % figs["gap"], ""]
    L += ["## 9. Data size: 40 vs 160 demos", "", "**Not a clean comparison:** 40-demo rows pool 3 training seeds, 160-demo rows are one "
          "seed. BC runs are epoch-budgeted, so more demos also meant more optimizer steps. ACT and ACT-LSTM have no 160-demo run, so only "
          "the 12-step models and BC appear. Fisher here also treats episodes as independent.", ""]
    L.append(md([[m, EXP_NAME[r.exp], "%d/%d" % (r.k_n40, r.n_n40), pct(r.rate_n40), "%d/%d" % (r.k_160, r.n_160), pct(r.rate_160),
                  "%+.1f" % (100 * r["diff"]), pv(r.fisher_p)] for m in AE.MODELS for _, r in T["sizes"][T["sizes"].model == m].iterrows()],
                ["model", "exp", "40 demos", "%", "160 demos", "%", "diff (pts)", "Fisher p"]))
    L += ["## Caveats, to carry into the write-up", "",
          "- **Scripted only.** The gates and these results are on scripted demonstrations; O31/O32 stay open until piloted data exists.",
          "- **D8 is pending adviser approval**, and it no longer changes one thing per rung: BC keeps W_o = 12 and batch 256, while "
          "ACT and ACT-LSTM use W_o = 1 and batch 8 (NOTES 2026-10-25b).",
          "- **Two levels of inference, neither over training seeds.** The pooled tests treat episodes as independent. The position-level "
          "test fixes that for positions, but its 0-3 counts still come from the same 3 seeds. With 3 seeds per model, an exact permutation "
          "test on the seed rates cannot go below p = 0.1 two-sided, so no test here says what a new training seed would do. Read the "
          "per-seed values beside every p.",
          "- **Selection is noisy.** Checkpoints were chosen on 20 episodes, and the choice differs widely between seeds: ACT-LSTM seed 0 "
          "was chosen at step 20,000 and scored lowest on the patch. Part of the between-seed spread may be selection, not training.",
          "- **Resolution.** At 100 episodes SE is about 5 points near 50% (CLAUDE.md section 8, 2026-09-11).",
          "- **Multiplicity.** Holm is applied within each experiment's family, not across experiments, metrics, tests or families.",
          "- **Evaluation lock rule.** Every run was evaluated under lock rule v3 on demonstrations recorded under v2 "
          "(CLAUDE.md section 8, 2026-10-24).",
          "- **Budgets differ.** ACT-family models ran 200k optimizer steps. BC is epoch-budgeted (300 epochs): BC 25,800 steps, BC (batch 8) "
          "823,200, BC at 160 demos 103,800, BC (batch 8) at 160 demos 3,314,400.",
          "- **BC comparison.** BC ran about 1/8 of the ACT family's optimizer steps at 40 demos, and BC (batch 8) reaches further stages than "
          "BC does. Neither ever succeeds, so the BC row is a floor at either batch size.",
          "- **BC selection is a tie-break.** All 8 BC-family runs scored 0/20 at selection, so the choice fell to the stage counts.",
          "- **ACT-LSTM deployment.** Deployed to 900 ticks, past any training episode, so the LSTM can read history lengths it never saw.",
          "- **The 160-demo runs are seed 0 only**, and there is no 160-demo ACT or ACT-LSTM. Their SD is undefined.",
          "- **Stage 'grasp' includes D19:** the box must still be within 50 mm of its spawn at the weld. See `d19_sensitivity.csv`."]
    text = "\n".join(L) + "\n"
    for junk in ("dtype:", "<bound method", "NaN", "| nan |"):      # a cell that printed a pandas object
        if junk in text:
            raise AssertionError("report cell contains %r" % junk)
    path = os.path.join(out, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def build(roots: Sequence[str] = ROOTS, out: str = OUT, d19_roots: Optional[Sequence[str]] = D19_ROOTS,
          strict: bool = True) -> dict:
    """Everything: tables, figures, report. Returns them, with the episode table."""
    os.makedirs(out, exist_ok=True)
    E, R = AE.load(roots)
    if strict:
        R = audit(E, R)
    used, rejected = AE.attach_grasp_detail(E, list(d19_roots or ()))
    by_run = AE.success_by_run(E)
    summary = AE.success_summary(by_run)
    gap, stages, fail = AE.generalization_gap(by_run), AE.stage_rates(E), AE.failure_phase(E)
    counts = position_counts(E)
    T = dict(by_run=arrange(by_run), summary=arrange(summary), gap=arrange(gap), stages=arrange(stages), fail=arrange(fail),
             thr=arrange(AE.thresholds(E)), d19=arrange(AE.d19_sensitivity(E)), sizes=AE.size_comparison(summary),
             position=position_tests(counts), position_counts=counts, proximity=proximity_summary(E),
             fisher=pd.DataFrame(fisher_family(E, "n40", PRIMARY, "primary") + fisher_family(E, "n40", CONTEXT, "exploratory")
                                 + fisher_family(E, "160", EXTRA_160, "exploratory, 160 demos, seed 0")),
             dist=pd.DataFrame(distance_family(E, "n40", PRIMARY, "primary", ["BC", "ACT", "ACT-LSTM"])
                               + distance_family(E, "n40", CONTEXT, "exploratory")))
    arrange(E).to_csv(os.path.join(out, "episodes.csv"), index=False)
    R.to_csv(os.path.join(out, "runs.csv"), index=False)
    for key, name in (("by_run", "success_by_run"), ("summary", "success_summary"), ("gap", "generalization_gap"),
                      ("stages", "stage_rates"), ("fail", "failure_phase"), ("fisher", "fisher"), ("dist", "distance_tests"),
                      ("thr", "thresholds"), ("sizes", "size_comparison"), ("d19", "d19_sensitivity"),
                      ("position", "position_tests"), ("position_counts", "position_counts"), ("proximity", "proximity_summary")):
        T[key].to_csv(os.path.join(out, name + ".csv"), index=False)
    cov = AE.fig_coverage(AE.coverage(), out, AE.region())
    figs = dict(success=fig_success(summary, by_run, out), stages=fig_stages(stages, out), failures=fig_failures(fail, out),
                proximity=fig_proximity(E, out), position=fig_position(E, out), gap=fig_gap(gap, by_run, out),
                coverage=os.path.basename(cov) if cov else None)
    report = write_report(E, R, T, figs, by_run, out)
    return dict(tables=T, episodes=E, runs=R, figures=figs, report=report, d19=(used, rejected))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", nargs="+", default=list(ROOTS), help="eval folders from tools/eval_scripted.py")
    ap.add_argument("--d19-root", nargs="*", default=list(D19_ROOTS), help="tools/rescore_d19.py output for older records")
    ap.add_argument("--out", default=OUT, help="the report folder")
    ap.add_argument("--no-strict", action="store_true", help="skip the checks that this is exactly the final set")
    a = ap.parse_args(argv)
    res = build(a.eval_root, a.out, a.d19_root, strict=not a.no_strict)
    print("runs %d, episodes %d, D19 re-runs used %d, rejected %d" % (len(res["runs"]), len(res["episodes"]), *res["d19"]))
    print("report: %s" % res["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
