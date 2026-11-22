#!/usr/bin/env python
"""Thesis Chapter 4 and appendix figures: Fig 4.1-4.8, Fig A.1-A.2, and a contact sheet.

    python thesis_figs/make_c4_figs.py            # -> thesis_figs/c4/

READ ONLY. No new analysis: every number comes from a file the analysis already wrote, or is a
re-binning of its per-episode table at the thesis's box sides.

  runs/human_v2/analysis_final/phase_pooled.csv         Fig 4.5, 4.6a, 4.8 (cumulative phase rates,
                                                         Wilson 95% CI; 3 seeds pooled, n = 300)
  runs/human_v2/analysis_final/generalization_gap.csv   Fig 4.6b (Exp 1 - Exp 2, Newcombe 95% CI)
  runs/human_v2/analysis_final/episodes.csv             Fig 4.7, A.1, A.2 (per episode); every pooled
                                                         rate recomputed from it must equal phase_pooled
  runs/human_v2/analysis_final/teleop/sessions.csv      Fig 4.1
  runs/human_v2/<run>/metrics.jsonl, eval/<run>/selection.json    Fig 4.3, 4.4
  data/processed/splits_human_v2.json, splits_scripted_v3_n40.json, data/eval/exp{1,2}_seeds.json
                                                         Fig 4.2 (spawn = box_reset's draw for the seed)

Box side (thesis definition, NOT the 3x3 grid thirds the report uses): Right y < -0.05 m,
Center |y| <= 0.05 m, Left y > 0.05 m.
Training curves: training loss = `recon_l1` (train mode, mean over the logging window; ACT's total
loss adds beta*KL in other units), validation loss = `val_loss` (the deployed function on the
validation demos). BC logs per epoch, ACT / ACT-LSTM per 1000 steps.

Every figure is checked after rendering: no text below 7 pt, no two text boxes overlapping, no
legend over an axes. A failure raises.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
sys.dont_write_bytecode = True

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402
from matplotlib.patches import Patch, Rectangle                         # noqa: E402
from matplotlib.text import Text                                        # noqa: E402

import analyze_experiments as AE                                        # noqa: E402

AN = os.path.join(ROOT, "runs", "human_v2", "analysis_final")
RUNS = os.path.join(ROOT, "runs", "human_v2")
OUT = os.path.join(HERE, "c4")

MODELS = ("BC", "ACT", "ACT-LSTM")
CODE = {"BC": "bc_b256", "ACT": "act_orig", "ACT-LSTM": "act_lstm_full"}
COL = {"ACT-LSTM": "#008300", "ACT": "#1c5cab", "BC": "#d55181"}
SINGLE = "#1c5cab"                       # single-series data marks (Fig 4.1, 4.2)
PATCH = "#e8f1fc"                        # held-out patch fill (Fig 4.2)
GRID = "#e6e6e3"
INK, INK2 = "#1f1f1d", "#52514e"         # text; reference lines (y = +-0.05 m, gap = 0)
RAMP = ("#cde2fb", "#0d366b")            # Fig A.1
FAIL_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b")   # Fig A.2
MIN_GAP_PT = 1.5                         # text boxes closer than this "touch" and fail the check
PHASES = ("walk_in", "grasp", "lift", "transport", "place", "success")
PNAME = {"walk_in": "Walk-in", "grasp": "Grasp", "lift": "Lift", "transport": "Transport", "place": "Place",
         "success": "Full success"}
EXPN = {"exp1": "Experiment 1", "exp2": "Experiment 2"}
SIDE_Y = 0.05
SIDES = ("Right", "Center", "Left")
W = 6.0
HATCH = "////"


def side_of(y: float) -> str:
    return "Right" if y < -SIDE_Y else ("Left" if y > SIDE_Y else "Center")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": "#8a8984", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "legend.frameon": False, "hatch.linewidth": 0.8, "lines.linewidth": 1.2,
        "svg.fonttype": "none", "axes.titlesize": 9, "axes.titleweight": "bold"})


def panel(ax, s):
    ax.set_title(s, loc="left", fontsize=9, fontweight="bold", pad=4)


def pct_axis(ax, label="Success rate (%)"):
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(label)


def fig_(h, **kw):
    if h > 5.5:
        raise SystemExit("figure height %.2f > 5.5 in" % h)
    return plt.subplots(figsize=(W, h), layout="constrained", **kw)


# ─── checks ───────────────────────────────────────────────────────────────────
def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Tick labels exist for ticks outside the view limits too and are never drawn: keep only the
    # labels of ticks inside the view interval.
    all_tick, shown_tick = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            tol = 1e-9 * max(1.0, abs(hi - lo))
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_tick.update((id(tk.label1), id(tk.label2)))
                if lo - tol <= tk.get_loc() <= hi + tol:
                    shown_tick.update((id(tk.label1), id(tk.label2)))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and t.get_window_extent(r).width > 0 and (id(t) not in all_tick or id(t) in shown_tick)]
    small = [(t.get_text(), t.get_fontsize()) for t in texts if t.get_fontsize() < 7]
    if small:
        raise SystemExit("%s: text below 7 pt: %s" % (name, small[:5]))
    gap = MIN_GAP_PT * fig.dpi / 72.0
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    clash = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ov_w = min(a.x1, b.x1) - max(a.x0, b.x0)     # > -gap: closer than MIN_GAP_PT horizontally
            ov_h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ov_w > -gap and ov_h > -gap:
                clash.append((boxes[i][0].get_text(), boxes[j][0].get_text()))
    if clash:
        raise SystemExit("%s: overlapping or touching text: %s" % (name, clash[:6]))
    check_colours(fig, name)
    for leg in fig.legends + [a.get_legend() for a in fig.axes if a.get_legend()]:
        lb = leg.get_window_extent(r)
        for ax in fig.axes:
            if ax.get_label() == "<colorbar>":
                continue
            ab = ax.get_window_extent(r)
            if min(lb.x1, ab.x1) - max(lb.x0, ab.x0) > 1 and min(lb.y1, ab.y1) - max(lb.y0, ab.y0) > 1:
                raise SystemExit("%s: legend overlaps an axes" % name)
    w, h = fig.get_size_inches()
    if abs(w - W) > 1e-6 or h > 5.5 + 1e-6:
        raise SystemExit("%s: size %.2f x %.2f in" % (name, w, h))


def check_colours(fig, name):
    """Every coloured mark inside an axes must be a policy colour, the single-series navy, a ramp step,
    the patch fill, the reference-line colour, the grid colour or white / ink. Nothing grey, red,
    orange or yellow can slip in. The colour-bar image is checked through its ramp ends."""
    import matplotlib.colors as mc
    from matplotlib.collections import Collection
    from matplotlib.patches import Patch as _P
    allowed = {mc.to_hex(c) for c in list(COL.values()) + [SINGLE, PATCH, GRID, INK, INK2, "white"]
               + list(RAMP) + list(FAIL_RAMP)}
    bad = set()

    def look(c):
        try:
            arr = np.atleast_2d(mc.to_rgba_array(c))
        except (ValueError, TypeError):
            return
        for rgba in arr:
            if rgba[3] == 0:
                continue
            h = mc.to_hex(rgba[:3])
            if h not in allowed:
                bad.add(h)
    def look_line(ln):
        """A marker-only line (ls 'none') never draws its line colour: check only what is drawn."""
        if ln.get_linestyle() not in ("None", "none", "", " "):
            look(ln.get_color())
        if ln.get_marker() not in (None, "None", "none", ""):
            look(ln.get_markerfacecolor()), look(ln.get_markeredgecolor())
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        for art in ax.get_children():
            if art is ax.patch or not art.get_visible() or art in ax.spines.values():
                continue
            if isinstance(art, Line2D):
                look_line(art)
            elif isinstance(art, _P):
                look(art.get_facecolor()), look(art.get_edgecolor())
            elif isinstance(art, Collection):
                if art.get_array() is not None:          # colour-mapped (the A.1 heatmap): ramp-checked
                    cm = art.get_cmap()
                    look(cm(0.0)), look(cm(1.0))
                    look(art.get_edgecolor())
                else:
                    look(art.get_facecolor()), look(art.get_edgecolor())
    for leg in fig.legends:
        for hnd in leg.legend_handles:
            if isinstance(hnd, Line2D):
                look_line(hnd)
            elif isinstance(hnd, _P):
                look(hnd.get_facecolor()), look(hnd.get_edgecolor())
    if bad:
        raise SystemExit("%s: colours outside the palette: %s" % (name, sorted(bad)))


def cibar(ax, x, r, lo, hi, color):
    """Wilson CI whiskers in the policy colour, with a thin white outline so the part over a
    same-coloured bar stays visible."""
    import matplotlib.patheffects as pe
    x, r, lo, hi = (np.atleast_1d(np.asarray(v, float)) for v in (x, r, lo, hi))
    cont = ax.errorbar(x, r, yerr=[r - lo, hi - r], fmt="none", ecolor=color, elinewidth=0.9, capsize=1.8,
                       capthick=0.9, zorder=3)
    halo = [pe.Stroke(linewidth=2.3, foreground="white"), pe.Normal()]
    for art in list(cont.lines[1]) + list(cont.lines[2]):
        art.set_path_effects(halo)


def text_on(fill: str) -> str:
    """White or ink, whichever contrasts more with `fill` (WCAG relative luminance)."""
    import matplotlib.colors as mc
    lum = lambda c: sum(w * (v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)   # noqa: E731
                        for w, v in zip((0.2126, 0.7152, 0.0722), mc.to_rgb(c)))
    lf, li = lum(fill), lum(INK)
    return "white" if (1.05 / (lf + 0.05)) > ((lf + 0.05) / (li + 0.05)) else INK


def save(fig, name, out):
    check(fig, name)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)), dpi=300)
    plt.close(fig)
    return os.path.join(out, name + ".png")


# ─── data ─────────────────────────────────────────────────────────────────────
def load():
    E = pd.read_csv(os.path.join(AN, "episodes.csv"))
    P = pd.read_csv(os.path.join(AN, "phase_pooled.csv"))
    G = pd.read_csv(os.path.join(AN, "generalization_gap.csv"))
    E["side_t"] = [side_of(y) for y in E.box_y]
    # the pooled table must equal a recount of the episode table
    for (data, model, exp), g in E.groupby(["data", "model", "exp"]):
        for p in PHASES:
            row = P[(P.data == data) & (P.model == model) & (P.exp == exp) & (P.phase == p)]
            k = int(g["cum_" + p].sum())
            if len(row) != 1 or int(row.k.iloc[0]) != k or int(row.n.iloc[0]) != len(g):
                raise SystemExit("phase_pooled disagrees with episodes.csv: %s %s %s %s" % (data, model, exp, p))
    for _, r in G[G.phase.isin(PHASES)].iterrows():
        a = P[(P.data == r.data) & (P.model == r.model) & (P.phase == r.phase)].set_index("exp")
        if int(a.loc["exp1", "k"]) != r.exp1_k or int(a.loc["exp2", "k"]) != r.exp2_k:
            raise SystemExit("generalization_gap disagrees with phase_pooled")
    return E, P, G


def rate_ci(P, data, model, exp, phase):
    r = P[(P.data == data) & (P.model == model) & (P.exp == exp) & (P.phase == phase)].iloc[0]
    return 100 * r.rate, 100 * r.ci_lo, 100 * r.ci_hi, int(r.k), int(r.n)


def wilson_pct(k, n):
    lo, hi = AE.wilson(int(k), int(n))
    return 100 * k / n, 100 * lo, 100 * hi


def histories():
    H = {}
    for m in MODELS:
        for s in (0, 1, 2):
            name = "%s_human40v2_seed%d" % (CODE[m], s)
            rows = [json.loads(l) for l in open(os.path.join(RUNS, name, "metrics.jsonl"), encoding="utf-8") if l.strip()]
            sel = json.load(open(os.path.join(RUNS, "eval", name, "selection.json"), encoding="utf-8"))
            by_epoch = "step" not in rows[0]
            x = np.array([r["epoch"] if by_epoch else r["step"] / 1000 for r in rows], float)
            cands = []
            for c in sel["candidates"]:
                if c.get("alias_of"):
                    continue
                ci = c["ckpt_info"]
                cands.append(dict(ckpt=c["ckpt"], x=float(ci["epoch"] if by_epoch else ci["step"] / 1000),
                                  success=int(c["success"]), val=float(ci["val_loss"])))
            cands.sort(key=lambda c: c["x"])
            ch = [c for c in cands if c["ckpt"] == sel["chosen"]][0]
            # the selected checkpoint's stored val loss is the logged one at that x: the dot sits on the line
            hit = [r for r, xx in zip(rows, x) if abs(xx - ch["x"]) < 1e-9]
            if len(hit) != 1 or abs(hit[0]["val_loss"] - ch["val"]) > 1e-9:
                raise SystemExit("%s: selected checkpoint's val loss is not on the logged curve" % name)
            H[(m, s)] = dict(x=x, train=np.array([r["recon_l1"] for r in rows]),
                             val=np.array([r["val_loss"] for r in rows]), cands=cands, chosen=ch, by_epoch=by_epoch)
    return H


# ─── figures ──────────────────────────────────────────────────────────────────
def fig_4_1(out):
    S = pd.read_csv(os.path.join(AN, "teleop", "sessions.csv"))
    S = S[S.session != "ALL"].copy()
    S["n"] = [int(str(s).split()[0]) for s in S.session]
    if S.accepted.sum() != 50 or S.attempts.sum() != 289:
        raise SystemExit("sessions.csv totals changed")
    fig, ax = fig_(2.8)
    x = S.n.to_numpy()
    rs = [wilson_pct(k, n) for k, n in zip(S.accepted, S.attempts)]
    r = np.array([v[0] for v in rs])
    ax.bar(x, r, width=0.62, color=SINGLE)
    cibar(ax, x, r, [v[1] for v in rs], [v[2] for v in rs], SINGLE)
    pct_axis(ax, "Accepted attempts (%)")
    ax.set_xticks(x)
    ax.set_xlim(0.4, 11.6)
    for xi, n in zip(x, S.attempts):
        ax.text(xi, -0.13, "n = %d" % n, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7,
                color="#8a8984")
    ax.set_xlabel("Recording session", labelpad=19)
    return save(fig, "Fig_4-1", out)


def fig_4_2(out, layout="stacked"):
    reg = AE.region()
    sh = json.load(open(os.path.join(ROOT, "data", "processed", "splits_human_v2.json")))
    ss = json.load(open(os.path.join(ROOT, "data", "processed", "splits_scripted_v3_n40.json")))
    ev = {k: json.load(open(os.path.join(ROOT, "data", "eval", "%s_seeds.json" % k)))["seeds"] for k in ("exp1", "exp2")}
    xy = lambda seeds: np.array([AE.spawn_xy(int(s)) for s in seeds])          # noqa: E731
    panels = [("(a) Human", [("Training", xy(sh["train"]), "o", "fill"), ("Validation", xy(sh["val"]), "o", "hollow")]),
              ("(b) Scripted", [("Training", xy(ss["train"]), "o", "fill")]),
              ("(c) Evaluation", [("Experiment 1", xy(ev["exp1"]), "s", "fill"), ("Experiment 2", xy(ev["exp2"]), "^", "fill")])]
    size = {"o": 14, "s": 8, "^": 14}                  # Experiment 1 = a SMALL filled square
    msz = {"o": 4.5, "s": 3.4, "^": 4.5}
    if layout == "stacked":
        fig, axes = fig_(5.3, nrows=3, sharex=True)
    else:
        fig, axes = fig_(2.0, ncols=3, sharey=True)
    for ax, (lab, series) in zip(axes, panels):
        ax.add_patch(Rectangle((reg["heldout_y"][0], reg["heldout_x"][0]), reg["heldout_y"][1] - reg["heldout_y"][0],
                               reg["heldout_x"][1] - reg["heldout_x"][0], fc=PATCH, ec="none", zorder=0))
        ax.add_patch(Rectangle((reg["y"][0], reg["x"][0]), reg["y"][1] - reg["y"][0], reg["x"][1] - reg["x"][0],
                               fill=False, ec=INK2, lw=0.8, zorder=1))
        for v in (-SIDE_Y, SIDE_Y):
            ax.axvline(v, color=INK2, lw=0.8, ls="--", zorder=1)
        for _, p, mk, fill in series:
            ax.scatter(p[:, 1], p[:, 0], s=size[mk], marker=mk, linewidths=0.8, zorder=3,
                       facecolors=SINGLE if fill == "fill" else "white", edgecolors=SINGLE)
        ax.set_xlim(reg["y"][1] + 0.012, reg["y"][0] - 0.012)        # robot's left (+y) on the left
        ax.set_ylim(reg["x"][0] - 0.01, reg["x"][1] + 0.01)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_yticks([1.45, 1.50, 1.55])
        panel(ax, lab)
        ax.set_ylabel("Box x (m)")
    # ONE legend: the symbols of all three panels
    seen = {}
    for _, series in panels:
        for n, _, mk, fill in series:
            seen.setdefault(n, Line2D([], [], ls="none", marker=mk, ms=msz[mk], mfc=SINGLE if fill == "fill" else "white",
                                      mec=SINGLE, mew=0.8, label=n))
    seen["Held-out patch"] = Patch(facecolor=PATCH, ec="none", label="Held-out patch")
    fig.legend(handles=list(seen.values()), loc="outside right upper", handletextpad=0.3)
    axes[-1].set_xlabel("Box y (m)")
    if layout != "stacked":
        for ax in axes:
            ax.set_xlabel("Box y (m)")
    return save(fig, "Fig_4-2", out)


def fig_4_3(H, out):
    fig, axes = fig_(2.9, ncols=3)
    for ax, m, lab in zip(axes, MODELS, "abc"):
        for s in (0, 1, 2):
            h = H[(m, s)]
            ax.plot(h["x"], h["train"], color=COL[m], lw=0.8, ls="-")
            ax.plot(h["x"], h["val"], color=COL[m], lw=0.8, ls=(0, (3, 1.6)))
            ax.plot([h["chosen"]["x"]], [h["chosen"]["val"]], ls="none", marker="o", ms=5, mfc=COL[m], mec="white",
                    mew=0.8, zorder=4)
        ax.set_yscale("log")
        import matplotlib.ticker as mt
        ax.yaxis.set_major_locator(mt.LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_formatter(mt.NullFormatter())
        ax.set_xlabel("Epoch" if H[(m, 0)]["by_epoch"] else "Training step (thousands)")
        panel(ax, "(%s) %s" % (lab, m))
    axes[0].set_ylabel("Loss (masked L1)")
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.0, label="Training loss"),
                        Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.6)), label="Validation loss"),
                        Line2D([], [], ls="none", marker="o", ms=5, mfc=INK, mec="white", mew=0.8, label="Selected checkpoint")],
               loc="outside upper center", ncol=3)
    return save(fig, "Fig_4-3", out)


def fig_4_4(H, out):
    fig, axes = fig_(2.9, ncols=3, sharey=True)
    marks = {0: "o", 1: "s", 2: "^"}
    top = max(c["success"] for h in H.values() for c in h["cands"])
    if top > 7:
        raise SystemExit("a candidate exceeds the 0-7 axis")
    for ax, m, lab in zip(axes, MODELS, "abc"):
        for s in (0, 1, 2):
            h = H[(m, s)]
            ax.plot([c["x"] for c in h["cands"]], [c["success"] for c in h["cands"]], color=COL[m], lw=0.8,
                    marker=marks[s], ms=2.8, mew=0)
            ax.plot([h["chosen"]["x"]], [h["chosen"]["success"]], ls="none", marker="o", ms=7, mfc=COL[m], mec="white",
                    mew=0.9, zorder=4)
        ax.set_ylim(-0.3, 7)
        ax.set_yticks(range(0, 8))
        ax.set_xlabel("Epoch" if H[(m, 0)]["by_epoch"] else "Training step (thousands)")
        panel(ax, "(%s) %s" % (lab, m))
    axes[0].set_ylabel("Successes (of 20)")
    fig.legend(handles=[Line2D([], [], color=INK, lw=0.8, marker=marks[s], ms=3.5, mew=0, label="Seed %d" % s)
                        for s in (0, 1, 2)]
               + [Line2D([], [], ls="none", marker="o", ms=7, mfc=INK, mec="white", mew=0.9, label="Selected checkpoint")],
               loc="outside upper center", ncol=4)
    return save(fig, "Fig_4-4", out)


def phase_bars(ax, P, exp, labels=True):
    x = np.arange(len(PHASES))
    w = 0.29
    for i, m in enumerate(MODELS):
        v = [rate_ci(P, "human", m, exp, p) for p in PHASES]
        r = np.array([a[0] for a in v])
        xs = x + (i - 1) * w
        ax.bar(xs, r, width=w * 0.92, color=COL[m], label=m)
        cibar(ax, xs, r, [a[1] for a in v], [a[2] for a in v], COL[m])
        if labels:
            ax.text(xs[-1], v[-1][2] + 2, "%.1f" % r[-1], ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [PNAME[p] for p in PHASES])
    pct_axis(ax)


def fig_4_5(P, out):
    fig, ax = fig_(2.9)
    phase_bars(ax, P, "exp1")
    fig.legend(handles=[Patch(color=COL[m], label=m) for m in MODELS], loc="outside upper center", ncol=3)
    return save(fig, "Fig_4-5", out)


def fig_4_6(P, G, out):
    fig, (a, b) = fig_(5.5, nrows=2, height_ratios=[1, 1.08])
    phase_bars(a, P, "exp2")
    panel(a, "(a)")
    g = G[(G.data == "human") & G.phase.isin(PHASES)]
    off = {"BC": 0.25, "ACT": 0.0, "ACT-LSTM": -0.25}
    for k, p in enumerate(PHASES):
        y0 = len(PHASES) - 1 - k
        for m in MODELS:
            r = g[(g.model == m) & (g.phase == p)].iloc[0]
            y = y0 + off[m]
            b.plot([100 * r.newcombe_lo, 100 * r.newcombe_hi], [y, y], color=COL[m], lw=1.2, solid_capstyle="butt")
            hollow = bool(r.floor)
            b.plot([100 * r.gap], [y], ls="none", marker="o", ms=4.5, mfc="white" if hollow else COL[m], mec=COL[m],
                   mew=1.0, zorder=3)
    b.axvline(0, color=INK2, lw=0.8)
    b.set_yticks(range(len(PHASES)), [PNAME[p] for p in reversed(PHASES)])
    b.set_ylim(-0.6, len(PHASES) - 0.4)
    b.grid(False)
    for yy in np.arange(len(PHASES) - 1) + 0.5:
        b.axhline(yy, color=GRID, lw=0.7, zorder=0)
    b.set_xlabel("Generalization gap, Experiment 1 minus Experiment 2 (percentage points)")
    panel(b, "(b)")
    fig.legend(handles=[Patch(color=COL[m], label=m) for m in MODELS]
               + [Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=COL["BC"], label="BC, 0% in both")],
               loc="outside upper center", ncol=4)
    return save(fig, "Fig_4-6", out)


def side_rates(E):
    H = E[E.data == "human"]
    out = {}
    for m in MODELS:
        for s in SIDES:
            g = H[(H.model == m) & (H.side_t == s)]
            out[(m, s)] = wilson_pct(int(g.cum_walk_in.sum()), len(g)) + (len(g),)
    return out


def fig_4_7(E, out):
    R = side_rates(E)
    fig, ax = fig_(2.8)
    x = np.arange(len(SIDES))
    w = 0.26
    for i, m in enumerate(MODELS):
        r = np.array([R[(m, s)][0] for s in SIDES])
        xs = x + (i - 1) * w
        ax.bar(xs, r, width=w * 0.92, color=COL[m], label=m)
        cibar(ax, xs, r, [R[(m, s)][1] for s in SIDES], [R[(m, s)][2] for s in SIDES], COL[m])
    ax.set_xticks(x, SIDES)
    ax.set_xlabel("Box side")
    pct_axis(ax, "Walk-in rate (%)")
    fig.legend(handles=[Patch(color=COL[m], label=m) for m in MODELS], loc="outside upper center", ncol=3)
    return save(fig, "Fig_4-7", out)


def fig_4_8(P, out):
    fig, axes = fig_(2.9, ncols=2, sharey=True)
    x = np.arange(len(MODELS))
    w = 0.36
    for ax, exp, lab in zip(axes, ("exp1", "exp2"), "ab"):
        for j, data in enumerate(("human", "scripted")):
            for i, m in enumerate(MODELS):
                r, lo, hi, k, n = rate_ci(P, data, m, exp, "success")
                xi = x[i] + (j - 0.5) * w
                if data == "human":
                    ax.bar(xi, r, width=w * 0.9, color=COL[m], ec=COL[m], lw=0.8)
                else:
                    ax.bar(xi, r, width=w * 0.9, facecolor="white", ec=COL[m], hatch=HATCH, lw=0.8)
                cibar(ax, [xi], [r], [lo], [hi], COL[m])
                ax.text(xi, hi + 2, "%.1f" % r, ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x, MODELS)
        pct_axis(ax, "Full success (%)")
        panel(ax, "(%s) %s" % (lab, EXPN[exp]))
    axes[1].set_ylabel("")
    fig.legend(handles=[Patch(facecolor=INK, ec=INK, label="Human"),
                        Patch(facecolor="white", ec=INK, hatch=HATCH, label="Scripted")],
               loc="outside upper center", ncol=2)
    return save(fig, "Fig_4-8", out)


def fig_a_1(E, out):
    import matplotlib.colors as mc
    reg = AE.region()
    xe, ye = AE._grid(reg)
    cmap = mc.LinearSegmentedColormap.from_list("blue", list(RAMP))
    cmap.set_bad("white")
    H = E[E.data == "human"]
    fig, axes = fig_(5.2, nrows=2, ncols=3, sharex=True, sharey=True)
    im = None
    for row, (col, rname) in zip(axes, (("cum_success", "Full success"), ("cum_walk_in", "Walk-in"))):
        for ax, m in zip(row, MODELS):
            g = H[H.model == m]
            tot, _, _ = np.histogram2d(g.box_x, g.box_y, bins=[xe, ye])
            ok, _, _ = np.histogram2d(g.box_x[g[col]], g.box_y[g[col]], bins=[xe, ye])
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.ma.masked_invalid(np.where(tot > 0, 100 * ok / tot, np.nan))
            im = ax.pcolormesh(ye, xe, rate, cmap=cmap, vmin=0, vmax=100, shading="flat", edgecolors="white", lw=0.6)
            ax.add_patch(Rectangle((reg["heldout_y"][0], reg["heldout_x"][0]), reg["heldout_y"][1] - reg["heldout_y"][0],
                                   reg["heldout_x"][1] - reg["heldout_x"][0], fill=False, ec=INK, lw=1.1, zorder=3))
            for v in (-SIDE_Y, SIDE_Y):
                ax.axvline(v, color=INK2, lw=0.8, ls="--", zorder=3)
            ax.set_xlim(reg["y"][1], reg["y"][0])
            ax.set_ylim(*reg["x"])
            ax.grid(False)
            ax.set_xticks([0.2, 0.1, 0.0, -0.1, -0.2])
            ax.set_yticks([1.45, 1.50, 1.55])
            panel(ax, "%s, %s" % (m, rname))
        row[0].set_ylabel("Box x (m)")
    for ax in axes[-1]:
        ax.set_xlabel("Box y (m)")
    for k, ax in enumerate(axes.ravel()):
        ax.set_title("(%s) %s" % ("abcdef"[k], ax.get_title(loc="left")), loc="left", fontsize=9, fontweight="bold", pad=4)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Rate (%)")
    cb.outline.set_visible(False)
    return save(fig, "Fig_A-1", out)


FAIL = tuple((c, n, colr) for (c, n), colr in zip(
    (("WALK_IN", "Walk-in"), ("GRASP", "Grasp"), ("LIFT", "Lift"), ("TRANSPORT", "Transport"), ("PLACE", "Place"),
     ("success", "Full success")), FAIL_RAMP))


def fig_a_2(E, out):
    H = E[E.data == "human"]
    known = {c for c, _, _ in FAIL}
    if set(H.first_failure) - known:
        raise SystemExit("unexpected first_failure values: %s" % (set(H.first_failure) - known))
    bars = []
    for m in MODELS:
        for exp in ("exp1", "exp2"):
            g = H[(H.model == m) & (H.exp == exp)]
            bars.append((m, exp, [100 * (g.first_failure == c).mean() for c, _, _ in FAIL]))
    ys = []
    y = 0.0
    for i, (m, exp, _) in enumerate(bars):
        ys.append(y)
        y -= 1.0 if exp == "exp1" else 1.5
    fig, ax = fig_(2.9)
    for yy, (m, exp, shares) in zip(ys, bars):
        left = 0.0
        for (c, name, colr), s in zip(FAIL, shares):
            ax.barh(yy, s, left=left, height=0.8, color=colr, ec="white", lw=0.6)
            if s >= 10:
                ax.text(left + s / 2, yy, "%.0f%%" % s, ha="center", va="center", fontsize=7, color=text_on(colr))
            left += s
    ax.set_yticks(ys, ["%s, %s" % (m, EXPN[e]) for m, e, _ in bars])
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Episodes (%), by first phase not reached")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(color=colr, label=name) for _, name, colr in FAIL], loc="outside upper center", ncol=6,
               handlelength=1.2, columnspacing=1.0)
    return save(fig, "Fig_A-2", out)


def contact_sheet(paths, out):
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p).convert("RGB") for p in paths]
    cols, cw = 2, 1800
    pad, cap = 40, 70
    rows = (len(ims) + cols - 1) // cols
    cells = [im.resize((cw, int(im.height * cw / im.width))) for im in ims]
    rh = [max(c.height for c in cells[r * cols:(r + 1) * cols]) for r in range(rows)]
    sheet = Image.new("RGB", (cols * cw + (cols + 1) * pad, sum(rh) + rows * (cap + pad) + pad), "white")
    try:
        font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf"), 44)
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(sheet)
    y = pad
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            if k >= len(cells):
                break
            x = pad + c * (cw + pad)
            d.text((x, y), os.path.splitext(os.path.basename(paths[k]))[0].replace("_", " "), fill=(31, 31, 29), font=font)
            sheet.paste(cells[k], (x, y + cap))
            d.rectangle([x - 1, y + cap - 1, x + cw, y + cap + cells[k].height], outline=(200, 200, 196), width=2)
        y += rh[r] + cap + pad
    p = os.path.join(out, "contact_sheet.png")
    sheet.save(p, dpi=(300, 300))
    return p


def main() -> int:
    style()
    os.makedirs(OUT, exist_ok=True)
    E, P, G = load()
    H = histories()
    paths = [fig_4_1(OUT), fig_4_2(OUT), fig_4_3(H, OUT), fig_4_4(H, OUT), fig_4_5(P, OUT), fig_4_6(P, G, OUT),
             fig_4_7(E, OUT), fig_4_8(P, OUT), fig_a_1(E, OUT), fig_a_2(E, OUT)]
    print("wrote", contact_sheet(paths, OUT), "and", len(paths), "figures (PNG + SVG)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
