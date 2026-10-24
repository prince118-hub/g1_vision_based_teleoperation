"""Operator metrics from a teleop recording folder (tools/record_teleop_live.py output).

    python tools/teleop_metrics.py recordings/demonstrations
    python tools/teleop_metrics.py recordings/teleop_fake --out runs/teleop_metrics_fake

Reads <folder>/ledger.jsonl and the saved episodes (<folder>/*.npz accepted,
<folder>/failed/*.npz failed); writes into --out (default
runs/teleop_metrics/<folder name>/, gitignored):

  attempts.csv         one row per attempt, in ledger order
  operators.csv        one row per operator, plus ALL
  learning_curve.png   success and duration vs attempt index, one column per operator
  teleop_metrics.md    both tables

READ ONLY, no mujoco / torch (tools/report_io.py). Works on fake-source dry runs
(operator "fake", source teleop_live_fake) exactly as on real ones; the source is
reported so a fake folder is never mistaken for collection data.

DEFINITIONS
-----------
attempt          an `issue` row, closed by the next accept / discard / error row
                 for the same seed. An issue never closed (a crash) is
                 "unfinished". A held-out-spawn `error` with no issue before it is
                 a skipped seed, not an attempt.
operator         the `operator` of the latest `session` row before the issue.
outcome          accept | failed (discard of a SAVED, task-failed episode) |
                 discard (operator discard, nothing written) | error | unfinished.
success rate     accepted / attempted, every outcome in the denominator.
wall_s           decision-row time minus issue-row time. Ledger times have 1 s
                 resolution; the interval includes scoring and saving (~ s) but
                 not the idle screen before ENTER (the issue is written on ENTER).
sim_s            recorded ticks / 25 Hz, from the decision row's `ticks`.
time per accepted demo   total wall_s of ALL the operator's attempts / accepted:
                 what one usable demonstration costs, failures included.
dropout %        degraded ticks / ticks (`tracking_ok`, O28), tick-weighted over
                 SAVED episodes; a discarded attempt wrote no file and is not in it.
tick rate, overruns      the episode's own `timing` block (host clock; an
                 overrun is a tick interval > 1.5 ticks). Headless fake runs are
                 unpaced (`timing.paced` False), so their tick rate is the
                 simulation's speed, not 25 Hz - reported, never compared with live.
learning curve   per operator, attempt index k = 1..n: success (0/1), cumulative
                 success rate, rolling success rate over --window attempts, wall_s.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import report_io as IO                                              # noqa: E402

DECISIONS = ("accept", "discard", "error")


def _t(s) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def _base(p) -> Optional[str]:
    """Basename of a ledger path, written on Windows or POSIX."""
    if not p:
        return None
    return str(p).replace("\\", "/").rsplit("/", 1)[-1]


def outcome_of(row: dict) -> str:
    if row["event"] == "accept":
        return "accept"
    if row["event"] == "error":
        return "error"
    if row.get("saved") or str(row.get("reason", "")).startswith("failed"):
        return "failed"
    return "discard"


def attempts_from_rows(rows: List[dict]) -> List[dict]:
    """The ledger rows as attempts, in order (see DEFINITIONS)."""
    session: dict = {}
    open_: Dict[int, dict] = {}
    out: List[dict] = []
    for r in rows:
        ev = r.get("event")
        if ev == "session":
            session = r
            continue
        if "seed" not in r:
            continue
        seed = int(r["seed"])
        if ev == "issue":
            if seed in open_:                       # re-issued with no decision: a crash
                open_.pop(seed)["outcome"] = "unfinished"
            a = dict(seed=seed, operator=str(session.get("operator") or "n/a"),
                     source=session.get("source") or session.get("label"),
                     t_issue=r.get("t"), outcome="unfinished", wall_s=None, ticks=None,
                     end_reason=None, file=None, reason=None)
            open_[seed] = a
            out.append(a)
        elif ev in DECISIONS and seed in open_:
            a = open_.pop(seed)
            a["outcome"] = outcome_of(r)
            t0, t1 = _t(a["t_issue"]), _t(r.get("t"))
            a["wall_s"] = (t1 - t0).total_seconds() if (t0 and t1) else None
            a["ticks"] = r.get("ticks")
            a["end_reason"] = r.get("end_reason")
            a["reason"] = r.get("reason")
            a["file"] = _base(r.get("path") or r.get("saved"))
    for a in out:
        a["sim_s"] = None if a["ticks"] is None else float(a["ticks"]) * IO.TICK_S
        a["success"] = int(a["outcome"] == "accept")
    return out


def attach_episodes(attempts: List[dict], folder: str) -> None:
    """Per-episode dropout and timing from the saved files the attempts name."""
    files = {os.path.basename(p): p for p in IO.episode_files(folder, include_failed=True)}
    for a in attempts:
        a.update(dropout_pct=None, degraded_ticks=None, tick_rate_hz=None, overrun_ticks=None,
                 paced=None, episode_source=None)
        p = files.get(a["file"]) if a["file"] else None
        if p is None:
            continue
        meta, arr, n = IO.read_episode(p, ("tracking_ok",))
        a["episode_source"] = meta.get("source")
        if a["ticks"] is None:
            a["ticks"], a["sim_s"] = n, n * IO.TICK_S
        if "tracking_ok" in arr:
            ok = np.asarray(arr["tracking_ok"], dtype=bool)
            a["degraded_ticks"] = int((~ok).sum())
            a["dropout_pct"] = 100.0 * a["degraded_ticks"] / max(ok.size, 1)
        tm = meta.get("timing") or {}
        a["tick_rate_hz"] = tm.get("tick_rate_hz")
        a["overrun_ticks"] = tm.get("overrun_ticks")
        a["paced"] = tm.get("paced")


def learning_curve(attempts: List[dict], window: int = 5) -> None:
    """Per-operator attempt index, cumulative and rolling success rate (in place)."""
    by: Dict[str, List[dict]] = {}
    for a in attempts:
        by.setdefault(a["operator"], []).append(a)
    for ops in by.values():
        wins = 0
        for k, a in enumerate(ops, 1):
            wins += a["success"]
            a["k"] = k
            a["cum_success_rate"] = wins / k
            last = ops[max(0, k - window):k]
            a["rolling_success_rate"] = sum(x["success"] for x in last) / len(last)


def summarize(attempts: List[dict], name: str) -> dict:
    n = len(attempts)
    acc = sum(a["success"] for a in attempts)
    walls = IO.finite(a["wall_s"] for a in attempts)
    sims = IO.finite(a["sim_s"] for a in attempts)
    acc_walls = IO.finite(a["wall_s"] for a in attempts if a["success"])
    saved = [a for a in attempts if a.get("degraded_ticks") is not None]
    deg = sum(a["degraded_ticks"] for a in saved)
    tot = sum(int(a["ticks"]) for a in saved if a["ticks"] is not None)
    rates = IO.finite(a.get("tick_rate_hz") for a in attempts)
    over = [int(a["overrun_ticks"]) for a in attempts if a.get("overrun_ticks") is not None]
    count = {o: sum(a["outcome"] == o for a in attempts)
             for o in ("accept", "failed", "discard", "error", "unfinished")}
    return dict(
        operator=name, sources=IO.joined(a["source"] for a in attempts), attempts=n,
        accepted=count["accept"], failed=count["failed"], discarded=count["discard"],
        errors=count["error"], unfinished=count["unfinished"],
        success_rate=(acc / n) if n else None,
        wall_s_per_attempt=float(np.mean(walls)) if walls else None,
        sim_s_per_attempt=float(np.mean(sims)) if sims else None,
        wall_s_per_accepted_attempt=float(np.mean(acc_walls)) if acc_walls else None,
        wall_s_per_accepted_demo=(sum(walls) / acc) if (acc and walls) else None,
        dropout_pct=(100.0 * deg / tot) if tot else None,
        saved_episodes_measured=len(saved),
        tick_rate_hz_mean=float(np.mean(rates)) if rates else None,
        overrun_ticks_total=sum(over) if over else None,
        paced=IO.joined(a.get("paced") for a in attempts))


ATTEMPT_COLS = ["operator", "k", "seed", "outcome", "success", "cum_success_rate",
                "rolling_success_rate", "wall_s", "sim_s", "ticks", "end_reason",
                "dropout_pct", "tick_rate_hz", "overrun_ticks", "paced", "source", "t_issue",
                "file", "reason"]
OP_COLS = ["operator", "sources", "attempts", "accepted", "failed", "discarded", "errors",
           "unfinished", "success_rate", "wall_s_per_attempt", "sim_s_per_attempt",
           "wall_s_per_accepted_attempt", "wall_s_per_accepted_demo", "dropout_pct",
           "saved_episodes_measured", "tick_rate_hz_mean", "overrun_ticks_total", "paced"]


def plot_learning_curve(attempts: List[dict], path: str) -> Optional[str]:
    if not attempts:
        return None
    plt = IO.pyplot()
    from matplotlib.ticker import MaxNLocator
    ops = list(dict.fromkeys(a["operator"] for a in attempts))
    fig, axes = plt.subplots(2, len(ops), figsize=(4.2 * len(ops), 5.6), sharex="col",
                             squeeze=False)
    for j, op in enumerate(ops):
        xs = [a for a in attempts if a["operator"] == op]
        k = [a["k"] for a in xs]
        top, bot = axes[0][j], axes[1][j]
        top.plot(k, [a["cum_success_rate"] for a in xs], color=IO.SERIES[0],
                 label="cumulative")
        top.plot(k, [a["rolling_success_rate"] for a in xs], color=IO.SERIES[1],
                 label="rolling")
        top.scatter(k, [a["success"] for a in xs], s=24, color=IO.INK_2, zorder=3,
                    edgecolor=IO.SURFACE, linewidth=1.2, label="attempt (1 = accepted)")
        top.set_ylim(-0.05, 1.05)
        top.set_title("operator %s (%d attempts)" % (op, len(xs)), fontsize=10, loc="left")
        if j == 0:
            top.set_ylabel("success rate")
        w = [(a["k"], a["wall_s"]) for a in xs if a["wall_s"] is not None]
        if w:
            bot.plot([p[0] for p in w], [p[1] for p in w], color=IO.SERIES[0], marker="o",
                     markersize=4)
        bot.set_ylim(bottom=0)
        bot.set_xlabel("attempt index")
        bot.xaxis.set_major_locator(MaxNLocator(integer=True))
        if j == 0:
            bot.set_ylabel("wall time per attempt (s)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def analyze(folder: str, out: Optional[str] = None, window: int = 5,
            operator: Optional[str] = None) -> dict:
    folder = os.path.abspath(folder)
    led = IO.open_ledger(folder)
    if led is None:
        raise SystemExit("%s has no %s" % (folder, IO.LEDGER_NAME))
    attempts = attempts_from_rows(led.rows())
    if operator:
        attempts = [a for a in attempts if a["operator"] == operator]
    attach_episodes(attempts, folder)
    learning_curve(attempts, window)
    ops = list(dict.fromkeys(a["operator"] for a in attempts))
    table = [summarize([a for a in attempts if a["operator"] == o], o) for o in ops]
    if len(ops) != 1:
        table.append(summarize(attempts, "ALL"))
    out = out or os.path.join(ROOT, "runs", "teleop_metrics", os.path.basename(folder))
    os.makedirs(out, exist_ok=True)
    IO.write_csv(os.path.join(out, "attempts.csv"), attempts, ATTEMPT_COLS)
    IO.write_csv(os.path.join(out, "operators.csv"), table, OP_COLS)
    png = plot_learning_curve(attempts, os.path.join(out, "learning_curve.png"))
    md = ["# Teleoperation metrics: %s" % os.path.basename(folder), "",
          "Source(s): %s. Definitions: tools/teleop_metrics.py docstring." %
          IO.joined(a["source"] for a in attempts), "",
          "## Per operator", "", IO.md_table(table, OP_COLS), "",
          "## Attempts", "", IO.md_table(attempts, ATTEMPT_COLS[:15]), ""]
    if png:
        md += ["![learning curve](learning_curve.png)", ""]
    with open(os.path.join(out, "teleop_metrics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return dict(attempts=attempts, operators=table, out=out, png=png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="recording folder holding ledger.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", type=int, default=5, help="rolling success window (attempts)")
    ap.add_argument("--operator", default=None, help="only this operator's attempts")
    a = ap.parse_args(argv)
    r = analyze(a.folder, a.out, a.window, a.operator)
    for row in r["operators"]:
        print("%-10s attempts %3d  accepted %3d  success %s  s/attempt %s  s/accepted demo %s  "
              "dropout %s%%" % (row["operator"], row["attempts"], row["accepted"],
                                IO.fmt(row["success_rate"], 2), IO.fmt(row["wall_s_per_attempt"], 1),
                                IO.fmt(row["wall_s_per_accepted_demo"], 1),
                                IO.fmt(row["dropout_pct"], 2)))
    print("-> %s" % r["out"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
