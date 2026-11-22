"""Chapter 4 data-quality tables and figures for the locked human_v2 study and scripted n40.

    python tools/data_quality_report.py          # runs/human_v2/analysis_final/data_quality/

READ ONLY on data and runs; no training, no deployment. Writes only into --out.

WHAT IS DESCRIBED
-----------------
human_v2 train (40) and val (10): the seeds of data/processed/splits_human_v2.json, episodes in
data/processed/episodes_human_v2 (walking labels smoothed, `centred_ma13_v1`, CLAUDE.md section 8
2026-11-19/20). Every human_v2 file names its raw source (`velocity_label_conversion.source`, in
recordings/demonstrations); the raw file is read for the walking command BEFORE smoothing and its
states are checked identical to the human_v2 copy.
scripted n40 train (40): the train seeds of data/processed/splits_scripted_v3_n40.json, episodes in
recordings/scripted_v3 (walking label = the command, `command_v1`).

DEFINITIONS
-----------
length            recorded ticks (25 Hz, 0.04 s each). SD is the sample SD (ddof 1).
ticks per phase   the 5 official phases (D23) over the OFFLINE labels of g1_data/phase_label.py,
                  bucketed by spec.SCORED_OF: walk-in = SETTLE (start -> first base lock); grasp =
                  REACH..GRASP (lock -> weld engage -> lift start); lift; transport = MOVE; place =
                  LOWER + RELEASE + VERIFY. The SAME labeller for both data sets (human episodes
                  carry no phase column, by construction). For scripted the demonstrator's own
                  labels are also tabulated (`recorded`), as a check on the labeller.
tracking dropout  ticks with tracking_ok == 0 / ticks (O28). Scripted measures nothing: n/a.
late ticks        the recorder's `timing.overrun_ticks`: a host-clock tick interval > 60 ms
                  (1.5 ticks). Per data set the tick-weighted % and the worst episode. Scripted
                  runs unpaced: n/a.
peak push         `box_pre_grasp_disturb_mm`: the recorder's peak box displacement before the weld.
placement error   `outcome.placement_error`, m (box to goal centre, xy).
spawn coverage    the 3x3 split bins of g1_data/dataset._spawn_bins over the spawn region (key
                  "i,j": i = x index, low x = nearest the start; j = y index) and the lateral thirds
                  of make_human_report.side_of (y < -0.07 right, -0.07..0.07 centre, > 0.07 left).
action stats      the 16 TRAINABLE dims (spec.TRAINABLE_ACTION_DIMS), raw units, every tick, the
                  labels as trained (human_v2 = smoothed walking dims).
walking command   dims 19-21. ONE definition for all three label sets: a command is ON when
                  |v| >= KEY_MIN (0.1); the smallest key value is 0.2 (backward 0.4 x precision 0.5;
                  locomotion_input.KeyboardCommand). forward vx >= 0.1, backward vx <= -0.1, strafe
                  |vy| >= 0.1, turn |wz| >= 0.1; shares over ALL ticks and over UNLOCKED ticks (state
                  dim 47 = 0; locked ticks are 0 by D17). A PATTERN CHANGE is a tick whose sign
                  pattern (sign of each dim at the threshold) differs from the previous tick; an
                  ON-RUN is a maximal run of ticks with one non-idle pattern.
                  The logged human command is NOT the bare key state: it also carries the teleop
                  loop's station-keeping / heading hold (recorder.TeleopRecorder), up to 0.25 in
                  these files, so the threshold counts some hold ticks as ON. For the RAW human
                  command the key presses are therefore also counted EXACTLY: a dim is a key press
                  when its value is a key value (KEY_VALUES, to 1e-4); any other non-zero value is
                  hold. KEY CHANGES = pattern changes over the exact key presses only.
late ticks        (cont.) the longest tick interval is read from `tick_wall_ns`, reported both with
                  and without the FIRST interval (the acceptance rule ignores the first tick).
attempt ledger    tools/teleop_metrics.attempts_from_rows on recordings/demonstrations/ledger.jsonl
                  (and the scripted_v3 ledger, for completeness); rejected attempts grouped by
                  reason; a discarded SUCCESS's recommendation reasons are tallied.
loss curves       metrics.jsonl of each human_v2 run: train = `recon_l1` (train mode, running
                  mean over the logging window; ACT's `train_loss` adds beta*KL, other units,
                  TR29), val = `val_loss` (train.score_deployment on the val split: eval mode,
                  model(obs) only). Both masked L1 on the 16 normalized trainable dims. BC logs per
                  epoch; its x is epoch x optimizer_steps / epochs (result.json).
selection         <eval>/<run>/selection.json: full successes on the 20 selection seeds per
                  deployed candidate (aliases skipped: same weights); the chosen one ringed.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import report_io as IO                                              # noqa: E402
import report_extras as RX                                          # noqa: E402
import teleop_metrics as TM                                         # noqa: E402

OUT = os.path.join(ROOT, "runs", "human_v2", "analysis_final", "data_quality")
SPLIT_H = os.path.join(ROOT, "data", "processed", "splits_human_v2.json")
SPLIT_S = os.path.join(ROOT, "data", "processed", "splits_scripted_v3_n40.json")
EP_H = os.path.join(ROOT, "data", "processed", "episodes_human_v2")
RAW_H = os.path.join(ROOT, "recordings", "demonstrations")
EP_S = os.path.join(ROOT, "recordings", "scripted_v3")
RUNS = os.path.join(ROOT, "runs", "human_v2")
EVAL = os.path.join(RUNS, "eval")
EVAL_SEEDS = {k: os.path.join(ROOT, "data", "eval", "%s_seeds.json" % k) for k in ("exp1", "exp2")}
PHASES5 = ("walk_in", "grasp", "lift", "transport", "place")
SIDES = ("right", "centre", "left")
KEY_MIN = 0.1
MODELS = (("bc_b256", "BC"), ("act_orig", "ACT"), ("act_lstm_full", "ACT-LSTM"))
SEEDS = (0, 1, 2)
VEL = slice(19, 22)
PATTERN_NAMES = ("forward", "backward", "strafe", "turn")


# ─── small pure helpers (tested) ──────────────────────────────────────────────
def describe(xs) -> dict:
    """mean / SD (ddof 1) / min / max / median / n over the finite values."""
    v = IO.finite(xs)
    if not v:
        return dict(n=0, mean=None, sd=None, min=None, max=None, median=None)
    return dict(n=len(v), mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                min=float(min(v)), max=float(max(v)), median=float(statistics.median(v)))


def key_pattern(vel: np.ndarray, thr: float = KEY_MIN) -> np.ndarray:
    """(T, 3) int8: the sign of vx, vy, wz at the ON threshold (0 = off)."""
    v = np.asarray(vel, dtype=np.float64)
    return (np.sign(v) * (np.abs(v) >= thr)).astype(np.int8)


def key_changes(pat: np.ndarray) -> int:
    """Ticks where the sign pattern differs from the previous tick."""
    p = np.asarray(pat)
    return int((p[1:] != p[:-1]).any(axis=1).sum()) if len(p) > 1 else 0


def on_runs(pat: np.ndarray) -> List[int]:
    """Lengths (ticks) of maximal runs of one NON-idle pattern."""
    p = np.asarray(pat)
    out, n = [], 0
    for t in range(len(p)):
        if p[t].any() and (n == 0 or (p[t] == p[t - 1]).all()):
            n += 1
        else:
            if n:
                out.append(n)
            n = 1 if p[t].any() else 0
    if n:
        out.append(n)
    return out


#: Every value a key can put on one dim (KeyboardCommand: 0.6 forward, 0.4 back, 0.4 strafe, 0.6
#: turn; precision mode halves them; forward + back together = 0.2 or 0.1 in precision).
KEY_VALUES = (0.6, 0.4, 0.3, 0.2, 0.1)


def exact_key_pattern(vel: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """(T, 3) int8: the sign of each dim where its value IS a key value, else 0 (idle or hold)."""
    v = np.asarray(vel, dtype=np.float64)
    is_key = (np.abs(np.abs(v)[..., None] - np.asarray(KEY_VALUES)) <= tol).any(axis=-1)
    return (np.sign(v) * is_key).astype(np.int8)


def command_shares(vel: np.ndarray, thr: float = KEY_MIN) -> Dict[str, np.ndarray]:
    """Per-tick bool masks for the four commands and idle."""
    v = np.asarray(vel, dtype=np.float64)
    m = dict(forward=v[:, 0] >= thr, backward=v[:, 0] <= -thr,
             strafe=np.abs(v[:, 1]) >= thr, turn=np.abs(v[:, 2]) >= thr)
    m["idle"] = ~(m["forward"] | m["backward"] | m["strafe"] | m["turn"])
    return m


def spawn_cell(xy, reg: dict, grid=(3, 3)) -> tuple:
    """g1_data/dataset._spawn_bins' cell for one spawn (same arithmetic)."""
    lo = np.array([reg["x"][0], reg["y"][0]])
    hi = np.array([reg["x"][1], reg["y"][1]])
    f = (np.asarray(xy, dtype=float) - lo) / np.maximum(hi - lo, 1e-9)
    return (int(np.clip(f[0] * grid[0], 0, grid[0] - 1)), int(np.clip(f[1] * grid[1], 0, grid[1] - 1)))


def side_of(y: float, reg: dict) -> str:
    """make_human_report.side_of (the y index of the 3x3 bins)."""
    lo, hi = reg["y"]
    return SIDES[int(np.clip(int((y - lo) / (hi - lo) * 3), 0, 2))]


def bucket_ticks(labels) -> Dict[str, int]:
    """Ticks per official phase from 10-phase labels, via spec.SCORED_OF."""
    from g1_data import spec
    name = {spec.ScoredPhase.WALK_IN: "walk_in", spec.ScoredPhase.GRASP: "grasp", spec.ScoredPhase.LIFT: "lift",
            spec.ScoredPhase.TRANSPORT: "transport", spec.ScoredPhase.PLACE: "place"}
    out = {p: 0 for p in PHASES5}
    for v, n in collections.Counter(int(x) for x in labels).items():
        if v == int(spec.Phase.UNKNOWN):
            raise ValueError("UNKNOWN phase label in the labels to bucket")
        out[name[spec.SCORED_OF[spec.Phase(v)]]] += n
    return out


# ─── loading ──────────────────────────────────────────────────────────────────
def _load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files if k != "meta"}
        d["meta"] = json.loads(str(z["meta"]))
    return d


def _seed_files(folder: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for p in IO.episode_files(folder):
        seed = int(json.loads(str(np.load(p, allow_pickle=False)["meta"]))["seed"])
        if seed in out:
            raise SystemExit("two episodes for seed %d in %s" % (seed, folder))
        out[seed] = p
    return out


def collect(reg: dict) -> List[dict]:
    """One dict per episode of the three described sets (arrays included)."""
    from g1_data import phase_label as PL
    from g1_data.success import geometry
    geo = geometry()
    sh, ss = json.load(open(SPLIT_H, encoding="utf-8")), json.load(open(SPLIT_S, encoding="utf-8"))
    fh, fs = _seed_files(EP_H), _seed_files(EP_S)
    sets = [("human_v2", "train", sh["train"], fh), ("human_v2", "val", sh["val"], fh),
            ("scripted_n40", "train", ss["train"], fs)]
    if len(sh["train"]) != 40 or len(sh["val"]) != 10 or len(ss["train"]) != 40:
        raise SystemExit("unexpected split sizes")
    eps = []
    for data, part, seeds, files in sets:
        for seed in seeds:
            d = _load(files[int(seed)])
            m = d["meta"]
            T = int(d["states"].shape[0])
            raw_vel = None
            if data == "human_v2":
                if m.get("velocity_label") != "centred_ma13_v1":
                    raise SystemExit("%s: not a smoothed human_v2 file" % files[seed])
                src = os.path.join(ROOT, m["velocity_label_conversion"]["source"].replace("\\", os.sep))
                r = _load(src)
                if not np.array_equal(r["states"], d["states"]):
                    raise SystemExit("%s: states differ from the raw source %s" % (files[seed], src))
                raw_vel = r["actions"][:, VEL].astype(np.float64)
            derived, diag = PL.label_episode(d, m, geo)
            row = dict(data=data, part=part, seed=int(seed), file=IO.repo_relpath(files[seed]), ticks=T,
                       seconds=T * IO.TICK_S, monotone_fixes=diag["corrected_by_monotone"])
            row.update({"ticks_" + k: v for k, v in bucket_ticks(derived).items()})
            if data == "scripted_n40":
                row.update({"recorded_ticks_" + k: v for k, v in bucket_ticks(d["phase_labels"]).items()})
            if "tracking_ok" in d:
                row["dropout_ticks"] = int((d["tracking_ok"] == 0).sum())
            tm = m.get("timing") or {}
            if tm.get("paced"):
                w = np.diff(d["tick_wall_ns"].astype(np.int64)) / 1e6
                row.update(late_ticks=int(tm["overrun_ticks"]), max_tick_ms=float(w.max()),
                           first_tick_ms=float(w[0]), max_tick_ms_after_first=float(w[1:].max()),
                           tick_rate_hz=float(tm["tick_rate_hz"]))
            out = m.get("outcome") or {}
            xy = m["box_spawn_xy"]
            cell = spawn_cell(xy, reg)
            row.update(push_mm=m.get("box_pre_grasp_disturb_mm"), placement_m=out.get("placement_error"),
                       outcome_ok=bool(out.get("ok")), spawn_x=float(xy[0]), spawn_y=float(xy[1]),
                       cell="%d,%d" % cell, side=side_of(xy[1], reg), heldout=bool(m.get("heldout")),
                       lock_rule=IO.lock_rule_of(m), velocity_label=m.get("velocity_label", "command_v1"))
            row["_actions"] = d["actions"].astype(np.float64)
            row["_locked"] = d["states"][:, 47] >= 0.5
            row["_raw_vel"] = raw_vel
            eps.append(row)
    for data, part, seeds, _ in sets:            # the split's bins agree with this binning
        split = sh if data == "human_v2" else ss
        for key, members in split["bins"].items():
            mine = {e["seed"] for e in eps if e["data"] == data and e["cell"] == key}
            if not set(int(s) for s in members if s in seeds) <= mine:
                raise SystemExit("%s bin %s disagrees with the split file" % (data, key))
    return eps


# ─── tables ───────────────────────────────────────────────────────────────────
GROUPS = (("human_v2 train", lambda e: e["data"] == "human_v2" and e["part"] == "train"),
          ("human_v2 val", lambda e: e["data"] == "human_v2" and e["part"] == "val"),
          ("human_v2 all", lambda e: e["data"] == "human_v2"),
          ("scripted n40 train", lambda e: e["data"] == "scripted_n40"))


def quality_rows(eps) -> List[dict]:
    rows = []
    for name, sel in GROUPS:
        g = [e for e in eps if sel(e)]
        T = sum(e["ticks"] for e in g)
        r = dict(set=name, episodes=len(g), total_ticks=T, total_s=T * IO.TICK_S)
        for k, v in describe(e["ticks"] for e in g).items():
            r["len_ticks_" + k] = v
        for k, v in describe(e["seconds"] for e in g).items():
            r["len_s_" + k] = v
        for p in PHASES5:
            ds = describe(e["ticks_" + p] for e in g)
            r.update({"%s_ticks_%s" % (p, k): ds[k] for k in ("mean", "sd", "min", "max")})
            r["%s_share_pct" % p] = 100.0 * sum(e["ticks_" + p] for e in g) / T
            if all("recorded_ticks_" + p in e for e in g):
                r["%s_recorded_ticks_mean" % p] = describe(e["recorded_ticks_" + p] for e in g)["mean"]
        meas = [e for e in g if "dropout_ticks" in e]
        if meas:
            r.update(dropout_pct=100.0 * sum(e["dropout_ticks"] for e in meas) / sum(e["ticks"] for e in meas),
                     dropout_episodes=sum(1 for e in meas if e["dropout_ticks"]))
        else:
            r.update(dropout_pct="n/a (not measured)", dropout_episodes="n/a")
        paced = [e for e in g if "late_ticks" in e]
        if paced:
            pct = [100.0 * e["late_ticks"] / e["ticks"] for e in paced]
            r.update(late_pct=100.0 * sum(e["late_ticks"] for e in paced) / sum(e["ticks"] for e in paced),
                     late_pct_episode_max=max(pct), late_pct_episode_median=float(statistics.median(pct)),
                     max_tick_ms=max(e["max_tick_ms"] for e in paced),
                     max_tick_ms_after_first=max(e["max_tick_ms_after_first"] for e in paced),
                     tick_rate_hz_mean=float(np.mean([e["tick_rate_hz"] for e in paced])))
        else:
            r.update(late_pct="n/a (unpaced)", late_pct_episode_max="n/a", late_pct_episode_median="n/a",
                     max_tick_ms="n/a", max_tick_ms_after_first="n/a", tick_rate_hz_mean="n/a")
        for k, v in describe(e["push_mm"] for e in g).items():
            r["push_mm_" + k] = v
        for k, v in describe(1000.0 * e["placement_m"] for e in g if e["placement_m"] is not None).items():
            r["placement_mm_" + k] = v
        r["outcome_ok"] = sum(e["outcome_ok"] for e in g)
        r["lock_rule"] = IO.joined(e["lock_rule"] for e in g)
        r["velocity_label"] = IO.joined(e["velocity_label"] for e in g)
        r["monotone_fixes_total"] = sum(e["monotone_fixes"] for e in g)
        rows.append(r)
    return rows


def coverage_rows(eps, reg) -> List[dict]:
    from analyze_experiments import spawn_xy
    sets = [(n, [(e["spawn_x"], e["spawn_y"]) for e in eps if sel(e)]) for n, sel in GROUPS if n != "human_v2 all"]
    for k, p in EVAL_SEEDS.items():
        sets.append(("Exp %s (eval seeds)" % k[-1], [spawn_xy(s) for s in json.load(open(p))["seeds"]]))
    rows = []
    for name, xys in sets:
        r = dict(set=name, n=len(xys))
        cells = collections.Counter("%d,%d" % spawn_cell(xy, reg) for xy in xys)
        for i in range(3):
            for j in range(3):
                r["cell_%d,%d" % (i, j)] = cells.get("%d,%d" % (i, j), 0)
        sides = collections.Counter(side_of(y, reg) for _, y in xys)
        r.update({s: sides.get(s, 0) for s in SIDES})
        r["in_heldout_patch"] = sum(1 for x, y in xys if reg["heldout_x"][0] <= x <= reg["heldout_x"][1]
                                    and reg["heldout_y"][0] <= y <= reg["heldout_y"][1])
        r["empty_cells"] = sum(1 for i in range(3) for j in range(3) if not cells.get("%d,%d" % (i, j)))
        rows.append(r)
    return rows


def action_rows(eps) -> List[dict]:
    from g1_data import spec
    sets = [("human_v2 train", [e for e in eps if e["data"] == "human_v2" and e["part"] == "train"]),
            ("human_v2 val", [e for e in eps if e["data"] == "human_v2" and e["part"] == "val"]),
            ("scripted n40 train", [e for e in eps if e["data"] == "scripted_n40"])]
    rows = []
    for d in spec.TRAINABLE_ACTION_DIMS:
        r = dict(dim=d, name=spec.ACTION_NAMES[d])
        for name, g in sets:
            a = np.concatenate([e["_actions"][:, d] for e in g])
            r.update({"%s_%s" % (name, k): v for k, v in
                      dict(mean=a.mean(), sd=a.std(ddof=1), min=a.min(), max=a.max()).items()})
        rows.append(r)
    return rows


def walking_rows(eps) -> (List[dict], dict):
    sets = [("human raw command (before smoothing)", [e for e in eps if e["data"] == "human_v2"], "_raw_vel"),
            ("human_v2 label (after smoothing)", [e for e in eps if e["data"] == "human_v2"], "_actions"),
            ("scripted n40 command", [e for e in eps if e["data"] == "scripted_n40"], "_actions")]
    rows = []
    hold = {}
    for name, g, key in sets:
        vels = [(e[key] if key == "_raw_vel" else e[key][:, VEL]) for e in g]
        allv = np.concatenate(vels)
        lock = np.concatenate([e["_locked"] for e in g])
        exact = exact_key_pattern(allv) if key == "_raw_vel" else None
        for scope, mask in (("all ticks", np.ones(len(allv), bool)), ("unlocked ticks", ~lock)):
            sh = command_shares(allv[mask])
            r = dict(command=name, scope=scope, episodes=len(g), ticks=int(mask.sum()))
            r.update({"%s_pct" % k: 100.0 * float(v.mean()) for k, v in sh.items()})
            if exact is not None:
                ex, v = exact[mask], allv[mask]
                r.update(key_forward_pct=100.0 * float((ex[:, 0] > 0).mean()),
                         key_backward_pct=100.0 * float((ex[:, 0] < 0).mean()),
                         key_strafe_pct=100.0 * float((ex[:, 1] != 0).mean()),
                         key_turn_pct=100.0 * float((ex[:, 2] != 0).mean()),
                         key_any_pct=100.0 * float((ex != 0).any(axis=1).mean()),
                         hold_any_pct=100.0 * float(((ex == 0) & (np.abs(v) > 1e-6)).any(axis=1).mean()))
            rows.append(r)
        per = {}
        pats = [key_pattern(v) for v in vels]
        ch = describe(key_changes(p) for p in pats)
        runs = [n for p in pats for n in on_runs(p)]
        per.update({"pattern_changes_per_ep_" + k: ch[k] for k in ("mean", "sd", "min", "max")})
        per.update(on_runs_per_ep_mean=float(np.mean([len(on_runs(p)) for p in pats])),
                   on_run_ticks_median=float(np.median(runs)) if runs else None,
                   on_run_ticks_mean=float(np.mean(runs)) if runs else None)
        if exact is not None:
            ek = [exact_key_pattern(v) for v in vels]
            kc = describe(key_changes(p) for p in ek)
            kr = [n for p in ek for n in on_runs(p)]
            per.update({"key_changes_per_ep_" + k: kc[k] for k in ("mean", "sd", "min", "max")})
            per.update(key_presses_per_ep_mean=float(np.mean([len(on_runs(p)) for p in ek])),
                       key_press_ticks_median=float(np.median(kr)) if kr else None)
            nk = np.abs(allv)[(exact == 0) & (np.abs(allv) > 1e-6)]
            hold = dict(max=float(nk.max()) if nk.size else 0.0, ge_thr=int((nk >= KEY_MIN).sum()),
                        n=int(nk.size), dim_ticks=int(allv.size))
        rows[-1].update(per)
        rows[-2].update(per)
    return rows, hold


def ledger_rows() -> (List[dict], List[dict], dict):
    """Human: rejected attempts by reason; recommendation reasons of discarded successes."""
    rows = [json.loads(l) for l in open(os.path.join(RAW_H, "ledger.jsonl"), encoding="utf-8") if l.strip()]
    att = TM.attempts_from_rows(rows)
    cat = collections.Counter()
    for a in att:
        if a["outcome"] == "accept":
            cat["accepted (in the data set)"] += 1
        elif a["revoked"]:
            cat["revoked after acceptance: " + str(a["reason"]).replace("acceptance revoked: ", "")] += 1
        elif a["outcome"] == "discard" and a["discarded_success"]:
            cat[str(a["reason"]).replace("operator discard of a successful episode",
                                         "task succeeded, operator discarded")] += 1
        elif a["outcome"] == "discard":
            cat["aborted by the operator during the episode (x)"] += 1
        elif a["outcome"] == "failed":
            cat["failed the success rule: %s" % a["reason"].replace("failed: ", "")] += 1
        else:
            cat[a["outcome"]] += 1
    out = [dict(reason=k, attempts=v, pct=100.0 * v / len(att)) for k, v in cat.most_common()]
    rec = collections.Counter()
    for r in rows:
        if r.get("event") == "discard" and (r.get("recommendation") or {}).get("recommend") == "discard":
            for why in r["recommendation"]["reasons"]:
                if "tick overruns" in why:
                    why = "late ticks " + ("> 5%" if "> 5%" in why else "> 2% (rule then)")
                elif "took" in why and "ms" in why:
                    why = "one tick > 150 ms"
                elif "pushed" in why and "before the weld" in why and "rule" not in why:
                    why = "box pushed > 50 mm before the weld"
                elif "success rule FAILED" in why:
                    why = "success rule failed at GRASP"
                rec[why] += 1
    rec_rows = [dict(recommendation_reason=k, episodes=v) for k, v in rec.most_common()]
    aborted = [a["ticks"] * IO.TICK_S for a in att if a["outcome"] == "discard" and not a["discarded_success"]
               and a["ticks"] is not None]
    srows = [json.loads(l) for l in open(os.path.join(EP_S, "ledger.jsonl"), encoding="utf-8") if l.strip()]
    sc = collections.Counter(("accepted" if r["event"] == "accept" else "discarded: " + str(r.get("reason")))
                             for r in srows if r["event"] in ("accept", "discard"))
    extra = dict(attempts=len(att), aborted_s=describe(aborted),
                 scripted=[dict(reason=k, attempts=v) for k, v in sc.most_common()],
                 scripted_issued=sum(1 for r in srows if r["event"] == "issue"))
    return out, rec_rows, extra


# ─── run histories ────────────────────────────────────────────────────────────
def run_histories() -> List[dict]:
    out = []
    for code, label in MODELS:
        for s in SEEDS:
            name = "%s_human40v2_seed%d" % (code, s)
            rd = os.path.join(RUNS, name)
            res = json.load(open(os.path.join(rd, "result.json"), encoding="utf-8"))
            meta = json.load(open(os.path.join(rd, "metadata.json"), encoding="utf-8"))
            hist = RX.metrics_rows(rd)
            sel = json.load(open(os.path.join(EVAL, name, "selection.json"), encoding="utf-8"))
            if res.get("final") is not True or res.get("status") not in ("finished", "hard_cap"):
                raise SystemExit("%s did not finish" % name)
            per_epoch = None
            if "step" not in hist[0]:
                per_epoch = res["optimizer_steps"] / meta["config"]["epochs"]
                if per_epoch != int(per_epoch):
                    raise SystemExit("%s: steps per epoch not whole" % name)
            step = [(h["step"] if per_epoch is None else h["epoch"] * per_epoch) for h in hist]
            cands = []
            for c in sel["candidates"]:
                if c.get("alias_of"):
                    continue
                ci = c.get("ckpt_info") or {}
                if per_epoch is None:
                    st = ci.get("step", c.get("progress"))
                else:
                    st = (ci.get("epoch", c.get("progress"))) * per_epoch
                cands.append(dict(ckpt=c["ckpt"], step=float(st), success=int(c["success"]), n=int(c["n"]),
                                  grasp=int(c["grasp"]), place=int(c["place"]), val_loss=ci.get("val_loss")))
            chosen = [c for c in cands if c["ckpt"] == sel["chosen"]]
            if len(chosen) != 1:
                raise SystemExit("%s: chosen checkpoint %s not among the candidates" % (name, sel["chosen"]))
            out.append(dict(run=name, model=label, seed=s, step=np.array(step, float),
                            train=np.array([h["recon_l1"] for h in hist], float),
                            val=np.array([h["val_loss"] for h in hist], float), cands=sorted(cands, key=lambda c: c["step"]),
                            chosen=chosen[0], steps=res["optimizer_steps"], steps_per_epoch=per_epoch,
                            final_train=hist[-1]["recon_l1"], final_val=hist[-1]["val_loss"],
                            min_val=min(h["val_loss"] for h in hist)))
    return out


# ─── figures ──────────────────────────────────────────────────────────────────
HUMAN_C, SCRIPTED_C = IO.SERIES[0], IO.SERIES[1]
MODEL_C = dict(zip([m for _, m in MODELS], IO.SERIES))


def _save(fig, out, stem) -> List[str]:
    paths = []
    for ext in ("png", "svg"):
        p = os.path.join(out, "%s.%s" % (stem, ext))
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def fig_loss(H, out):
    plt = IO.pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.4), sharey="row")
    for i, (_, m) in enumerate(MODELS):
        for j, s in enumerate(SEEDS):
            ax = axes[i, j]
            h = [x for x in H if x["model"] == m and x["seed"] == s][0]
            ax.plot(h["step"] / 1000, h["train"], color=MODEL_C[m], lw=1.6, label="train (recon L1, train mode)")
            ax.plot(h["step"] / 1000, h["val"], color=IO.INK_2, lw=1.6, ls="--", label="val (deployed L1)")
            ax.axvline(h["chosen"]["step"] / 1000, color=IO.INK, lw=0.9, ls=":", label="selected checkpoint")
            ax.set_yscale("log")
            ax.set_title("%s, seed %d" % (m, s), fontsize=9.5, loc="left")
            if i == 2:
                ax.set_xlabel("optimizer steps (thousands)", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("masked L1, normalized", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MODEL_C[m], lw=1.6, label="train, %s (recon L1, train mode)" % m) for _, m in MODELS]
    handles += [Line2D([], [], color=IO.INK_2, lw=1.6, ls="--", label="val (deployed function, 10 val demos)"),
                Line2D([], [], color=IO.INK, lw=0.9, ls=":", label="selected checkpoint")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.955), ncol=5, fontsize=7.5)
    fig.suptitle("human_v2 training curves, 9 runs: masked L1 on the 16 normalized trainable dims, log scale; "
                 "y shared within a model (row).", x=0.02, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out, "fig_loss_curves")


def fig_selection(H, out):
    plt = IO.pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    styles = {0: ("-", "o"), 1: ("--", "s"), 2: (":", "^")}
    for ax, (_, m) in zip(axes, MODELS):
        for s in SEEDS:
            h = [x for x in H if x["model"] == m and x["seed"] == s][0]
            xs = np.array([c["step"] for c in h["cands"]]) / 1000
            ys = [c["success"] for c in h["cands"]]
            ls, mk = styles[s]
            ax.plot(xs, ys, color=MODEL_C[m], ls=ls, marker=mk, ms=6, lw=1.6, mec=IO.SURFACE, mew=1.2,
                    label="seed %d" % s)
            c = h["chosen"]
            ax.plot([c["step"] / 1000], [c["success"]], ls="none", marker="o", ms=13, mfc="none",
                    mec=IO.INK, mew=1.2)
        ax.set_title(m, fontsize=10, loc="left")
        ax.set_xlabel("optimizer steps (thousands)", fontsize=8.5)
        top = max(c["success"] for h in H for c in h["cands"])
        ax.set_ylim(-0.4, top + 2.5)
        ax.yaxis.set_major_locator(__import__("matplotlib.ticker", fromlist=["MaxNLocator"]).MaxNLocator(integer=True))
        ax.legend(fontsize=7.5, loc="upper right", ncol=3)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("full successes (of 20 selection seeds)", fontsize=8.5)
    fig.suptitle("Checkpoint selection, human_v2: every deployed candidate (aliases skipped); ring = the selected one.\n"
                 "Rule: most successes, then place, transport, lift, grasp, placement error; tie -> least training. "
                 "Axis stops at %d of 20: no candidate exceeds %d." % (top + 2, top),
                 x=0.02, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, out, "fig_selection_success")


def fig_lengths(eps, out):
    plt = IO.pyplot()
    hu = [e["seconds"] for e in eps if e["data"] == "human_v2"]
    sc = [e["seconds"] for e in eps if e["data"] == "scripted_n40"]
    bins = np.arange(np.floor(min(hu + sc)), np.ceil(max(hu + sc)) + 1, 1.0)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True, sharey=True)
    for ax, xs, c, lab in ((axes[0], hu, HUMAN_C, "human_v2 train + val (n=%d)" % len(hu)),
                           (axes[1], sc, SCRIPTED_C, "scripted n40 train (n=%d)" % len(sc))):
        ax.hist(xs, bins=bins, color=c, edgecolor=IO.SURFACE, lw=1.5)
        v = statistics.median(xs)
        ax.axvline(v, color=IO.INK, lw=1.0, ls="--")
        ax.text(v, 1.0, "  median %.1f s" % v, transform=ax.get_xaxis_transform(), va="top", fontsize=7.5,
                color=IO.INK_2)
        ax.set_title(lab, fontsize=9, loc="left")
        ax.set_ylabel("episodes", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("episode length (s; 25 Hz ticks / 25), 1 s bins", fontsize=8.5)
    fig.suptitle("Demonstration length, human vs scripted", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig_episode_length_hist")


def fig_spawn(eps, reg, out):
    from analyze_experiments import spawn_xy
    import matplotlib.patches as mp
    plt = IO.pyplot()
    ex = {k: np.array([spawn_xy(s) for s in json.load(open(p))["seeds"]]) for k, p in EVAL_SEEDS.items()}

    def pts(sel):
        return np.array([(e["spawn_x"], e["spawn_y"]) for e in eps if sel(e)])
    panels = [("human_v2 demonstrations", [("train (40)", pts(GROUPS[0][1]), HUMAN_C, "o"),
                                          ("val (10)", pts(GROUPS[1][1]), SCRIPTED_C, "^")]),
              ("scripted n40 demonstrations", [("train (40)", pts(GROUPS[3][1]), HUMAN_C, "o")]),
              ("evaluation seeds (all policies)", [("Exp 1, in-region (100)", ex["exp1"], HUMAN_C, "o"),
                                                   ("Exp 2, held-out patch (100)", ex["exp2"], SCRIPTED_C, "^")])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (title, series) in zip(axes, panels):
        ax.add_patch(mp.Rectangle((reg["y"][0], reg["x"][0]), reg["y"][1] - reg["y"][0], reg["x"][1] - reg["x"][0],
                                  fill=False, ec=IO.INK_2, lw=1.0))
        ax.add_patch(mp.Rectangle((reg["heldout_y"][0], reg["heldout_x"][0]),
                                  reg["heldout_y"][1] - reg["heldout_y"][0], reg["heldout_x"][1] - reg["heldout_x"][0],
                                  fc=IO.GRID, ec=IO.INK_2, lw=1.0, ls="--", alpha=0.8, zorder=0))
        for v in np.linspace(*reg["y"], 4)[1:-1]:
            ax.axvline(v, color=IO.INK_2, lw=0.6, ls=":")
        for v in np.linspace(*reg["x"], 4)[1:-1]:
            ax.axhline(v, color=IO.INK_2, lw=0.6, ls=":")
        for lab, p, c, mk in series:
            ax.scatter(p[:, 1], p[:, 0], s=34, color=c, marker=mk, edgecolors=IO.SURFACE, linewidths=1.0,
                       label=lab, zorder=3)
        ax.set_xlim(reg["y"][1] + 0.01, reg["y"][0] - 0.01)     # robot's view: its left (+y) on the left
        ax.set_ylim(reg["x"][0] - 0.006, reg["x"][1] + 0.006)
        ax.grid(False)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("box y (m); robot's left <-  -> right", fontsize=8.5)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax.tick_params(labelsize=7.5)
        for k, sname in enumerate(("left", "centre", "right")):
            ax.text(reg["y"][1] - (k + 0.5) * (reg["y"][1] - reg["y"][0]) / 3, reg["x"][1] + 0.003, sname,
                    ha="center", va="bottom", fontsize=7, color=IO.INK_2)
    axes[0].set_ylabel("box x, depth (m)", fontsize=8.5)
    fig.suptitle("Box spawn positions. Solid outline: the spawn region; dotted: the 3x3 split bins (columns = the "
                 "right / centre / left thirds); shaded dashed: the held-out patch (Exp 2).", x=0.02, ha="left",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out, "fig_spawn_coverage")


# ─── markdown ─────────────────────────────────────────────────────────────────
def _f(x, nd=1):
    return IO.fmt(x, nd)


def _msmm(r, p, nd=1):
    """mean ± SD [min, max]"""
    return "%s ± %s [%s, %s]" % (_f(r[p + "mean"], nd), _f(r[p + "sd"], nd), _f(r[p + "min"], nd), _f(r[p + "max"], nd))


def write_md(out, Q, C, A, W, sub, L, REC, LX, H, folders) -> str:
    L_ = ["# Data quality: human_v2 and scripted n40", "",
          "Generated by `python tools/data_quality_report.py` (read-only on data and runs). Definitions are in the "
          "tool's docstring; the CSVs in this folder carry every number at full precision.", "",
          "Sets: **human_v2 train** (40) and **val** (10) = `data/processed/splits_human_v2.json`, files in "
          "`data/processed/episodes_human_v2` (smoothed walking labels; states checked identical to the raw "
          "`recordings/demonstrations` files); **scripted n40 train** (40) = `splits_scripted_v3_n40.json`, files "
          "in `recordings/scripted_v3`. Lengths are 25 Hz ticks (0.04 s).", ""]
    Q4 = [q for q in Q if q["set"] != "human_v2 all"]
    L_ += ["## 1. Data quality", "", "### 1a. Episodes and length", ""]
    L_.append("| set | episodes | total (min) | length, ticks: mean ± SD [min, max] | length, s: mean ± SD [min, max] | median (s) |")
    L_.append("|---|---|---|---|---|---|")
    for q in Q:
        L_.append("| %s | %d | %.1f | %s | %s | %.1f |" % (q["set"], q["episodes"], q["total_s"] / 60,
                                                         _msmm(q, "len_ticks_", 0), _msmm(q, "len_s_", 1), q["len_s_median"]))
    L_ += ["", "### 1b. Ticks per phase (offline labels, `g1_data/phase_label.py`; the five official phases, D23)", "",
           "Cell: mean ± SD [min, max] ticks; share = the phase's ticks / all ticks of the set.", ""]
    L_.append("| set | " + " | ".join(p.replace("_", "-") for p in PHASES5) + " |")
    L_.append("|---|" + "---|" * len(PHASES5))
    for q in Q:
        L_.append("| %s | " % q["set"] + " | ".join(
            "%s (%.1f%%)" % (_msmm(q, "%s_ticks_" % p, 0), q["%s_share_pct" % p]) for p in PHASES5) + " |")
    sq = [q for q in Q if q["set"] == "scripted n40 train"][0]
    L_ += ["", "Labeller check on scripted (the demonstrator's own labels, same buckets), mean ticks: " + ", ".join(
        "%s %s vs %s derived" % (p.replace("_", "-"), _f(sq["%s_recorded_ticks_mean" % p]), _f(sq["%s_ticks_mean" % p]))
        for p in PHASES5) + ". Monotone corrections applied by the labeller: %s." % ", ".join(
        "%s %d" % (q["set"], q["monotone_fixes_total"]) for q in Q4),
        "", "The derived LIFT starts only when the box is %.3f m above rest (`phase_label.LIFT_MIN`), while the "
        "demonstrator enters LIFT when it commands the lift, so derived labels move %.0f scripted ticks (mean) "
        "from lift to grasp; walk-in, transport and place agree to within %.0f ticks (mean). Read lift durations "
        "as box-off-the-platform time, the same rule for both data sets." % (
            0.055, sq["lift_recorded_ticks_mean"] - sq["lift_ticks_mean"],
            max(abs(sq["%s_recorded_ticks_mean" % p] - sq["%s_ticks_mean" % p]) for p in ("walk_in", "transport", "place"))), ""]
    L_ += ["### 1c. Recording quality", "",
           "| set | tracking dropout (% ticks) | episodes with dropout | late ticks (% ticks, > 60 ms) | late, worst episode (%) | "
           "longest tick, after the first / incl. the first (ms) | mean tick rate (Hz) | peak push before weld (mm): mean ± SD [min, max] | "
           "placement error (mm): mean ± SD [min, max] | success rule passed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for q in Q:
        L_.append("| %s | %s | %s | %s | %s | %s / %s | %s | %s | %s | %d/%d |" % (
            q["set"], _f(q["dropout_pct"], 2), _f(q["dropout_episodes"]), _f(q["late_pct"], 2),
            _f(q["late_pct_episode_max"], 2), _f(q["max_tick_ms_after_first"], 0), _f(q["max_tick_ms"], 0),
            _f(q["tick_rate_hz_mean"], 2),
            _msmm(q, "push_mm_", 1), _msmm(q, "placement_mm_", 1), q["outcome_ok"], q["episodes"]))
    L_ += ["", "Scripted episodes run unpaced and measure no tracking: dropout and late ticks do not apply (n/a). "
           "Lock rule stored: " + "; ".join("%s %s" % (q["set"], q["lock_rule"]) for q in Q4) + ".", ""]
    L_ += ["### 1d. Spawn coverage", "", "3x3 split bins, key `i,j`: i = x (depth) index, 0 = nearest the start; "
           "j = y index, 0 = right (most negative y). Sides are the j thirds.", ""]
    cells = ["%d,%d" % (i, j) for i in range(3) for j in range(3)]
    L_.append("| set | n | " + " | ".join(cells) + " | empty bins | right | centre | left | in held-out patch |")
    L_.append("|---|---|" + "---|" * (len(cells) + 5))
    for c in C:
        L_.append("| %s | %d | " % (c["set"], c["n"]) + " | ".join(str(c["cell_" + k]) for k in cells)
                  + " | %d | %d | %d | %d | %d |" % (c["empty_cells"], c["right"], c["centre"], c["left"], c["in_heldout_patch"]))
    L_ += ["", "### 1e. Collection folders (`tools/report_extras.data_quality`)", ""]
    L_.append(IO.md_table(folders, ["dataset", "attempts", "accepted", "rejected", "episodes_on_disk", "len_min_ticks",
                                    "len_median_ticks", "len_max_ticks", "spec_version", "lock_rule", "grip_label",
                                    "source", "dropout_pct", "replay_check"], nd=2))
    L_ += ["", "These are the WHOLE collection folders (human: all 50 accepted; scripted: all 200), not the "
           "training subsets above. `accepted` / `rejected` count SEEDS by their last decision "
           "(`EpisodeLedger.summary`): every human seed was eventually accepted, hence rejected 0; the "
           "rejected ATTEMPTS are in section 3. The replay gate for the human data was run by `tools/pilot_check.py` "
           "(CLAUDE.md section 8 2026-11-17b: replay from the lock 50/50), whose output is not saved in the folder.", ""]
    L_ += ["## 2. Action statistics", "", "### 2a. The 16 trainable action dims (raw units: rad; gripper = weld bit; "
           "m/s, rad/s), every tick, labels as trained", "", "Cell: mean ± SD [min, max].", ""]
    sets = ("human_v2 train", "human_v2 val", "scripted n40 train")
    L_.append("| dim | name | " + " | ".join(sets) + " |")
    L_.append("|---|---|" + "---|" * len(sets))
    for a in A:
        L_.append("| %d | `%s` | " % (a["dim"], a["name"]) + " | ".join(
            "%s ± %s [%s, %s]" % tuple(IO.fmt(a["%s_%s" % (s, k)], 3) for k in ("mean", "sd", "min", "max")) for s in sets) + " |")
    L_ += ["", "### 2b. Walking command (dims 19-21), before and after smoothing", "",
           "One definition for all three label sets: ON = |v| >= %.1f (the smallest key value is 0.2). A pattern "
           "change = a tick whose sign pattern (vx, vy, wz at the threshold) differs from the previous tick; an "
           "on-run = consecutive ticks with one non-idle pattern. Forward + backward + strafe + turn can exceed "
           "100 - idle: a tick can carry two commands. The per-episode columns do not depend on the scope and "
           "repeat on both rows." % KEY_MIN, ""]
    L_.append("| command | ticks | forward % | backward % | strafe % | turn % | idle % | pattern changes / episode: mean ± SD [min, max] | on-runs / episode | on-run median (ticks) |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|")
    for w in W:
        L_.append("| %s, %s | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %s | %.1f | %s |" % (
            w["command"], w["scope"], w["ticks"], w["forward_pct"], w["backward_pct"], w["strafe_pct"], w["turn_pct"],
            w["idle_pct"], _msmm(w, "pattern_changes_per_ep_", 1), w["on_runs_per_ep_mean"], _f(w["on_run_ticks_median"], 0)))
    raw = [w for w in W if "key_any_pct" in w]
    L_ += ["", "**The raw human command, key presses counted exactly.** The logged command is the key state PLUS "
           "the teleop loop's station-keeping / heading hold (`recorder.TeleopRecorder`): %d of %d dim-ticks carry "
           "a non-zero value that is no key value, up to %.2f, and %d of them reach the %.1f threshold, so the "
           "table above counts some hold ticks as commands. Exact key presses (a dim's value is a key value, "
           "`KEY_VALUES`, to 1e-4):" % (sub["n"], sub["dim_ticks"], sub["max"], sub["ge_thr"], KEY_MIN), "",
           "| scope | any key % | forward key % | backward key % | strafe key % | turn key % | hold on some dim % | "
           "key changes / episode: mean ± SD [min, max] | key presses / episode | press median (ticks) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for w in raw:
        L_.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %s | %.1f | %s |" % (
            w["scope"], w["key_any_pct"], w["key_forward_pct"], w["key_backward_pct"], w["key_strafe_pct"],
            w["key_turn_pct"], w["hold_any_pct"], _msmm(w, "key_changes_per_ep_", 1), w["key_presses_per_ep_mean"],
            _f(w["key_press_ticks_median"], 0)))
    L_ += ["", "`hold on some dim %` = a tick where at least one dim carries a non-zero value that is no key value (a key may be held on another dim). The smoothed "
           "human_v2 label has no key state (a moving average of key + hold), so only the threshold table applies "
           "to it; the scripted command is the station-keeper's continuous output.", ""]
    L_ += ["## 3. Attempt ledger", "", "### 3a. Human (`recordings/demonstrations/ledger.jsonl`, "
           "`tools/teleop_metrics.attempts_from_rows`), %d attempts" % LX["attempts"], ""]
    L_.append(IO.md_table(L, ["reason", "attempts", "pct"], ["outcome / reason", "attempts", "% of attempts"], nd=1))
    ab = LX["aborted_s"]
    L_ += ["", "Operator aborts (x during the episode) record no reason; their length when aborted: median %.1f s, "
           "mean %.1f s, [%.1f, %.1f] s. A success counts as `recommended discard` when the recorder's verdict "
           "screen flagged it; those flags:" % (ab["median"], ab["mean"], ab["min"], ab["max"]), ""]
    L_.append(IO.md_table(REC, ["recommendation_reason", "episodes"], ["recorder's discard reason", "episodes"]))
    L_ += ["", "Of the 50 accepted, 10 are val and 40 train (the 54 accept rows include the 4 later revoked, counted "
           "above as revoked).", "",
           "### 3b. Scripted (`recordings/scripted_v3/ledger.jsonl`), %d issued" % LX["scripted_issued"], ""]
    L_.append(IO.md_table(LX["scripted"], ["reason", "attempts"], ["outcome / reason", "attempts"]))
    L_ += ["", "A scripted discard is the demonstrator failing the recorder's checks; the reason lists the checks "
           "that FAILED (`tools/record_episodes.py`). The 8 discarded seeds are not in any split; the n40 train "
           "seeds are all accepted.", ""]
    L_ += ["## 4. Training runs (figures)", "",
           "| run | optimizer steps | final train recon L1 | final val | min val | selected | its successes /20 |",
           "|---|---|---|---|---|---|---|"]
    for h in H:
        L_.append("| %s | %d | %.4f | %.4f | %.4f | %s | %d |" % (h["run"], h["steps"], h["final_train"], h["final_val"],
                                                                h["min_val"], h["chosen"]["ckpt"], h["chosen"]["success"]))
    L_ += ["", "BC is epoch-budgeted (300 epochs x %d steps/epoch); ACT and ACT-LSTM step-budgeted (200k). BC "
           "(K = 1) and ACT / ACT-LSTM (K = 100, a whole chunk) score different targets: compare curves within a "
           "model, not across." % [h for h in H if h["steps_per_epoch"]][0]["steps_per_epoch"], "",
           "Figures (300 dpi PNG + SVG):", "",
           "- `fig_loss_curves`: train and val vs step, all 9 runs.",
           "- `fig_selection_success`: selection successes vs checkpoint, all 9 runs.",
           "- `fig_episode_length_hist`: episode length, human vs scripted.",
           "- `fig_spawn_coverage`: spawn positions, train / val / Exp 1 / Exp 2.", ""]
    p = os.path.join(out, "data_quality.md")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L_))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from analyze_experiments import region
    reg = region()
    eps = collect(reg)
    Q = quality_rows(eps)
    C = coverage_rows(eps, reg)
    A = action_rows(eps)
    W, sub = walking_rows(eps)
    L, REC, LX = ledger_rows()
    H = run_histories()
    folders = [RX.data_quality(RAW_H), RX.data_quality(EP_S)]
    flat = [{k: v for k, v in e.items() if not k.startswith("_")} for e in eps]
    IO.write_csv(os.path.join(a.out, "episodes.csv"), flat)
    IO.write_csv(os.path.join(a.out, "quality.csv"), Q)
    IO.write_csv(os.path.join(a.out, "spawn_coverage.csv"), C)
    IO.write_csv(os.path.join(a.out, "action_stats.csv"), A)
    IO.write_csv(os.path.join(a.out, "walking_commands.csv"), W)
    IO.write_csv(os.path.join(a.out, "attempts_by_reason.csv"), L)
    IO.write_csv(os.path.join(a.out, "discard_recommendations.csv"), REC)
    IO.write_csv(os.path.join(a.out, "collection_folders.csv"), folders, RX.DQ_COLS)
    IO.write_csv(os.path.join(a.out, "selection_candidates.csv"),
                 [dict(run=h["run"], chosen=c["ckpt"] == h["chosen"]["ckpt"], **c) for h in H for c in h["cands"]])
    IO.write_csv(os.path.join(a.out, "runs.csv"), [{k: v for k, v in h.items() if k not in ("step", "train", "val", "cands", "chosen")}
                                                   | dict(chosen=h["chosen"]["ckpt"], chosen_success=h["chosen"]["success"])
                                                   for h in H])
    figs = fig_loss(H, a.out) + fig_selection(H, a.out) + fig_lengths(eps, a.out) + fig_spawn(eps, reg, a.out)
    md = write_md(a.out, Q, C, A, W, sub, L, REC, LX, H, folders)
    print("wrote %s and %d figure files to %s" % (os.path.basename(md), len(figs), IO.repo_relpath(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
