"""Readers shared by tools/report_extras.py and tools/teleop_metrics.py.

READ ONLY, AND NO MUJOCO / TORCH IMPORT
---------------------------------------
These tools summarize folders that already exist (recordings, run directories,
eval directories) and must run on any machine, the CPU laptop included. So:

  * the ledger is read through `g1_data/ledger.py` ITSELF - one definition of
    "accepted" / "discarded" / "last decision per seed" - but loaded from its
    file, because `import g1_data.ledger` runs `g1_data/__init__.py`, which
    imports `reset` and with it mujoco;
  * episode files are opened with numpy directly (`meta` is the JSON string
    `EpisodeBuffer.save` writes), not through `recorder.load_episode`, which
    imports mujoco and refuses another spec version. A report has to be able to
    DESCRIBE a folder of 1.1.0 episodes, so it reports the version instead of
    refusing it.

Nothing here writes into a recording or a run directory.
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import math
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_NAME = "ledger.jsonl"            # g1_data/recorder.py LEDGER_NAME
FAILED_DIR = "failed"                   # tools/record_teleop_live.py FAILED_DIR
TICK_S = 0.04                           # 25 Hz record rate (D4), spec.PHYSICS_STEPS_PER_TICK * 0.002

_LEDGER_MOD = None


def ledger_module():
    """g1_data/ledger.py, loaded from its file (see the module docstring)."""
    global _LEDGER_MOD
    if _LEDGER_MOD is None:
        p = os.path.join(ROOT, "g1_data", "ledger.py")
        s = importlib.util.spec_from_file_location("_g1_ledger_ro", p)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        _LEDGER_MOD = mod
    return _LEDGER_MOD


def _paths_module():
    """g1_data/paths.py, loaded from its file (importing g1_data loads mujoco)."""
    s = importlib.util.spec_from_file_location("_g1_paths_ro", os.path.join(ROOT, "g1_data", "paths.py"))
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


#: g1_data.paths.repo_relpath - the repo's one relpath (test_dataset enforces it).
repo_relpath = _paths_module().repo_relpath


def open_ledger(folder: str):
    """The folder's own ledger (`<folder>/ledger.jsonl`), or None if absent.
    Never creates one: `EpisodeLedger.__init__` only makes the directory, which
    exists already, and no row is ever appended here."""
    p = os.path.join(os.path.abspath(folder), LEDGER_NAME)
    if not os.path.exists(p):
        return None
    return ledger_module().EpisodeLedger(p)


# ─── episode files ───────────────────────────────────────────────────────────
def episode_files(folder: str, include_failed: bool = False) -> List[str]:
    """Episode .npz files directly in `folder` (and `failed/` if asked).
    Raw keypoint takes (`keypoints/*_kp.npz`) are never episodes."""
    out = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if include_failed:
        out += sorted(glob.glob(os.path.join(folder, FAILED_DIR, "*.npz")))
    return [p for p in out if not p.endswith("_kp.npz")]


def read_episode(path: str, arrays: Sequence[str] = ("tracking_ok", "tick_wall_ns")):
    """(meta, {name: array}) for the requested arrays that the file HAS, plus
    `n_ticks` (from `states` when present, else meta). Missing arrays are simply
    absent from the dict - absence is information (`tracking_ok` absent = not
    measured, recorder.tracking_measured)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        got = {k: z[k].copy() for k in arrays if k in z.files}
        n = int(z["states"].shape[0]) if "states" in z.files else int(meta.get("n_ticks", 0))
    return meta, got, n


def lock_rule_of(meta: dict) -> str:
    """The stored rule's version. No field = v1 (CLAUDE.md section 8, 2026-10-17);
    `lock_rule: null` = an episode run without a predicate."""
    if "lock_rule" not in meta:
        return "v1 (no field)"
    r = meta["lock_rule"]
    if r is None:
        return "none"
    return str(r.get("version", "?")) if isinstance(r, dict) else str(r)


# ─── small numeric helpers ───────────────────────────────────────────────────
def finite(xs: Iterable) -> List[float]:
    out = []
    for x in xs:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def fmt(x, nd: int = 3, na: str = "n/a") -> str:
    if x is None:
        return na
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    f = float(x)
    return na if not math.isfinite(f) else ("%.*f" % (nd, f))


def joined(values: Iterable, na: str = "n/a") -> str:
    """Sorted distinct values; more than one is shown as 'MIXED: a, b' because a
    dataset is meant to be uniform in every field this is used for."""
    vs = sorted({str(v) for v in values if v is not None})
    if not vs:
        return na
    return vs[0] if len(vs) == 1 else "MIXED: " + ", ".join(vs)


# ─── output ──────────────────────────────────────────────────────────────────
def write_csv(path: str, rows: List[dict], columns: Optional[List[str]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = columns or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return path


def md_table(rows: List[dict], columns: List[str], headers: Optional[List[str]] = None,
             nd: int = 3) -> str:
    hs = headers or columns
    out = ["| " + " | ".join(hs) + " |", "|" + "|".join("---" for _ in hs) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c), nd).replace("|", "/") for c in columns) + " |")
    return "\n".join(out)


# ─── plotting (matplotlib, Agg; the dataviz reference palette, light mode) ───
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
#: Categorical slots 1-3 in FIXED order - the three that validate all-pairs
#: (scatter) in the reference palette. More series are faceted, never cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
        "lines.linewidth": 2.0})
    return plt
