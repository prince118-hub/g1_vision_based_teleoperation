"""Open one saved episode (.npz) and show the real numbers, with their names.

    python tools/inspect_episode.py dims
    python tools/inspect_episode.py info  recordings/scripted_v3/ep_seed0000.npz
    python tools/inspect_episode.py table recordings/scripted_v3/ep_seed0000.npz --ticks 195:200
    python tools/inspect_episode.py stats recordings/scripted_v3/ep_seed0000.npz --dims action
    python tools/inspect_episode.py csv   recordings/scripted_v3/ep_seed0000.npz --out ep0.csv
    python tools/inspect_episode.py plot  recordings/scripted_v3/ep_seed0000.npz --out ep0.png

READ-ONLY. Works on scripted and teleop episodes. Every name, index and mask comes
from g1_data/spec.py; the units are added here. See docs/DATA_FLOW.md for what each
number means.

--dims takes names (`a_vx`), globs (`box_pos_*`), indices (`s28`, `a17`), the groups
of spec.py (`box_pos`, `arm_left`, `velocity`) or `state`, `action`, `aux`, `all`.
`dims` lists them. --ticks is START:END with END excluded, like a Python slice.

PHASES. A scripted episode stores its demonstrator's phase per tick. A teleop episode
stores UNKNOWN (no live classifier exists), so its phases are derived here with
g1_data.phase_label, which loads the MuJoCo scene for the platform geometry, and the
output says which of the two you are reading.

OLDER FILES. An older 47-D file (g1-spec-1.1.0, no `base_locked`) opens with a
warning: its lock is read from the metadata, first lock only, and its gripper action
is the command sampled at the tick rather than the next weld bit. The dataset code
refuses these files on purpose; this tool only reads them.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from g1_data import phase_label as PL                               # noqa: E402
from g1_data import recorder as REC                                 # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402
import report_io as RIO                                             # noqa: E402

TICK_S = 1.0 / spec.CONTROL_HZ

#: Units per spec.py group. spec.py defines the groups; it does not state units.
STATE_UNITS = dict(box_pos="m", box_quat="wxyz", base_pos="m", base_quat="wxyz",
                   palmL_pos="m", palmL_quat="wxyz", palmR_pos="m", palmR_quat="wxyz",
                   gripper="0/1", arm_left_q="rad", arm_right_q="rad", waist_q="rad",
                   base_locked="0/1")
ACTION_UNITS = dict(arm_left="rad", arm_right="rad", waist="rad", gripper="0/1",
                    velocity=None)                 # per dim: see VELOCITY_UNITS
VELOCITY_UNITS = dict(a_vx="m/s", a_vy="m/s", a_wz="rad/s")
#: Per-tick arrays beyond the two vectors, shown when the file has them.
AUX_UNITS = dict(grip_cmd="0/1", tracking_ok="0/1", frame_seq="count")

DEFAULT_TABLE_TICKS = 20
DEFAULT_TABLE_DIMS = ("box_pos_x", "box_pos_y", "box_pos_z", "base_pos_x", "base_pos_y",
                      "a_vx", "a_vy", "a_wz")
DEFAULT_PLOT_DIMS = ("box_pos", "base_pos", "velocity", "q_left_elbow", "q_right_elbow")
#: Above this many dims a table lists one dim per row and one tick per column.
HORIZONTAL_MAX_DIMS = 8
VERTICAL_TICKS_PER_BLOCK = 10
#: The fixed categorical slots validate for 3 series; more are faceted, never cycled.
MAX_SERIES_PER_PANEL = len(RIO.SERIES)
MAX_PANELS = 24
LOCK_TINT, WELD_TINT, TRACKING_TINT = RIO.INK_2, "#eda100", "#eb6834"
PHASE_TINTS = ("#e4e3df", "#f0efec")


class InspectError(Exception):
    """A file or a request this tool cannot honour. The CLI prints it and exits 2."""


@dataclass(frozen=True)
class Dim:
    """One named per-tick column."""
    kind: str               # "state" | "action" | "aux"
    index: int              # position in its vector; -1 for aux
    name: str
    unit: str
    group: str
    masked: bool            # excluded from the loss and from normalization


def state_dims(width: int = spec.STATE_DIM) -> List[Dim]:
    out = []
    for group, sl in spec.STATE_GROUPS:
        for i in range(*sl.indices(spec.STATE_DIM)):
            if i < width:
                out.append(Dim("state", i, spec.STATE_NAMES[i], STATE_UNITS[group], group,
                               not bool(spec.STATE_MASK[i])))
    return sorted(out, key=lambda d: d.index)


def action_dims() -> List[Dim]:
    out = []
    for group, sl in spec.ACTION_GROUPS:
        for i in range(*sl.indices(spec.ACTION_DIM)):
            name = spec.ACTION_NAMES[i]
            out.append(Dim("action", i, name, ACTION_UNITS[group] or VELOCITY_UNITS[name], group,
                           not bool(spec.ACTION_MASK[i])))
    return sorted(out, key=lambda d: d.index)


@dataclass
class Episode:
    """One opened file: the arrays, plus the per-tick lock, weld and phase columns."""
    path: str
    meta: dict
    arrays: Dict[str, np.ndarray]
    dims: List[Dim]
    n_ticks: int
    phases: np.ndarray
    phase_origin: str
    locked: np.ndarray
    welded: np.ndarray
    tracking: Optional[np.ndarray]
    warnings: List[str] = field(default_factory=list)

    def column(self, d: Dim) -> np.ndarray:
        if d.kind == "state":
            return self.arrays["states"][:, d.index]
        if d.kind == "action":
            return self.arrays["actions"][:, d.index]
        return self.arrays[d.name]

    @property
    def source_label(self) -> str:
        info = REC.SOURCES.get(self.meta.get("source"))
        return info["label"] if info else "unregistered: %r" % (self.meta.get("source"),)


def _ticks_from_meta(meta: dict, n: int) -> np.ndarray:
    """The lock column of a file with no `base_locked` dim: first lock only."""
    on = np.zeros(n, dtype=bool)
    lo, hi = int(meta.get("lock_engage_tick", -1)), int(meta.get("lock_release_tick", -1))
    if lo >= 0:
        on[lo:(hi if hi >= 0 else n)] = True
    return on


def _phases_of(arrays: dict, meta: dict, warnings: List[str]) -> Tuple[np.ndarray, str]:
    """The stored labels if the file has any, else labels derived offline."""
    stored = np.asarray(arrays["phase_labels"], dtype=np.int8)
    if (stored != int(spec.Phase.UNKNOWN)).any():
        return stored, "recorded by the demonstrator"
    try:
        derived, _ = PL.label_episode(arrays, meta)
    except Exception as e:                                      # noqa: BLE001
        warnings.append("phases could not be derived (%s: %s); shown as UNKNOWN"
                        % (type(e).__name__, str(e).splitlines()[0] if str(e) else ""))
        return stored, "not available"
    return np.asarray(derived, dtype=np.int8), "derived offline by g1_data.phase_label"


def open_episode(path: str) -> Episode:
    """Read one episode file. Raises InspectError on anything that is not one."""
    if not os.path.isfile(path):
        raise InspectError("no such file: %s" % path)
    try:
        with np.load(path, allow_pickle=False) as z:
            files = list(z.files)
            if "states" not in files or "actions" not in files or "meta" not in files:
                raise InspectError(
                    "%s is not an episode file: it holds %s (an episode has states, actions "
                    "and meta)" % (repo_relpath(path), ", ".join(files)))
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k].copy() for k in files if k != "meta"}
    except (zipfile.BadZipFile, ValueError, OSError) as e:
        raise InspectError("cannot read %s: %s" % (repo_relpath(path), e))
    try:
        n = REC.assert_tick_lengths(arrays, meta, where=os.path.basename(path))
    except REC.EpisodeLengthError as e:
        raise InspectError(str(e))
    width = int(arrays["states"].shape[1])
    if width not in (spec.STATE_DIM, spec.STATE_DIM - 1) \
            or int(arrays["actions"].shape[1]) != spec.ACTION_DIM:
        raise InspectError("%s has states %s and actions %s; this build reads %d-D (or an "
                           "older 47-D) states and %d-D actions"
                           % (repo_relpath(path), arrays["states"].shape, arrays["actions"].shape,
                              spec.STATE_DIM, spec.ACTION_DIM))
    warnings = []
    version = meta.get("spec_version")
    if version != spec.SPEC_VERSION:
        warnings.append("written under %s, this build is %s: the dataset code refuses this "
                        "file, and its gripper action may mean something else (docs/DATA_FLOW.md)"
                        % (version, spec.SPEC_VERSION))
    states = arrays["states"]
    if width == spec.STATE_DIM:
        locked = np.asarray(states[:, spec.BASE_LOCKED] >= 0.5)
    else:
        locked = _ticks_from_meta(meta, n)
        warnings.append("no base_locked dim: the lock shown is the FIRST lock only, from the metadata")
    aux = [Dim("aux", -1, k, u, "aux", False) for k, u in AUX_UNITS.items() if k in arrays]
    tracking = np.asarray(arrays["tracking_ok"], dtype=bool) if "tracking_ok" in arrays else None
    phases, origin = _phases_of(arrays, meta, warnings)
    return Episode(path=path, meta=meta, arrays=arrays,
                   dims=state_dims(width) + action_dims() + aux, n_ticks=n, phases=phases,
                   phase_origin=origin, locked=locked,
                   welded=np.asarray(states[:, spec.GRIP_L] >= 0.5), tracking=tracking,
                   warnings=warnings)


# ─── selecting dims and ticks ─────────────────────────────────────────────────
def select_dims(dims: Sequence[Dim], selectors: Sequence[str]) -> List[Dim]:
    """The dims named by `selectors`, in the order asked, without duplicates."""
    picked: List[Dim] = []
    for raw in selectors:
        for sel in (s.strip() for s in raw.split(",")):
            if not sel:
                continue
            m = re.fullmatch(r"([sa])(\d+)", sel)
            if sel == "all":
                hits = list(dims)
            elif sel in ("state", "action", "aux"):
                hits = [d for d in dims if d.kind == sel]
            elif m:
                kind = "state" if m.group(1) == "s" else "action"
                hits = [d for d in dims if d.kind == kind and d.index == int(m.group(2))]
            else:
                hits = [d for d in dims if d.name == sel or d.group == sel]
                hits = hits or [d for d in dims if fnmatch.fnmatchcase(d.name, sel)]
            if not hits:
                raise InspectError("no dim matches %r in this file (`dims` lists the names)" % sel)
            picked.extend(d for d in hits if d not in picked)
    return picked


def tick_range(text: Optional[str], n: int) -> range:
    """`START:END` (END excluded), `START:`, `:END` or one tick `T`, within 0..n."""
    if text is None:
        return range(n)
    m = re.fullmatch(r"(\d*)(:?)(\d*)", text.strip())
    if not m or (not m.group(2) and m.group(3)) or text.strip() in ("", ":"):
        raise InspectError("--ticks must look like 190:210, 600:, :50 or 195; got %r" % text)
    lo = int(m.group(1)) if m.group(1) else 0
    hi = (int(m.group(3)) if m.group(3) else n) if m.group(2) else lo + 1
    r = range(lo, min(hi, n))
    if len(r) == 0:
        raise InspectError("--ticks %s selects nothing: this episode has ticks 0..%d" % (text, n - 1))
    return r


def segments(mask) -> List[Tuple[int, int]]:
    """(start, end excluded) of every run of True."""
    m = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, m, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def phase_segments(labels) -> List[Tuple[int, int, str]]:
    """(start, end excluded, phase name) of every run of one label."""
    L = np.asarray(labels)
    if len(L) == 0:
        return []
    cuts = [0] + [int(i) + 1 for i in np.flatnonzero(L[1:] != L[:-1])] + [len(L)]
    return [(a, b, spec.phase_name(L[a])) for a, b in zip(cuts, cuts[1:])]


# ─── formatting ───────────────────────────────────────────────────────────────
def exact(v) -> str:
    """The shortest decimal that reads back to the same stored value: 0.1, not
    0.10000000149. No exponent, so Excel shows every value as a plain number."""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return np.format_float_positional(v, unique=True, trim="-")


def _shown(v, d: Dim, digits: int) -> str:
    if d.unit in ("0/1", "count"):
        return str(int(v))
    return "%.*f" % (digits, float(v))


def text_table(headers: Sequence[Sequence[str]], rows: Sequence[Sequence[str]]) -> List[str]:
    """Right-aligned columns under one or more header rows."""
    all_rows = list(headers) + list(rows)
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(all_rows[0]))]
    fmt = lambda r: "  ".join(c.rjust(w) for c, w in zip(r, widths)).rstrip()   # noqa: E731
    return [fmt(h) for h in headers] + ["  ".join("-" * w for w in widths)] + [fmt(r) for r in rows]


def _flag(b) -> str:
    return "x" if b else "."


def table_lines(ep: Episode, dims: Sequence[Dim], ticks: Sequence[int], digits: int = 4) -> List[str]:
    """Per-tick numbers: one tick per row, or one dim per row when `dims` is long."""
    if len(dims) <= HORIZONTAL_MAX_DIMS:
        fixed = ["tick", "t_s", "phase", "lock", "weld"]
        head = [fixed + [d.name for d in dims], [""] * len(fixed) + ["[%s]" % d.unit for d in dims]]
        rows = [[str(t), "%.2f" % (t * TICK_S), spec.phase_name(ep.phases[t]),
                 _flag(ep.locked[t]), _flag(ep.welded[t])]
                + [_shown(ep.column(d)[t], d, digits) for d in dims] for t in ticks]
        return text_table(head, rows)
    out: List[str] = []
    ticks = list(ticks)
    for k in range(0, len(ticks), VERTICAL_TICKS_PER_BLOCK):
        block = ticks[k:k + VERTICAL_TICKS_PER_BLOCK]
        left = ["kind", "idx", "name", "unit"]
        head = [left + [str(t) for t in block],
                ["", "", "phase", ""] + [spec.phase_name(ep.phases[t]) for t in block],
                ["", "", "lock", ""] + [_flag(ep.locked[t]) for t in block],
                ["", "", "weld", ""] + [_flag(ep.welded[t]) for t in block]]
        rows = [[d.kind, str(d.index) if d.index >= 0 else "", d.name, d.unit]
                + [_shown(ep.column(d)[t], d, digits) for t in block] for d in dims]
        out.extend(text_table(head, rows) + [""])
    return out[:-1]


def stats_rows(ep: Episode, dims: Sequence[Dim], ticks: Sequence[int]) -> List[dict]:
    """Per-dim summary over `ticks`, computed in float64 from the stored values."""
    idx = np.asarray(list(ticks), dtype=np.int64)
    rows = []
    for d in dims:
        x = np.asarray(ep.column(d)[idx], dtype=np.float64)
        rows.append(dict(kind=d.kind, index=d.index if d.index >= 0 else "", name=d.name,
                         unit=d.unit, masked="yes" if d.masked else "no",
                         min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                         std=float(x.std()), first=float(x[0]), last=float(x[-1]),
                         constant="yes" if x.min() == x.max() else "no"))
    return rows


def _write(path: str, write_to) -> str:
    """Create folders, and turn "the file is open in Excel" into a clear message."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    try:
        write_to(path)
    except PermissionError:
        raise InspectError("cannot write %s: is it open in Excel? Close it and run this again" % path)
    return path


def write_episode_csv(path: str, ep: Episode, dims: Sequence[Dim], ticks: Sequence[int]) -> str:
    """One row per tick, named columns. UTF-8 with BOM and exact values, for Excel."""
    def _go(p):
        cols = [ep.column(d) for d in dims]
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["tick", "time_s", "phase", "locked", "welded"] + [d.name for d in dims])
            for t in ticks:
                w.writerow([t, "%.2f" % (t * TICK_S), spec.phase_name(ep.phases[t]),
                            int(ep.locked[t]), int(ep.welded[t])] + [exact(c[t]) for c in cols])
    return _write(path, _go)


def write_stats_csv(path: str, rows: Sequence[dict]) -> str:
    def _go(p):
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return _write(path, _go)


# ─── the plot ─────────────────────────────────────────────────────────────────
def panels_of(dims: Sequence[Dim]) -> List[List[Dim]]:
    """Consecutive dims of one group and unit share a panel, up to the validated
    series count; a panel has one y-axis, so it never mixes units."""
    panels: List[List[Dim]] = []
    for d in dims:
        last = panels[-1] if panels else None
        if last and last[0].group == d.group and last[0].unit == d.unit \
                and len(last) < MAX_SERIES_PER_PANEL:
            last.append(d)
        else:
            panels.append([d])
    return panels


def _plot_title(ep: Episode) -> str:
    seed = ep.meta.get("seed")
    return "%s  (seed %s, %s, %d ticks = %.1f s)" % (os.path.basename(ep.path), seed, ep.source_label,
                                                     ep.n_ticks, ep.n_ticks * TICK_S)


def write_plot(path: str, ep: Episode, dims: Sequence[Dim], ticks: Sequence[int]) -> str:
    """Chosen dims over time. A strip names the phases, lock, weld and any tracking
    loss, and the lock and weld are shaded through every panel."""
    panels = panels_of(dims)
    if len(panels) > MAX_PANELS:
        raise InspectError("%d dims need %d panels (limit %d): choose fewer, or use `csv`"
                           % (len(dims), len(panels), MAX_PANELS))
    plt = RIO.pyplot()
    idx = np.asarray(list(ticks), dtype=np.int64)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    rows = [("phase", None), ("base lock", ep.locked), ("weld", ep.welded)]
    if ep.tracking is not None and not ep.tracking.all():
        rows.append(("tracking lost", ~ep.tracking))
    fig, axes = plt.subplots(
        len(panels) + 1, 1, sharex=True, squeeze=False,
        figsize=(11, 0.35 * len(rows) + 0.5 + 1.7 * len(panels)),
        gridspec_kw=dict(height_ratios=[0.3 * len(rows)] + [1.3] * len(panels)))
    axes = axes[:, 0]
    strip = axes[0]

    def clipped(pairs):
        return [(max(a, lo), min(b, hi)) for a, b in pairs if b > lo and a < hi]

    phases = [(max(a, lo), min(b, hi), n) for a, b, n in phase_segments(ep.phases) if b > lo and a < hi]
    top = len(rows) - 1
    for k, (a, b, name) in enumerate(phases):
        strip.broken_barh([(a, b - a)], (top - 0.42, 0.84), facecolor=PHASE_TINTS[k % 2],
                          edgecolor=RIO.SURFACE, linewidth=1.0)
        share = (b - a) / max(hi - lo, 1)
        if share >= 0.015:          # a narrow phase gets its first letters
            strip.text((a + b) / 2, top, name if share >= 0.055 else name[:3], ha="center",
                       va="center", fontsize=7.5, color=RIO.INK)
    tints = {"base lock": LOCK_TINT, "weld": WELD_TINT, "tracking lost": TRACKING_TINT}
    for r, (label, mask) in enumerate(rows):
        if mask is not None:
            spans = clipped(segments(mask))
            strip.broken_barh([(a, b - a) for a, b in spans], (top - r - 0.32, 0.64),
                              facecolor=tints[label], edgecolor=RIO.SURFACE)
    strip.set_yticks(range(len(rows)))
    strip.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.5)
    strip.set_ylim(-0.6, len(rows) - 0.4)
    strip.grid(False)
    strip.tick_params(axis="y", length=0)
    strip.spines["left"].set_visible(False)
    strip.set_title(_plot_title(ep), loc="left", fontsize=10.5)

    boundaries = [a for a, _, _ in phases[1:]]
    for ax, panel in zip(axes[1:], panels):
        for a, b in clipped(segments(ep.locked)):
            ax.axvspan(a, b, facecolor=LOCK_TINT, alpha=0.10, linewidth=0)
        for a, b in clipped(segments(ep.welded)):
            ax.axvspan(a, b, facecolor=WELD_TINT, alpha=0.14, linewidth=0)
        for x in boundaries:
            ax.axvline(x, color=RIO.INK_2, linewidth=0.6, linestyle=(0, (2, 3)))
        for k, d in enumerate(panel):
            ax.plot(idx, ep.column(d)[idx], color=RIO.SERIES[k], label=d.name, linewidth=1.8,
                    drawstyle="steps-post" if d.unit == "0/1" else "default")
        ax.set_ylabel(panel[0].unit if len(panel) > 1 else "%s\n%s" % (panel[0].name, panel[0].unit),
                      fontsize=8.5)
        if len(panel) > 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), fontsize=8.5, handlelength=1.4)
    axes[-1].set_xlabel("tick (1 tick = %.2f s)   |   phases: %s" % (TICK_S, ep.phase_origin), fontsize=9)
    axes[-1].set_xlim(lo, hi)
    fig.tight_layout()
    try:
        return _write(path, lambda p: fig.savefig(p, dpi=130))
    finally:
        plt.close(fig)


# ─── the commands ─────────────────────────────────────────────────────────────
def _warn(ep: Episode) -> None:
    for w in ep.warnings:
        print("WARNING: %s" % w)


def cmd_dims(a) -> int:
    rows = [[d.kind, str(d.index), d.name, d.unit, d.group, "yes" if d.masked else "no"]
            for d in state_dims() + action_dims()]
    print("\n".join(text_table([["kind", "idx", "name", "unit", "group", "masked"]], rows)))
    print("\nFiles may add the aux columns: %s" % ", ".join(AUX_UNITS))
    return 0


def cmd_info(a) -> int:
    ep = open_episode(a.episode)
    m = ep.meta
    _warn(ep)
    print("file         %s" % repo_relpath(ep.path))
    print("source       %s" % ep.source_label)
    print("spec         %s" % m.get("spec_version"))
    print("seed         %s" % m.get("seed"))
    print("box spawn    %s  heldout %s" % (m.get("box_spawn_xy"), m.get("heldout")))
    print("length       %d ticks = %.2f s" % (ep.n_ticks, ep.n_ticks * TICK_S))
    print("lock rule    %s" % ((m.get("lock_rule") or {}).get("name") or m.get("lock_rule_id")))
    print("end reason   %s" % m.get("end_reason", "n/a"))
    outcome = m.get("outcome")
    if outcome:
        print("outcome      %s" % json.dumps(outcome, sort_keys=True))
    print("\narrays in the file")
    print("\n".join("  %-14s %-12s %s" % (k, tuple(v.shape), v.dtype) for k, v in ep.arrays.items()))
    print("\nphases (%s)" % ep.phase_origin)
    rows = [[n, str(s), str(e - 1), str(e - s), "%.2f" % ((e - s) * TICK_S)]
            for s, e, n in phase_segments(ep.phases)]
    print("\n".join(text_table([["phase", "from", "to", "ticks", "s"]], rows)))
    for label, mask in (("base lock", ep.locked), ("weld", ep.welded)):
        print("\n%-10s %s" % (label, ", ".join("%d..%d" % (s, e - 1) for s, e in segments(mask)) or "never"))
    if ep.tracking is not None:
        lost = int((~ep.tracking).sum())
        print("\ntracking   lost on %d of %d ticks (%.1f%%)" % (lost, ep.n_ticks, 100.0 * lost / ep.n_ticks))
    return 0


def cmd_table(a) -> int:
    ep = open_episode(a.episode)
    _warn(ep)
    dims = select_dims(ep.dims, a.dims or DEFAULT_TABLE_DIMS)
    if a.ticks is None:
        ticks = range(min(DEFAULT_TABLE_TICKS, ep.n_ticks))
        print("(first %d ticks of %d; --ticks START:END for others)" % (len(ticks), ep.n_ticks))
    else:
        ticks = tick_range(a.ticks, ep.n_ticks)
    print("\n".join(table_lines(ep, dims, list(ticks)[::a.every], digits=a.digits)))
    print("\nphases %s" % ep.phase_origin)
    return 0


def cmd_stats(a) -> int:
    ep = open_episode(a.episode)
    _warn(ep)
    dims = select_dims(ep.dims, a.dims or ["all"])
    ticks = tick_range(a.ticks, ep.n_ticks)
    rows = stats_rows(ep, dims, ticks)
    print("ticks %d..%d (%d)" % (ticks[0], ticks[-1], len(ticks)))
    num = lambda v: "%.5f" % v                                          # noqa: E731
    print("\n".join(text_table(
        [["kind", "idx", "name", "unit", "masked", "min", "max", "mean", "std", "first", "last", "constant"]],
        [[r["kind"], str(r["index"]), r["name"], r["unit"], r["masked"], num(r["min"]), num(r["max"]),
          num(r["mean"]), num(r["std"]), num(r["first"]), num(r["last"]), r["constant"]] for r in rows])))
    if a.out:
        print("wrote %s" % repo_relpath(write_stats_csv(a.out, rows)))
    return 0


def cmd_csv(a) -> int:
    ep = open_episode(a.episode)
    _warn(ep)
    dims = select_dims(ep.dims, a.dims or ["all"])
    ticks = list(tick_range(a.ticks, ep.n_ticks))[::a.every]
    write_episode_csv(a.out, ep, dims, ticks)
    print("wrote %s: %d ticks, %d named columns; phases %s"
          % (repo_relpath(a.out), len(ticks), len(dims) + 5, ep.phase_origin))
    return 0


def cmd_plot(a) -> int:
    ep = open_episode(a.episode)
    _warn(ep)
    dims = select_dims(ep.dims, a.dims or DEFAULT_PLOT_DIMS)
    write_plot(a.out, ep, dims, tick_range(a.ticks, ep.n_ticks))
    print("wrote %s: %d dims; phases %s" % (repo_relpath(a.out), len(dims), ep.phase_origin))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dims", help="every state and action dim: index, name, unit, masked").set_defaults(fn=cmd_dims)

    def add(name, fn, help_, out=None):
        p = sub.add_parser(name, help=help_)
        p.add_argument("episode", help="an episode .npz")
        p.set_defaults(fn=fn)
        if name != "info":
            p.add_argument("--dims", nargs="+", default=None, help="names, globs, s28/a17, groups, "
                           "state, action, aux or all; comma- or space-separated")
            p.add_argument("--ticks", default=None, help="START:END (END excluded), START:, :END or T")
        if out == "required":
            p.add_argument("--out", required=True, help="the file to write")
        elif out == "optional":
            p.add_argument("--out", default=None, help="also write this CSV")
        return p

    add("info", cmd_info, "what the file holds: source, seed, phases, lock and weld ticks")
    t = add("table", cmd_table, "per-tick numbers for a tick range")
    t.add_argument("--every", type=int, default=1, help="show every Nth tick")
    t.add_argument("--digits", type=int, default=4)
    add("stats", cmd_stats, "min, max, mean, std, first, last per dim", out="optional")
    c = add("csv", cmd_csv, "write a named-column CSV that opens in Excel", out="required")
    c.add_argument("--every", type=int, default=1, help="write every Nth tick")
    add("plot", cmd_plot, "PNG of chosen dims over time, with lock, weld and phase shading", out="required")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "every", 1) < 1:
        print("error: --every must be >= 1", file=sys.stderr)
        return 2
    try:
        return a.fn(a)
    except InspectError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
