"""Relabel the walking-velocity action dims of EXISTING episodes, writing smoothed COPIES.

    python tools/convert_velocity_label.py recordings/demonstrations --out-dir data/processed/episodes_human_v2

A TEST, not adopted (2026-11-19). Human teleop walk-in labels are on/off key
taps (each keypress is a 0.25 s latch pulse), and ACT trained on them collapsed
vx/vy to the constant "no key" at every checkpoint, even on its own training
demos. This writes copies whose dims 19-21 are `spec.velocity_label_smoothed`
(centred moving average over spec.VELOCITY_LABEL_WINDOW ticks, locked ticks 0),
so the label can be tested without touching a recording.

WHAT IT DOES, per file:
  * actions[:, 19:22] := spec.velocity_label_smoothed(states, actions)[:, 19:22]
  * meta["velocity_label"] := spec.VELOCITY_LABEL_SMOOTH, plus
    `velocity_label_conversion`: the source path, its sha256, the window, this
    tool, commit, UTC time
Every other array, every other action dim and every other metadata field is
copied unchanged (checked before writing).

WHAT IT DOES NOT DO:
  * overwrite: the output directory must differ from every source directory,
    and an existing output file is refused;
  * convert a file that already carries a `velocity_label`;
  * copy anything but `ep_seed*.npz` from the top of a source folder (no
    discarded/ or failed/ subfolders, no ledger: a converted folder is not a
    recording namespace);
  * change `spec_version` - the layout is the same.

`--accepted-only` (default on) skips files whose meta is not `accepted`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from g1_data import spec                                            # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402
from g1_data.recorder import _git_commit, assert_tick_lengths       # noqa: E402

TOOL = "tools/convert_velocity_label.py"


class ConvertError(ValueError):
    pass


def _sources(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "ep_seed*.npz")))
        else:
            out += sorted(glob.glob(p))
    if not out:
        raise ConvertError("no episode .npz found in %s" % (paths,))
    return out


def convert_arrays(arrays: dict, meta: dict, name: str = "") -> tuple:
    """(new arrays, new meta) - pure, so the tests can check the rule without files."""
    if meta.get("spec_version") != spec.SPEC_VERSION:
        raise ConvertError("%s: spec_version %r; this tool relabels %s files only"
                           % (name, meta.get("spec_version"), spec.SPEC_VERSION))
    if "velocity_label" in meta:
        raise ConvertError("%s already carries velocity_label %r" % (name, meta["velocity_label"]))
    assert_tick_lengths(arrays, meta, where=name)
    S, A = arrays["states"], arrays["actions"]
    A2 = spec.velocity_label_smoothed(S, A)
    keep = [i for i in range(spec.ACTION_DIM) if i not in range(19, 22)]
    if not np.array_equal(A2[:, keep], A[:, keep]) or A2.dtype != A.dtype:
        raise ConvertError("%s: a dim other than 19-21 changed - refusing" % name)
    out = dict(arrays)
    out["actions"] = A2
    m = dict(meta)
    m["velocity_label"] = spec.VELOCITY_LABEL_SMOOTH
    return out, m


def convert_file(src: str, dst: str) -> dict:
    """Write the converted copy of `src` to `dst`. Returns a short report."""
    if os.path.exists(dst):
        raise ConvertError("refusing to overwrite %s" % dst)
    with open(src, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with np.load(src, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k].copy() for k in z.files if k != "meta"}
    name = os.path.basename(src)
    new, m = convert_arrays(arrays, meta, name)
    m["velocity_label_conversion"] = dict(
        source=repo_relpath(src), source_sha256=digest, window_ticks=spec.VELOCITY_LABEL_WINDOW,
        rule=spec.VELOCITY_LABELS[spec.VELOCITY_LABEL_SMOOTH]["rule"],
        tool=TOOL, git_commit=_git_commit(),
        utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        note="actions[:, 19:22] smoothed; every other array and dim unchanged; spec_version unchanged")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    np.savez_compressed(dst, meta=np.array(json.dumps(m)), **new)
    A0, A1 = arrays["actions"][:, 19:22].astype(np.float64), new["actions"][:, 19:22].astype(np.float64)
    return dict(file=name, ticks=len(A0), mean_abs_change=float(np.abs(A1 - A0).mean()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="episode folder(s) or .npz glob(s)")
    ap.add_argument("--out-dir", required=True, help="where the copies go (must be new)")
    ap.add_argument("--include-unaccepted", action="store_true",
                    help="also convert files whose meta is not `accepted`")
    a = ap.parse_args(argv)
    srcs = _sources(a.sources)
    out = os.path.abspath(a.out_dir)
    for d in {os.path.dirname(os.path.abspath(p)) for p in srcs}:
        if os.path.normcase(d) == os.path.normcase(out):
            raise SystemExit("--out-dir is a source folder (%s); copies only" % repo_relpath(d))
    for p in srcs:
        dst = os.path.join(out, os.path.basename(p))
        if os.path.exists(dst):
            raise SystemExit("refusing to overwrite %s" % repo_relpath(dst))
    n = skipped = 0
    for p in srcs:
        with np.load(p, allow_pickle=False) as z:
            accepted = bool(json.loads(str(z["meta"])).get("accepted"))
        if not accepted and not a.include_unaccepted:
            skipped += 1
            continue
        r = convert_file(p, os.path.join(out, os.path.basename(p)))
        n += 1
        print("  %s  %d ticks, mean |change| dims 19-21 %.4f" % (r["file"], r["ticks"], r["mean_abs_change"]))
    print("converted %d file(s) -> %s (originals untouched)%s"
          % (n, repo_relpath(out), "; %d not accepted, skipped" % skipped if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
