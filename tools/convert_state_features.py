"""Declare an observation feature set on EXISTING episodes, writing COPIES.

    python tools/convert_state_features.py data/processed/episodes_human_v2 \
        --out-dir data/processed/episodes_human_v3

"fix #1" (2026-11-20). The copies declare `meta["state_features"] =
spec.STATE_FEATURES_ROBOT_FRAME`, so every loader and normalizer reading them
builds the policy's observation as `spec.observation(state, robot_frame_v1)`:
the recorded 48-D state plus box and goal (x, y, sin bearing, cos bearing) in
the base heading frame (`spec.robot_frame_features`). The features are NOT
stored: they are a deterministic function of the recorded state and the fixed
goal, computed by the one function the loader and deploy share, so a stored
copy could only ever disagree with it.

WHAT IT DOES, per file:
  * meta["state_features"] := spec.STATE_FEATURES_ROBOT_FRAME, plus
    `state_features_conversion`: the source path, its sha256, the rule, this
    tool, commit, UTC time
  * every array and every other metadata field copied unchanged (checked)

WHAT IT DOES NOT DO:
  * overwrite: the output directory must differ from every source directory,
    and an existing output file is refused;
  * convert a file that already declares `state_features`;
  * copy anything but `ep_seed*.npz` from the top of a source folder;
  * touch `recordings/` (it only reads, and refuses to write there);
  * change `spec_version` - the recorded layout is the same.

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

TOOL = "tools/convert_state_features.py"
RECORDINGS = os.path.join(ROOT, "recordings")


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


def convert_arrays(arrays: dict, meta: dict, name: str = "",
                   features: str = spec.STATE_FEATURES_ROBOT_FRAME) -> tuple:
    """(arrays, new meta) - pure. The arrays are returned unchanged."""
    if meta.get("spec_version") != spec.SPEC_VERSION:
        raise ConvertError("%s: spec_version %r; this tool handles %s files only"
                           % (name, meta.get("spec_version"), spec.SPEC_VERSION))
    if "state_features" in meta:
        raise ConvertError("%s already declares state_features %r" % (name, meta["state_features"]))
    if features == spec.STATE_FEATURES_RAW or features not in spec.STATE_FEATURES:
        raise ConvertError("%s: convert to a non-raw feature set, got %r" % (name, features))
    assert_tick_lengths(arrays, meta, where=name)
    S = np.asarray(arrays["states"])
    if S.ndim != 2 or S.shape[1] != spec.STATE_DIM:
        raise ConvertError("%s: states %s, expected (T, %d)" % (name, S.shape, spec.STATE_DIM))
    F = spec.robot_frame_features(S.astype(np.float64))
    if not np.all(np.isfinite(F)):
        raise ConvertError("%s: non-finite robot-frame feature" % name)
    m = dict(meta)
    m["state_features"] = features
    return dict(arrays), m


def convert_file(src: str, dst: str) -> dict:
    """Write the converted copy of `src` to `dst`. Returns a short report."""
    if os.path.exists(dst):
        raise ConvertError("refusing to overwrite %s" % dst)
    if os.path.normcase(os.path.abspath(dst)).startswith(os.path.normcase(RECORDINGS) + os.sep):
        raise ConvertError("refusing to write under recordings/ (%s)" % dst)
    with open(src, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with np.load(src, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k].copy() for k in z.files if k != "meta"}
    name = os.path.basename(src)
    new, m = convert_arrays(arrays, meta, name)
    m["state_features_conversion"] = dict(
        source=repo_relpath(src), source_sha256=digest,
        rule=spec.STATE_FEATURES[m["state_features"]]["rule"],
        goal_xy=list(spec.GOAL_XY), tool=TOOL, git_commit=_git_commit(),
        utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        note="arrays copied unchanged; the features are computed from states by "
             "spec.observation at load and at deploy; spec_version unchanged")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    np.savez_compressed(dst, meta=np.array(json.dumps(m)), **new)
    with np.load(dst, allow_pickle=False) as z:                 # read back: arrays identical
        for k, v in arrays.items():
            if not np.array_equal(z[k], v) or z[k].dtype != v.dtype:
                raise ConvertError("%s: array %r changed on write" % (name, k))
    F = spec.robot_frame_features(np.asarray(arrays["states"], dtype=np.float64))
    return dict(file=name, ticks=len(F),
                box_x0=float(F[0, 0]), box_y0=float(F[0, 1]),
                goal_bearing0_deg=float(np.degrees(np.arctan2(F[0, 6], F[0, 7]))))


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
        print("  %s  %d ticks, box at tick 0: x %+.3f y %+.3f m, goal bearing %+.1f deg"
              % (r["file"], r["ticks"], r["box_x0"], r["box_y0"], r["goal_bearing0_deg"]))
    print("converted %d file(s) -> %s (sources untouched)%s"
          % (n, repo_relpath(out), "; %d not accepted, skipped" % skipped if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
