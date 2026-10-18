"""Relabel the gripper action of EXISTING episodes, writing converted COPIES.

    python tools/convert_grip_label.py recordings/scripted_v2 --out-dir recordings/scripted_v2_grip
    python tools/convert_grip_label.py "recordings/scripted_v2/ep_seed000*.npz" --out-dir D:/tmp/conv

Open item 29 (2026-10-18). Before spec 1.2.0 the two gripper action dims were
the weld command SAMPLED at the tick, which flips mid-tick, so they equalled the
weld bit of the same state on every tick and BC learned to copy its input. The
label is now the NEXT state's weld bit (`spec.next_weld_label`, the one rule the
recorder also applies at save). This applies it to files recorded before.

WHAT IT DOES, per file:
  * actions[:, 17:19] := next_weld_label(states[:, 28])   (same indices in 1.1.0)
  * grip_cmd           := the old actions[:, 17] (the sampled command), unless
                          the file already has one
  * meta["grip_label"] := spec.GRIP_LABEL, plus `grip_label_conversion`: the
                          source path, its sha256, this tool, commit, UTC time
Every other array and metadata field is copied unchanged.

WHAT IT DOES NOT DO:
  * overwrite: the output directory must differ from every source directory,
    and an existing output file is refused;
  * change `spec_version`. A converted 1.1.0 file is STILL 1.1.0 - it has no
    `base_locked` dim (spec 1.2.0), which cannot be recovered offline: the lock
    is an equality state, not stored per tick - so this build's loader refuses
    it like any other 1.1.0 file. It is for analysis under the old layout, and
    for proving the rule;
  * convert a file that already carries a `grip_label`;
  * copy the ledger. A converted folder is not a recording namespace.
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

#: Layouts whose weld bit is state 28 and gripper action 17/18. A version not
#: listed is refused rather than relabelled at indices it may not use.
KNOWN = {"g1-spec-1.1.0": 47, "g1-spec-1.2.0": 48}


class ConvertError(ValueError):
    pass


def _sources(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.npz")))
        else:
            out += sorted(glob.glob(p))
    if not out:
        raise ConvertError("no .npz found in %s" % (paths,))
    return out


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
    ver = meta.get("spec_version")
    if ver not in KNOWN:
        raise ConvertError("%s: spec_version %r is not a layout this tool knows %s"
                           % (name, ver, sorted(KNOWN)))
    if "grip_label" in meta:
        raise ConvertError("%s already carries grip_label %r" % (name, meta["grip_label"]))
    assert_tick_lengths(arrays, meta, where=name)
    S, A = arrays["states"], arrays["actions"]
    if S.shape[1] != KNOWN[ver] or A.shape[1] != spec.ACTION_DIM:
        raise ConvertError("%s: shapes %s / %s do not match %s"
                           % (name, S.shape, A.shape, ver))
    old = A[:, spec.GRIP_L_A].copy()
    new = spec.next_weld_label(S[:, spec.GRIP_L]).astype(A.dtype)
    A = A.copy()
    A[:, spec.GRIP_L_A] = new
    A[:, spec.GRIP_R_A] = new
    arrays["actions"] = A
    arrays.setdefault("grip_cmd", old.astype(np.float32))
    meta["grip_label"] = spec.GRIP_LABEL
    meta["grip_label_conversion"] = dict(
        source=repo_relpath(src), source_sha256=digest,
        tool="tools/convert_grip_label.py", git_commit=_git_commit(),
        utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        note="actions[:, 17:19] := state[t+1, 28]; grip_cmd = the sampled command; "
             "spec_version unchanged")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    np.savez_compressed(dst, meta=np.array(json.dumps(meta)), **arrays)
    changed = int(np.sum(old != new))
    return dict(file=name, spec_version=ver, ticks=len(S), labels_changed=changed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="episode folder(s) or .npz glob(s)")
    ap.add_argument("--out-dir", required=True, help="where the copies go (must be new)")
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
    n = 0
    for p in srcs:
        r = convert_file(p, os.path.join(out, os.path.basename(p)))
        n += 1
        print("  %s  %s  %d ticks, %d gripper label(s) changed"
              % (r["file"], r["spec_version"], r["ticks"], r["labels_changed"]))
    print("converted %d file(s) -> %s (originals untouched)" % (n, repo_relpath(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
