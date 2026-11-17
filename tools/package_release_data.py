"""Build the GitHub Release data zips, and the manifest the fetcher checks them against.

    python tools/package_release_data.py                 # all assets -> release_build/
    python tools/package_release_data.py --only analysis_scripted_v3 --dry-run

Data never goes into git history. It goes into zips uploaded as assets of ONE
GitHub Release (tag in RELEASE_TAG), and `tools/fetch_release_data.py` puts each
file back at the same repo-relative path it was packed from.

Every zip is DETERMINISTIC (sorted members, fixed 1980-01-01 timestamps), so
rebuilding from unchanged files gives the same SHA-256. Checkpoints (.pt, already
zip archives) are stored, everything else deflated. GitHub's limit is 2 GiB per
asset; this refuses anything over 2.0e9 bytes.

The manifest, `tools/release_manifest.json`, is committed: it records each zip's
size and SHA-256, so a fetched file that is not the packed one is refused.

WHAT GOES IN (the final checkpoint per run is the one tools/eval_scripted.py chose,
read from runs/scripted_v3/analysis_final/runs.csv, `chosen_ckpt`):
    scripted_v3_dataset      recordings/scripted_v3/ (200 episodes + ledger)
    human_demonstrations     recordings/demonstrations/ WITHOUT any keypoints/ folder, and
                             with `camera.host` (the recording laptop's hostname) removed
                             from each episode's meta - in the ZIP only (see strip_host)
    analysis_scripted_v3     eval/, eval_d19/, analysis_final/, report_extras/, summaries,
                             and each final run's metadata.json, result.json, train.log, ...
    checkpoints_scripted_v3  the chosen .pt of the 16 runs under runs/scripted_v3/
    checkpoints_memory_design the chosen .pt of the 6 runs under runs/scripted_v3_memory_design/
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "release_manifest.json")
RELEASE_REPO = "prince118-hub/g1_vision_based_teleoperation"
RELEASE_TAG = "data-v1"
MAX_ASSET_BYTES = 2_000_000_000
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
RUNS_CSV = "runs/scripted_v3/analysis_final/runs.csv"
#: Small per-run files that describe a training run (config, data provenance, result).
RUN_FILES = ("metadata.json", "result.json", "resources.json", "train.log", "metrics.jsonl")
SKIP_SUFFIXES = (".part", ".tmp", ".bak", ".pyc")


def _walk(rel_dir: str, exclude_dirs=()) -> List[str]:
    out = []
    base = os.path.join(ROOT, rel_dir)
    if not os.path.isdir(base):
        raise SystemExit("missing folder %s" % rel_dir)
    for d, dirs, files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in exclude_dirs and x != "__pycache__")
        for f in files:
            if f.endswith(SKIP_SUFFIXES):
                continue
            out.append(os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/"))
    return sorted(out)


#: Removed from released human episodes. Operator name and ZED serial stay (Charles, 2026-11-17).
STRIP_META_KEYS = (("camera", "host"),)


def _npz_bytes(arrays: Dict[str, np.ndarray]) -> bytes:
    """An .npz as np.savez_compressed writes it, but with fixed member timestamps so
    the release zip stays deterministic (savez stamps the current time)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for k, v in arrays.items():
            info = zipfile.ZipInfo(k + ".npy", FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            with z.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(v), allow_pickle=False)
    return buf.getvalue()


def strip_host(src: str) -> bytes:
    """The episode at `src` with STRIP_META_KEYS removed from its meta JSON. The file on
    disk is NOT modified. Every other array, and every other meta field, must come back
    identical from the rewritten bytes, or this raises."""
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    s = str(arrays["meta"])
    meta = json.loads(s)
    if json.dumps(meta) != s:
        raise SystemExit("%s: meta JSON does not round-trip; refusing to rewrite it" % src)
    for path in STRIP_META_KEYS:
        d = meta
        for k in path[:-1]:
            d = d.get(k, {})
        d.pop(path[-1], None)
    arrays["meta"] = np.array(json.dumps(meta))
    out = _npz_bytes(arrays)
    with np.load(io.BytesIO(out), allow_pickle=False) as z:
        if list(z.files) != list(arrays):
            raise SystemExit("%s: array order changed" % src)
        for k, v in arrays.items():
            if z[k].dtype != v.dtype or z[k].shape != v.shape or not np.array_equal(z[k], v):
                raise SystemExit("%s: array %s did not survive the rewrite" % (src, k))
        back = json.loads(str(z["meta"]))
    with np.load(src, allow_pickle=False) as z:
        orig = json.loads(str(z["meta"]))
    for path in STRIP_META_KEYS:
        d = orig
        for k in path[:-1]:
            d = d.get(k, {})
        d.pop(path[-1], None)
    if back != orig:
        raise SystemExit("%s: meta changed beyond the stripped keys" % src)
    return out


def final_runs() -> List[Dict[str, str]]:
    """(run folder, chosen checkpoint) for the 22 final runs, from the final report's runs.csv."""
    rows = []
    with open(os.path.join(ROOT, RUNS_CSV), encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            path = r["path"].replace("\\", "/")                  # runs/<root>/eval/<run>
            root = path.split("/eval/")[0]
            rows.append(dict(run_dir="%s/%s" % (root, r["run"]), ckpt=r["chosen_ckpt"],
                             model=r["model"], size=r["size"]))
    if len(rows) != 22:
        raise SystemExit("%s lists %d runs, expected the final 22" % (RUNS_CSV, len(rows)))
    return rows


def asset_files() -> Dict[str, Dict]:
    runs = final_runs()
    analysis = []
    for d in ("runs/scripted_v3/eval", "runs/scripted_v3/eval_d19",
              "runs/scripted_v3/analysis_final", "runs/scripted_v3/report_extras",
              "runs/scripted_v3_memory_design/eval"):
        analysis += _walk(d)
    for f in ("runs/scripted_v3/summary.txt", "runs/scripted_v3/summary_n40.txt",
              "runs/scripted_v3/summary_bc_b256.txt",
              "runs/scripted_v3_memory_design/summary_memory_design_n40.txt"):
        if os.path.exists(os.path.join(ROOT, f)):
            analysis.append(f)
    for r in runs:
        for f in RUN_FILES:
            p = "%s/%s" % (r["run_dir"], f)
            if os.path.exists(os.path.join(ROOT, p)):
                analysis.append(p)
            elif f == "metadata.json":
                raise SystemExit("missing %s" % p)
    ck = lambda root: ["%s/%s" % (r["run_dir"], r["ckpt"]) for r in runs
                       if r["run_dir"].startswith(root + "/")]
    return {
        "scripted_v3_dataset": dict(
            files=_walk("recordings/scripted_v3"),
            description="Scripted demonstrations (g1-spec-1.2.0, 200 episodes) -> recordings/scripted_v3/"),
        "human_demonstrations": dict(
            files=_walk("recordings/demonstrations", exclude_dirs=("keypoints",)),
            transform={".npz": strip_host},
            description="Teleoperated demonstrations (accepted, discarded/, failed/, ledger.jsonl), "
                        "WITHOUT the raw ZED keypoints/ and with camera.host removed from each "
                        "episode's meta (so the 4 file sha256 values in ledger.jsonl no longer "
                        "match) -> recordings/demonstrations/"),
        "analysis_scripted_v3": dict(
            files=sorted(set(analysis)),
            description="Exp 1/2 evaluation records, checkpoint selection, final report, and the "
                        "22 final runs' metadata -> runs/scripted_v3/, runs/scripted_v3_memory_design/"),
        "checkpoints_scripted_v3": dict(
            files=ck("runs/scripted_v3"),
            description="Selected checkpoint of each of the 16 runs under runs/scripted_v3/ "
                        "(needs analysis_scripted_v3 for each run's metadata.json)"),
        "checkpoints_memory_design": dict(
            files=ck("runs/scripted_v3_memory_design"),
            description="Selected checkpoint of each of the 6 ACT / ACT-LSTM (memory design) runs "
                        "(needs analysis_scripted_v3 for each run's metadata.json)"),
    }


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(name: str, files: List[str], out_dir: str,
          transform: Optional[Dict[str, Callable[[str], bytes]]] = None) -> Dict:
    zpath = os.path.join(out_dir, name + ".zip")
    tmp = zpath + ".part"
    raw = rewritten = 0
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for rel in files:
            src = os.path.join(ROOT, rel)
            info = zipfile.ZipInfo(rel, FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED if rel.endswith(".pt") else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            fn = (transform or {}).get(os.path.splitext(rel)[1])
            if fn is not None:
                data = fn(src)
                raw += len(data)
                rewritten += 1
                with z.open(info, "w", force_zip64=True) as fo:
                    fo.write(data)
                continue
            raw += os.path.getsize(src)
            with open(src, "rb") as fi, z.open(info, "w", force_zip64=True) as fo:
                for b in iter(lambda: fi.read(1 << 20), b""):
                    fo.write(b)
    if rewritten:
        print("  rewrote %d file(s) in the zip copy (sources untouched)" % rewritten)
    os.replace(tmp, zpath)
    size = os.path.getsize(zpath)
    if size > MAX_ASSET_BYTES:
        raise SystemExit("%s is %.2f GB, over the 2 GB asset limit - split it" % (zpath, size / 1e9))
    return dict(name=name + ".zip", size=size, sha256=_sha256(zpath), n_files=len(files),
                unpacked_bytes=raw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "release_build"))
    ap.add_argument("--only", nargs="+", default=None, help="asset names (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="list files and sizes, write nothing")
    a = ap.parse_args(argv)
    assets = asset_files()
    names = a.only or list(assets)
    unknown = set(names) - set(assets)
    if unknown:
        raise SystemExit("unknown asset(s) %s; known: %s" % (sorted(unknown), list(assets)))
    if a.dry_run:
        for n in names:
            fs = assets[n]["files"]
            raw = sum(os.path.getsize(os.path.join(ROOT, f)) for f in fs)
            print("%-28s %5d files  %9.1f MB unpacked" % (n, len(fs), raw / 1e6))
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    old = {}
    if os.path.exists(MANIFEST):
        old = {x["name"]: x for x in json.load(open(MANIFEST, encoding="utf-8"))["assets"]}
    for n in names:
        print("building %s ..." % n, flush=True)
        rec = build(n, assets[n]["files"], a.out_dir, assets[n].get("transform"))
        rec["description"] = assets[n]["description"]
        old[rec["name"]] = rec
        print("  %s  %.1f MB  sha256 %s" % (rec["name"], rec["size"] / 1e6, rec["sha256"][:16]))
    order = [n + ".zip" for n in assets]
    man = dict(repo=RELEASE_REPO, tag=RELEASE_TAG,
               note="Built by tools/package_release_data.py; fetched by tools/fetch_release_data.py.",
               assets=[old[k] for k in order if k in old])
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")
    print("manifest -> %s" % os.path.relpath(MANIFEST, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
