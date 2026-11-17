"""Download the thesis data from the GitHub Release and unpack it into the repo.

    python tools/fetch_release_data.py                       # everything (~2.4 GB)
    python tools/fetch_release_data.py --only scripted_v3_dataset analysis_scripted_v3
    python tools/fetch_release_data.py --list                # what is available
    python tools/fetch_release_data.py --from-dir D:/release_build   # zips already on disk

Each asset's size and SHA-256 are checked against `tools/release_manifest.json`
(committed; written by tools/package_release_data.py) before anything is
unpacked. Every file lands at the repo-relative path it was packed from, e.g.
`recordings/scripted_v3/ep_seed0000.npz` or `runs/scripted_v3/act_n40_seed0/best.pt`.
An existing file with the same content is left alone; an existing file with
DIFFERENT content is refused unless --force.

Which assets you need:
    replay a scripted demo            scripted_v3_dataset
    train a small run                 scripted_v3_dataset
    evaluate a final checkpoint       scripted_v3_dataset + analysis_scripted_v3 + a checkpoints_* asset
    regenerate the final report       analysis_scripted_v3

Downloads go to .release_cache/ (gitignored) and are kept, so a rerun does not
download again; delete that folder to reclaim the space.
Uses only the Python standard library. If the direct download fails (e.g. a
private repository), it falls back to the GitHub CLI: `gh release download`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "release_manifest.json")
CACHE = os.path.join(ROOT, ".release_cache")
#: Members may only unpack under these top-level folders.
ALLOWED_TOP = ("recordings", "runs")
NEEDS = {"checkpoints_scripted_v3": "analysis_scripted_v3",
         "checkpoints_memory_design": "analysis_scripted_v3"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _verify(path, rec):
    if not os.path.exists(path):
        return "missing"
    if os.path.getsize(path) != rec["size"]:
        return "size %d != %d" % (os.path.getsize(path), rec["size"])
    if _sha256(path) != rec["sha256"]:
        return "sha256 mismatch"
    return None


def _download(url, dest):
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_release_data"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
            done += len(b)
            pct = int(100 * done / total) if total else -1
            if pct != last and (pct % 5 == 0):
                print("\r  %s  %5.0f / %.0f MB" % (os.path.basename(dest), done / 1e6, total / 1e6),
                      end="", flush=True)
                last = pct
    print()
    os.replace(tmp, dest)


def _gh_download(repo, tag, name, dest_dir):
    if not shutil.which("gh"):
        return False
    r = subprocess.run(["gh", "release", "download", tag, "--repo", repo, "--pattern", name,
                        "--dir", dest_dir, "--clobber"])
    return r.returncode == 0


def obtain(man, rec, from_dir):
    """Path to a verified local copy of the asset zip."""
    name = rec["name"]
    if from_dir:
        p = os.path.join(from_dir, name)
        bad = _verify(p, rec)
        if bad:
            raise SystemExit("%s: %s (from %s)" % (name, bad, from_dir))
        return p
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name)
    if _verify(p, rec) is None:
        print("  %s already downloaded and verified" % name)
        return p
    url = "https://github.com/%s/releases/download/%s/%s" % (man["repo"], man["tag"], name)
    print("  downloading %s" % url)
    try:
        _download(url, p)
    except Exception as e:                                   # noqa: BLE001 - report and fall back
        print("  direct download failed (%s); trying the GitHub CLI" % e)
        if not _gh_download(man["repo"], man["tag"], name, CACHE):
            raise SystemExit("could not download %s. Download it by hand from "
                             "https://github.com/%s/releases/tag/%s into %s and rerun."
                             % (name, man["repo"], man["tag"], CACHE))
    bad = _verify(p, rec)
    if bad:
        raise SystemExit("%s: downloaded file fails verification (%s)" % (name, bad))
    return p


def _target(member):
    parts = member.split("/")
    if (member.startswith("/") or "\\" in member or ":" in parts[0] or ".." in parts
            or parts[0] not in ALLOWED_TOP):
        raise SystemExit("refusing unsafe or unexpected path in archive: %r" % member)
    return os.path.join(ROOT, *parts)


def unpack(zpath, force):
    new = same = 0
    with zipfile.ZipFile(zpath) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        conflicts = []
        for i in infos:
            t = _target(i.filename)
            if os.path.exists(t) and not _same(t, i):
                conflicts.append(os.path.relpath(t, ROOT))
        if conflicts and not force:
            raise SystemExit("%d file(s) already exist with DIFFERENT content, e.g. %s\n"
                             "Nothing was unpacked from %s. Move them aside, or rerun with --force "
                             "to overwrite." % (len(conflicts), conflicts[:3], os.path.basename(zpath)))
        for i in infos:
            t = _target(i.filename)
            if os.path.exists(t) and _same(t, i):
                same += 1
                continue
            os.makedirs(os.path.dirname(t), exist_ok=True)
            with z.open(i) as src, open(t + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(t + ".part", t)
            new += 1
    return new, same


def _same(path, info):
    if os.path.getsize(path) != info.file_size:
        return False
    crc = 0
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            crc = zlib.crc32(b, crc)
    return crc == info.CRC


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None,
                    help="asset names without .zip (default: all); see --list")
    ap.add_argument("--list", action="store_true", help="list the assets and exit")
    ap.add_argument("--from-dir", default=None, help="use zips already in this folder, no download")
    ap.add_argument("--force", action="store_true", help="overwrite files whose content differs")
    a = ap.parse_args(argv)
    man = json.load(open(MANIFEST, encoding="utf-8"))
    recs = {x["name"][:-4]: x for x in man["assets"]}
    if a.list:
        print("Release %s of %s:" % (man["tag"], man["repo"]))
        for n, r in recs.items():
            print("  %-27s %8.1f MB  %s" % (n, r["size"] / 1e6, r["description"]))
        return 0
    names = a.only or list(recs)
    unknown = [n for n in names if n not in recs]
    if unknown:
        raise SystemExit("unknown asset(s) %s; known: %s" % (unknown, list(recs)))
    for n in list(names):
        dep = NEEDS.get(n)
        if dep and dep not in names:
            print("note: %s needs %s (each run's metadata.json); adding it" % (n, dep))
            names.insert(0, dep)
    for n in names:
        print("%s:" % n)
        zp = obtain(man, recs[n], a.from_dir)
        new, same = unpack(zp, a.force)
        print("  unpacked %d file(s), %d already present" % (new, same))
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
