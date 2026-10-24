"""Back up a recordings folder to one or more destinations, verified by sha256.

    # after every recording session (Windows laptop):
    python tools/backup_recordings.py recordings/demonstrations --dest D:\\thesis_backup\\demonstrations --dest E:\\thesis_backup\\demonstrations

    # first backup to a destination that does not exist yet:
    python tools/backup_recordings.py recordings/demonstrations --dest E:\\thesis_backup\\demonstrations --create

    # what is NOT yet backed up (hashes both sides, copies nothing):
    python tools/backup_recordings.py recordings/demonstrations --dest ... --status

WHAT IS COPIED
--------------
Every regular file under the source folder, at the same relative path: the
accepted episodes (top level), `failed/`, `discarded/`, `keypoints/` and the
ledger. Each destination is a MIRROR of the source folder - point it at a folder
of its own (`...\\demonstrations`), not at the drive root. The recordings are
the irreplaceable human-generated data (`.gitignore`, PLAN.md Phase 6); nothing
here ever writes into the source.

THE RULES
---------
* Every file is VERIFIED, every run: a file already at the destination is
  hashed and compared, never trusted because its name exists.
* A copy goes to `<name>.bkpart`, is fsynced, hashed FROM DISK and compared with
  the bytes read from the source; the source is hashed again and must not have
  changed (a session still writing it); only then is it renamed into place.
* A DIFFERENT file at the destination is NEVER overwritten - it is reported as a
  CONFLICT and left as it is. The one exception is append-only growth: a
  destination file whose bytes are an exact PREFIX of the source's (the ledger
  after another session; `ledger.jsonl` is append-only by construction,
  g1_data/ledger.py) is replaced, because the new file contains every byte of
  the old one. Nothing is lost, and the manifest says `grown`.
* Files that exist only at the destination (an episode moved from the top level
  to `discarded/` after its backup) are listed, never deleted.
* A destination that does not exist is UNAVAILABLE (a USB stick not plugged in,
  a cloud folder not synced) and is not created unless `--create` is given: on
  Linux a missing mount point would otherwise be silently created on the
  laptop's own disk.

Each destination gets `_backup_manifests/backup_<stamp>.json`: date, source,
host, counts by kind (episodes / failed / discarded / keypoints / ledger /
other) and every file's path, size, sha256 and status. The run ends with what is
NOT yet backed up - per file, which destinations lack a verified copy - and
exits 1 if anything is.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_data.paths import repo_relpath                              # noqa: E402

PART = ".bkpart"
MANIFEST_DIR = "_backup_manifests"
CHUNK = 1 << 20

#: status of one file at one destination
COPIED, VERIFIED, GROWN = "copied", "verified", "grown"
CONFLICT, MISSING, ERROR, UNAVAILABLE = "CONFLICT", "missing", "ERROR", "unavailable"
OK_STATUSES = (COPIED, VERIFIED, GROWN)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def is_prefix(short: str, long: str) -> bool:
    """True if the bytes of `short` are exactly the first bytes of `long`."""
    if os.path.getsize(short) > os.path.getsize(long):
        return False
    with open(short, "rb") as a, open(long, "rb") as b:
        for chunk in iter(lambda: a.read(CHUNK), b""):
            if b.read(len(chunk)) != chunk:
                return False
    return True


def kind_of(rel: str) -> str:
    """The manifest's category for a source-relative path (forward slashes)."""
    parts = rel.split("/")
    if len(parts) == 1:
        if rel.endswith(".npz"):
            return "episodes"
        if rel.endswith(".jsonl"):
            return "ledger"
        return "other"
    if parts[0] in ("failed", "discarded", "keypoints"):
        return parts[0]
    return "other"


def list_source(src: str) -> List[str]:
    """Source-relative paths (forward slashes) of every regular file, sorted.
    Our own temp files and any manifest folder are skipped."""
    out = []
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != MANIFEST_DIR)
        for f in sorted(files):
            if f.endswith(PART):
                continue
            p = os.path.join(root, f)
            if os.path.isfile(p) and not os.path.islink(p):
                out.append(repo_relpath(p, start=src).replace(os.sep, "/"))
    return out


def _within(a: str, b: str) -> bool:
    """True if path a is b or lies inside b."""
    a, b = os.path.normcase(os.path.realpath(a)), os.path.normcase(os.path.realpath(b))
    return a == b or a.startswith(b.rstrip(os.sep) + os.sep)


def check_paths(src: str, dests: List[str]) -> None:
    if not os.path.isdir(src):
        raise SystemExit("source %s is not a folder" % src)
    for d in dests:
        if _within(d, src) or _within(src, d):
            raise SystemExit("refusing: destination %s and source %s contain one another "
                             "- a backup must live outside the folder it copies" % (d, src))
    real = [os.path.normcase(os.path.realpath(d)) for d in dests]
    if len(set(real)) != len(real):
        raise SystemExit("the same destination is given twice: %s" % dests)


def copy_verified(src_file: str, dst_file: str, expect_sha: str) -> str:
    """Copy through a temp file; return the sha256 of what was installed.
    Raises IOError if the written bytes or the source changed underneath."""
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    tmp = dst_file + PART
    h = hashlib.sha256()
    try:
        with open(src_file, "rb") as fi, open(tmp, "wb") as fo:
            for chunk in iter(lambda: fi.read(CHUNK), b""):
                h.update(chunk)
                fo.write(chunk)
            fo.flush()
            os.fsync(fo.fileno())
        read_sha = h.hexdigest()
        if read_sha != expect_sha:
            raise IOError("source changed while it was being backed up (is a session "
                          "still writing it?) - re-run after the session")
        written = sha256_of(tmp)
        if written != read_sha:
            raise IOError("the copy on disk does not match the bytes written "
                          "(sha256 %s vs %s) - the destination is unreliable"
                          % (written[:12], read_sha[:12]))
        if sha256_of(src_file) != read_sha:
            raise IOError("source changed while it was being backed up - re-run")
        st = os.stat(src_file)
        os.utime(tmp, (st.st_atime, st.st_mtime))
        os.replace(tmp, dst_file)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    final = sha256_of(dst_file)
    if final != expect_sha:
        raise IOError("installed copy does not verify (sha256 %s vs %s)"
                      % (final[:12], expect_sha[:12]))
    return final


def backup_one(src: str, dest: str, files: List[str], hashes: Dict[str, str],
               status_only: bool) -> Dict[str, dict]:
    """Per file: {status, detail}. Never deletes, never overwrites a different file."""
    out = {}
    for rel in files:
        s = os.path.join(src, *rel.split("/"))
        d = os.path.join(dest, *rel.split("/"))
        try:
            if os.path.isfile(d):
                have = sha256_of(d)
                if have == hashes[rel]:
                    out[rel] = dict(status=VERIFIED)
                elif is_prefix(d, s):
                    if status_only:
                        out[rel] = dict(status=MISSING, detail="destination holds an older "
                                        "(prefix) copy: %d of %d bytes"
                                        % (os.path.getsize(d), os.path.getsize(s)))
                    else:
                        copy_verified(s, d, hashes[rel])
                        out[rel] = dict(status=GROWN, detail="append-only growth: the "
                                        "old copy was an exact prefix of the source")
                else:
                    out[rel] = dict(status=CONFLICT, detail="a DIFFERENT file is at the "
                                    "destination (sha256 %s, source %s); left untouched"
                                    % (have[:12], hashes[rel][:12]))
            elif os.path.exists(d):
                out[rel] = dict(status=CONFLICT, detail="destination path exists and is "
                                "not a file")
            elif status_only:
                out[rel] = dict(status=MISSING)
            else:
                copy_verified(s, d, hashes[rel])
                out[rel] = dict(status=COPIED)
        except (IOError, OSError) as e:
            out[rel] = dict(status=ERROR, detail="%s: %s" % (type(e).__name__, e))
    return out


def dest_only(dest: str, files: List[str]) -> List[str]:
    """Files at the destination with no counterpart in the source (never deleted)."""
    if not os.path.isdir(dest):
        return []
    have = set(list_source(dest))
    return sorted(have - set(files))


def write_manifest(src: str, dest: str, files: List[str], hashes: Dict[str, str],
                   result: Dict[str, dict], extra: List[str], stamp: str) -> str:
    counts: Dict[str, dict] = {}
    for rel in files:
        c = counts.setdefault(kind_of(rel), dict(files=0, bytes=0))
        c["files"] += 1
        c["bytes"] += os.path.getsize(os.path.join(src, *rel.split("/")))
    by_status: Dict[str, int] = {}
    for r in result.values():
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    now = _dt.datetime.now()
    doc = dict(
        tool="tools/backup_recordings.py", format=1,
        date_local=now.isoformat(timespec="seconds"),
        date_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        host=platform.node(), source=os.path.abspath(src), destination=os.path.abspath(dest),
        n_files=len(files), total_bytes=sum(c["bytes"] for c in counts.values()),
        counts_by_kind=counts, counts_by_status=by_status,
        complete=all(r["status"] in OK_STATUSES for r in result.values()),
        destination_only=extra,
        files=[dict(path=rel, kind=kind_of(rel),
                    size=os.path.getsize(os.path.join(src, *rel.split("/"))),
                    sha256=hashes[rel], **result[rel]) for rel in files])
    mdir = os.path.join(dest, MANIFEST_DIR)
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, "backup_%s.json" % stamp)
    k = 1
    while os.path.exists(path):              # two runs in the same second
        path = os.path.join(mdir, "backup_%s_%d.json" % (stamp, k))
        k += 1
    tmp = path + PART
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def run(src: str, dests: List[str], create: bool = False, status_only: bool = False,
        quiet: bool = False) -> dict:
    """The whole backup. Returns {files, hashes, per_dest, not_backed_up, manifests}."""
    say = (lambda *a: None) if quiet else print
    check_paths(src, dests)
    files = list_source(src)
    hashes = {rel: sha256_of(os.path.join(src, *rel.split("/"))) for rel in files}
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    per_dest, manifests, extras = {}, {}, {}
    say("source %s: %d file(s), %.1f MB"
        % (src, len(files), sum(os.path.getsize(os.path.join(src, *r.split("/")))
                                for r in files) / 1e6))
    for dest in dests:
        if not os.path.isdir(dest):
            if create and not status_only:
                os.makedirs(dest)
                say("created %s" % dest)
            else:
                per_dest[dest] = {rel: dict(status=UNAVAILABLE) for rel in files}
                say("\n%s: UNAVAILABLE - the folder does not exist (drive not plugged in, "
                    "cloud folder not synced?)%s"
                    % (dest, "" if status_only else "; pass --create if this is its first backup"))
                continue
        res = backup_one(src, dest, files, hashes, status_only)
        per_dest[dest] = res
        extras[dest] = dest_only(dest, files)
        tally = {}
        for r in res.values():
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        say("\n%s: %s" % (dest, ", ".join("%s %d" % kv for kv in sorted(tally.items()))
                          or "nothing to do"))
        for rel, r in res.items():
            if r["status"] in (CONFLICT, ERROR, GROWN):
                say("  %-8s %s%s" % (r["status"], rel, "  - " + r["detail"]
                                     if r.get("detail") else ""))
        if extras[dest]:
            say("  %d file(s) only at the destination (kept, never deleted): %s%s"
                % (len(extras[dest]), extras[dest][:5], " ..." if len(extras[dest]) > 5 else ""))
        if not status_only:
            manifests[dest] = write_manifest(src, dest, files, hashes, res, extras[dest], stamp)
            say("  manifest %s" % manifests[dest])
    not_backed = {}
    for rel in files:
        lacking = [d for d in dests if per_dest[d][rel]["status"] not in OK_STATUSES]
        if lacking:
            not_backed[rel] = lacking
    return dict(files=files, hashes=hashes, per_dest=per_dest, not_backed_up=not_backed,
                destination_only=extras, manifests=manifests)


def report_not_backed_up(res: dict, dests: List[str]) -> None:
    nb = res["not_backed_up"]
    n = len(res["files"])
    print("\n================ NOT YET BACKED UP ================")
    if not nb:
        print("nothing: all %d file(s) verified (sha256) on all %d destination(s)"
              % (n, len(dests)))
        return
    print("%d of %d file(s) lack a verified copy on at least one destination:" % (len(nb), n))
    for rel, lacking in sorted(nb.items()):
        why = sorted({res["per_dest"][d][rel]["status"] for d in lacking})
        print("  %-60s %s  (%s)" % (rel, ", ".join(why), "; ".join(lacking)))
    covered = {d: sum(1 for rel in res["files"]
                      if res["per_dest"][d][rel]["status"] in OK_STATUSES) for d in dests}
    print("verified per destination: %s"
          % ", ".join("%s %d/%d" % (d, c, n) for d, c in covered.items()))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the recordings folder, e.g. recordings/demonstrations")
    ap.add_argument("--dest", action="append", required=True,
                    help="a destination folder (the source's mirror); repeat for several")
    ap.add_argument("--create", action="store_true",
                    help="create a destination folder that does not exist (first backup)")
    ap.add_argument("--status", action="store_true",
                    help="copy nothing: hash both sides and report what is not backed up")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    res = run(a.source, a.dest, create=a.create, status_only=a.status)
    report_not_backed_up(res, a.dest)
    ok = not res["not_backed_up"]
    print("\nBACKUP %s" % ("COMPLETE" if ok else "INCOMPLETE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
