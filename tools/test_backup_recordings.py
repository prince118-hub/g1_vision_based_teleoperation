"""Tests for tools/backup_recordings.py on temporary fake recordings folders.

    python tools/test_backup_recordings.py

No MuJoCo, no torch: the folder holds files laid out the way
tools/record_teleop_live.py writes them (episodes at the top level, failed/,
discarded/, keypoints/, ledger.jsonl), with random bytes standing in for the
.npz contents - the tool copies and hashes bytes and never parses them.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import backup_recordings as BR                                      # noqa: E402

LAYOUT = {
    "ep_seed0000_20261101-100000.npz": 3000,
    "ep_seed0001_20261101-100500.npz": 2500,
    "failed/ep_seed0002_20261101-101000.npz": 1800,
    "discarded/ep_seed0003_20261101-101500.npz": 1200,
    "keypoints/ep_seed0000_20261101-100000_kp.npz": 900,
    "keypoints/ep_seed0002_20261101-101000_kp.npz": 700,
}
LEDGER = "ledger.jsonl"


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _fake_recordings(root: str) -> str:
    src = os.path.join(root, "demonstrations")
    for rel, n in LAYOUT.items():
        p = os.path.join(src, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(os.urandom(n))
    _append_ledger(src, dict(event="session", operator="tester"))
    _append_ledger(src, dict(event="accept", seed=0))
    return src


def _append_ledger(src: str, row: dict) -> None:
    with open(os.path.join(src, LEDGER), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _mk():
    root = tempfile.mkdtemp(prefix="backup_test_")
    src = _fake_recordings(root)
    d1, d2 = os.path.join(root, "D_backup", "demos"), os.path.join(root, "usb", "demos")
    return root, src, d1, d2


def _manifests(dest):
    return sorted(glob.glob(os.path.join(dest, BR.MANIFEST_DIR, "*.json")))


def test_first_backup_copies_everything_to_every_destination_and_verifies():
    root, src, d1, d2 = _mk()
    try:
        assert BR.main([src, "--dest", d1, "--dest", d2, "--create"]) == 0
        files = BR.list_source(src)
        assert len(files) == len(LAYOUT) + 1
        for d in (d1, d2):
            for rel in files:
                a, b = os.path.join(src, rel), os.path.join(d, rel)
                assert BR.sha256_of(a) == BR.sha256_of(b), rel
            assert not glob.glob(os.path.join(d, "**", "*" + BR.PART), recursive=True)
            (m,) = _manifests(d)
            doc = json.load(open(m, encoding="utf-8"))
            assert doc["complete"] is True and doc["n_files"] == len(files)
            assert doc["counts_by_status"] == {"copied": len(files)}
            k = doc["counts_by_kind"]
            assert (k["episodes"]["files"], k["failed"]["files"], k["discarded"]["files"],
                    k["keypoints"]["files"], k["ledger"]["files"]) == (2, 1, 1, 2, 1)
            assert doc["date_local"] and doc["date_utc"] and doc["source"] == os.path.abspath(src)
            by = {f["path"]: f for f in doc["files"]}
            assert by["failed/ep_seed0002_20261101-101000.npz"]["sha256"] == BR.sha256_of(
                os.path.join(src, "failed", "ep_seed0002_20261101-101000.npz"))
        # nothing is ever written into the source
        assert sorted(BR.list_source(src)) == sorted(files)
        assert not os.path.exists(os.path.join(src, BR.MANIFEST_DIR))
    finally:
        shutil.rmtree(root)


def test_second_run_verifies_by_hash_and_copies_only_the_new_session():
    root, src, d1, d2 = _mk()
    try:
        assert BR.main([src, "--dest", d1, "--dest", d2, "--create"]) == 0
        # a second session: one more episode, its keypoints, more ledger rows
        new = "ep_seed0004_20261102-090000.npz"
        with open(os.path.join(src, new), "wb") as fh:
            fh.write(os.urandom(1500))
        _append_ledger(src, dict(event="accept", seed=4))
        r = BR.run(src, [d1, d2], quiet=True)
        assert not r["not_backed_up"]
        for d in (d1, d2):
            st = {rel: v["status"] for rel, v in r["per_dest"][d].items()}
            assert st[new] == BR.COPIED
            assert st[LEDGER] == BR.GROWN, st[LEDGER]       # append-only growth
            assert all(v == BR.VERIFIED for k, v in st.items() if k not in (new, LEDGER))
            assert BR.sha256_of(os.path.join(d, LEDGER)) == BR.sha256_of(os.path.join(src, LEDGER))
            assert len(_manifests(d)) == 2
    finally:
        shutil.rmtree(root)


def test_a_different_file_is_never_overwritten():
    root, src, d1, _ = _mk()
    try:
        assert BR.main([src, "--dest", d1, "--create"]) == 0
        rel = "ep_seed0000_20261101-100000.npz"
        theirs = os.path.join(d1, rel)
        with open(theirs, "wb") as fh:                   # same name, other bytes
            fh.write(b"not the same episode" * 10)
        before = BR.sha256_of(theirs)
        # a ledger at the destination that is NOT a prefix (rewritten history)
        with open(os.path.join(d1, LEDGER), "w", encoding="utf-8") as fh:
            fh.write('{"event": "accept", "seed": 99}\n')
        r = BR.run(src, [d1], quiet=True)
        st = r["per_dest"][d1]
        assert st[rel]["status"] == BR.CONFLICT and st[LEDGER]["status"] == BR.CONFLICT
        assert BR.sha256_of(theirs) == before, "a conflicting file was overwritten"
        assert set(r["not_backed_up"]) == {rel, LEDGER}
        assert BR.main([src, "--dest", d1]) == 1
        doc = json.load(open(_manifests(d1)[-1], encoding="utf-8"))
        assert doc["complete"] is False and doc["counts_by_status"]["CONFLICT"] == 2
    finally:
        shutil.rmtree(root)


def test_a_corrupted_backup_copy_is_detected_and_reported():
    """A bit flip at the destination (same size) is caught by the hash, not the
    name or size, and reported as not backed up - and is NOT silently repaired."""
    root, src, d1, _ = _mk()
    try:
        assert BR.main([src, "--dest", d1, "--create"]) == 0
        rel = "keypoints/ep_seed0002_20261101-101000_kp.npz"
        p = os.path.join(d1, *rel.split("/"))
        b = bytearray(open(p, "rb").read())
        b[100] ^= 0x01
        open(p, "wb").write(bytes(b))
        r = BR.run(src, [d1], status_only=True, quiet=True)
        assert r["per_dest"][d1][rel]["status"] == BR.CONFLICT
        assert list(r["not_backed_up"]) == [rel]
    finally:
        shutil.rmtree(root)


def test_missing_destination_is_unavailable_not_created_and_status_reports_it():
    root, src, d1, d2 = _mk()
    try:
        assert BR.main([src, "--dest", d1, "--create"]) == 0
        # d2 (the USB stick) is not there: nothing is created, everything is listed
        r = BR.run(src, [d1, d2], quiet=True)
        assert not os.path.exists(d2)
        assert set(r["not_backed_up"]) == set(BR.list_source(src))
        assert all(v == [d2] for v in r["not_backed_up"].values())
        assert BR.main([src, "--dest", d1, "--dest", d2]) == 1
        # --status copies nothing and writes no manifest
        n_before = len(_manifests(d1))
        new = os.path.join(src, "ep_seed0005_20261103-090000.npz")
        open(new, "wb").write(os.urandom(64))
        r = BR.run(src, [d1], status_only=True, quiet=True)
        assert r["per_dest"][d1]["ep_seed0005_20261103-090000.npz"]["status"] == BR.MISSING
        assert not os.path.exists(os.path.join(d1, "ep_seed0005_20261103-090000.npz"))
        assert len(_manifests(d1)) == n_before
    finally:
        shutil.rmtree(root)


def test_destination_only_files_are_kept_and_listed():
    """An episode moved from the top level to discarded/ after its backup: the
    old copy stays at the destination and is listed, the new path is copied."""
    root, src, d1, _ = _mk()
    try:
        assert BR.main([src, "--dest", d1, "--create"]) == 0
        old = "ep_seed0001_20261101-100500.npz"
        shutil.move(os.path.join(src, old), os.path.join(src, "discarded", old))
        r = BR.run(src, [d1], quiet=True)
        assert r["destination_only"][d1] == [old]
        assert os.path.exists(os.path.join(d1, old))
        assert r["per_dest"][d1]["discarded/" + old]["status"] == BR.COPIED
        assert not r["not_backed_up"]
    finally:
        shutil.rmtree(root)


def test_source_changing_during_the_copy_is_an_error_and_installs_nothing():
    root, src, d1, _ = _mk()
    try:
        os.makedirs(d1)
        rel = "ep_seed0000_20261101-100000.npz"
        s, d = os.path.join(src, rel), os.path.join(d1, rel)
        stale = BR.sha256_of(s)
        open(s, "ab").write(b"session still writing")         # changes after hashing
        _raises(IOError, BR.copy_verified, s, d, stale)
        assert not os.path.exists(d) and not os.path.exists(d + BR.PART)
    finally:
        shutil.rmtree(root)


def test_refusals():
    root, src, d1, _ = _mk()
    try:
        _raises(SystemExit, BR.main, [src, "--dest", os.path.join(src, "inside")])
        _raises(SystemExit, BR.main, [src, "--dest", root])           # contains the source
        _raises(SystemExit, BR.main, [src, "--dest", d1, "--dest", d1, "--create"])
        _raises(SystemExit, BR.main, [os.path.join(root, "nope"), "--dest", d1])
    finally:
        shutil.rmtree(root)


def test_kinds():
    assert BR.kind_of("ep_seed0000_x.npz") == "episodes"
    assert BR.kind_of("ledger.jsonl") == "ledger"
    assert BR.kind_of("failed/a.npz") == "failed"
    assert BR.kind_of("discarded/a.npz") == "discarded"
    assert BR.kind_of("keypoints/a_kp.npz") == "keypoints"
    assert BR.kind_of("notes.txt") == "other"


def _tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _tests():
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(_tests()) - failed}/{len(_tests())} passed")
    raise SystemExit(1 if failed else 0)
