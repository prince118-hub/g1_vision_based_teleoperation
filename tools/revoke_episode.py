"""Revoke an ACCEPTED episode: move it to <out-dir>/discarded/ and append a
ledger DISCARD row that says so. The seed becomes pending again.

For an acceptance that should not stand (a mis-keyed `a` on the verdict
screen). The recorder itself has no way back from an ACCEPT.

    python tools/revoke_episode.py recordings/demonstrations/ep_seed0019_20261113-131806.npz \
        --reason "operator mis-key: pressed a on a RECOMMEND: DISCARD verdict"

WHAT IT DOES, AND WHAT IT NEVER DOES
------------------------------------
- The ledger is append-only (g1_data/ledger.py). ONE row is appended; no
  existing row is edited or removed. The row is an ordinary `discard`, so every
  reader of the ledger (`EpisodeLedger.state`, `.pending`, `.summary`,
  `g1_data.dataset.coverage`) already treats the seed as re-issuable; the extra
  fields (`revokes`, `revoked_accept_t`, `sha256`) say it was a revocation.
- The episode file is MOVED, never rewritten: it is byte-identical in
  discarded/ (sha256 before the move is stored in the row and re-checked after
  it). Its own metadata therefore still says `accepted: true` - the ledger row
  is the record of the revocation. The keypoint take stays in keypoints/, where
  the recorder keeps the takes of discarded episodes too.
- It refuses unless the ledger's LAST decision for that seed is the ACCEPT of
  exactly this file, the file is in the out-dir itself, and nothing is in the
  way at the destination. A second run on the same episode is refused.
- If the ledger row cannot be written, the file is moved back.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from g1_data.ledger import EpisodeLedger                              # noqa: E402
from g1_data.paths import repo_relpath                                # noqa: E402

DISCARDED_DIR = "discarded"            # = tools/record_teleop_live.DISCARDED_DIR (tested)
LEDGER_NAME = "ledger.jsonl"
_NAME = re.compile(r"^ep_seed(\d+)_.+\.npz$")


class RevokeError(Exception):
    """The revocation was refused; nothing was changed."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _ledger_path_abs(stored: str) -> str:
    """A path as the recorder stored it (repo-relative, either separator)."""
    p = stored.replace("\\", os.sep).replace("/", os.sep)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def revoke(episode: str, reason: str, by: str = "operator") -> dict:
    """Revoke the acceptance of `episode`. Returns the appended ledger row.
    Raises RevokeError, having changed nothing, when it is not safe."""
    if not (reason or "").strip():
        raise RevokeError("a reason is required")
    src = os.path.abspath(episode)
    out_dir = os.path.dirname(src)
    m = _NAME.match(os.path.basename(src))
    if not m:
        raise RevokeError("not an episode file name: %s" % os.path.basename(src))
    seed = int(m.group(1))
    if not os.path.isfile(src):
        raise RevokeError("no such file: %s" % src)
    ledger_file = os.path.join(out_dir, LEDGER_NAME)
    if not os.path.isfile(ledger_file):
        raise RevokeError("no %s beside the episode (accepted episodes live in the "
                          "out-dir itself): %s" % (LEDGER_NAME, out_dir))
    ledger = EpisodeLedger(ledger_file)
    last = ledger.state().get(seed)
    if last is None or last["event"] != "accept":
        raise RevokeError("seed %d is not ACCEPTED in the ledger (last decision: %s)"
                          % (seed, last["event"] if last else "none"))
    if not _same(_ledger_path_abs(last.get("path", "")), src):
        raise RevokeError("the ledger's ACCEPT for seed %d is a different file: %s"
                          % (seed, last.get("path")))
    dst = os.path.join(out_dir, DISCARDED_DIR, os.path.basename(src))
    if os.path.exists(dst):
        raise RevokeError("already exists, not overwritten: %s" % dst)

    digest = _sha256(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    try:
        if _sha256(dst) != digest:
            raise RevokeError("the file changed while it was moved")
        fields = {k: last[k] for k in ("label", "keypoints", "checks", "end_reason", "ticks",
                                       "placement_error", "recommendation") if k in last}
        row = ledger.append(
            "discard", seed=seed, reason="acceptance revoked: %s" % reason.strip(),
            saved=repo_relpath(dst), revokes=last.get("path"), revoked_accept_t=last.get("t"),
            sha256=digest, operator_choice="discard", decided_by="%s (revoke_episode.py)" % by,
            **fields)
    except BaseException:
        shutil.move(dst, src)            # the ledger did not record it: undo the move
        raise
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="the accepted episode .npz, in its out-dir")
    ap.add_argument("--reason", required=True, help="why the acceptance does not stand")
    ap.add_argument("--by", default="operator", help="who revokes (default %(default)s)")
    a = ap.parse_args(argv)
    try:
        row = revoke(a.episode, a.reason, a.by)
    except RevokeError as e:
        print("REFUSED, nothing changed: %s" % e)
        return 2
    ledger = EpisodeLedger(os.path.join(os.path.dirname(os.path.abspath(a.episode)), LEDGER_NAME))
    s = ledger.summary()
    print("revoked seed %d -> %s" % (row["seed"], row["saved"]))
    print("ledger: %d lines, accepted %d, discarded %d; seed %d is pending again"
          % (s["lines"], s["accepted"], s["discarded"], row["seed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
