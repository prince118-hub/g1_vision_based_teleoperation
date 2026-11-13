"""tools/revoke_episode.py: an acceptance is revoked by ONE appended ledger row
and a byte-identical move; everything unsafe is refused with nothing changed."""
import hashlib
import os
import shutil
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import revoke_episode as RV                                           # noqa: E402
from g1_data.ledger import EpisodeLedger                              # noqa: E402
from g1_data.paths import repo_relpath                                # noqa: E402


@pytest.fixture
def out():
    d = tempfile.mkdtemp(prefix="revoke_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _episode(out, seed, stamp="20261113-131806", payload=b"episode-bytes"):
    path = os.path.join(out, "ep_seed%04d_%s.npz" % (seed, stamp))
    with open(path, "wb") as fh:
        fh.write(payload + bytes([seed]))
    return path


def _accept(ledger, path, seed):
    ledger.append("issue", seed=seed, label="t")
    # the recorder stores repo-relative paths with the platform separator
    ledger.append("accept", seed=seed, path=repo_relpath(path), heldout=False, label="t",
                  ticks=633, end_reason="success",
                  recommendation=dict(recommend="discard", reasons=["14 tick overruns"]))


def _snapshot(out):
    files = {}
    for base, _, names in os.walk(out):
        for n in names:
            p = os.path.join(base, n)
            files[os.path.relpath(p, out)] = open(p, "rb").read()
    return files


def test_revoke_moves_the_file_appends_one_row_and_the_seed_is_pending_again(out):
    ledger = EpisodeLedger(os.path.join(out, "ledger.jsonl"))
    p18, p19 = _episode(out, 18), _episode(out, 19)
    _accept(ledger, p18, 18)
    _accept(ledger, p19, 19)
    before = open(ledger.path, "rb").read()
    sha = hashlib.sha256(open(p19, "rb").read()).hexdigest()
    assert ledger.pending(range(17, 21)) == [17, 20]

    assert RV.main([p19, "--reason", "operator mis-key"]) == 0

    dst = os.path.join(out, "discarded", os.path.basename(p19))
    assert not os.path.exists(p19) and os.path.isfile(dst)
    assert hashlib.sha256(open(dst, "rb").read()).hexdigest() == sha
    after = open(ledger.path, "rb").read()
    assert after.startswith(before), "existing ledger rows must be untouched"
    assert after[len(before):].count(b"\n") == 1, "exactly one row appended"
    row = ledger.rows()[-1]
    assert row["event"] == "discard" and row["seed"] == 19
    assert row["reason"] == "acceptance revoked: operator mis-key"
    assert row["revokes"] == repo_relpath(p19) and row["saved"] == repo_relpath(dst)
    assert row["sha256"] == sha and row["decided_by"] == "operator (revoke_episode.py)"
    assert row["recommendation"]["recommend"] == "discard" and row["ticks"] == 633
    assert ledger.pending(range(17, 21)) == [17, 19, 20]
    s = ledger.summary()
    assert (s["accepted"], s["discarded"]) == (1, 1)
    assert os.path.isfile(p18), "other episodes are not touched"


def test_refusals_change_nothing(out):
    ledger = EpisodeLedger(os.path.join(out, "ledger.jsonl"))
    p19 = _episode(out, 19)
    _accept(ledger, p19, 19)
    other = _episode(out, 19, stamp="20261113-140000")        # on disk, never accepted
    p7 = _episode(out, 7)
    ledger.append("discard", seed=7, reason="failed", saved=None)
    os.makedirs(os.path.join(out, "discarded"))
    snap = _snapshot(out)

    with pytest.raises(RV.RevokeError, match="reason"):
        RV.revoke(p19, "  ")
    with pytest.raises(RV.RevokeError, match="different file"):
        RV.revoke(other, "x")
    with pytest.raises(RV.RevokeError, match="not ACCEPTED"):
        RV.revoke(p7, "x")
    with pytest.raises(RV.RevokeError, match="no such file"):
        RV.revoke(os.path.join(out, "ep_seed0003_x.npz"), "x")
    with pytest.raises(RV.RevokeError, match="not an episode"):
        RV.revoke(ledger.path, "x")
    assert RV.main([other, "--reason", "x"]) == 2
    assert _snapshot(out) == snap

    # something already at the destination: refused, not overwritten
    blocker = os.path.join(out, "discarded", os.path.basename(p19))
    open(blocker, "wb").write(b"old")
    snap = _snapshot(out)
    with pytest.raises(RV.RevokeError, match="already exists"):
        RV.revoke(p19, "x")
    assert _snapshot(out) == snap
    os.remove(blocker)

    # a second revocation of the same episode is refused
    RV.revoke(p19, "mis-key")
    snap = _snapshot(out)
    with pytest.raises(RV.RevokeError):
        RV.revoke(p19, "again")
    with pytest.raises(RV.RevokeError):
        RV.revoke(blocker, "again")                            # the moved file: no ledger beside it
    assert _snapshot(out) == snap


def test_a_failed_ledger_write_puts_the_file_back(out, monkeypatch):
    ledger = EpisodeLedger(os.path.join(out, "ledger.jsonl"))
    p19 = _episode(out, 19)
    _accept(ledger, p19, 19)
    snap = _snapshot(out)

    def boom(self, event, **fields):
        raise OSError("disk full")
    monkeypatch.setattr(EpisodeLedger, "append", boom)
    with pytest.raises(OSError):
        RV.revoke(p19, "mis-key")
    assert _snapshot(out) == snap and os.path.isfile(p19)


def test_a_re_recorded_seed_is_accepted_again_after_a_revocation(out):
    """The next session's accept of the same seed is the last decision."""
    ledger = EpisodeLedger(os.path.join(out, "ledger.jsonl"))
    p19 = _episode(out, 19)
    _accept(ledger, p19, 19)
    RV.revoke(p19, "mis-key")
    new = _episode(out, 19, stamp="20261114-090000")
    _accept(ledger, new, 19)
    assert ledger.pending([19]) == [] and ledger.accepted()[19]["path"] == repo_relpath(new)


def test_folder_name_is_the_recorders():
    src = open(os.path.join(ROOT, "tools", "record_teleop_live.py"), encoding="utf-8").read()
    assert 'DISCARDED_DIR = "keypoints", "failed", "discarded"' in src
    assert RV.DISCARDED_DIR == "discarded"
