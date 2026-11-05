"""The evaluation episode cap for a demonstration dataset: ceil(1.3 x its longest ACCEPTED demo).

    python tools/eval_cap.py recordings/demonstrations            # report
    python tools/eval_cap.py recordings/demonstrations --ticks    # just the number

WHY A RULE
----------
A closed-loop evaluation episode ends at a fall, at a held success, or at the
cap (`tools/deploy.py`). Scripted evaluation caps at 900 ticks (36 s); scripted_v3
demonstrations run 548-692 ticks (NOTES 2026-10-17), so a policy has ~30% more
time than the slowest demonstration. Human teleoperated demonstrations are
slower and their length is not known in advance (40-120 s expected), so a fixed
900 could end a policy's episode before the demonstrated task fits in it. The
rule states the same margin for any dataset:

    max_ticks = ceil(1.3 x the longest accepted demonstration, in ticks)

computed in INTEGERS, ceil(13 L / 10), so no floating-point rounding can move it
by a tick (math.ceil(1.3 * L) happens to agree for every L up to 5000, because
the double nearest 1.3 is just below it - an accident this rule does not rely on).

WHAT COUNTS AS AN ACCEPTED DEMONSTRATION
----------------------------------------
The episodes at the TOP LEVEL of the folder (`dataset.scan`, not recursive) -
where every recorder puts accepted episodes; `failed/` and `discarded/` are
subfolders and never counted. When the folder has its ledger, the two must agree
exactly (`dataset.coverage`): an ACCEPT with no file, or a file with no ACCEPT,
refuses (fail closed - "cannot tell" never passes as a cap). Lengths are each
episode's stored `n_ticks`.

SCRIPTED EVALUATION KEEPS 900
-----------------------------
`deploy.py` and `eval_scripted.py` default to `deploy.MAX_TICKS` = 900 and the
rule is applied only when asked for (`--max-ticks-from <folder>`). For a
scripted folder the report says so. (If the longest scripted_v3 episode is the
692 ticks NOTES records, the rule gives exactly 900: 13 * 692 = 8996 -> 900.)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from g1_data import dataset as DS
from g1_data import spec
from g1_data.ledger import EpisodeLedger
from g1_data.paths import repo_relpath

#: 1.3 as an exact fraction: the cap is computed in integers.
FACTOR_NUM, FACTOR_DEN = 13, 10
FACTOR = FACTOR_NUM / FACTOR_DEN
RULE = "ceil(1.3 x longest accepted demonstration, in ticks)"
#: Scripted evaluation's cap (tools/deploy.py MAX_TICKS), restated so this
#: module does not import the deploy tool; tools/test_eval_cap.py checks the
#: two agree.
SCRIPTED_MAX_TICKS = 900
TICK_S = 1.0 / spec.CONTROL_HZ                      # 0.04 s (25 Hz, D4)


class CapError(DS.DatasetError):
    """The cap cannot be stated for this folder."""


def cap_for_length(longest_ticks: int) -> int:
    """ceil(1.3 * longest_ticks), exactly (integer arithmetic)."""
    n = int(longest_ticks)
    if n < 1:
        raise CapError("a demonstration length must be >= 1 tick, got %r" % (longest_ticks,))
    return -(-FACTOR_NUM * n // FACTOR_DEN)


def _n_ticks(e: DS.Episode) -> int:
    n = e.meta.get("n_ticks")
    if n is None:
        arrays, _ = e.load()
        n = len(arrays["states"])
    return int(n)


def eval_cap(folder: str, ledger_path: Optional[str] = None) -> dict:
    """The cap and everything it was computed from, as a JSON-able dict
    (recorded verbatim in deploy headers and selection.json)."""
    from g1_data.recorder import SOURCE_SCRIPTED, ledger_for
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise CapError("no such folder %s" % folder)
    eps = DS.scan(folder, require_uniform=False)
    if not eps:
        raise CapError("%s holds no accepted episode (top-level .npz) - the cap "
                       "cannot be stated" % repo_relpath(folder))
    led_path = ledger_path or ledger_for(folder)
    ledger_checked = os.path.exists(led_path)
    if ledger_checked:
        cov = DS.coverage(EpisodeLedger(led_path), eps)
        if cov["missing_from_disk"] or cov["not_in_ledger"]:
            raise CapError("the ledger %s and the folder disagree on what was accepted "
                           "(ACCEPT with no file: seeds %s; file with no ACCEPT: seeds %s) "
                           "- refusing to state a cap" % (repo_relpath(led_path),
                                                           cov["missing_from_disk"],
                                                           cov["not_in_ledger"]))
    lengths = [(_n_ticks(e), e) for e in eps]
    longest, e_long = max(lengths, key=lambda x: (x[0], -x[1].seed))
    n = np.asarray([x[0] for x in lengths], dtype=np.int64)
    sources = sorted({str(e.meta.get("source")) for e in eps})
    cap = cap_for_length(longest)
    return dict(
        rule=RULE, factor=FACTOR, max_ticks=int(cap), max_s=round(cap * TICK_S, 2),
        longest_ticks=int(longest), longest_s=round(longest * TICK_S, 2),
        longest_seed=int(e_long.seed), longest_file=os.path.basename(e_long.path),
        n_accepted=len(eps),
        lengths_ticks=dict(min=int(n.min()), median=float(np.median(n)), max=int(n.max())),
        folder=repo_relpath(folder), ledger_checked=bool(ledger_checked),
        sources=sources, scripted=sources == [SOURCE_SCRIPTED],
        scripted_max_ticks=SCRIPTED_MAX_TICKS)


def report_lines(c: dict) -> list:
    out = ["evaluation cap for %s: %d ticks (%.2f s)" % (c["folder"], c["max_ticks"], c["max_s"]),
           "  rule      %s" % c["rule"],
           "  longest   %d ticks (%.2f s), seed %d, %s"
           % (c["longest_ticks"], c["longest_s"], c["longest_seed"], c["longest_file"]),
           "  accepted  %d episode(s), length min %d / median %.1f / max %d ticks; ledger %s"
           % (c["n_accepted"], c["lengths_ticks"]["min"], c["lengths_ticks"]["median"],
              c["lengths_ticks"]["max"], "checked" if c["ledger_checked"] else "ABSENT (not checked)"),
           "  sources   %s" % ", ".join(c["sources"])]
    if c["scripted"]:
        out.append("  NOTE: a scripted folder - scripted evaluation keeps %d ticks "
                   "(deploy.py default); the rule is applied only with --max-ticks-from"
                   % c["scripted_max_ticks"])
    return out
