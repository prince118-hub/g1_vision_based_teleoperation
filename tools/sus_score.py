"""Standard System Usability Scale scoring (Brooke 1996) from a CSV of responses.

    python tools/sus_score.py responses.csv
    python tools/sus_score.py responses.csv --out runs/sus

Input: the CSV described in docs/SUS_FORM.md - header `operator,q1..q10`, an
optional `session` column, one row per completed form, each answer an integer
1-5. A blank or out-of-range answer REFUSES the file with the row number: a
guessed answer would be a fabricated response.

Scoring, per response: odd items contribute (answer - 1), even items
(5 - answer); SUS = 2.5 x the sum, 0-100. Per operator: the mean of that
operator's responses (an operator who filled the form after several sessions
has several rows). Overall: the mean over OPERATORS (each operator counts once,
so a repeat respondent does not outweigh the others), plus the mean over all
responses, both with n and the sample SD.

Writes (with --out) sus_responses.csv, sus_operators.csv and sus.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ITEMS = tuple("q%d" % i for i in range(1, 11))


class SUSError(ValueError):
    """A response that cannot be scored as given."""


def sus_score(answers: Sequence[int]) -> float:
    """One response (10 answers, item 1 first) -> SUS 0-100."""
    if len(answers) != 10:
        raise SUSError("SUS has 10 items, got %d answer(s)" % len(answers))
    total = 0
    for i, a in enumerate(answers, 1):
        if isinstance(a, bool) or int(a) != a or not 1 <= int(a) <= 5:
            raise SUSError("item %d: answer %r is not an integer 1-5" % (i, a))
        total += (int(a) - 1) if i % 2 else (5 - int(a))
    return 2.5 * total


def read_responses(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        cols = [c.strip().lower() for c in (rd.fieldnames or [])]
        missing = [c for c in ("operator",) + ITEMS if c not in cols]
        if missing:
            raise SUSError("%s: missing column(s) %s" % (path, ", ".join(missing)))
        out = []
        for n, raw in enumerate(rd, 2):                         # row 1 is the header
            r = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            if not any(r.values()):
                continue
            ans = []
            for q in ITEMS:
                try:
                    ans.append(int(r[q]))
                except ValueError:
                    raise SUSError("%s row %d (%s): %s = %r is not an integer 1-5"
                                   % (path, n, r.get("operator"), q, r[q])) from None
            try:
                score = sus_score(ans)
            except SUSError as e:
                raise SUSError("%s row %d (%s): %s" % (path, n, r.get("operator"), e)) from None
            if not r["operator"]:
                raise SUSError("%s row %d: no operator id" % (path, n))
            out.append(dict(row=n, operator=r["operator"], session=r.get("session", ""),
                            **dict(zip(ITEMS, ans)), sus=score))
    if not out:
        raise SUSError("%s holds no responses" % path)
    return out


def _sd(xs: List[float]):
    return statistics.stdev(xs) if len(xs) > 1 else None


def summarize(responses: List[dict]) -> dict:
    by: Dict[str, List[float]] = {}
    for r in responses:
        by.setdefault(r["operator"], []).append(r["sus"])
    ops = [dict(operator=o, n_responses=len(v), sus_mean=statistics.mean(v), sus_sd=_sd(v),
                sus_min=min(v), sus_max=max(v)) for o, v in by.items()]
    means = [o["sus_mean"] for o in ops]
    allv = [r["sus"] for r in responses]
    return dict(operators=ops,
                mean_over_operators=statistics.mean(means), sd_over_operators=_sd(means),
                n_operators=len(ops),
                mean_over_responses=statistics.mean(allv), sd_over_responses=_sd(allv),
                n_responses=len(allv))


def write(out: str, responses: List[dict], s: dict) -> None:
    import report_io as IO
    os.makedirs(out, exist_ok=True)
    rc = ["row", "operator", "session"] + list(ITEMS) + ["sus"]
    oc = ["operator", "n_responses", "sus_mean", "sus_sd", "sus_min", "sus_max"]
    IO.write_csv(os.path.join(out, "sus_responses.csv"), responses, rc)
    IO.write_csv(os.path.join(out, "sus_operators.csv"), s["operators"], oc)
    md = ["# SUS scores", "",
          "Standard SUS (Brooke 1996), 0-100. Overall: **%.1f** (mean over %d operator(s), SD %s); "
          "%.1f over %d response(s)." % (s["mean_over_operators"], s["n_operators"],
                                        IO.fmt(s["sd_over_operators"], 1),
                                        s["mean_over_responses"], s["n_responses"]), "",
          IO.md_table(s["operators"], oc, nd=1), "", "## Responses", "",
          IO.md_table(responses, rc, nd=1), ""]
    with open(os.path.join(out, "sus.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default=None, help="write sus_*.csv and sus.md here")
    a = ap.parse_args(argv)
    try:
        resp = read_responses(a.csv)
    except SUSError as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2
    s = summarize(resp)
    for o in s["operators"]:
        print("%-16s n=%d  SUS %.1f%s" % (o["operator"], o["n_responses"], o["sus_mean"],
                                         "" if o["sus_sd"] is None else "  (SD %.1f)" % o["sus_sd"]))
    print("%-16s n=%d  SUS %.1f  (mean over operators; %.1f over %d responses)"
          % ("MEAN", s["n_operators"], s["mean_over_operators"], s["mean_over_responses"],
             s["n_responses"]))
    if a.out:
        write(a.out, resp, s)
        print("-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
