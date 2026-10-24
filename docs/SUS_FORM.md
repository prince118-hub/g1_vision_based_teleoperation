# System Usability Scale (SUS) — teleoperation system

**Standard 10-item SUS (Brooke, 1996), unmodified wording.** Give it to each operator **after** their
collection session(s), before any discussion of results. "The system" here means the whole teleoperation
setup the operator used: the ZED 2i body tracking, the live robot view, and the keyboard controls for
walking and grasping.

Score with `python tools/sus_score.py responses.csv` (see `docs/REPORT_EXTRAS.md`). Do not change the
item wording or order: the scoring alternates positive (odd) and negative (even) items, and the
published benchmarks (mean ≈ 68) only apply to the standard form.

---

**Operator ID:** ____________________ **Date:** ______________ **Session(s):** ____________

**Instructions.** For each statement, tick ONE box that best describes your reaction to the system
today. Answer every item. If you are unsure, tick the middle box (3). Respond quickly; your first
reaction is what is wanted.

| # | Statement | 1<br>Strongly disagree | 2 | 3 | 4 | 5<br>Strongly agree |
|---|---|:-:|:-:|:-:|:-:|:-:|
| 1 | I think that I would like to use this system frequently. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | I found the system unnecessarily complex. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | I thought the system was easy to use. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | I think that I would need the support of a technical person to be able to use this system. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | I found the various functions in this system were well integrated. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | I thought there was too much inconsistency in this system. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 7 | I would imagine that most people would learn to use this system very quickly. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | I found the system very cumbersome to use. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9 | I felt very confident using the system. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | I needed to learn a lot of things before I could get going with this system. | ☐ | ☐ | ☐ | ☐ | ☐ |

Comments (optional): ______________________________________________________________________

_____________________________________________________________________________________________

---

## For the researcher: transcription and scoring

Transcribe each form as one row of a CSV (UTF-8, header required):

```csv
operator,session,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
charles,2026-11-02,4,1,5,2,3,3,4,2,5,1
```

`session` is optional free text (a date or label). Every `q` must be an integer 1–5; a blank or out-of-range
answer is refused, not guessed — re-ask the operator, or record the centre point 3 by hand if the operator
could not answer (Brooke's instruction) and say so in the write-up.

Standard scoring, per response:

- odd items (1, 3, 5, 7, 9): contribution = answer − 1
- even items (2, 4, 6, 8, 10): contribution = 5 − answer
- SUS = 2.5 × (sum of the ten contributions), range 0–100

The example row above scores (3+4+2+3+4) + (4+3+2+3+4) = 32 → **80.0**. A SUS score is not a percentage.

Reference: J. Brooke, "SUS: a 'quick and dirty' usability scale," in *Usability Evaluation in Industry*,
Taylor & Francis, 1996, pp. 189–194.
