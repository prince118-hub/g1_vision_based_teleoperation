"""Train any model of the thesis: a thin, behaviour-identical wrapper of tools/train_bc.py.

    python tools/train_policy.py train --model act_lstm_full --chunk 100 --obs-window 1 \
        --batch 8 --max-steps 200000 --window-steps 1000 --checkpoint-every 20000 \
        --budget-reason "..." --episodes recordings/scripted_v3 \
        --splits data/processed/splits_scripted_v3_n40.json \
        --norm data/processed/norm_stats_scripted_v3_n40.npz --out-dir runs/<name>
    python tools/train_policy.py --help          (train_bc.py's own help)

WHY THIS FILE EXISTS
--------------------
`tools/train_bc.py` is the SHARED trainer of every model - BC, chunked BC, ACT
(act_orig), ACT-LSTM (act_lstm_full) and the W_o = 12 ablations (act, act_lstm).
`--model` chooses the model, exactly as the original ACT repository's
`imitate_episodes.py` trains ACT or CNNMLP through `--policy_class`. The name
`train_bc.py` is historical: it was written for BC in Stage 2 and grew into the
one loop the whole ladder shares (g1_model/train.py explains why there is only
one). Launchers, NOTES.md, CLAUDE.md and other tools name it, so it is NOT
renamed. This file only lets a command read naturally.

WHAT IT DOES
------------
It imports `train_bc` and calls `train_bc.main()` in this same process with the
SAME arguments. Nothing is parsed, defaulted, added or removed here: the
subcommands, every flag and its default, the refusals, the run directory,
metadata.json, the checkpoints and the exit code are train_bc.py's own
(tools/test_train_policy.py checks two runs bit for bit). The only visible
difference is cosmetic: argparse prints `train_policy.py` as the program name in
its usage and error lines.

ONE THING IT DOES NOT INHERIT: THE ONE-HEAVY-PROCESS GUARDS
------------------------------------------------------------
tools/eval_scripted.py (`HEAVY`, `_heavy_others`) and the PowerShell launchers
(tools/train_*.ps1) detect a running training by the script name `train_bc.py`
in the process command line. A run started as `train_policy.py` is INVISIBLE to
them, so they could start a second heavy process beside it - and the 4 GB laptop
GPU does not hold two (NOTES.md 2026-10-26). Whenever a launcher or
eval_scripted.py may run at the same time, start the training as
`tools/train_bc.py` (the launchers already do, unchanged).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                     # already sys.path[0] when run as a script
    sys.path.insert(0, HERE)

import train_bc                                                      # noqa: E402


def main(argv=None):
    """Run `train_bc.main()` with `argv` (default: this process's own arguments).

    Like train_bc.main() it never returns: it ends in SystemExit with train_bc's
    exit code. `argv` is for in-process callers (tests); sys.argv is restored."""
    if argv is None:
        return train_bc.main()
    saved = sys.argv
    sys.argv = [saved[0] if saved else "train_policy.py"] + [str(a) for a in argv]
    try:
        return train_bc.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
