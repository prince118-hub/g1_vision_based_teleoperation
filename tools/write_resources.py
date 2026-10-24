"""resources.json for each run, from the training launchers' summary lines.

    python tools/write_resources.py --run-root runs/scripted_v3 [--force]

The launchers (tools/train_scripted_v3*.ps1) poll the training process every
5 s and write one line per run to <run-root>/summary*.txt:

    act_n40_seed0  status=hard_cap  steps=200000  exit=0  270.6 min  peak RAM 1.47 GB  peak GPU 2328 MiB

train_bc.py itself logs neither number, so tools/report_extras.py reads them from
a `resources.json` sidecar in the run folder. This writes that sidecar:

    peak_ram_gb   the training process's PeakWorkingSet64 (Windows), GB
    peak_gpu_gb   nvidia-smi memory.used, DEVICE-WIDE (includes the desktop and
                  any other process), MiB / 1024 - an upper bound on the run's own
    source        which summary line it came from

A run with several lines (an interrupted first attempt, a SKIPPED relaunch) takes
its LAST line whose status is "finished" or "hard_cap"; lines for other
statuses are ignored. An existing resources.json is kept unless --force.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import report_io as IO                                              # noqa: E402

LINE = re.compile(r"^\W*(?P<run>\S+)\s+status=(?P<status>\S+)\s+steps=(?P<steps>\S*)\s+"
                  r"exit=(?P<exit>\S*)\s+(?P<min>[\d.]+) min\s+peak RAM (?P<ram>[\d.]+) GB\s+"
                  r"peak GPU (?P<gpu>-?\d+) MiB")
DONE = ("finished", "hard_cap")


def parse(summary_paths):
    """{run: (fields, 'file:line')} - the last clean-finish line per run."""
    out = {}
    for p in summary_paths:
        for i, line in enumerate(open(p, encoding="utf-8-sig", errors="replace"), 1):
            m = LINE.match(line.strip())
            if m and m.group("status") in DONE:
                out[m.group("run")] = (m.groupdict(), "%s:%d" % (IO.repo_relpath(p), i))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=os.path.join(IO.ROOT, "runs", "scripted_v3"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    found = parse(sorted(glob.glob(os.path.join(a.run_root, "summary*.txt"))))
    wrote, kept, missing = [], [], []
    for d in sorted(glob.glob(os.path.join(a.run_root, "*"))):
        name = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "result.json")):
            continue
        dst = os.path.join(d, "resources.json")
        if name not in found:
            missing.append(name)
            continue
        if os.path.exists(dst) and not a.force:
            kept.append(name)
            continue
        f, src = found[name]
        gpu = int(f["gpu"])
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(dict(peak_ram_gb=float(f["ram"]),
                           peak_gpu_gb=round(gpu / 1024.0, 3) if gpu >= 0 else None,
                           peak_gpu_mib=gpu if gpu >= 0 else None,
                           wall_min=float(f["min"]), status=f["status"],
                           ram_what="training process PeakWorkingSet64, polled every 5 s",
                           gpu_what="nvidia-smi memory.used, DEVICE-WIDE (includes other "
                                    "processes), polled every 5 s - an upper bound",
                           source=src), fh, indent=1)
        wrote.append(name)
    print("wrote %d: %s" % (len(wrote), " ".join(wrote)))
    if kept:
        print("kept (exists; --force to rewrite): %s" % " ".join(kept))
    if missing:
        print("no clean-finish summary line: %s" % " ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
