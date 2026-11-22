"""Independent recount of human_v2 Exp 1/2 from the raw eval JSONL (stdlib only).

Does NOT import make_human_report.py, analyze_experiments.py or any project module, so it
is an independent check of runs/human_v2/analysis_final/report.md (2026-11-22: every
number matched). Read-only.

Usage: python tools/verify_human_v2.py [runs/human_v2/eval]

Walk-in boundary: a lock on the SAME tick as the weld counts as a pickup lock (t <= weld),
the rule make_human_report.lock_flags uses. One episode depends on it (act_orig seed 1,
exp2, box seed 599: lock and weld both at tick 377); with t < weld, ACT Exp 2 walk-in is
128 not 129 and there are 75 unlocked welds, not 74.
"""
import json, sys, os
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("runs", "human_v2", "eval")
MODELS = {"bc_b256": "BC", "act_orig": "ACT", "act_lstm_full": "ACT-LSTM"}
PHASES = ["walk-in", "grasp", "lift", "transport", "place"]


def model_of(run):
    for code, name in MODELS.items():
        if run.startswith(code + "_human40v2_seed"):
            return name, int(run.rsplit("seed", 1)[1])
    raise ValueError(run)


def walkin(r, strict=True):
    """Pickup lock = a lock engaged while the box was not welded.
    strict: lock tick <= weld tick (or never welded). loose: also ticks after a release."""
    w, rel = r["weld_engage_tick"], r["weld_release_tick"]
    for t in r["lock_ticks"]:
        if w < 0 or t <= w:
            return True
        if not strict and rel >= 0 and t > rel:
            return True
    return False


def reached(r):
    out = [walkin(r), r["grasp"], r["lift"], r["transport"], r["place"]]
    cum, acc = [], True
    for x in out:
        acc = acc and bool(x)
        cum.append(acc)
    return cum


data = {}  # (model, seed, exp) -> {box_seed: record}
for run in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, run)
    if not os.path.isdir(d):
        continue
    model, seed = model_of(run)
    chosen = json.load(open(os.path.join(d, "selection.json")))["chosen"]
    for exp in ("exp1", "exp2"):
        recs = [json.loads(l) for l in open(os.path.join(d, exp + ".jsonl")) if l.strip()]
        assert len(recs) == 100, (run, exp, len(recs))
        assert all(r["policy"].endswith("/" + chosen) for r in recs), (run, exp, "not the chosen ckpt")
        by = {r["seed"]: r for r in recs}
        assert len(by) == 100
        data[(model, seed, exp)] = by

# box positions identical across runs?
for exp in ("exp1", "exp2"):
    keys = {k: set(v) for k, v in data.items() if k[2] == exp}
    s0 = next(iter(keys.values()))
    assert all(s == s0 for s in keys.values()), exp
    boxes = {k: {s: tuple(r["box_xy"]) for s, r in v.items()} for k, v in data.items() if k[2] == exp}
    b0 = next(iter(boxes.values()))
    assert all(b == b0 for b in boxes.values()), exp + " box_xy differs"

print("(1) FULL SUCCESS per run (Exp 1 | Exp 2)")
for m in MODELS.values():
    tot = {"exp1": 0, "exp2": 0}
    parts = []
    for s in range(3):
        a = sum(r["success"] for r in data[(m, s, "exp1")].values())
        b = sum(r["success"] for r in data[(m, s, "exp2")].values())
        tot["exp1"] += a; tot["exp2"] += b
        parts.append(f"s{s} {a}|{b}")
    print(f"  {m:9s} {', '.join(parts)}  pooled {tot['exp1']}/300 | {tot['exp2']}/300")
allx = sum(r["success"] for v in data.values() for r in v.values())
print(f"  all 1800 episodes: {allx}")

print("\n(2) CUMULATIVE 5-PHASE COUNTS (pooled /300) + per run; also success==place check")
for strict in (True, False):
    if not strict:
        print("  -- loose walk-in (lock after a release also counts) --")
    for m in MODELS.values():
        for exp in ("exp1", "exp2"):
            c = [0] * 5
            per = []
            for s in range(3):
                cs = [0] * 5
                for r in data[(m, s, exp)].values():
                    w = walkin(r, strict)
                    cum = [w] + reached(r)[1:]
                    acc = True
                    for i, x in enumerate(cum):
                        acc = acc and bool(x)
                        cs[i] += acc
                    if strict:
                        assert bool(r["success"]) == bool(acc), "success != place-cumulative"
                per.append(cs)
                c = [a + b for a, b in zip(c, cs)]
            print(f"  {m:9s} {exp}: " + " ".join(f"{p} {n}" for p, n in zip(PHASES, c))
                  + ("   per seed walk-in " + ",".join(str(p[0]) for p in per) if strict else ""))

# edge case: lock tick == weld tick
eq = sum(1 for v in data.values() for r in v.values() if r["weld_engage_tick"] in r["lock_ticks"])
print(f"  episodes with a lock tick equal to the weld tick: {eq}")
# raw (unchained) grasp without walk-in
g_nolock = sum(1 for v in data.values() for r in v.values() if r["grasp"] and not walkin(r))
print(f"  grasp=True with no pickup lock: {g_nolock}")


def pos_counts(m, exp, idx):
    out = defaultdict(int)
    for s in range(3):
        for bs, r in data[(m, s, exp)].items():
            if idx == "success":
                out[bs] += bool(r["success"])
            else:
                out[bs] += reached(r)[idx]
    return out


print("\n(3) ACT-LSTM vs ACT, position level (favour ACT-LSTM / favour ACT / tie)")
for label, exp, idx in [("Exp 1 full success", "exp1", "success"), ("Exp 2 walk-in", "exp2", 0),
                        ("Exp 1 walk-in", "exp1", 0), ("Exp 1 place", "exp1", 4)]:
    a, b = pos_counts("ACT-LSTM", exp, idx), pos_counts("ACT", exp, idx)
    fa = sum(a[k] > b[k] for k in a); fb = sum(a[k] < b[k] for k in a)
    both0 = sum(a[k] == b[k] == 0 for k in a)
    print(f"  {label:20s} {fa} / {fb} / {100 - fa - fb} ({both0} both 0)"
          f"   mean {sum(a.values())/100:.2f} vs {sum(b.values())/100:.2f}")

print("\n(4) WELDS WITH NO PICKUP LOCK: how many counted as grasp (raw, unchained = the old")
print("    4-phase table) and how many ended in SUCCESS (Exp 1 + Exp 2)")
tot = 0
for m in MODELS.values():
    row = []
    n_all = gr = succ = 0
    for exp in ("exp1", "exp2"):
        n = g = sc = welded = 0
        for s in range(3):
            for r in data[(m, s, exp)].values():
                if r["weld_engage_tick"] >= 0:
                    welded += 1
                    if not walkin(r):
                        n += 1; g += bool(r["grasp"]); sc += bool(r["success"])
        row.append(f"{exp}: welded {welded}, unlocked {n}, grasp {g}, success {sc}")
        n_all += n; gr += g; succ += sc
    tot += n_all
    print(f"  {m:9s} {'; '.join(row)}  => {n_all} unlocked, {gr} grasp, {succ} success")
print(f"  total unlocked welds: {tot}")
