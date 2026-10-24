"""Readiness check before collection / training: the data contract and a leakage audit.

    python tools/readiness_check.py --teleop recordings/teleop_fake/<session> \
        [--reference recordings/scripted_v3] [--folders recordings/teleop_fake ...]

PART 1 - DATA CONTRACT. Every episode .npz in --teleop (its failed/ subfolder
included: a failed episode's arrays still have to obey the contract) is checked,
and compared with --reference (the scripted_v3 training data):

  masked      action dims 6, 13, 14, 15, 16 exactly 0.0; dim 18 == dim 17
              (what spec.CONSTANT_ACTION_DIMS and deploy's post-processing assume)
  vel_locked  action dims 19-21 exactly 0 on every tick whose state has
              base_locked (dim 47) = 1 (D17)
  vel_clip    every velocity within spec.VELOCITY_CLIP (spec.assert_velocity_within_clip)
  grip_label  meta grip_label == next_weld_v1 and action dim 17 == the NEXT
              state's weld bit (spec.next_weld_label; D22)
  weld_bits   state dims 28 == 29 (both the weld bit, D14), values in {0, 1}
  base_locked state dim 47 in {0, 1}
  finite      no NaN / inf in states or actions; state 48-D, action 22-D
  spec        spec_version == spec.SPEC_VERSION
  norm_fit    dataset.fit_norm_stats(..., allow_synthetic=<not real>) succeeds -
              it runs assert_masked_actions_constant - and labels the source
              from the registry
  ranges      per-dim min/max against the reference's, in the reference's own
              normalized units (INFORMATION: a fake take is a replayed synthetic
              motion; the list says which dims leave the scripted envelope)

PART 2 - LEAKAGE AUDIT:
  * data/eval train / select / exp1 / exp2 seed streams pairwise disjoint;
    every exp2 spawn inside the held-out patch, no train/select/exp1 spawn in it
  * every splits file in data/processed: train and val disjoint, both inside the
    train stream (so outside select/exp1/exp2 and the patch); a nested split's
    train inside its parent's train with the same val
  * every norm stats file in data/processed: fitted on split "train" and on
    EXACTLY the train seeds of its splits file (dataset.assert_norm_stats_match)
  * every run under --run-root: metadata data.seeds == its splits' train, norm
    stats file = the one fitted on them; every eval exp1/exp2/select JSONL holds
    exactly its stream's seeds
  * every recording folder in --folders (+ --reference, --teleop): episode seeds
    inside the train stream, none in the patch

Prints OK / PROBLEM per check; exit code 1 if any PROBLEM.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from g1_data import dataset as DS                                   # noqa: E402
from g1_data import recorder as REC                                 # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402

RESULTS = []


def report(part, name, ok, detail=""):
    RESULTS.append((part, name, bool(ok), detail))
    print("  %-8s %-34s %s" % ("OK" if ok else "PROBLEM", name, detail))


def episode_files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.npz")) +
                  glob.glob(os.path.join(folder, "failed", "*.npz")))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return ({k: z[k] for k in ("states", "actions") if k in z.files},
                json.loads(str(z["meta"])))


# ─── part 1 ──────────────────────────────────────────────────────────────────
def contract(teleop, reference):
    print("\nPART 1 - DATA CONTRACT: %s vs %s" % (repo_relpath(teleop), repo_relpath(reference)))
    files = episode_files(teleop)
    if not files:
        report("contract", "episodes found", False, "no .npz in %s" % teleop)
        return
    eps, bad = [], {k: [] for k in ("masked", "vel_locked", "vel_clip", "grip_label", "weld_bits",
                                    "base_locked", "finite", "spec")}
    zero_dims = [d for d in spec.CONSTANT_ACTION_DIMS if d != spec.GRIP_R_A]
    for p in files:
        a, m = load(p)
        S, A = np.asarray(a["states"], np.float64), np.asarray(a["actions"], np.float64)
        tag = "%s(seed %s)" % (os.path.basename(p), m.get("seed"))
        eps.append((p, S, A, m))
        if (S.ndim != 2 or S.shape[1] != spec.STATE_DIM or A.shape != (len(S), spec.ACTION_DIM)
                or not np.isfinite(S).all() or not np.isfinite(A).all()):
            bad["finite"].append(tag)
            continue
        if np.any(A[:, zero_dims] != 0.0) or np.any(A[:, spec.GRIP_R_A] != A[:, spec.GRIP_L_A]):
            bad["masked"].append(tag)
        locked = S[:, spec.BASE_LOCKED] == 1.0
        if np.any(A[locked][:, spec.VEL_A] != 0.0):
            bad["vel_locked"].append(tag)
        try:
            spec.assert_velocity_within_clip(A[:, spec.VEL_A], where=tag)
        except Exception:                                   # noqa: BLE001
            bad["vel_clip"].append(tag)
        if (m.get("grip_label") != "next_weld_v1"
                or not np.array_equal(A[:, spec.GRIP_L_A], spec.next_weld_label(S[:, spec.GRIP_L]))):
            bad["grip_label"].append(tag)
        w = S[:, [spec.GRIP_L, spec.GRIP_R]]
        if not np.array_equal(w[:, 0], w[:, 1]) or not np.isin(w, (0.0, 1.0)).all():
            bad["weld_bits"].append(tag)
        if not np.isin(S[:, spec.BASE_LOCKED], (0.0, 1.0)).all():
            bad["base_locked"].append(tag)
        if m.get("spec_version") != spec.SPEC_VERSION:
            bad["spec"].append(tag)
    n_lock = sum(int((S[:, spec.BASE_LOCKED] == 1).sum()) for _, S, _, _ in eps)
    n_weld = sum(int((S[:, spec.GRIP_L] == 1).sum()) for _, S, _, _ in eps)
    srcs = sorted({str(m.get("source")) for *_, m in eps})
    print("  %d episode(s) (%s), %d ticks, %d locked ticks, %d welded ticks, lock rule(s) %s"
          % (len(eps), ", ".join(srcs), sum(len(S) for _, S, _, _ in eps), n_lock, n_weld,
             sorted({str((m.get("lock_rule") or {}).get("version", m.get("lock_rule"))) for *_, m in eps})))
    for k, v in bad.items():
        report("contract", k, not v, ("%d bad: %s" % (len(v), v[:3])) if v else "all %d" % len(eps))
    if n_lock == 0 or n_weld == 0:
        report("contract", "lock and weld exercised", False,
               "locked ticks %d, welded ticks %d - the rules above were not tested" % (n_lock, n_weld))
    # norm fit on the teleop episodes (the masked-dims assertion lives there)
    E = [DS.Episode(p, m) for p, _, _, m in eps]
    real = all(REC.source_info(m.get("source"))["real"] for *_, m in eps)
    try:
        _, _, rep = DS.fit_norm_stats(E, "train", allow_synthetic=not real)
        report("contract", "norm_fit", True, "fitted on %d episodes, source label %r"
               % (rep["n_episodes"], rep["source"]))
    except Exception as e:                                  # noqa: BLE001
        report("contract", "norm_fit", False, "%s: %s" % (type(e).__name__, e))
    # ranges vs the reference, in the reference's normalized units
    ref_files = episode_files(reference)
    if ref_files:
        RS = np.concatenate([load(p)[0]["states"] for p in ref_files]).astype(np.float64)
        RA = np.concatenate([load(p)[0]["actions"] for p in ref_files]).astype(np.float64)
        TS = np.concatenate([S for _, S, _, _ in eps])
        TA = np.concatenate([A for _, _, A, _ in eps])
        out = []
        for kind, R, T, names, mask in (("state", RS, TS, spec.STATE_NAMES, spec.STATE_MASK),
                                        ("action", RA, TA, spec.ACTION_NAMES, spec.ACTION_MASK)):
            mu, sd = R.mean(0), np.maximum(R.std(0), 1e-2)
            lo, hi = R.min(0), R.max(0)
            for d in range(R.shape[1]):
                if not mask[d]:
                    continue
                below, above = (lo[d] - T[:, d].min()) / sd[d], (T[:, d].max() - hi[d]) / sd[d]
                if max(below, above) > 1.0:
                    out.append("%s %d %s: %.2f sd %s the reference range [%.3f, %.3f] (teleop %.3f..%.3f)"
                               % (kind, d, names[d], max(below, above),
                                  "below" if below > above else "above", lo[d], hi[d],
                                  T[:, d].min(), T[:, d].max()))
        report("contract", "ranges (information)", True,
               "%d dim(s) leave the reference envelope by > 1 sd" % len(out))
        for line in out:
            print("             - " + line)


# ─── part 2 ──────────────────────────────────────────────────────────────────
def leakage(run_root, folders):
    from g1_teleop.box_reset import in_heldout, sample_box_pose
    from g1_teleop.config import TeleopConfig
    cfg = TeleopConfig().box
    print("\nPART 2 - LEAKAGE AUDIT")
    streams = {k: DS.load_seed_file(os.path.join(DS.EVAL, "%s_seeds.json" % k))
               for k in ("train", "select", "exp1", "exp2")}
    for a, b in itertools.combinations(streams, 2):
        inter = sorted(set(streams[a]) & set(streams[b]))
        report("leakage", "streams %s/%s disjoint" % (a, b), not inter, str(inter[:5]) if inter else "")
    held = {s: bool(in_heldout(sample_box_pose(cfg, s)[0][:2], cfg))
            for s in set().union(*streams.values())}
    for k, v in streams.items():
        n_in = sum(held[s] for s in v)
        want = len(v) if k == "exp2" else 0
        report("leakage", "stream %s held-out spawns" % k, n_in == want,
               "%d/%d in the patch (want %d)" % (n_in, len(v), want))
    train_stream = set(streams["train"])
    # The splits / norm files IN USE: every one a run under run_root names, plus
    # the scripted_v3 ones. Older files (v1/v2, other spec versions) are listed,
    # not audited - no current run reads them.
    used = set()
    for md in glob.glob(os.path.join(run_root, "*", "metadata.json")):
        dp = json.load(open(md, encoding="utf-8"))["config"]["notes"]["data_paths"]
        used |= {os.path.basename(dp["splits"]), os.path.basename(dp["norm"])}
    used |= {os.path.basename(p) for p in glob.glob(os.path.join(DS.PROCESSED, "*scripted_v3*"))}
    legacy = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DS.PROCESSED, "*"))
                    if os.path.basename(p) not in used)
    splits = {}
    for name in sorted(n for n in used if n.startswith("splits")):
        sp = json.load(open(os.path.join(DS.PROCESSED, name), encoding="utf-8"))
        tr, va = set(sp.get("train", [])), set(sp.get("val", []))
        splits[name] = (tr, va)
        problems = []
        if tr & va:
            problems.append("train/val share %s" % sorted(tr & va)[:5])
        outside = sorted((tr | va) - train_stream)
        if outside:
            problems.append("%d seed(s) outside the train stream: %s" % (len(outside), outside[:5]))
        if any(bool(in_heldout(sample_box_pose(cfg, s)[0][:2], cfg)) for s in tr | va):
            problems.append("held-out spawn in the split")
        report("leakage", "split %s" % name, not problems,
               "; ".join(problems) or "%d train / %d val, inside the train stream" % (len(tr), len(va)))
    if "splits_scripted_v3.json" in splits and "splits_scripted_v3_n40.json" in splits:
        (t160, v160), (t40, v40) = splits["splits_scripted_v3.json"], splits["splits_scripted_v3_n40.json"]
        report("leakage", "n40 nested in 160 (same val)", t40 <= t160 and v40 == v160,
               "n40 train subset %s, val equal %s" % (t40 <= t160, v40 == v160))
    for name in sorted(n for n in used if n.startswith("norm_stats")):
        sp_name = name.replace("norm_stats", "splits").replace(".npz", ".json")
        if sp_name not in splits:
            report("leakage", "norm %s" % name, False, "no matching splits file %s" % sp_name)
            continue
        try:
            _, _, meta = DS.load_norm_stats(os.path.join(DS.PROCESSED, name))
            DS.assert_norm_stats_match(meta, sorted(splits[sp_name][0]))
            ok = meta.get("split") == "train" and not (set(meta["seeds"]) & splits[sp_name][1])
            why = "split %r, fitted on exactly the %d train seeds of %s, no val seed" % (
                meta.get("split"), len(meta["seeds"]), sp_name)
        except Exception as e:                              # noqa: BLE001
            ok, why = False, "%s: %s" % (type(e).__name__, str(e)[:160])
        report("leakage", "norm %s" % name, ok, why)
    if legacy:
        print("  (not audited - no current run reads them: %s)" % ", ".join(legacy))
    for md in sorted(glob.glob(os.path.join(run_root, "*", "metadata.json"))):
        run = os.path.dirname(md)
        m = json.load(open(md, encoding="utf-8"))
        dp = m["config"]["notes"]["data_paths"]
        tr = splits.get(os.path.basename(dp["splits"]), (set(), set()))[0]
        seeds = set(int(s) for s in (m.get("data") or {}).get("seeds", []))
        _, _, nm = DS.load_norm_stats(os.path.join(ROOT, dp["norm"]))
        ok = seeds == tr and set(nm["seeds"]) == tr
        report("leakage", "run %s" % os.path.basename(run), ok,
               "trained on its split's %d train seeds, norm fitted on them" % len(tr) if ok else
               "data.seeds %d vs split train %d vs norm %d" % (len(seeds), len(tr), len(nm["seeds"])))
    ev = os.path.join(run_root, "eval")
    n_files = mism = 0
    for run in sorted(os.listdir(ev)) if os.path.isdir(ev) else []:
        for tag, stream in (("exp1", "exp1"), ("exp2", "exp2")):
            p = os.path.join(ev, run, "%s.jsonl" % tag)
            if os.path.exists(p):
                n_files += 1
                got = [json.loads(l)["seed"] for l in open(p, encoding="utf-8") if l.strip()]
                mism += sorted(got) != sorted(streams[stream])
        for p in glob.glob(os.path.join(ev, run, "select", "*.jsonl")):
            n_files += 1
            got = [json.loads(l)["seed"] for l in open(p, encoding="utf-8") if l.strip()]
            mism += sorted(got) != sorted(streams["select"])
    report("leakage", "eval files hold their stream", mism == 0,
           "%d file(s) checked, %d mismatched" % (n_files, mism))
    for f in folders:
        files = episode_files(f)
        if not files:
            report("leakage", "folder %s" % repo_relpath(f), True, "(no episodes)")
            continue
        seeds = [int(DS.read_meta(p)["seed"]) for p in files]
        out = sorted(set(seeds) - train_stream)
        inp = sorted(s for s in set(seeds) if bool(in_heldout(sample_box_pose(cfg, s)[0][:2], cfg)))
        report("leakage", "folder %s" % repo_relpath(f), not out and not inp,
               "%d episode(s); outside train stream %s; in patch %s" % (len(files), out[:5], inp[:5]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teleop", required=True, help="teleop episode folder (fake or real)")
    ap.add_argument("--reference", default=os.path.join(ROOT, "recordings", "scripted_v3"))
    ap.add_argument("--run-root", default=os.path.join(ROOT, "runs", "scripted_v3"))
    ap.add_argument("--folders", nargs="*", default=[], help="more recording folders to audit")
    a = ap.parse_args(argv)
    contract(a.teleop, a.reference)
    leakage(a.run_root, [a.reference, a.teleop] + list(a.folders))
    bad = [r for r in RESULTS if not r[2]]
    print("\n%d check(s), %d PROBLEM(S)" % (len(RESULTS), len(bad)))
    for part, name, _, d in bad:
        print("  PROBLEM %s / %s: %s" % (part, name, d))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
