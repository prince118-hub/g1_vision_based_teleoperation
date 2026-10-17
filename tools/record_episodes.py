"""Record demonstrations from the SCRIPTED demonstrator, with operator control.

    python tools/record_episodes.py --count 5 --seeds data/eval/train_seeds.json             # interactive
    python tools/record_episodes.py --count 5 --seeds data/eval/train_seeds.json --auto pass # accept if all checks pass
    python tools/record_episodes.py --invariance 0                # recording changes nothing?
    python tools/record_episodes.py --ledger-summary

    # watch and film (tools/episode_view.py; observation only):
    python tools/record_episodes.py --count 2 --seeds data/eval/train_seeds.json --auto pass \
        --viewer --realtime --video-dir D:/thesis_videos/test --out-dir recordings/scripted_test

--out-dir is the episode folder AND its ledger (default recordings/episodes/).
Writing into recordings/episodes/ requires --seeds: the contiguous fallback
stream contains held-out-patch seeds (g1_data/dataset.py:183-186).
--video-dir writes ep_seedNNNN.mp4 per accepted episode and
ep_seedNNNN_FAIL.mp4 for a discarded, retried or errored one.

Phase 3, first chunk. The recorder is `g1_data/recorder.py` and is OBSERVATION
ONLY: it wraps `mj_step` and reads the demonstrator's frame. `scripted_demo.py`
is not modified, which is why the gates can still be byte-identical.

WHY THE VERDICT IS PRINTED, NOT LEFT TO THE EYE
-----------------------------------------------
Proposal 3.3.8 requires per-episode visual inspection and re-recording of
failures. An operator watching 150 episodes cannot see a 0.8 rad wrist wedge, a
2 mm pre-grasp box nudge, or a 12 mm hand-slab penetration, and those are
exactly the failures that survive into a dataset looking like successes. So each
check is printed with its NUMBER and a pass/fail, then a recommendation. The
operator still decides: ACCEPT / DISCARD / RETRY.

Thresholds are from measured populations, not taste - each one carries its
provenance below. Two of them (clearance, pre-grasp disturbance) are deliberately
set at the edge of the MEASURED demonstrator envelope rather than at zero,
because O26 is mitigated but not closed: the hand does enter the slab in every
scripted episode, and a check that fails 40/40 healthy episodes tells the
operator nothing.

RETRY, on the scripted demonstrator, reruns the same seed and gets the same
episode - it is deterministic. It is here because it is what teleop will need,
and because the ledger has to record the attempt either way.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from contextlib import nullcontext

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g1_data import spec
from g1_data.ledger import EpisodeLedger, seed_stream
from g1_data.paths import repo_relpath
from g1_data.recorder import (NS_COLLECTION, NS_SCRIPTED, SOURCE_SCRIPTED,
                             ScriptedRecorder, assert_namespace, label_of,
                             ledger_for)
from g1_teleop.config import TeleopConfig

# A1: the destination is a named constant in the recorder registry, not a
# string typed at the command line. NS_COLLECTION is the real collection's
# namespace; this tool drives the scripted demonstrator and may not write
# there (A3), which `assert_namespace` enforces below.
OUT_DIR = NS_SCRIPTED

# ---- rejection checks. (label, provenance, predicate, formatter) ------------
# Measured envelopes over the 40-seed predicate gate (docs/measurements/
# bprime_gates_adopted_pred_40.json) unless stated otherwise.
PLACE_LIMIT = 0.10          # m, Q4 placement tolerance
WEDGE_RAD = 0.8             # TR23: pitch pair, post-REACH
DISTURB_MM = 10.0           # measured demonstrator range 0.1-1.9 mm; teleop 65-197 mm
PENETRATION_MM = -35.0      # measured demonstrator range 0.0 to -28.3 mm (O26 open)


def verdict(meta: dict) -> tuple:
    """[(name, ok, text)], recommend_accept."""
    o = meta["outcome"]
    rows = [
        ("weld fired", bool(o["engaged"]),
         "engaged=%s" % o["engaged"]),
        ("box not dropped", bool(o["resting"]) and not bool(o["fell"]),
         "resting=%s tilt=%.1f deg" % (o["resting"], o["tilt_deg"])),
        ("placement <= %.2f m" % PLACE_LIMIT, o["placement_error"] <= PLACE_LIMIT,
         "%.4f m" % o["placement_error"]),
        ("no fall", not bool(o["fell"]),
         "max pitch %.1f deg" % o["max_pitch_deg"]),
        ("wrist dev <= %.1f rad" % WEDGE_RAD, meta["wrist_dev_rad"] <= WEDGE_RAD,
         "%.3f rad (%s)" % (meta["wrist_dev_rad"], meta["wrist_dev_joint"] or "-")),
        ("box undisturbed pre-grasp", meta["box_pre_grasp_disturb_mm"] <= DISTURB_MM,
         "%.1f mm" % meta["box_pre_grasp_disturb_mm"]),
        ("hand/platform penetration",
         meta["min_hand_platform_clearance_mm"] >= PENETRATION_MM,
         "%.1f mm (0.0 = never touched, TR19)"
         % meta["min_hand_platform_clearance_mm"]),
        # The demonstrator's OWN verdict. Added 2026-10-17: without it a
        # transport scrape (box < DemoConfig.min_clearance over a platform,
        # scripted_demo.py:1251) passed every row above, so --auto pass
        # accepted an episode the demonstrator had scored as failed.
        ("demonstrator ok", bool(o["ok"]),
         "fail_phase=%s, box clearance %.1f mm"
         % (o["fail_phase"], 1000.0 * meta["min_box_platform_clearance_m"])),
    ]
    return rows, all(ok for _, ok, _ in rows)


def print_verdict(meta: dict, rows, recommend: bool) -> None:
    print("  %-28s %-6s %s" % ("check", "result", "measurement"))
    for name, ok, text in rows:
        print("  %-28s %-6s %s" % (name, "PASS" if ok else "FAIL", text))
    print("  ticks %d | sim %.1f s | sim-to-wall %.2fx | demonstrator ok=%s"
          % (meta["n_ticks"], meta["sim_seconds"], meta["sim_to_wall_ratio"],
             meta["outcome"]["ok"]))
    print("  RECOMMEND: %s" % ("ACCEPT" if recommend else "DISCARD"))


def run_one(seed: int, cfg, demo, book, policy, record: bool = True, view=None):
    """(result, recorder or None). The demonstrator call is untouched either way.

    `view` is an optional `episode_view.EpisodeView`. It is entered BEFORE the
    recorder so the recorder stays the outermost mj_step wrapper (see
    episode_view's docstring).
    """
    from g1_data import scripted_demo as SD
    ctx = view if view is not None else nullcontext()
    if not record:
        with ctx:
            return SD.run_episode(seed, cfg=cfg, demo=demo, book=book, policy=policy), None
    with ctx, ScriptedRecorder() as rec:
        result = SD.run_episode(seed, cfg=cfg, demo=demo, book=book, policy=policy)
    return result, rec


def make_view(seed: int, a):
    """An EpisodeView if any watch/film flag is set, else None."""
    if not (a.viewer or a.realtime or a.video_dir):
        return None
    from episode_view import EpisodeView
    return EpisodeView(seed, viewer=a.viewer, realtime=a.realtime,
                       video_dir=a.video_dir)


def finalize_view(view, accepted: bool) -> None:
    if view is None:
        return
    path = view.finalize(accepted)
    if path:
        print("  video -> %s (%d frames, %.1f MB)"
              % (path, view.n_frames, os.path.getsize(path) / 1e6))


def _setup():
    import torch
    import walk_test as W
    from g1_data import scripted_demo as SD
    cfg = TeleopConfig()
    # 2026-10-17, setting "A", chosen by Charles (CLAUDE.md section 8):
    #   reposition_s 6.0 -> 0.5  REPOSITION let a FREE base close the standoff
    #                          after the reach (scripted_demo.py:88-92); under
    #                          the D12 lock the base cannot move, so it was 6 s
    #                          of dead time (base 0.000 m/s, arm cmd 0.001 rad/s).
    #   hold_max stays 0.80    (the demonstrator default). Measured on the first
    #                          50 train seeds, headless: 0.80 -> 49/50, 0.60 ->
    #                          43/50 (7 placement failures at box y < -0.09),
    #                          0.40 -> 49/50 but goal-platform clearance < 10 mm
    #                          on 45/50. A's one failure: seed 25, PLACE(xy).
    # Episode timing changes: 548-692 ticks, so the D13 range (694-846, measured
    # with reposition_s 6.0) does not describe episodes recorded with this.
    demo = SD.DemoConfig(walk_place=True, start_xy=(0.60, 0.00), settle_s=14.0,
                         lock_predicate=True, hold_max=0.80, reposition_s=0.5)
    return cfg, demo, SD.PoseBook(cfg, demo), torch.jit.load(W.POLICY_PATH)


def cmd_invariance(a):
    """Does RECORDING change the episode? It must not (observation only)."""
    cfg, demo, book, policy = _setup()
    seed = a.invariance
    print("seed %d: running WITHOUT the recorder..." % seed)
    r_plain, _ = run_one(seed, cfg, demo, book, policy, record=False)
    print("seed %d: running WITH the recorder..." % seed)
    r_rec, rec = run_one(seed, cfg, demo, book, policy, record=True)
    keys = sorted(set(r_plain) & set(r_rec))
    diffs = []
    for k in keys:
        x, y = r_plain[k], r_rec[k]
        try:
            same = bool(np.array_equal(np.asarray(x, dtype=float),
                                       np.asarray(y, dtype=float), equal_nan=True))
        except (TypeError, ValueError):
            same = (x == y)
        if not same:
            diffs.append((k, x, y))
    print("compared %d result fields; %d differ" % (len(keys), len(diffs)))
    for k, x, y in diffs:
        print("   %-24s %r -> %r" % (k, x, y))
    print("recorded %d ticks" % len(rec.buf))
    print("OBSERVATION ONLY: %s" % ("CONFIRMED" if not diffs else "VIOLATED"))
    return 0 if not diffs else 1


def cmd_record(a):
    # A3: the scripted demonstrator may not write into the collection namespace.
    assert_namespace(a.out, SOURCE_SCRIPTED, where="record_episodes --out")
    ledger = EpisodeLedger(ledger_for(a.out))
    if a.ledger_summary:
        s = ledger.summary()
        print("\n".join("%-18s %s" % (k, v) for k, v in s.items()))
        return 0
    cfg, demo, book, policy = _setup()
    stream = seed_stream(a.seeds, a.count + 1000, a.start_seed)
    pending = ledger.pending(stream)[:a.count]
    if not pending:
        print("nothing pending: every seed in the stream already has an ACCEPT")
        return 0
    # A2: DERIVED, not operator-supplied. This tool drives the scripted
    # demonstrator and nothing else, so its source string is known here and the
    # ledger label follows from it. The old `--label` defaulted to "scripted",
    # which meant a piloted episode recorded without the flag was labelled
    # scripted - a provenance field that is wrong by default is worse than none.
    label = label_of(SOURCE_SCRIPTED, where="record_episodes")
    ledger.append("session", label=label, seeds=pending, auto=a.auto,
                  spec_version=spec.SPEC_VERSION)
    print("session %r: %d seed(s) pending -> %s" % (label, len(pending), pending))
    accepted = attempted = 0
    for seed in pending:
        # --until-accepted: stop once enough are SAVED; --count stays the cap on
        # attempts, so a run of failures cannot consume the stream unbounded.
        if a.until_accepted and accepted >= a.until_accepted:
            break
        attempted += 1
        while True:
            ledger.append("issue", seed=seed, label=label)
            print("\n=== seed %d ===" % seed)
            t0 = time.perf_counter()
            view = make_view(seed, a)
            try:
                result, rec = run_one(seed, cfg, demo, book, policy, view=view)
            except Exception as e:                          # noqa: BLE001
                # Includes episode_view.ViewerPerturbation: a viewer that
                # changed the physics fails the episode outright.
                ledger.append("error", seed=seed, reason=repr(e))
                print("  ERROR: %r (recorded in the ledger, seed left pending)" % e)
                finalize_view(view, accepted=False)
                break
            # The model the episode ran on is the one the recorder saw.
            meta = rec.metadata(rec.model, result, cfg, demo, seed,
                                extra=dict(label=label,
                                           wall_seconds_total=time.perf_counter() - t0))
            rows, recommend = verdict(meta)
            print_verdict(meta, rows, recommend)
            choice = decide(a.auto, recommend)
            finalize_view(view, accepted=(choice == "accept"))
            if choice == "accept":
                path = os.path.join(a.out, "ep_seed%04d.npz" % seed)
                rec.buf.save(path, meta)
                ledger.append("accept", seed=seed, path=repo_relpath(path),
                              heldout=meta["heldout"], label=label,
                              checks={n: ok for n, ok, _ in rows},
                              placement_error=meta["outcome"]["placement_error"])
                print("  ACCEPTED -> %s (%.0f KB)"
                      % (path, os.path.getsize(path) / 1024))
                accepted += 1
                break
            if choice == "discard":
                reason = ", ".join(n for n, ok, _ in rows if not ok) or "operator"
                ledger.append("discard", seed=seed, reason=reason, label=label,
                              checks={n: ok for n, ok, _ in rows})
                print("  DISCARDED (%s) - nothing written" % reason)
                break
            print("  RETRY (the scripted demonstrator is deterministic: the same "
                  "seed produces the same episode)")
    print("\naccepted %d of %d attempted (cap %d); ledger %s"
          % (accepted, attempted, len(pending), ledger.path))
    print("\n".join("%-18s %s" % (k, v) for k, v in ledger.summary().items()))
    return 0


def decide(auto: str, recommend: bool) -> str:
    if auto == "pass":
        return "accept" if recommend else "discard"
    if auto == "all":
        return "accept"
    while True:
        r = input("  [a]ccept / [d]iscard / [r]etry ? ").strip().lower()
        if r in ("a", "accept"):
            return "accept"
        if r in ("d", "discard"):
            return "discard"
        if r in ("r", "retry"):
            return "retry"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=1,
                    help="seeds to attempt (the cap, with --until-accepted)")
    ap.add_argument("--until-accepted", type=int, default=0,
                    help="stop once this many episodes are accepted (0 = attempt all --count)")
    # No --label: it is derived from the recorder-set source (A2). An operator
    # flag that defaults to a provenance claim is a field that lies quietly.
    ap.add_argument("--seeds", default=None, help="JSON file of the pre-partitioned stream")
    ap.add_argument("--start-seed", type=int, default=0)
    # A2: ONE argument. The ledger is DERIVED from the output directory and is
    # not separately settable - an operator who can pair the new directory with
    # the old ledger eventually will, and that pairing silently skips every
    # seed the old ledger already accepted.
    ap.add_argument("--out-dir", "--out", dest="out", default=OUT_DIR,
                    help="output folder for episodes; its ledger is %s inside it "
                         "(default recordings/episodes/, which requires --seeds)"
                    % "ledger.jsonl")
    ap.add_argument("--viewer", action="store_true",
                    help="open the MuJoCo passive viewer (observation only)")
    ap.add_argument("--realtime", action="store_true",
                    help="pace the episode to wall-clock time (sleep only)")
    ap.add_argument("--video-dir", default=None,
                    help="save one MP4 per episode here (1280x720, 25 fps)")
    ap.add_argument("--ledger-summary", action="store_true")
    ap.add_argument("--auto", choices=("pass", "all", "off"), default="off",
                    help="pass: accept when every check passes (unattended runs)")
    ap.add_argument("--invariance", type=int, default=None,
                    help="run this seed with and without the recorder and diff the result")
    a = ap.parse_args()
    if (a.invariance is None and not a.ledger_summary and a.seeds is None
            and os.path.abspath(a.out) == os.path.abspath(OUT_DIR)):
        ap.error("refusing to write into %s without --seeds: the contiguous "
                 "fallback stream includes held-out-patch seeds. Pass --seeds "
                 "data/eval/train_seeds.json, or a different --out-dir."
                 % repo_relpath(OUT_DIR))
    raise SystemExit(cmd_invariance(a) if a.invariance is not None else cmd_record(a))


if __name__ == "__main__":
    main()
