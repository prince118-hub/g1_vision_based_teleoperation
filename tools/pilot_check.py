"""After the 25-episode human pilot: does the data hold up, and does it train?

    python tools/pilot_check.py                         # recordings/demonstrations, 25 expected
    python tools/pilot_check.py --episodes recordings/demonstrations --expect 25 --steps 300

    # a dry run on fake-source episodes (never demonstrations):
    python tools/pilot_check.py --episodes recordings/teleop_fake --allow-fake --expect 3

ONE COMMAND, SIX CHECKS, ONE VERDICT
------------------------------------
1. inventory   accepted episodes (top level) >= --expect; one episode per seed;
               the ledger agrees with the disk (dataset.coverage); failed/ and
               discarded/ counted, failures by attributed phase; the attempt
               -> accept rate (the first number on teleop success).
2. guards      the dataset builder's own refusals, each run separately so ALL
               failures are listed: spec version and per-tick lengths
               (`load_episode`), uniform spec + contact contract, the contact
               contract equal to the CURRENT code's D18 contract, one lock
               rule (WARN if it is not `LockConfig()`'s), the grip label, one
               demonstrator setting, success (`assert_successful`), no held-out
               spawn, every source a REAL demonstration (`teleop_live`; fake
               only with --allow-fake), every seed a training seed and none an
               exp1 / exp2 / selection seed.
3. replay      tools/replay_check.replay on EVERY accepted episode: reset
               fingerprint and MANIPULATION reproduced (the criterion
               replay_check scores). The whole task is not expected to replay
               open loop (the walk; NOTES 2026-09-16) and is reported only.
4. split+norm  tools/build_dataset.py split and norm, into the SCRATCH folder
               only (never data/processed): the val split must be non-empty.
5. BC smoke    tools/train_bc.py train, BC (K=1, W_o=12), --steps optimizer
               steps on the scratch split: finished, every loss finite, train
               loss falling, and the final deployed reconstruction below the
               ZERO baseline (the constant predictor). Says the data TRAINS -
               nothing about BC (untuned, a few hundred steps, O31).
6. timing      per-episode tick rate, overruns and tracking dropout from each
               episode's metadata. FAIL on the MEDIAN (a systematic camera or
               machine problem); WARN per episode (the operator accepted it).
               Thresholds: --min-tick-rate 24 Hz (LIVE_RECORDING_CHECKLIST
               step 2); --max-dropout 0.05 and --max-overrun-frac 0.02 (the
               operator guide's PROVISIONAL recommend-discard limits).

Steps 4-5 run only when the guards pass (build_dataset would refuse with the
same reason). Verdict: PASS (exit 0) - no FAIL and nothing skipped; FAIL (1);
INCOMPLETE (2) - nothing failed but a check was skipped (--skip-replay,
--skip-train). Everything is written under --scratch (default
runs/pilot_check/<stamp>/, gitignored): logs of each tool, splits.json,
norm_stats.npz, bc_smoke/, pilot_report.txt and pilot_report.json.

Run it on the NVIDIA laptop (O34 Option B); CPU works, the smoke test is small.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import math
import os
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data.ledger import EpisodeLedger                            # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402
from g1_data import recorder as REC                                 # noqa: E402

PASS, FAIL, WARN, SKIP, NA = "PASS", "FAIL", "WARN", "SKIP", "N/A"
FAILED_DIR, DISCARDED_DIR, KP_DIR = "failed", "discarded", "keypoints"
SEED_FILES = ("exp1_seeds.json", "exp2_seeds.json", "select_seeds.json")
#: g1_model.train's final status of a run that completed: an epoch budget ends
#: "finished", a step budget "hard_cap" (or "stop_rule" if one was set).
DONE_STATUSES = ("finished", "hard_cap", "stop_rule")


@dataclass
class Check:
    name: str
    status: str
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)
    #: one line of numbers printed under the check, whatever its status
    summary: str = ""


def _first_lines(e: BaseException, n: int = 4) -> str:
    lines = [ln for ln in str(e).splitlines() if ln.strip()]
    return "%s: %s" % (type(e).__name__, " | ".join(lines[:n]) or "(no message)")


def _guard(name: str, fn: Callable[[], object]) -> Check:
    """Run one refusal; PASS if it does not raise, FAIL with its message if it does."""
    try:
        out = fn()
    except Exception as e:                                  # noqa: BLE001
        return Check(name, FAIL, [_first_lines(e)])
    if isinstance(out, Check):
        return out
    return Check(name, PASS)


# ─── 1. inventory ────────────────────────────────────────────────────────────
def inventory(ep_dir: str, expect: int):
    """(Check, accepted Episodes). Scans without the uniformity refusal - the
    guards apply it, so a mixed folder is reported there, not crashed on here."""
    eps = DS.scan(ep_dir, require_uniform=False)
    failed = DS.scan(os.path.join(ep_dir, FAILED_DIR), require_uniform=False)
    discarded = DS.scan(os.path.join(ep_dir, DISCARDED_DIR), require_uniform=False)
    kp = os.path.join(ep_dir, KP_DIR)
    n_kp = len([f for f in os.listdir(kp) if f.endswith(".npz")]) if os.path.isdir(kp) else 0
    reasons, status = [], PASS
    seeds = Counter(e.seed for e in eps)
    dup = sorted(s for s, n in seeds.items() if n > 1)
    if len(eps) < expect:
        status = FAIL
        reasons.append("%d accepted episode(s), %d expected (--expect)" % (len(eps), expect))
    if dup:
        status = FAIL
        reasons.append("seed(s) %s have more than one accepted episode - one per seed" % dup)
    led_path = REC.ledger_for(ep_dir)
    details: Dict[str, object] = dict(
        folder=repo_relpath(ep_dir), accepted=len(eps), failed=len(failed),
        discarded=len(discarded), keypoint_files=n_kp, seeds=sorted(seeds),
        operators=dict(Counter(str(e.meta.get("operator")) for e in eps)),
        failed_by_phase=dict(Counter(str((e.meta.get("outcome") or {}).get("attributed_to")
                                         or "?") for e in failed)))
    if not os.path.exists(led_path):
        status = FAIL
        reasons.append("no ledger at %s - the episodes cannot be traced to a session"
                       % repo_relpath(led_path))
    else:
        led = EpisodeLedger(led_path)
        cov = DS.coverage(led, eps)
        rows = led.rows()
        attempts = sum(1 for r in rows if r["event"] == "issue")
        accepts = sum(1 for r in rows if r["event"] == "accept")
        details.update(
            ledger=repo_relpath(led_path), sessions=sum(1 for r in rows if r["event"] == "session"),
            attempts=attempts, accept_rows=accepts,
            accept_rate=(accepts / attempts) if attempts else None,
            missing_from_disk=cov["missing_from_disk"], not_in_ledger=cov["not_in_ledger"])
        if cov["missing_from_disk"]:
            status = FAIL
            reasons.append("ledger ACCEPTs with no episode on disk: seeds %s"
                           % cov["missing_from_disk"])
        if cov["not_in_ledger"]:
            status = FAIL
            reasons.append("episodes on disk with no ledger ACCEPT: seeds %s"
                           % cov["not_in_ledger"])
    return Check("inventory", status, reasons, details), eps


# ─── 2. the dataset guards ───────────────────────────────────────────────────
def _assert_loads(eps) -> None:
    bad = []
    for e in eps:
        try:
            REC.load_episode(e.path)
        except Exception as ex:                             # noqa: BLE001
            bad.append("%s: %s" % (os.path.basename(e.path), _first_lines(ex, 2)))
    if bad:
        raise DS.DatasetError("%d episode(s) do not load:\n  %s" % (len(bad), "\n  ".join(bad)))


def _assert_current_contract(eps) -> None:
    from g1_teleop.config import TeleopConfig
    from g1_teleop.contact_contract import expected_contract
    want = expected_contract(TeleopConfig().contact)
    bad = sorted({e.meta.get("contact_contract") for e in eps} - {want})
    if bad:
        raise DS.DatasetError("contact contract differs from this code's D18 contract %r: %s"
                              % (want, bad))


def _lock_rule_is_default(eps) -> Check:
    from g1_data.phases import LockConfig, lock_config_from_meta
    want = LockConfig().version
    have = sorted({lock_config_from_meta(e.meta).version for e in eps})
    if have and have != [want]:
        return Check("lock rule is the default", WARN,
                     ["episodes are lock rule %s; LockConfig() is %s - deliberate "
                      "(--lock-rule) or a mistake? Training and deploy must use the "
                      "episodes' rule" % (have, want)])
    return Check("lock rule is the default", PASS, details=dict(rule=want))


def _assert_sources(eps, allow_fake: bool) -> None:
    kinds = Counter(e.meta.get("source") for e in eps)
    for src in kinds:
        REC.source_info(src, where="pilot_check")           # unknown -> raises
    fake = {s: n for s, n in kinds.items() if not REC.SOURCES[s]["real"]}
    if fake and not allow_fake:
        raise DS.DatasetError("%s: not real demonstrations (only %r is). A dry run "
                              "needs --allow-fake." % (fake, REC.SOURCE_TELEOP_LIVE))


def _assert_training_seeds(eps) -> None:
    train = set(DS.load_seed_file(os.path.join(DS.EVAL, "train_seeds.json")))
    held = {}
    for name in SEED_FILES:
        p = os.path.join(DS.EVAL, name)
        if os.path.exists(p):
            for s in DS.load_seed_file(p):
                held[int(s)] = name
    bad = sorted((e.seed, held[e.seed]) for e in eps if e.seed in held)
    off = sorted(e.seed for e in eps if e.seed not in train)
    if bad or off:
        raise DS.DatasetError("seeds that may not be demonstrated: %s%s"
                              % (["%d (%s)" % b for b in bad],
                                 "; not in train_seeds.json: %s" % off if off else ""))


def guards(eps, allow_fake: bool) -> List[Check]:
    return [
        _guard("spec version + tick lengths", lambda: _assert_loads(eps)),
        _guard("uniform spec + contact contract", lambda: DS.assert_uniform(eps)),
        _guard("contact contract is D18's", lambda: _assert_current_contract(eps)),
        _guard("one lock rule", lambda: DS.assert_uniform_lock_rule(eps)),
        _guard("lock rule is the default", lambda: _lock_rule_is_default(eps)),
        _guard("grip label", lambda: DS.assert_uniform_grip_label(eps)),
        _guard("one demonstrator setting", lambda: DS.assert_uniform_demonstrator(eps)),
        _guard("every episode succeeded", lambda: DS.assert_successful(eps)),
        _guard("no held-out spawn", lambda: DS.assert_no_leak(eps)),
        _guard("real demonstrations", lambda: _assert_sources(eps, allow_fake)),
        _guard("training seeds only", lambda: _assert_training_seeds(eps)),
    ]


# ─── 3. replay ───────────────────────────────────────────────────────────────
def replay_all(eps, log_path: str) -> Check:
    import replay_check as RC
    rows, reasons = [], []
    with open(log_path, "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        for e in eps:
            name = os.path.basename(e.path)
            try:
                r = RC.replay(e.path, verbose=True)
            except Exception as ex:                         # noqa: BLE001
                traceback.print_exc()
                rows.append(dict(episode=name, seed=e.seed, error=_first_lines(ex, 2)))
                reasons.append("%s: replay raised %s" % (name, _first_lines(ex, 2)))
                continue
            row = dict(episode=name, seed=e.seed,
                       fingerprint_ok=bool(r["reset_fingerprint_ok"]),
                       manipulation_ok=bool(r.get("manipulation_ok")),
                       task_reproduced=bool(r.get("task_success")),
                       weld_tick_delta=r.get("weld_tick_delta"),
                       lock_tick_delta=r.get("lock_tick_delta"),
                       box_while_locked_m=(r.get("locked") or {}).get("box"))
            rows.append(row)
            if not row["fingerprint_ok"]:
                reasons.append("%s: reset fingerprint MISMATCH" % name)
            elif not row["manipulation_ok"]:
                reasons.append("%s: manipulation NOT reproduced (dWeld %s, box while "
                               "locked %s m)" % (name, row["weld_tick_delta"],
                                                 row["box_while_locked_m"]))
    n = len(eps)
    details = dict(
        episodes=n, log=repo_relpath(log_path),
        fingerprint_matched=sum(1 for r in rows if r.get("fingerprint_ok")),
        manipulation_reproduced=sum(1 for r in rows if r.get("manipulation_ok")),
        whole_task_reproduced=sum(1 for r in rows if r.get("task_reproduced")),
        per_episode=rows,
        note="whole task is NOT expected to replay open loop (the walk); reported only")
    return Check("replay", FAIL if reasons else PASS, reasons, details,
                 "fingerprint %d/%d, manipulation reproduced %d/%d, whole task %d/%d (not scored)"
                 % (details["fingerprint_matched"], n, details["manipulation_reproduced"], n,
                    details["whole_task_reproduced"], n))


# ─── 4-5. split, norm, BC smoke: the repo's own tools, in this process ──────
def run_tool(main_fn: Callable[[], object], argv: List[str], log_path: str):
    """Run a tool's `main()` with `argv`, its output into `log_path`.
    Returns (exit code, last lines of the log). In-process, so a caller's
    monkeypatch (the tests) reaches the tool the same way it reaches us."""
    old = sys.argv
    sys.argv = [argv[0]] + list(argv[1:])
    code = 0
    with open(log_path, "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print("$ %s" % " ".join(argv))
        try:
            r = main_fn()
            code = int(r or 0)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if not isinstance(e.code, (int, type(None))):
                print(e.code)
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
            code = 1
        finally:
            sys.argv = old
    with open(log_path, encoding="utf-8") as fh:
        tail = [ln.rstrip() for ln in fh.readlines()[-12:]]
    return code, tail


def split_norm(ep_dir: str, scratch: str, val_frac: float):
    import build_dataset as BD
    splits = os.path.join(scratch, "splits.json")
    norm = os.path.join(scratch, "norm_stats.npz")
    details = dict(splits=repo_relpath(splits), norm=repo_relpath(norm))
    code, tail = run_tool(BD.main, ["build_dataset.py", "split", "--episodes", ep_dir,
                                    "--splits", splits, "--val-frac", str(val_frac)],
                          os.path.join(scratch, "split.log"))
    if code or not os.path.exists(splits):
        return Check("split + norm", FAIL, ["split exited %s:" % code] + tail, details), None
    sp = DS.load_splits(splits)
    details.update(train=len(sp["train"]), val=len(sp["val"]), val_seeds=sp["val"],
                   bins_without_val=sp.get("bins_without_val"))
    if not sp["val"] or not sp["train"]:
        return Check("split + norm", FAIL,
                     ["split has %d train / %d val - both must be non-empty (more "
                      "episodes, or --val-frac)" % (len(sp["train"]), len(sp["val"]))],
                     details), None
    code, tail = run_tool(BD.main, ["build_dataset.py", "norm", "--episodes", ep_dir,
                                    "--splits", splits, "--norm", norm],
                          os.path.join(scratch, "norm.log"))
    if code or not os.path.exists(norm):
        return Check("split + norm", FAIL, ["norm exited %s:" % code] + tail, details), None
    _, _, meta = DS.load_norm_stats(norm)
    details.update(norm_source=meta.get("source"), norm_episodes=meta.get("n_episodes"))
    return Check("split + norm", PASS, [], details,
                 "%d train / %d val, norm fitted on %s episode(s) (source %s)"
                 % (len(sp["train"]), len(sp["val"]), meta.get("n_episodes"),
                    meta.get("source"))), (splits, norm)


def smoke_verdict(res: dict) -> List[str]:
    """Reasons the BC smoke run does NOT show the data trains ([] = it does)."""
    why = []
    if res.get("status") not in DONE_STATUSES:
        why.append("run status %r - did not complete (%s)"
                   % (res.get("status"), "/".join(DONE_STATUSES)))
    vals = dict(final_train_loss=res.get("final_train_loss"),
                final_deployment_recon=res.get("final_deployment_recon"),
                final_val_loss=res.get("final_val_loss"))
    for k, v in vals.items():
        if v is None or not math.isfinite(float(v)):
            why.append("%s is %r (not finite)" % (k, v))
    hist = [h.get("train_loss") for h in res.get("history") or []]
    if len(hist) >= 2 and all(h is not None and math.isfinite(h) for h in hist) \
            and not hist[-1] < hist[0]:
        why.append("train loss did not fall: first window %.6f, last %.6f" % (hist[0], hist[-1]))
    zero = (res.get("baselines") or {}).get("zero")
    dep = vals["final_deployment_recon"]
    if zero is not None and dep is not None and math.isfinite(float(dep)) \
            and not float(dep) < float(zero):
        why.append("deployed reconstruction %.6f is not below the ZERO baseline %.6f "
                   "(the constant predictor)" % (float(dep), float(zero)))
    return why


def bc_smoke(ep_dir: str, paths, scratch: str, steps: int, window: int, batch: int) -> Check:
    import train_bc as TB
    splits, norm = paths
    run_dir = os.path.join(scratch, "bc_smoke")
    argv = ["train_bc.py", "train", "--episodes", ep_dir, "--splits", splits, "--norm", norm,
            "--model", "bc", "--chunk", "1", "--obs-window", "12",
            "--max-steps", str(steps), "--window-steps", str(window),
            "--batch", str(batch), "--out-dir", run_dir,
            "--budget-reason", "pilot_check smoke test: does the pilot data train "
                               "at all (not a result)"]
    code, tail = run_tool(TB.main, argv, os.path.join(scratch, "bc_smoke.log"))
    rpath = os.path.join(run_dir, "result.json")
    details = dict(run_dir=repo_relpath(run_dir), steps=steps, window_steps=window, batch=batch)
    if code or not os.path.exists(rpath):
        return Check("BC smoke", FAIL, ["train_bc exited %s:" % code] + tail, details)
    with open(rpath, encoding="utf-8") as fh:
        res = json.load(fh)
    base = res.get("baselines") or {}
    details.update(status=res.get("status"),
                   first_window_train_loss=(res.get("history") or [{}])[0].get("train_loss"),
                   final_train_loss=res.get("final_train_loss"),
                   final_deployment_recon=res.get("final_deployment_recon"),
                   final_val_loss=res.get("final_val_loss"),
                   zero_baseline=base.get("zero"), copy_baseline=base.get("copy"),
                   note="beating COPY is NOT required here (a few hundred steps)")
    why = smoke_verdict(res)
    num = lambda k: ("%.4f" % details[k]) if isinstance(details[k], (int, float)) else "n/a"
    return Check("BC smoke", FAIL if why else PASS, why, details,
                 "%d steps: train loss %s -> %s, deployed recon %s vs zero %s / copy %s, val %s"
                 % (steps, num("first_window_train_loss"), num("final_train_loss"),
                    num("final_deployment_recon"), num("zero_baseline"),
                    num("copy_baseline"), num("final_val_loss")))


# ─── 6. timing and dropout ───────────────────────────────────────────────────
def _stats(v):
    v = [float(x) for x in v if x is not None and math.isfinite(float(x))]
    if not v:
        return None
    return dict(n=len(v), min=min(v), median=float(np.median(v)), max=max(v))


def timing_summary(metas: List[dict], min_tick_rate: float, max_dropout: float,
                   max_overrun_frac: float) -> Check:
    """Tick rate, overruns and dropout from the episodes' own metadata."""
    reasons, warns, rows = [], [], []
    for m in metas:
        t = m.get("timing") or {}
        n = int(m.get("n_ticks") or 0)
        rows.append(dict(seed=m.get("seed"), n_ticks=n, paced=bool(t.get("paced")),
                         tick_rate_hz=t.get("tick_rate_hz"),
                         overrun_ticks=t.get("overrun_ticks"),
                         overrun_frac=(t.get("overrun_ticks") or 0) / n if n else None,
                         max_tick_interval_ms=t.get("max_tick_interval_ms"),
                         dropout_rate=m.get("dropout_rate"),
                         degraded_ticks=m.get("degraded_ticks"),
                         end_reason=m.get("end_reason")))
    real = [r for m, r in zip(metas, rows)
            if REC.SOURCES.get(m.get("source"), {}).get("real")]
    unpaced_real = [r["seed"] for r in real if not r["paced"]]
    if unpaced_real:
        reasons.append("real episode(s) without wall-clock timing (paced false): seeds %s"
                       % unpaced_real)
    paced = [r for r in rows if r["paced"]]
    details = dict(
        thresholds=dict(min_tick_rate_hz=min_tick_rate, max_dropout=max_dropout,
                        max_overrun_frac=max_overrun_frac,
                        note="tick rate: LIVE_RECORDING_CHECKLIST step 2; dropout and "
                             "overruns: the operator guide's PROVISIONAL limits"),
        n_ticks=_stats([r["n_ticks"] for r in rows]),
        dropout_rate=_stats([r["dropout_rate"] for r in rows]),
        tick_rate_hz=_stats([r["tick_rate_hz"] for r in paced]),
        overrun_frac=_stats([r["overrun_frac"] for r in paced]),
        max_tick_interval_ms=_stats([r["max_tick_interval_ms"] for r in paced]),
        end_reasons=dict(Counter(str(r["end_reason"]) for r in rows)),
        per_episode=rows)
    d = details["dropout_rate"]
    if d and d["median"] > max_dropout:
        reasons.append("median tracking dropout %.3f > %.3f - camera placement, lighting "
                       "or clothing (docs/RECORDING_PROTOCOL.md)" % (d["median"], max_dropout))
    for r in rows:
        if r["dropout_rate"] is not None and r["dropout_rate"] > max_dropout:
            warns.append("seed %s: dropout %.3f" % (r["seed"], r["dropout_rate"]))
    if paced:
        tr, ov = details["tick_rate_hz"], details["overrun_frac"]
        if tr and tr["median"] < min_tick_rate:
            reasons.append("median tick rate %.2f Hz < %.1f - the loop did not hold 25 Hz"
                           % (tr["median"], min_tick_rate))
        if ov and ov["median"] > max_overrun_frac:
            reasons.append("median overrun fraction %.3f > %.3f" % (ov["median"], max_overrun_frac))
        for r in paced:
            if r["tick_rate_hz"] is not None and r["tick_rate_hz"] < min_tick_rate:
                warns.append("seed %s: tick rate %.2f Hz" % (r["seed"], r["tick_rate_hz"]))
            if r["overrun_frac"] is not None and r["overrun_frac"] > max_overrun_frac:
                warns.append("seed %s: overruns %.3f of ticks" % (r["seed"], r["overrun_frac"]))
    elif rows:
        details["tick_rate_note"] = ("not measured: every episode is headless (paced false), "
                                     "a fake-source dry run")
    if reasons:
        return Check("timing + dropout", FAIL, reasons + warns, details)
    if warns:
        return Check("timing + dropout", WARN, warns, details)
    return Check("timing + dropout", PASS if paced or not rows else NA, [], details)


# ─── the run ─────────────────────────────────────────────────────────────────
def verdict(checks: List[Check]) -> str:
    st = {c.status for c in checks}
    if FAIL in st:
        return "FAIL"
    if SKIP in st:
        return "INCOMPLETE"
    return "PASS"


def render(checks: List[Check], header: List[str]) -> List[str]:
    out = list(header)
    for c in checks:
        out.append("[%-4s] %s" % (c.status, c.name))
        if c.summary:
            out.append("         %s" % c.summary)
        for r in c.reasons:
            out.append("         - %s" % r)
    return out


def _scratch_dir(path: Optional[str]) -> str:
    d = path or os.path.join(ROOT, "runs", "pilot_check",
                             _dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if os.path.isdir(d) and os.listdir(d):
        raise SystemExit("--scratch %s is not empty - refusing to mix two checks" % d)
    os.makedirs(d, exist_ok=True)
    return d


def run(a) -> int:
    ep_dir = os.path.abspath(a.episodes)
    if not os.path.isdir(ep_dir):
        raise SystemExit("no such folder %s" % ep_dir)
    if a.steps % a.window_steps:
        raise SystemExit("--steps (%d) must be a multiple of --window-steps (%d)"
                         % (a.steps, a.window_steps))
    scratch = _scratch_dir(a.scratch)
    say = print
    say("pilot check of %s -> scratch %s" % (repo_relpath(ep_dir), repo_relpath(scratch)))

    checks: List[Check] = []
    inv, eps = inventory(ep_dir, a.expect)
    checks.append(inv)
    say("[%s] inventory: %d accepted, %d failed, %d discarded"
        % (inv.status, inv.details["accepted"], inv.details["failed"], inv.details["discarded"]))
    g = guards(eps, a.allow_fake) if eps else [Check("dataset guards", FAIL, ["no episodes"])]
    checks.extend(g)
    guards_ok = all(c.status in (PASS, WARN) for c in g)
    say("[%s] dataset guards (%d)" % (PASS if guards_ok else FAIL, len(g)))

    if a.skip_replay:
        checks.append(Check("replay", SKIP, ["--skip-replay"]))
    elif eps:
        say("replaying %d episode(s) open loop ..." % len(eps))
        checks.append(replay_all(eps, os.path.join(scratch, "replay.log")))
        say("[%s] replay" % checks[-1].status)

    paths = None
    if not guards_ok:
        checks.append(Check("split + norm", SKIP, ["blocked: the dataset guards failed "
                                                   "(build_dataset refuses the same)"]))
    else:
        c, paths = split_norm(ep_dir, scratch, a.val_frac)
        checks.append(c)
        say("[%s] split + norm" % c.status)
    if a.skip_train:
        checks.append(Check("BC smoke", SKIP, ["--skip-train"]))
    elif paths is None:
        checks.append(Check("BC smoke", SKIP, ["blocked: no split / norm stats"]))
    else:
        say("BC smoke test: %d steps ..." % a.steps)
        checks.append(bc_smoke(ep_dir, paths, scratch, a.steps, a.window_steps, a.batch))
        say("[%s] BC smoke" % checks[-1].status)

    metas = [e.meta for e in eps]
    checks.append(timing_summary(metas, a.min_tick_rate, a.max_dropout, a.max_overrun_frac))

    v = verdict(checks)
    head = ["", "=" * 72,
            "PILOT CHECK  %s  (%s)" % (repo_relpath(ep_dir),
                                      _dt.datetime.now().isoformat(timespec="seconds")),
            "spec %s; scratch %s" % (spec.SPEC_VERSION, repo_relpath(scratch))]
    if a.allow_fake:
        head.append("DRY RUN: --allow-fake - fake-source episodes are NOT demonstrations")
    d = inv.details
    head.append("accepted %d (expected %d), failed %d, discarded %d; ledger attempts %s, "
                "accept rate %s" % (d["accepted"], a.expect, d["failed"], d["discarded"],
                                    d.get("attempts"),
                                    "%.0f%%" % (100 * d["accept_rate"])
                                    if d.get("accept_rate") is not None else "n/a"))
    tm = checks[-1].details
    for k, unit in (("dropout_rate", ""), ("tick_rate_hz", " Hz"), ("n_ticks", " ticks")):
        s = tm.get(k)
        if s:
            head.append("%-13s median %.3f%s (min %.3f, max %.3f, n %d)"
                        % (k, s["median"], unit, s["min"], s["max"], s["n"]))
    head.append("=" * 72)
    lines = render(checks, head)
    lines += ["", "PILOT CHECK: %s" % v]
    if v == "PASS":
        lines.append("  next: record the next 25 (docs/RECORDING_PROTOCOL.md section 11)")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(scratch, "pilot_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    with open(os.path.join(scratch, "pilot_report.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(verdict=v, episodes=repo_relpath(ep_dir), scratch=repo_relpath(scratch),
                       spec_version=spec.SPEC_VERSION, allow_fake=bool(a.allow_fake),
                       written_local=_dt.datetime.now().isoformat(timespec="seconds"),
                       args={k: v_ for k, v_ in vars(a).items()},
                       checks=[asdict(c) for c in checks]), fh, indent=1, default=str)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[v]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default=REC.NS_COLLECTION,
                    help="the episode folder (default recordings/demonstrations)")
    ap.add_argument("--expect", type=int, default=25,
                    help="accepted episodes required (default 25, the pilot)")
    ap.add_argument("--scratch", default=None,
                    help="output folder (default runs/pilot_check/<stamp>); must be empty")
    ap.add_argument("--allow-fake", action="store_true",
                    help="accept fake-source episodes (a dry run of this check)")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="validation fraction for the scratch split (default 0.2, the design)")
    ap.add_argument("--steps", type=int, default=300, help="BC smoke optimizer steps")
    ap.add_argument("--window-steps", type=int, default=50,
                    help="steps per logged window (--steps must be a multiple)")
    ap.add_argument("--batch", type=int, default=256, help="BC smoke batch size")
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--min-tick-rate", type=float, default=24.0)
    ap.add_argument("--max-dropout", type=float, default=0.05)
    ap.add_argument("--max-overrun-frac", type=float, default=0.02)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
