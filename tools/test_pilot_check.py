"""Tests for tools/pilot_check.py on FAKE-source recordings. CPU only, no ZED.

    python tools/test_pilot_check.py

The folder is recorded by tools/record_teleop_live.py itself (`--source fake
--headless`), one synthetic grasp-and-lift take per seed, exactly as
tools/test_record_teleop_live.py records its fixture. That fixture grasps and
lifts and NEVER places, so every episode fails `success.evaluate`; the positive
path forces the verdict the way test_record_teleop_live's accept test does
(`_forced_success`), and the negative path runs the same folder WITHOUT it and
must FAIL on exactly that. A fourth seed is recorded unforced into failed/.

The BC smoke test runs a few dozen steps here (the tool's default is 300).
"""
from __future__ import annotations

import contextlib
import glob
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import pilot_check as PC                                            # noqa: E402
import record_teleop_live as RT                                     # noqa: E402
from g1_data import dataset as DS                                   # noqa: E402
from g1_data import success as SU                                   # noqa: E402

SEEDS = (0, 1, 2)          # the first training seeds (data/eval/train_seeds.json)
FAILED_SEED = 3
STANDOFF = 0.32
STOP_S = 14.0
SMOKE = ["--steps", "40", "--window-steps", "20", "--batch", "64"]
_CACHE: dict = {}


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _tmp() -> str:
    if "tmp" not in _CACHE:
        _CACHE["tmp"] = tempfile.mkdtemp(prefix="pilot_check_test_")
    return _CACHE["tmp"]


@contextlib.contextmanager
def _forced_success():
    """The fixture never places; force the four criteria to pass (as
    test_record_teleop_live.test_accept_path... does). Patches the module, so the
    recorder, the dataset guards, build_dataset and the loader all see it."""
    real = SU.evaluate

    def forced(arrays, meta, cfg=None, geo=None):
        r = real(arrays, meta, cfg, geo)
        r.update(episode_success=True, place=True, attributed_to="")
        return r
    SU.evaluate = forced
    try:
        yield
    finally:
        SU.evaluate = real


def _record(out: str, seed: int, stop_s: float) -> None:
    take = os.path.join(_tmp(), "take_s%d.npz" % seed)
    if not os.path.exists(take):
        assert RT.main(["--make-fake-take", take, "--seed-list", str(seed),
                        "--start-standoff", str(STANDOFF)]) == 0
    from g1_teleop.synthetic_source import load_recording
    g = load_recording(take).meta["grasp_on_s"]
    assert RT.main(["--source", "fake", "--replay", take, "--seed-list", str(seed),
                    "--start-standoff", str(STANDOFF), "--headless", "--operator", "tester",
                    "--out-dir", out, "--script", "%.3f:g,%.3f:e" % (g + 0.1, stop_s)]) == 0


def _pilot_folder() -> str:
    """3 accepted (forced) grasp episodes + 1 failed, one ledger. Cached."""
    if "folder" not in _CACHE:
        out = os.path.join(_tmp(), "pilot")
        with _forced_success():
            for s in SEEDS:
                _record(out, s, STOP_S)
        _record(out, FAILED_SEED, 1.0)
        assert len(glob.glob(os.path.join(out, "*.npz"))) == len(SEEDS)
        assert len(glob.glob(os.path.join(out, "failed", "*.npz"))) == 1
        _CACHE["folder"] = out
    return _CACHE["folder"]


def _argv(folder, scratch, *extra):
    return ["--episodes", folder, "--scratch", scratch, "--allow-fake",
            "--expect", str(len(SEEDS)), "--val-frac", "0.34"] + SMOKE + list(extra)


def _report(scratch) -> dict:
    with open(os.path.join(scratch, "pilot_report.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["by_name"] = {c["name"]: c for c in doc["checks"]}
    return doc


def _processed_snapshot():
    return sorted((f, os.path.getmtime(os.path.join(DS.PROCESSED, f)))
                  for f in os.listdir(DS.PROCESSED)) if os.path.isdir(DS.PROCESSED) else []


# ─── end to end ──────────────────────────────────────────────────────────────
def test_pilot_passes_end_to_end_on_a_good_folder():
    folder = _pilot_folder()
    scratch = os.path.join(_tmp(), "scratch_pass")
    before = _processed_snapshot()
    with _forced_success():
        rc = PC.main(_argv(folder, scratch))
    doc = _report(scratch)
    bad = {n: (c["status"], c["reasons"]) for n, c in doc["by_name"].items()
           if c["status"] not in (PC.PASS, PC.NA)}
    assert rc == 0 and doc["verdict"] == "PASS", bad
    by = doc["by_name"]
    inv = by["inventory"]["details"]
    assert (inv["accepted"], inv["failed"], inv["discarded"]) == (3, 1, 0)
    assert inv["seeds"] == list(SEEDS) and inv["attempts"] == 4 and inv["accept_rows"] == 3
    rp = by["replay"]["details"]
    assert rp["fingerprint_matched"] == rp["manipulation_reproduced"] == 3
    sn = by["split + norm"]["details"]
    assert sn["train"] == 2 and sn["val"] == 1
    sm = by["BC smoke"]["details"]
    assert sm["status"] == "hard_cap"            # a step budget run to its cap
    assert sm["final_deployment_recon"] < sm["zero_baseline"]
    assert sm["final_train_loss"] < sm["first_window_train_loss"]
    assert by["timing + dropout"]["status"] == PC.NA      # headless: not measured
    # everything in the scratch folder, nothing in data/processed
    for f in ("splits.json", "norm_stats.npz", "bc_smoke/result.json", "pilot_report.txt",
              "replay.log", "split.log", "norm.log", "bc_smoke.log"):
        assert os.path.exists(os.path.join(scratch, f)), f
    assert _processed_snapshot() == before, "pilot_check wrote into data/processed"
    txt = open(os.path.join(scratch, "pilot_report.txt"), encoding="utf-8").read()
    assert "PILOT CHECK: PASS" in txt and "DRY RUN" in txt


def test_unsuccessful_episodes_fail_the_pilot_and_block_split_and_training():
    """The same folder with the REAL success criteria: the fixture never places."""
    folder = _pilot_folder()
    scratch = os.path.join(_tmp(), "scratch_fail")
    rc = PC.main(_argv(folder, scratch, "--skip-replay"))
    doc = _report(scratch)
    by = doc["by_name"]
    assert rc == 1 and doc["verdict"] == "FAIL"
    assert by["every episode succeeded"]["status"] == PC.FAIL
    assert "did not succeed" in by["every episode succeeded"]["reasons"][0]
    assert by["split + norm"]["status"] == by["BC smoke"]["status"] == PC.SKIP
    # the other guards still ran and passed: every failure is listed, not just the first
    for name in ("spec version + tick lengths", "uniform spec + contact contract",
                 "contact contract is D18's", "one lock rule", "grip label",
                 "no held-out spawn", "real demonstrations", "training seeds only"):
        assert by[name]["status"] == PC.PASS, (name, by[name])


def test_fake_sources_and_a_short_pilot_are_refused_without_the_flags():
    folder = _pilot_folder()
    scratch = os.path.join(_tmp(), "scratch_real")
    with _forced_success():
        rc = PC.main(["--episodes", folder, "--scratch", scratch, "--skip-replay",
                      "--skip-train"])
    by = _report(scratch)["by_name"]
    assert rc == 1
    assert by["real demonstrations"]["status"] == PC.FAIL
    assert "--allow-fake" in by["real demonstrations"]["reasons"][0]
    assert by["inventory"]["status"] == PC.FAIL
    assert "3 accepted episode(s), 25 expected" in by["inventory"]["reasons"][0]


def test_duplicate_seed_and_ledger_disagreement_fail_the_inventory():
    folder = os.path.join(_tmp(), "pilot_dup")
    shutil.copytree(_pilot_folder(), folder)
    first = sorted(glob.glob(os.path.join(folder, "ep_seed0000_*.npz")))[0]
    shutil.copy2(first, first.replace(".npz", "_copy.npz"))          # seed 0 twice
    with open(os.path.join(folder, "ledger.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(event="accept", seed=5, path="gone.npz")) + "\n")
    scratch = os.path.join(_tmp(), "scratch_dup")
    with _forced_success():
        rc = PC.main(_argv(folder, scratch, "--skip-replay", "--skip-train"))
    inv = _report(scratch)["by_name"]["inventory"]
    assert rc == 1 and inv["status"] == PC.FAIL
    text = " ".join(inv["reasons"])
    assert "more than one accepted episode" in text and "[0]" in text
    assert "no episode on disk: seeds [5]" in text


def test_skipping_a_check_makes_the_verdict_incomplete_not_pass():
    scratch = os.path.join(_tmp(), "scratch_skip")
    with _forced_success():
        rc = PC.main(_argv(_pilot_folder(), scratch, "--skip-replay", "--skip-train"))
    doc = _report(scratch)
    assert rc == 2 and doc["verdict"] == "INCOMPLETE"
    assert doc["by_name"]["split + norm"]["status"] == PC.PASS


# ─── the pieces ──────────────────────────────────────────────────────────────
def _good_result(**kw):
    r = dict(status="finished", final_train_loss=0.2, final_deployment_recon=0.15,
             final_val_loss=0.3, baselines=dict(zero=0.8, copy=0.05),
             history=[dict(train_loss=0.6), dict(train_loss=0.2)])
    r.update(kw)
    return r


def test_smoke_verdict():
    assert PC.smoke_verdict(_good_result()) == []
    assert "did not complete" in PC.smoke_verdict(_good_result(status="crashed"))[0]
    assert PC.smoke_verdict(_good_result(status="hard_cap")) == []
    assert "not finite" in PC.smoke_verdict(_good_result(final_val_loss=math.nan))[0]
    assert "did not fall" in PC.smoke_verdict(
        _good_result(history=[dict(train_loss=0.2), dict(train_loss=0.3)]))[0]
    assert "ZERO baseline" in PC.smoke_verdict(_good_result(final_deployment_recon=0.9))[0]


def _meta(seed, rate=25.0, dropout=0.0, overruns=0, n=600, paced=True, source="teleop_live"):
    return dict(seed=seed, n_ticks=n, source=source, dropout_rate=dropout,
                degraded_ticks=int(dropout * n), end_reason="success",
                timing=dict(paced=paced, tick_rate_hz=rate, overrun_ticks=overruns,
                            max_tick_interval_ms=55.0))


def test_timing_summary_fails_on_the_median_and_warns_per_episode():
    th = (24.0, 0.05, 0.02)
    ok = [_meta(s) for s in range(5)]
    assert PC.timing_summary(ok, *th).status == PC.PASS
    one_bad = ok[:4] + [_meta(9, rate=20.0, dropout=0.2, overruns=30)]
    c = PC.timing_summary(one_bad, *th)
    assert c.status == PC.WARN and len(c.reasons) == 3, c.reasons
    slow = [_meta(s, rate=21.0) for s in range(5)]
    c = PC.timing_summary(slow, *th)
    assert c.status == PC.FAIL and "median tick rate 21.00" in c.reasons[0]
    lost = [_meta(s, dropout=0.08) for s in range(5)]
    assert PC.timing_summary(lost, *th).status == PC.FAIL
    # a REAL episode without wall-clock timing is a FAIL; a headless fake one is N/A
    assert PC.timing_summary([_meta(0, paced=False)], *th).status == PC.FAIL
    assert PC.timing_summary([_meta(0, paced=False, source="teleop_live_fake")],
                             *th).status == PC.NA


def test_verdict_and_refusals():
    C = PC.Check
    assert PC.verdict([C("a", PC.PASS), C("b", PC.WARN), C("c", PC.NA)]) == "PASS"
    assert PC.verdict([C("a", PC.PASS), C("b", PC.SKIP)]) == "INCOMPLETE"
    assert PC.verdict([C("a", PC.FAIL), C("b", PC.SKIP)]) == "FAIL"
    full = os.path.join(_tmp(), "scratch_full")
    os.makedirs(full, exist_ok=True)
    open(os.path.join(full, "x"), "w").close()
    _raises(SystemExit, PC.main, ["--episodes", _tmp(), "--scratch", full])
    _raises(SystemExit, PC.main, ["--episodes", _tmp(), "--steps", "45",
                                  "--window-steps", "20"])
    _raises(SystemExit, PC.main, ["--episodes", os.path.join(_tmp(), "nope")])


def test_zzz_cleanup():
    """Last (sorted): remove the temp tree."""
    if "tmp" in _CACHE:
        shutil.rmtree(_CACHE.pop("tmp"), ignore_errors=True)
        _CACHE.clear()


def _tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _tests():
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(_tests()) - failed}/{len(_tests())} passed")
    raise SystemExit(1 if failed else 0)
