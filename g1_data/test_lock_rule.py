"""Tests for the versioned lock rule (phases.LockConfig v1 / v2, 2026-10-17).

    python -m g1_data.test_lock_rule                    # fast + one scripted episode
    G1_SLOW_TESTS=1 python -m g1_data.test_lock_rule    # + the 20-spare-seed v2 run (~5 min)

v2 ("stance band only") is the default; v1 ("stance band + 10 mm / 0.8 s") stays
for every episode recorded under it. Each episode stores the rule it was
collected under, and replay / the phase labeller / deploy apply THAT rule.
"""
from __future__ import annotations

import functools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from g1_data import dataset as DS                                   # noqa: E402
from g1_data.phases import (LOCK_RULE_V1, LOCK_RULE_V2, LOCK_RULE_V3,  # noqa: E402
                            LOCK_RULES, LockConfig, lock_config_from_meta,
                            lock_rule_meta)

#: A v1 episode at the current spec: the staged fixtures were recorded under v1
#: and re-derived at 1.2.0 (2026-10-18) with every 1.1.0 column bit-identical.
#: (recordings/scripted_v1 is spec 1.1.0, which this build refuses.)
REC0 = os.path.join(ROOT, "data", "synthetic", "ep_seed0000.npz")
SPARE = [60, 63, 64, 65, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _ep(seed, rule=None, drop=False):
    from g1_data import spec
    meta = dict(seed=seed, spec_version=spec.SPEC_VERSION, contact_contract="x",
                box_spawn_xy=[1.5, -0.1])
    if not drop:
        meta["lock_rule"] = lock_rule_meta(rule)
    return DS.Episode("mem://seed%d.npz" % seed, meta)


def test_v3_is_the_default_and_v1_v2_are_kept():
    """v3 is the default since 2026-10-24 (deploy and new scripted recordings);
    v1 and v2 stay selectable and keep the carry release at every lock."""
    assert LockConfig() == LOCK_RULE_V3 and LOCK_RULE_V3.version == "v3"
    assert LOCK_RULE_V2.version == "v2"
    assert LOCK_RULE_V2.disp_max == float("inf")
    assert LOCK_RULE_V1.version == "v1" and LOCK_RULE_V1.disp_max == 0.010
    # Everything but the stillness term is shared.
    for f in ("settle_window", "debounce", "fwd_lo", "fwd_hi", "lat_max", "head_max_deg",
              "lift_min", "rest_tol", "withdraw", "done_radius"):
        assert getattr(LOCK_RULE_V1, f) == getattr(LOCK_RULE_V2, f), f
    assert set(LOCK_RULES) == {"v1", "v2", "v3"}
    # v3 is v2 with the carry release restricted to the pickup lock - nothing else
    assert LOCK_RULE_V3 == LockConfig(version="v3", carry_release_goal=False) == LockConfig()
    assert LOCK_RULE_V2.carry_release_goal is True and LOCK_RULE_V1.carry_release_goal is True


def test_v3_round_trips_and_old_v2_metadata_stays_v2():
    """A v3 rule stored and read back is v3; a v2 rule stored BEFORE the
    carry_release_goal field existed (scripted_v3's metadata) still reads as v2."""
    s = json.dumps(lock_rule_meta(LOCK_RULE_V3), allow_nan=False)
    assert lock_config_from_meta({"lock_rule": json.loads(s)}) == LOCK_RULE_V3
    assert json.loads(s)["name"] == "stance band only, carry-release at pickup only"
    old = lock_rule_meta(LOCK_RULE_V2)
    old.pop("carry_release_goal")
    assert lock_config_from_meta({"lock_rule": old}) == LOCK_RULE_V2
    bad = lock_rule_meta(LOCK_RULE_V3)
    bad["carry_release_goal"] = True                            # says v3, is not v3
    _raises(ValueError, lock_config_from_meta, {"lock_rule": bad})


def _state(geo, base_xy, box_xyz, welded):
    from g1_data import spec
    s = np.zeros(spec.STATE_DIM)
    s[spec.BASE_POS] = [base_xy[0], base_xy[1], 0.78]
    s[spec.BASE_QUAT] = [1.0, 0.0, 0.0, 0.0]                    # facing +x
    s[spec.BOX_POS] = box_xyz
    s[spec.BOX_QUAT] = [1.0, 0.0, 0.0, 0.0]
    s[spec.PALM_L_POS] = s[spec.PALM_R_POS] = box_xyz          # hands on the box
    s[spec.GRIP_L] = s[spec.GRIP_R] = 1.0 if welded else 0.0
    return s


def _run(pred, s, n):
    return [pred.update(s) for _ in range(n)]


def test_v3_carry_release_applies_at_the_pickup_lock_only():
    """Pickup lock (engaged unwelded), then welded and lifted: released under v2
    AND v3. Goal lock (engaged welded) with the box still 0.10 m up: released
    under v2 after the debounce, HELD under v3 - until the box is placed and the
    hands withdraw, which releases it under both."""
    from g1_data.phases import LockPredicate, PlatformGeometry
    from g1_teleop.config import TeleopConfig
    from g1_teleop.contact_contract import load_model
    geo = PlatformGeometry.resolve(load_model(TeleopConfig()))
    rest = geo.top_z + geo.box_half
    px, py = geo.pickup_xy
    gx, gy = geo.goal_xy
    for cfg, goal_release in ((LOCK_RULE_V2, True), (LOCK_RULE_V3, False)):
        pred = LockPredicate(geo, cfg)
        n = cfg.debounce
        # pickup: lock on the box, unwelded
        assert _run(pred, _state(geo, (px - 0.33, py), (px, py, rest), False), n)[-1] == "lock"
        assert pred.locked_welded is False
        # welded and lifted 0.10 m: the carry release fires (both rules)
        assert _run(pred, _state(geo, (px - 0.33, py), (px, py, rest + 0.10), True), n)[-1] == "release"
        # goal: lock while welded, box still at carry height
        high = _state(geo, (gx - 0.33, gy), (gx, gy, rest + 0.10), True)
        assert _run(pred, high, n)[-1] == "lock" and pred.locked_welded is True
        ev = _run(pred, high, n)
        assert (ev[-1] == "release") is goal_release, (cfg.version, ev)
        if not goal_release:
            # placed, released, hands withdrawn: the normal goal release
            from g1_data import spec
            placed = _state(geo, (gx - 0.33, gy), (gx, gy, rest), False)
            placed[spec.PALM_L_POS] = placed[spec.PALM_R_POS] = [gx - 0.40, gy, rest]
            assert _run(pred, placed, n)[-1] == "release"


def test_rule_is_stored_and_read_back_through_strict_json():
    for cfg in (LOCK_RULE_V1, LOCK_RULE_V2):
        s = json.dumps(lock_rule_meta(cfg), allow_nan=False)     # valid JSON: inf -> null
        assert lock_config_from_meta({"lock_rule": json.loads(s)}) == cfg
    assert json.loads(json.dumps(lock_rule_meta(LOCK_RULE_V2)))["name"] == "stance band only"
    assert lock_config_from_meta({}) == LOCK_RULE_V1, "no field means v1"
    assert lock_config_from_meta(None) == LOCK_RULE_V1
    _raises(ValueError, lock_config_from_meta, {"lock_rule": None})
    bad = lock_rule_meta(LOCK_RULE_V1)
    bad["disp_max"] = 0.030                                      # says v1, is not v1
    _raises(ValueError, lock_config_from_meta, {"lock_rule": bad})


def test_mixed_lock_rules_are_refused():
    DS.assert_uniform_lock_rule([_ep(0, LOCK_RULE_V2), _ep(1, LOCK_RULE_V2)])
    # an old episode (no field) IS v1, so it groups with an explicit v1
    DS.assert_uniform_lock_rule([_ep(0, drop=True), _ep(1, LOCK_RULE_V1)])
    e = _raises(DS.DatasetError, DS.assert_uniform_lock_rule,
                [_ep(0, LOCK_RULE_V1), _ep(1, LOCK_RULE_V2)])
    assert "v1" in str(e) and "v2" in str(e) and "stance band only" in str(e)
    _raises(DS.DatasetError, DS.assert_uniform_lock_rule, [_ep(0, drop=True), _ep(1, LOCK_RULE_V2)])


def test_scripted_episode_records_the_rule_in_force():
    """Read from the episode's own LockPredicate, not from a default: forcing
    v1 into the demonstrator must record v1."""
    import record_episodes as RE
    from g1_data import scripted_demo as SD
    cfg, demo, book, policy = RE._setup()
    for rule in (None, LOCK_RULE_V1):
        orig = SD.LockConfig
        if rule is not None:
            SD.LockConfig = lambda: rule
        try:
            r, rec = RE.run_one(60, cfg, demo, book, policy)
        finally:
            SD.LockConfig = orig
        meta = rec.metadata(rec.model, r, cfg, demo, 60, extra={})
        want = rule or LOCK_RULE_V3
        assert meta["lock_rule"]["version"] == want.version, meta["lock_rule"]
        assert lock_config_from_meta(meta) == want


def test_v1_episode_replays_with_its_own_rule():
    """A v1 episode with NO lock_rule field (as all of scripted_v1 was): replay
    must apply v1 and reproduce the recorded lock tick (seed 0: dLock 0).
    Bit-identity against the pre-change code was checked once on 2026-10-17
    (seeds 0, 1, 43). The field is dropped from a temp copy of a staged v1
    episode, so both readings - explicit v1 and "no field" - are exercised."""
    if not os.path.exists(REC0):
        print("      (skipped: no %s)" % REC0)
        return
    import shutil
    import tempfile
    import replay_check as RC
    from g1_data.recorder import load_episode
    _, meta = load_episode(REC0)
    assert lock_config_from_meta(meta) == LOCK_RULE_V1, meta.get("lock_rule")
    tmp = tempfile.mkdtemp()
    try:
        with np.load(REC0, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
        meta.pop("lock_rule", None)
        p = os.path.join(tmp, os.path.basename(REC0))
        np.savez_compressed(p, meta=np.array(json.dumps(meta)), **arrays)
        assert "lock_rule" not in load_episode(p)[1]
        r = RC.replay(p, verbose=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert r["lock_rule"] == "v1"
    assert r["lock_tick"] == meta["lock_engage_tick"], (r["lock_tick"], meta["lock_engage_tick"])


def test_slow_v2_on_the_spare_seeds_reproduces_19_of_20():
    """The adoption measurement: v2 scores 19/20 on the spare seeds, seed 72
    failing PLACE(upright), pickup lock at tick 32-48 (2026-10-17)."""
    if os.environ.get("G1_SLOW_TESTS") != "1":
        print("      (skipped: set G1_SLOW_TESTS=1, ~5 min)")
        return
    import record_episodes as RE
    from g1_data import scripted_demo as SD
    cfg, demo, book, policy = RE._setup()
    ok, fails, locks = 0, [], []
    orig = SD.LockConfig
    SD.LockConfig = lambda: LOCK_RULE_V2          # v2 explicitly: v3 is the default now
    try:
        for s in SPARE:
            r, rec = RE.run_one(s, cfg, demo, book, policy)
            ok += bool(r["ok"])
            locks.append(rec.lock_engage_tick)
            if not r["ok"]:
                fails.append((s, r["fail_phase"]))
            assert rec.lock_rule["version"] == "v2"
    finally:
        SD.LockConfig = orig
    print("      v2 spare seeds: %d/20, failures %s, pickup lock ticks %d-%d"
          % (ok, fails, min(locks), max(locks)))
    assert ok == 19 and fails == [(72, "PLACE(upright)")], (ok, fails)
    assert 32 <= min(locks) and max(locks) <= 48, locks


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
