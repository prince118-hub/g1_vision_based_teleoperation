"""Tests for tools/deploy.py - closed-loop evaluation.

    python tools/test_deploy.py

Needs the scripted_v1 recordings and trained runs (gitignored); a test whose
inputs are missing prints "skipped" rather than failing, like the other suites.

The sanity test is the one that matters: a "policy" that returns a recorded
episode's actions, pushed through deploy.py's loop, must reproduce
replay_check's trajectory BIT-FOR-BIT. That proves the loop routes an action
exactly as the replay test - and therefore collection - does, so a policy's
failure in deploy.py is the policy's, not the harness's.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import deploy as D                                                  # noqa: E402
from g1_data import spec                                            # noqa: E402

REC = os.path.join(ROOT, "recordings", "scripted_v1", "ep_seed0000.npz")
RUNS = os.path.join(ROOT, "runs", "scripted_v1")


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def test_post_process_overwrites_untrained_dims_and_thresholds_the_weld():
    a = np.full(spec.ACTION_DIM, 0.3)
    a[spec.GRIP_L_A], a[spec.GRIP_R_A] = 0.49, 0.9       # dim 18 is untrained noise
    a[spec.VEL_A] = [2.0, -2.0, 0.7]
    p = D.post_process(a)
    assert np.all(p[[6, 13, 14, 15, 16]] == 0.0)
    assert p[spec.GRIP_L_A] == 0.0 and p[spec.GRIP_R_A] == 0.0
    assert np.allclose(p[spec.VEL_A], spec.VELOCITY_CLIP[:, 1] * [1, -1, 1])
    keep = [d for d in range(spec.ACTION_DIM)
            if d not in (6, 13, 14, 15, 16, 17, 18, 19, 20, 21)]
    assert np.all(p[keep] == 0.3), "trainable joint targets must pass through untouched"
    for g, want in ((0.5, 1.0), (1.7, 1.0), (-0.2, 0.0), (0.499999, 0.0)):
        a[spec.GRIP_L_A] = g
        p = D.post_process(a)
        assert p[spec.GRIP_L_A] == want == p[spec.GRIP_R_A], (g, p[spec.GRIP_L_A])
    spec.assert_velocity_within_clip(p[spec.VEL_A])


def test_natural_start_matches_the_demonstrators_tick_zero():
    """reset_episode + base at (0.60, 0.00) is EXACTLY the recorded start."""
    if not os.path.exists(REC):
        print("      (skipped: no %s)" % REC)
        return
    from g1_data.phases import lock_config_from_meta
    pol = D.ReplayPolicy(REC)
    res, arrays = D.deploy_episode(int(pol.meta["seed"]), pol, max_ticks=1,
                                   early_success=False, raw_actions=True,
                                   lock_rule=lock_config_from_meta(pol.meta))
    assert res["reset_fingerprint"] == pol.meta["reset_fingerprint"]
    from g1_data.recorder import load_episode
    rec, _ = load_episode(REC)
    s0 = arrays["states"][0].astype(np.float32)
    assert np.array_equal(s0, rec["states"][0]), np.abs(s0 - rec["states"][0]).max()


def test_replayed_actions_through_the_deploy_loop_match_replay_check():
    """Bit-for-bit, from the same start, with the same actions."""
    if not os.path.exists(REC):
        print("      (skipped: no %s)" % REC)
        return
    import replay_check as RC
    from g1_data.recorder import load_episode
    rec, meta = load_episode(REC)
    pol = D.ReplayPolicy(REC)
    T = len(rec["actions"])
    from g1_data.phases import lock_config_from_meta
    res, arrays = D.deploy_episode(
        int(meta["seed"]), pol, max_ticks=T, early_success=False, raw_actions=True,
        initial_state=(rec["qpos"][0], rec["qvel"][0]),
        lock_rule=lock_config_from_meta(meta))       # the episode's own rule: v1
    assert res["lock_rule"]["version"] == "v1"
    rep = RC.replay(REC, verbose=False)
    div = np.abs(arrays["states"] - np.asarray(rec["states"], dtype=np.float64))
    assert res["ticks"] == T and not res["fell"]
    assert np.array_equal(div, rep["div"]), \
        "deploy loop diverges from replay_check: max %.3e" % np.abs(div - rep["div"]).max()
    assert res["weld_engage_tick"] == rep["weld_tick"], (res["weld_engage_tick"], rep["weld_tick"])
    assert res["lock_ticks"][0] == rep["lock_tick"], (res["lock_ticks"], rep["lock_tick"])


def _determinism(run, ckpt, seed, max_ticks):
    from g1_model import train as T
    T.set_determinism(0, strict=True)
    pol = D.CheckpointPolicy(os.path.join(RUNS, run), ckpt)
    out = []
    for _ in range(2):
        res, arrays = D.deploy_episode(seed, pol, max_ticks=max_ticks)
        res.pop("wall_seconds"), res.pop("sim_to_wall")
        out.append((res, arrays))
    (r1, a1), (r2, a2) = out
    assert r1 == r2, {k: (r1[k], r2[k]) for k in r1 if r1[k] != r2[k]}
    for k in a1:
        assert np.array_equal(a1[k], a2[k]), k


def test_same_checkpoint_and_seed_twice_is_identical_bc():
    if not os.path.exists(os.path.join(RUNS, "bc_seed0", "best.pt")):
        print("      (skipped: no bc_seed0/best.pt)")
        return
    _determinism("bc_seed0", "best.pt", 60, D.MAX_TICKS)


def test_same_checkpoint_and_seed_twice_is_identical_act_lstm():
    """Shortened: ACT-LSTM inference on the GPU, ensembling every tick."""
    if not os.path.exists(os.path.join(RUNS, "act_lstm_seed0", "best.pt")):
        print("      (skipped: no act_lstm_seed0/best.pt)")
        return
    _determinism("act_lstm_seed0", "best.pt", 60, 150)


def test_out_file_is_never_appended_to():
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "x.jsonl")
    open(p, "w").close()
    _raises(SystemExit, D.main, ["--run", RUNS, "--seed-list", "60", "--out", p])


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
