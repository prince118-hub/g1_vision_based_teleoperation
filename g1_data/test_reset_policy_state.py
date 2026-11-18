"""Tests for g1_data.reset.reset_policy_state on the real locomotion policy.

    python -m g1_data.test_reset_policy_state
    pytest g1_data/test_reset_policy_state.py

The 2026-11-18 leak: `motion.pt` runs with grad mode on, each forward pass
records an autograd graph that saves its inputs, and the in-place copy into
`hidden_state` / `cell_state` chains every step's graph to the previous one's.
A reset by `zero_()` alone kept that history, so it grew across every episode
of a process until allocation failed. Each test runs the OLD reset as a
control, so a test that cannot fail would be caught.
"""
from __future__ import annotations

import gc
import os
import sys
import weakref

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_data.reset import RECURRENT_BUFFERS, reset_policy_state    # noqa: E402
from g1_teleop import operator_config as OC                        # noqa: E402

N_OBS = 47


def _loco():
    return torch.jit.load(OC.POLICY_PATH)


def _obs_seq(n=40, seed=0):
    return np.random.default_rng(seed).normal(size=(n, N_OBS)).astype(np.float32)


def _run(loco, seq):
    """The deploy loop's exact call: grad mode on, from_numpy, detach."""
    return np.stack([loco(torch.from_numpy(o).unsqueeze(0)).detach().numpy().squeeze()
                     for o in seq])


def _old_reset(loco):
    for name in RECURRENT_BUFFERS:
        getattr(loco, name).zero_()


def _input_survives_reset(reset) -> bool:
    """Feed one tracked input, run more steps, reset, collect: is it still alive?"""
    loco = _loco()
    probe = torch.from_numpy(_obs_seq(1, seed=1)[0])   # the base: what the graph keeps
    ref = weakref.ref(probe)
    loco(probe.unsqueeze(0))
    del probe
    _run(loco, _obs_seq(10))
    reset(loco)
    gc.collect()
    return ref() is not None


def test_reset_drops_autograd_history():
    assert torch.is_grad_enabled()                 # the deploy loop's condition
    assert _input_survives_reset(_old_reset), "control: the old reset should keep the graph"
    assert not _input_survives_reset(reset_policy_state)


def test_reset_leaves_buffers_zero_and_detached():
    loco = _loco()
    _run(loco, _obs_seq(10))
    assert all(getattr(loco, n).grad_fn is not None for n in RECURRENT_BUFFERS)
    assert reset_policy_state(loco) == list(RECURRENT_BUFFERS)
    for n in RECURRENT_BUFFERS:
        b = getattr(loco, n)
        assert b.grad_fn is None and not b.requires_grad
        assert torch.count_nonzero(b) == 0


def test_episode_after_reset_is_bit_identical_to_a_fresh_policy():
    """Episode 1 of a fresh policy, then episode 2 after each reset: same bits.
    The old reset gives the same bits too - the fix changes memory, not values."""
    seq = _obs_seq()
    fresh = _run(_loco(), seq)
    for reset in (reset_policy_state, _old_reset):
        loco = _loco()
        _run(loco, _obs_seq(25, seed=7))           # a different "previous episode"
        reset(loco)
        assert np.array_equal(_run(loco, seq), fresh), reset.__name__


def _all_tests():
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    failed = 0
    tests = _all_tests()
    for name, fn in tests:
        try:
            fn()
            print("ok   ", name)
        except Exception as e:                    # noqa: BLE001
            failed += 1
            print("FAIL ", name, "-", repr(e))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
