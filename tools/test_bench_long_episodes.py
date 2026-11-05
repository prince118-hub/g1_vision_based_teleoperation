"""tools/bench_long_episodes.py: the projection arithmetic, on hand-made timing rows.

    python -m pytest tools/test_bench_long_episodes.py -q

No model is timed here (the tool itself does that); the rows are fabricated so
every expected number follows from the module's stated anchors.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import bench_long_episodes as B                                     # noqa: E402


def _rows(scale_lstm_step, scale_score):
    """act_orig: step 0.7 s at every length, score linear in T; act_lstm_full:
    + an LSTM step part and an LSTM score part scaled as given."""
    rows = {}
    for T in (600, 3000):
        rows[("act_orig", T)] = dict(step_s=0.7, score_s_per_ep=6.0 * T / 600,
                                     peak_rss_gb=1.6, train_rss_gb=0.8)
        rows[("act_lstm_full", T)] = dict(
            step_s=0.7 + 1.7 * scale_lstm_step[T], peak_rss_gb=1.8, train_rss_gb=1.0,
            score_s_per_ep=6.0 * T / 600 + 2.0 * scale_score[T])
    return rows


def _mem():
    saved = {B.padded(T): int(1e8 + 1e5 * B.padded(T)) for T in (600, 692, 3000)}
    return dict(saved=saved, n_params=41_000_000, fixed_bytes=16 * 41_000_000)


def test_padding_matches_the_segmented_read():
    from g1_model.act_memory import READ_SEGMENT
    assert B.SEG == READ_SEGMENT == 64
    assert [B.padded(t) for t in (1, 64, 65, 600, 692, 3000)] == [64, 64, 128, 640, 704, 3008]


def test_timestep_counts_follow_the_segmented_read_and_the_scoring_order():
    # one sample per batch: lengths 1..64 pad to 64; 65..128 to 128
    assert B.train_timesteps(64, batch=1) == 64
    assert B.train_timesteps(128, batch=1) == pytest.approx(96)
    # more samples per batch: the longest dominates
    assert B.train_timesteps(600, batch=8) > B.train_timesteps(600, batch=1)
    assert B.train_timesteps(600, batch=8) == pytest.approx(565, abs=10)
    # scoring: consecutive batches of 32 in (episode, t) order
    assert B.score_timesteps(64) == 128                     # [1..32] -> 64, [33..64] -> 64
    assert B.score_timesteps(100) == 64 + 64 + 128 + 128    # ..., [97..100] -> 128
    assert B.score_timesteps(600) == 6400
    # O(T^2): five times the length, up to 25 times the reading (22.6: the
    # 64-tick padding weighs more on short histories)
    assert 22 < B.score_timesteps(3000) / B.score_timesteps(600) < 25


def test_projection_reproduces_the_laptop_at_600_and_scales_each_part():
    rows = _rows({600: 1.0, 3000: 2.0}, {600: 1.0, 3000: 25.0})
    p600, p3000 = B.project(rows, _mem(), val_episodes=40)
    # at the reference length the projection IS the laptop's measured window
    hours = B.GPU_WINDOW_S * 200 / 3600
    assert p600["gpu_run_hours_projected"] == pytest.approx(hours, abs=0.06)
    assert p600["gpu_run_hours_lower_bound"] == pytest.approx(hours, abs=0.06)
    g_step = B.GPU_BENCH_STEP_FULL / B.GPU_BENCH_HIGH
    g_lstm = g_step * (B.GPU_BENCH_STEP_FULL - B.GPU_BENCH_STEP_ORIG) / B.GPU_BENCH_STEP_FULL
    tau = g_lstm / B.train_timesteps(600)
    # expected: the LSTM part grows with its timesteps, the rest of a step does not
    assert p3000["gpu_step_s_projected"] == pytest.approx(
        g_step - g_lstm + tau * B.train_timesteps(3000), abs=1e-4)
    g_val_ep = (B.GPU_WINDOW_S - 1000 * g_step) / 40
    lstm600 = B.GPU_FWD_FRACTION * tau * B.score_timesteps(600, 40)
    want = (g_val_ep - lstm600) * 5 + B.GPU_FWD_FRACTION * tau * B.score_timesteps(3000, 40)
    assert p3000["gpu_score_s_per_ep_projected"] == pytest.approx(want, abs=1e-3)
    # lower: each part grows as it did on the CPU rows (LSTM step x2, score x(80/8))
    lo = (200_000 * (g_step - g_lstm + 2 * g_lstm) + 200 * 40 * g_val_ep * 80 / 8) / 3600
    assert p3000["gpu_run_hours_lower_bound"] == pytest.approx(lo, abs=0.06)
    assert p3000["gpu_run_hours_lower_bound"] < p3000["gpu_run_hours_projected"]
    # memory: the anchor plus the saved-activation growth from 704 to 3008 ticks
    grow = 1e5 * (3008 - 704) / 1024 ** 3
    assert p3000["gpu_peak_gib_projected"] == pytest.approx(B.GPU_PEAK_MIB / 1024 + grow, abs=0.01)
    assert p3000["cpu_run_hours"] == pytest.approx(
        (200_000 * (0.7 + 3.4) + 200 * 40 * (30 + 50)) / 3600, abs=0.06)
