"""Walking-velocity label (2026-11-19, a TEST): the smoothing rule, its versioning, the guards.

    python -m pytest g1_data/test_velocity_label.py

`spec.velocity_label_smoothed` is the one rule; `tools/convert_velocity_label.py`
writes relabelled COPIES; a file without `velocity_label` carries the recorded
command. The dataset tools and the loader refuse a mix of labels and a
normalizer fitted on the other label. Tests that need real human recordings
are skipped without recordings/demonstrations.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from g1_data import spec, dataset as DS                              # noqa: E402
import convert_velocity_label as CV                                  # noqa: E402

HUMAN = sorted(glob.glob(os.path.join(ROOT, "recordings", "demonstrations", "ep_seed*.npz")))
needs_human = pytest.mark.skipif(len(HUMAN) < 2, reason="no recordings/demonstrations")


def _episode(T=60, locked=(40, 60)):
    S = np.zeros((T, spec.STATE_DIM), dtype=np.float32)
    S[locked[0]:locked[1], spec.BASE_LOCKED] = 1.0
    rng = np.random.default_rng(0)
    A = rng.normal(size=(T, spec.ACTION_DIM)).astype(np.float32)
    A[:, spec.VEL_A] = 0.0
    A[5:11, 19] = 0.6                     # one 6-tick latch pulse
    A[20:26, 21] = -0.6
    return S, A


# ─── the rule ────────────────────────────────────────────────────────────────
def test_only_walking_dims_change_and_dtype_is_kept():
    S, A = _episode()
    B = spec.velocity_label_smoothed(S, A)
    keep = [i for i in range(spec.ACTION_DIM) if i not in (19, 20, 21)]
    assert np.array_equal(B[:, keep], A[:, keep]) and B.dtype == A.dtype
    assert not np.array_equal(B[:, 19], A[:, 19])


def test_centred_average_values_and_zero_phase():
    S, A = _episode(locked=(59, 60))
    A[:, 19] = 0.0
    A[15:21, 19] = 0.6                    # a 6-tick pulse well inside the episode
    B = spec.velocity_label_smoothed(S, A, window=13).astype(np.float64)
    # a 6-tick pulse of 0.6 under a 13-tick window: peak 6 * 0.6 / 13, symmetric about the pulse centre
    assert B[:, 19].max() == pytest.approx(6 * 0.6 / 13, abs=1e-6)
    nz = np.flatnonzero(B[:, 19])
    assert (nz.min() + nz.max()) / 2 == pytest.approx(15 + 2.5)
    assert nz.min() == 15 - 6 and nz.max() == 20 + 6         # spreads h = 6 ticks each way, no more
    assert np.allclose(B[nz.min():nz.max() + 1, 19], B[nz.min():nz.max() + 1, 19][::-1])   # zero phase
    assert B[:, 19].sum() == pytest.approx(A[:, 19].astype(np.float64).sum(), rel=1e-6)   # displacement kept


def test_locked_ticks_are_exactly_zero_and_window_one_is_the_command():
    S, A = _episode(locked=(8, 60))
    A[8:, spec.VEL_A] = 0.0
    B = spec.velocity_label_smoothed(S, A)
    assert (B[8:, spec.VEL_A] == 0).all()
    assert np.array_equal(spec.velocity_label_smoothed(S, A, window=1), A)


def test_refuses_bad_input():
    S, A = _episode()
    A2 = A.copy()
    A2[50, 19] = 0.3                      # nonzero on a locked tick: D17 violated
    with pytest.raises(ValueError, match="locked"):
        spec.velocity_label_smoothed(S, A2)
    for w in (0, 2, 12):
        with pytest.raises(ValueError, match="odd"):
            spec.velocity_label_smoothed(S, A, window=w)
    with pytest.raises(ValueError):
        spec.velocity_label_smoothed(S[:-1], A)


def test_label_registry():
    assert spec.velocity_label_of({}) == spec.VELOCITY_LABEL_COMMAND == "command_v1"
    assert spec.velocity_label_of({"velocity_label": spec.VELOCITY_LABEL_SMOOTH}) == "centred_ma13_v1"
    assert spec.VELOCITY_LABELS[spec.VELOCITY_LABEL_SMOOTH]["window"] == spec.VELOCITY_LABEL_WINDOW == 13
    with pytest.raises(ValueError, match="unknown velocity_label"):
        spec.velocity_label_of({"velocity_label": "nope"})


# ─── the guards ──────────────────────────────────────────────────────────────
def _eps(*labels):
    out = []
    for i, lab in enumerate(labels):
        m = {"seed": i}
        if lab is not None:
            m["velocity_label"] = lab
        out.append(DS.Episode("ep_seed%04d.npz" % i, m))
    return out


def test_uniform_label_groups_absent_with_command_and_refuses_a_mix():
    C, M = spec.VELOCITY_LABEL_COMMAND, spec.VELOCITY_LABEL_SMOOTH
    assert DS.assert_uniform_velocity_label(_eps(None, C, None)) == C
    assert DS.assert_uniform_velocity_label(_eps(M, M)) == M
    with pytest.raises(DS.DatasetError, match="mixes walking-velocity labels"):
        DS.assert_uniform_velocity_label(_eps(None, M))


def test_norm_label_must_match_the_episodes():
    C, M = spec.VELOCITY_LABEL_COMMAND, spec.VELOCITY_LABEL_SMOOTH
    DS.assert_norm_velocity_label({}, C)                         # an old stats file = the command
    DS.assert_norm_velocity_label({"velocity_label": M}, M)
    with pytest.raises(DS.DatasetError, match="velocity label"):
        DS.assert_norm_velocity_label({}, M)
    with pytest.raises(DS.DatasetError, match="velocity label"):
        DS.assert_norm_velocity_label({"velocity_label": M}, C)


def test_norm_stats_store_the_label_only_when_given(tmp_path):
    st = spec.NormStats(kind="state", mean=np.zeros(spec.STATE_DIM), std=np.ones(spec.STATE_DIM), split="train")
    ac = spec.NormStats(kind="action", mean=np.zeros(spec.ACTION_DIM), std=np.ones(spec.ACTION_DIM), split="train")
    p_old = DS.save_norm_stats(st, ac, path=str(tmp_path / "a.npz"), seeds=[1, 2], split="train")
    p_new = DS.save_norm_stats(st, ac, path=str(tmp_path / "b.npz"), seeds=[1, 2], split="train",
                               velocity_label=spec.VELOCITY_LABEL_SMOOTH)
    assert "velocity_label" not in DS.load_norm_stats(p_old)[2]
    assert DS.load_norm_stats(p_new)[2]["velocity_label"] == spec.VELOCITY_LABEL_SMOOTH
    with pytest.raises(DS.DatasetError):
        DS.save_norm_stats(st, ac, path=str(tmp_path / "c.npz"), seeds=[1], split="train", velocity_label="x")


# ─── the converter ───────────────────────────────────────────────────────────
def test_convert_arrays_changes_only_dims_19_21_and_refuses_twice():
    S, A = _episode()
    T = len(S)
    arrays = dict(states=S, actions=A, grip_cmd=np.zeros(T, np.float32), phase_labels=np.zeros(T, np.int8),
                  gait_phase=np.zeros((T, 2), np.float32), qpos=np.zeros((T, 3)), qvel=np.zeros((T, 3)),
                  step_index=np.arange(T))
    meta = {"spec_version": spec.SPEC_VERSION, "n_ticks": T}
    new, m = CV.convert_arrays(arrays, meta, "x")
    assert m["velocity_label"] == spec.VELOCITY_LABEL_SMOOTH and "velocity_label" not in meta
    assert np.array_equal(new["actions"], spec.velocity_label_smoothed(S, A))
    for k in arrays:
        if k != "actions":
            assert new[k] is arrays[k], k
    with pytest.raises(CV.ConvertError, match="already carries"):
        CV.convert_arrays(new, m, "x")
    with pytest.raises(CV.ConvertError, match="spec_version"):
        CV.convert_arrays(arrays, dict(meta, spec_version="g1-spec-1.1.0"), "x")


@needs_human
def test_converter_writes_a_copy_and_leaves_the_original(tmp_path):
    src = HUMAN[0]
    before = open(src, "rb").read()
    out = tmp_path / "conv"
    assert CV.main([src, "--out-dir", str(out)]) == 0
    assert open(src, "rb").read() == before
    dst = out / os.path.basename(src)
    with np.load(src) as a, np.load(str(dst)) as b:
        ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
        assert set(a.files) == set(b.files)
        for k in a.files:
            if k not in ("meta", "actions"):
                assert np.array_equal(a[k], b[k]), k
        keep = [i for i in range(spec.ACTION_DIM) if i not in (19, 20, 21)]
        assert np.array_equal(a["actions"][:, keep], b["actions"][:, keep])
        assert np.array_equal(b["actions"], spec.velocity_label_smoothed(a["states"], a["actions"]))
    assert mb["velocity_label"] == spec.VELOCITY_LABEL_SMOOTH
    assert {k: v for k, v in mb.items() if k not in ("velocity_label", "velocity_label_conversion")} == ma
    assert len(mb["velocity_label_conversion"]["source_sha256"]) == 64
    with pytest.raises(SystemExit):
        CV.main([src, "--out-dir", str(out)])                   # refuses to overwrite
    with pytest.raises(SystemExit):
        CV.main([src, "--out-dir", os.path.dirname(src)])      # refuses a source folder


@needs_human
def test_loader_refuses_a_mix_and_a_mismatched_normalizer(tmp_path):
    from g1_model.loader import ChunkDataset, LoaderConfig, TrackingPolicy
    raw = tmp_path / "raw"
    raw.mkdir()
    for p in HUMAN[:2]:
        shutil.copy(p, raw)
    conv = tmp_path / "conv"
    CV.main([str(raw), "--out-dir", str(conv)])
    eps_raw, eps_conv = DS.scan(str(raw)), DS.scan(str(conv))
    cfg = LoaderConfig(chunk_size=1, obs_window=1, tracking=TrackingPolicy())
    st, ac, _ = DS.fit_norm_stats(eps_conv, "train")
    seeds = [e.seed for e in eps_conv]
    ok = ChunkDataset(eps_conv, cfg, st, ac, {"seeds": seeds, "velocity_label": spec.VELOCITY_LABEL_SMOOTH})
    assert len(ok) > 0
    ChunkDataset(eps_raw, cfg, st, ac, {"seeds": seeds})          # absent on both sides = the command
    with pytest.raises(DS.DatasetError, match="velocity label"):
        ChunkDataset(eps_conv, cfg, st, ac, {"seeds": seeds})     # stats fitted on the command
    with pytest.raises(DS.DatasetError, match="velocity label"):
        ChunkDataset(eps_raw, cfg, st, ac, {"seeds": seeds, "velocity_label": spec.VELOCITY_LABEL_SMOOTH})
    mixed = [eps_raw[0], eps_conv[1]]
    with pytest.raises(DS.DatasetError, match="mixes walking-velocity labels"):
        ChunkDataset(mixed, cfg, st, ac, None)
