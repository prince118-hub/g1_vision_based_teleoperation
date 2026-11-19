"""Tests for the observation feature sets (spec.STATE_FEATURES, "fix #1", 2026-11-20).

    python -m pytest tools/test_state_features.py -q

What is checked:
  * DEFINITION: robot_frame_features IS the lock predicate's fwd / lat / head
    (phases.LockPredicate.terms) for the box before the weld and the goal after
    it, on recorded states; yaw is walk_test's; GOAL_XY is the compiled scene's.
  * PROPERTIES: invariant to a rigid motion of the whole world (box, base, goal);
    the goal bearing stays continuous when the goal passes behind the robot.
  * RAW IS UNCHANGED: observation(raw) returns its input object; NormStats,
    save/load and BC metadata carry no new field for raw.
  * GUARDS: mixed feature sets, a normalizer of another set, a model of another
    width - each refused, at the dataset, the loader and deploy.
  * TRAIN/DEPLOY PARITY on recorded demos: for EVERY tick of every human_v3
    episode, the normalized observation `deploy.CheckpointPolicy.act` feeds the
    model equals, bit for bit, the loader's training observation for that tick,
    and the whole W_o window equals the loader's sample. (That deploy's 48-D
    state is the recorder's is tools/test_deploy.py's
    test_observation_is_the_recorded_state_at_every_tick.)
Skips the recorded-demo tests when data/processed/episodes_human_v3 is absent
(it is gitignored; tools/convert_state_features.py builds it).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402

V3 = os.path.join(ROOT, "data", "processed", "episodes_human_v3")
V3_SPLITS = os.path.join(ROOT, "data", "processed", "splits_human_v3.json")
V3_NORM = os.path.join(ROOT, "data", "processed", "norm_stats_human_v3.npz")
V2 = os.path.join(ROOT, "data", "processed", "episodes_human_v2")
HAVE_V3 = os.path.isdir(V3) and os.path.isfile(V3_NORM) and os.path.isfile(V3_SPLITS)
needs_v3 = pytest.mark.skipif(not HAVE_V3, reason="human_v3 not built (tools/convert_state_features.py)")
RF = spec.STATE_FEATURES_ROBOT_FRAME
RAW = spec.STATE_FEATURES_RAW


def _state(box_xy, base_xy, yaw, weld=0.0):
    s = np.zeros(spec.STATE_DIM)
    s[spec.BOX_POS] = [box_xy[0], box_xy[1], 0.8]
    s[spec.BOX_QUAT] = [1, 0, 0, 0]
    s[spec.BASE_POS] = [base_xy[0], base_xy[1], 0.78]
    s[spec.BASE_QUAT] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    s[spec.GRIP_L] = s[spec.GRIP_R] = weld
    return s


def _model():
    from g1_teleop.config import TeleopConfig
    from g1_teleop.contact_contract import load_model
    return load_model(TeleopConfig())


# ─── definition ──────────────────────────────────────────────────────────────
def test_goal_constant_is_the_scenes_goal():
    from g1_data.phases import PlatformGeometry
    from g1_data.success import geometry
    m = _model()
    assert np.array_equal(PlatformGeometry.resolve(m).goal_xy, np.asarray(spec.GOAL_XY))
    assert np.array_equal(geometry()["goal_xy"], np.asarray(spec.GOAL_XY))


def test_yaw_is_walk_tests_yaw():
    import walk_test as W
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        s = np.zeros(spec.STATE_DIM)
        s[spec.BASE_QUAT] = q
        assert spec.heading_yaw(s) == pytest.approx(float(W.yaw_from_quat(q)), abs=1e-15)


def _lock_terms_check(states):
    """features vs LockPredicate.terms on each state: box terms with the weld
    bit off, goal terms with it on (terms switches its target on the weld)."""
    from g1_data.phases import LOCK_RULE_V3, LockPredicate, PlatformGeometry
    pred = LockPredicate(PlatformGeometry.resolve(_model()), LOCK_RULE_V3)
    F = spec.robot_frame_features(states)
    for s, f in zip(states, F):
        for k, weld in ((0, 0.0), (1, 1.0)):
            s2 = np.array(s, dtype=np.float64)
            s2[spec.GRIP_L] = weld
            t = pred.terms(s2)
            x, y, sn, cs = f[4 * k:4 * k + 4]
            assert x == pytest.approx(t["fwd"], abs=1e-12)
            assert y == pytest.approx(t["lat"], abs=1e-12)
            assert np.degrees(np.arctan2(sn, cs)) == pytest.approx(t["head"], abs=1e-9)
            assert sn * sn + cs * cs == pytest.approx(1.0, abs=1e-12)


def test_features_are_the_lock_predicates_terms_synthetic():
    rng = np.random.default_rng(1)
    S = np.stack([_state(rng.uniform(1.2, 1.8, 2), rng.uniform(0.2, 1.6, 2), rng.uniform(-np.pi, np.pi))
                  for _ in range(40)])
    _lock_terms_check(S)


@needs_v3
def test_features_are_the_lock_predicates_terms_on_recorded_demos():
    eps = DS.scan(V3)
    for e in eps[:5]:
        a, _ = e.load()
        S = np.asarray(a["states"], dtype=np.float64)
        _lock_terms_check(S[::25])


def test_rigid_motion_of_the_world_leaves_the_features_unchanged():
    rng = np.random.default_rng(2)
    for _ in range(20):
        box, base, yaw = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)
        goal = rng.uniform(-2, 2, 2)
        th, t = rng.uniform(-np.pi, np.pi), rng.uniform(-3, 3, 2)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        f0 = spec.robot_frame_features(_state(box, base, yaw), goal_xy=goal)
        f1 = spec.robot_frame_features(_state(R @ box + t, R @ base + t, yaw + th),
                                       goal_xy=R @ goal + t)
        assert np.allclose(f0, f1, atol=1e-12)


def test_goal_bearing_is_continuous_when_the_goal_passes_behind():
    yaws = np.linspace(-np.pi, np.pi, 2001)
    S = np.stack([_state((1.5, 0.1), (0.9, 0.0), y) for y in yaws])
    F = spec.robot_frame_features(S)
    step = np.abs(np.diff(F[:, 6:8], axis=0)).max()
    assert step < 0.01, step                    # (sin, cos) never jumps
    ang = np.degrees(np.arctan2(F[:, 6], F[:, 7]))
    assert np.abs(np.diff(ang)).max() > 300     # where an angle WOULD have wrapped


def test_zero_distance_bearing_is_defined():
    f = spec.robot_frame_features(_state((1.0, 0.0), (1.0, 0.0), 0.3))
    assert list(f[:4]) == [0.0, 0.0, 0.0, 1.0]


def test_observation_layout():
    s = _state((1.4, 0.1), (0.6, 0.0), 0.2).astype(np.float32)
    assert spec.observation(s, RAW) is s
    o = spec.observation(s.astype(np.float64), RF)
    assert o.shape == (56,) and o.dtype == np.float64
    assert np.array_equal(o[:48], s.astype(np.float64))
    assert np.array_equal(o[spec.TASK_FEATURES], spec.robot_frame_features(s.astype(np.float64)))
    assert spec.state_dim_of(RAW) == 48 and spec.state_dim_of(RF) == 56
    assert spec.state_names_of(RF)[48:] == spec.TASK_FEATURE_NAMES
    assert spec.state_mask_of(RAW) is spec.STATE_MASK and spec.state_mask_of(RF).all()
    with pytest.raises(ValueError):
        spec.observation(s, "nope")
    with pytest.raises(ValueError):
        spec.state_features_of({"state_features": "nope"})
    assert spec.state_features_of({}) == RAW


# ─── raw is unchanged ────────────────────────────────────────────────────────
def test_raw_norm_file_has_no_new_field_and_round_trips(tmp_path):
    st = spec.NormStats.identity("state")
    ac = spec.NormStats.identity("action")
    p = DS.save_norm_stats(st, ac, path=str(tmp_path / "n.npz"), seeds=[1], split="train")
    with np.load(p) as z:
        meta = json.loads(str(z["meta"]))
    assert "state_features" not in meta
    st2, _, _ = DS.load_norm_stats(p)
    assert st2.state_features == RAW and st2.mean.shape == (48,)


def test_robot_frame_norm_round_trips_and_widths_are_enforced(tmp_path):
    st = spec.NormStats.identity("state", state_features=RF)
    assert st.mean.shape == (56,)
    ac = spec.NormStats.identity("action")
    p = DS.save_norm_stats(st, ac, path=str(tmp_path / "n.npz"), seeds=[1], split="train")
    st2, _, meta = DS.load_norm_stats(p)
    assert meta["state_features"] == RF and st2.state_features == RF
    with pytest.raises(ValueError):
        spec.NormStats(kind="state", mean=np.zeros(48), std=np.ones(48), state_features=RF)
    with pytest.raises(ValueError):
        spec.NormStats(kind="action", mean=np.zeros(22), std=np.ones(22), state_features=RF)
    with pytest.raises(DS.DatasetError):
        DS.save_norm_stats(st, ac, path=str(tmp_path / "m.npz"), seeds=[1], state_features=RAW)


def test_bc_config_and_hparams_omit_state_dim_at_48():
    from g1_model.models import BCConfig, build_bc
    c = BCConfig(obs_window=12, chunk_size=1)
    assert "state_dim" not in c.as_metadata()
    assert "state_dim" not in build_bc(cfg=c).hparams
    c56 = BCConfig(obs_window=12, chunk_size=1, state_dim=56)
    assert c56.as_metadata()["state_dim"] == 56
    m = build_bc(cfg=c56)
    assert m.hparams["state_dim"] == 56
    assert build_bc(**m.hparams).net[0].in_features == 12 * 56
    import torch
    with pytest.raises(ValueError):
        m(torch.zeros(1, 12, 48))


def test_act_model_takes_its_width_from_its_config():
    import torch
    from g1_model.act import ACTConfig, build_act
    cfg = ACTConfig(obs_window=1, chunk_size=4, lr=1e-5, hidden_dim=32, dim_feedforward=64,
                    nheads=4, enc_layers=1, dec_layers=1, state_dim=56, use_lstm=True)
    m = build_act(cfg=cfg).eval()
    assert tuple(m(torch.zeros(2, 1, 56)).shape) == (2, 4, 22)
    with pytest.raises(ValueError):
        m(torch.zeros(2, 1, 48))


# ─── guards ──────────────────────────────────────────────────────────────────
@needs_v3
def test_dataset_refuses_a_mix_of_feature_sets():
    eps = DS.scan(V3)[:2] + DS.scan(V2)[2:4]
    with pytest.raises(DS.DatasetError, match="mixes observation feature sets"):
        DS.assert_uniform_state_features(eps)
    assert DS.assert_uniform_state_features(DS.scan(V3)) == RF
    assert DS.assert_uniform_state_features(DS.scan(V2)) == RAW


@needs_v3
def test_loader_refuses_a_normalizer_of_another_feature_set():
    from g1_model.loader import ChunkDataset, LoaderConfig, LoaderError, TrackingPolicy
    sp = DS.load_splits(V3_SPLITS)
    cfg = LoaderConfig(chunk_size=1, obs_window=1, tracking=TrackingPolicy())
    st2, ac2, m2 = DS.load_norm_stats(os.path.join(ROOT, "data/processed/norm_stats_human_v2.npz"))
    st3, ac3, m3 = DS.load_norm_stats(V3_NORM)
    with pytest.raises(LoaderError, match="state_features"):            # v2 stats on v3 episodes
        ChunkDataset.from_directory(V3, cfg, st2, ac2, m2, seeds=sp["train"])
    with pytest.raises(LoaderError, match="state_features"):            # v3 stats on v2 episodes
        ChunkDataset.from_directory(V2, cfg, st3, ac3, m3, seeds=sp["train"])
    with pytest.raises(DS.DatasetError, match="state_features"):        # meta says otherwise
        DS.assert_norm_state_features(m2, RF)


# ─── train/deploy parity on recorded demos ───────────────────────────────────
def _v3_run(kind, W, tmp):
    """A run directory around an UNTRAINED checkpoint on the human_v3 norm."""
    import torch
    sp = DS.load_splits(V3_SPLITS)
    torch.manual_seed(0)
    if kind == "bc":
        from g1_model.models import BCConfig, build_bc
        model, K = build_bc(cfg=BCConfig(obs_window=W, chunk_size=1, hidden=(32,), state_dim=56)), 1
    else:
        from g1_model.act import ACTConfig, build_act
        K = 4
        model = build_act(cfg=ACTConfig(obs_window=W, chunk_size=K, lr=1e-5, hidden_dim=32,
                                        dim_feedforward=64, nheads=4, enc_layers=1, dec_layers=1,
                                        state_dim=56))
    run = os.path.join(tmp, "%s_W%d" % (kind, W))
    os.makedirs(run)
    torch.save(dict(state_dict=model.state_dict(), model_cls=type(model).__name__,
                    model_kwargs=model.hparams, spec_version=spec.SPEC_VERSION,
                    config={}, step=0, epoch=0, val_loss=None, git_commit=None),
               os.path.join(run, "ckpt.pt"))
    with open(os.path.join(run, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(config=dict(notes=dict(
            model=kind, chunk_size=K, state_features=RF,
            data_paths=dict(episodes=V3, splits=V3_SPLITS, norm=V3_NORM)))), fh)
    return run, sp


@needs_v3
@pytest.mark.parametrize("kind,W", [("bc", 12), ("act", 1)])
def test_train_deploy_feature_parity_on_every_recorded_tick(kind, W, tmp_path):
    import torch
    import deploy as D
    from g1_model.loader import ChunkDataset, LoaderConfig, TrackingPolicy
    run, sp = _v3_run(kind, W, str(tmp_path))
    pol = D.CheckpointPolicy(run, "ckpt.pt", device=torch.device("cpu"))
    assert pol.features == RF and pol.state_dim == 56
    pol.check_goal(_model())
    st, ac, meta = DS.load_norm_stats(V3_NORM)
    ds = ChunkDataset.from_directory(
        V3, LoaderConfig(chunk_size=pol.K, obs_window=W, tracking=TrackingPolicy()),
        st, ac, meta, norm_fit_seeds=sp["train"])
    assert ds.state_features == RF and ds.state_dim == 56 and len(ds.seeds) == 50
    starts = np.cumsum([0] + ds.lengths[:-1])
    n_ticks = 0
    for ei, path in enumerate(ds.paths):
        a, _ = DS.Episode(path, DS.read_meta(path)).load()
        S = np.asarray(a["states"])                         # float32, as recorded
        pol.reset()
        for t in range(len(S)):
            pol.act(S[t], t)
            assert np.array_equal(pol.window[-1], ds.states[ei][t]), (path, t)
            if t % 97 == 0:                                 # the whole window = the sample
                item = ds[int(starts[ei]) + t]
                assert int(item["start_tick"]) == t
                assert np.array_equal(pol.window, item["obs"].numpy()), (path, t)
        n_ticks += len(S)
    assert n_ticks == sum(ds.lengths)


@needs_v3
def test_deploy_refuses_a_width_or_feature_mismatch(tmp_path):
    import torch
    import deploy as D
    run, _ = _v3_run("bc", 12, str(tmp_path))
    # a 48-wide checkpoint on the 56-D normalizer
    from g1_model.models import build_bc
    m48 = build_bc(obs_window=12, chunk_size=1, hidden=(32,))
    torch.save(dict(state_dict=m48.state_dict(), model_cls="BCPolicy", model_kwargs=m48.hparams,
                    spec_version=spec.SPEC_VERSION, config={}, step=0, epoch=0, val_loss=None,
                    git_commit=None), os.path.join(run, "w48.pt"))
    with pytest.raises(SystemExit, match="48-D observations"):
        D.CheckpointPolicy(run, "w48.pt", device=torch.device("cpu"))
    # run metadata that says raw while the normalizer is robot-frame
    p = os.path.join(run, "metadata.json")
    m = json.load(open(p, encoding="utf-8"))
    del m["config"]["notes"]["state_features"]
    json.dump(m, open(p, "w", encoding="utf-8"))
    with pytest.raises(SystemExit, match="trained on state_features"):
        D.CheckpointPolicy(run, "ckpt.pt", device=torch.device("cpu"))
