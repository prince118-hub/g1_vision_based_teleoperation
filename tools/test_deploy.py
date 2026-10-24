"""Tests for tools/deploy.py - closed-loop evaluation.

    python tools/test_deploy.py

Self-contained since spec 1.2.0 (2026-10-18). The suite used to read
recordings/scripted_v1 and the runs trained on it; both are spec 1.1.0, which
this build refuses. It now records ONE scripted episode in memory (seed 0,
setting A, lock rule v2 - the configuration collection uses), saves it to a
temp folder, and builds small UNTRAINED checkpoints whose norm stats are fitted
on that episode. Nothing is trained and nothing is written outside a temp dir.
Checkpoint inference runs on the CPU (cuDNN's LSTM crashes Python at exit,
CLAUDE.md section 8 2026-09-29).

The sanity test is the one that matters: a "policy" that returns a recorded
episode's actions, pushed through deploy.py's loop, must reproduce
replay_check's trajectory BIT-FOR-BIT. That proves the loop routes an action
exactly as the replay test - and therefore collection - does, so a policy's
failure in deploy.py is the policy's, not the harness's.

TRAIN/DEPLOY PARITY (spec 1.2.0: `base_locked` and the gripper label)
---------------------------------------------------------------------
A whole replayed episode does NOT reproduce its recording, and cannot: the
demonstrator changes its walking command every 10 physics steps and the file
holds one per 20-step tick (replay_check's docstring; NOTES 2026-09-16), so the
walk-in drifts (seed 0: first lock at tick 43, recorded 40). What parity CAN
mean, and what is tested:
  1. the observation deploy builds IS the state the recorder stores - same
     builder, same instant - checked on every tick of a full episode;
  2. tick 0 of a natural start is bit-identical, all 48 dims;
  3. `base_locked` means the same thing on both sides: 1 exactly on the ticks
     the lock is held at the observation, rising on the tick the lock is taken;
  4. the gripper label has the same one-tick lead on both sides: replayed
     through deploy, the weld is observed on the SAME tick as in the recording
     (onset and release), and the 1.1.0 label - the sampled command - is
     observed one tick LATE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import deploy as D                                                  # noqa: E402
from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402

SEED = 0
_CACHE: dict = {}


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _recorded():
    """(path, per-tick observe()==recorded flags, loco): one scripted episode at
    the CURRENT spec, recorded with deploy's observer evaluated at every sampled
    tick. Cached per process."""
    if "rec" not in _CACHE:
        import record_episodes as RE
        from g1_data import recorder as REC
        cfg, demo, book, loco = RE._setup()
        same = []
        orig = REC.EpisodeBuffer.tick

        def tick(self, model, data, *a, **k):
            orig(self, model, data, *a, **k)
            same.append(bool(np.array_equal(D.observe(self.sp, model, data, self.ix),
                                            self.states[-1])))
        REC.EpisodeBuffer.tick = tick
        try:
            r, rec = RE.run_one(SEED, cfg, demo, book, loco)
        finally:
            REC.EpisodeBuffer.tick = orig
        path = os.path.join(tempfile.mkdtemp(), "ep_seed%04d.npz" % SEED)
        rec.buf.save(path, rec.metadata(rec.model, r, cfg, demo, SEED))
        _CACHE["rec"] = (path, same, loco)
    return _CACHE["rec"]


def _replay(actions=None, **kw):
    """The recorded episode's actions (or `actions`) through deploy's loop, from
    the natural start, under the episode's own lock rule."""
    from g1_data.phases import lock_config_from_meta
    from g1_data.recorder import load_episode
    path, _, loco = _recorded()
    rec, meta = load_episode(path)
    pol = D.ReplayPolicy(path)
    if actions is not None:
        pol.actions = np.asarray(actions, dtype=np.float64)
    return D.deploy_episode(SEED, pol, loco=loco, max_ticks=len(rec["actions"]),
                            early_success=False, raw_actions=True,
                            lock_rule=lock_config_from_meta(meta), **kw)


def _run(kind: str) -> str:
    """A run directory around an UNTRAINED checkpoint (seed-0 init), with norm
    stats fitted on the recorded episode and a split naming it as training."""
    key = "run_" + kind
    if key in _CACHE:
        return _CACHE[key]
    import torch
    path, _, _ = _recorded()
    tmp = tempfile.mkdtemp()
    ep = DS.Episode(path, DS.read_meta(path))
    st, ac, rep = DS.fit_norm_stats([ep], "train", allow_synthetic=True)
    norm = os.path.join(tmp, "norm.npz")
    DS.save_norm_stats(st, ac, path=norm, source=rep["source"], n_episodes=1,
                       seeds=rep["seeds"], split="train")
    splits = os.path.join(tmp, "splits.json")
    with open(splits, "w", encoding="utf-8") as fh:
        json.dump(dict(train=[SEED], val=[], spec_version=spec.SPEC_VERSION), fh)
    torch.manual_seed(0)
    if kind == "bc":
        from g1_model.models import build_bc
        model, K = build_bc(obs_window=12, chunk_size=1, hidden=(64, 64)), 1
    else:
        from g1_model.act import ACTConfig, build_act
        K = 8
        model = build_act(cfg=ACTConfig(obs_window=12, chunk_size=K, lr=1e-5, hidden_dim=64,
                                        dim_feedforward=128, nheads=4, enc_layers=2,
                                        dec_layers=1, use_lstm=True))
    run = os.path.join(tmp, "%s_seed0" % kind)
    os.makedirs(run)
    torch.save(dict(state_dict=model.state_dict(), model_cls=type(model).__name__,
                    model_kwargs=model.hparams, spec_version=spec.SPEC_VERSION,
                    config={}, step=0, epoch=0, val_loss=None, git_commit=None),
               os.path.join(run, "ckpt.pt"))
    with open(os.path.join(run, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(config=dict(notes=dict(
            model=kind, chunk_size=K,
            data_paths=dict(episodes=os.path.dirname(path), splits=splits, norm=norm)))), fh)
    _CACHE[key] = run
    return run


def _policy(kind, **kw):
    import torch
    return D.CheckpointPolicy(_run(kind), "ckpt.pt", device=torch.device("cpu"), **kw)


def _onsets(x):
    x = np.asarray(x) >= 0.5
    return (np.flatnonzero(~x[:-1] & x[1:]) + 1).tolist()


def _offsets(x):
    x = np.asarray(x) >= 0.5
    return (np.flatnonzero(x[:-1] & ~x[1:]) + 1).tolist()


# ─── post-processing ─────────────────────────────────────────────────────────
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


# ─── the loop is the replay loop ─────────────────────────────────────────────
def test_natural_start_matches_the_demonstrators_tick_zero():
    """reset_episode + base at (0.60, 0.00) is EXACTLY the recorded start, all
    48 dims (base_locked 0 on both sides)."""
    from g1_data.phases import lock_config_from_meta
    from g1_data.recorder import load_episode
    path, _, loco = _recorded()
    pol = D.ReplayPolicy(path)
    res, arrays = D.deploy_episode(SEED, pol, loco=loco, max_ticks=1,
                                   early_success=False, raw_actions=True,
                                   lock_rule=lock_config_from_meta(pol.meta))
    assert res["reset_fingerprint"] == pol.meta["reset_fingerprint"]
    rec, _ = load_episode(path)
    s0 = arrays["states"][0].astype(np.float32)
    assert s0.shape == (spec.STATE_DIM,) == (48,)
    assert np.array_equal(s0, rec["states"][0]), np.abs(s0 - rec["states"][0]).max()
    assert s0[spec.BASE_LOCKED] == 0.0


def test_replayed_actions_through_the_deploy_loop_match_replay_check():
    """Bit-for-bit, from the same start, with the same actions."""
    import replay_check as RC
    from g1_data.phases import lock_config_from_meta
    from g1_data.recorder import load_episode
    path, _, loco = _recorded()
    rec, meta = load_episode(path)
    pol = D.ReplayPolicy(path)
    T = len(rec["actions"])
    res, arrays = D.deploy_episode(
        SEED, pol, loco=loco, max_ticks=T, early_success=False, raw_actions=True,
        initial_state=(rec["qpos"][0], rec["qvel"][0]),
        lock_rule=lock_config_from_meta(meta))       # the episode's own rule: v3 (default)
    assert res["lock_rule"]["version"] == "v3"
    rep = RC.replay(path, verbose=False)
    div = np.abs(arrays["states"] - np.asarray(rec["states"], dtype=np.float64))
    assert res["ticks"] == T and not res["fell"]
    assert np.array_equal(div, rep["div"]), \
        "deploy loop diverges from replay_check: max %.3e" % np.abs(div - rep["div"]).max()
    assert res["weld_engage_tick"] == rep["weld_tick"], (res["weld_engage_tick"], rep["weld_tick"])
    assert res["lock_ticks"][0] == rep["lock_tick"], (res["lock_ticks"], rep["lock_tick"])


# ─── train/deploy parity (spec 1.2.0) ────────────────────────────────────────
def test_observation_is_the_recorded_state_at_every_tick():
    """Parity 1: `deploy.observe` on the recorder's MjData, at the recorder's
    sampling instant, equals the stored training state to the bit - all 48
    dims, every tick of a full episode."""
    from g1_data.recorder import load_episode
    path, same, _ = _recorded()
    rec, _ = load_episode(path)
    assert len(same) == len(rec["states"]) > 500, (len(same), len(rec["states"]))
    assert all(same), "observe() != recorded state on %d tick(s)" % same.count(False)


def test_base_locked_means_the_same_in_the_data_and_in_deploy():
    """Parity 3. In the data, dim 47 rises on the recorder's own lock tick and
    every locked tick logs zero velocity (D17 reads the same equality). In
    deploy, dim 47 is 1 on exactly the ticks the lock is held when the
    observation is built, rising on the tick deploy takes the lock."""
    from g1_data.recorder import load_episode
    path, _, _ = _recorded()
    rec, meta = load_episode(path)
    S, A = rec["states"], rec["actions"]
    bl = S[:, spec.BASE_LOCKED]
    assert set(np.unique(bl)) == {0.0, 1.0}
    assert _onsets(bl)[0] == meta["lock_engage_tick"] and _offsets(bl)[0] == meta["lock_release_tick"]
    assert np.all(A[bl == 1][:, spec.VEL_A] == 0.0)
    assert len(_onsets(bl)) == 2, "pickup lock and goal lock"
    res, out = _replay()
    T = len(out["states"])
    held = np.zeros(T, dtype=bool)
    for a, b in zip(res["lock_ticks"], res["release_ticks"] + [T]):
        held[a:b] = True
    dbl = out["states"][:, spec.BASE_LOCKED]
    assert np.array_equal(dbl == 1.0, held)
    assert _onsets(dbl) == res["lock_ticks"] and _offsets(dbl) == res["release_ticks"]
    assert np.all(out["actions"][dbl == 1.0][:, spec.VEL_A] == 0.0)


def test_grip_label_has_the_same_one_tick_lead_in_the_data_and_in_deploy():
    """Parity 4 (open item 29). The label leads the observed weld bit by one
    tick in the data; replayed through deploy the weld is OBSERVED on the same
    tick as in the recording, onset and release. The 1.1.0 label - the sampled
    command, kept as `grip_cmd` - is observed one tick late: the mismatch the
    relabel removes."""
    from g1_data.recorder import load_episode
    path, _, _ = _recorded()
    rec, meta = load_episode(path)
    S, A = rec["states"], rec["actions"]
    on, off = _onsets(S[:, spec.GRIP_L]), _offsets(S[:, spec.GRIP_L])
    assert len(on) == 1 and len(off) == 1, (on, off)
    assert _onsets(A[:, spec.GRIP_L_A]) == [on[0] - 1] and _offsets(A[:, spec.GRIP_L_A]) == [off[0] - 1]
    assert meta["grip_label"] == spec.GRIP_LABEL
    _, out = _replay()
    assert _onsets(out["states"][:, spec.GRIP_L]) == on
    assert _offsets(out["states"][:, spec.GRIP_L]) == off
    old = A.copy()
    old[:, spec.GRIP_L_A] = old[:, spec.GRIP_R_A] = rec["grip_cmd"]
    _, out_old = _replay(actions=old)
    assert _onsets(out_old["states"][:, spec.GRIP_L])[0] == on[0] + 1


# ─── determinism ─────────────────────────────────────────────────────────────
def _determinism(kind, max_ticks):
    from g1_model import train as T
    T.set_determinism(0, strict=True)
    pol = _policy(kind)
    _, _, loco = _recorded()
    out = []
    for _ in range(2):
        res, arrays = D.deploy_episode(SEED, pol, loco=loco, max_ticks=max_ticks)
        res.pop("wall_seconds"), res.pop("sim_to_wall")
        out.append((res, arrays))
    (r1, a1), (r2, a2) = out
    assert r1 == r2, {k: (r1[k], r2[k]) for k in r1 if r1[k] != r2[k]}
    for k in a1:
        assert np.array_equal(a1[k], a2[k]), k
    assert a1["states"].shape[1] == spec.STATE_DIM


def test_same_checkpoint_and_seed_twice_is_identical_bc():
    _determinism("bc", 300)


def test_same_checkpoint_and_seed_twice_is_identical_act_lstm():
    """Shortened: ACT-LSTM inference, ensembling every tick."""
    _determinism("act_lstm", 150)


def test_no_ensemble_applies_the_newest_chunks_first_action():
    """--no-ensemble (open item 32): query every tick, apply pred[0, 0]. With
    one chunk in the buffer (tick 0) the ensemble IS that action; after it the
    two must differ, or the flag does nothing."""
    import torch
    from g1_data.recorder import load_episode
    path, _, _ = _recorded()
    rec, _ = load_episode(path)
    off = _policy("act_lstm", ensemble=False)
    on = _policy("act_lstm")
    assert off.ensemble is False and off.ens is None and on.ensemble is True
    off.reset(); on.reset()
    window = np.zeros((off.W, spec.STATE_DIM), dtype=np.float32)
    differs = False
    for t in range(6):
        s = rec["states"][t]
        a_off, a_on = off.act(s, t), on.act(s, t)
        z = off.st.normalize(np.asarray(s, np.float32).astype(np.float64)).astype(np.float32)
        window = np.concatenate([window[1:], z[None]], axis=0)
        with torch.no_grad():
            first = off.model(torch.from_numpy(window).unsqueeze(0).to(off.device))[0, 0]
        want = off.ac.denormalize(first.cpu().numpy().astype(np.float64))
        assert np.array_equal(a_off, want), t
        if t == 0:
            assert np.allclose(a_on, a_off, atol=1e-6), "one chunk: ensemble == first action"
        differs |= not np.allclose(a_on, a_off, atol=1e-6)
    assert differs, "ensembling on and off gave identical actions for 6 ticks"


def _actions(pol, states, n=10):
    pol.reset()
    return [pol.act(states[t], t) for t in range(n)]


def test_ensemble_horizon_default_is_the_reference_ensemble():
    """No horizon and horizon = K give the same actions (the default is unchanged)."""
    from g1_data.recorder import load_episode
    rec, _ = load_episode(_recorded()[0])
    a = _actions(_policy("act_lstm"), rec["states"])
    pol = _policy("act_lstm", ensemble_horizon=8)
    assert pol.H == pol.K == 8
    assert all(np.array_equal(x, y) for x, y in zip(a, _actions(pol, rec["states"])))


def test_ensemble_horizon_one_is_no_ensemble():
    from g1_data.recorder import load_episode
    rec, _ = load_episode(_recorded()[0])
    a = _actions(_policy("act_lstm", ensemble=False), rec["states"])
    b = _actions(_policy("act_lstm", ensemble_horizon=1), rec["states"])
    assert all(np.allclose(x, y, atol=1e-6) for x, y in zip(a, b))


def test_ensemble_horizon_averages_only_the_last_n_chunks():
    """Horizon 3 at tick t: the weighted mean (oldest first, exp(-m i)) of the
    entries for t in the chunks predicted at t-2, t-1 and t - recomputed here
    from the raw model outputs - and it differs from the full ensemble."""
    import torch
    from g1_data.recorder import load_episode
    from g1_model.act import TEMPORAL_ENSEMBLE_M
    rec, _ = load_episode(_recorded()[0])
    pol = _policy("act_lstm", ensemble_horizon=3)
    pol.reset()
    window = np.zeros((pol.W, spec.STATE_DIM), dtype=np.float32)
    chunks = []
    for t in range(7):
        s = rec["states"][t]
        got = pol.act(s, t)
        z = pol.st.normalize(np.asarray(s, np.float32).astype(np.float64)).astype(np.float32)
        window = np.concatenate([window[1:], z[None]], axis=0)
        with torch.no_grad():
            chunks.append(pol.model(torch.from_numpy(window).unsqueeze(0).to(pol.device))[0].cpu().numpy())
        made = list(range(max(0, t - 2), t + 1))                     # oldest first
        rows = np.stack([chunks[k][t - k] for k in made]).astype(np.float64)
        w = np.exp(-TEMPORAL_ENSEMBLE_M * np.arange(len(made)))
        want = pol.ac.denormalize((rows * (w / w.sum())[:, None]).sum(axis=0))
        assert np.allclose(got, want, atol=1e-5), t
    full = _actions(_policy("act_lstm"), rec["states"], 7)
    assert not np.allclose(full[-1], got, atol=1e-6), "horizon 3 must differ from horizon 8 by tick 6"
    _raises(SystemExit, _policy, "act_lstm", ensemble_horizon=0)
    _raises(SystemExit, _policy, "act_lstm", ensemble_horizon=9)


def test_episode_metrics_on_a_hand_built_trajectory():
    """Open item 46. Palms approach the box until the weld at tick 3; after it
    they are on the box (ignored). The box passes 0.05 m from the goal at tick 5
    and ends 0.40 m from it."""
    from g1_data.success import geometry
    goal = np.asarray(geometry()["goal_xy"], float)
    T = 8
    S = np.zeros((T, spec.STATE_DIM))
    S[:, spec.BOX_POS] = [1.5, 0.0, 0.9]
    far = [0.50, 0.40, 0.30, 0.20, 0.0, 0.0, 0.0, 0.0]       # left palm, x offset
    near = [0.30, 0.25, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0]      # right palm
    for t in range(T):
        S[t, spec.PALM_L_POS] = S[t, spec.BOX_POS] + [far[t], 0, 0]
        S[t, spec.PALM_R_POS] = S[t, spec.BOX_POS] + [0, near[t], 0]
    S[5, spec.BOX_POS.start:spec.BOX_POS.start + 2] = goal + [0.05, 0.0]
    S[7, spec.BOX_POS.start:spec.BOX_POS.start + 2] = goal + [0.0, 0.40]
    m = D.episode_metrics(S, weld_tick=3)
    assert m["palm_box_min_m"] == 0.20 and m["palm_box_nearest_min_m"] == 0.10, m
    assert m["final_box_goal_m"] == 0.40 and m["closest_box_goal_m"] == 0.05, m
    assert m["closest_box_goal_tick"] == 5
    m = D.episode_metrics(S, weld_tick=1)                     # only ticks 0-1 count
    assert m["palm_box_min_m"] == 0.40 and m["palm_box_nearest_min_m"] == 0.25, m
    m = D.episode_metrics(S, weld_tick=-1)                    # never welded: all ticks
    assert m["palm_box_min_m"] == 0.0


def test_episode_metrics_in_a_deployed_episode():
    """In the deploy result: the final box-goal distance is the scored placement
    error, and an episode that welded had both palms near the box before it."""
    res, _ = _replay()
    assert abs(res["final_box_goal_m"] - res["placement_error_m"]) < 1e-4
    assert res["weld_engage_tick"] >= 0 and res["palm_box_min_m"] < 0.20, res["palm_box_min_m"]
    assert res["closest_box_goal_m"] <= res["final_box_goal_m"]


def test_scripted_walkin_is_the_demonstrators_walkin():
    """Item 44 diagnostic. Up to the first lock the scripted walk-in IS the
    demonstrator: states and logged actions bit-identical to the recording, and
    the lock fires on the recorded tick. (The policy here is a replay; its
    walk-in outputs are discarded, so only the controller is being tested.)"""
    from g1_data.phases import lock_config_from_meta
    from g1_data.recorder import load_episode
    path, _, loco = _recorded()
    rec, meta = load_episode(path)
    lk = meta["lock_engage_tick"]
    res, arr = D.deploy_episode(SEED, D.ReplayPolicy(path), loco=loco, max_ticks=lk + 2,
                                early_success=False, raw_actions=True,
                                lock_rule=lock_config_from_meta(meta), walkin="scripted")
    assert res["lock_ticks"][0] == lk and res["scripted_walkin_ticks"] == lk
    assert np.array_equal(arr["states"][:lk + 1].astype(np.float32), rec["states"][:lk + 1])
    assert np.array_equal(arr["actions"][:lk].astype(np.float32), rec["actions"][:lk])
    assert res["walkin"] == "scripted"
    _raises(ValueError, D.deploy_episode, SEED, D.ReplayPolicy(path), loco=loco, max_ticks=1,
            walkin="teleport")


# ─── seeds (open item 38) ────────────────────────────────────────────────────
def test_selection_uses_its_own_seed_file():
    a = argparse.Namespace(seed_list=None, seeds=DS.SELECT_SEEDS)
    sel = D._seeds(a)
    assert sel == DS.load_seed_file(DS.SELECT_SEEDS) and len(sel) == 20
    others = set()
    for k in ("train", "exp1", "exp2"):
        others |= set(DS.load_seed_file(os.path.join(DS.EVAL, "%s_seeds.json" % k)))
    assert not set(sel) & others


def test_seeds_the_run_trained_on_are_refused_before_anything_loads():
    run = _run("bc")
    p = os.path.join(tempfile.mkdtemp(), "x.jsonl")
    e = _raises(SystemExit, D.main, ["--run", run, "--seed-list", "%d,700" % SEED, "--out", p])
    assert "train/val split" in str(e) and not os.path.exists(p)


def test_out_file_is_never_appended_to():
    p = os.path.join(tempfile.mkdtemp(), "x.jsonl")
    open(p, "w").close()
    _raises(SystemExit, D.main, ["--run", "unused", "--seed-list", "679", "--out", p])


# ─── review 2026-10-24 ───────────────────────────────────────────────────────
class _StandStill:
    """Zero action: the arms at zero, no walking - the robot never leaves the
    start, so it never locks and never welds: a walk-in failure by construction."""
    name, kind = "stand_still", "test"

    def reset(self):
        pass

    def act(self, state, t):
        return np.zeros(spec.ACTION_DIM)


def test_walk_in_failure_is_charged_to_walk_in_not_grasp():
    """M1. A never-grasped episode that ends in SETTLE is a WALK_IN failure
    (success.attribute). deploy scored the episode BEFORE labelling it, so the
    labels never reached attribute() and every such episode read GRASP."""
    from g1_data import success as SU
    _, _, loco = _recorded()
    res, arrays = D.deploy_episode(SEED, _StandStill(), loco=loco, max_ticks=60)
    assert res["weld_engage_tick"] < 0 and not res["lock_ticks"], res["lock_ticks"]
    assert res["phase_reached"] == "SETTLE", res["phase_reached"]
    assert res["fail_reason"] == "WALK_IN", res["fail_reason"]
    assert res["grasp_detail"]["reason"] == "the weld never engaged"
    # and the test discriminates: scored WITHOUT the labels, as before the fix
    old = SU.evaluate(arrays, dict(box_spawn_xy=res["box_xy"]))
    assert old["attributed_to"] == "GRASP"


def _run_variant(kind, **meta_edit):
    """A copy of _run(kind) whose metadata.json is edited."""
    import shutil
    src = _run(kind)
    dst = os.path.join(tempfile.mkdtemp(), os.path.basename(src))
    shutil.copytree(src, dst)
    p = os.path.join(dst, "metadata.json")
    m = json.load(open(p, encoding="utf-8"))
    for k, v in meta_edit.items():
        if k in ("splits", "episodes", "norm"):
            m["config"]["notes"]["data_paths"][k] = v
        else:
            m[k] = v
    json.dump(m, open(p, "w", encoding="utf-8"))
    return dst


def test_norm_stats_from_another_split_are_refused():
    """The normalizer must be the one fitted on the run's training seeds."""
    import torch
    run = _run_variant("bc", data=dict(seeds=[SEED + 1]))
    e = _raises(DS.DatasetError, D.CheckpointPolicy, run, "ckpt.pt",
                device=torch.device("cpu"))
    assert "seed" in str(e).lower(), e
    D.CheckpointPolicy(_run("bc"), "ckpt.pt", device=torch.device("cpu"))   # matching: loads


def test_seed_guard_fails_closed_when_the_split_is_unknown():
    """It used to return no seeds - and so wave every seed through - when the
    splits file was missing."""
    run = _run_variant("bc", splits="no/such/splits.json")
    p = os.path.join(tempfile.mkdtemp(), "x.jsonl")
    e = _raises(SystemExit, D.main, ["--run", run, "--seed-list", "700", "--out", p])
    assert "cannot determine" in str(e) and not os.path.exists(p), e


def test_lock_rule_guard_fails_closed():
    """A rule that cannot be read refuses; a mismatch refuses unless stated -
    both before anything is loaded."""
    from g1_data.phases import LOCK_RULES, lock_config_from_meta
    path, _, _ = _recorded()
    have = lock_config_from_meta(DS.read_meta(path)).version
    other = next(v for v in sorted(LOCK_RULES) if v != have)
    p = os.path.join(tempfile.mkdtemp(), "x.jsonl")
    e = _raises(SystemExit, D.main, ["--run", _run("bc"), "--seed-list", "700",
                                     "--lock-rule", other, "--out", p])
    assert "refusing: evaluating under lock rule" in str(e), e
    gone = _run_variant("bc", episodes=os.path.join(tempfile.mkdtemp(), "nothing_here"))
    e = _raises(SystemExit, D.main, ["--run", gone, "--seed-list", "700",
                                     "--lock-rule", have, "--out", p])
    assert "not found" in str(e), e
    empty = tempfile.mkdtemp()
    e = _raises(SystemExit, D.main, ["--run", _run_variant("bc", episodes=empty),
                                     "--seed-list", "700", "--lock-rule", have, "--out", p])
    assert "missing" in str(e), e
    assert not os.path.exists(p)
    m = json.load(open(os.path.join(_run("bc"), "metadata.json"), encoding="utf-8"))
    assert D.training_lock_rule(m).version == have


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
