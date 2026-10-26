"""Tests for the memory-design pair: act_orig and act_lstm_full (g1_model/act_memory.py).

    python -m g1_model.test_act_memory

What they hold the two models to:
  * act_orig IS ACT at W_o = 1 - same config, same weights, same function.
  * act_lstm_full is act_orig plus the LSTM token, and nothing else: every
    act_orig parameter bit-identical from the same seed, the configs differ in
    `use_lstm` alone, and on a one-tick history it IS ACT-LSTM at W_o = 1.
  * the LSTM reads EVERY step of the history 0..t, in order, and never the
    padding; a sample's output does not depend on what it is batched with.
  * deployment (`step`, carried (h, c)) equals re-reading the whole history.
  * it trains, checkpoints, resumes bit-exactly and is scored through the
    UNCHANGED shared loop and guard.
Tests that train need a staged episode in data/synthetic (test_train._tiny) and
say so when they skip.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from g1_data import spec                                            # noqa: E402
from g1_model import act_memory as AM                               # noqa: E402
from g1_model import train as T                                     # noqa: E402
from g1_model.act import ACTConfig, ACTPolicy, build_act            # noqa: E402
from g1_model.history import (CarriedObs, HistoryChunkDataset,      # noqa: E402
                              HistoryObs, collate_history, history_of)
from g1_model.loader import (ChunkDataset, LoaderConfig,            # noqa: E402
                             LoaderError, TrackingPolicy, collate_chunks)

TEST_LR, TEST_WD = 1e-4, 0.0


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _small(cls=ACTConfig, K=8, W=1, **kw):
    """The small test model of test_act.py, of either config class."""
    kw.setdefault("lr", TEST_LR)
    kw.setdefault("weight_decay", TEST_WD)
    return cls(obs_window=W, chunk_size=K, hidden_dim=64, dim_feedforward=128, nheads=4,
               enc_layers=2, dec_layers=1, **kw)


def _full(K=8, seed=0, **kw):
    torch.manual_seed(seed)
    return AM.build_act_lstm_full(cfg=_small(AM.ACTFullHistoryConfig, K=K, **kw)).eval()


def _orig(K=8, seed=0, **kw):
    torch.manual_seed(seed)
    return build_act(cfg=_small(ACTConfig, K=K, **kw)).eval()


def _tb():
    import importlib
    return importlib.import_module("train_bc")


# ═══ act_orig IS ACT at the original window ══════════════════════════════════
def test_act_orig_is_act_at_w1_config_weights_and_function():
    """The runner's act_orig config equals its act config at --obs-window 1, at
    the gate and outside it, and builds the same model from the same seed."""
    tb = _tb()
    for gate in (True, False):
        orig = tb._model_config("act_orig", 100, None, 8, gate=gate)
        act1 = tb._model_config("act", 100, 1, 8, gate=gate)
        assert type(orig) is type(act1) is ACTConfig and orig == act1, (orig, act1)
        assert orig.obs_window == 1 and orig.use_lstm is False
        assert orig.optimizer_config() == act1.optimizer_config()
    # the runner's own build path, full size: the same model to the bit
    tcfg = tb._train_config(orig, 1, 8, seed=3)
    a = tb._make_model(orig, tcfg).eval()
    b = tb._make_model(act1, tcfg).eval()
    assert type(a) is type(b) is ACTPolicy
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    obs = torch.randn(2, 1, spec.STATE_DIM)
    with torch.no_grad():
        assert torch.equal(a(obs), b(obs))


def test_memory_design_models_are_defined_by_w1():
    tb = _tb()
    for kind in AM.MEMORY_DESIGN_MODELS:
        assert tb._model_config(kind, 100, None, 8, gate=False).obs_window == 1
        assert tb._model_config(kind, 100, 1, 8, gate=False).obs_window == 1
        e = _raises(SystemExit, tb._model_config, kind, 100, 12, 8, gate=False)
        assert "W_o = 1" in str(e) and "--obs-window 1" in str(e), e


def test_the_pair_differs_only_in_use_lstm():
    """As act and act_lstm do (test_act.py): same keys, one differing value; the
    TYPE carries the history input; the optimizer settings are ACT's."""
    tb = _tb()
    from g1_model.act import LR, WEIGHT_DECAY
    for gate in (True, False):
        o = tb._model_config("act_orig", 100, None, 8, gate=gate)
        f = tb._model_config("act_lstm_full", 100, None, 8, gate=gate)
        do, df = dataclasses.asdict(o), dataclasses.asdict(f)
        assert set(do) == set(df) == set(dataclasses.asdict(ACTConfig(1, 1, lr=1.0)))
        assert {k for k in do if do[k] != df[k]} == {"use_lstm"}
        assert type(f) is AM.ACTFullHistoryConfig and isinstance(f, ACTConfig)
        assert o.optimizer_config() == f.optimizer_config()
        assert (o.lr, o.chunk_size, o.kl_weight) == (LR, 100, 10.0)
        assert o.weight_decay == (0.0 if gate else WEIGHT_DECAY)
    # and they differ from act_lstm in the window and the LSTM's input only
    al = dataclasses.asdict(tb._model_config("act_lstm", 100, 12, 8, gate=False))
    df = dataclasses.asdict(tb._model_config("act_lstm_full", 100, None, 8, gate=False))
    assert {k for k in al if al[k] != df[k]} == {"obs_window"}


def test_full_config_and_class_refuse_anything_else():
    _raises(ValueError, _small, AM.ACTFullHistoryConfig, W=12)
    _raises(ValueError, _small, AM.ACTFullHistoryConfig, use_lstm=False)
    _raises(TypeError, AM.ACTLSTMFullPolicy, _small(ACTConfig, use_lstm=True))
    m = _full()
    e = _raises(ValueError, m, torch.randn(2, 1, spec.STATE_DIM))
    assert "history" in str(e), e
    _raises(ValueError, m.step, torch.randn(2, 3, spec.STATE_DIM))
    _raises(ValueError, AM.build_act_lstm_full, cfg=dataclasses.asdict(
        _small(AM.ACTFullHistoryConfig)), lstm_input="window")


# ═══ act_lstm_full is act_orig + the LSTM token ══════════════════════════════
def test_act_lstm_full_shares_every_act_orig_parameter():
    """Same seed: every act_orig tensor bit-identical (the LSTM is built last),
    and the only extra tensors are ACT-LSTM's three LSTM modules."""
    so, sf = _orig(seed=5).state_dict(), _full(seed=5).state_dict()
    for k, v in so.items():
        assert k in sf and torch.equal(v, sf[k]), "act_orig tensor %s differs" % k
    extra = set(sf) - set(so)
    assert extra and {k.split(".")[0] for k in extra} == {"lstm", "lstm_proj",
                                                         "lstm_pos_embed"}, sorted(extra)
    m = _full()
    assert isinstance(m, ACTPolicy) and m.cfg.use_lstm is True
    assert (m.lstm.num_layers, m.lstm.hidden_size, m.lstm.dropout) == (2, 256, 0.3)
    assert m.lstm.input_size == spec.STATE_DIM and m.lstm.batch_first
    assert not m.lstm.bidirectional
    assert m.TARGET_READING_MODULES == ACTPolicy.TARGET_READING_MODULES
    assert m.PRIOR_LATENT_MODULES == ("latent_out_proj",)
    assert m.hparams["lstm_input"] == AM.LSTM_INPUT


def test_first_tick_is_act_lstm_at_window_one():
    """On a one-tick history the LSTM reads exactly ACT-LSTM's W_o = 1 window from
    zero memory, so the output IS ACT-LSTM's - through ACT-LSTM's own code."""
    full = _full(seed=2)
    torch.manual_seed(2)
    lstm1 = build_act(cfg=_small(ACTConfig, use_lstm=True)).eval()
    lstm1.load_state_dict(full.state_dict())
    s = torch.randn(5, 1, spec.STATE_DIM)
    with torch.no_grad():
        a = full(HistoryObs(s, s.clone(), torch.ones(5, dtype=torch.int64)))
        b = lstm1(s)
    assert torch.allclose(a, b, atol=1e-6, rtol=0), float((a - b).abs().max())
    # training mode too: the CVAE encoder sees the current state, never the token
    act = torch.randn(5, 8, spec.ACTION_DIM)
    with torch.no_grad():
        ma, la = full.encode(HistoryObs(s, s, torch.ones(5, dtype=torch.int64)), act)
        mb, lb = lstm1.encode(s, act)
    assert torch.equal(ma, mb) and torch.equal(la, lb)


def test_the_lstm_reads_every_step_in_order_and_never_the_padding():
    m = _full()
    x = torch.randn(3, 30, spec.STATE_DIM)
    L = [12, 30, 1]
    with torch.no_grad():
        y = m(history_of(x, L))
        # the OLDEST real step matters (it reads 0..t, not a window)
        # (EXACT inequality: with PyTorch's default LSTM init - forget-gate bias 0,
        # as ACT-LSTM's - an UNTRAINED LSTM keeps ~1e-6 of tick 0 after 30 ticks;
        # the dependence is computed, its strength is for training to learn)
        x0 = x.clone(); x0[:2, 0] += 1.0
        y0 = m(history_of(x0, L))
        for b in (0, 1):
            assert not torch.equal(y0[b], y[b]), b
        # the ORDER matters (it recurs)
        flipped = x.clone(); flipped[0, :11] = x[0, :11].flip(0)
        assert float((m(history_of(flipped, L))[0] - y[0]).abs().max()) > 1e-5
        # the padding never does: rows past each length changed -> output identical
        pad = x.clone(); pad[0, 12:] = 99.0; pad[2, 1:] = -7.0
        assert torch.equal(m(history_of(pad, L)), y)
        # and a sample does not depend on its batch-mates. Tolerance 1e-5, not
        # 1e-6: on the NVIDIA laptop's Intel CPU the alone-vs-batched difference
        # is 6.6e-7..1.0e-6 on outputs of order 1 (float32 GEMM rounding varies
        # with the batch size; measured 2026-10-25) - a leak from a batch-mate or
        # the padding would be O(0.1), and the padding check above is exact.
        for b, n in enumerate(L):
            alone = m(history_of(x[b:b + 1, :n], [n]))
            assert torch.allclose(alone, y[b:b + 1], atol=1e-5, rtol=0), \
                (b, float((alone - y[b:b + 1]).abs().max()))
    # the gradient reaches the OLDEST real step of every sample, and no padded one
    xg = x.clone().requires_grad_(True)
    m(history_of(xg, L)).sum().backward()
    g = xg.grad.abs().sum(-1)                                       # (B, T)
    for b, n in enumerate(L):
        assert float(g[b, 0]) > 0.0 and float(g[b, :n].min()) > 0.0, (b, g[b, :n].min())
        assert float(g[b, n:].sum()) == 0.0, b


def test_tick_by_tick_deployment_equals_rereading_the_whole_history():
    """`step` from the carried (h, c) vs `forward` re-reading states 0..t from
    zero memory, at EVERY tick of a 200-tick episode, two episodes in a batch."""
    m = _full(seed=7)
    T_ = 200
    x = torch.randn(2, T_, spec.STATE_DIM)
    mem, worst = None, 0.0
    with torch.no_grad():
        for t in range(T_):
            got, mem = m.step(x[:, t:t + 1], mem)
            want = m(history_of(x[:, :t + 1]))
            worst = max(worst, float((got - want).abs().max()))
        h_full = m.read_history(x, torch.full((2,), T_, dtype=torch.int64))
    assert worst < 1e-5, "carried vs re-read: max |diff| %.3e" % worst
    assert torch.allclose(mem[0][-1], h_full, atol=1e-5, rtol=0)
    # the state is the caller's: nothing is kept on the model, and the same
    # memory gives the same answer
    assert T.hidden_state(m) == []
    with torch.no_grad():
        a, _ = m.step(x[:, :1], mem)
        b, _ = m.step(x[:, :1], mem)
    assert torch.equal(a, b)


def test_segmented_read_equals_one_unpacked_lstm_call():
    """2026-10-26: read_history reads in fixed READ_SEGMENT-tick segments carrying
    (h, c). It must equal ONE unpacked LSTM call over the same padded batch,
    gathered at length - 1 (within 1e-6), for lengths below, at and across the
    segment boundary - on the CPU, and on CUDA when present."""
    S = AM.READ_SEGMENT
    lengths = [1, S - 1, S, S + 1, 2 * S + 5, 3 * S]
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for dev in devices:
        m = _full().to(dev)
        x = torch.randn(len(lengths), max(lengths) + 7, spec.STATE_DIM, device=dev)
        L = torch.tensor(lengths, dtype=torch.int64)
        with torch.no_grad():
            seg = m.read_history(x, L)
            out, _ = m.lstm(x)
            one = out[torch.arange(len(lengths), device=dev), (L - 1).to(dev)]
        d = float((seg - one).abs().max())
        assert d < 1e-6, (str(dev), d)


def test_only_the_history_dataset_changes_how_it_is_scored():
    """Fixes A/C (2026-10-26) are attributes of HistoryChunkDataset alone; the
    plain ChunkDataset every other model trains on declares neither, so
    score_deployment keeps batch 256 and the cache for them."""
    from g1_model.history import HistoryChunkDataset
    assert HistoryChunkDataset.score_batch_size == 32
    assert HistoryChunkDataset.empty_cache_after_scoring is True
    assert not hasattr(ChunkDataset, "score_batch_size")
    assert not hasattr(ChunkDataset, "empty_cache_after_scoring")


def test_step_from_no_memory_is_tick_zero():
    m = _full()
    s = torch.randn(1, 1, spec.STATE_DIM)
    with torch.no_grad():
        a, mem = m.step(s, None)
        b = m(history_of(s))
    assert torch.allclose(a, b, atol=1e-6, rtol=0)
    assert mem[0].shape == (2, 1, 256) and mem[1].shape == (2, 1, 256)


def test_gradient_reaches_the_lstm_through_the_history():
    m = _full()
    m.train()
    x = torch.randn(4, 20, spec.STATE_DIM)
    obs = history_of(x, [3, 20, 9, 1])
    act = torch.randn(4, 8, spec.ACTION_DIM)
    loss, l1, n, extras = m.loss_terms(obs, act, torch.ones(4, 8, dtype=torch.bool))
    loss.backward()
    for name in ("lstm.weight_ih_l0", "lstm.weight_hh_l1", "lstm_proj.weight",
                 "lstm_pos_embed.weight", "encoder_joint_proj.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and float(g.abs().sum()) > 0.0, name
    assert extras["kl"] > 0.0


# ═══ the loader ═══════════════════════════════════════════════════════════════
def _datasets(K=6, n_eps=2, n_ticks=40):
    from g1_model.test_train import _tiny
    got = _tiny(n_eps=n_eps, n_ticks=n_ticks)
    if got is None:
        return None
    tmp, _ = got
    st, ac = spec.NormStats.identity("state"), spec.NormStats.identity("action")
    cfg = LoaderConfig(chunk_size=K, obs_window=1, tracking=TrackingPolicy())
    return (tmp, ChunkDataset.from_directory(tmp, cfg, st, ac, None),
            HistoryChunkDataset.from_directory(tmp, cfg, st, ac, None))


def test_history_dataset_is_chunkdataset_plus_the_history():
    got = _datasets()
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, plain, hist = got
    try:
        assert len(plain) == len(hist) and np.array_equal(plain.index, hist.index)
        for i in (0, 1, 39, 40, 79):
            a, b = plain[i], hist[i]
            assert set(b) - set(a) == {"history"}
            for k in a:
                assert torch.equal(a[k], b[k]), (i, k)
            ei, t = (int(v) for v in hist.index[i])
            assert torch.equal(b["history"], torch.from_numpy(hist.states[ei][:t + 1]))
            assert torch.equal(b["history"][-1], b["obs"][-1])
        assert T._collate_for(plain) is collate_chunks
        assert T._collate_for(hist) is collate_history
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collate_history_pads_and_keeps_everything_else_as_collate_chunks():
    got = _datasets()
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, plain, hist = got
    try:
        idx = [3, 0, 77, 40]
        b = collate_history([hist[i] for i in idx])
        p = collate_chunks([plain[i] for i in idx])
        assert isinstance(b["obs"], HistoryObs)
        assert torch.equal(b["obs"].current, p["obs"])
        for k in p:
            if k != "obs":
                assert torch.equal(b[k], p[k]), k
        L = b["obs"].lengths
        assert L.tolist() == [int(hist.index[i][1]) + 1 for i in idx] == b["history_length"].tolist()
        H = b["obs"].history
        assert H.shape == (4, int(L.max()), spec.STATE_DIM)
        for r, n in enumerate(L.tolist()):
            assert float(H[r, n:].abs().sum()) == 0.0, "padding must be zeros"
        moved = b["obs"].to("cpu")
        assert moved.lengths.device.type == "cpu" and torch.equal(moved.history, H)
        # the newest history row must BE the current state
        bad = [dict(hist[i]) for i in idx]
        bad[1]["history"] = bad[1]["history"] + 1.0
        _raises(LoaderError, collate_history, bad)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══ the shared loop, unchanged ═══════════════════════════════════════════════
def _train_cfg(mcfg, out, **kw):
    kw.setdefault("epochs", 2)
    return T.TrainConfig(batch_size=8, seed=3, **mcfg.optimizer_config(),
                         run_name="actfull", log_every=0, **kw)


def test_act_lstm_full_trains_checkpoints_and_is_scored_through_the_shared_loop():
    got = _datasets()
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, _, hist = got
    out = tempfile.mkdtemp()
    try:
        mcfg = _small(AM.ACTFullHistoryConfig, K=6)
        cfg = _train_cfg(mcfg, out, epochs=3)
        m = T.seeded_build(cfg, AM.build_act_lstm_full, cfg=mcfg)
        r = T.train(m, hist, cfg, run_dir=out)
        assert r["history"][-1]["train_loss"] < r["history"][0]["train_loss"]
        assert r["history"][0]["train_kl"] > 0.0
        assert r["metadata"]["model"]["cls"] == "ACTLSTMFullPolicy"
        # the guard passes and the score is reproducible
        q1, q2 = T.score_deployment(m, hist), T.score_deployment(m, hist)
        assert q1.value == q2.value and np.isfinite(q1.value)
        # checkpoint: rebuilt from model_kwargs alone, identical predictions
        batch = collate_history([hist[i] for i in (0, 5, 41, 79)])
        m = m.cpu().eval()
        with torch.no_grad():
            before = m(batch["obs"]).clone()
        back, ck = T.load_checkpoint(os.path.join(out, "last.pt"), AM.build_act_lstm_full)
        assert ck["model_cls"] == "ACTLSTMFullPolicy"
        assert ck["model_kwargs"]["lstm_input"] == AM.LSTM_INPUT
        assert isinstance(back.cfg, AM.ACTFullHistoryConfig)
        with torch.no_grad():
            assert torch.equal(before, back(batch["obs"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_two_identical_act_lstm_full_runs_are_bit_identical():
    got = _datasets()
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, _, hist = got
    out = tempfile.mkdtemp()
    try:
        mcfg = _small(AM.ACTFullHistoryConfig, K=6)
        losses = []
        for i in (1, 2):
            cfg = _train_cfg(mcfg, out)
            m = T.seeded_build(cfg, AM.build_act_lstm_full, cfg=mcfg)
            r = T.train(m, hist, cfg, run_dir=os.path.join(out, "r%d" % i))
            losses.append([x["train_loss"] for x in r["history"]])
        assert losses[0] == losses[1], losses
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_cuda_resume_of_an_lstm_with_dropout_is_not_bit_exact():
    """A KNOWN LIMITATION, pinned (2026-10-25, NVIDIA laptop): cuDNN's LSTM keeps
    its own dropout RNG state, seeded once per module and advanced inside cuDNN,
    which a full-state resume (torch/CUDA generator states) cannot restore. So on
    CUDA a resumed act_lstm_full - and any LSTM with between-layer dropout, i.e.
    act_lstm too - does NOT reproduce the uninterrupted curve; with LSTM dropout
    0 it does, which isolates the cause. Uninterrupted runs ARE deterministic
    (test_two_identical_act_lstm_full_runs_are_bit_identical). No result depends
    on a resume: the launchers restart an interrupted run from scratch."""
    import dataclasses
    from g1_model.test_train import _losses
    if not torch.cuda.is_available():
        print("      (skipped: no CUDA)")
        return
    got = _datasets(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, _, hist = got
    outcome = {}
    try:
        for drop in (0.3, 0.0):
            mcfg = dataclasses.replace(_small(AM.ACTFullHistoryConfig, K=5), lstm_dropout=drop)
            dirs = [tempfile.mkdtemp() for _ in range(3)]

            def run(out, max_steps, resume=None):
                cfg = T.TrainConfig(epochs=None, batch_size=8, seed=3, max_steps=max_steps,
                                    window_steps=7, stop_rule=None, log_every=0,
                                    **mcfg.optimizer_config())
                m = T.seeded_build(cfg, AM.build_act_lstm_full, cfg=mcfg)
                return T.train(m, hist, cfg, run_dir=out, resume=resume)
            try:
                whole = run(dirs[0], 28)
                run(dirs[1], 14)
                resumed = run(dirs[2], 28, resume=T.Resume(os.path.join(dirs[1], "state.pt")))
                outcome[drop] = _losses(resumed["history"]) == _losses(whole["history"])
            finally:
                for d in dirs:
                    shutil.rmtree(d, ignore_errors=True)
        assert outcome == {0.3: False, 0.0: True}, outcome
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_full_state_resume_is_bit_exact_for_act_lstm_full():
    """As test_train's ACT resume test: killed mid-pass, the resumed curve is the
    uninterrupted one to the bit - LSTM dropout RNG and the shuffled history
    samples included."""
    from g1_model.test_train import _losses
    got = _datasets(K=5)
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, _, hist = got
    a, b, c = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    mcfg = _small(AM.ACTFullHistoryConfig, K=5)

    # ON THE CPU: bit-exact there. On CUDA it is NOT, with LSTM dropout > 0 -
    # see test_cuda_resume_of_an_lstm_with_dropout_is_not_bit_exact.
    def run(out, max_steps, resume=None):
        cfg = T.TrainConfig(epochs=None, batch_size=8, seed=3, max_steps=max_steps,
                            window_steps=7, stop_rule=None, log_every=0, prefer_cuda=False,
                            **mcfg.optimizer_config())
        m = T.seeded_build(cfg, AM.build_act_lstm_full, cfg=mcfg)
        return T.train(m, hist, cfg, run_dir=out, resume=resume)
    try:
        whole = run(a, 28)
        run(b, 14)
        resumed = run(c, 28, resume=T.Resume(os.path.join(b, "state.pt")))
        assert _losses(resumed["history"]) == _losses(whole["history"])
        assert resumed["metadata"]["resume"]["kind"] == "full_state"
    finally:
        for d in (tmp, a, b, c):
            shutil.rmtree(d, ignore_errors=True)


def test_score_deployment_refuses_a_plain_window_for_act_lstm_full():
    """Scored on the plain loader (no history), act_lstm_full must refuse rather
    than read the one-step window as the episode."""
    got = _datasets()
    if got is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    tmp, plain, _ = got
    try:
        _raises(ValueError, T.score_deployment, _full(K=6), plain)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_carried_obs_validates_its_shapes():
    _raises(ValueError, CarriedObs, torch.randn(2, 1, spec.STATE_DIM), torch.randn(3, 256))
    _raises(ValueError, HistoryObs, torch.randn(2, 1, spec.STATE_DIM),
            torch.randn(2, 5, spec.STATE_DIM), torch.tensor([6, 1]))
    _raises(ValueError, HistoryObs, torch.randn(2, 1, spec.STATE_DIM),
            torch.randn(2, 5, spec.STATE_DIM), torch.tensor([0, 1]))
    o = history_of(torch.randn(2, 5, spec.STATE_DIM), [5, 2])
    assert o.dim() == 3 and tuple(o.shape) == (2, 1, spec.STATE_DIM)
    assert o.reshape(2, -1).shape == (2, spec.STATE_DIM) and o.dtype == torch.float32


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
            print("FAIL  %s\n        %s: %s" % (name, type(e).__name__, e))
        else:
            print("ok    %s" % name)
    print("\n%d/%d passed" % (len(_tests()) - failed, len(_tests())))
    raise SystemExit(1 if failed else 0)
