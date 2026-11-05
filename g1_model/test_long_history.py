"""act_lstm_full on LONG episodes: up to 3000 ticks (120 s), human-demonstration length.

    python -m g1_model.test_long_history

Human teleoperated demonstrations are expected to run 40-120 s (1000-3000
ticks), against scripted_v3's 548-692. act_lstm_full's LSTM reads the whole
episode, so episode length is its sequence length. What these check, at 3000:
  * TRAINING: the 64-tick segmented read (`read_history`, READ_SEGMENT) equals
    one LSTM call over the same 3000-tick padded batch; samples drawn from a
    3000-tick episode (t = 0 .. 2999) batch, train and are scored through the
    loop's own pieces (`_collate_for`, `forward_loss`, `score_deployment` and
    its guard).
  * DEPLOYMENT: the carried (h, c) of `step`, tick by tick for 3000 ticks,
    equals re-reading states 0..t from zero memory - no drift; through
    `deploy.CheckpointPolicy` too; and `deploy.deploy_episode` runs a
    3000-tick closed-loop episode to its cap.
Small transformer (test width), the REAL LSTM (2 x 256, dropout 0.3), CPU.
Nothing here is trained beyond single optimizer steps.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from g1_data import spec                                            # noqa: E402
from g1_model import act_memory as AM                               # noqa: E402
from g1_model import train as T                                     # noqa: E402
from g1_model.history import HistoryChunkDataset, collate_history, history_of  # noqa: E402

LONG = 3000                    # ticks: 120 s at 25 Hz
#: carried vs re-read, and segmented vs one call: the tolerance test_act_memory
#: uses at 200 ticks. Measured here at 3000 ticks: see the asserts' messages.
TOL = 1e-5


def _full(K=8, seed=0):
    from g1_model.test_act_memory import _small
    torch.manual_seed(seed)
    return AM.build_act_lstm_full(cfg=_small(AM.ACTFullHistoryConfig, K=K)).eval()


def test_segmented_read_equals_one_lstm_call_at_3000_ticks():
    S = AM.READ_SEGMENT
    lengths = [1, S, 1500, LONG - 1, LONG]
    m = _full()
    x = torch.randn(len(lengths), LONG, spec.STATE_DIM)
    L = torch.tensor(lengths, dtype=torch.int64)
    with torch.no_grad():
        seg = m.read_history(x, L)
        out, _ = m.lstm(x)
        one = out[torch.arange(len(lengths)), L - 1]
    d = float((seg - one).abs().max())
    assert d < TOL, "segmented vs one call at %d ticks: max |diff| %.3e" % (LONG, d)
    # 3000 is not a multiple of 64: the read pads to 3008 and stops there
    assert (LONG + S - 1) // S * S == 3008


def test_carried_memory_equals_rereading_3000_ticks():
    """`step` from the carried (h, c) at every one of 3000 ticks, against the
    policy re-reading 0..t from zero memory at ticks across the episode
    (segment boundaries included), and the final hidden state against one read
    of the whole episode. Two episodes in a batch."""
    m = _full(seed=7)
    x = torch.randn(2, LONG, spec.STATE_DIM)
    check = {0, 63, 64, 999, 2047, 2048, LONG - 1}
    mem, worst, at = None, 0.0, {}
    with torch.no_grad():
        for t in range(LONG):
            got, mem = m.step(x[:, t:t + 1], mem)
            if t in check:
                want = m(history_of(x[:, :t + 1]))
                at[t] = float((got - want).abs().max())
                worst = max(worst, at[t])
        h_full = m.read_history(x, torch.full((2,), LONG, dtype=torch.int64))
    assert worst < TOL, "carried vs re-read: %s" % {k: "%.2e" % v for k, v in sorted(at.items())}
    dh = float((mem[0][-1] - h_full).abs().max())
    assert dh < TOL, "final hidden state carried vs read: %.3e" % dh
    assert all(torch.isfinite(t_).all() for t_ in mem)


def _long_dataset(n_eps=2, K=8, seed=0):
    import bench_memory_design as B
    return B._synthetic_dataset(HistoryChunkDataset, K, 1, n_eps, LONG, seed)


def test_samples_from_a_3000_tick_episode_train_through_the_shared_loop():
    ds = _long_dataset()
    assert len(ds) == 2 * LONG and ds.lengths == [LONG, LONG]
    assert T._collate_for(ds) is collate_history
    idx = [LONG - 1, 0, 1500, 2 * LONG - 1]                     # t = 2999, 0, 1500, 2999
    b = collate_history([ds[i] for i in idx])
    assert b["history_length"].tolist() == [LONG, 1, 1501, LONG]
    assert tuple(b["obs"].history.shape) == (4, LONG, spec.STATE_DIM)
    m = _full()
    m.train()
    dim_mask = torch.from_numpy(np.asarray(spec.ACTION_MASK, dtype=bool).copy())
    loss, l1, n, extras = T.forward_loss(m, b["obs"], b["action"], b["action_mask"], dim_mask)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    for name in ("lstm.weight_ih_l0", "lstm.weight_hh_l1", "lstm_proj.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0, name
    # the gradient reaches the RECENT end of a 3000-tick history and never its
    # padding. (How far back it reaches is for training to learn: an untrained,
    # default-init LSTM forgets tick 0 long before tick 2999 - see NOTES.)
    xg = torch.randn(2, LONG, spec.STATE_DIM, requires_grad=True)
    m(history_of(xg, [LONG, 100])).sum().backward()
    g = xg.grad.abs().sum(-1)
    assert float(g[0, LONG - 64:].min()) > 0 and float(g[1, 100:].sum()) == 0.0


def test_score_deployment_on_a_3000_tick_episode():
    """The selection quantity, guard included, on one whole 3000-tick episode -
    3000 samples, each re-reading its history (32 per batch, segmented)."""
    ds = _long_dataset(n_eps=1, seed=1)
    m = _full()
    q1 = T.score_deployment(m, ds)
    assert np.isfinite(q1.value) and q1.value > 0
    q2 = T.score_deployment(m, ds)
    assert q1.value == q2.value


def _states_3000():
    import test_deploy as TD
    from g1_data.recorder import load_episode
    rec, _ = load_episode(TD._recorded()[0])
    S = np.asarray(rec["states"], dtype=np.float64)
    return np.resize(S, (LONG, S.shape[1]))                     # a real episode, repeated


def test_deploy_wrapper_carries_memory_for_3000_ticks():
    """deploy.CheckpointPolicy (ensemble off, so each action is one chunk's first
    step) over 3000 ticks of real states: its carried-memory action equals the
    model re-reading the normalized states 0..t, at ticks across the episode."""
    import test_memory_design as TM
    pol = TM._policy("act_lstm_full", ensemble=False)
    S = _states_3000()
    pol.reset()
    got = [pol.act(S[t], t) for t in range(LONG)]
    assert pol.memory is not None and np.isfinite(np.asarray(got)).all()
    Z = np.stack([pol.st.normalize(np.asarray(s, np.float32).astype(np.float64)).astype(np.float32)
                  for s in S])
    worst = 0.0
    for t in (0, 64, 1000, 2500, LONG - 1):
        with torch.no_grad():
            chunk = pol.model(history_of(torch.from_numpy(Z[:t + 1]).unsqueeze(0)))[0]
        want = pol.ac.denormalize(chunk[0].numpy().astype(np.float64))
        worst = max(worst, float(np.abs(got[t] - want).max()))
    assert worst < 1e-4, "wrapper carried vs re-read at 3000 ticks: %.3e" % worst
    pol.reset()
    assert pol.memory is None


def test_closed_loop_deploy_runs_3000_ticks_to_the_cap():
    """deploy.deploy_episode, the untrained small act_lstm_full policy, cap 3000:
    nothing grasps, so the episode runs to the cap (or falls) and is scored."""
    import deploy as D
    import test_deploy as TD
    import test_memory_design as TM
    _, _, loco = TD._recorded()
    res, arrays = D.deploy_episode(TD.SEED, TM._policy("act_lstm_full"), loco=loco,
                                   max_ticks=LONG)
    assert res["ticks"] == len(arrays["states"]) == len(arrays["actions"])
    if res["fell"]:
        assert res["end_reason"] == "fell" and res["ticks"] <= LONG
    else:
        assert res["end_reason"] == "time_limit" and res["ticks"] == LONG, res["end_reason"]
    assert np.isfinite(arrays["actions"]).all() and res["success"] is False
    assert len(res["diagnostics"]) > 0 and res["trajectory_sha256"]


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
