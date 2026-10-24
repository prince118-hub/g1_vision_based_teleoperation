"""The episode-history loader for act_lstm_full: (current window, history 0..t) -> chunk.

WHAT THIS ADDS, AND ONLY THIS
-----------------------------
act_lstm_full (g1_model/act_memory.py) is act_orig - ACT on the CURRENT state
alone, W_o = 1 - plus an LSTM that reads the WHOLE episode so far, tick 0 to the
current tick. Everything a sample shares with act_orig comes from the parent
`ChunkDataset` unchanged: the same `(episode, start_tick)` index, the same
normalization (applied once, at load), the same W_o = 1 observation window, the
same K-step action chunk and its padding mask, the same refusals. A sample adds
ONE thing, `history`: the normalized states 0..t of its own episode, oldest
first, variable length (t + 1 rows).

Training draws samples exactly as for every other model: the shared loop's
DataLoader shuffles the index, so each sample is a random (episode, t), and the
LSTM re-reads that episode from tick 0 each time. Nothing is carried between
samples, so the loop, the loss and `train._DeploymentGuard` apply unchanged.

THE BATCH
---------
Histories differ in length, so `collate_chunks`' plain stack cannot batch them.
`collate_history` pads each batch's histories with zeros to its longest and
records each one's length; the model reads them with `pack_padded_sequence`, so
the padding is never computed on (test_act_memory checks that changing it moves
nothing). Every other key is stacked by `collate_chunks` itself. The batch's
`obs` is a `HistoryObs`. The shared loop finds the batcher through the dataset's
`collate_fn` attribute (`train._collate_for`); a `ChunkDataset` has none, so
every existing model's batches are built exactly as before.

`HistoryObs` and `CarriedObs` present the current (B, 1, 48) window to every ACT
component through the four tensor methods `ACTPolicy` calls on its observation
(`dim`, `shape`, `reshape`, `dtype`/`device`), and carry what the LSTM reads.
That is what lets act_lstm_full run ACT-LSTM's own forward, encode and decode
code with only the LSTM's input changed.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from g1_data import spec
from g1_model.loader import ChunkDataset, LoaderError, collate_chunks


class _CurrentWindow:
    """The (B, W_o, 48) current-state window, as `ACTPolicy` reads its `obs`:
    `forward` checks `dim()` and `shape` and builds z with `dtype`/`device`;
    `encode` and `_decode` flatten it with `reshape`. Nothing else is called on
    an observation, so nothing else is delegated."""

    __slots__ = ()

    def dim(self) -> int:
        return self.current.dim()

    @property
    def shape(self) -> torch.Size:
        return self.current.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.current.dtype

    @property
    def device(self) -> torch.device:
        return self.current.device

    def reshape(self, *shape) -> torch.Tensor:
        return self.current.reshape(*shape)


class HistoryObs(_CurrentWindow):
    """act_lstm_full's observation in training and scoring.

      current : (B, W_o, 48)  the window act_orig is given (W_o = 1: the state at t)
      history : (B, T, 48)    each sample's normalized states 0..t, oldest first,
                              right-padded with zeros to the longest in the batch
      lengths : (B,) int64    t + 1 per sample; kept on the CPU, where
                              `pack_padded_sequence` requires it

    Only `history[b, :lengths[b]]` is data. `collate_history` checks that the
    newest history row IS the current state, so the LSTM's last input is the
    observation the rest of ACT sees.
    """

    __slots__ = ("current", "history", "lengths")

    def __init__(self, current: torch.Tensor, history: torch.Tensor,
                 lengths: torch.Tensor):
        if current.dim() != 3 or current.shape[-1] != spec.STATE_DIM:
            raise ValueError("current must be (B, W_o, %d), got %s"
                             % (spec.STATE_DIM, tuple(current.shape)))
        if history.dim() != 3 or history.shape[-1] != spec.STATE_DIM:
            raise ValueError("history must be (B, T, %d), got %s"
                             % (spec.STATE_DIM, tuple(history.shape)))
        lengths = torch.as_tensor(lengths, dtype=torch.int64).cpu()
        B = current.shape[0]
        if history.shape[0] != B or tuple(lengths.shape) != (B,):
            raise ValueError("batch sizes disagree: current %d, history %d, lengths %s"
                             % (B, history.shape[0], tuple(lengths.shape)))
        if B and (int(lengths.min()) < 1 or int(lengths.max()) > history.shape[1]):
            raise ValueError("lengths must be in [1, %d], got [%d, %d]"
                             % (history.shape[1], int(lengths.min()), int(lengths.max())))
        self.current, self.history, self.lengths = current, history, lengths

    def to(self, *args, **kwargs) -> "HistoryObs":
        """Moves the two state tensors; `lengths` stays on the CPU."""
        return HistoryObs(self.current.to(*args, **kwargs),
                          self.history.to(*args, **kwargs), self.lengths)

    def __repr__(self) -> str:
        return "HistoryObs(current=%s, history=%s, lengths %d..%d)" % (
            tuple(self.current.shape), tuple(self.history.shape),
            int(self.lengths.min()), int(self.lengths.max()))


class CarriedObs(_CurrentWindow):
    """act_lstm_full's observation at DEPLOYMENT, built by `ACTLSTMFullPolicy.step`.

      current     : (B, 1, 48)  the state at t, as in `HistoryObs`
      lstm_hidden : (B, H)      the LSTM's last-layer hidden state after reading
                                tick t, its (h, c) carried from tick 0 of this
                                episode one tick at a time

    The deployed counterpart of re-reading `HistoryObs.history`: same weights,
    same inputs in the same order, so the same token up to floating-point
    rounding (test_act_memory checks it on whole episodes).
    """

    __slots__ = ("current", "lstm_hidden")

    def __init__(self, current: torch.Tensor, lstm_hidden: torch.Tensor):
        if current.dim() != 3 or current.shape[-1] != spec.STATE_DIM:
            raise ValueError("current must be (B, W_o, %d), got %s"
                             % (spec.STATE_DIM, tuple(current.shape)))
        if lstm_hidden.dim() != 2 or lstm_hidden.shape[0] != current.shape[0]:
            raise ValueError("lstm_hidden must be (B, H) with B = %d, got %s"
                             % (current.shape[0], tuple(lstm_hidden.shape)))
        self.current, self.lstm_hidden = current, lstm_hidden

    def to(self, *args, **kwargs) -> "CarriedObs":
        return CarriedObs(self.current.to(*args, **kwargs),
                          self.lstm_hidden.to(*args, **kwargs))


def collate_history(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Batch `HistoryChunkDataset` samples: everything but `history` through
    `collate_chunks` itself, the histories padded to the batch's longest.

    The result's `obs` is a `HistoryObs`; `history_length` is also returned as a
    plain tensor for anything that inspects the batch."""
    hist = [b["history"] for b in batch]
    out = collate_chunks([{k: v for k, v in b.items() if k != "history"} for b in batch])
    lengths = torch.tensor([int(h.shape[0]) for h in hist], dtype=torch.int64)
    padded = pad_sequence(hist, batch_first=True)               # zeros after each length
    newest = padded[torch.arange(len(hist)), lengths - 1]
    if not torch.equal(newest, out["obs"][:, -1]):
        raise LoaderError("the newest history row is not the current state: the "
                          "LSTM would read a different observation from the one "
                          "the rest of the policy sees")
    out["obs"] = HistoryObs(out["obs"], padded, lengths)
    out["history_length"] = lengths
    return out


class HistoryChunkDataset(ChunkDataset):
    """`ChunkDataset`, each sample also carrying its episode's states 0..t.

    Construction, normalization, the sample index and every refusal are the
    parent's. `__getitem__` returns the parent's sample plus `history`, a
    (t + 1, 48) float32 view of the already-normalized states - no copy is made
    until `collate_history` pads the batch.
    """

    #: Read by `train._collate_for`: the shared loop's batcher for this dataset.
    collate_fn = staticmethod(collate_history)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = super().__getitem__(i)
        ei, t = (int(v) for v in self.index[int(i)])
        s["history"] = torch.from_numpy(self.states[ei][:t + 1])
        return s

    def summary(self) -> dict:
        d = super().summary()
        d.update(history="episode states 0..t per sample, variable length "
                         "(1..%d), padded per batch" % int(max(self.lengths)))
        return d


def history_of(states: torch.Tensor, lengths: Optional[Sequence[int]] = None) -> HistoryObs:
    """A `HistoryObs` from (B, T, 48) histories whose newest real row is the
    current state: `lengths` default to T for every sample. For tests and
    diagnostics; the training batches come from `collate_history`."""
    B, T = states.shape[0], states.shape[1]
    lengths = torch.full((B,), T, dtype=torch.int64) if lengths is None else \
        torch.as_tensor(lengths, dtype=torch.int64)
    current = states[torch.arange(B), lengths - 1].unsqueeze(1)
    return HistoryObs(current, states, lengths)
