"""The episode-history loader for act_lstm_full: (current window, history 0..t) -> chunk.

Adds only `history` to `ChunkDataset`; `HistoryObs`/`CarriedObs` look like the window to ACT.
why: see DESIGN_NOTES.md#history-module
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from g1_data import spec
from g1_model.loader import ChunkDataset, LoaderError, collate_chunks


class _CurrentWindow:
    """The current window as `ACTPolicy` reads its `obs`; nothing else is delegated."""

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
    """Training and scoring obs: current (B, W_o, 48), padded history (B, T, 48), lengths (B,)."""
    # WARNING: only `history[b, :lengths[b]]` is data. why: see DESIGN_NOTES.md#history-obs

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
    """Deployment obs, built by `ACTLSTMFullPolicy.step`: current (B, 1, 48), lstm_hidden (B, H)."""
    # why: see DESIGN_NOTES.md#history-obs

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
    """Batch through `collate_chunks`, padding the histories; `obs` becomes a `HistoryObs`."""
    # why: see DESIGN_NOTES.md#history-batch
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
    """`ChunkDataset`, each sample also carrying a view of its episode's states 0..t."""

    #: Read by `train._collate_for`: the shared loop's batcher for this dataset.
    collate_fn = staticmethod(collate_history)
    #: Read by `train.score_deployment`: score 32 histories at a time, then release the CUDA cache.
    # Same math. why: see DESIGN_NOTES.md#history-dataset
    score_batch_size = 32
    empty_cache_after_scoring = True

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
    """A `HistoryObs` from (B, T, 48) histories; for tests and diagnostics."""
    B, T = states.shape[0], states.shape[1]
    lengths = torch.full((B,), T, dtype=torch.int64) if lengths is None else \
        torch.as_tensor(lengths, dtype=torch.int64)
    current = states[torch.arange(B), lengths - 1].unsqueeze(1)
    return HistoryObs(current, states, lengths)
