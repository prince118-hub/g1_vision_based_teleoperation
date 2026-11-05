"""The memory-design pair: act_orig (ACT, W_o = 1) and act_lstm_full (+ whole-episode LSTM).

Only the LSTM's input differs from ACT-LSTM; training re-reads 0..t, deployment carries (h, c).
why: see DESIGN_NOTES.md#act-memory-pair
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


from g1_model.act import ACTConfig, ACTPolicy
from g1_model.history import CarriedObs, HistoryObs

#: The model names, as `tools/train_bc.py --model` takes them and as every run's
#: metadata (`config.notes.model`) records them.
# why: see DESIGN_NOTES.md#act-memory-models
ACT_ORIG = "act_orig"
ACT_LSTM_FULL = "act_lstm_full"
MEMORY_DESIGN_MODELS = (ACT_ORIG, ACT_LSTM_FULL)

# ACT ref: utils.py:37, detr/models/detr_vae.py:80: the original observation is ONE timestep.
ORIGINAL_OBS_WINDOW: int = 1

#: Written into act_lstm_full's `hparams` (and so into every checkpoint's
#: `model_kwargs`), so a checkpoint states what its LSTM reads.
LSTM_INPUT = "episode_history"
#: Ticks per LSTM call in `read_history`: one fixed shape keeps the CUDA allocator from fragmenting.
# why: see DESIGN_NOTES.md#act-memory-read-history
READ_SEGMENT = 64
LSTM_INPUT_DESCRIPTION = (
    "the WHOLE episode so far: normalized states from tick 0 to the current tick, "
    "variable length, oldest first. Training re-reads it from zero memory for every "
    "sample (packed variable-length batches); deployment carries the LSTM's (h, c) "
    "tick by tick and resets it per episode")

INPUT_DESCRIPTION = {
    ACT_ORIG: "the current observation only (W_o = 1, no history) - the original "
              "ACT paper's input",
    ACT_LSTM_FULL: "the current observation (W_o = 1) for every ACT component, plus "
                   "an LSTM token over " + LSTM_INPUT_DESCRIPTION,
}


@dataclass(frozen=True)
class ACTFullHistoryConfig(ACTConfig):
    """act_lstm_full's config: an `ACTConfig` with `use_lstm=True` at W_o = 1; adds no field."""
    # why: see DESIGN_NOTES.md#act-memory-config

    use_lstm: bool = True

    def __post_init__(self):
        super().__post_init__()
        if int(self.obs_window) != ORIGINAL_OBS_WINDOW:
            raise ValueError("act_lstm_full is defined by W_o = %d (the current state; "
                             "the history is the LSTM's), got obs_window=%r"
                             % (ORIGINAL_OBS_WINDOW, self.obs_window))
        if self.use_lstm is not True:
            raise ValueError("act_lstm_full IS the LSTM variant: use_lstm must be True, "
                             "got %r" % (self.use_lstm,))

    def as_metadata(self) -> dict:
        d = super().as_metadata()
        d.update(lstm_input=LSTM_INPUT, lstm_input_description=LSTM_INPUT_DESCRIPTION,
                 memory_design_model=ACT_LSTM_FULL)
        return d


class ACTLSTMFullPolicy(ACTPolicy):
    """ACT-LSTM at W_o = 1 whose LSTM reads the whole episode history.

    WARNING: a plain tensor is refused; it has no history.
    why: see DESIGN_NOTES.md#act-memory-policy
    """

    #: Read by `tools/deploy.py`: carry the LSTM's (h, c) tick by tick through
    #: `step`, reset per episode. Every other model is a function of the window.
    # why: see DESIGN_NOTES.md#act-memory-two-reads
    CARRIES_MEMORY = True

    def __init__(self, cfg: ACTFullHistoryConfig):
        if not isinstance(cfg, ACTFullHistoryConfig):
            raise TypeError("ACTLSTMFullPolicy needs an ACTFullHistoryConfig, got %s"
                            % type(cfg).__name__)
        super().__init__(cfg)
        self.hparams = dict(self.hparams, lstm_input=LSTM_INPUT)

    def forward(self, obs, actions: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None):
        if not isinstance(obs, (HistoryObs, CarriedObs)):
            raise ValueError(
                "act_lstm_full reads the episode history: pass a HistoryObs "
                "(g1_model/history.py; training and scoring) or deploy through "
                "step(), got %s" % type(obs).__name__)
        return super().forward(obs, actions, action_mask)

    # ---- the only override of ACT-LSTM: what the LSTM reads ---------------
    def _lstm_token(self, obs) -> torch.Tensor:
        """(B, h): the projected last-layer hidden state after the CURRENT tick -
        re-read from tick 0 (`HistoryObs`) or carried (`CarriedObs`)."""
        if isinstance(obs, CarriedObs):
            return self.lstm_proj(obs.lstm_hidden)
        return self.lstm_proj(self.read_history(obs.history, obs.lengths))

    def read_history(self, history: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, T, 48) padded, (B,) lengths -> (B, H) hidden state after each last real step."""
        # Reads READ_SEGMENT ticks at a time, carrying (h, c).
        # why: see DESIGN_NOTES.md#act-memory-read-history
        B, T = history.shape[0], history.shape[1]
        idx = (torch.as_tensor(lengths, dtype=torch.int64) - 1).to(history.device)
        S = READ_SEGMENT
        end = (int(idx.max()) // S + 1) * S if B else 0
        if end > T:
            history = torch.nn.functional.pad(history, (0, 0, 0, end - T))
        rows = torch.arange(B, device=history.device)
        token, memory = None, None
        for s in range(0, end, S):
            out, memory = self.lstm(history[:, s:s + S], memory)        # (B, S, H)
            here = (idx >= s) & (idx < s + S)
            pick = out[rows, (idx - s).clamp(0, S - 1)]
            token = pick if token is None else torch.where(here[:, None], pick, token)
        return token

    # ---- deployment -------------------------------------------------------
    def step(self, obs: torch.Tensor,
             memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """One deployed tick: ((B, 1, 48), carried (h, c) or None) -> ((B, K, 22), new (h, c))."""
        # Pure: the caller keeps the state. why: see DESIGN_NOTES.md#act-memory-step
        # WARNING: deployment can outrun every training episode.
        # why: see DESIGN_NOTES.md#act-memory-disclosed
        if obs.dim() != 3 or tuple(obs.shape[1:]) != (ORIGINAL_OBS_WINDOW,
                                                      self.cfg.state_dim):
            raise ValueError("step takes the (B, 1, %d) current window, got %s"
                             % (self.cfg.state_dim, tuple(obs.shape)))
        _, memory = self.lstm(obs, memory)
        return self(CarriedObs(obs, memory[0][-1])), memory


def build_act_lstm_full(cfg: ACTFullHistoryConfig = None, **kw) -> ACTLSTMFullPolicy:
    """Factory for `train.seeded_build` and `train.load_checkpoint`: takes an
    `ACTFullHistoryConfig`, or a checkpoint's `model_kwargs` (cfg=..., lstm_input=...)."""
    kw = dict(kw)
    tag = kw.pop("lstm_input", LSTM_INPUT)
    if tag != LSTM_INPUT:
        raise ValueError("checkpoint LSTM input %r is not %r" % (tag, LSTM_INPUT))
    if cfg is not None and kw:
        raise ValueError("pass an ACTFullHistoryConfig or keyword arguments, not both")
    if cfg is None:
        cfg = ACTFullHistoryConfig(**kw)
    elif isinstance(cfg, dict):
        cfg = ACTFullHistoryConfig(**cfg)
    return ACTLSTMFullPolicy(cfg)
