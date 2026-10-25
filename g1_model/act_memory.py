"""The memory-design pair: act_orig and act_lstm_full. Opt-in, behind new model names.

    act_orig       ACT with the ORIGINAL PAPER'S INPUT: the current observation
                   only, W_o = 1, no history (`reference/act/utils.py:37`,
                   `detr_vae.py:80`). Not a new model: it is `ACTPolicy` from
                   `ACTConfig(obs_window=1)`, which act.py has always accepted
                   as "1 reproduces the reference's observation". Everything
                   else is ACT's: K = 100, beta 10, lr 1e-5, batch 8, temporal
                   ensembling at deployment. The NAME exists so a run states
                   which design it is, in its metadata and its folder.

    act_lstm_full  act_orig PLUS an LSTM reading the WHOLE episode so far, tick
                   0 to the current tick, variable length. Its last-layer final
                   hidden state becomes the third encoder token, exactly where
                   ACT-LSTM puts its window summary: [latent, proprio, lstm]
                   with `lstm_pos_embed`. Same LSTM (2 layers, hidden 256,
                   dropout 0.3 between layers), same projection, same
                   construction order.

THE PAIR DIFFERS IN ONE THING, AS EVERY RUNG OF THE LADDER DOES
---------------------------------------------------------------
`ACTLSTMFullPolicy` IS `ACTPolicy` with `use_lstm=True` at W_o = 1. Only the
LSTM's INPUT is overridden (`_lstm_token`): ACT-LSTM reads its W_o window from
zero memory, act_lstm_full reads the episode history from zero memory. The
forward pass, the CVAE encoder (which sees the current state and never the
token), the decoder, the loss and the guard declarations are ACT-LSTM's own
code, not a copy. The LSTM is built after every ACT module, so act_lstm_full
from a seed starts with every act_orig parameter bit-identical, and the two
configs differ in `use_lstm` alone (test_act_memory).

TRAINING AND DEPLOYMENT READ THE SAME HISTORY TWO WAYS
------------------------------------------------------
Training (and the gate, and validation) samples a random (episode, t) like every
other model, and the LSTM re-reads states 0..t from zero memory: packed
variable-length batches (`g1_model/history.py`). No state is carried, so the
model is still a pure function of (weights, observation) and
`train._DeploymentGuard` applies unchanged. Deployment cannot afford to re-read
up to 900 ticks every tick, so `step` advances the LSTM ONE tick from the
carried (h, c) and `tools/deploy.py` keeps that (h, c) - in the policy wrapper,
never on the model - and resets it per episode. The two agree up to
floating-point rounding (test_act_memory, on whole episodes).

DISCLOSED: in deployment an episode may run to 900 ticks (`deploy.MAX_TICKS`),
longer than any training episode (scripted_v3: 548-692). Past the longest
training episode the LSTM reads a history length it never saw in training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


from g1_model.act import ACTConfig, ACTPolicy
from g1_model.history import CarriedObs, HistoryObs

#: The model names, as `tools/train_bc.py --model` takes them and as every run's
#: metadata (`config.notes.model`) records them.
ACT_ORIG = "act_orig"
ACT_LSTM_FULL = "act_lstm_full"
MEMORY_DESIGN_MODELS = (ACT_ORIG, ACT_LSTM_FULL)

#: The original paper's observation: ONE timestep (`reference/act/utils.py:37`,
#: `detr_vae.py:80`; correspondence row 53). Both models are defined by it.
ORIGINAL_OBS_WINDOW: int = 1

#: Written into act_lstm_full's `hparams` (and so into every checkpoint's
#: `model_kwargs`), so a checkpoint states what its LSTM reads.
LSTM_INPUT = "episode_history"
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
    """act_lstm_full's config: an `ACTConfig` with `use_lstm=True` at W_o = 1.

    No field is added, so `asdict` has exactly ACTConfig's keys and an
    act_orig config and an act_lstm_full config differ in `use_lstm` alone. The
    TYPE is what selects the full-history LSTM input; `__post_init__` refuses
    anything that is not W_o = 1 with the LSTM on.
    """

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

    Inputs:
      `HistoryObs`  training and scoring: the current window plus the padded
                    history 0..t and its lengths (`g1_model/history.py`)
      `CarriedObs`  deployment, built by `step` from the carried LSTM state
    A plain tensor is refused: it has no history, and reading its one step as
    the whole episode would silently be ACT-LSTM at W_o = 1.
    """

    #: Read by `tools/deploy.py`: carry the LSTM's (h, c) tick by tick through
    #: `step`, reset per episode. Every other model is a function of the window.
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
        """(B, T, 48) padded histories, (B,) lengths -> (B, H) last-layer hidden
        state after each sample's LAST REAL step, from zero memory.

        UNPACKED (2026-10-25): the LSTM runs over the whole padded batch and the
        last layer's output at index length - 1 - which IS that layer's hidden
        state after the last real step - is gathered per sample. An LSTM is
        causal, so the padding after it cannot reach that output or its gradient
        (tested to the bit). Until 2026-10-25 this packed the batch
        (`pack_padded_sequence`), and cuDNN's packed-sequence LSTM aborted the
        process natively on the laptop (0xC0000409, act_lstm_full_n40_seed0 at
        ~step 25k); the unpacked call is the cuDNN path act_lstm uses. Costs the
        compute on the padding."""
        out, _ = self.lstm(history)                                 # (B, T, H)
        idx = (torch.as_tensor(lengths, dtype=torch.int64) - 1).to(out.device)
        return out[torch.arange(out.shape[0], device=out.device), idx]

    # ---- deployment -------------------------------------------------------
    def step(self, obs: torch.Tensor,
             memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """One deployed tick: ((B, 1, 48) current window, carried (h, c) or None
        at tick 0) -> ((B, K, 22) chunk, (h, c) after this tick).

        Advances the LSTM one step from `memory` and runs the policy on the new
        last-layer hidden state. Pure: the caller keeps the returned state and
        passes it back next tick; nothing is stored on the model."""
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
