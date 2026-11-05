"""The policies: BC at K=1, chunked BC at K>1, ONE class.

Every hyperparameter is PROVISIONAL, not tuned: chosen on SCRIPTED data, re-opened on piloted.
why: see DESIGN_NOTES.md#models-provisional
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from g1_data import spec

#: Read by `tools/train_bc.py` and copied into every run's metadata, so the
#: warning above travels with the numbers rather than staying in this file.
PROVISIONAL = (
    "PROVISIONAL HYPERPARAMETERS, NOT TUNED. Chosen on SCRIPTED demonstrator "
    "data, whose action distribution differs from piloted teleoperation. No "
    "sweep or search was run, deliberately. Every value is re-opened when "
    "piloted data exists; no result obtained with them may be cited as tuned."
)

# PROVISIONAL, NOT SWEPT: ACT's published chunk (ACT ref: README.md:76), 4.0 s at 25 Hz.
# SETTLE IT ON PILOTED DATA. why: see DESIGN_NOTES.md#models-k-provisional
K_PROVISIONAL: int = 100

#: THE global-norm clip for EVERY model (a ladder invariant): change it HERE, for every model.
# why: see DESIGN_NOTES.md#models-grad-clip
LADDER_GRAD_CLIP: float = 1.0

#: Written into every run's metadata. The deviation is stated where the numbers go.
GRAD_CLIP_DISCLOSURE = (
    "DISCLOSED DEVIATION FROM THE ACT REFERENCE: every model clips gradients to a "
    "global L2 norm of %.1f. The reference never clips: `--clip_max_norm 0.1` is "
    "marked '# not used' (reference/act/detr/main.py:20) and reference/act contains "
    "no clip_grad_norm_ call. MEASURED (docs/ACT_AUDIT_REPORT.md R1): the norm is ~678 "
    "at ACT's initialisation and median 1.35 at the converged ACT gate weights, above "
    "the threshold on 19 of 20 batches - the clip is active throughout training, not a "
    "safety net. KEPT, identically for every model, because (1) ACT's passing "
    "overfit-10 gate already used it, (2) recurrent models are where clipping matters "
    "most, and (3) a clip that differed between ACT and ACT-LSTM would sit inside the "
    "headline RQ3 comparison: identical-across-models matters more here than matching "
    "the reference." % LADDER_GRAD_CLIP)


@dataclass(frozen=True)
class BCConfig:
    """Every hyperparameter of the BC policy, in one place. See PROVISIONAL.

    `obs_window` and `chunk_size` are required: K must not be inherited silently.
    why: see DESIGN_NOTES.md#models-bc-config
    """

    obs_window: int              # required (C3)
    chunk_size: int              # required (C3): 1 = BC, >1 = chunked BC
    hidden: Tuple[int, ...] = (512, 512)   # ordinary 2x512 MLP
    activation: str = "relu"
    dropout: float = 0.0         # the overfit-10 gate must NOT be regularized
    layer_norm: bool = False

    # training. DECLARED by the model (`optimizer_config`, attached by `build_bc`)
    # and checked against the TrainConfig, as ACT's are (TR28).
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    optimizer: str = "adamw"
    #: The ladder's one value; see LADDER_GRAD_CLIP and GRAD_CLIP_DISCLOSURE.
    grad_clip: float = LADDER_GRAD_CLIP

    provisional: str = PROVISIONAL

    def optimizer_config(self) -> dict:
        """The training hyperparameters BC declares for itself (TR28). Exactly
        `train.MODEL_DECLARED_FIELDS`; the loop refuses a partial declaration."""
        return dict(lr=float(self.lr), weight_decay=float(self.weight_decay),
                    optimizer=str(self.optimizer), grad_clip=float(self.grad_clip))

    def as_metadata(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


_ACT = dict(relu=nn.ReLU, gelu=nn.GELU, tanh=nn.Tanh, silu=nn.SiLU)


class BCPolicy(nn.Module):
    """Behavioural cloning: one observation in, K actions out. K=1 is BC, K>1 is chunked BC.

    One class, so that K is the only difference. why: see DESIGN_NOTES.md#models-bc-policy
    Why an MLP: DESIGN_NOTES.md#models-why-mlp; shapes: DESIGN_NOTES.md#models-shape-contract
    """

    def __init__(self, obs_window: int, chunk_size: int,
                 hidden: Tuple[int, ...] = (512, 512), activation: str = "relu",
                 dropout: float = 0.0, layer_norm: bool = False):
        super().__init__()
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1, got %r" % (chunk_size,))
        if int(obs_window) < 1:
            raise ValueError("obs_window must be >= 1, got %r" % (obs_window,))
        if activation not in _ACT:
            raise ValueError("unknown activation %r (have %s)"
                             % (activation, ", ".join(sorted(_ACT))))
        self.obs_window = int(obs_window)
        self.chunk_size = int(chunk_size)
        #: Everything needed to rebuild this module from a checkpoint alone.
        self.hparams = dict(obs_window=int(obs_window),
                            chunk_size=int(chunk_size),
                            hidden=tuple(hidden), activation=activation,
                            dropout=float(dropout), layer_norm=bool(layer_norm))

        in_dim = self.obs_window * spec.STATE_DIM
        layers, d = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(_ACT[activation]())
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            d = h
        layers.append(nn.Linear(d, self.chunk_size * spec.ACTION_DIM))
        self.net = nn.Sequential(*layers)
        #: Set by `build_bc`; while None, `train.make_optimizer` refuses to train this model.
        self._declared_training: Optional[dict] = None

    def declare_training(self, optimizer_config: dict) -> "BCPolicy":
        """Attach the training hyperparameters this model is trained with. A
        declaration, not a change to the forward pass."""
        self._declared_training = dict(optimizer_config)
        return self

    def optimizer_config(self) -> Optional[dict]:
        """Hook read by `train.make_optimizer`; None when nothing was declared."""
        return (None if self._declared_training is None
                else dict(self._declared_training))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, W_o, 48) -> (B, K, 22)."""
        if obs.dim() != 3 or obs.shape[-1] != spec.STATE_DIM:
            raise ValueError("obs must be (B, W_o, %d), got %s"
                             % (spec.STATE_DIM, tuple(obs.shape)))
        if obs.shape[1] != self.obs_window:
            raise ValueError(
                "this policy was built for obs_window=%d but was given %d. W_o "
                "is not a free parameter at call time: it changes what the model "
                "can see, which is the thing the ACT/ACT-LSTM comparison is "
                "about." % (self.obs_window, obs.shape[1]))
        B = obs.shape[0]
        return self.net(obs.reshape(B, -1)).view(B, self.chunk_size,
                                                 spec.ACTION_DIM)


def first_action(pred: torch.Tensor) -> torch.Tensor:
    """The action a chunked policy executes, (B, K, 22) -> (B, 22); no temporal ensembling (C2)."""
    # why: see DESIGN_NOTES.md#models-first-action
    if pred.dim() != 3 or pred.shape[-1] != spec.ACTION_DIM:
        raise ValueError("expected (B, K, %d), got %s"
                         % (spec.ACTION_DIM, tuple(pred.shape)))
    return pred[:, 0, :]


def build_bc(cfg: BCConfig = None, **kw) -> BCPolicy:
    """Factory used by `train.load_checkpoint` and by the training scripts."""
    if cfg is not None and kw:
        raise ValueError("pass a BCConfig or keyword arguments, not both")
    if cfg is not None:
        return BCPolicy(obs_window=cfg.obs_window, chunk_size=cfg.chunk_size,
                        hidden=cfg.hidden, activation=cfg.activation,
                        dropout=cfg.dropout, layer_norm=cfg.layer_norm
                        ).declare_training(cfg.optimizer_config())
    return BCPolicy(**kw)
