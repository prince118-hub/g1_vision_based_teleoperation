"""State-only ACT: the CVAE that chunked BC is missing; ACT-LSTM is this class with `use_lstm=True`.

ACT ref: paths are relative to reference/act/ at tonyzhaozh/act@742c753c; paper arXiv:2304.13705v1.
why: see DESIGN_NOTES.md#act-module
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from g1_data import spec
from g1_model.models import LADDER_GRAD_CLIP

# ─── reference constants ──────────────────────────────────────────────────────
# ACT ref: values match the reference unless marked DIFFERS or OURS (docs/ACT_CORRESPONDENCE.md).
# why: see DESIGN_NOTES.md#act-reference
REF_COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"
REF_PAPER = "arXiv:2304.13705v1"

# ACT ref: detr/models/detr_vae.py:67 (latent_dim = 32, flagged "TODO tune" there)
LATENT_DIM: int = 32
# ACT ref: README.md:76 --hidden_dim 512; paper Table III (overrides detr/main.py:41's 256)
HIDDEN_DIM: int = 512
# ACT ref: README.md:76 --dim_feedforward 3200; paper Table III
DIM_FEEDFORWARD: int = 3200
# ACT ref: imitate_episodes.py:56 nheads = 8; paper Table III
NHEADS: int = 8
# ACT ref: imitate_episodes.py:54 enc_layers = 4; CVAE encoder too (detr/models/detr_vae.py:217)
ENC_LAYERS: int = 4
# ACT ref: imitate_episodes.py:55 dec_layers = 7; paper Table III
REF_DEC_LAYERS: int = 7
# DIFFERS from ACT: one layer built, not 7; bit-identical because the reference reads hs[0].
# why: see DESIGN_NOTES.md#act-decoder-layers
DEC_LAYERS: int = 1
# ACT ref: detr/main.py:43 --dropout 0.1 (README.md:76 does not override it)
DROPOUT: float = 0.1
# ACT ref: detr/models/detr_vae.py:219 (hardcoded "relu")
ACTIVATION: str = "relu"
# ACT ref: detr/main.py:49 --pre_norm is not passed in README.md:76, so POST-norm
PRE_NORM: bool = False
# ACT ref: README.md:76 --kl_weight 10; policy.py:34 (l1 + kl * kl_weight); paper Table III
KL_WEIGHT: float = 10.0
# ACT ref: imitate_episodes.py:255 (k = 0.01, renamed m); the paper gives no value (§IV-A).
# why: see DESIGN_NOTES.md#act-constants
TEMPORAL_ENSEMBLE_M: float = 0.01

# ACT ref: README.md:77 --lr 1e-5; paper Table III (overrides detr/main.py:14's 1e-4).
# WARNING: state it here, never inherit it from another model (TR28).
# why: see DESIGN_NOTES.md#act-constants
LR: float = 1e-5
# ACT ref: detr/main.py:17 --weight_decay 1e-4 (README.md:76 does not override it)
WEIGHT_DECAY: float = 1e-4
# ACT ref: detr/main.py:87 (torch.optim.AdamW)
OPTIMIZER: str = "adamw"

# ACT ref: detr/models/detr_vae.py:131 reads hs[0].
# WARNING: never "fix" to hs[-1]: a different model. why: see DESIGN_NOTES.md#act-decoder-read
DECODER_LAYER_READ: int = 0

# ─── ACT-LSTM constants (OURS: the reference has no recurrence, row 54) ───────
# why: see DESIGN_NOTES.md#act-lstm-flag
#: The proposal's value, "2 stacked" (PLAN.md Phase 4; CLAUDE.md §8 2026-09-25).
LSTM_LAYERS: int = 2
#: OURS, CLAUDE.md §8 2026-09-25. Not from the proposal, not tuned.
LSTM_HIDDEN: int = 256
#: The proposal's value; `nn.LSTM` applies it BETWEEN stacked layers only, never on the token.
# why: see DESIGN_NOTES.md#act-lstm-dropout
LSTM_DROPOUT: float = 0.3


def _get_activation_fn(activation: str):
    # ACT ref: detr/models/transformer.py:306-314 (_get_activation_fn)
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu/glu, not %r" % activation)


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    # ACT ref: detr/models/transformer.py:289-290 (_get_clones)
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def sinusoid_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Fixed (not learned) sinusoidal position table, (1, n_position, d_hid)."""
    # ACT ref: detr/models/detr_vae.py:23-31 (get_sinusoid_encoding_table)
    # WARNING: must stay a buffer. why: see DESIGN_NOTES.md#act-sinusoid
    def angle_vec(position):
        return [position / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]

    table = np.array([angle_vec(p) for p in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])      # dim 2i
    table[:, 1::2] = np.cos(table[:, 1::2])      # dim 2i+1
    return torch.FloatTensor(table).unsqueeze(0)


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = mu + std * eps, the CVAE's reparameterisation."""
    # ACT ref: detr/models/detr_vae.py:17-20 (reparametrize)
    # WARNING: keep the div(2), asserted in the tests. why: see DESIGN_NOTES.md#act-reparametrize
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


# ─── transformer, mirroring reference/act/detr/models/transformer.py ──────────
# why: see DESIGN_NOTES.md#act-transformer-reimplemented
class _EncoderLayer(nn.Module):
    # ACT ref: detr/models/transformer.py:144-201 (TransformerEncoderLayer)

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 normalize_before):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos(tensor, pos):
        # ACT ref: detr/models/transformer.py:163-164 (with_pos_embed): pos goes to Q and K, never V
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask=None, pos=None):
        if self.normalize_before:  # ACT ref: detr/models/transformer.py:181-193
            src2 = self.norm1(src)
            q = k = self._with_pos(src2, pos)
            src2 = self.self_attn(q, k, value=src2,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            return src + self.dropout2(src2)
        # ACT ref: detr/models/transformer.py:166-179 (post-norm, the published configuration)
        q = k = self._with_pos(src, pos)
        src2 = self.self_attn(q, k, value=src,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class _Encoder(nn.Module):
    # ACT ref: detr/models/transformer.py:79-100 (TransformerEncoder)

    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, pos=None):
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


class _DecoderLayer(nn.Module):
    # ACT ref: detr/models/transformer.py:204-286 (TransformerDecoderLayer)

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 normalize_before):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None,
                memory_key_padding_mask=None):
        if self.normalize_before:  # ACT ref: detr/models/transformer.py:252-273
            tgt2 = self.norm1(tgt)
            q = k = self._with_pos(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(query=self._with_pos(tgt2, query_pos),
                                       key=self._with_pos(memory, pos),
                                       value=memory,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            return tgt + self.dropout3(tgt2)
        # ACT ref: detr/models/transformer.py:229-250 (post-norm, the published configuration)
        q = k = self._with_pos(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(query=self._with_pos(tgt, query_pos),
                                   key=self._with_pos(memory, pos),
                                   value=memory,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class _Decoder(nn.Module):
    # ACT ref: detr/models/transformer.py:103-141 (TransformerDecoder); stacks every layer's output.
    # why: see DESIGN_NOTES.md#act-decoder-intermediate

    def __init__(self, layer, num_layers, norm, return_intermediate=True):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, pos=None, query_pos=None,
                memory_key_padding_mask=None):
        out = tgt
        intermediate: List[torch.Tensor] = []
        for layer in self.layers:
            out = layer(out, memory, pos=pos, query_pos=query_pos,
                        memory_key_padding_mask=memory_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(out))
        if self.norm is not None:
            out = self.norm(out)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(out)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return out.unsqueeze(0)


# ─── configuration ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ACTConfig:
    """ACT's hyperparameters; reference values are cited at their constants above.

    `obs_window` and `chunk_size` are required: neither may be inherited silently.
    why: see DESIGN_NOTES.md#act-config
    """

    obs_window: int                 # required; 1 = reference and act_orig, 12 = ablation
    chunk_size: int                 # required; K = number of decoder queries
    #: REQUIRED, no default (TR28): ACT's published value is `LR`, never another model's.
    # why: see DESIGN_NOTES.md#act-config
    lr: float

    hidden_dim: int = HIDDEN_DIM
    dim_feedforward: int = DIM_FEEDFORWARD
    nheads: int = NHEADS
    enc_layers: int = ENC_LAYERS
    dec_layers: int = DEC_LAYERS
    dropout: float = DROPOUT
    activation: str = ACTIVATION
    pre_norm: bool = PRE_NORM
    latent_dim: int = LATENT_DIM
    kl_weight: float = KL_WEIGHT
    weight_decay: float = WEIGHT_DECAY
    optimizer: str = OPTIMIZER
    #: DIFFERS from ACT: never clipped there (detr/main.py:20); see `models.GRAD_CLIP_DISCLOSURE`.
    grad_clip: float = LADDER_GRAD_CLIP

    #: OURS: 48-D state, 22-D action; the reference hardcodes 14 (detr/models/detr_vae.py:52, :230).
    state_dim: int = spec.STATE_DIM
    action_dim: int = spec.ACTION_DIM

    #: ACT-LSTM (D8). False is ACT, bit-identical to the model before the flag existed; the
    #: `lstm_*` fields are INERT then, so the two configs differ in `use_lstm` alone.
    use_lstm: bool = False
    lstm_layers: int = LSTM_LAYERS
    lstm_hidden: int = LSTM_HIDDEN
    lstm_dropout: float = LSTM_DROPOUT

    def __post_init__(self):
        if int(self.obs_window) < 1:
            raise ValueError("obs_window must be >= 1, got %r" % (self.obs_window,))
        if int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1, got %r" % (self.chunk_size,))
        if self.hidden_dim % self.nheads:
            raise ValueError("hidden_dim %d must be divisible by nheads %d"
                             % (self.hidden_dim, self.nheads))
        if not isinstance(self.use_lstm, bool):
            raise ValueError("use_lstm must be a bool, got %r" % (self.use_lstm,))
        if int(self.lstm_layers) < 1 or int(self.lstm_hidden) < 1:
            raise ValueError("lstm_layers and lstm_hidden must be >= 1, got %r, %r"
                             % (self.lstm_layers, self.lstm_hidden))
        if not 0.0 <= float(self.lstm_dropout) < 1.0:
            raise ValueError("lstm_dropout must be in [0, 1), got %r"
                             % (self.lstm_dropout,))

    def optimizer_config(self) -> dict:
        """The optimizer settings ACT declares; `train.make_optimizer` refuses a mismatch (TR28)."""
        return dict(lr=float(self.lr), weight_decay=float(self.weight_decay),
                    optimizer=str(self.optimizer), grad_clip=float(self.grad_clip))

    def as_metadata(self) -> dict:
        d = asdict(self)
        d.update(reference_commit=REF_COMMIT, reference_paper=REF_PAPER,
                 decoder_layer_read=DECODER_LAYER_READ,
                 reference_lr=LR, reference_weight_decay=WEIGHT_DECAY,
                 reference_grad_clip="none (reference/act/detr/main.py:20 '# not used')")
        if self.use_lstm:
            d.update(lstm_dropout_placement=(
                "between stacked LSTM layers only (nn.LSTM semantics); not applied "
                "to the final hidden state that becomes the token"))
        return d


# ─── the model ────────────────────────────────────────────────────────────────
class ACTPolicy(nn.Module):
    """State-only ACT: a CVAE whose decoder is a transformer emitting K actions.

    `forward(obs)` is inference; adding `actions`, `action_mask` is training and returns mu, logvar.
    why: see DESIGN_NOTES.md#act-policy (ACT-LSTM: DESIGN_NOTES.md#act-policy-lstm)
    """

    #: The modules that read the TARGET chunk (CVAE encoder); `train._DeploymentGuard` bars them.
    # why: see DESIGN_NOTES.md#act-guard-declarations
    TARGET_READING_MODULES = ("cls_embed", "encoder_action_proj",
                              "encoder_joint_proj", "latent_proj", "encoder")

    #: The modules whose INPUT is z; `train.score_deployment` verifies z == 0 on every call.
    # ACT ref: detr/models/detr_vae.py:113; paper §IV-B.
    # why: see DESIGN_NOTES.md#act-guard-declarations
    PRIOR_LATENT_MODULES = ("latent_out_proj",)

    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        #: Everything needed to rebuild this module from a checkpoint alone.
        self.hparams = dict(cfg=asdict(cfg))
        h, K = cfg.hidden_dim, cfg.chunk_size

        # ---- CVAE encoder --------------------------------------------------
        # ACT ref: detr/models/detr_vae.py:66-72. [CLS] + proprio + chunk -> mu, logvar; training only.
        self.cls_embed = nn.Embedding(1, h)                        # :68
        self.encoder_action_proj = nn.Linear(cfg.action_dim, h)    # :69 (theirs 14)
        self.encoder_joint_proj = nn.Linear(
            cfg.state_dim * cfg.obs_window, h)                     # :70 (theirs 14)
        self.latent_proj = nn.Linear(h, cfg.latent_dim * 2)        # :71
        # FIXED sinusoidal, length 1 + 1 + K = [CLS], qpos, a_seq   # :72
        self.register_buffer("pos_table", sinusoid_table(1 + 1 + K, h))

        enc_layer = _EncoderLayer(h, cfg.nheads, cfg.dim_feedforward, cfg.dropout,
                                  cfg.activation, cfg.pre_norm)
        # ACT ref: detr/models/detr_vae.py:212-226 (build_encoder): norm only when pre_norm
        self.encoder = _Encoder(enc_layer, cfg.enc_layers,
                                nn.LayerNorm(h) if cfg.pre_norm else None)

        # ---- decoder side; ACT ref: detr/models/detr_vae.py:74-76 --------------
        self.latent_out_proj = nn.Linear(cfg.latent_dim, h)        # :75
        # LEARNED position embedding for exactly [latent, proprio], sized as in the reference.
        # why: see DESIGN_NOTES.md#act-additional-pos
        self.additional_pos_embed = nn.Embedding(2, h)             # :76
        self.input_proj_robot_state = nn.Linear(
            cfg.state_dim * cfg.obs_window, h)                     # :58 (theirs 14)

        # transformer encoder over the two tokens, then a K-query decoder
        t_enc_layer = _EncoderLayer(h, cfg.nheads, cfg.dim_feedforward,
                                    cfg.dropout, cfg.activation, cfg.pre_norm)
        self.t_encoder = _Encoder(t_enc_layer, cfg.enc_layers,
                                  nn.LayerNorm(h) if cfg.pre_norm else None)
        t_dec_layer = _DecoderLayer(h, cfg.nheads, cfg.dim_feedforward,
                                    cfg.dropout, cfg.activation, cfg.pre_norm)
        # ACT ref: detr/models/transformer.py:35: the decoder ALWAYS has a final LayerNorm
        self.t_decoder = _Decoder(t_dec_layer, cfg.dec_layers, nn.LayerNorm(h),
                                  return_intermediate=True)

        # ACT ref: detr/models/detr_vae.py:54 (learned); DIFFERS from the paper's "fixed" (§IV-C).
        # why: see DESIGN_NOTES.md#act-queries
        self.query_embed = nn.Embedding(K, h)                      # :54
        self.action_head = nn.Linear(h, cfg.action_dim)            # :52 (theirs 14)
        # DIFFERS from ACT: `is_pad_head` (detr/models/detr_vae.py:53) dropped; no loss uses it.
        # why: see DESIGN_NOTES.md#act-queries

        self._reset_parameters()

        # ---- ACT-LSTM (D8): LAST, so every ACT parameter stays bit-identical ------
        # why: see DESIGN_NOTES.md#act-policy-lstm
        if cfg.use_lstm:
            self.lstm = nn.LSTM(cfg.state_dim, cfg.lstm_hidden,
                                num_layers=cfg.lstm_layers, batch_first=True,
                                dropout=cfg.lstm_dropout)
            self.lstm_proj = nn.Linear(cfg.lstm_hidden, h)
            # position of the third token; `additional_pos_embed` stays the
            # reference's (2, h) for [latent, proprio]
            self.lstm_pos_embed = nn.Embedding(1, h)

    def optimizer_config(self) -> dict:
        """Hook read by `train.make_optimizer`. See `ACTConfig.optimizer_config`."""
        return self.cfg.optimizer_config()

    def _reset_parameters(self):
        """Xavier-uniform on the policy transformer and the CVAE encoder."""
        # ACT ref: detr/models/transformer.py:44-47 (_reset_parameters)
        # DIFFERS from ACT: the CVAE encoder too. why: see DESIGN_NOTES.md#act-reset-parameters
        for m in (self.encoder, self.t_encoder, self.t_decoder):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # ---- the CVAE encoder --------------------------------------------------
    def encode(self, obs: torch.Tensor, actions: torch.Tensor,
               action_mask: Optional[torch.Tensor] = None):
        """(mu, logvar) from [CLS] + proprio + action chunk. TRAINING ONLY."""
        # ACT ref: detr/models/detr_vae.py:88-108
        # WARNING: padded steps must be masked or z is quietly wrong.
        # why: see DESIGN_NOTES.md#act-encode-mask
        B = obs.shape[0]
        flat = obs.reshape(B, -1)
        action_embed = self.encoder_action_proj(actions)            # (B, K, h)
        qpos_embed = self.encoder_joint_proj(flat).unsqueeze(1)     # (B, 1, h)
        cls = self.cls_embed.weight.unsqueeze(0).repeat(B, 1, 1)    # (B, 1, h)
        src = torch.cat([cls, qpos_embed, action_embed], dim=1)     # (B, K+2, h)
        src = src.permute(1, 0, 2)                                  # (K+2, B, h)

        key_padding = None
        if action_mask is not None:
            is_pad = ~action_mask.bool()                            # (B, K)
            head = torch.zeros((B, 2), dtype=torch.bool, device=obs.device)
            key_padding = torch.cat([head, is_pad], dim=1)          # (B, K+2)

        pos = self.pos_table.clone().detach().permute(1, 0, 2)      # (K+2, 1, h)
        out = self.encoder(src, src_key_padding_mask=key_padding, pos=pos)
        latent_info = self.latent_proj(out[0])                      # CLS only, :105
        mu = latent_info[:, :self.cfg.latent_dim]
        logvar = latent_info[:, self.cfg.latent_dim:]
        return mu, logvar

    # ---- the policy --------------------------------------------------------
    def _lstm_token(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, W_o, 48) -> (B, h): the LSTM's summary of the window, from zero memory each call."""
        # WARNING: nothing crosses calls (Option B). why: see DESIGN_NOTES.md#act-lstm-token
        _, (h_n, _) = self.lstm(obs)                                # (L, B, H)
        return self.lstm_proj(h_n[-1])                              # (B, h)

    def _decode(self, obs: torch.Tensor, latent_input: torch.Tensor) -> torch.Tensor:
        """The transformer encoder over [latent, proprio] (+ lstm), then the K-query decoder."""
        # ACT ref: detr/models/transformer.py:59-63 (the vision path, image tokens removed)
        # WARNING: never the `backbones is None` branch: it drops z.
        # why: see DESIGN_NOTES.md#act-state-only-path
        B = obs.shape[0]
        proprio = self.input_proj_robot_state(obs.reshape(B, -1))   # (B, h)
        src = torch.stack([latent_input, proprio], dim=0)           # (2, B, h)
        pos = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)
        if self.cfg.use_lstm:
            src = torch.cat([src, self._lstm_token(obs).unsqueeze(0)], dim=0)
            pos = torch.cat([pos, self.lstm_pos_embed.weight.unsqueeze(1)
                             .repeat(1, B, 1)], dim=0)              # (3, B, h)
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (K, B, h)
        tgt = torch.zeros_like(query)  # ACT ref: detr/models/transformer.py:72

        memory = self.t_encoder(src, pos=pos)
        hs = self.t_decoder(tgt, memory, pos=pos, query_pos=query)  # (L, K, B, h)
        hs = hs.transpose(1, 2)                                     # (L, B, K, h)
        return hs[DECODER_LAYER_READ]  # ACT ref: detr/models/detr_vae.py:131

    def forward(self, obs: torch.Tensor, actions: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None):
        """(B, W_o, 48) -> (B, K, 22), plus (mu, logvar) when training."""
        # ACT ref: detr/models/detr_vae.py:85 (the mode is whether an action chunk was supplied)
        # why: see DESIGN_NOTES.md#act-forward
        if obs.dim() != 3 or obs.shape[-1] != self.cfg.state_dim:
            raise ValueError("obs must be (B, W_o, %d), got %s"
                             % (self.cfg.state_dim, tuple(obs.shape)))
        if obs.shape[1] != self.cfg.obs_window:
            raise ValueError("this policy was built for obs_window=%d, got %d"
                             % (self.cfg.obs_window, obs.shape[1]))
        B = obs.shape[0]

        if actions is None:                                # inference
            # WARNING: z must be zero at inference (ACT ref: detr/models/detr_vae.py:113)
            mu = logvar = None
            z = torch.zeros(B, self.cfg.latent_dim, dtype=obs.dtype,
                            device=obs.device)
        else:                                              # training
            if actions.shape[1] != self.cfg.chunk_size:
                raise ValueError("action chunk must be K=%d, got %d"
                                 % (self.cfg.chunk_size, actions.shape[1]))
            mu, logvar = self.encode(obs, actions, action_mask)
            z = reparametrize(mu, logvar)

        latent_input = self.latent_out_proj(z)
        a_hat = self.action_head(self._decode(obs, latent_input))
        if actions is None:
            return a_hat
        return a_hat, mu, logvar


    # ---- the loop's optional hook (train.forward_loss) --------------------
    def loss_terms(self, obs, target, pad_mask, dim_mask=None):
        """(loss, reconstruction_l1, n_elements, extras): ACT's L1 + beta*KL."""
        # ACT ref: policy.py:34. why: see DESIGN_NOTES.md#act-loss-terms
        pred, mu, logvar = self(obs, actions=target, action_mask=pad_mask)
        loss, l1, kl, n = act_loss(pred, target, pad_mask, mu, logvar,
                                   self.cfg.kl_weight, dim_mask)
        return loss, l1, n, dict(kl=float(kl.detach()),
                                 kl_weighted=float((kl * self.cfg.kl_weight).detach()))


# ─── the objective ────────────────────────────────────────────────────────────
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL to the standard normal: summed over latent dims, averaged over the batch."""
    # ACT ref: policy.py:71-84 (kl_divergence, the total_kld branch)
    # WARNING: `mean_kld` would rescale the KL 32x and still train. why: see DESIGN_NOTES.md#act-kl
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean(0)


def act_loss(pred, target, pad_mask, mu, logvar, kl_weight: float,
             dim_mask=None):
    """L1 + beta * KL; the reconstruction term is the shared `train.masked_l1`."""
    # ACT ref: policy.py:34
    # DIFFERS from ACT: L1 divides by contributing elements, 1.4704x its scale.
    # why: see DESIGN_NOTES.md#act-loss-reduction
    from g1_model.train import masked_l1
    l1, n = masked_l1(pred, target, pad_mask, dim_mask)
    kl = kl_divergence(mu, logvar)
    return l1 + kl * float(kl_weight), l1, kl, n


# ─── A3: temporal ensembling (INFERENCE ONLY) ─────────────────────────────────
class TemporalEnsembler:
    """Inference only: overlapping chunk predictions weighted exp(-m * i), index 0 the OLDEST."""
    # ACT ref: imitate_episodes.py:218-259
    # why: see DESIGN_NOTES.md#act-temporal-ensembler

    def __init__(self, chunk_size: int, action_dim: int = spec.ACTION_DIM,
                 m: float = TEMPORAL_ENSEMBLE_M):
        self.K = int(chunk_size)
        self.action_dim = int(action_dim)
        self.m = float(m)
        self.reset()

    def reset(self) -> None:
        """Per episode. A buffer carried across episodes would average one
        episode's intent into the next."""
        self._buf: List[Optional[torch.Tensor]] = []
        self.t = 0

    @staticmethod
    def weights(n: int, m: float = TEMPORAL_ENSEMBLE_M) -> np.ndarray:
        """Normalized exp(-m * i); index 0 is the OLDEST."""
        # ACT ref: imitate_episodes.py:256-257
        w = np.exp(-m * np.arange(n))
        return w / w.sum()

    def step(self, chunk: torch.Tensor) -> torch.Tensor:
        """Absorb a (K, 22) prediction made at the current tick; return the
        ensembled action for this tick, (22,)."""
        if chunk.shape != (self.K, self.action_dim):
            raise ValueError("expected a (%d, %d) chunk, got %s"
                             % (self.K, self.action_dim, tuple(chunk.shape)))
        self._buf.append(chunk)
        if len(self._buf) > self.K:
            self._buf.pop(0)
        # prediction made i ticks ago contributes its entry for the current tick
        n = len(self._buf)
        stack = torch.stack([self._buf[j][n - 1 - j] for j in range(n)], dim=0)
        w = torch.from_numpy(self.weights(n, self.m)).to(stack.device,
                                                         stack.dtype).unsqueeze(1)
        self.t += 1
        return (stack * w).sum(dim=0)


def build_act(cfg: ACTConfig = None, **kw) -> ACTPolicy:
    """Factory for `train.seeded_build` and `train.load_checkpoint` (D10)."""
    if cfg is not None and kw:
        raise ValueError("pass an ACTConfig or keyword arguments, not both")
    if cfg is None:
        cfg = ACTConfig(**kw["cfg"]) if "cfg" in kw else ACTConfig(**kw)
    elif isinstance(cfg, dict):
        cfg = ACTConfig(**cfg)
    return ACTPolicy(cfg)
