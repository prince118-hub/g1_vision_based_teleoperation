"""The ONE training loop, the ONE loss, and the ONE evaluation path, shared by every model.

The model is a parameter; everything else is held fixed so a difference is the architecture's.
why: see DESIGN_NOTES.md#train-module
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from g1_data import spec
from g1_data.paths import ROOT, repo_relpath
from g1_data.recorder import _git_commit
from g1_model.loader import ChunkDataset, collate_chunks

RUNS = os.path.join(ROOT, "runs")


class TrainError(AssertionError):
    """A training setup that would produce a number nobody should cite."""


# ─── B2: the device ───────────────────────────────────────────────────────────
def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report(device: torch.device) -> dict:
    """What was selected, and what it can hold. Printed at every startup."""
    r = dict(device=str(device), torch=torch.__version__,
             cuda_available=bool(torch.cuda.is_available()),
             cuda_version=torch.version.cuda, python=platform.python_version())
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        r.update(name=p.name, total_vram_gb=round(p.total_memory / 1024 ** 3, 2),
                 capability="%d.%d" % (p.major, p.minor),
                 multiprocessors=p.multi_processor_count)
    else:
        r.update(name=platform.processor() or "cpu", total_vram_gb=None)
    return r


# ─── B3: determinism ──────────────────────────────────────────────────────────
def set_determinism(seed: int, strict: bool = True) -> dict:
    """Seed everything, and report what could not be made deterministic (goes into run metadata)."""
    # why: see DESIGN_NOTES.md#train-determinism
    notes: List[str] = []
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False       # benchmark picks by timing: varies
    deterministic_algorithms = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_algorithms = True
        except Exception as e:                              # noqa: BLE001
            notes.append("use_deterministic_algorithms refused: %s" % e)
    else:
        notes.append("strict determinism NOT requested by this run")
    # TF32 changes the arithmetic, not just the order, so a run that enables it
    # is not comparable with one that does not. Off, and recorded.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(seed=int(seed),
                deterministic_algorithms=deterministic_algorithms,
                cudnn_deterministic=True, cudnn_benchmark=False, tf32=False,
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                notes=notes)


def seeded_build(train_cfg: "TrainConfig", factory: Callable[..., nn.Module],
                 /, **kw) -> nn.Module:
    """Seed, THEN construct the model. Use this instead of building it yourself."""
    # WARNING: `train()` seeds too late to control weight init.
    # why: see DESIGN_NOTES.md#train-seeded-build
    set_determinism(train_cfg.seed, strict=train_cfg.strict_determinism)
    return factory(**kw)


def seed_worker(worker_id: int) -> None:
    """Per-worker seeding, so `num_workers > 0` does not reintroduce variance."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def _collate_for(ds) -> Callable:
    """The batcher for `ds`: its own `collate_fn` if it declares one, else `collate_chunks`."""
    # why: see DESIGN_NOTES.md#train-collate
    return getattr(ds, "collate_fn", None) or collate_chunks


# ─── B4: the loss ─────────────────────────────────────────────────────────────
# why: see DESIGN_NOTES.md#train-why-l1, DESIGN_NOTES.md#train-two-masks
def masked_l1(pred: torch.Tensor, target: torch.Tensor,
              pad_mask: torch.Tensor,
              dim_mask: Optional[torch.Tensor] = None,
              reduce: bool = True):
    """Mean |pred - target| over (real timesteps) x (trainable dims); returns (loss, n_elements).

    `reduce=False` returns (err, mask). why: see DESIGN_NOTES.md#train-masked-l1
    """
    if dim_mask is None:
        dim_mask = torch.from_numpy(
            np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(pred.device)
    if pred.shape != target.shape:
        raise TrainError("pred %s and target %s must have the same shape"
                         % (tuple(pred.shape), tuple(target.shape)))
    if pred.shape[-1] != spec.ACTION_DIM:
        raise TrainError("actions must have last dimension %d, got %s"
                         % (spec.ACTION_DIM, tuple(pred.shape)))
    m = pad_mask.bool().unsqueeze(-1) & dim_mask.bool().view(1, 1, -1)
    err = (pred - target).abs() * m
    n = m.sum()
    if not reduce:
        return err, m
    total = err.sum()
    return total / n.clamp(min=1), n


# ─── the optional extra-loss hook ─────────────────────────────────────────────
def forward_loss(model: nn.Module, obs, target, pad, dim_mask):
    """(loss, reconstruction_l1, n_elements, extras) for ANY model in the ladder."""
    # WARNING: a `loss_terms` hook must build its reconstruction from `masked_l1` (D6).
    # why: see DESIGN_NOTES.md#train-forward-loss
    fn = getattr(model, "loss_terms", None)
    if fn is not None:
        return fn(obs, target, pad, dim_mask)
    pred = model(obs)
    loss, n = masked_l1(pred, target, pad, dim_mask)
    return loss, loss, n, {}


# ─── B6: the baselines ────────────────────────────────────────────────────────
def baselines(ds: ChunkDataset, device: Optional[torch.device] = None,
              batch_size: int = 512) -> dict:
    """The ZERO and COPY baselines, through the same loss and samples as training."""
    # why: see DESIGN_NOTES.md#train-baselines
    device = device or torch.device("cpu")
    dim_mask = torch.from_numpy(
        np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(device)
    sums = dict(zero=0.0, copy=0.0)
    n_total = 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=_collate_for(ds), num_workers=0)
    offset = 0
    for batch in loader:
        target = batch["action"].to(device)
        pad = batch["action_mask"].to(device)
        B, K, A = target.shape

        # the previous action for each sample, read from the episode arrays
        prev = np.zeros((B, A), dtype=np.float32)
        for b in range(B):
            ei, t = (int(v) for v in ds.index[offset + b])
            if t > 0:
                prev[b] = ds.actions[ei][t - 1]
        offset += B
        copy_pred = torch.from_numpy(prev).to(device).unsqueeze(1).expand(B, K, A)

        for name, pred in (("zero", torch.zeros_like(target)),
                           ("copy", copy_pred)):
            err, m = masked_l1(pred, target, pad, dim_mask, reduce=False)
            sums[name] += float(err.sum())
        n_total += int((pad.bool().unsqueeze(-1)
                        & dim_mask.view(1, 1, -1)).sum())
    return dict(zero=sums["zero"] / max(n_total, 1),
                copy=sums["copy"] / max(n_total, 1),
                elements=n_total, samples=len(ds))


# ─── B1/B5: the loop ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StopRule:
    """When a step-budgeted run counts as CONVERGED: every monitored quantity stalled."""
    # why: see DESIGN_NOTES.md#train-stop-rule

    patience_windows: int
    min_rel_improvement: float
    monitor: Tuple[str, ...] = ("recon_l1", "train_loss")

    def __post_init__(self):
        if int(self.patience_windows) < 1:
            raise TrainError("patience_windows must be >= 1")
        if not (0.0 < float(self.min_rel_improvement) < 1.0):
            raise TrainError("min_rel_improvement must be a fraction in (0, 1)")

    def check(self, windows: Sequence[dict]) -> Optional[dict]:
        """None while improving; otherwise the per-quantity relative improvement
        over the trailing span, which is below threshold for every one."""
        P = int(self.patience_windows)
        if len(windows) <= P:
            return None
        rel = {}
        for key in self.monitor:
            vals = [float(w[key]) for w in windows]
            prior, recent = min(vals[:-P]), min(vals[-P:])
            rel[key] = (prior - recent) / prior if prior > 0 else 0.0
        if all(r < float(self.min_rel_improvement) for r in rel.values()):
            return dict(rel_improvement_over_span=rel)
        return None

    def describe(self, window_steps: int) -> str:
        return ("stop when, for EVERY one of %s, the best windowed mean over the "
                "last %d windows (%d optimizer steps) improves on the best before "
                "them by less than %.2f%% relative; windows of %d steps"
                % (list(self.monitor), self.patience_windows,
                   self.patience_windows * window_steps,
                   100 * self.min_rel_improvement, window_steps))


class _Unstated:
    """Sentinel for a TrainConfig field the caller did not state. One instance;
    survives copy, deepcopy and pickling as itself."""
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "UNSTATED"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_Unstated, ())


UNSTATED = _Unstated()


@dataclass
class TrainConfig:
    """Everything a run needs that is not the model or the data; written into the run directory.

    Exactly one budget; no field that shapes the weights has a default (TR28).
    why: see DESIGN_NOTES.md#train-config
    """

    epochs: Optional[int]
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: str
    grad_clip: float
    seed: int
    num_workers: int = 0
    prefer_cuda: bool = True
    strict_determinism: bool = True
    log_every: int = 1
    #: 0 = only last.pt and best.pt; N > 0 also keeps a checkpoint every N steps or epochs
    #: (step budget: N must be a multiple of window_steps). why: see DESIGN_NOTES.md#train-config
    checkpoint_every: int = 0
    run_name: str = "run"
    #: Free-form. The gate runner puts its data-provenance statement here, and
    #: it ends up in metadata.json (C5).
    notes: Dict[str, object] = field(default_factory=dict)
    #: Step budget. When set, `epochs` must be None and the run ends at the first
    #: of: the stop rule firing, or this HARD CAP.
    max_steps: Optional[int] = None
    #: Step-budgeted runs must STATE this - `None` to run to the cap, or a rule.
    #: Left unstated it is `UNSTATED`, which a step-budgeted run refuses.
    stop_rule: Optional[StopRule] = UNSTATED
    #: Optimizer steps per logged window (required in a step budget): decides the final weights.
    window_steps: Optional[int] = UNSTATED
    #: Write the resumable `state.pt` every this many windows (about 3x the weights on disk).
    state_every_windows: int = 1

    def __post_init__(self):
        if (self.epochs is None) == (self.max_steps is None):
            raise TrainError(
                "state exactly ONE budget: epochs=%r, max_steps=%r. An epoch is a "
                "dataset-dependent unit; a step budget is the comparable one."
                % (self.epochs, self.max_steps))
        if self.max_steps is None:
            if self.stop_rule not in (None, UNSTATED):
                raise TrainError("a stop rule needs a step budget (max_steps) as its cap")
            if self.window_steps not in (None, UNSTATED):
                raise TrainError("window_steps applies to a step budget (max_steps) only")
            self.stop_rule, self.window_steps = None, None
        else:
            unstated = [k for k in ("stop_rule", "window_steps")
                        if getattr(self, k) is UNSTATED]
            if unstated:
                raise TrainError(
                    "a step-budgeted run must STATE %s (stop_rule=None means run to "
                    "the cap). Both decide which weights the run ends on, so neither "
                    "may arrive as a loop default (TR28)." % " and ".join(unstated))
            if int(self.window_steps) < 1:
                raise TrainError("window_steps must be >= 1")


#: EVERY TrainConfig field, classified (test-enforced): model, run, environment or plumbing.
# why: see DESIGN_NOTES.md#train-config-fields
TRAIN_CONFIG_FIELDS: Dict[str, Tuple[str, str]] = dict(
    lr=("model", "optimizer step size"),
    weight_decay=("model", "AdamW decoupled weight decay"),
    optimizer=("model", "optimizer family"),
    grad_clip=("model", "global-norm gradient clip; also a LADDER INVARIANT"),
    epochs=("run", "budget; exactly one of epochs / max_steps"),
    max_steps=("run", "budget; exactly one of epochs / max_steps"),
    batch_size=("run", "batch size; changes the gradient noise and the step count"),
    seed=("run", "initial weights (via seeded_build), data order, dropout and "
                 "reparameterisation streams"),
    stop_rule=("run", "step budgets: when training ends, so which weights are final"),
    window_steps=("run", "step budgets: stop-rule granularity and selection cadence"),
    num_workers=("environment", "loader worker count; order comes from the main-"
                                "process generator and the dataset draws no randomness"),
    prefer_cuda=("environment", "device; recorded as metadata.device"),
    strict_determinism=("environment", "deterministic kernels; recorded as "
                                       "metadata.determinism"),
    log_every=("plumbing", "console printing"),
    checkpoint_every=("plumbing", "extra checkpoint files"),
    run_name=("plumbing", "run directory name"),
    notes=("plumbing", "free-form provenance written to metadata"),
    state_every_windows=("plumbing", "resumable state.pt cadence; resume is bit-exact"),
)

#: The fields a model must declare - exactly these, no more, no fewer.
MODEL_DECLARED_FIELDS = tuple(k for k, (c, _) in TRAIN_CONFIG_FIELDS.items()
                              if c == "model")


def _ladder_invariants() -> dict:
    """Values that must be IDENTICAL for every model in the ladder, because a
    difference would sit inside an RQ2/RQ3 comparison. One source each."""
    from g1_model.models import LADDER_GRAD_CLIP
    return dict(grad_clip=float(LADDER_GRAD_CLIP))


class OptimizerSourceError(TrainError):
    """A TrainConfig whose hyperparameters are not the model's own, or a bad declaration (TR28)."""


def assert_optimizer_source(model: nn.Module, cfg: TrainConfig) -> None:
    """The model's DECLARED optimizer settings must be the ones being used (TR28)."""
    # why: see DESIGN_NOTES.md#train-optimizer-source
    declared = getattr(model, "optimizer_config", None)
    want = declared() if declared is not None else None
    if want is None:
        raise OptimizerSourceError(
            "%s declares no training hyperparameters (optimizer_config). The loop "
            "would have to supply %s from its own config - which is TR28. Build "
            "the model through its factory from its config (build_bc(cfg=...), "
            "build_act(cfg=...))." % (type(model).__name__, list(MODEL_DECLARED_FIELDS)))
    missing = sorted(set(MODEL_DECLARED_FIELDS) - set(want))
    extra = sorted(set(want) - set(MODEL_DECLARED_FIELDS))
    if missing or extra:
        raise OptimizerSourceError(
            "%s declares an incomplete or unknown set of training hyperparameters: "
            "missing %s, unknown %s. A partial declaration leaves the rest to the "
            "loop's config, which is TR28 again." % (type(model).__name__, missing, extra))
    for k, v in _ladder_invariants().items():
        if abs(float(want[k]) - v) > 1e-12:
            raise OptimizerSourceError(
                "%s declares %s=%r, but every model in the ladder must use %r "
                "(ladder invariant; see models.GRAD_CLIP_DISCLOSURE). Change it for "
                "EVERY model or for none." % (type(model).__name__, k, want[k], v))
    got = {k: getattr(cfg, k) for k in MODEL_DECLARED_FIELDS}
    bad = {}
    for k, w in want.items():
        g = got[k]
        differs = (abs(float(w) - float(g)) > 1e-12 if isinstance(w, float)
                   else w != g)
        if differs:
            bad[k] = (w, g)
    if bad:
        lines = ["  %-13s model declares %r, TrainConfig has %r" % (k, w, g)
                 for k, (w, g) in sorted(bad.items())]
        raise OptimizerSourceError(
            "%s declares optimizer settings the TrainConfig does not carry, so "
            "the run would use another model's hyperparameters:" % type(model).__name__
            + "".join("\n" + L for L in lines)
            + "\n  Build the TrainConfig from THIS model's config. The loop is "
              "shared; the hyperparameter source is not (TR28).")


def _training_provenance(model: nn.Module, opt: torch.optim.Optimizer) -> dict:
    """Where every training hyperparameter of this run came from, for metadata.
    Written by every run, so the grad-clip deviation travels with the numbers."""
    from g1_model.models import GRAD_CLIP_DISCLOSURE
    return dict(
        source="declared by the model (optimizer_config); checked against the "
               "TrainConfig by assert_optimizer_source before the first step",
        declared=model.optimizer_config(),
        ladder_invariants=_ladder_invariants(),
        optimizer_defaults={k: v for k, v in opt.defaults.items()
                            if k not in MODEL_DECLARED_FIELDS},
        field_policy={k: c for k, (c, _) in TRAIN_CONFIG_FIELDS.items()},
        disclosed_deviations=dict(grad_clip=GRAD_CLIP_DISCLOSURE))


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    assert_optimizer_source(model, cfg)
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay)
    raise TrainError("unknown optimizer %r" % cfg.optimizer)


def make_run_dir(cfg: TrainConfig, root: str = RUNS) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, "%s_%s" % (stamp, cfg.run_name))
    os.makedirs(path, exist_ok=True)
    return path


def dataset_provenance(ds: ChunkDataset) -> dict:
    """WHICH DATA this ran on (C5), written into every run's metadata."""
    # why: see DESIGN_NOTES.md#train-dataset-provenance
    sources = sorted({str(s) for s in ds.sources}) or ["unknown"]
    real = bool(ds.real_demonstration) and all(ds.real_demonstration)
    return dict(
        episode_sources=sources,
        episodes=len(ds.lengths), seeds=list(ds.seeds), ticks=int(sum(ds.lengths)),
        samples=len(ds),
        tracking_measured_episodes=int(sum(ds.tracking_is_measured)),
        real_demonstrations=bool(real),
        caveat=(
            "SCRIPTED demonstrator data. It is smoother and more consistent "
            "than piloted teleoperation will be, and tracking_ok cannot be "
            "exercised by episodes recorded without a camera. NO GATE RESULT "
            "FROM THIS RUN MAY BE CITED without a re-run on piloted data."
            if not real else
            "Real teleoperated demonstrations."))


#: What best.pt is chosen on and what the gate scores: ONE quantity.
# why: see DESIGN_NOTES.md#train-selection
SELECTION_CRITERION = ("deployment reconstruction (score_deployment): validation "
                       "set if given, else the training set")


class ScoringError(TrainError):
    """Scoring reached something the deployed policy never runs or never has."""


# ─── the deployment guard ─────────────────────────────────────────────────────
# WARNING: every check is STRUCTURAL, on what the model does: a leak moves the score only 1.0e-7.
# why: see DESIGN_NOTES.md#train-deployment-guard (the attack table)

#: What every nn.Module keeps in its own __dict__: bookkeeping, never model state.
_MODULE_INTERNALS = frozenset(vars(nn.Module()).keys()) | {"forward"}


def _holds_foreign_array(v, own: set, depth: int = 0) -> bool:
    """True if `v` is or holds a tensor outside the model's own (or an alias of one), or any numpy array."""
    if torch.is_tensor(v):
        return id(v) not in own
    if isinstance(v, np.ndarray):
        return True
    if depth >= 3:
        return False
    if isinstance(v, dict):
        return any(_holds_foreign_array(x, own, depth + 1) for x in v.values())
    if isinstance(v, (list, tuple, set, frozenset)):
        return any(_holds_foreign_array(x, own, depth + 1) for x in v)
    return False


def hidden_state(model: nn.Module) -> List[str]:
    """Every tensor or array a module holds OUTSIDE parameters and buffers; empty if stateless."""
    own = {id(t) for t in model.parameters()} | {id(t) for t in model.buffers()}
    found = []
    for name, mod in model.named_modules():
        for k, v in vars(mod).items():
            if k not in _MODULE_INTERNALS and _holds_foreign_array(v, own):
                found.append("%s.%s" % (name or type(model).__name__, k))
    return found


def _weights_fingerprint(model: nn.Module) -> dict:
    """Identity, storage and in-place version counter of every parameter and
    buffer: any mutation during scoring changes one of them, at no copying cost."""
    out = {}
    named = (list(model.named_parameters(remove_duplicate=False))
             + list(model.named_buffers(remove_duplicate=False)))
    for n, t in named:
        out[n] = (id(t), t.data_ptr(), t._version, tuple(t.shape), str(t.dtype))
    return out


def _rng_snapshot() -> dict:
    snap = dict(torch=torch.get_rng_state().clone(), numpy=np.random.get_state(),
                python=random.getstate())
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        snap["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return snap


def _rng_advanced(a: dict, b: dict) -> List[str]:
    moved = []
    if not torch.equal(a["torch"], b["torch"]):
        moved.append("torch CPU")
    na, nb = a["numpy"], b["numpy"]
    if not (na[0] == nb[0] and np.array_equal(na[1], nb[1]) and tuple(na[2:]) == tuple(nb[2:])):
        moved.append("numpy")
    if a["python"] != b["python"]:
        moved.append("python random")
    if "cuda" in a and "cuda" in b:
        if len(a["cuda"]) != len(b["cuda"]) or any(
                not torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])):
            moved.append("CUDA")
    return moved


class _DeploymentGuard:
    """Eval mode, enforced; the encoder unreachable; z verified; no randomness; no state."""
    # Entered by `score_deployment`, which also runs `finish`; every change is undone on exit.

    def __init__(self, model: nn.Module):
        self.model = model
        self.declared = tuple(getattr(model, "TARGET_READING_MODULES", ()))
        self.prior = tuple(getattr(model, "PRIOR_LATENT_MODULES", ()))
        name = type(model).__name__
        if hasattr(model, "encode") and not self.declared:
            raise ScoringError(
                "%s has an `encode` method but declares no TARGET_READING_MODULES; "
                "scoring cannot prove its encoder is unreachable" % name)
        if hasattr(model, "encode") and not self.prior:
            raise ScoringError(
                "%s has an `encode` method but declares no PRIOR_LATENT_MODULES; "
                "scoring cannot verify that z is the prior mean" % name)
        self.z_calls = 0
        self._z_at_call = None
        self._handles, self._patched = [], []

    # -- helpers -------------------------------------------------------------
    def _patch(self, mod: nn.Module, make):
        had = "forward" in vars(mod)
        old = vars(mod).get("forward")
        mod.forward = make(mod.forward)
        self._patched.append((mod, had, old))

    def _forbidden(self, name):
        def make(_orig):
            def forward(*_a, **_k):
                raise ScoringError(
                    "scoring reached %r, which reads the target action chunk; the "
                    "deployed policy never runs it (called via __call__ or "
                    ".forward() - both are refused)" % name)
            return forward
        return make

    def _prior_check(self, name):
        def make(orig):
            def forward(*a, **k):
                z = a[0] if a else next(iter(k.values()), None)
                if not torch.is_tensor(z):
                    raise ScoringError("%r was called without a tensor latent" % name)
                if z.numel() and bool((z != 0).any()):
                    raise ScoringError(
                        "the latent entering %r is not the prior mean: max|z| = %.3e. "
                        "The deployed ACT uses z = 0 (detr_vae.py:113; paper §IV-B)"
                        % (name, float(z.detach().abs().nan_to_num(float("inf")).max())))
                self.z_calls += 1
                return orig(*a, **k)
            return forward
        return make

    # -- context -------------------------------------------------------------
    def __enter__(self):
        m = self.model
        held = hidden_state(m)
        if held:
            raise ScoringError(
                "%s holds state outside its parameters and buffers: %s. The deployed "
                "function must be a function of (weights, observation) alone - a "
                "leftover recurrent hidden state or a cached target would reach the "
                "gate, and would make it score BETTER. A recurrent model needs an "
                "explicit per-episode reset protocol, which score_deployment does not "
                "implement." % (type(m).__name__, held))
        self.was_training = m.training
        self.fp0 = _weights_fingerprint(m)
        self.rng0 = _rng_snapshot()
        try:
            m.eval()

            def _no_train_mode(mod, _inp):
                if mod.training:
                    raise ScoringError(
                        "%s ran in TRAIN mode during scoring; the gate scores the "
                        "deployed function only" % type(mod).__name__)
            for mod in m.modules():
                self._handles.append(mod.register_forward_pre_hook(_no_train_mode))
            seen = set()
            for name in self.declared:
                for sub in getattr(m, name).modules():
                    if id(sub) not in seen:
                        seen.add(id(sub))
                        self._patch(sub, self._forbidden(name))
            for name in self.prior:
                self._patch(getattr(m, name), self._prior_check(name))
            if self.prior:
                def _pre(_mod, _inp):
                    self._z_at_call = self.z_calls

                def _post(_mod, _inp, _out):
                    if self.z_calls <= self._z_at_call:
                        raise ScoringError(
                            "a forward pass of %s returned without passing a latent "
                            "through %s, so z = 0 could not be verified"
                            % (type(m).__name__, list(self.prior)))
                self._handles.append(m.register_forward_pre_hook(_pre))
                self._handles.append(m.register_forward_hook(_post))
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        for mod, had, old in reversed(self._patched):
            if had:
                mod.forward = old
            else:
                del mod.forward
        self._patched = []
        self.model.train(self.was_training)

    def __exit__(self, *exc):
        self._restore()
        return False

    # -- the checks that need the finished pass --------------------------------
    def finish(self, probe_obs, probe_out):
        """Called inside the context, after the scoring pass. Order matters: the
        NaN probe writes to parameters, so the fingerprint is checked first."""
        m = self.model
        with torch.no_grad():
            again = m(probe_obs)
        if not torch.equal(again, probe_out):
            raise ScoringError(
                "the first batch scored differently after the full pass (max |diff| "
                "%.3e): the deployed function carried state between calls, or is not "
                "deterministic. An un-reset recurrent hidden state does exactly this."
                % float((again - probe_out).abs().max()))
        held = hidden_state(m)
        if held:
            raise ScoringError("scoring left state outside parameters and buffers on "
                               "%s: %s" % (type(m).__name__, held))
        fp = _weights_fingerprint(m)
        if fp != self.fp0:
            changed = sorted(k for k in set(fp) | set(self.fp0)
                             if fp.get(k) != self.fp0.get(k))
            raise ScoringError(
                "parameters or buffers changed during scoring: %s. The gate scores "
                "FIXED weights; a buffer updated per call is carried state."
                % changed[:8])
        moved = _rng_advanced(self.rng0, _rng_snapshot())
        if moved:
            raise ScoringError(
                "scoring drew randomness (%s RNG advanced): the deployed function "
                "must be deterministic - dropout active, a sampled latent, or any "
                "other stochastic op" % ", ".join(moved))
        if self.declared:
            self._nan_probe(probe_obs, probe_out)

    def _nan_probe(self, probe_obs, probe_out):
        """Fill the target-reading modules' weights with NaN, re-score: the output must not move."""
        m = self.model
        params, seen = [], set()
        for name in self.declared:
            for p in list(getattr(m, name).parameters()) + list(getattr(m, name).buffers()):
                if id(p) not in seen and p.is_floating_point():
                    seen.add(id(p))
                    params.append(p)
        saved = [p.detach().to("cpu", copy=True) for p in params]
        try:
            with torch.no_grad():
                for p in params:
                    p.fill_(float("nan"))
                poisoned = m(probe_obs)
        finally:
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s.to(p.device))
        if not torch.equal(poisoned, probe_out):
            raise ScoringError(
                "the deployed output depends on the weights of %s (target-reading "
                "modules): with them set to NaN the probe batch changed. The deployed "
                "policy must not use the CVAE encoder by any route." % list(self.declared))


def score_deployment(model: nn.Module, ds: ChunkDataset,
                     device: Optional[torch.device] = None,
                     dim_mask: Optional[torch.Tensor] = None,
                     batch_size: int = 256):
    """THE gated number: masked-L1 reconstruction of the DEPLOYED function, as a `Quantity`."""
    # WARNING: `model(obs)` alone: `forward_loss` and `loss_terms` feed the target to the encoder.
    # why: see DESIGN_NOTES.md#train-score-deployment
    from g1_model.ambiguity import Quantity, RECON_UNITS, MEASURED_AT_DEPLOYMENT
    batch_size = int(getattr(ds, "score_batch_size", batch_size))
    device = device or next(model.parameters()).device
    if dim_mask is None:
        dim_mask = torch.from_numpy(
            np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(device)
    total, n = 0.0, 0
    probe_obs = probe_out = None
    with _DeploymentGuard(model) as guard, torch.no_grad():
        for i in range(0, len(ds), int(batch_size)):
            batch = _collate_for(ds)([ds[j] for j in range(i, min(i + int(batch_size), len(ds)))])
            obs = batch["obs"].to(device, non_blocking=True)
            pred = model(obs)                        # the observation, and nothing else
            if probe_obs is None:
                probe_obs, probe_out = obs, pred.clone()
            target = batch["action"].to(device, non_blocking=True)
            pad = batch["action_mask"].to(device, non_blocking=True)
            l1, count = masked_l1(pred, target, pad, dim_mask)
            total += float(l1) * int(count)
            n += int(count)
        if probe_obs is not None:
            guard.finish(probe_obs, probe_out)
    if getattr(ds, "empty_cache_after_scoring", False) and torch.device(device).type == "cuda":
        del probe_obs, probe_out
        torch.cuda.empty_cache()
    return Quantity(total / max(n, 1), RECON_UNITS, measured=MEASURED_AT_DEPLOYMENT)


def evaluate(model: nn.Module, ds: ChunkDataset, device: torch.device,
             dim_mask: torch.Tensor) -> float:
    """The ONE evaluation path: reconstruction of the deployed function, for every model."""
    return float(score_deployment(model, ds, device, dim_mask).value)


# ─── survivability: atomic writes, resumable state ────────────────────────────
def _atomic_json(path: str, obj) -> None:
    """Write-then-rename, so a kill mid-write leaves the PREVIOUS file intact
    rather than a truncated one. `os.replace` is atomic on the same volume."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


#: Training settings that must be IDENTICAL between a run and its resumption.
#: `max_steps` is deliberately absent: extending a cap is a legitimate resume.
RESUME_MUST_MATCH = ("batch_size", "lr", "weight_decay", "optimizer", "grad_clip",
                     "seed", "window_steps", "strict_determinism", "stop_rule")


@dataclass(frozen=True)
class Resume:
    """Where a step-budgeted run continues from: a full `state.pt` (bit-exact) or weights only.

    why: see DESIGN_NOTES.md#train-resume
    """

    path: str
    prior_run_dir: Optional[str] = None


def _norm(x):
    """JSON round trip, so a tuple and a list, or a dataclass and its dict,
    compare equal when they carry the same values."""
    return json.loads(json.dumps(x, default=str))


def _check_resume_config(saved: dict, cfg: TrainConfig, where: str) -> None:
    now = asdict(cfg)
    bad = {k: (saved.get(k), now.get(k)) for k in RESUME_MUST_MATCH
           if _norm(saved.get(k)) != _norm(now.get(k))}
    if bad:
        raise TrainError(
            "refusing to resume %s under a different training configuration - the "
            "two segments would not be one run:\n" % where
            + "\n".join("  %-18s saved %r, now %r" % (k, a, b)
                        for k, (a, b) in sorted(bad.items())))


def _load_resume(resume: Resume, model: nn.Module, opt, cfg: TrainConfig) -> tuple:
    """(resume_info, state). Loads weights (and, for a full state, the optimizer)
    into `model` / `opt`; everything else is returned for the loop to apply."""
    ck = torch.load(resume.path, map_location="cpu", weights_only=False)
    spec.assert_spec_version(ck["spec_version"], where=os.path.basename(resume.path))
    if ck.get("model_cls") != type(model).__name__:
        raise TrainError("checkpoint holds a %s, this run builds a %s"
                         % (ck.get("model_cls"), type(model).__name__))
    if _norm(ck.get("model_kwargs")) != _norm(getattr(model, "hparams", {})):
        raise TrainError("checkpoint architecture %r differs from this run's %r"
                         % (ck.get("model_kwargs"), getattr(model, "hparams", {})))
    model.load_state_dict(ck["state_dict"])

    if "optimizer" in ck:                                       # FULL STATE
        _check_resume_config(ck["config"], cfg, resume.path)
        opt.load_state_dict(ck["optimizer"])
        same = ck.get("selection") == SELECTION_CRITERION
        state = dict(history=ck["history"],
                     best=float(ck["best"]) if same else float("inf"),
                     step=int(ck["step"]), epoch=int(ck["epoch"]),
                     batches_done=int(ck["batches_done"]),
                     epoch_gen_state=ck["epoch_gen_state"], rng=ck["rng"],
                     elapsed=float(ck["elapsed_seconds"]))
        info = dict(kind="full_state", source=resume.path, resumed_at_step=state["step"],
                    restored=["model weights", "optimizer state (AdamW moments, step)",
                              "RNG: torch CPU, torch CUDA, numpy, python",
                              "data-order generator and position within the pass",
                              "history, best value, elapsed time"],
                    not_restored=[] if same else [
                        "best value: the prior run selected best.pt on %r, not %r; "
                        "selection restarts, and the first new window writes best.pt"
                        % (ck.get("selection") or "train-mode window mean",
                           SELECTION_CRITERION)],
                    scheduler="none exists in this loop",
                    prior_windows=len(state["history"]))
        return info, state

    # WEIGHTS ONLY: identify the step from the prior run's own records
    prior = resume.prior_run_dir or os.path.dirname(os.path.abspath(resume.path))
    with open(os.path.join(prior, "metadata.json"), encoding="utf-8") as fh:
        prior_meta = json.load(fh)
    _check_resume_config(prior_meta["config"], cfg, resume.path)
    rows = [json.loads(L) for L in open(os.path.join(prior, "metrics.jsonl"),
                                        encoding="utf-8") if L.strip()]
    # An EXACT match of the stored total loss identifies the window the weights are from.
    match = [r for r in rows if r.get("train_loss") == ck.get("train_loss")]
    if len(match) != 1 or "step" not in match[0]:
        raise TrainError(
            "cannot identify which step %s holds: %d metrics rows match its stored "
            "loss %r. Resume from a full state file instead."
            % (resume.path, len(match), ck.get("train_loss")))
    at = match[0]
    history = [dict(r) for r in rows if r["step"] <= at["step"]]
    # Selection quantity only (val, else the deployment score); older rows have neither: restart.
    watch = [r["val_loss"] if r.get("val_loss") is not None else r.get("eval_recon_l1")
             for r in history]
    watch = [w for w in watch if w is not None]
    state = dict(history=history, best=float(min(watch)) if watch else float("inf"),
                 step=int(at["step"]),
                 epoch=int(at["epoch"]), batches_done=None, epoch_gen_state=None,
                 rng=None, elapsed=float(at["elapsed_seconds"]))
    info = dict(
        kind="weights_only", source=resume.path, prior_run_dir=prior,
        resumed_at_step=state["step"], prior_windows=len(history),
        identified_by="exact match of the checkpoint's stored total loss to the "
                      "step-%d metrics row" % at["step"],
        restored=["model weights", "history (from the prior metrics.jsonl)",
                  "step and best value", "training config (checked, must match)"],
        not_restored=["optimizer state: AdamW m, v and step count restart at zero, "
                      "so bias correction restarts and early steps are sign-like "
                      "steps of size ~lr - expect a short transient",
                      "RNG streams (dropout, reparameterization, shuffle): reseeded",
                      "position within the data pass: a fresh shuffled pass begins"],
        scheduler="none exists in this loop",
        consequence="the curve continues from the resume point, but it is not the "
                    "curve the uninterrupted run would have drawn")
    return info, state


def train(model: nn.Module, train_ds: ChunkDataset, cfg: TrainConfig,
          val_ds: Optional[ChunkDataset] = None,
          run_dir: Optional[str] = None,
          on_epoch: Optional[Callable[[dict], None]] = None,
          gate_reference=None,
          resume: Optional[Resume] = None) -> dict:
    """Train any model on any ChunkDataset; it knows nothing about BC or ACT.

    WARNING: build the model with `seeded_build`; seeding here is too late for its init.
    why: see DESIGN_NOTES.md#train-loop (resuming: DESIGN_NOTES.md#train-resume)
    """
    det = set_determinism(cfg.seed, strict=cfg.strict_determinism)
    device = select_device(cfg.prefer_cuda)
    dev = device_report(device)
    run_dir = run_dir or make_run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)

    model = model.to(device)
    opt = make_optimizer(model, cfg)
    dim_mask = torch.from_numpy(
        np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(device)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=_collate_for(train_ds), num_workers=cfg.num_workers,
                    generator=g, worker_init_fn=seed_worker, drop_last=False)
    # Validation and selection are scored by `score_deployment` on the dataset, with no DataLoader.
    # why: see DESIGN_NOTES.md#train-score-deployment
    vl = val_ds
    # The training set is scored the same way whenever it is the selection
    # quantity (no validation set) or the gate needs it.
    score_train = val_ds is None or gate_reference is not None
    from g1_model.ambiguity import (Quantity, RECON_UNITS, TOTAL_LOSS_UNITS,
                                    MEASURED_TRAIN_MODE_WINDOW, gate as _gate)
    dep = {"q": None, "step": None}     # latest deployment score of the train set

    base = baselines(train_ds, device)
    prov = dataset_provenance(train_ds)

    meta = dict(
        config=asdict(cfg), device=dev, determinism=det,
        git_commit=_git_commit(), spec_version=spec.SPEC_VERSION,
        started_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        model=dict(cls=type(model).__name__,
                   parameters=int(sum(p.numel() for p in model.parameters())),
                   trainable=int(sum(p.numel() for p in model.parameters()
                                     if p.requires_grad)),
                   repr=str(model)),
        loader=dict(chunk_size=train_ds.cfg.chunk_size,
                    obs_window=train_ds.cfg.obs_window,
                    tracking=asdict(train_ds.cfg.tracking)),
        data=prov, baselines=base,
        val=dict(episodes=len(val_ds.lengths), seeds=list(val_ds.seeds),
                 samples=len(val_ds)) if val_ds is not None else None,
        training_hyperparameters=_training_provenance(model, opt),
    )
    step_mode = cfg.max_steps is not None
    if resume is not None and not step_mode:
        raise TrainError("resume is supported for step-budgeted runs (max_steps)")
    ckpt_every = int(cfg.checkpoint_every or 0)
    if ckpt_every < 0:
        raise TrainError("checkpoint_every must be >= 0, got %d" % ckpt_every)
    if ckpt_every and step_mode and ckpt_every % int(cfg.window_steps):
        raise TrainError(
            "checkpoint_every (%d steps) must be a multiple of window_steps (%d), "
            "so each periodic checkpoint carries its window's losses"
            % (ckpt_every, int(cfg.window_steps)))
    if step_mode:
        meta["budget"] = dict(unit="optimizer_steps", max_steps=int(cfg.max_steps),
                              window_steps=int(cfg.window_steps),
                              stop_rule=(cfg.stop_rule.describe(cfg.window_steps)
                                         if cfg.stop_rule else None))

    history: List[dict] = []
    best = float("inf")
    step, epoch, stop_info = 0, 0, None
    elapsed0 = 0.0
    skip_batches = 0
    rng_to_restore = None
    segment = 1
    resumed_from_step = 0
    meta["selection"] = SELECTION_CRITERION
    if resume is not None:
        info, st = _load_resume(resume, model, opt, cfg)
        meta["resume"] = info
        history, best, step = st["history"], st["best"], st["step"]
        resumed_from_step = step
        elapsed0 = st["elapsed"]
        for r in history:
            r.setdefault("segment", 1)
        segment = max(int(r["segment"]) for r in history) + 1
        if st["epoch_gen_state"] is not None:                   # full state
            g.set_state(st["epoch_gen_state"])
            epoch, skip_batches = st["epoch"] - 1, st["batches_done"]
            rng_to_restore = st["rng"]
        else:                                                   # weights only
            epoch = st["epoch"]

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)

    print("run           %s" % repo_relpath(run_dir))
    print("device        %s%s" % (dev["device"],
                                  ("  %s, %.2f GB VRAM" % (dev["name"], dev["total_vram_gb"]))
                                  if dev["total_vram_gb"] else "  (%s)" % dev["name"]))
    print("determinism   seed %d, deterministic_algorithms=%s%s"
          % (det["seed"], det["deterministic_algorithms"],
             ("  NOTES: " + "; ".join(det["notes"])) if det["notes"] else ""))
    print("model         %s, %d parameters"
          % (meta["model"]["cls"], meta["model"]["parameters"]))
    print("data          %d episodes, %d samples, K=%d W_o=%d"
          % (prov["episodes"], prov["samples"], train_ds.cfg.chunk_size,
             train_ds.cfg.obs_window))
    print("              SOURCE: %s" % ", ".join(prov["episode_sources"]))
    print("              %s" % prov["caveat"])
    print("baselines     zero %.6f   copy %.6f   (normalized L1, trainable dims)"
          % (base["zero"], base["copy"]))
    if step_mode:
        print("budget        %d optimizer steps (hard cap)%s"
              % (cfg.max_steps, "; " + meta["budget"]["stop_rule"]
                 if cfg.stop_rule else ""))
    if resume is not None:
        print("RESUME        %s from step %d (%s); %d prior windows copied in"
              % (meta["resume"]["kind"], step, repo_relpath(resume.path),
                 len(history)))
        for s_ in meta["resume"]["not_restored"]:
            print("              NOT restored: %s" % s_)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if history:
        with open(metrics_path, "w", encoding="utf-8") as fh:
            for r in history:
                fh.write(json.dumps(r) + "\n")
    t0 = time.perf_counter()
    status = {"value": "running"}
    counters = {"windows_since_state": 0}

    def _elapsed():
        return elapsed0 + (time.perf_counter() - t0)

    def _write_progress(final: bool = False, error: Optional[str] = None):
        """result.json and gate.json as of NOW: a kill costs at most one window."""
        if not history:
            return
        last = history[-1]
        seg_s = time.perf_counter() - t0
        doc = dict(status=status["value"], final=final, error=error,
                   run_dir=run_dir, optimizer_steps=int(step), segment=segment,
                   resume=meta.get("resume"), stop=stop_info,
                   final_train_loss=last["train_loss"],
                   final_recon_l1=last["recon_l1"],
                   final_val_loss=last.get("val_loss"),
                   best=best, baselines=base,
                   segment_steps_per_second=round((step - resumed_from_step)
                                                  / max(seg_s, 1e-9), 3),
                   wall_seconds=round(_elapsed(), 3),
                   written_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(
                       timespec="seconds"),
                   history=history)
        _atomic_json(os.path.join(run_dir, "result.json"), doc)
        if gate_reference is not None and dep["q"] is not None:
            q = dep["q"]
            v = _gate(q, gate_reference)
            _atomic_json(os.path.join(run_dir, "gate.json"), dict(
                status=status["value"], final=final, error=error,
                note=(None if final else
                      "PROVISIONAL: the verdict at the step the run was last "
                      "alive, not a final verdict"),
                passed=v.passed, train_error=v.train_error,
                train_error_units=q.units, train_error_measured=q.measured,
                scored_weights=("final weights" if final else
                                "weights at the last completed window"),
                scored_at_step=dep["step"],
                train_mode_window_recon=dict(
                    value=last["recon_l1"], measured=MEASURED_TRAIN_MODE_WINDOW,
                    note="logged for comparison; NOT gated"),
                total_loss=last["train_loss"],
                reference=v.reference, ratio=v.ratio, at_step=int(step),
                optimizer_steps=int(step), stop=stop_info,
                steps_per_second=doc["segment_steps_per_second"],
                detail=gate_reference.as_metadata()))

    def _save_state(batches_done: int, epoch_gen_state):
        rng = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                   python=random.getstate(),
                   cuda=(torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None))
        _atomic_torch_save(dict(
            state_dict=model.state_dict(), optimizer=opt.state_dict(),
            scheduler=None,          # no LR scheduler exists in this loop
            config=asdict(cfg), model_cls=type(model).__name__,
            model_kwargs=getattr(model, "hparams", {}),
            spec_version=spec.SPEC_VERSION, git_commit=meta["git_commit"],
            step=int(step), epoch=int(epoch), batches_done=int(batches_done),
            epoch_gen_state=epoch_gen_state, rng=rng, history=history,
            best=float(best), selection=SELECTION_CRITERION,
            elapsed_seconds=float(_elapsed())),
            os.path.join(run_dir, "state.pt"))

    w_tot, w_rec, w_n, w_ext, w_t0 = 0.0, 0.0, 0, {}, time.perf_counter()

    def _window_row():
        """Close the current window of optimizer steps into one logged row."""
        nonlocal w_tot, w_rec, w_n, w_ext, w_t0, best
        secs = time.perf_counter() - w_t0
        tl_, rl_ = w_tot / max(w_n, 1), w_rec / max(w_n, 1)
        val = evaluate(model, vl, device, dim_mask) if vl is not None else None
        if score_train:
            dep["q"], dep["step"] = score_deployment(model, train_ds, device, dim_mask), step
        model.train()
        row = dict(step=step, epoch=epoch, train_loss=tl_, recon_l1=rl_,
                   eval_recon_l1=float(dep["q"].value) if score_train else None,
                   **{("train_" + k_): v_ / max(w_n, 1) for k_, v_ in w_ext.items()},
                   val_loss=val, lr=float(opt.param_groups[0]["lr"]),
                   window_seconds=round(secs, 3),
                   elapsed_seconds=round(_elapsed(), 3),
                   segment=segment, git_commit=meta["git_commit"])
        history.append(row)
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        watch = val if val is not None else row["eval_recon_l1"]
        if watch < best:
            best = watch
            save_checkpoint(os.path.join(run_dir, "best.pt"), model, cfg, meta,
                            epoch, tl_, val, step=step, selected_on=watch)
        if cfg.log_every:
            kl = row.get("train_kl")
            print("  step %8d  recon %.6f  total %.6f%s  %.1fs"
                  % (step, rl_, tl_, ("  kl %.5f" % kl) if kl is not None else "",
                     secs))
        w_tot, w_rec, w_n, w_ext, w_t0 = 0.0, 0.0, 0, {}, time.perf_counter()

    if rng_to_restore is not None:
        # WARNING: LAST thing before the first step (baselines draws from the global generator).
        torch.set_rng_state(rng_to_restore["torch"])
        np.random.set_state(rng_to_restore["numpy"])
        random.setstate(rng_to_restore["python"])
        if rng_to_restore.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_to_restore["cuda"])

    try:
        while True:
            if not step_mode and epoch >= int(cfg.epochs):
                break
            epoch += 1
            model.train()
            e0 = time.perf_counter()
            total, recon, n = 0.0, 0.0, 0
            extra_sums = {}
            epoch_gen_state = g.get_state()      # before the iterator draws from g
            batches_done, skip = 0, skip_batches
            skip_batches = 0
            for batch in tl:
                if skip:
                    # Full-state resume: pass over the batches already trained on.
                    skip -= 1
                    batches_done += 1
                    continue
                obs = batch["obs"].to(device, non_blocking=True)
                target = batch["action"].to(device, non_blocking=True)
                pad = batch["action_mask"].to(device, non_blocking=True)
                loss, l1, count, extras = forward_loss(model, obs, target, pad, dim_mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                opt.step()
                total += float(loss.detach()) * int(count)
                recon += float(l1.detach()) * int(count)
                n += int(count)
                for k_, v_ in extras.items():
                    extra_sums[k_] = extra_sums.get(k_, 0.0) + float(v_) * int(count)
                batches_done += 1
                step += 1
                if not step_mode:
                    continue
                # ---- step-budgeted: windows, the stop rule, the hard cap -------
                w_tot += float(loss.detach()) * int(count)
                w_rec += float(l1.detach()) * int(count)
                w_n += int(count)
                for k_, v_ in extras.items():
                    w_ext[k_] = w_ext.get(k_, 0.0) + float(v_) * int(count)
                if step % int(cfg.window_steps) == 0:
                    _window_row()
                    if ckpt_every and step % ckpt_every == 0:
                        save_checkpoint(
                            os.path.join(run_dir, "ckpt_step%07d.pt" % step),
                            model, cfg, meta, epoch, history[-1]["train_loss"],
                            history[-1]["val_loss"], step=step)
                    if cfg.stop_rule is not None:
                        fired = cfg.stop_rule.check(history)
                        if fired:
                            stop_info = dict(reason="stop_rule", step=step, **fired)
                    counters["windows_since_state"] += 1
                    if counters["windows_since_state"] >= int(cfg.state_every_windows):
                        _save_state(batches_done, epoch_gen_state)
                        counters["windows_since_state"] = 0
                    _write_progress()
                    if stop_info is not None:
                        break
                if step >= int(cfg.max_steps):
                    if w_n:
                        _window_row()
                    stop_info = dict(reason="hard_cap", step=step)
                    _save_state(batches_done, epoch_gen_state)
                    break
            if step_mode:
                if stop_info is not None:
                    break
                continue
            train_loss = total / max(n, 1)
            recon_loss = recon / max(n, 1)
            extra_means = {k_: v_ / max(n, 1) for k_, v_ in extra_sums.items()}
            val_loss = evaluate(model, vl, device, dim_mask) if vl is not None else None
            if score_train:
                dep["q"], dep["step"] = (score_deployment(model, train_ds, device, dim_mask),
                                         step)

            row = dict(epoch=epoch, train_loss=train_loss, recon_l1=recon_loss,
                       eval_recon_l1=float(dep["q"].value) if score_train else None,
                       **{("train_" + k_): v_ for k_, v_ in extra_means.items()},
                       val_loss=val_loss,
                       lr=float(opt.param_groups[0]["lr"]),
                       epoch_seconds=round(time.perf_counter() - e0, 3),
                       elapsed_seconds=round(time.perf_counter() - t0, 3),
                       # TR29: baselines are reconstruction quantities.
                       frac_of_zero_baseline=recon_loss / base["zero"] if base["zero"] else None,
                       frac_of_copy_baseline=recon_loss / base["copy"] if base["copy"] else None,
                       git_commit=meta["git_commit"])
            history.append(row)
            with open(metrics_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            watch = val_loss if val_loss is not None else row["eval_recon_l1"]
            if watch < best:
                best = watch
                save_checkpoint(os.path.join(run_dir, "best.pt"), model, cfg, meta,
                                epoch, train_loss, val_loss, selected_on=watch)
            if ckpt_every and epoch % ckpt_every == 0:
                save_checkpoint(os.path.join(run_dir, "ckpt_epoch%04d.pt" % epoch),
                                model, cfg, meta, epoch, train_loss, val_loss,
                                step=step)
            if cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.epochs):
                print("  epoch %4d  train %.6f%s  (%.3fx zero, %.3fx copy)  %.1fs"
                      % (epoch, train_loss,
                         "  val %.6f" % val_loss if val_loss is not None else "",
                         row["frac_of_zero_baseline"], row["frac_of_copy_baseline"],
                         row["epoch_seconds"]))
            _write_progress()
            if on_epoch:
                on_epoch(row)
    except BaseException as e:                                  # noqa: BLE001
        # Record WHY and WHERE before dying (a hard OS kill leaves the last window's files instead).
        status["value"] = "crashed"
        msg = str(e).splitlines()[0] if str(e) else ""
        try:
            _write_progress(final=False, error="%s: %s" % (type(e).__name__, msg))
        finally:
            raise

    status["value"] = (stop_info["reason"] if stop_info else "finished")
    save_checkpoint(os.path.join(run_dir, "last.pt"), model, cfg, meta,
                    len(history), history[-1]["train_loss"],
                    history[-1]["val_loss"], step=step if step_mode else None)
    # THE GATED NUMBER: one pass, after training, on the FINAL weights - the
    # weights in last.pt - through the deployed function.
    dep["q"], dep["step"] = score_deployment(model, train_ds, device, dim_mask), step
    wall = round(time.perf_counter() - t0, 3)
    seg_steps = step - resumed_from_step
    result = dict(run_dir=run_dir, history=history, baselines=base,
                  metadata=meta, best=best,
                  final_train_loss=history[-1]["train_loss"],
                  final_recon_l1=history[-1]["recon_l1"],
                  final_deployment_recon=float(dep["q"].value),
                  final_val_loss=history[-1]["val_loss"],
                  # Labelled AT THE SOURCE (TR29): only this loop knows which number is which.
                  quantities=dict(
                      # the gate accepts ONLY this one
                      deployment_recon=dep["q"],
                      recon_l1=Quantity(float(history[-1]["recon_l1"]), RECON_UNITS,
                                        measured=MEASURED_TRAIN_MODE_WINDOW),
                      train_loss=Quantity(float(history[-1]["train_loss"]),
                                          TOTAL_LOSS_UNITS,
                                          measured=MEASURED_TRAIN_MODE_WINDOW)),
                  optimizer_steps=int(step),
                  segment_steps=int(seg_steps),
                  steps_per_second=round(seg_steps / wall, 3) if wall else None,
                  stop=stop_info,
                  wall_seconds=wall,
                  total_wall_seconds=round(elapsed0 + wall, 3))
    _write_progress(final=True)
    # the richer final record, overwriting the progress version
    _atomic_json(os.path.join(run_dir, "result.json"),
                 dict({k: v for k, v in result.items() if k != "metadata"},
                      status=status["value"], final=True))
    return result


# ─── checkpoints ──────────────────────────────────────────────────────────────
def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, meta: dict,
                    epoch: int, train_loss: float,
                    val_loss: Optional[float], step: Optional[int] = None,
                    selected_on: Optional[float] = None) -> str:
    """Weights and provenance, NOT a resumable state (`train()` writes that, as `state.pt`)."""
    # why: see DESIGN_NOTES.md#train-checkpoints
    torch.save(dict(state_dict=model.state_dict(), config=asdict(cfg),
                    model_cls=type(model).__name__,
                    model_kwargs=getattr(model, "hparams", {}),
                    spec_version=spec.SPEC_VERSION,
                    git_commit=meta.get("git_commit"),
                    data=meta.get("data"), baselines=meta.get("baselines"),
                    epoch=int(epoch), train_loss=float(train_loss),
                    val_loss=None if val_loss is None else float(val_loss),
                    step=None if step is None else int(step),
                    selection=None if selected_on is None else dict(
                        criterion=SELECTION_CRITERION, value=float(selected_on))),
               path)
    return path


def load_checkpoint(path: str, model_factory: Callable[..., nn.Module]) -> tuple:
    """(model, checkpoint). Refuses another spec version: the dim layout may differ."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec.assert_spec_version(ck["spec_version"], where=os.path.basename(path))
    model = model_factory(**(ck.get("model_kwargs") or {}))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
