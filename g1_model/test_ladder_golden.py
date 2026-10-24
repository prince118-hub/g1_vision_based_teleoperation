"""Golden fingerprints of the EXISTING runner configurations, pinned before act_orig
and act_lstm_full were added (2026-10-24).

    python -m g1_model.test_ladder_golden

What `tools/train_bc.py` builds for bc (K = 1, K = 100), act and act_lstm (K = 100),
at the gate and outside it, W_o = 12, batch 8, seed 0 - `_model_config` then
`_make_model` - must stay exactly what it was. RECORDED BY THE OLD CODE (TR32): main
@ e10052b, through `tools/check_memory_design_regression.py`'s probe, on a Linux
CPU with torch 2.14.1+cu130; the same probe on the new code reproduced every value.

Build-independent fields are checked everywhere: class, config type, parameter and
tensor counts, the config and hparams (sha256 of their sorted JSON), the declared
optimizer settings and the regularization statement written into run metadata.
The weights (sha256 of every tensor, by name) and the output on a fixed input are a
property of the torch BUILD - seeded init draws through torch's own RNG - so they
are checked on the build they were recorded on and skipped, with a note, elsewhere
(as test_act.py's golden test does). On the NVIDIA laptop run
`tools/check_memory_design_regression.py`, which compares old and new code there.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from g1_data import spec                                            # noqa: E402

#: The torch build the weight and output values were recorded on.
_GOLDEN_TORCH = ("2.14.1+cu130",)

_GOLDEN = {
    "bc_K1_train": dict(cls="BCPolicy", config_type="BCConfig", n_params=569366, n_tensors=6,
         config_sha256="da571c27d2dcfa914ca1c54d86bb2e47472b7d7d85eee58125aa3b26e367b9a9",
         hparams_sha256="9312447bdcaaa742501d624ba320cee85cbbf8559d18c584d56bb4262266403c",
         optimizer={'lr': 0.001, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grad_clip': 1.0},
         regularization='NONE: dropout 0, weight_decay 0, no augmentation',
         state_sha256="59e1a59d28a2b3ec9b7569d79f89c093240bb56723f2387128491a282e4c5882",
         out_fsum=-1.0182613199576735),
    "bc_K1_gate": dict(cls="BCPolicy", config_type="BCConfig", n_params=569366, n_tensors=6,
         config_sha256="da571c27d2dcfa914ca1c54d86bb2e47472b7d7d85eee58125aa3b26e367b9a9",
         hparams_sha256="9312447bdcaaa742501d624ba320cee85cbbf8559d18c584d56bb4262266403c",
         optimizer={'lr': 0.001, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grad_clip': 1.0},
         regularization='NONE: dropout 0, weight_decay 0, no augmentation',
         state_sha256="59e1a59d28a2b3ec9b7569d79f89c093240bb56723f2387128491a282e4c5882",
         out_fsum=-1.0182613199576735),
    "bc_K100_train": dict(cls="BCPolicy", config_type="BCConfig", n_params=1686680, n_tensors=6,
         config_sha256="4d29dd0d7118b71e20e8acb88f012675d909e666c553f2c2760eab7545a712fb",
         hparams_sha256="1ad831bc4b225e81bf2d6ae234b844cd7c7b752594eff58f7d46057037c05c5e",
         optimizer={'lr': 0.001, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grad_clip': 1.0},
         regularization='NONE: dropout 0, weight_decay 0, no augmentation',
         state_sha256="a61a9d2782af291e393f2105ee1caf5044882c16f1a0c98ae5d8f6062ad49a2b",
         out_fsum=-14.82858623622451),
    "act_K100_train": dict(cls="ACTPolicy", config_type="ACTConfig", n_params=40767702, n_tensors=132,
         config_sha256="b747232b8a6b0826dc3c2f2a97b9e91e53d51f032019c9ef85e9e06241637ee0",
         hparams_sha256="52f5e94713baadd34f61e31237096eeffe25eae076f77483689412fe767f0d61",
         optimizer={'lr': 1e-05, 'weight_decay': 0.0001, 'optimizer': 'adamw', 'grad_clip': 1.0},
         regularization='dropout 0.1 (active in TRAINING; the gate scores eval mode, where it is off), weight_decay 0.0001, no augmentation',
         state_sha256="baf4f19f2d946af9b3f924196c9176e1bc5be2cd23bc433d21335aa68f339be7",
         out_fsum=-336.6006974161137),
    "act_K100_gate": dict(cls="ACTPolicy", config_type="ACTConfig", n_params=40767702, n_tensors=132,
         config_sha256="de99d82ad42e5a6174bc2b2b36166f73e05506d25c8e9ddb6c2d211525c342fa",
         hparams_sha256="79574c44bfe8c1d360d1da7f9922d76ce9d7cfc564f64d5b49994a9dd28c7097",
         optimizer={'lr': 1e-05, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grad_clip': 1.0},
         regularization='dropout 0.1 (active in TRAINING; the gate scores eval mode, where it is off), weight_decay 0, no augmentation',
         state_sha256="baf4f19f2d946af9b3f924196c9176e1bc5be2cd23bc433d21335aa68f339be7",
         out_fsum=-336.6006974161137),
    "act_lstm_K100_train": dict(cls="ACTPolicy", config_type="ACTConfig", n_params=41739478, n_tensors=143,
         config_sha256="5775024e2c898864a7c57a85e523275a459e97d02be0776be8d3684bffdac2f3",
         hparams_sha256="18dd700f8f68272bebe1de875f30e33eacc98e7d76406e82ef852597a2fc05e4",
         optimizer={'lr': 1e-05, 'weight_decay': 0.0001, 'optimizer': 'adamw', 'grad_clip': 1.0},
         regularization='dropout 0.1, LSTM dropout 0.3 between stacked layers only (active in TRAINING; the gate scores eval mode, where it is off), weight_decay 0.0001, no augmentation',
         state_sha256="039fe977f8193dceb744ff87907a14b466f8ce0b529c93a93873104b7622abc3",
         out_fsum=-280.6646006538067),
    "act_lstm_K100_gate": dict(cls="ACTPolicy", config_type="ACTConfig", n_params=41739478, n_tensors=143,
         config_sha256="d08ffb53fa8f23f95f8f91adc27993f99b779c347d5272f0a16647ac21e1236d",
         hparams_sha256="a4fd5160b68d102e97e125c6104117d14b6c682f5ab0dd397e4730d0f0c3167b",
         optimizer={'lr': 1e-05, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grad_clip': 1.0},
         regularization='dropout 0.1, LSTM dropout 0.3 between stacked layers only (active in TRAINING; the gate scores eval mode, where it is off), weight_decay 0, no augmentation',
         state_sha256="039fe977f8193dceb744ff87907a14b466f8ce0b529c93a93873104b7622abc3",
         out_fsum=-280.6646006538067),
}

_BUILD_DEPENDENT = ("state_sha256", "out_fsum")


def _sha_json(x) -> str:
    x = json.loads(json.dumps(x, default=str))
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def fingerprint(kind: str, K: int, gate: bool) -> dict:
    """The probe of tools/check_memory_design_regression.py, for one configuration."""
    import importlib
    TB = importlib.import_module("train_bc")
    mcfg = TB._model_config(kind, K, 12, 8, gate=gate)
    tcfg = TB._train_config(mcfg, 1, 8, seed=0)
    model = TB._make_model(mcfg, tcfg).eval()
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    g = torch.Generator().manual_seed(123)
    obs = torch.randn(4, 12, spec.STATE_DIM, generator=g)
    with torch.no_grad():
        out = model(obs)
    return dict(cls=type(model).__name__, config_type=type(mcfg).__name__,
                n_params=int(sum(p.numel() for p in model.parameters())), n_tensors=len(sd),
                config_sha256=_sha_json(dataclasses.asdict(mcfg)),
                hparams_sha256=_sha_json(getattr(model, "hparams", {})),
                optimizer=json.loads(json.dumps(mcfg.optimizer_config())),
                regularization=TB.regularization_statement(mcfg),
                state_sha256=h.hexdigest(),
                out_fsum=math.fsum(out.double().flatten().tolist()))


def test_existing_runner_configs_build_exactly_what_they_built_before():
    pinned = torch.__version__ in _GOLDEN_TORCH
    if not pinned:
        print("      (weights and outputs NOT checked: torch %s is not the recording build %s; "
              "every build-independent field checked. Run "
              "tools/check_memory_design_regression.py on this machine.)"
              % (torch.__version__, _GOLDEN_TORCH))
    for key, want in _GOLDEN.items():
        kind, k, mode = key.rsplit("_", 2)
        got = fingerprint(kind, int(k[1:]), gate=(mode == "gate"))
        for f, v in want.items():
            if f in _BUILD_DEPENDENT and not pinned:
                continue
            if f == "out_fsum":
                assert abs(got[f] - v) <= 1e-6 * max(1.0, abs(v)), (key, f, got[f], v)
            else:
                assert got[f] == v, (key, f, got[f], v)


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
