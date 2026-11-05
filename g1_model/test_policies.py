"""The g1_model/policies/ entry modules are aliases, and build exactly what the trainer builds.

    python -m g1_model.test_policies          (or: python -m pytest g1_model/test_policies.py)

For every entry - BC, ACT (act_orig), ACT-LSTM (act_lstm_full), ACT-12 (act) and
ACT-LSTM-12 (act_lstm):
  * `Config`, `Policy` and `build` ARE the real objects (identity, not copies), and
    the code names cover exactly `tools/train_bc.py --model`'s choices;
  * the documented `TRAIN_ARGS`, read by `tools/train_bc.py`'s own `_model_config`,
    give exactly `Config(**CONFIG_KWARGS)` - and, at the overfit-10 gate, that config
    with weight decay 0 - so the settings written in the entry cannot drift from the
    trainer unnoticed;
  * a FULL-SIZE model built through the entry, seeded as the trainer seeds it
    (`train.seeded_build`, seed 0), equals the model `train_bc._make_model` builds:
    class, hparams, every parameter and buffer to the bit, the optimizer declaration,
    and the output on a fixed input - for act_lstm_full both the re-read history and
    the carried `step` path that deployment uses;
  * a checkpoint written by `train.save_checkpoint` loads through the entry's `build`
    to the same weights and the same outputs as through deploy's own loader
    (`train_bc._factory_for`, which `tools/deploy.CheckpointPolicy` calls).
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import shutil
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from g1_data import spec                                            # noqa: E402
from g1_model import policies as P                                  # noqa: E402

#: Where each entry's real objects live (module of the class, as imported).
_REAL = {
    "bc": ("g1_model.models", "BCConfig", "BCPolicy", "build_bc"),
    "act_orig": ("g1_model.act", "ACTConfig", "ACTPolicy", "build_act"),
    "act_lstm_full": ("g1_model.act_memory", "ACTFullHistoryConfig",
                      "ACTLSTMFullPolicy", "build_act_lstm_full"),
    "act": ("g1_model.act", "ACTConfig", "ACTPolicy", "build_act"),
    "act_lstm": ("g1_model.act", "ACTConfig", "ACTPolicy", "build_act"),
}

_ENTRIES = (P.bc, P.act, P.act_lstm, P.act_12, P.act_lstm_12)


def _tb():
    return importlib.import_module("train_bc")


def _args(entry) -> dict:
    """`TRAIN_ARGS` as {flag: value}."""
    tok = entry.TRAIN_ARGS.split()
    assert len(tok) % 2 == 0 and all(t.startswith("--") for t in tok[0::2]), tok
    return dict(zip(tok[0::2], tok[1::2]))


def _trainer_config(entry, gate: bool):
    a = _args(entry)
    return _tb()._model_config(a["--model"], int(a["--chunk"]), int(a["--obs-window"]),
                               int(a["--batch"]), gate=gate)


def _same_state(m1, m2, where):
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert list(s1) == list(s2), (where, "tensor names differ")
    for k in s1:
        assert s1[k].dtype == s2[k].dtype and torch.equal(s1[k], s2[k]), (where, k)
    assert type(m1) is type(m2), where
    assert getattr(m1, "hparams", None) == getattr(m2, "hparams", None), where


def _same_outputs(m1, m2, cfg, where, steps=20):
    """The output on fixed inputs, to the bit. act_lstm_full: re-read histories of
    different lengths, and the carried tick-by-tick `step` path."""
    from g1_model.history import history_of
    m1.eval(), m2.eval()
    g = torch.Generator().manual_seed(123)
    with torch.no_grad():
        if getattr(m1, "CARRIES_MEMORY", False):
            states = torch.randn(4, 40, spec.STATE_DIM, generator=g)
            obs = history_of(states, lengths=[40, 17, 1, 33])
            assert torch.equal(m1(obs), m2(obs)), (where, "history forward")
            mem1 = mem2 = None
            for t in range(steps):
                x = states[:1, t:t + 1]
                a1, mem1 = m1.step(x, mem1)
                a2, mem2 = m2.step(x, mem2)
                assert torch.equal(a1, a2), (where, "step", t)
                assert all(torch.equal(u, v) for u, v in zip(mem1, mem2)), (where, "h,c", t)
        else:
            obs = torch.randn(4, cfg.obs_window, spec.STATE_DIM, generator=g)
            assert torch.equal(m1(obs), m2(obs)), (where, "forward")


# ─── the aliases ─────────────────────────────────────────────────────────────
def test_entries_alias_the_real_objects():
    for entry in _ENTRIES:
        mod, cfg, pol, build = _REAL[entry.CODE_NAME]
        real = importlib.import_module(mod)
        assert entry.Config is getattr(real, cfg), entry.__name__
        assert entry.Policy is getattr(real, pol), entry.__name__
        assert entry.build is getattr(real, build), entry.__name__
        assert entry.ROLE in ("headline", "ablation"), entry.__name__
    from g1_model.act import TemporalEnsembler
    for entry in _ENTRIES[1:]:
        assert entry.Ensembler is TemporalEnsembler, entry.__name__
    from g1_model.history import HistoryChunkDataset
    assert P.act_lstm.Dataset is HistoryChunkDataset


def test_code_names_cover_train_bcs_model_choices():
    TB = _tb()
    choices = {"bc"} | set(TB.ACT_FAMILY)            # tools/train_bc.py --model choices
    assert set(P.BY_CODE_NAME) == choices, (sorted(P.BY_CODE_NAME), sorted(choices))
    for name, entry in P.BY_CODE_NAME.items():
        assert entry.CODE_NAME == name and _args(entry)["--model"] == name, name
    assert {k: v.THESIS_NAME for k, v in P.HEADLINE.items()} == {
        "BC": "BC", "ACT": "ACT", "ACT-LSTM": "ACT-LSTM"}
    assert {k: v.THESIS_NAME for k, v in P.ABLATION.items()} == {
        "ACT-12": "ACT-12", "ACT-LSTM-12": "ACT-LSTM-12"}
    assert all(e.ROLE == "headline" for e in P.HEADLINE.values())
    assert all(e.ROLE == "ablation" for e in P.ABLATION.values())


def test_docstrings_name_thesis_name_code_name_and_real_class():
    for entry in _ENTRIES:
        doc = entry.__doc__ or ""
        assert entry.THESIS_NAME in doc and entry.CODE_NAME in doc, entry.__name__
        assert entry.Policy.__name__ in doc and entry.Config.__name__ in doc, entry.__name__
        real_file = entry.Policy.__module__.replace(".", "/") + ".py"
        assert real_file in doc, (entry.__name__, real_file)
        assert os.path.isfile(os.path.join(ROOT, real_file)), real_file


# ─── the documented settings are the trainer's ──────────────────────────────
def test_documented_settings_are_what_train_bc_builds():
    for entry in _ENTRIES:
        mine = entry.Config(**entry.CONFIG_KWARGS)
        got = _trainer_config(entry, gate=False)
        assert type(got) is type(mine) and got == mine, (entry.__name__, got, mine)
        gate = _trainer_config(entry, gate=True)
        assert type(gate) is type(mine), entry.__name__
        # the overfit-10 gate is unregularised: weight decay 0 (BC: also dropout 0,
        # which are already its defaults), every other field unchanged
        assert gate == dataclasses.replace(mine, weight_decay=0.0), entry.__name__
    assert P.bc.Config(**P.bc.CONFIG_KWARGS).batch_size == int(_args(P.bc)["--batch"])


# ─── the entries build the trainer's model, to the bit ──────────────────────
def test_each_entry_builds_exactly_the_model_train_bc_builds():
    """Full size, seed 0, as `tools/train_bc.py` builds it (`_make_model`)."""
    TB = _tb()
    from g1_model import train as T
    for entry in _ENTRIES:
        a = _args(entry)
        mcfg = _trainer_config(entry, gate=False)
        tcfg = TB._train_config(mcfg, 1, int(a["--batch"]), seed=0)
        theirs = TB._make_model(mcfg, tcfg)
        mine = T.seeded_build(tcfg, entry.build, cfg=entry.Config(**entry.CONFIG_KWARGS))
        assert isinstance(mine, entry.Policy), entry.__name__
        _same_state(mine, theirs, entry.__name__)
        assert mine.optimizer_config() == theirs.optimizer_config(), entry.__name__
        _same_outputs(mine, theirs, mcfg, entry.__name__)
        del theirs, mine


def test_entry_build_loads_a_checkpoint_as_deploy_does():
    """A checkpoint written by the trainer's `save_checkpoint` (test width) loads
    through `entry.build` exactly as through `train_bc._factory_for`, the builder
    `tools/deploy.CheckpointPolicy` uses."""
    TB = _tb()
    from g1_model import train as T
    from g1_model.act import ACTConfig
    tmp = tempfile.mkdtemp(prefix="policies_test_")
    try:
        for entry in _ENTRIES:
            a = _args(entry)
            mcfg = _trainer_config(entry, gate=False)
            mcfg = (dataclasses.replace(mcfg, hidden_dim=64, dim_feedforward=128,
                                        nheads=4, enc_layers=1)
                    if isinstance(mcfg, ACTConfig) else
                    dataclasses.replace(mcfg, hidden=(64, 64)))
            tcfg = TB._train_config(mcfg, 1, int(a["--batch"]), seed=0)
            model = TB._make_model(mcfg, tcfg)
            path = os.path.join(tmp, "%s.pt" % entry.CODE_NAME)
            T.save_checkpoint(path, model, tcfg, dict(git_commit=None), epoch=0,
                              train_loss=0.0, val_loss=None, step=0)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            factory, note = TB._factory_for(ck)
            assert note is None, note
            via_deploy, _ = T.load_checkpoint(path, factory)
            via_entry, _ = T.load_checkpoint(path, entry.build)
            _same_state(via_entry, via_deploy, entry.__name__)
            _same_state(via_entry, model, entry.__name__ + " vs saved")
            _same_outputs(via_entry, via_deploy, mcfg, entry.__name__)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
