"""Tests for the memory-design wiring: train_bc.py, deploy.py, eval_scripted.py, the
launcher and the cost benchmark, for act_orig and act_lstm_full.

    python tools/test_memory_design.py

The models themselves are tested in g1_model/test_act_memory.py. Here:
  * train_bc.py trains both through its CLI at --obs-window 1, records the name
    in metadata.json and refuses any other window, the parser's default 12
    included - and an existing model still gets the default W_o = 12;
  * deploy.py loads both, carries act_lstm_full's LSTM state tick by tick (equal
    to re-reading the whole history every tick), resets it per episode, and runs
    them closed loop; the existing models do not carry anything;
  * eval_scripted.py selects among a new run's checkpoints with no
    model-specific code, and waits for the new launcher;
  * the launcher runs the requested steps in the requested order with ACT's
    settings, and its completion checks read files, not exit codes.
Training tests need a staged episode in data/synthetic (test_train._tiny); the
deploy tests record one scripted episode in memory (test_deploy._recorded).
"""
from __future__ import annotations

import atexit
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402

LAUNCHER = os.path.join(HERE, "train_memory_design_n40.ps1")
_CACHE: dict = {}


def _tmpdir() -> str:
    """A temp dir removed when the process exits: nothing here outlives the run."""
    d = tempfile.mkdtemp(prefix="memdesign_test_")
    atexit.register(shutil.rmtree, d, True)
    return d


@contextlib.contextmanager
def _small_models():
    """train_bc's OWN configs at test width: every field but the transformer's
    width and depth (or BC's hidden sizes) kept, and the config TYPE kept, so the
    CLI wiring under test is the real one - at full size a 4-step run writes
    ~1 GB of checkpoints."""
    import dataclasses
    import train_bc as TB
    from g1_model.act import ACTConfig
    real = TB._model_config

    def small(kind, K, W, batch, gate):
        c = real(kind, K, W, batch, gate)
        if isinstance(c, ACTConfig):
            return dataclasses.replace(c, hidden_dim=64, dim_feedforward=128, nheads=4,
                                       enc_layers=1)
        return dataclasses.replace(c, hidden=(64, 64))
    TB._model_config = small
    try:
        yield
    finally:
        TB._model_config = real


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


# ─── train_bc.py, end to end ─────────────────────────────────────────────────
def _data():
    """(episodes dir, splits, norm): three tiny real-format episodes, seeds 0-1
    train and 2 val, stats fitted on the train split. Cached per process."""
    if "data" in _CACHE:
        return _CACHE["data"]
    from g1_model.test_train import _tiny
    got = _tiny(n_eps=3, n_ticks=30, masked_consistent=True)
    if got is None:
        return None
    ep_dir, _ = got
    atexit.register(shutil.rmtree, ep_dir, True)
    tmp = _tmpdir()
    eps = [e for e in DS.scan(ep_dir) if e.seed in (0, 1)]
    st, ac, rep = DS.fit_norm_stats(eps, "train", allow_synthetic=True)
    norm = DS.save_norm_stats(st, ac, path=os.path.join(tmp, "norm.npz"), source=rep["source"],
                              n_episodes=2, seeds=rep["seeds"], split="train")
    splits = os.path.join(tmp, "splits.json")
    with open(splits, "w", encoding="utf-8") as fh:
        json.dump(dict(train=[0, 1], val=[2], spec_version=spec.SPEC_VERSION), fh)
    _CACHE["data"] = (ep_dir, splits, norm)
    return _CACHE["data"]


def _train_bc(*args):
    """Run tools/train_bc.py's main() in-process; (exit code, output)."""
    import train_bc as TB
    old_argv, old_data = sys.argv, dict(TB.DATA)
    sys.argv = ["train_bc.py"] + [str(a) for a in args]
    buf = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(buf):
            TB.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else str(e.code))
    finally:
        sys.argv = old_argv
        TB.DATA.clear()
        TB.DATA.update(old_data)
    return code, buf.getvalue()


def _trained(kind, *extra):
    """A finished 4-step `train_bc.py train` run of `kind` (K = 4, batch 4,
    checkpoint every 2 steps), at test width (`_small_models`). Cached per process."""
    key = ("trained", kind) + extra
    if key in _CACHE:
        return _CACHE[key]
    d = _data()
    if d is None:
        return None
    ep, splits, norm = d
    out = os.path.join(_tmpdir(), "%s_seed0" % kind)
    if kind in ("act_orig", "act_lstm_full"):
        extra = ("--obs-window", 1) + tuple(extra)
    with _small_models():
        code, log = _train_bc("train", "--model", kind, "--chunk", 4, "--batch", 4,
                              "--max-steps", 4, "--window-steps", 2, "--checkpoint-every", 2,
                              "--budget-reason", "test", "--episodes", ep, "--splits", splits,
                              "--norm", norm, "--out-dir", out, *extra)
    assert code == 0, (code, log[-3000:])
    _CACHE[key] = out
    return out


def _meta(run):
    return json.load(open(os.path.join(run, "metadata.json"), encoding="utf-8"))


def test_train_bc_trains_both_and_records_the_name():
    import torch
    import train_bc as TB
    from g1_model import train as T
    from g1_model.act_memory import ACTFullHistoryConfig, ACTLSTMFullPolicy, LSTM_INPUT
    for kind, cls in (("act_orig", "ACTPolicy"), ("act_lstm_full", "ACTLSTMFullPolicy")):
        run = _trained(kind)
        if run is None:
            print("      (skipped: nothing staged in data/synthetic)")
            return
        m = _meta(run)
        notes = m["config"]["notes"]
        assert notes["model"] == kind and notes["memory_design"]["model"] == kind, notes
        assert notes["chunk_size"] == 4 and notes["kl_weight_used"] == 10.0
        assert m["loader"]["obs_window"] == 1 and m["model"]["cls"] == cls
        assert m["config"]["lr"] == 1e-5 and m["config"]["batch_size"] == 4
        res = json.load(open(os.path.join(run, "result.json"), encoding="utf-8"))
        assert res["status"] == "hard_cap" and res["optimizer_steps"] == 4
        names = sorted(os.listdir(run))
        assert {"ckpt_step0000002.pt", "ckpt_step0000004.pt", "best.pt", "last.pt"} <= set(names)
        ck = torch.load(os.path.join(run, "last.pt"), map_location="cpu", weights_only=False)
        factory, note = TB._factory_for(ck)
        model, _ = T.load_checkpoint(os.path.join(run, "last.pt"), factory)
        assert type(model).__name__ == cls and note is None
        assert model.cfg.obs_window == 1
        if kind == "act_lstm_full":
            assert isinstance(model, ACTLSTMFullPolicy)
            assert isinstance(model.cfg, ACTFullHistoryConfig)
            assert ck["model_kwargs"]["lstm_input"] == LSTM_INPUT
            assert "re-reads states 0..t" in notes["memory_design"]["sampling"]


def test_train_bc_refuses_another_window_for_the_new_models():
    d = _data()
    if d is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    ep, splits, norm = d
    for kind in ("act_orig", "act_lstm_full"):
        for window in (("--obs-window", 12), ()):            # stated 12, or the default
            out = os.path.join(_tmpdir(), "x")
            code, _ = _train_bc("train", "--model", kind, "--chunk", 4, *window,
                                "--max-steps", 2, "--budget-reason", "test", "--episodes",
                                ep, "--splits", splits, "--norm", norm, "--out-dir", out)
            assert isinstance(code, str) and "pass --obs-window 1" in code, (window, code)
            assert not os.listdir(out), "refused before anything was written"


def test_overfit10_gate_runs_for_act_lstm_full():
    """The gate path too: reference computed on the W_o = 1 view (stated in the
    metadata), gate.json written, weight decay 0 as for every model at the gate."""
    d = _data()
    if d is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    ep, splits, norm = d
    from g1_model import train as T
    runs = _tmpdir()
    real = T.make_run_dir
    T.make_run_dir = lambda cfg, root=None: real(cfg, runs)     # never the repo's runs/
    try:
        with _small_models():
            code, log = _train_bc("overfit10", "--model", "act_lstm_full", "--chunk", 4,
                                  "--obs-window", 1, "--batch", 4, "--max-steps", 2,
                                  "--window-steps", 2, "--budget-reason", "test",
                                  "--episodes", ep, "--splits", splits, "--norm", norm)
    finally:
        T.make_run_dir = real
    assert code in (0, 1), (code, log[-3000:])            # 1 = the gate FAILED: allowed
    (run,) = [os.path.join(runs, f) for f in os.listdir(runs)]
    assert os.path.basename(run).endswith("act_lstm_full_overfit10_K4_W1")
    m = _meta(run)
    notes = m["config"]["notes"]
    assert notes["ambiguity_reference"]["obs_window"] == 1
    assert "W_o = 1 current observation only" in notes["memory_design"]["ambiguity_reference"]
    assert m["config"]["weight_decay"] == 0.0 and m["model"]["cls"] == "ACTLSTMFullPolicy"
    assert "LSTM dropout 0.3" in notes["regularization"]
    assert os.path.isfile(os.path.join(run, "gate.json"))


def test_existing_models_keep_the_shared_default_window():
    """--obs-window keeps its default 12: an existing model run without the flag
    (chunked BC here, the smallest) is W_o = 12, and its notes carry nothing new."""
    run = _trained("bc")
    if run is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    m = _meta(run)
    assert m["loader"]["obs_window"] == 12 and m["model"]["cls"] == "BCPolicy"
    assert "memory_design" not in m["config"]["notes"]


# ─── deploy.py ───────────────────────────────────────────────────────────────
def _run_new(kind):
    """A run directory around an UNTRAINED small checkpoint of `kind`, norm stats
    fitted on test_deploy's recorded episode (as test_deploy._run does)."""
    key = "deploy_run_" + kind
    if key in _CACHE:
        return _CACHE[key]
    import torch
    import test_deploy as TD
    from g1_model.act import ACTConfig, build_act
    from g1_model import act_memory as AM
    path, _, _ = TD._recorded()
    tmp = _tmpdir()
    ep = DS.Episode(path, DS.read_meta(path))
    st, ac, rep = DS.fit_norm_stats([ep], "train", allow_synthetic=True)
    norm = os.path.join(tmp, "norm.npz")
    DS.save_norm_stats(st, ac, path=norm, source=rep["source"], n_episodes=1,
                       seeds=rep["seeds"], split="train")
    splits = os.path.join(tmp, "splits.json")
    with open(splits, "w", encoding="utf-8") as fh:
        json.dump(dict(train=[TD.SEED], val=[], spec_version=spec.SPEC_VERSION), fh)
    K = 8
    small = dict(obs_window=1, chunk_size=K, lr=1e-5, hidden_dim=64, dim_feedforward=128,
                 nheads=4, enc_layers=2, dec_layers=1)
    torch.manual_seed(0)
    model = (AM.build_act_lstm_full(cfg=AM.ACTFullHistoryConfig(**small))
             if kind == "act_lstm_full" else build_act(cfg=ACTConfig(**small)))
    run = os.path.join(tmp, "%s_seed0" % kind)
    os.makedirs(run)
    torch.save(dict(state_dict=model.state_dict(), model_cls=type(model).__name__,
                    model_kwargs=model.hparams, spec_version=spec.SPEC_VERSION,
                    config={}, step=0, epoch=0, val_loss=None, git_commit=None),
               os.path.join(run, "ckpt.pt"))
    with open(os.path.join(run, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(config=dict(notes=dict(
            model=kind, chunk_size=K,
            data_paths=dict(episodes=os.path.dirname(path), splits=splits, norm=norm)))), fh)
    _CACHE[key] = run
    return run


def _policy(kind, **kw):
    import torch
    import deploy as D
    return D.CheckpointPolicy(_run_new(kind), "ckpt.pt", device=torch.device("cpu"), **kw)


def _states(n):
    import test_deploy as TD
    from g1_data.recorder import load_episode
    rec, _ = load_episode(TD._recorded()[0])
    return np.asarray(rec["states"][:n])


def test_deploy_loads_both_and_only_act_lstm_full_carries_memory():
    import test_deploy as TD
    import deploy as D
    import torch
    full, orig = _policy("act_lstm_full"), _policy("act_orig")
    assert (full.kind, full.W, full.K, full.recurrent) == ("act_lstm_full", 1, 8, True)
    assert (orig.kind, orig.W, orig.K, orig.recurrent) == ("act_orig", 1, 8, False)
    assert full.ensemble and orig.ensemble, "K > 1: temporal ensembling, as ACT"
    for kind in ("bc", "act_lstm"):
        old = D.CheckpointPolicy(TD._run(kind), "ckpt.pt", device=torch.device("cpu"))
        assert old.recurrent is False and old.memory is None


def test_deployed_memory_equals_rereading_the_whole_history_every_tick():
    """CheckpointPolicy.act carries (h, c); the reference re-reads states 0..t
    from zero memory at every tick through the model's forward, with its own
    ensembler. Equal up to rounding over 150 ticks of a real episode."""
    import torch
    from g1_model.act import TEMPORAL_ENSEMBLE_M, TemporalEnsembler
    from g1_model.history import history_of
    pol = _policy("act_lstm_full")
    S = _states(150)
    pol.reset()
    got = [pol.act(S[t], t) for t in range(len(S))]
    ens = TemporalEnsembler(pol.K, m=TEMPORAL_ENSEMBLE_M)
    Z = np.stack([pol.st.normalize(np.asarray(s, np.float32).astype(np.float64)).astype(np.float32)
                  for s in S])
    worst = 0.0
    for t in range(len(S)):
        with torch.no_grad():
            chunk = pol.model(history_of(torch.from_numpy(Z[:t + 1]).unsqueeze(0)))[0]
        want = pol.ac.denormalize(ens.step(chunk).numpy().astype(np.float64))
        worst = max(worst, float(np.abs(got[t] - want).max()))
    assert worst < 1e-4, "deployed (carried) vs re-read: max |diff| %.3e" % worst


def test_deployed_memory_is_reset_per_episode():
    pol = _policy("act_lstm_full")
    S = _states(40)
    pol.reset()
    a = [pol.act(S[t], t) for t in range(40)]
    assert pol.memory is not None
    pol.reset()
    assert pol.memory is None
    b = [pol.act(S[t], t) for t in range(40)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "memory leaked across episodes"
    # and the memory matters: without ensembling, tick 39's action differs from
    # the action for the same state with no ticks before it
    pol = _policy("act_lstm_full", ensemble=False)
    pol.reset()
    with_memory = [pol.act(S[t], t) for t in range(40)][-1]
    pol.reset()
    assert not np.allclose(pol.act(S[39], 0), with_memory, atol=1e-7)


def test_both_run_closed_loop_through_deploy():
    import test_deploy as TD
    import deploy as D
    _, _, loco = TD._recorded()
    for kind in ("act_orig", "act_lstm_full"):
        res, arrays = D.deploy_episode(TD.SEED, _policy(kind), loco=loco, max_ticks=40)
        assert res["kind"] == kind and res["ticks"] == len(arrays["states"]) > 0
        assert arrays["actions"].shape[1] == spec.ACTION_DIM


def test_deploy_main_writes_the_kind_and_the_memory_note():
    import deploy as D
    for kind in ("act_orig", "act_lstm_full"):
        out = os.path.join(_tmpdir(), "x.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            assert D.main(["--run", _run_new(kind), "--ckpt", "ckpt.pt", "--seed-list", "700",
                           "--max-ticks", "15", "--out", out]) == 0
        rows = [json.loads(L) for L in open(out, encoding="utf-8") if L.strip()]
        h = rows[0]["header"]
        assert len(rows) == 1 and h["kind"] == kind and h["obs_window"] == 1, h
        assert h["temporal_ensemble"] is True
        assert ("lstm_memory" in h) == (kind == "act_lstm_full"), h


# ─── eval_scripted.py ────────────────────────────────────────────────────────
def test_eval_scripted_selects_among_a_new_runs_checkpoints():
    """No model-specific code: a trained act_lstm_full run is DONE, step-budgeted,
    its last.pt is an alias of the final periodic checkpoint, and selection runs
    over the rest (deploy faked: the rows a deploy would write)."""
    import eval_scripted as E
    run = _trained("act_lstm_full")
    if run is None:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    assert E.run_status(run) in E.DONE_STATUS and not E.epoch_budgeted(run)
    cands = E.candidates(run)
    assert ("last.pt", "ckpt_step0000004.pt") in cands, cands
    assert [f for f, a in cands if a is None][:2] == ["ckpt_step0000002.pt", "ckpt_step0000004.pt"]
    step = {"ckpt_step0000002.pt": 2, "ckpt_step0000004.pt": 4, "best.pt": 4, "last.pt": 4}

    def fake(run_dir, ckpt, seed_args, n, out, lock_rule, logdir, allow=False):
        assert seed_args == ["--selection"] and lock_rule == "v3" and allow is True
        ok = ckpt == "ckpt_step0000002.pt"
        return [dict(success=ok, grasp=True, lift=ok, transport=ok, place=ok,
                     placement_error_m=0.1, header=dict(ckpt=dict(file=ckpt, step=step[ckpt])))
                for _ in range(n)]
    real, E.deploy = E.deploy, fake
    ev = _tmpdir()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sel = E.select(run, ev, "v3", True)
    finally:
        E.deploy = real
    assert sel["chosen"] == "ckpt_step0000002.pt", sel["chosen"]
    assert json.load(open(os.path.join(ev, "selection.json"), encoding="utf-8"))["chosen"] \
        == "ckpt_step0000002.pt"


def test_eval_scripted_waits_for_the_new_launcher():
    import eval_scripted as E
    for name in ("train_memory_design_n40.ps1", "train_scripted_v3_n40.ps1", "train_bc.py",
                 "deploy.py", "eval_scripted.py"):
        assert E.HEAVY.fullmatch(name), name
    assert not E.HEAVY.fullmatch("train_memory_design_n40.ps1.bak")


# ─── the launcher ────────────────────────────────────────────────────────────
def _ps1():
    return open(LAUNCHER, encoding="utf-8").read()


def _var(src, name):
    m = re.search(r'^\$%s = "(.*)"$' % re.escape(name), src, re.M)
    assert m, name
    return m.group(1)


def test_launcher_runs_the_requested_steps_in_the_requested_order():
    src = _ps1()
    block = src[src.index("$Steps = @("):src.index("\n)\n", src.index("$Steps = @("))]
    steps = []
    for line in block.splitlines():
        t = re.search(r'\(Train "(\w+)" "(\w+)" (\d)\)', line)
        e = re.search(r'Kind = "eval"; Runs = @\((.*)\)', line)
        if t:
            steps.append(("train", t.group(1), t.group(2), int(t.group(3))))
        elif e:
            steps.append(("eval", tuple(re.findall(r'"(\w+)"', e.group(1)))))
    assert steps == [
        ("train", "act_orig_n40_seed0", "act_orig", 0),
        ("train", "act_lstm_full_n40_seed0", "act_lstm_full", 0),
        ("eval", ("act_orig_n40_seed0", "act_lstm_full_n40_seed0")),
        ("train", "act_orig_n40_seed1", "act_orig", 1),
        ("train", "act_orig_n40_seed2", "act_orig", 2),
        ("train", "act_lstm_full_n40_seed1", "act_lstm_full", 1),
        ("train", "act_lstm_full_n40_seed2", "act_lstm_full", 2),
        ("eval", ("act_orig_n40_seed1", "act_orig_n40_seed2",
                  "act_lstm_full_n40_seed1", "act_lstm_full_n40_seed2")),
    ], steps
    assert 'foreach ($s in $Steps)' in src and 'if ($s.Kind -eq "train") { RunTraining $s }' in src


def test_launcher_uses_acts_settings_and_the_n40_data():
    src = _ps1()
    n40 = open(os.path.join(HERE, "train_scripted_v3_n40.ps1"), encoding="utf-8").read()
    assert _var(src, "Data40") == _var(n40, "Data40"), "same data as Run A"
    assert "splits_scripted_v3_n40.json" in _var(src, "Data40")
    assert _var(src, "Common") == "--chunk 100 --obs-window 1 --batch 8"
    budget = _var(src, "ActBudget")
    assert "--max-steps 200000 --window-steps 1000 --checkpoint-every 20000" in budget
    assert budget == _var(n40, "ActBudget"), "ACT's budget text, word for word"
    assert '"--model $Model --seed $Seed $Common $ActBudget $Data40"' in src
    assert r'-u tools\train_bc.py train $($r.Args) --out-dir' in src
    # every train_bc.py option the launcher passes exists, with the right model names
    import train_bc as TB
    for kind in ("act_orig", "act_lstm_full"):
        assert kind in TB.ACT_FAMILY


def test_launcher_evaluates_through_eval_scripted_under_lock_rule_v3():
    src = _ps1()
    call = [L for L in src.splitlines() if "tools\\eval_scripted.py" in L and "& python" in L]
    assert len(call) == 1, call
    for arg in ("--run-root $RunRoot", "--lock-rule v3", "--allow-lock-rule-mismatch",
                "--runs $Todo"):
        assert arg in call[0], (arg, call[0])
    assert "--wait-for-training" not in call[0], "the launcher runs it after training itself"


def test_launcher_completion_is_read_from_files_not_exit_codes():
    src = _ps1()
    run_status = src[src.index("function RunStatus"):src.index("function Done")]
    assert '"result.json"' in run_status and ".status" in run_status
    assert 'function Done($Status) { return ($Status -eq "finished" -or $Status -eq "hard_cap") }' in src
    evaluated = src[src.index("function Evaluated"):src.index("function RunTraining")]
    for f in ('"selection.json"', '"exp1.jsonl"', '"exp2.jsonl"', "$NExp1", "$NExp2"):
        assert f in evaluated, f
    # a skipped training is decided BEFORE anything is started, from result.json
    rt = src[src.index("function RunTraining"):src.index("function RunEval")]
    assert rt.index("if (Done $st)") < rt.index("Start-Process")
    assert "if (-not (Done $Status))" in rt and "$Code" in rt     # the exit code is only logged
    # an interrupted run is kept, never deleted, and takes any evaluation of it along
    assert 'Move-Item $Dir $Dest' in rt and 'Move-Item $OldEval "$Dest.eval"' in rt
    assert "Remove-Item" not in rt
    # the summary lines are the format tools/write_resources.py reads
    import write_resources as WR
    line = ("act_lstm_full_n40_seed0  status=hard_cap     steps=200000   exit=-1073740791 "
            "270.6 min  peak RAM 1.47 GB  peak GPU 2328 MiB")
    assert WR.LINE.match(line)
    assert ('status={1,-12} steps={2,-8} exit={3,-12} {4} min  peak RAM {5} GB  '
            'peak GPU {6} MiB') in src
    assert "training started" in src and "training ended" in src, "eval_scripted's _launcher_ended"
    assert "Tee-Object -FilePath $Summary" not in src, "UTF-16 in PowerShell 5.1"


def test_launcher_dry_run_is_200_steps_into_its_own_folder():
    src = _ps1()
    dry = src[src.index("if ($DryRun) {\n    $ActBudget"):]
    assert '--max-steps 200 --window-steps 100' in dry.split("}")[0]
    assert '"scripted_v3_memory_design_dryrun"' in src
    assert "param([switch]$DryRun" in src
    dd = src[src.index("if ($DryRun) {\n        # one short deploy"):]
    assert "tools\\deploy.py" in dd and "--max-ticks 100" in dd and "--allow-lock-rule-mismatch" in dd


def _pwsh():
    return os.environ.get("PWSH") or shutil.which("pwsh") or shutil.which("powershell")


def test_launcher_parses_in_powershell():
    exe = _pwsh()
    if not exe:
        print("      (skipped: no PowerShell here; set PWSH to check the parse)")
        return
    cmd = ("$t=$null; $e=$null; $null=[System.Management.Automation.Language.Parser]::"
           "ParseFile('%s', [ref]$t, [ref]$e); $e.Count" % LAUNCHER.replace("'", "''"))
    out = subprocess.run([exe, "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    assert out.stdout.strip().splitlines()[-1] == "0", out.stdout + out.stderr


def test_launcher_completion_checks_run_in_powershell():
    """The launcher's own RunStatus / Done / Evaluated, executed on fake files: a
    training is done only on a finished result.json; an evaluation only with
    selection.json AND one line per seed in exp1.jsonl and exp2.jsonl."""
    exe = _pwsh()
    if not exe:
        print("      (skipped: no PowerShell here; set PWSH to run the launcher's checks)")
        return
    src = _ps1()
    funcs = src[src.index("function RunStatus"):src.index("function RunTraining")]
    funcs = funcs.replace("\\", "/")                   # Windows accepts both; Linux one
    tmp = _tmpdir()
    q = lambda p: p.replace("'", "''")                 # noqa: E731
    script = ("$Root = '%s'\n$EvalRoot = '%s'\n" % (q(ROOT), q(os.path.join(tmp, "eval")))
              + funcs + r"""
$o = @()
$run = Join-Path '%s' 'run'
New-Item -ItemType Directory -Force $run | Out-Null
$o += (RunStatus $run)
'{"status": "running"}' | Out-File -Encoding utf8 (Join-Path $run 'result.json')
$o += (Done (RunStatus $run))
'{"status": "hard_cap"}' | Out-File -Encoding utf8 (Join-Path $run 'result.json')
$o += (Done (RunStatus $run))
$d = Join-Path $EvalRoot 'r'
New-Item -ItemType Directory -Force $d | Out-Null
$o += (Evaluated 'r')
'{}' | Out-File -Encoding utf8 (Join-Path $d 'selection.json')
1..$NExp1 | ForEach-Object { '{"seed": 1}' } | Out-File -Encoding utf8 (Join-Path $d 'exp1.jsonl')
1..($NExp2 - 1) | ForEach-Object { '{"seed": 1}' } | Out-File -Encoding utf8 (Join-Path $d 'exp2.jsonl')
$o += (Evaluated 'r')
1..$NExp2 | ForEach-Object { '{"seed": 1}' } | Out-File -Encoding utf8 (Join-Path $d 'exp2.jsonl')
$o += (Evaluated 'r')
"RESULT $NExp1 $NExp2 " + ($o -join '|')
""" % q(tmp))
    path = os.path.join(tmp, "check.ps1")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    out = subprocess.run([exe, "-NoProfile", "-File", path], capture_output=True, text=True)
    line = [L for L in out.stdout.splitlines() if L.startswith("RESULT ")]
    n1, n2 = (len(DS.load_seed_file(os.path.join(ROOT, "data", "eval", "%s_seeds.json" % e)))
              for e in ("exp1", "exp2"))
    assert line == ["RESULT %d %d no result.json|False|True|False|False|True" % (n1, n2)], \
        out.stdout + out.stderr
    shutil.rmtree(tmp, ignore_errors=True)


# ─── the cost benchmark ──────────────────────────────────────────────────────
def test_bench_measures_each_model_through_the_real_pieces():
    import bench_memory_design as B
    from g1_model.history import HistoryChunkDataset
    ds = B._synthetic_dataset(HistoryChunkDataset, 4, 1, 2, 12, seed=0)
    s = ds[13]
    assert set(s) >= {"obs", "action", "action_mask", "history"} and s["history"].shape == (2, 48)
    assert len(ds) == 24
    rows = [B.measure(kind, W, steps=1, warmup=0, n_eps=1, ticks=12, batch=2, score_eps=1,
                      device="cpu") for kind, W in B.MODELS]
    assert [r["model"] for r in rows] == ["act_lstm", "act_orig", "act_lstm_full"]
    assert [r["obs_window"] for r in rows] == [12, 1, 1]
    for r in rows:
        assert r["step_s"] > 0 and r["score_s_per_ep"] > 0 and r["parameters"] > 0
    rat = B.ratios(rows)
    assert rat["act_lstm"]["step_s"] == 1.0 and "act_lstm_full" in rat
    assert "ratios vs act_lstm" in B.render(rows, rat)


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
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(_tests()) - failed}/{len(_tests())} passed")
    raise SystemExit(1 if failed else 0)
