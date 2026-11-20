"""tools/human_runs.py: status classification, the no-resume rule, restart, and settings parity."""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import human_runs as HR                                             # noqa: E402


def _result(d, status, steps=10):
    os.makedirs(d, exist_ok=True)
    json.dump(dict(status=status, optimizer_steps=steps, total_wall_seconds=36.0),
              open(os.path.join(d, "result.json"), "w"))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(HR, "_check_data", lambda: None)
    monkeypatch.setattr(HR, "_refuse_if_heavy", lambda: None)
    monkeypatch.setattr(HR, "heavy_processes", lambda exclude_pid=None: [])
    return str(tmp_path)


def test_train_status_states(root):
    name = HR.run_name("act_orig", 0)
    assert HR.train_status(root, name)["state"] == "not_started"
    os.makedirs(os.path.join(root, name))
    assert HR.train_status(root, name)["state"] == "not_started"          # empty folder
    open(os.path.join(root, name, "train.log"), "w").write("x")
    assert HR.train_status(root, name)["state"] == "incomplete"
    _result(os.path.join(root, name), "running")
    assert HR.train_status(root, name)["state"] == "incomplete"
    for s in ("finished", "hard_cap"):
        _result(os.path.join(root, name), s)
        assert HR.train_status(root, name)["state"] == "done"


def test_interrupted_run_is_refused_not_resumed(root, monkeypatch):
    called = []
    monkeypatch.setattr(HR, "_run_logged", lambda *a, **k: called.append(a))
    name = HR.run_name("act_lstm_full", 1)
    d = os.path.join(root, name)
    os.makedirs(d)
    open(os.path.join(d, "ckpt_step0020000.pt"), "w").write("partial")
    with pytest.raises(SystemExit) as e:
        HR.train("act_lstm_full", 1, root=root)
    assert "INTERRUPTED" in str(e.value) and "-Restart" in str(e.value)
    assert not called and os.path.exists(os.path.join(d, "ckpt_step0020000.pt"))


def test_restart_moves_the_incomplete_run_and_trains_from_scratch(root, monkeypatch):
    name = HR.run_name("bc", 2)
    d = os.path.join(root, name)
    os.makedirs(d)
    open(os.path.join(d, "train.log"), "w").write("partial")
    os.makedirs(os.path.join(root, "eval", name))
    seen = {}

    def fake_run(cmd, out, err):
        seen["cmd"] = cmd
        seen["fresh"] = os.listdir(d) == []                   # nothing left from the old run
        _result(d, "finished")
        return dict(exit_code=0, wall_seconds=1.0, peak_ram_gb=0.1, peak_gpu_mib=1, interrupted_by_user=False)
    monkeypatch.setattr(HR, "_run_logged", fake_run)
    assert HR.train("bc", 2, restart=True, root=root) == 0
    assert seen["fresh"] and "--resume-from" not in seen["cmd"]
    moved = os.listdir(os.path.join(root, "_failed"))
    assert any(m.startswith(name) and not m.endswith(".eval") for m in moved)
    assert any(m.endswith(".eval") for m in moved), "the old run's evaluation must move with it"
    assert json.load(open(os.path.join(root, "summaries", name + ".train.txt")))["state"] == "done"
    assert HR.train("bc", 2, root=root) == 0                   # done -> nothing to do


def test_cudnn_exit_crash_after_finished_result_counts_as_done(root, monkeypatch):
    name = HR.run_name("act_lstm_full", 0)

    def fake_run(cmd, out, err):
        _result(os.path.join(root, name), "hard_cap", 200000)
        return dict(exit_code=3221226505, wall_seconds=1.0, peak_ram_gb=0.1, peak_gpu_mib=1,
                    interrupted_by_user=False)
    monkeypatch.setattr(HR, "_run_logged", fake_run)
    assert HR.train("act_lstm_full", 0, root=root) == 0


def test_eval_refuses_unfinished_training(root):
    with pytest.raises(SystemExit) as e:
        HR.evaluate("act_orig", 2, root=root)
    assert "REFUSED" in str(e.value)


def test_eval_status_states(root):
    name = HR.run_name("act_orig", 0)
    assert HR.eval_status(root, name)["state"] == "not_started"
    ev = os.path.join(root, "eval", name)
    os.makedirs(ev)
    json.dump(dict(chosen="best.pt"), open(os.path.join(ev, "selection.json"), "w"))
    line = json.dumps(dict(success=True, header=dict(max_ticks=1234)))
    open(os.path.join(ev, "exp1.jsonl"), "w").write((line + "\n") * 100)
    assert HR.eval_status(root, name)["state"] == "incomplete"
    open(os.path.join(ev, "exp2.jsonl"), "w").write((line + "\n") * 99)
    assert HR.eval_status(root, name)["state"] == "incomplete"
    open(os.path.join(ev, "exp2.jsonl"), "w").write((line + "\n") * 100)
    s = HR.eval_status(root, name)
    assert s["state"] == "done" and s["exp1"] == "100/100" and s["max_ticks"] == 1234


@pytest.fixture
def dataset_reset():
    yield
    HR.use_dataset("human_v1")


def test_human_v1_names_and_paths_are_unchanged(dataset_reset):
    HR.use_dataset("human_v1")
    assert HR.run_name("act_orig", 0) == "act_orig_human40_seed0"
    assert HR.run_name("bc", 2) == "bc_b256_human40_seed2"
    assert (HR.EPISODES, HR.SPLITS, HR.NORM, HR.RUN_ROOT) == (
        "recordings/demonstrations", "data/processed/splits_human_v1.json",
        "data/processed/norm_stats_human_v1.npz", "runs/human_v1")
    assert HR._reason("act_orig") == HR.MODELS["act_orig"]["reason"]
    assert HR.CAP_FROM == "recordings/demonstrations"


def test_human_v2_cannot_write_into_human_v1(dataset_reset):
    HR.use_dataset("human_v2")
    assert HR.RUN_ROOT == "runs/human_v2" and HR.run_name("act_orig", 0) == "act_orig_human40v2_seed0"
    assert all("human_v1" not in p for p in (HR.EPISODES, HR.SPLITS, HR.NORM, HR.RUN_ROOT))
    assert HR._reason("act_orig").startswith("human_v2") and "200k steps" in HR._reason("act_orig")
    assert {HR.run_name(m, s) for m, s in HR.all_runs()}.isdisjoint(
        {"%s_seed%d" % (HR.MODELS[m]["stem"], s) for m, s in HR.all_runs()})


def test_human_v2_train_and_eval_commands(root, monkeypatch, dataset_reset):
    HR.use_dataset("human_v2")
    seen = []

    def fake_run(cmd, out, err):
        seen.append(cmd)
        if "train_bc.py" in " ".join(cmd):
            _result(os.path.join(root, HR.run_name("act_orig", 0)), "finished")
        return dict(exit_code=0, wall_seconds=1.0, peak_ram_gb=0.1, peak_gpu_mib=1, interrupted_by_user=False)
    monkeypatch.setattr(HR, "_run_logged", fake_run)
    assert HR.train("act_orig", 0, root=root) == 0
    t = seen[0]
    for flag, val in (("--episodes", "data/processed/episodes_human_v2"),
                      ("--splits", "data/processed/splits_human_v2.json"),
                      ("--norm", "data/processed/norm_stats_human_v2.npz")):
        assert t[t.index(flag) + 1] == val
    assert " ".join(HR.MODELS["act_orig"]["budget"]) in " ".join(t)       # same settings as human_v1
    assert t[t.index("--budget-reason") + 1].startswith("human_v2")
    HR.evaluate("act_orig", 0, root=root)
    e = seen[1]
    assert e[e.index("--max-ticks-from") + 1] == "recordings/demonstrations"   # the same cap
    assert e[e.index("--runs") + 1] == "act_orig_human40v2_seed0" and "v3" in e


def test_ps1_launchers_pass_data_with_human_v1_default():
    # every launcher offers exactly the datasets human_runs.py knows (2026-11-20: + human_v3)
    want = "[ValidateSet(%s)][string]$Data = \"human_v1\"" % ", ".join('"%s"' % d for d in sorted(HR.DATASETS))
    for f in ("run_human.ps1", "eval_human.ps1", "status_human.ps1", "queue_human.ps1"):
        s = _ps1(f)
        assert want in s, (f, want)
        assert "--data" in s and "$Data" in s, f


def _ps1(path):
    return open(os.path.join(HERE, path), encoding="utf-8").read()


def test_settings_are_the_scripted_main_runs():
    bc = _ps1("train_scripted_v3_bc_b256.ps1")
    md = _ps1("train_memory_design_n40.ps1")
    assert '$Common = "--obs-window 12 --batch 256"' in bc
    assert '$BcBudget = "--epochs 300 --checkpoint-every 30 ' in bc
    assert "--model bc --chunk 1" in bc
    assert '$Common = "--chunk 100 --obs-window 1 --batch 8"' in md
    assert re.search(r'\$ActBudget = "--max-steps 200000 --window-steps 1000 --checkpoint-every 20000 ', md)
    m = HR.MODELS
    assert " ".join(m["bc"]["args"]) == "--model bc --chunk 1 --obs-window 12 --batch 256"
    assert " ".join(m["bc"]["budget"]) == "--epochs 300 --checkpoint-every 30"
    for k in ("act_orig", "act_lstm_full"):
        assert " ".join(m[k]["args"]) == "--model %s --chunk 100 --obs-window 1 --batch 8" % k
        assert " ".join(m[k]["budget"]) == "--max-steps 200000 --window-steps 1000 --checkpoint-every 20000"
    assert HR.SEEDS == (0, 1, 2)
