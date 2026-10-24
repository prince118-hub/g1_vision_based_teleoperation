"""tools/report_extras.py on synthetic dataset / run / eval folders with known answers.

    python -m pytest tools/test_report_extras.py -q

Every fixture is written in the format its real producer writes (ledger rows as
record_episodes.py / record_teleop_live.py append them, episode `meta` as
EpisodeBuffer.save stores it, result.json / metadata.json / metrics.jsonl as
train_bc.py writes them, selection.json as eval_scripted.select writes it),
reduced to the fields the report reads.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import report_extras as RE                                          # noqa: E402
import report_io as IO                                              # noqa: E402

SCRIPTED_META = dict(spec_version="g1-spec-1.2.0", lock_rule=dict(version="v3"),
                     grip_label="next_weld_v1", source="scripted_demo.run_episode",
                     demonstrator=dict(hold_max=0.8, reposition_s=0.5))


def _episode(path, n, meta, tracking=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = dict(meta, n_ticks=n)
    arrays = dict(states=np.zeros((n, 48), np.float32), meta=np.array(json.dumps(m)))
    if tracking is not None:
        arrays["tracking_ok"] = np.asarray(tracking, np.uint8)
    np.savez_compressed(path, **arrays)


def _ledger(folder, events):
    led = IO.ledger_module().EpisodeLedger(os.path.join(folder, "ledger.jsonl"))
    for ev, kw in events:
        led.append(ev, **kw)


# ═══ (a) data quality ═════════════════════════════════════════════════════════
@pytest.fixture
def scripted_folder(tmp_path):
    d = str(tmp_path / "episodes")
    _ledger(d, [("session", dict(label="scripted", seeds=[0, 1, 2, 3])),
                ("issue", dict(seed=0)), ("accept", dict(seed=0, path="ep_seed0000.npz")),
                ("issue", dict(seed=1)), ("discard", dict(seed=1, reason="placement")),
                ("issue", dict(seed=1)), ("accept", dict(seed=1, path="ep_seed0001.npz")),
                ("issue", dict(seed=2)), ("error", dict(seed=2, reason="RuntimeError()")),
                ("issue", dict(seed=3)), ("accept", dict(seed=3, path="ep_seed0003.npz")),
                ("issue", dict(seed=4)), ("discard", dict(seed=4, reason="tilt"))])
    for s, n in ((0, 600), (1, 700), (3, 650)):
        _episode(os.path.join(d, "ep_seed%04d.npz" % s), n, SCRIPTED_META)
    with open(os.path.join(d, "replay_check.txt"), "w") as fh:
        fh.write("MANIPULATION reproduced        : 3/3  (weld fired ...)\n\nVERDICT\n"
                 "REPLAY PASS\n")
    return d


def test_data_quality_scripted(scripted_folder):
    r = RE.data_quality(scripted_folder)
    assert r["ledger"] == "present"
    assert r["attempts"] == 6                       # issue rows
    assert r["seeds_issued"] == 5
    assert r["accepted"] == 3                       # seed 1 discarded THEN accepted: accepted
    assert r["rejected"] == 1                       # seed 4
    assert r["errors"] == 1                         # seed 2
    assert (r["len_min_ticks"], r["len_median_ticks"], r["len_max_ticks"]) == (600, 650.0, 700)
    assert r["len_median_s"] == pytest.approx(26.0)
    assert r["spec_version"] == "g1-spec-1.2.0"
    assert r["lock_rule"] == "v3"
    assert r["grip_label"] == "next_weld_v1"
    assert r["demonstrator_operator"] == "scripted (hold_max=0.8, reposition_s=0.5)"
    assert r["dropout_pct"] == "n/a (not measured)"
    assert r["tracking_measured_episodes"] == 0
    assert r["replay_check"] == "PASS (manipulation 3/3) [replay_check.txt]"


def test_data_quality_teleop_dropout_and_mixed(tmp_path):
    d = str(tmp_path / "teleop")
    _ledger(d, [("session", dict(label="teleop-live-fake", operator="fake")),
                ("issue", dict(seed=0)), ("accept", dict(seed=0, path="a.npz", ticks=100)),
                ("issue", dict(seed=1)), ("accept", dict(seed=1, path="b.npz", ticks=300))])
    tm = dict(spec_version="g1-spec-1.2.0", grip_label="next_weld_v1",
              source="teleop_live_fake", operator="fake")
    _episode(os.path.join(d, "a.npz"), 100, dict(tm, lock_rule=dict(version="v3")),
             tracking=[0] * 10 + [1] * 90)
    _episode(os.path.join(d, "b.npz"), 300, dict(tm, lock_rule=dict(version="v2")),
             tracking=[1] * 300)
    _episode(os.path.join(d, "failed", "c.npz"), 50, dict(tm, lock_rule=dict(version="v3")),
             tracking=[0] * 50)                     # failed/ is not part of the dataset
    r = RE.data_quality(d)
    assert r["dropout_pct"] == pytest.approx(2.5)  # 10 / 400
    assert r["failed_on_disk"] == 1 and r["episodes_on_disk"] == 2
    assert r["lock_rule"] == "MIXED: v2, v3"
    assert r["demonstrator_operator"] == "operator: fake"
    assert r["replay_check"] == "not run / not saved"


def test_data_quality_without_ledger_or_fields(tmp_path):
    d = str(tmp_path / "old")
    _episode(os.path.join(d, "x.npz"), 10, dict(spec_version="g1-spec-1.1.0"))
    r = RE.data_quality(d)
    assert r["ledger"] == "absent" and r["accepted"] is None
    assert r["lock_rule"] == "v1 (no field)"
    assert r["grip_label"] == "absent (pre-1.2.0)"


# ═══ (b) compute ══════════════════════════════════════════════════════════════
def _run(root, name, model, result=None, metrics=None, params=1000, resources=None):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    json.dump(dict(config=dict(notes=dict(model=model)), model=dict(parameters=params),
                   device=dict(device="cuda", name="NVIDIA GeForce RTX 3050 Laptop GPU")),
              open(os.path.join(d, "metadata.json"), "w"))
    if result is not None:
        json.dump(result, open(os.path.join(d, "result.json"), "w"))
    if metrics is not None:
        with open(os.path.join(d, "metrics.jsonl"), "w") as fh:
            for r in metrics:
                fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    if resources is not None:
        json.dump(resources, open(os.path.join(d, "resources.json"), "w"))
    return d


def test_compute_row_finished(tmp_path):
    d = _run(str(tmp_path), "act_160_seed0", "act",
             result=dict(total_wall_seconds=7200.0, optimizer_steps=100000,
                         steps_per_second=13.9, status="hard_cap"), params=41726166)
    r = RE.compute_row(d, bench=True)
    assert r["model"] == "act" and r["status"] == "hard_cap"
    assert r["wall_h"] == 2.0 and r["optimizer_steps"] == 100000 and r["params"] == 41726166
    assert r["train_device"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert r["peak_ram_gb"] is None and r["peak_source"].startswith("not recorded")
    assert r["infer_note"] == "n/a: no checkpoint in the run directory"


def test_compute_row_killed_run_and_sidecar(tmp_path):
    d = _run(str(tmp_path), "bc_n40_seed1", "bc",
             metrics=[dict(step=1000, elapsed_seconds=50.0), dict(step=2000, elapsed_seconds=99.5),
                      '{"step": 3000, "elaps'],    # torn final line of a killed run
             resources=dict(peak_ram_gb=3.5, peak_gpu_gb=1.25))
    r = RE.compute_row(d, bench=False)
    assert r["status"] == "no result.json (last metrics.jsonl row)"
    assert (r["wall_s"], r["optimizer_steps"]) == (99.5, 2000)
    assert (r["peak_ram_gb"], r["peak_gpu_gb"], r["peak_source"]) == (3.5, 1.25, "resources.json")
    assert r["infer_note"] == "n/a: --no-bench"


def test_time_calls_with_a_fake_clock():
    t = [0.0]

    def clock():                                    # every read advances 1 ms
        t[0] += 0.001
        return t[0]
    calls = []
    r = RE.time_calls(lambda: calls.append(1), n_iter=50, n_warmup=5, clock=clock)
    assert len(calls) == 55
    assert r["n"] == 50
    assert r["ms_mean"] == pytest.approx(1.0) and r["ms_p95"] == pytest.approx(1.0)


def test_benchmark_without_torch_is_na(tmp_path):
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed here; this checks the torch-less path")
    except ImportError:
        pass
    r = RE.benchmark_checkpoint(str(tmp_path), "best.pt")
    assert r["infer_note"] == "n/a: torch not installed"


# ═══ (c) offline vs real ══════════════════════════════════════════════════════
def test_benchmark_times_act_lstm_full_through_step(tmp_path):
    """act_lstm_full refuses a plain (1, 1, 48) tensor; the benchmark must time
    the deployed call, step() with carried (h, c) - and still time ACT as before."""
    torch = pytest.importorskip("torch")
    try:
        from g1_model import act_memory as AM
        from g1_model.act import ACTConfig, build_act
        from g1_data import spec
    except Exception:                                   # noqa: BLE001  (no mujoco here)
        pytest.skip("model stack not importable")
    small = dict(obs_window=1, chunk_size=8, lr=1e-5, hidden_dim=64, dim_feedforward=128,
                 nheads=4, enc_layers=1, dec_layers=1)
    for kind, model in (("act_lstm_full", AM.build_act_lstm_full(cfg=AM.ACTFullHistoryConfig(**small))),
                        ("act", build_act(cfg=ACTConfig(**small)))):
        d = tmp_path / kind
        d.mkdir()
        torch.save(dict(state_dict=model.state_dict(), model_cls=type(model).__name__,
                        model_kwargs=model.hparams, spec_version=spec.SPEC_VERSION,
                        config={}, step=0, epoch=0, val_loss=None, git_commit=None),
                   str(d / "best.pt"))
        r = RE.benchmark_checkpoint(str(d), "best.pt", device="cpu", n_iter=5)
        assert "infer_ms_mean" in r and r["infer_ms_mean"] > 0, r
        want = "step() with carried LSTM memory" if kind == "act_lstm_full" else "model forward only"
        assert r["infer_note"].startswith(want), r["infer_note"]


def _selection(eval_root, run, cands, chosen):
    d = os.path.join(eval_root, run)
    os.makedirs(d, exist_ok=True)
    json.dump(dict(run=run, chosen=chosen, candidates=cands), open(os.path.join(d, "selection.json"), "w"))


def _cand(f, step, val, success, alias=None):
    if alias:
        return dict(ckpt=f, alias_of=alias)
    return dict(ckpt=f, alias_of=None, progress=step, n=20, success=success, grasp=success + 5,
                lift=success + 3, transport=success + 1, place=success,
                mean_placement_error_m=0.1, ckpt_info=dict(file=f, step=step, epoch=None,
                                                           val_loss=val))


def test_spearman_hand_checked():
    # ranks x 1..5, y 2,1,4,3,5: d = -1,1,-1,1,0, sum d^2 = 4, rho = 1 - 6*4/(5*24) = 0.8
    assert RE.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])["rho"] == pytest.approx(0.8)
    assert RE.spearman([1, 2, 3], [0, 0, 0])["note"] == "undefined: selection success is constant"
    assert RE.spearman([1, 2], [1, 2])["note"] == "undefined: fewer than 3 points"


@pytest.fixture
def runs(tmp_path):
    root = str(tmp_path / "scripted_v3")
    ev = os.path.join(root, "eval")
    a = _run(root, "act_160_seed0", "act", result=dict(total_wall_seconds=10, status="hard_cap"),
             metrics=[dict(step=20000, val_loss=0.3), dict(step=40000, val_loss=0.2)])
    _selection(ev, "act_160_seed0", [
        _cand("ckpt_step0020000.pt", 20000, None, 2),      # val from metrics.jsonl: 0.3
        _cand("ckpt_step0040000.pt", 40000, None, 3),      # 0.2
        _cand("ckpt_step0060000.pt", 60000, 0.1, 5),
        _cand("best.pt", 10000, 0.4, 1),
        _cand("last.pt", None, None, None, alias="ckpt_step0060000.pt")],
        chosen="ckpt_step0060000.pt")
    b = _run(root, "bc_160_seed0", "bc", result=dict(total_wall_seconds=10, status="finished"))
    _selection(ev, "bc_160_seed0", [_cand("ckpt_epoch%04d.pt" % e, e, v, 0)
                                     for e, v in ((10, 0.05), (20, 0.04), (30, 0.03))],
               chosen="ckpt_epoch0010.pt")
    return root, ev, a, b


def test_selection_points_and_rho(runs):
    root, ev, a, _ = runs
    pts = RE.selection_points(a, RE.eval_dir_for(a, ev))
    assert [p["ckpt"] for p in pts] == ["ckpt_step0020000.pt", "ckpt_step0040000.pt",
                                        "ckpt_step0060000.pt", "best.pt"]   # alias dropped
    assert [p["val_loss"] for p in pts] == [0.3, 0.2, 0.1, 0.4]
    assert [p["val_loss_source"] for p in pts] == ["metrics.jsonl"] * 2 + ["checkpoint"] * 2
    assert [p["chosen"] for p in pts] == [False, False, True, False]
    sp = RE.spearman([p["val_loss"] for p in pts], [p["success"] for p in pts])
    assert sp["rho"] == pytest.approx(-1.0)        # lower loss <-> more successes, perfectly


def test_build_end_to_end(runs, scripted_folder, tmp_path):
    root, ev, a, b = runs
    out = str(tmp_path / "report")
    r = RE.build([scripted_folder], RE.find_runs(root), ev, out, bench=False)
    sp = {(s["scope"], s["name"]): s for s in r["spearman"]}
    assert sp[("run", "act_160_seed0")]["rho"] == pytest.approx(-1.0)
    assert sp[("run", "bc_160_seed0")]["rho"] is None
    assert sp[("run", "bc_160_seed0")]["note"] == "undefined: selection success is constant"
    assert sp[("model (pooled runs)", "act")]["n"] == 4
    for f in ("data_quality.csv", "compute.csv", "offline_vs_real.csv",
              "offline_vs_real_spearman.csv", "offline_vs_real.png", "report_extras.md"):
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    md = open(os.path.join(out, "report_extras.md"), encoding="utf-8").read()
    assert "## (a) Data quality" in md and "## (c) Offline vs real" in md
    assert [c["run"] for c in r["compute"]] == ["act_160_seed0", "bc_160_seed0"]   # eval/ is no run


def test_cli(runs, tmp_path, capsys):
    root, _, _, _ = runs
    out = str(tmp_path / "cli")
    assert RE.main(["--run-root", root, "--out", out, "--no-bench"]) == 0
    assert "selection points 7" in capsys.readouterr().out
