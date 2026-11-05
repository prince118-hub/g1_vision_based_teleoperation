"""tools/render_final_videos.py with a fake runner and tiny videos: no simulation, no rendering.

    python -m pytest tools/test_render_final_videos.py -q
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import render_final_videos as RF                                    # noqa: E402
import video_common as VC                                           # noqa: E402

TICKS = 12


def _video(path, n=TICKS):
    with VC.Mp4Writer(path, VC.SRC_W, VC.SRC_H, VC.TICK_HZ) as w:
        for t in range(n):
            w.write(np.full((VC.SRC_H, VC.SRC_W, 3), t * 5, np.uint8))


def _record(seed, ok, reason, sha, run="", model="", ticks=TICKS):
    return dict(seed=seed, success=ok, fail_reason="" if ok else reason, ticks=ticks, trajectory_sha256=sha, box_xy=[1.5, 0.1], heldout=False,
                placement_error_m=0.04 if ok else 0.5, final_box_goal_m=0.04 if ok else 0.5, phase_reached="VERIFY" if ok else "REACH",
                end_reason="success" if ok else "time_limit", weld_engage_tick=2, weld_release_tick=8, lock_ticks=[1], release_ticks=[5],
                phase_boundaries=dict(SETTLE=0, REACH=1, GRASP=3, LIFT=5, MOVE=6, LOWER=8, RELEASE=9, VERIFY=10), kind=model,
                header=dict(ckpt=dict(file="c.pt", step=1), lock_rule=dict(version="v3"), training_data_lock_rule="v2", max_ticks=900, K=100))


def test_start_command_title_is_a_quoted_string():
    """Start-Process drops double quotes from -Command: that title reached PowerShell unquoted, as a command that failed."""
    line = next(L.strip() for L in RF.__doc__.splitlines() if L.strip().startswith("Start-Process"))
    command = re.search(r"'-Command','((?:[^']|'')*)'$", line).group(1).replace("''", "'")
    assert '"' not in command
    assert command == "$Host.UI.RawUI.WindowTitle = 'Thesis videos: scripted_final'; python -u tools/render_final_videos.py"


@pytest.fixture
def results(tmp_path, monkeypatch):
    """Evaluation folders for the 15 runs: 2 successes and 3 failures per (run, exp)."""
    res = dict(scripted_v3=str(tmp_path / "a"), memory_design=str(tmp_path / "b"))
    for m, (key, _) in RF.MODELS.items():
        for s in RF.SEEDS:
            d = os.path.join(res[key], "eval", RF.run_name(m, s))
            os.makedirs(d)
            for exp in VC.EXPS:
                with open(os.path.join(d, exp + ".jsonl"), "w", encoding="utf-8") as fh:
                    for p in range(5):
                        ok = m != "BC" and p < 2
                        fh.write(json.dumps(_record(p + (0 if exp == "exp1" else 100), ok, "GRASP" if p < 4 else "TRANSPORT", "sha%d" % p, model=m)) + "\n")
    monkeypatch.setattr(RF, "typical_failures", lambda rows, k=2: [(r, "failure-%s (3/3 failures)" % r["fail_reason"]) for r in [x for x in rows if not x["success"]][:k]])
    monkeypatch.setattr(RF, "RESULTS", res)
    return res


def _fake_film(root, results_, items, tamper=None):
    """What record_exp_videos leaves: episodes.jsonl, roles.json, a video and an index.csv."""
    idx = {}
    for it in items:
        d = os.path.join(RF.video_root(root, it["key"]), it["run"], it["exp"])
        os.makedirs(d, exist_ok=True)
        stored = RF.read_jsonl(os.path.join(results_[it["key"]], "eval", it["run"], it["exp"] + ".jsonl"))
        pick = [r for r in stored if r["success"]] + [r for r, _ in RF.typical_failures(stored)]
        roles, rows = {}, []
        for r in pick:
            v = os.path.join(d, "clip_%s_%d.mp4" % (it["run"], r["seed"]))
            _video(v)
            role = "success" if r["success"] else "failure-%s (3/3 failures)" % r["fail_reason"]
            roles[str(r["seed"])] = role
            rerun = dict(r, video=v)
            if tamper and tamper(it, r):
                rerun["trajectory_sha256"] = "DIFFERENT"
            rows.append(rerun)
            idx.setdefault(it["key"], []).append(dict(run=it["run"], exp=it["exp"], seed=r["seed"], role=role, success=r["success"], fail_reason=r["fail_reason"],
                                                      phase_reached=r["phase_reached"], placement_error_m=r["placement_error_m"], ticks=r["ticks"], ckpt="c.pt",
                                                      sha_stored=r["trajectory_sha256"], sha_rerun=rerun["trajectory_sha256"],
                                                      hash_match=r["trajectory_sha256"] == rerun["trajectory_sha256"], video=v))
        with open(os.path.join(d, "episodes.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in rows)
        with open(os.path.join(d, "roles.json"), "w", encoding="utf-8") as fh:
            json.dump(roles, fh)
    for key, rows in idx.items():
        with open(os.path.join(RF.video_root(root, key), "index.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def test_expected_films_counts_successes_plus_two_failures(results):
    items = RF.expected_films(results, list(RF.MODELS), list(RF.SEEDS))
    assert len(items) == 5 * 3 * 2
    by = {(i["model"], i["seed"], i["exp"]): i for i in items}
    assert by[("ACT-LSTM", 1, "exp2")]["n"] == 4 and by[("ACT-LSTM", 1, "exp2")]["successes"] == 2     # 2 successes + 2 failures
    assert by[("BC", 0, "exp1")]["n"] == 2 and by[("BC", 0, "exp1")]["successes"] == 0
    assert by[("ACT", 2, "exp1")]["run"] == "act_orig_n40_seed2" and by[("BC", 0, "exp1")]["run"] == "bc_b256_n40_seed0"
    assert by[("ACT-12", 0, "exp1")]["key"] == "scripted_v3" and by[("ACT-LSTM", 0, "exp1")]["key"] == "memory_design"


def test_estimate_scales_with_ticks(results):
    items = RF.expected_films(results, list(RF.MODELS), list(RF.SEEDS))
    e = RF.estimate_seconds(items)
    assert e["total_s"] == e["film_s"] + e["side_by_side_s"] + e["reel_s"] and e["disk_gb"] > 0


def test_film_skips_what_exists_retries_once_and_reports_what_is_missing(results, tmp_path):
    root = str(tmp_path / "out")
    items = RF.expected_films(results, ["BC"], [0, 1])
    _fake_film(root, results, [i for i in items if i["seed"] == 0])       # seed 0 is finished already
    calls = []
    assert RF.film(root, results, items, runner=lambda args, lg: calls.append(args)) == ["bc_b256_n40_seed1 exp1", "bc_b256_n40_seed1 exp2"]
    assert len(calls) == 2 and all("bc_b256_n40_seed1" in c for c in calls)   # one run, tried twice
    assert "--failures-per" in calls[0] and calls[0][calls[0].index("--lock-rule") + 1] == "v3"

    def works(args, lg):
        _fake_film(root, results, items)
    assert RF.film(root, results, items, runner=works) == []


def _doc(seed, picks):
    return dict(runs=[RF.run_name(m, seed) for m in RF.PANEL_MODELS], labels={RF.run_name(m, seed): m for m in RF.PANEL_MODELS}, results=["r"], picks=picks)


def _pick(cat, exp, seed, **kw):
    return dict(category=cat, exp=exp, seed=seed, why=cat, **kw)


def test_select_picks_follows_the_plan_and_skips_a_position_already_taken():
    docs = {0: _doc(0, [_pick("contrast", "exp2", 10), _pick("contrast_reverse", "exp2", 11), _pick("contrast_reverse", "exp2", 12), _pick("contrast_reverse", "exp2", 13)]),
            1: _doc(1, [_pick("contrast", "exp2", 20), _pick("contrast", "exp2", 21), _pick("contrast", "exp2", 22), _pick("contrast_reverse", "exp1", 23),
                        _pick("typical_failure", "exp2", 20, run="act_lstm_full_n40_seed1"), _pick("typical_failure", "exp2", 24, run="act_lstm_full_n40_seed1"),
                        _pick("typical_failure", "exp2", 25, run="act_orig_n40_seed1")]),
            2: _doc(2, [_pick("contrast", "exp2", 30), _pick("contrast", "exp2", 31), _pick("typical_failure", "exp2", 32, run="act_orig_n40_seed2")])}
    sel = RF.select_picks(docs)
    got = [(s["train_seed"], s["pick"]["category"], s["pick"]["exp"], s["pick"]["seed"]) for s in sel]
    assert got == [(1, "contrast", "exp2", 20), (1, "contrast", "exp2", 21), (2, "contrast", "exp2", 30), (2, "contrast", "exp2", 31), (0, "contrast", "exp2", 10),
                   (0, "contrast_reverse", "exp2", 11), (0, "contrast_reverse", "exp2", 12), (1, "contrast_reverse", "exp1", 23),
                   (1, "typical_failure", "exp2", 24), (2, "typical_failure", "exp2", 32)]      # 20 was taken: ACT-LSTM's next candidate
    assert sum(n for _, _, _, n, _ in RF.PICK_PLAN) == 10


def test_side_by_side_skips_a_finished_video_and_reports_a_missing_one(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "side_by_side", "picks"))
    done = dict(train_seed=1, pick=_pick("contrast", "exp2", 20), runs=["r"], labels={}, results=["x"])
    todo = dict(train_seed=1, pick=_pick("contrast", "exp2", 21), runs=["r"], labels={}, results=["x"])
    d = os.path.join(root, "side_by_side", "exp2", "seed0020")
    os.makedirs(d)
    open(os.path.join(d, "panels_bc_b256_n40_seed1__x.json"), "w").close()
    calls = []
    miss = RF.side_by_side(root, [done, todo], runner=lambda a, lg: calls.append(a))
    assert len(calls) == 2 and len(miss) == 1 and "position 21" in miss[0]
    assert json.load(open(os.path.join(root, "side_by_side", "picks", "pick_01.json")))["picks"][0]["seed"] == 21


def test_relabelled_clip_has_the_same_frames_and_no_code_name(tmp_path):
    src, dst = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    _video(src)
    n = RF.relabel_clip(src, dst, _record(5, True, "", "s", ticks=TICKS), "ACT-LSTM", 1, "exp2")
    assert n == TICKS and VC.probe(dst)["frames"] == TICKS and (VC.probe(dst)["w"], VC.probe(dst)["h"]) == (VC.SRC_W, VC.SRC_H)
    frame = next(iter(VC.iter_frames(dst)))
    assert frame[:VC.OVERLAY_BAR_PX].max() > 200                       # the new bar has white text


def test_reel_manifest_uses_thesis_names_and_pooled_tallies(results, tmp_path):
    root = str(tmp_path / "out")
    items = RF.expected_films(results, list(RF.MODELS), list(RF.SEEDS))
    _fake_film(root, results, items)
    man = RF.reel_manifest(root, results, list(RF.MODELS), list(RF.SEEDS))
    text = json.dumps([{k: v for k, v in s.items() if k != "video"} for s in man["segments"]])
    for code in ("act_orig", "act_lstm_full", "bc_b256", "act_n40", "act_lstm_n40"):
        assert code not in text, code
    cards = [s["title"] for s in man["segments"] if s["type"] == "card"]
    assert cards[1:6] == list(RF.MODELS)
    clips = [s for s in man["segments"] if s["type"] == "clip"]
    assert len(clips) == 4 + 5                                         # best success (not BC), one failure each
    assert all(os.path.basename(c["video"]).startswith(("ACT", "BC")) and os.path.exists(c["video"]) for c in clips)
    assert man["segments"][-1]["lines"][0] == "BC:  Exp 1  0/15    Exp 2  0/15"
    assert man["segments"][-1]["lines"][2] == "ACT-LSTM:  Exp 1  6/15    Exp 2  6/15"
    assert man["size"] == [1920, 1080]


def test_a_pushed_grasp_failure_says_so_in_its_caption(results, tmp_path):
    root = str(tmp_path / "out")
    items = RF.expected_films(results, ["ACT"], [0])
    _fake_film(root, results, items)
    for p in [os.path.join(RF.video_root(root, i["key"]), i["run"], i["exp"], "episodes.jsonl") for i in items]:
        rows = RF.read_jsonl(p)
        for r in rows:
            r["grasp_detail"] = dict(box_near_spawn=False, box_above_platform=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in rows)
    man = RF.reel_manifest(root, results, ["ACT"], [0])
    caps = [s["caption"] for s in man["segments"] if s["type"] == "clip"]
    assert any("typical failure: GRASP" in c and "pushed over 50 mm" in c for c in caps), caps
    assert not any("best success" in c and "pushed" in c for c in caps), caps


def test_a_mismatched_clip_stays_out_of_the_reel_and_is_reported(results, tmp_path):
    root = str(tmp_path / "out")
    items = RF.expected_films(results, list(RF.MODELS), list(RF.SEEDS))
    _fake_film(root, results, items, tamper=lambda it, r: r["success"])       # every success differs
    man = RF.reel_manifest(root, results, list(RF.MODELS), list(RF.SEEDS))
    assert not any(s["type"] == "clip" and "best success" in s.get("caption", "") for s in man["segments"])
    rows = RF.build_index(root, results)
    bad = RF.mismatches(rows)
    assert bad and all(b["result"] == "success" and "differs" in b["problem"] for b in bad)


def test_index_has_the_requested_columns_in_thesis_names(results, tmp_path):
    root = str(tmp_path / "out")
    items = RF.expected_films(results, ["BC", "ACT"], [1])
    _fake_film(root, results, items)
    d = os.path.join(root, "side_by_side", "exp2", "seed0100")
    os.makedirs(d)
    panel = lambda run, ok: dict(run=run, ckpt="c.pt", video_frames=TICKS, hash_match=True, rerun=dict(success=ok, fail_reason="" if ok else "GRASP", ticks=TICKS, trajectory_sha256="h"),
                                 stored=dict(success=ok, trajectory_sha256="h"))
    with open(os.path.join(d, "panels_x.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(exp="exp2", seed=100, box_xy=[1.5, 0.1], heldout=True, category="contrast", why="w", video=os.path.join(d, "s.mp4"),
                       panels=[panel("bc_b256_n40_seed1", False), panel("act_lstm_full_n40_seed1", True)]), fh)
    rows = RF.build_index(root, results)
    assert RF.mismatches(rows) == []
    with open(os.path.join(root, "index.csv"), encoding="utf-8-sig", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    for col in ("model", "seed", "exp", "position", "result", "failure_phase", "file"):
        assert col in csv_rows[0]
    per = [r for r in csv_rows if r["kind"] == "per_model"]
    assert {r["model"] for r in per} == {"BC", "ACT"} and {r["seed"] for r in per} == {"1"}
    fail = next(r for r in per if r["model"] == "BC")
    assert fail["result"] == "failure: GRASP" and fail["failure_phase"] == "GRASP" and fail["hash_match"] == "True"
    good = next(r for r in per if r["model"] == "ACT" and r["result"] == "success")
    assert good["failure_phase"] == "" and os.path.exists(good["file"]) and good["frames"] == str(TICKS)
    sbs = [r for r in csv_rows if r["kind"] == "side_by_side"]
    assert [(r["model"], r["result"]) for r in sbs] == [("BC", "failure: GRASP"), ("ACT-LSTM", "success")]
    assert sbs[0]["position"] == "100" and sbs[0]["role"] == "contrast"


def test_a_wrong_frame_count_or_outcome_counts_as_a_mismatch():
    base = dict(kind="per_model", model="ACT", seed=0, exp="exp1", position=1, hash_match=True, frames=10, ticks=10)
    assert RF.mismatches([base]) == []
    assert "frames" in RF.mismatches([dict(base, frames=9)])[0]["problem"]
    assert RF.mismatches([dict(base, hash_match=False)])
    assert RF.mismatches([dict(kind="reel", hash_match="", frames=5, ticks="")]) == []


def test_plan_runs_nothing(results, tmp_path, capsys):
    root = str(tmp_path / "never")
    assert RF.main(["--root", root, "--plan"]) == 0
    out = capsys.readouterr().out
    assert "estimate" in out and "ACT-LSTM-12" in out and not os.path.exists(root)
