"""Documentation videos of the finished scripted Exp 1 / Exp 2 results: one restart-safe job.

    python -u tools/render_final_videos.py --plan     # what it would film and how long; runs nothing
    python -u tools/render_final_videos.py            # D:\\thesis_videos\\scripted_final

In its own window, from PowerShell. Start-Process drops double quotes from -Command, so the title is in
single quotes, doubled inside the single-quoted argument:

    Start-Process powershell -WorkingDirectory "D:\\Charles_Aninon\\Thesis Project" -ArgumentList '-NoExit','-NoProfile','-ExecutionPolicy','Bypass','-Command','$Host.UI.RawUI.WindowTitle = ''Thesis videos: scripted_final''; python -u tools/render_final_videos.py'

Re-runs existing evaluation episodes only: no training, no new experiments. Stop it at any time and run
the same command again: finished work is kept, and at most one (run, experiment) is filmed again.

  1. PER MODEL. tools/record_exp_videos.py, 40 demos, 3 seeds: every success, plus 2 typical failures per
     model x seed x experiment (its rule: one from each of the two commonest failure phases, or both from
     the only one).
  2. SIDE BY SIDE. tools/pick_video_episodes.py + tools/video_side_by_side.py, BC | ACT | ACT-LSTM of one
     training seed on one box position, 10 picks (see PICK_PLAN).
  3. REEL. tools/video_highlight_reel.py: title cards, best success per model, typical failures, one
     side-by-side clip. Its clips are copies whose top bar reads thesis names instead of deploy's code
     names; the per-model files are not touched.
  4. index.csv (one row per video, one per panel of a side-by-side), mismatches.csv, status.json.

Every re-run's trajectory hash is compared with its evaluation record. A mismatch is reported, never
hidden, and a mismatched clip stays out of the reel. Exit 0: complete, all match. 1: mismatch. 2: incomplete.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import analyze_experiments as AE                                    # noqa: E402
import video_common as VC                                           # noqa: E402
import video_highlight_reel as RL                                   # noqa: E402

OUT_ROOT = r"D:\thesis_videos\scripted_final"
RESULTS = {"scripted_v3": os.path.join(ROOT, "runs", "scripted_v3"),
           "memory_design": os.path.join(ROOT, "runs", "scripted_v3_memory_design")}
#: Thesis name -> (results folder, run-folder prefix), in display order.
MODELS = {"BC": ("scripted_v3", "bc_b256"), "ACT": ("memory_design", "act_orig"),
          "ACT-LSTM": ("memory_design", "act_lstm_full"), "ACT-12": ("scripted_v3", "act"),
          "ACT-LSTM-12": ("scripted_v3", "act_lstm")}
SEEDS = (0, 1, 2)
FAILURES_PER = 2
LOCK_RULE = "v3"
#: Side-by-side selection, in order: (training seed, category, experiment, how many, typical-failure model).
#: Exp 2 reversals exist only on seed 0 (ACT's Exp 2 successes are all from it).
PICK_PLAN = ((1, "contrast", "exp2", 2, None), (2, "contrast", "exp2", 2, None), (0, "contrast", "exp2", 1, None),
             (0, "contrast_reverse", "exp2", 2, None), (1, "contrast_reverse", "exp1", 1, None),
             (1, "typical_failure", "exp2", 1, "ACT-LSTM"), (2, "typical_failure", "exp2", 1, "ACT"))
PANEL_MODELS = ("BC", "ACT", "ACT-LSTM")
#: Measured on this machine, 2026-11-05, with rendering (3 pilot episodes): seconds per tick, seconds per deploy start.
S_PER_TICK = {"BC": 0.015}
S_PER_TICK_DEFAULT = 0.025
S_PER_DEPLOY = 12.0
MB_PER_TICK = 0.0115
REEL_TARGET_S = 100.0
INDEX_COLS = ["kind", "model", "seed", "exp", "position", "box_x", "box_y", "heldout", "result", "failure_phase", "role", "ticks",
              "frames", "ckpt", "hash_match", "sha_stored", "sha_rerun", "file", "why"]


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    line = "[%s] %s" % (now(), msg)
    print(line, flush=True)
    if _LOG["path"]:
        with open(_LOG["path"], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


_LOG = {"path": None}


def run_name(model: str, seed: int) -> str:
    return "%s_n40_seed%d" % (MODELS[model][1], seed)


def video_root(root: str, results_key: str) -> str:
    """One record_exp_videos root per results folder (docs/VIDEOS.md)."""
    return os.path.join(root, "per_model", results_key)


def read_jsonl(path: str) -> List[dict]:
    return VC.read_jsonl(path) if os.path.exists(path) else []


def thesis_of(run: str) -> dict:
    p = AE.parse_run(run)
    return dict(model=p["model"] or run, seed=p["train_seed"])


# ─── what should exist ────────────────────────────────────────────────────────
def typical_failures(rows, k: int = FAILURES_PER) -> list:
    """record_exp_videos' own rule. Imported late: its module imports MuJoCo through eval_scripted."""
    import record_exp_videos as RV
    return RV.typical_failures(rows, k)


def expected_films(results: Dict[str, str], models: Sequence[str], seeds: Sequence[int]) -> List[dict]:
    """One item per (run, exp): which episodes, how many ticks."""
    items = []
    for m in models:
        key, _ = MODELS[m]
        for s in seeds:
            run = run_name(m, s)
            for exp in VC.EXPS:
                rows = read_jsonl(os.path.join(results[key], "eval", run, "%s.jsonl" % exp))
                if not rows:
                    raise SystemExit("no evaluation for %s %s under %s" % (run, exp, results[key]))
                pick = [r for r in rows if r["success"]] + [r for r, _ in typical_failures(rows)]
                items.append(dict(model=m, seed=s, run=run, exp=exp, key=key, n=len(pick), successes=sum(r["success"] for r in rows),
                                  ticks=sum(int(r["ticks"]) for r in pick)))
    return items


def estimate_seconds(items: Sequence[dict], n_picks: int = 10) -> dict:
    """Filming from measured rates, plus side-by-side panels and composition and the reel."""
    film = sum(i["ticks"] * S_PER_TICK.get(i["model"], S_PER_TICK_DEFAULT) + S_PER_DEPLOY for i in items if i["n"])
    side = n_picks * (2 * (S_PER_DEPLOY + 700 * S_PER_TICK_DEFAULT) + 75.0)     # about 2 of 3 panels are new
    reel = 240.0
    mb = sum(i["ticks"] for i in items) * MB_PER_TICK + n_picks * 3 * 900 * MB_PER_TICK
    return dict(film_s=film, side_by_side_s=side, reel_s=reel, total_s=film + side + reel, disk_gb=mb / 1000.0)


# ─── running the tools ────────────────────────────────────────────────────────
def run_tool(args: Sequence[str], logfile: str) -> int:
    """A tool in its own process, its output to a log. Judged by the files it writes, not by this code."""
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write("\n[%s] %s\n" % (now(), " ".join(args)))
        fh.flush()
        return subprocess.call([sys.executable, "-u"] + list(args), stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)


def film_done(root: str, it: dict) -> bool:
    return len(read_jsonl(os.path.join(video_root(root, it["key"]), it["run"], it["exp"], "episodes.jsonl"))) == it["n"]


def film(root: str, results: Dict[str, str], items: Sequence[dict], runner=run_tool) -> List[str]:
    """Step 1, one run at a time. Returns what is still missing."""
    rv = os.path.join(HERE, "record_exp_videos.py")
    runs = list(dict.fromkeys((i["run"], i["key"]) for i in items))
    total = sum(i["ticks"] for i in items)
    t0, ticks_at_start = time.time(), sum(i["ticks"] for i in items if film_done(root, i))
    for run, key in runs:
        mine = [i for i in items if i["run"] == run]
        if all(film_done(root, i) for i in mine):
            continue
        name = thesis_of(run)
        log("filming %s, training seed %s (%d episodes)" % (name["model"], name["seed"], sum(i["n"] for i in mine)))
        args = [rv, "--run-root", results[key], "--video-root", video_root(root, key), "--runs", run, "--failures-per", str(FAILURES_PER),
                "--lock-rule", LOCK_RULE]
        for attempt in (1, 2):
            runner(args, os.path.join(root, "logs", "per_model_%s.log" % run))
            if all(film_done(root, i) for i in mine):
                break
            if attempt == 1:
                log("  %s incomplete: one more try" % run)
        done = sum(i["ticks"] for i in items if film_done(root, i))
        new = done - ticks_at_start
        if new > 0:
            left = (total - done) * (time.time() - t0) / new
            log("  %d of %d ticks filmed, about %.0f min of filming left" % (done, total, left / 60.0))
    return ["%s %s" % (i["run"], i["exp"]) for i in items if not film_done(root, i)]


def select_picks(docs: Dict[int, dict], plan=PICK_PLAN) -> List[dict]:
    """The side-by-side list, from one pick_video_episodes document per training seed. Skips a box position already taken for that
    seed and experiment, and takes the next candidate instead. Fewer candidates than asked is fine, and shows in the count."""
    out, taken = [], set()
    for seed, category, exp, n, model in plan:
        doc = docs.get(seed)
        if doc is None:
            continue
        cands = [p for p in doc["picks"] if p["category"] == category and p["exp"] == exp]
        if model:
            cands = [p for p in cands if p.get("run") == run_name(model, seed)]
        got = 0
        for p in cands:
            if got >= n:
                break
            if (seed, exp, p["seed"]) in taken:
                continue
            taken.add((seed, exp, p["seed"]))
            out.append(dict(train_seed=seed, pick=p, runs=doc["runs"], labels=doc["labels"], results=doc["results"]))
            got += 1
    return out


def make_picks(root: str, results: Dict[str, str], seeds: Sequence[int], runner=run_tool) -> Dict[int, dict]:
    """tools/pick_video_episodes.py per training seed (eval JSONLs only, deterministic)."""
    d = os.path.join(root, "side_by_side", "picks")
    os.makedirs(d, exist_ok=True)
    docs = {}
    for s in seeds:
        path = os.path.join(d, "all_seed%d.json" % s)
        if not os.path.exists(path):
            specs = ["%s=%s" % (run_name(m, s), m) for m in PANEL_MODELS]
            runner([os.path.join(HERE, "pick_video_episodes.py"), "--results", results["scripted_v3"], results["memory_design"], "--runs"] + specs
                   + ["--contrast", "%s:%s" % (run_name("ACT-LSTM", s), run_name("ACT", s)), "--per-category", "3", "--out", path],
                   os.path.join(root, "logs", "picks.log"))
        docs[s] = VC.load_json(path)
        if docs[s] is None:
            raise SystemExit("pick_video_episodes wrote no %s: see logs/picks.log" % path)
    return docs


def panels_done(root: str, sel: dict) -> Optional[str]:
    p = sel["pick"]
    files = glob.glob(os.path.join(root, "side_by_side", p["exp"], "seed%04d" % p["seed"], "panels_*seed%d*.json" % sel["train_seed"]))
    return files[0] if files else None


def side_by_side(root: str, selection: Sequence[dict], runner=run_tool) -> List[str]:
    """Step 2, one pick at a time, so a finished video is never rendered again."""
    d = os.path.join(root, "side_by_side")
    reuse = [video_root(root, k) for k in RESULTS]
    missing = []
    for i, sel in enumerate(selection):
        p = sel["pick"]
        tag = "pick %d/%d: %s %s, position %d, training seed %d" % (i + 1, len(selection), p["category"], p["exp"], p["seed"], sel["train_seed"])
        if panels_done(root, sel):
            continue
        log("side by side %s" % tag)
        one = os.path.join(d, "picks", "pick_%02d.json" % i)
        with open(one, "w", encoding="utf-8") as fh:
            json.dump(dict(results=sel["results"], runs=sel["runs"], labels=sel["labels"], picks=[p]), fh, indent=1)
        args = [os.path.join(HERE, "video_side_by_side.py"), "--picks", one, "--out-dir", d, "--reuse"] + reuse
        for attempt in (1, 2):
            runner(args, os.path.join(root, "logs", "side_by_side.log"))
            if panels_done(root, sel):
                break
            if attempt == 1:
                log("  no video yet: one more try")
        if not panels_done(root, sel):
            missing.append(tag)
    return missing


# ─── the reel ─────────────────────────────────────────────────────────────────
def per_model_rows(root: str) -> List[dict]:
    """Every record_exp_videos index row, from each results folder's own index.csv."""
    rows = []
    for key in RESULTS:
        p = os.path.join(video_root(root, key), "index.csv")
        if os.path.exists(p):
            rows += RL.read_csv(p)
    return rows


def thesis_bar(img, text: str) -> None:
    """Replace deploy's top bar, which ends in the run's code name."""
    cv2.rectangle(img, (0, 0), (img.shape[1], VC.OVERLAY_BAR_PX), VC.BLACK, -1)
    VC.put_text(img, text, (14, 31), VC.fit_scale(text, img.shape[1] - 28, 0.8, 2), VC.WHITE, 2, shadow=False)


def relabel_clip(src: str, dst: str, row: dict, model: str, train_seed: int, exp: str) -> int:
    """A copy of a per-model clip whose bar says what deploy's said, with the thesis name. `row` is the re-run's record."""
    import video_side_by_side as VS
    n_ticks = int(row.get("ticks") or 0)
    w, h = VC.SRC_W, VC.SRC_H
    with VC.Mp4Writer(dst, w, h, VC.TICK_HZ) as out:
        for t, frame in enumerate(VC.iter_frames(src)):
            thesis_bar(frame, "%s, training seed %d   %s, position %d   t %5.2f s   %s   weld %s   lock %s" % (
                model, train_seed, VC.EXP_LABEL[exp].split(" (")[0], int(row["seed"]), t / float(VC.TICK_HZ),
                VS.phase_at(row.get("phase_boundaries"), t, n_ticks or t + 1), "ON" if VS.weld_on(row, t) else "off",
                "ON" if VS.lock_on(row, t) else "off"))
            out.write(frame)
    return out.n


def pooled_tally(results: Dict[str, str], models: Sequence[str], seeds: Sequence[int]) -> Dict[str, Dict[str, tuple]]:
    out = {}
    for m in models:
        key, _ = MODELS[m]
        out[m] = {}
        for exp in VC.EXPS:
            rows = [r for s in seeds for r in read_jsonl(os.path.join(results[key], "eval", run_name(m, s), "%s.jsonl" % exp))]
            out[m][exp] = (sum(bool(r["success"]) for r in rows), len(rows))
    return out


def reel_manifest(root: str, results: Dict[str, str], models: Sequence[str], seeds: Sequence[int], target_s: float = REEL_TARGET_S) -> dict:
    """The tool's automatic plan, with thesis names in every caption and on every clip's bar."""
    idx = per_model_rows(root)
    sbs_path = os.path.join(root, "side_by_side", "side_by_side_index.csv")
    sbs = RL.read_csv(sbs_path) if os.path.exists(sbs_path) else []
    runs = [run_name(m, s) for m in models for s in seeds]
    labels = {run_name(m, s): m for m in models for s in seeds}
    man = RL.auto_manifest(idx, sbs, runs, labels, list(VC.EXPS), failures_per_model=1, sbs_max=1,
                           title="Bimanual box pickup on the Unitree G1",
                           data_label="learned from 40 scripted demonstrations, 3 training seeds per model")
    by_video = {os.path.normpath(r["video"]): r for r in idx}
    sbs_by_video = {os.path.normpath(r["video"]): r for r in sbs}
    clip_dir = os.path.join(root, "reel", "clips")
    os.makedirs(clip_dir, exist_ok=True)
    for seg in man["segments"]:
        if seg["type"] != "clip":
            continue
        key = os.path.normpath(seg["video"])
        if key in by_video:
            r = by_video[key]
            who = thesis_of(r["run"])
            stored = {int(x["seed"]): x for x in read_jsonl(os.path.join(os.path.dirname(r["video"]), "episodes.jsonl"))}
            dst = os.path.join(clip_dir, "%s_seed%d_%s_position%04d.mp4" % (VC.safe_name(who["model"]), who["seed"], r["exp"], int(r["seed"])))
            if not os.path.exists(dst):
                relabel_clip(r["video"], dst, stored[int(r["seed"])], who["model"], who["seed"], r["exp"])
            seg["video"] = dst
            seg["detail"] = "%s, training seed %d | %s, box position %s" % (who["model"], who["seed"], VC.EXP_LABEL[r["exp"]], r["seed"])
            # A clip can show the box carried and still read "GRASP": the weld came after the box was pushed (D19).
            rec = stored[int(r["seed"])]
            if rec.get("fail_reason") == "GRASP" and int(rec.get("weld_engage_tick", -1)) >= 0 \
                    and (rec.get("grasp_detail") or {}).get("box_near_spawn") is False:
                seg["caption"] += " - box pushed over 50 mm before the weld"
        elif key in sbs_by_video:
            r = sbs_by_video[key]
            train_seed = thesis_of(r["runs"].split(";")[0])["seed"]
            seg["detail"] = "%s, training seed %d | %s, box position %s" % (r["labels"].replace(";", " | "), train_seed, VC.EXP_LABEL[r["exp"]], r["seed"])
    end = man["segments"][-1]
    tally = pooled_tally(results, models, seeds)
    end["title"] = "Successes, all training seeds pooled"
    end["lines"] = ["%s:  Exp 1  %d/%d    Exp 2  %d/%d" % (m, *t["exp1"], *t["exp2"]) for m, t in tally.items()]
    end["lines"].append("Exp 1: fresh seeds from the training region | Exp 2: held-out patch")
    man["size"] = [1920, 1080]
    return RL.plan_timing(man, target_s, 4.0)


def h264_copy(src: str, dst: str, ffmpeg: Optional[str] = None) -> Optional[str]:
    """An H.264 copy, which PowerPoint plays. ffmpeg with libx264 if available, else Windows Media Foundation through OpenCV.
    Returns the file, or None with a log line if neither works: the mp4v reel still exists."""
    exe = ffmpeg or shutil.which("ffmpeg")
    if exe:
        shutil.copyfile(src, dst)
        VC.transcode_h264(dst, exe)
        return dst
    probe = VC.probe(src)
    tmp = os.path.splitext(dst)[0] + ".partial.mp4"
    w = cv2.VideoWriter(tmp, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), float(VC.TICK_HZ), (probe["w"], probe["h"]))
    if not w.isOpened():
        log("no H.264 encoder here (no ffmpeg, no Media Foundation): the reel stays mp4v")
        return None
    for f in VC.iter_frames(src):
        w.write(f)
    w.release()
    got = VC.probe(tmp) if os.path.exists(tmp) and os.path.getsize(tmp) else dict(frames=0)
    if got["frames"] != probe["frames"]:
        log("H.264 copy has %s frames, the reel %s: not kept" % (got["frames"], probe["frames"]))
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    os.replace(tmp, dst)
    return dst


def make_reel(root: str, results: Dict[str, str], models: Sequence[str], seeds: Sequence[int], force: bool = False) -> dict:
    """Step 3. Never simulates."""
    out = os.path.join(root, "reel", "highlight_reel.mp4")
    side = os.path.splitext(out)[0] + ".json"
    if os.path.exists(out) and os.path.exists(side) and not force:
        return VC.load_json(side)
    man = reel_manifest(root, results, models, seeds)
    with open(os.path.join(root, "reel", "reel_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    lo, hi = RL.REEL_RANGE_S
    log("reel plan: %d segments, %.0f s" % (len(man["segments"]), man["planned_seconds"]))
    if not lo <= man["planned_seconds"] <= hi:
        log("WARNING: the reel plan is outside %d-%d s" % (lo, hi))
    info = RL.render(man, out)
    doc = dict(man, rendered=info, h264=h264_copy(out, os.path.splitext(out)[0] + "_h264.mp4"),
               written=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    with open(side, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    log("reel: %.1f s, %s" % (info["seconds"], VC.rel(out)))
    return doc


# ─── the index ────────────────────────────────────────────────────────────────
def _frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    cap.release()
    return n


def _result(success, fail_reason) -> str:
    return "success" if success else "failure: %s" % (fail_reason or "?")


def build_index(root: str, results: Dict[str, str]) -> List[dict]:
    """One row per per-model video, per side-by-side panel and for the reel. The outcome and hash of a per-model video are compared
    with the evaluation record here, whatever the tools concluded."""
    rows = []
    for key, res in results.items():
        for p in sorted(glob.glob(os.path.join(video_root(root, key), "*", "exp[12]", "episodes.jsonl"))):
            exp_dir = os.path.dirname(p)
            run, exp = os.path.basename(os.path.dirname(exp_dir)), os.path.basename(exp_dir)
            stored = {int(r["seed"]): r for r in read_jsonl(os.path.join(res, "eval", run, "%s.jsonl" % exp))}
            roles = VC.load_json(os.path.join(exp_dir, "roles.json")) or {}
            who = thesis_of(run)
            for r in read_jsonl(p):
                s = stored.get(int(r["seed"]), {})
                box = s.get("box_xy") or [None, None]
                match = s.get("trajectory_sha256") == r.get("trajectory_sha256") and bool(s.get("success")) == bool(r.get("success"))
                rows.append(dict(kind="per_model", model=who["model"], seed=who["seed"], exp=exp, position=r["seed"], box_x=box[0], box_y=box[1],
                                 heldout=s.get("heldout"), result=_result(r["success"], r.get("fail_reason")),
                                 failure_phase="" if r["success"] else (r.get("fail_reason") or "?"), role=roles.get(str(r["seed"]), ""),
                                 ticks=r.get("ticks"), frames=_frame_count(r.get("video", "")), ckpt=(r.get("header") or {}).get("ckpt", {}).get("file"),
                                 hash_match=match, sha_stored=s.get("trajectory_sha256"), sha_rerun=r.get("trajectory_sha256"),
                                 file=r.get("video", ""), why=""))
    for p in sorted(glob.glob(os.path.join(root, "side_by_side", "exp*", "seed*", "panels_*.json"))):
        d = VC.load_json(p)
        box = d.get("box_xy") or [None, None]
        for pn in d["panels"]:
            who = thesis_of(pn["run"])
            re_, st = pn["rerun"], pn["stored"]
            rows.append(dict(kind="side_by_side", model=who["model"], seed=who["seed"], exp=d["exp"], position=d["seed"], box_x=box[0], box_y=box[1],
                             heldout=d.get("heldout"), result=_result(re_["success"], re_.get("fail_reason")),
                             failure_phase="" if re_["success"] else (re_.get("fail_reason") or "?"), role=d.get("category", ""), ticks=re_.get("ticks"),
                             frames=pn.get("video_frames"), ckpt=pn.get("ckpt"),
                             hash_match=bool(pn.get("hash_match")) and bool(st.get("success")) == bool(re_.get("success")),
                             sha_stored=st.get("trajectory_sha256"), sha_rerun=re_.get("trajectory_sha256"), file=d["video"], why=d.get("why", "")))
    side = VC.load_json(os.path.join(root, "reel", "highlight_reel.json"))
    if side:
        out = os.path.join(root, "reel", "highlight_reel.mp4")
        rows.append(dict(kind="reel", model=", ".join(MODELS), seed="", exp="exp1, exp2", position="", box_x="", box_y="", heldout="", result="",
                         failure_phase="", role="%.1f s" % side["rendered"]["seconds"], ticks="", frames=side["rendered"]["frames"], ckpt="", hash_match="",
                         sha_stored="", sha_rerun="", file=side.get("h264") or out, why=""))
    order = {m: i for i, m in enumerate(MODELS)}
    rows.sort(key=lambda r: ({"per_model": 0, "side_by_side": 1, "reel": 2}[r["kind"]], order.get(r["model"], 9), str(r["seed"]), r["exp"], str(r["position"])))
    with open(os.path.join(root, "index.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def mismatches(rows: Sequence[dict]) -> List[dict]:
    """A hash or outcome that differs from the evaluation record, or a video whose frames are not one per tick."""
    bad = []
    for r in rows:
        if r["kind"] == "reel":
            continue
        if not r["hash_match"]:
            bad.append(dict(r, problem="re-run differs from the evaluation record"))
        elif r["frames"] != r["ticks"]:
            bad.append(dict(r, problem="video has %s frames, the episode %s ticks" % (r["frames"], r["ticks"])))
    return bad


def write_status(root: str, **fields) -> None:
    p = os.path.join(root, "status.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(fields, updated=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")), fh, indent=1)
    os.replace(tmp, p)


# ─── the job ──────────────────────────────────────────────────────────────────
def keep_awake(on: bool) -> None:
    """Windows may not sleep while this runs (it can still sleep on a closed lid, depending on the power plan)."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except (AttributeError, OSError):
        pass


def busy_others() -> List[str]:
    import eval_scripted as E
    return E._heavy_others(re.compile(r"train_bc\.py|train_policy\.py|deploy\.py|eval_scripted\.py|record_exp_videos\.py|video_side_by_side\.py|render_final_videos\.py"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=OUT_ROOT, help="where everything goes")
    ap.add_argument("--plan", action="store_true", help="say what would run and how long; run nothing")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS), help="a smaller trial: only these")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="a smaller trial: only these training seeds")
    ap.add_argument("--limit-picks", type=int, default=0, help="a smaller trial: at most N side-by-sides")
    a = ap.parse_args(argv)
    full = list(a.models) == list(MODELS) and list(a.seeds) == list(SEEDS) and not a.limit_picks
    items = expected_films(RESULTS, a.models, a.seeds)
    est = estimate_seconds(items, a.limit_picks or 10)
    n_eps, n_succ = sum(i["n"] for i in items), sum(i["successes"] for i in items)
    print("per-model videos: %d (%d successes + typical failures), %d ticks" % (n_eps, n_succ, sum(i["ticks"] for i in items)))
    print("estimate: filming %.0f min, side-by-sides %.0f min, reel %.0f min; total about %.1f h, %.1f GB -> %s"
          % (est["film_s"] / 60, est["side_by_side_s"] / 60, est["reel_s"] / 60, est["total_s"] / 3600, est["disk_gb"], a.root))
    if a.plan:
        for it in items:
            print("  %-12s seed %d %s  %2d videos, %5d ticks" % (it["model"], it["seed"], it["exp"], it["n"], it["ticks"]))
        return 0
    busy = busy_others()
    if busy:
        raise SystemExit("refusing: another heavy process is running (one at a time):\n  " + "\n  ".join(busy))
    os.makedirs(os.path.join(a.root, "logs"), exist_ok=True)
    _LOG["path"] = os.path.join(a.root, "job.log")
    if shutil.disk_usage(a.root).free < (est["disk_gb"] + 2.0) * 1e9:
        raise SystemExit("not enough disk under %s for about %.1f GB" % (a.root, est["disk_gb"]))
    keep_awake(True)
    t0 = time.time()
    log("started, %d per-model videos planned%s" % (n_eps, "" if full else " (a smaller trial)"))
    write_status(a.root, state="running", step="per_model", planned=n_eps)
    try:
        missing = film(a.root, RESULTS, items)
        write_status(a.root, state="running", step="side_by_side", planned=n_eps)
        seeds_for_sbs = [s for s in SEEDS if s in a.seeds]
        selection = select_picks(make_picks(a.root, RESULTS, seeds_for_sbs))
        if a.limit_picks:
            selection = selection[:a.limit_picks]
        with open(os.path.join(a.root, "side_by_side", "picks", "selected.json"), "w", encoding="utf-8") as fh:
            json.dump([dict(train_seed=s["train_seed"], category=s["pick"]["category"], exp=s["pick"]["exp"], position=s["pick"]["seed"],
                            why=s["pick"]["why"]) for s in selection], fh, indent=1)
        missing += side_by_side(a.root, selection)
        write_status(a.root, state="running", step="reel", planned=n_eps)
        make_reel(a.root, RESULTS, a.models, a.seeds, force=True)
    except SystemExit as e:
        log("STOPPED: %s" % e)
        write_status(a.root, state="stopped", error=str(e))
        keep_awake(False)
        return 2
    rows = build_index(a.root, RESULTS)
    bad = mismatches(rows)
    with open(os.path.join(a.root, "mismatches.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["problem"] + INDEX_COLS)
        w.writeheader()
        w.writerows(bad)
    kinds = Counter(r["kind"] for r in rows)
    code = 2 if missing else (1 if bad else 0)
    write_status(a.root, state="finished", exit_code=code, videos=dict(kinds), checked=sum(r["kind"] != "reel" for r in rows), mismatches=len(bad),
                 missing=missing, side_by_side_videos=len({r["file"] for r in rows if r["kind"] == "side_by_side"}), hours=round((time.time() - t0) / 3600.0, 2))
    log("finished in %.2f h: %d per-model videos, %d side-by-side panels, reel %s; %d mismatches; %d missing"
        % ((time.time() - t0) / 3600.0, kinds["per_model"], kinds["side_by_side"], "yes" if kinds["reel"] else "NO", len(bad), len(missing)))
    for b in bad:
        log("  MISMATCH %s seed %s %s position %s: %s" % (b["model"], b["seed"], b["exp"], b["position"], b["problem"]))
    for m in missing:
        log("  MISSING %s" % m)
    log("index: %s" % os.path.join(a.root, "index.csv"))
    keep_awake(False)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
