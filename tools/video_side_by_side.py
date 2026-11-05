"""Side-by-side video of ONE box position (seed) for several models.

    python tools/video_side_by_side.py --results runs/scripted_v3 --exp exp1 --seed 1234 \
        --runs bc_160_seed0 act_160_seed0 act_lstm_160_seed0 --out-dir D:/thesis_videos/side_by_side
    python tools/video_side_by_side.py --picks picks.json --category contrast heldout_success \
        --out-dir D:/thesis_videos/side_by_side          # seeds from tools/pick_video_episodes.py
    ... --plan                                           # say what would run, run nothing

Every panel's episode is RE-RUN, not looked up. tools/deploy.py runs on the
checkpoint the evaluation used (header.ckpt.file in the run's eval record), with
--video-dir and every deploy option the evaluation used, all read from the
stored record's header: lock rule, lock-rule-mismatch flag, max ticks, temporal
ensemble on/off and horizon, walk-in mode. So the re-run is the same experiment.
Closed-loop deployment is deterministic and rendering only reads `data`
(tools/episode_view.py), so the re-run must reproduce the evaluated episode
BIT-FOR-BIT. Its trajectory_sha256 is compared with the eval record's. A
mismatch is NOT hidden: the panel gets a red frame and "RE-RUN != EVAL RECORD",
the index records it, and the exit code is 1. Expect mismatches on any machine
other than the one that ran the evaluation (CLAUDE.md O34).

PANELS. deploy's own top text bar is cropped (--keep-overlay keeps it) and
replaced by a header. The header shows:
  * the model label, plus the run's training-demo count and training seed;
  * the outcome: SUCCESS, or FAIL: <bucket>, where the bucket is deploy.py's
    fail_reason (the ScoredPhase the failure is charged to);
  * per frame: seed, time (tick / 25 Hz), and the offline phase label
    (phase_boundaries, g1_data/phase_label.py);
  * weld and base lock as the policy observes them at that tick (the state's
    weld bit: on for engage < t <= release; lock on from a lock tick until a
    release tick).
When a panel's episode ends (success hold, fall, or the tick cap), it freezes on
its last frame, darkened, with ENDED and its time. One clock drives every
panel: frame k is tick k. Title bar: experiment, seed, box spawn (x, y),
training region or held-out patch. Footer: how many re-runs matched their eval
hash.

Output (--out-dir):
  <exp>/seedNNNN/side_by_side_<exp>_seedNNNN_<tag>.mp4
  <exp>/seedNNNN/panels_<tag>.json   per panel: run, label, checkpoint, stored and
                                     re-run outcome, both hashes, deploy options, video
  _panels/<run>/<exp>/seedNNNN/      the re-run (deploy's episodes.jsonl, its MP4, a
                                     log). A finished one is reused, so a restart is safe
  side_by_side_index.csv             one row per side-by-side video, rebuilt from every
                                     panels_*.json present
--reuse ROOT takes a panel's video from tools/record_exp_videos.py output
(<ROOT>/<run>/<exp>/episodes.jsonl + MP4) when it holds the seed. The hash
check still applies.

ONE heavy process at a time: refuses to start while a train_bc.py or deploy.py runs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import video_common as VC                                           # noqa: E402

PHASE_ORDER = ("SETTLE", "REACH", "GRASP", "LIFT", "MOVE", "LOWER", "RELEASE", "VERIFY")
HEADER_H = 60            # px per panel at the default 640 px width (scaled with it)
TITLE_H, FOOTER_H = 44, 30
INDEX_COLS = ["exp", "seed", "box_x", "box_y", "heldout", "tag", "category", "why", "n_panels",
              "hash_matches", "runs", "labels", "outcomes", "frames", "video"]


# ─── what each tick shows ────────────────────────────────────────────────────
def phase_at(boundaries: Optional[dict], t: int, n_ticks: int) -> str:
    """The offline phase label at tick t from phase_label's boundaries (start
    tick per phase, non-decreasing; a phase never reached sits at the episode
    length). The last phase whose segment has started wins, as in label_episode."""
    if not boundaries:
        return "?"
    cur = "SETTLE"
    for p in PHASE_ORDER:
        b = boundaries.get(p)
        if b is not None and int(b) <= t and int(b) < n_ticks:
            cur = p
    return cur


def weld_on(row: dict, t: int) -> bool:
    """The state's weld bit at tick t. The weld engages (or releases) during a
    tick's physics, and the state is sampled before them, so it shows the
    change one tick later: on for engage < t <= release."""
    e, r = int(row.get("weld_engage_tick", -1)), int(row.get("weld_release_tick", -1))
    return e >= 0 and t > e and (r < 0 or t <= r)


def lock_on(row: dict, t: int) -> bool:
    """Base lock at tick t: the lock/release update runs BEFORE the observation,
    so a lock at tick L holds from L, a release at R frees from R."""
    locks = [x for x in row.get("lock_ticks", []) if x <= t]
    rels = [x for x in row.get("release_ticks", []) if x <= t]
    return bool(locks) and (not rels or max(locks) > max(rels))


# ─── the deploy options of the evaluation ────────────────────────────────────
def deploy_options(header: dict) -> dict:
    """Everything the stored episode was deployed with, as deploy.py flags.
    Fields absent from an older header take deploy.py's defaults of the time."""
    ck = (header.get("ckpt") or {}).get("file")
    if not ck:
        raise SystemExit("eval record header has no ckpt.file - cannot re-run it")
    lr = (header.get("lock_rule") or {}).get("version")
    if not lr:
        raise SystemExit("eval record header has no lock_rule.version - cannot re-run it")
    allow = header.get("lock_rule_mismatch_allowed")
    if allow is None:                         # before the flag: infer from the two rules
        tr = header.get("training_data_lock_rule")
        allow = bool(tr) and tr != lr
    extra: List[str] = []
    if header.get("max_ticks") is not None:
        extra += ["--max-ticks", str(int(header["max_ticks"]))]
    K = int(header.get("K") or 1)
    if K > 1 and header.get("temporal_ensemble") is False:
        extra += ["--no-ensemble"]
    h = header.get("ensemble_horizon")
    if h is not None and K > 1 and int(h) != K and header.get("temporal_ensemble") is not False:
        extra += ["--ensemble-horizon", str(int(h))]
    if header.get("walkin") not in (None, "policy"):
        extra += ["--walkin", str(header["walkin"])]
    if header.get("train_seeds_evaluated"):
        extra += ["--allow-train-seeds"]
    return dict(ckpt=ck, lock_rule=lr, allow_mismatch=bool(allow), extra=extra)


def deploy_command(run_dir: str, opts: dict, seed: int, pdir: str) -> List[str]:
    """For --plan and the panels file: the command eval_scripted.deploy runs."""
    return (["python", "tools/deploy.py", "--run", run_dir, "--ckpt", opts["ckpt"],
             "--lock-rule", opts["lock_rule"], "--out", os.path.join(pdir, "episodes.jsonl"),
             "--seed-list", str(seed), "--video-dir", pdir] + opts["extra"]
            + (["--allow-lock-rule-mismatch"] if opts["allow_mismatch"] else []))


def run_deploy(run_dir: str, opts: dict, seed: int, pdir: str) -> dict:
    """ONE episode through tools/eval_scripted.deploy (deploy.py in a subprocess,
    .part file, one retry, a finished episodes.jsonl is returned as is)."""
    import eval_scripted as E                     # imports MuJoCo; only when deploying
    rows = E.deploy(run_dir, opts["ckpt"],
                    ["--seed-list", str(seed), "--video-dir", pdir] + opts["extra"], 1,
                    os.path.join(pdir, "episodes.jsonl"), opts["lock_rule"],
                    os.path.join(pdir, "logs"), allow_mismatch=opts["allow_mismatch"])
    return rows[0]


def heavy_busy() -> List[str]:
    import re
    import eval_scripted as E
    return E._heavy_others(re.compile(r"train_bc\.py|deploy\.py"))


# ─── panels ──────────────────────────────────────────────────────────────────
@dataclass
class Panel:
    run: str
    label: str
    detail: str
    stored: dict                    # the eval record of this seed
    rerun: Optional[dict] = None    # deploy's record of the re-run
    video: Optional[str] = None
    source: str = ""                # deployed | cached | reused | planned
    options: dict = field(default_factory=dict)

    @property
    def hash_match(self) -> Optional[bool]:
        if self.rerun is None:
            return None
        return self.rerun.get("trajectory_sha256") == self.stored.get("trajectory_sha256")

    @property
    def shown(self) -> dict:
        """The record of what the VIDEO shows (the re-run), else the stored one."""
        return self.rerun if self.rerun is not None else self.stored


def panel_dir(out_dir: str, run: str, exp: str, seed: int) -> str:
    return os.path.join(out_dir, "_panels", run, exp, "seed%04d" % seed)


def find_reused(roots: Sequence[str], run: str, exp: str, seed: int):
    """(row, video) from a tools/record_exp_videos.py folder holding this seed."""
    for root in roots:
        p = os.path.join(root, run, exp, "episodes.jsonl")
        if not os.path.exists(p):
            continue
        for r in VC.read_jsonl(p):
            if int(r["seed"]) == seed and r.get("video") and os.path.exists(r["video"]):
                return r, r["video"]
    return None


def prepare_panel(res: VC.Results, run: str, label: Optional[str], exp: str, seed: int,
                  out_dir: str, reuse: Sequence[str] = (), plan: bool = False,
                  deploy_fn=None) -> Panel:
    stored = VC.load_exp(res, run, exp).get(seed)
    if stored is None:
        raise SystemExit("%s has no %s record for seed %d (%s) - only evaluated episodes "
                         "can be hash-checked" % (run, exp, seed, VC.rel(res.exp_path(run, exp))))
    meta = VC.run_meta(res, run)
    p = Panel(run=run, label=VC.run_label(run, stored.get("kind"), label),
              detail=VC.run_detail(run, meta), stored=stored,
              options=deploy_options(stored.get("header") or {}))
    sel = VC.load_json(os.path.join(res.eval_root, run, "selection.json"))
    if sel and sel.get("chosen") != p.options["ckpt"]:
        print("WARNING %s: eval record used %s, selection.json says %s - re-running what the "
              "record used" % (run, p.options["ckpt"], sel.get("chosen")))
    pdir = panel_dir(out_dir, run, exp, seed)
    done = os.path.join(pdir, "episodes.jsonl")
    got = find_reused(reuse, run, exp, seed)
    if got is not None:
        p.rerun, p.video, p.source = got[0], got[1], "reused"
    elif os.path.exists(done):
        p.rerun = VC.read_jsonl(done)[0]
        p.video, p.source = p.rerun.get("video"), "cached"
    elif plan:
        p.source = "planned"
    else:
        run_dir = res.run_dir(run)
        if not os.path.exists(os.path.join(run_dir, "metadata.json")):
            raise SystemExit("cannot re-run %s: %s has no metadata.json (the run folder "
                             "must sit beside its eval folder)" % (run, VC.rel(run_dir)))
        p.rerun = (deploy_fn or run_deploy)(run_dir, p.options, seed, pdir)
        p.video, p.source = p.rerun.get("video"), "deployed"
    if p.rerun is not None and int(p.rerun["seed"]) != seed:
        raise SystemExit("%s: re-run record is seed %s, expected %d" % (done, p.rerun["seed"], seed))
    if p.source != "planned" and (not p.video or not os.path.exists(p.video)):
        raise SystemExit("%s seed %d: the re-run's video %r is missing - delete %s to re-run"
                         % (run, seed, p.video, VC.rel(pdir)))
    return p


# ─── composition ─────────────────────────────────────────────────────────────
class _Track:
    """Lock-step reader: seek(s) advances to source frame s, holding the last
    frame once the video has ended."""

    def __init__(self, frames: Iterable[np.ndarray]):
        self.it = iter(frames)
        self.pos, self.cur, self.done = -1, None, False

    def seek(self, s: int):
        while not self.done and self.pos < s:
            try:
                f = next(self.it)
            except StopIteration:
                self.done = True
                break
            self.pos, self.cur = self.pos + 1, f
        if self.cur is None:
            raise SystemExit("a panel video has no frames")
        return self.cur, self.done and self.pos < s

    @property
    def n(self) -> int:
        return self.pos + 1


def layout(n: int, cols: Optional[int], panel_w: int, src_w: int = VC.SRC_W,
           src_h: int = VC.SRC_H, crop_top: int = VC.OVERLAY_BAR_PX) -> dict:
    """Panel grid geometry. Default columns: all in one row up to 3, else 2 rows."""
    cols = int(cols) if cols else (n if n <= 3 else int(math.ceil(n / 2.0)))
    rows = int(math.ceil(n / float(cols)))
    pw = VC.even(panel_w)
    ph_img = VC.even(round(pw * (src_h - crop_top) / float(src_w)))
    hh = VC.even(round(HEADER_H * pw / 640.0))
    W = cols * pw
    H = TITLE_H + rows * (hh + ph_img) + FOOTER_H
    cells = [(TITLE_H + (i // cols) * (hh + ph_img), (i % cols) * pw) for i in range(n)]
    return dict(cols=cols, rows=rows, panel_w=pw, img_h=ph_img, header_h=hh, W=VC.even(W),
                H=VC.even(H), cells=cells, crop_top=crop_top)


def draw_panel(src, row: dict, label: str, detail: str, hash_match: Optional[bool],
               tick: int, ended: bool, geo: dict):
    """One panel (header + picture) for source frame `tick` of `row`'s episode."""
    pw, ih, hh = geo["panel_w"], geo["img_h"], geo["header_h"]
    k = pw / 640.0
    out = VC.blank(pw, hh + ih)
    pic = VC.resize(src[geo["crop_top"]:], pw, ih)
    n_ticks = int(row.get("ticks") or 0) or tick + 1
    t_show = min(tick, n_ticks - 1)
    if ended:
        pic = VC.darken(pic)
        msg = "ENDED  %.2f s" % (n_ticks / float(VC.TICK_HZ))
        s = 0.9 * k
        w, h = VC.text_size(msg, s, 2)
        VC.put_text(pic, msg, ((pw - w) // 2, (ih + h) // 2), s, VC.WHITE, 2)
        why = {"success": "success held 2.5 s", "fell": "fell", "time_limit": "tick cap"}.get(
            row.get("end_reason"), row.get("end_reason") or "")
        if why:
            s2 = 0.55 * k
            w2 = VC.text_size(why, s2)[0]
            VC.put_text(pic, why, ((pw - w2) // 2, (ih + h) // 2 + int(30 * k)), s2, VC.GREY)
    out[hh:] = pic
    # header row 1: label + detail ... outcome badge
    ok = bool(row.get("success"))
    bx = VC.badge(out, VC.outcome_text(row), pw - int(8 * k), int(6 * k), int(26 * k),
                  VC.GREEN if ok else VC.RED, 0.55 * k)
    ls = VC.fit_scale(label, int(bx - 16 * k), 0.68 * k, 2)
    x = VC.put_text(out, label, (int(10 * k), int(26 * k)), ls, VC.WHITE, 2)
    if detail and x + 12 * k < bx:
        ds = VC.fit_scale(detail, int(bx - x - 18 * k), 0.45 * k)
        VC.put_text(out, detail, (x + int(10 * k), int(26 * k)), ds, VC.GREY)
    # header row 2: per-tick state
    line = "seed %d   t %5.2f s   phase %s   weld %s   lock %s" % (
        int(row["seed"]), t_show / float(VC.TICK_HZ),
        phase_at(row.get("phase_boundaries"), t_show, n_ticks),
        "ON" if weld_on(row, t_show) else "off", "ON" if lock_on(row, t_show) else "off")
    if hash_match is False:
        line += "   RE-RUN != EVAL RECORD"
    VC.put_text(out, line, (int(10 * k), int(50 * k)), VC.fit_scale(line, pw - int(20 * k), 0.46 * k),
                VC.RED if hash_match is False else VC.GREY)
    if hash_match is False:
        cv2.rectangle(out, (0, 0), (pw - 1, out.shape[0] - 1), VC.RED, max(2, int(4 * k)))
    return out


def title_text(exp: str, seed: int, box_xy, heldout: Optional[bool]) -> str:
    """'Exp 2 | seed 5012 | box (1.512, 0.034) m in the held-out patch' - the
    region is the record's own `heldout` flag, not the experiment's name."""
    box = "box (%.3f, %.3f) m" % (box_xy[0], box_xy[1]) if box_xy else "box ?"
    if heldout is not None:
        box += " in the %s" % ("held-out patch" if heldout else "training region")
    return "   |   ".join((VC.EXP_LABEL.get(exp, exp).split(" (")[0], "seed %d" % seed, box))


def compose(sources: Sequence[Tuple[Iterable[np.ndarray], dict, str, str, Optional[bool]]],
            out_path: str, title: str, footer: str, panel_w: int = 640,
            cols: Optional[int] = None, speed: float = 1.0, tail_s: float = 2.0,
            crop_top: int = VC.OVERLAY_BAR_PX, src_size=(VC.SRC_W, VC.SRC_H),
            fps: float = VC.TICK_HZ) -> dict:
    """Write the side-by-side MP4. `sources` = [(frames, row, label, detail,
    hash_match)], frames being BGR images at the source size, frame k = tick k.
    Returns {frames written, per-panel source frame counts, geometry}."""
    if speed <= 0:
        raise ValueError("speed must be > 0")
    geo = layout(len(sources), cols, panel_w, src_size[0], src_size[1], crop_top)
    tracks = [_Track(s[0]) for s in sources]
    tail = int(round(tail_s * fps))
    with VC.Mp4Writer(out_path, geo["W"], geo["H"], fps) as w:
        k, extra = 0, 0
        while True:
            s = int(round(k * speed))
            frame = VC.blank(geo["W"], geo["H"])
            ended_all = True
            for (src_frames, row, label, detail, hm), tr, (y, x) in zip(sources, tracks, geo["cells"]):
                img, ended = tr.seek(s)
                ended_all &= ended
                tick = tr.pos if ended else s
                p = draw_panel(img, row, label, detail, hm, tick, ended, geo)
                frame[y:y + p.shape[0], x:x + p.shape[1]] = p
            if ended_all:
                if extra >= tail:
                    break
                extra += 1
            VC.put_text(frame, title, (12, 30), VC.fit_scale(title, geo["W"] - 260, 0.7), VC.WHITE, 2)
            clock = "t %5.2f s" % (min(s, max(t.n for t in tracks) - 1) / float(VC.TICK_HZ))
            if abs(speed - 1.0) > 1e-9:
                clock += "   x%g speed" % speed
            cw = VC.text_size(clock, 0.6)[0]
            VC.put_text(frame, clock, (geo["W"] - cw - 12, 30), 0.6, VC.GREY)
            VC.put_text(frame, footer, (12, geo["H"] - 10),
                        VC.fit_scale(footer, geo["W"] - 24, 0.48), VC.GREY)
            w.write(frame)
            k += 1
    return dict(frames=w.n, panel_frames=[t.n for t in tracks], geometry=geo)


# ─── one side-by-side video ──────────────────────────────────────────────────
def default_tag(runs: Sequence[str]) -> str:
    t = "__".join(runs)
    if len(t) <= 80:
        return VC.safe_name(t)
    return "%dpanels_%s" % (len(runs), hashlib.sha1(t.encode()).hexdigest()[:8])


def footer_text(panels: Sequence[Panel]) -> str:
    n = sum(bool(p.hash_match) for p in panels)
    rules = sorted({p.options["lock_rule"] for p in panels})
    return ("Each panel: tools/deploy.py re-run of the evaluated episode on the run's evaluated "
            "checkpoint | trajectory hash = eval record: %d/%d | lock rule %s"
            % (n, len(panels), "/".join(rules)))


def render(panels: Sequence[Panel], exp: str, seed: int, out_dir: str, tag: str,
           category: str = "", why: str = "", **kw) -> dict:
    box = [p.stored.get("box_xy") for p in panels]
    if any(b != box[0] for b in box):
        print("WARNING seed %d: the runs' records disagree on the box spawn: %s" % (seed, box))
    held = panels[0].stored.get("heldout")
    d = os.path.join(out_dir, exp, "seed%04d" % seed)
    out = os.path.join(d, "side_by_side_%s_seed%04d_%s.mp4" % (exp, seed, tag))
    srcs = [(VC.iter_frames(p.video), p.shown, p.label, p.detail, p.hash_match) for p in panels]
    info = compose(srcs, out, title_text(exp, seed, box[0], held), footer_text(panels), **kw)
    record = dict(exp=exp, seed=seed, box_xy=box[0], heldout=held, tag=tag, category=category,
                  why=why, video=out, frames=info["frames"], geometry={
                      k: v for k, v in info["geometry"].items() if k != "cells"},
                  panels=[])
    for p, nf in zip(panels, info["panel_frames"]):
        sh = p.shown
        record["panels"].append(dict(
            run=p.run, label=p.label, detail=p.detail, ckpt=p.options["ckpt"],
            deploy_options=p.options, source=p.source, video=p.video, video_frames=nf,
            frames_match_ticks=nf == int(sh.get("ticks", -1)),
            stored=dict(success=p.stored.get("success"), fail_reason=p.stored.get("fail_reason"),
                        ticks=p.stored.get("ticks"), trajectory_sha256=p.stored.get("trajectory_sha256"),
                        device=(p.stored.get("header") or {}).get("device"),
                        git_commit=(p.stored.get("header") or {}).get("git_commit")),
            rerun=dict(success=sh.get("success"), fail_reason=sh.get("fail_reason"),
                       ticks=sh.get("ticks"), trajectory_sha256=sh.get("trajectory_sha256"),
                       device=(sh.get("header") or {}).get("device"),
                       git_commit=(sh.get("header") or {}).get("git_commit")),
            hash_match=p.hash_match))
        if nf != int(sh.get("ticks", -1)):
            print("WARNING %s seed %d: video has %d frames, episode %s ticks"
                  % (p.run, seed, nf, sh.get("ticks")))
    with open(os.path.join(d, "panels_%s.json" % tag), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    return record


def write_index(out_dir: str) -> List[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "exp*", "seed*", "panels_*.json"))):
        r = VC.load_json(p)
        rows.append(dict(
            exp=r["exp"], seed=r["seed"], box_x=(r["box_xy"] or [None])[0],
            box_y=(r["box_xy"] or [None, None])[1], heldout=r["heldout"], tag=r["tag"],
            category=r.get("category", ""), why=r.get("why", ""), n_panels=len(r["panels"]),
            hash_matches=sum(bool(x["hash_match"]) for x in r["panels"]),
            runs=";".join(x["run"] for x in r["panels"]),
            labels=";".join(x["label"] for x in r["panels"]),
            outcomes=";".join("SUCCESS" if x["rerun"]["success"] else
                              "FAIL:%s" % x["rerun"]["fail_reason"] for x in r["panels"]),
            frames=r["frames"], video=r["video"]))
    with open(os.path.join(out_dir, "side_by_side_index.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


# ─── CLI ─────────────────────────────────────────────────────────────────────
def jobs_from_args(a) -> List[dict]:
    """[{exp, seed, runs: [(name, label)], category, why}]"""
    specs = [VC.parse_run_spec(s) for s in (a.runs or [])]
    if a.picks:
        pk = VC.load_json(a.picks)
        if pk is None:
            raise SystemExit("--picks %s not found" % a.picks)
        cats = set(a.category or [])
        out, seen = [], set()
        for p in pk["picks"]:
            if cats and p["category"] not in cats:
                continue
            if a.exp and p["exp"] not in a.exp:
                continue
            runs = specs or [(r, (pk.get("labels") or {}).get(r)) for r in pk["runs"]]
            key = (p["exp"], p["seed"], tuple(n for n, _ in runs))
            if key in seen:                       # one seed picked for two reasons
                continue
            seen.add(key)
            out.append(dict(exp=p["exp"], seed=int(p["seed"]), runs=runs,
                            category=p["category"], why=p.get("why", "")))
        return out[:a.limit] if a.limit else out
    if not (a.seed and a.exp and specs):
        raise SystemExit("give --picks, or --exp, --seed and --runs")
    if len(a.exp) != 1:
        raise SystemExit("--seed takes ONE --exp")
    return [dict(exp=a.exp[0], seed=int(s), runs=specs, category="", why="") for s in a.seed]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=None,
                    help="results folder(s): run folders + eval/ (eval_scripted --run-root); "
                         "default: the ones recorded in --picks")
    ap.add_argument("--runs", nargs="+", default=None, metavar="RUN[=LABEL]",
                    help="panels, in order (default with --picks: the picks' runs)")
    ap.add_argument("--exp", nargs="+", choices=VC.EXPS, default=None)
    ap.add_argument("--seed", nargs="+", type=int, default=None, help="box position(s)")
    ap.add_argument("--picks", default=None, help="picks.json from tools/pick_video_episodes.py")
    ap.add_argument("--category", nargs="+", default=None, help="with --picks: only these")
    ap.add_argument("--limit", type=int, default=0, help="with --picks: at most N videos")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None, help="file-name tag (default: the run names)")
    ap.add_argument("--reuse", nargs="*", default=[],
                    help="tools/record_exp_videos.py video root(s) to take panels from")
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (frames skipped)")
    ap.add_argument("--tail-s", type=float, default=2.0, help="hold after the last panel ends")
    ap.add_argument("--keep-overlay", action="store_true",
                    help="keep deploy's own top text bar in each panel")
    ap.add_argument("--h264", action="store_true", help="re-encode with ffmpeg (libx264)")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--plan", action="store_true", help="print what would run; run nothing")
    a = ap.parse_args(argv)

    jobs = jobs_from_args(a)
    roots = a.results
    if not roots and a.picks:
        roots = VC.load_json(a.picks).get("results")
    if not roots:
        raise SystemExit("--results is required")
    where = VC.find_runs([VC.open_results(r) for r in roots])
    for j in jobs:
        for name, _ in j["runs"]:
            if name not in where:
                raise SystemExit("run %s not found under %s (found: %s)"
                                 % (name, ", ".join(roots), ", ".join(sorted(where)) or "none"))
    if not a.plan:
        busy = heavy_busy()
        if busy:
            raise SystemExit("refusing: another heavy process is running (one at a time):\n  "
                             + "\n  ".join(busy))
    os.makedirs(a.out_dir, exist_ok=True)
    print("%d side-by-side video(s) -> %s" % (len(jobs), a.out_dir))
    bad = 0
    for j in jobs:
        panels = [prepare_panel(where[n], n, lb, j["exp"], j["seed"], a.out_dir, a.reuse, a.plan)
                  for n, lb in j["runs"]]
        tag = a.tag or default_tag([p.run for p in panels])
        if a.plan:
            print("\n%s seed %d  [%s] %s" % (j["exp"], j["seed"], j["category"], j["why"]))
            for p in panels:
                if p.source == "planned":
                    cmd = deploy_command(where[p.run].run_dir(p.run), p.options, j["seed"],
                                         panel_dir(a.out_dir, p.run, j["exp"], j["seed"]))
                    ok = os.path.exists(os.path.join(where[p.run].run_dir(p.run), "metadata.json"))
                    print("  %-28s deploy: %s%s" % (p.run, " ".join(cmd), "" if ok else
                                                   "   <- RUN FOLDER MISSING: cannot re-run"))
                else:
                    print("  %-28s %s: %s (hash %s)" % (p.run, p.source, VC.rel(p.video),
                                                       "match" if p.hash_match else "MISMATCH"))
            continue
        rec = render(panels, j["exp"], j["seed"], a.out_dir, tag, j["category"], j["why"],
                     panel_w=a.panel_width, cols=a.cols, speed=a.speed, tail_s=a.tail_s,
                     crop_top=0 if a.keep_overlay else VC.OVERLAY_BAR_PX)
        if a.h264:
            VC.transcode_h264(rec["video"], a.ffmpeg)
        n_bad = sum(p.hash_match is False for p in panels)
        bad += n_bad
        print("%s seed %4d  %s  %s%s" % (
            j["exp"], j["seed"], "  ".join("%s=%s" % (p.label, VC.outcome_text(p.shown)) for p in panels),
            VC.rel(rec["video"]), ("  %d HASH MISMATCH(ES)" % n_bad) if n_bad else ""))
    if a.plan:
        return 0
    idx = write_index(a.out_dir)
    print("%d side-by-side videos in the index, %d panel hash mismatch(es) this pass -> %s"
          % (len(idx), bad, VC.rel(os.path.join(a.out_dir, "side_by_side_index.csv"))))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
