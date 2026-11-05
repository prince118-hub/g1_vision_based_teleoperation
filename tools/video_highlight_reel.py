"""A short (1-2 min) highlight montage from MP4s that already exist.

    python tools/video_highlight_reel.py --index D:/thesis_videos/scripted_exp/index.csv \
        --side-by-side D:/thesis_videos/side_by_side/side_by_side_index.csv \
        --runs bc_160_seed0 act_160_seed0 act_lstm_160_seed0 --results runs/scripted_v3 \
        --out D:/thesis_videos/reel_scripted_160.mp4
    ... --plan --write-manifest reel.json    # choose and time the clips, render nothing
    python tools/video_highlight_reel.py --manifest reel.json --out reel.mp4   # edited plan

Nothing is simulated. The clips are tools/record_exp_videos.py videos (its
index.csv: run, exp, seed, role, outcome, hash check, path) and
tools/video_side_by_side.py videos (its side_by_side_index.csv).

THE AUTOMATIC PLAN, in order:
  1. a title card (--title, --data-label, the experiments);
  2. per model (the run-name model, or a --runs NAME=LABEL label; runs of one
     model, e.g. its training seeds, are pooled):
       a card, then its BEST SUCCESS (lowest placement error, then fewest ticks,
       then lower seed), then --failures-per-model TYPICAL FAILURES (the index's
       failure-<bucket> clips, commonest bucket first; record_exp_videos chose
       each as the episode nearest its bucket's median). A model with no
       success says so on its card;
  3. if side-by-side videos are given, a card and up to --side-by-side-max of
     them, preferring the categories contrast, heldout_success,
     contrast_reverse, all_succeed, heldout_spread; each is captioned with its
     pick reason;
  4. an end card: Exp 1 / Exp 2 successes per run (counted from the eval JSONLs
     when --results is given) and how the clips were made.
Clips whose re-run did NOT reproduce the eval record's trajectory hash are left
out (--allow-mismatch puts them in, captioned RE-RUN != EVAL RECORD), as are
missing files.

TIMING: cards have fixed lengths. Every clip is shown WHOLE and sped up to fit
an equal share of --target-s (default 90 s). The speed is rounded up to 0.25x,
never below 1x, and capped at --max-speed. Every caption states the speed. The
planned length is printed, with a warning when it falls outside 60-120 s.
Captions sit on a shaded band at the bottom of the frame: line 1 says what the
clip shows, line 2 its source (run, experiment, seed, speed).
--write-manifest saves the plan as JSON (cards and clips, with their captions,
speeds, start/end) so it can be edited and rendered with --manifest. The
rendered reel gets a <reel>.json sidecar: the manifest plus the measured length.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import video_common as VC                                           # noqa: E402

TITLE_S, MODEL_CARD_S, SECTION_S, END_S = 5.0, 2.5, 2.5, 6.0
SBS_PREF = ("contrast", "heldout_success", "contrast_reverse", "all_succeed", "heldout_spread")
REEL_RANGE_S = (60.0, 120.0)
_ROLE_N = re.compile(r"\((\d+)/(\d+) failures\)")


def _bool(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


def _float(v, default=float("inf")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ─── choosing the clips ──────────────────────────────────────────────────────
def usable(row: dict, allow_mismatch: bool) -> bool:
    if not row.get("video") or not os.path.exists(row["video"]):
        return False
    if "hash_match" in row:
        return allow_mismatch or _bool(row["hash_match"])
    return allow_mismatch or int(row.get("hash_matches") or 0) == int(row.get("n_panels") or -1)


def group_rows(rows: Sequence[dict], runs: Sequence[str], labels: Dict[str, str]) -> Dict[str, List[dict]]:
    """{label: rows} in display order (the --runs order, else the model order)."""
    groups: Dict[str, List[dict]] = {}
    keep = set(runs)
    for r in rows:
        if keep and r["run"] not in keep:
            continue
        groups.setdefault(VC.run_label(r["run"], None, labels.get(r["run"])), []).append(r)
    if runs:
        order = []
        for n in runs:
            lb = VC.run_label(n, None, labels.get(n))
            if lb in groups and lb not in order:
                order.append(lb)
    else:
        order = sorted(groups, key=lambda lb: (min(VC.model_rank(VC.model_key(r["run"]))
                                                   for r in groups[lb]), lb))
    return {lb: groups[lb] for lb in order}


def best_success(rows: Sequence[dict]) -> Optional[dict]:
    ok = [r for r in rows if _bool(r["success"])]
    if not ok:
        return None
    return min(ok, key=lambda r: (_float(r.get("placement_error_m")), int(r.get("ticks") or 0),
                                  int(r["seed"])))


def typical_failures(rows: Sequence[dict], k: int) -> List[dict]:
    """The index's failure-<bucket> clips, commonest bucket first, one per bucket."""
    def n_of(r):
        m = _ROLE_N.search(r.get("role", ""))
        return int(m.group(1)) if m else 0
    fails = sorted((r for r in rows if r.get("role", "").startswith("failure-")),
                   key=lambda r: (-n_of(r), r.get("fail_reason", ""), r["exp"], int(r["seed"])))
    out, seen = [], set()
    for r in fails:
        if r.get("fail_reason") in seen:
            continue
        seen.add(r.get("fail_reason"))
        out.append(r)
        if len(out) >= k:
            break
    return out


def clip_frames(row: dict) -> int:
    v = row.get("ticks") or row.get("frames")
    return int(v) if v not in (None, "") else VC.probe(row["video"])["frames"]


def _exp_seed(row: dict) -> str:
    return "%s seed %s" % (VC.EXP_LABEL.get(row["exp"], row["exp"]).split(" (")[0], row["seed"])


def tallies(results: Sequence[VC.Results], runs: Sequence[str]) -> Dict[str, Dict[str, tuple]]:
    """{run: {exp: (successes, episodes)}} from the eval JSONLs."""
    where = VC.find_runs(results)
    out = {}
    for n in runs:
        if n in where:
            out[n] = {}
            for e in VC.EXPS:
                rows = VC.load_exp(where[n], n, e)
                if rows:
                    out[n][e] = (sum(bool(r["success"]) for r in rows.values()), len(rows))
    return out


def data_label_from(results: Sequence[VC.Results], runs: Sequence[str]) -> str:
    where = VC.find_runs(results)
    eps = set()
    for n in runs:
        meta = VC.run_meta(where[n], n) if n in where else None
        p = (((meta or {}).get("config") or {}).get("notes") or {}).get("data_paths", {}).get("episodes")
        if p:
            eps.add(os.path.basename(os.path.normpath(p)))
    return ("training data: %s" % ", ".join(sorted(eps))) if eps else ""


def auto_manifest(index_rows: Sequence[dict], sbs_rows: Sequence[dict], runs: Sequence[str],
                  labels: Dict[str, str], exps: Sequence[str], failures_per_model: int = 1,
                  sbs_max: int = 2, allow_mismatch: bool = False, title: str = "",
                  data_label: str = "", tally: Optional[dict] = None) -> dict:
    """The segment list, without timing (see plan_timing)."""
    rows = [r for r in index_rows if r["exp"] in exps and usable(r, allow_mismatch)]
    groups = group_rows(rows, runs, labels)
    segs: List[dict] = [dict(type="card", title=title or "Bimanual box pickup on the Unitree G1",
                             lines=[x for x in (data_label,
                                                "learned policies, closed loop in MuJoCo",
                                                " | ".join(VC.EXP_LABEL[e] for e in exps)) if x],
                             footer="Every clip is a re-run of an evaluated episode",
                             seconds=TITLE_S)]
    for lb, grp in groups.items():
        best = best_success(grp)
        fails = typical_failures(grp, failures_per_model)
        lines = ["best success" if best else "no successful episode among the filmed ones"]
        if fails:
            lines[0] += " and %s" % ("a typical failure" if len(fails) == 1 else
                                     "%d typical failures" % len(fails))
        segs.append(dict(type="card", title=lb, lines=lines, seconds=MODEL_CARD_S))
        for r, what in ([(best, "best success")] if best else []) + [(f, "typical failure") for f in fails]:
            if what == "best success":
                cap = "%s - best success, placement error %.1f cm" % (
                    lb, 100 * _float(r.get("placement_error_m"), float("nan")))
            else:
                m = _ROLE_N.search(r.get("role", ""))
                share = (" (%s of %s failures)" % m.groups()) if m else ""
                cap = "%s - typical failure: %s%s" % (lb, r.get("fail_reason") or "?", share)
            if not _bool(r.get("hash_match", True)):
                cap += " - RE-RUN != EVAL RECORD"
            segs.append(dict(type="clip", video=r["video"], caption=cap,
                             detail="%s | %s | %s" % (r["run"], r.get("ckpt", ""), _exp_seed(r)),
                             frames=clip_frames(r), start_s=0.0, end_s=None))
    sbs = [r for r in sbs_rows if r["exp"] in exps and usable(r, allow_mismatch)]
    if runs:
        sbs = [r for r in sbs if set(r["runs"].split(";")) <= set(runs)]
    sbs.sort(key=lambda r: (SBS_PREF.index(r["category"]) if r.get("category") in SBS_PREF
                            else len(SBS_PREF), r["exp"], int(r["seed"])))
    picked, cats = [], []
    for r in sbs:                                # one per category first, then the rest
        if r.get("category") not in cats:
            picked.append(r)
            cats.append(r.get("category"))
    picked += [r for r in sbs if r not in picked]
    picked = picked[:max(0, sbs_max)]
    if picked:
        segs.append(dict(type="card", title="Same box position, side by side",
                         lines=["one seed, every model, one clock",
                                "panels: model | outcome | time | phase | weld | lock"],
                         seconds=SECTION_S))
        for r in picked:
            cap = r.get("why") or "%s: %s" % (_exp_seed(r), r.get("outcomes", ""))
            if int(r.get("hash_matches") or 0) != int(r.get("n_panels") or -1):
                cap += " - RE-RUN != EVAL RECORD"
            segs.append(dict(type="clip", video=r["video"], caption=cap,
                             detail="%s | %s | box (%s, %s)" % (
                                 _exp_seed(r), r.get("labels", "").replace(";", " / "),
                                 _short(r.get("box_x")), _short(r.get("box_y"))),
                             frames=clip_frames(r), start_s=0.0, end_s=None))
    end_lines = []
    for n, t in (tally or {}).items():
        lb = VC.run_label(n, None, labels.get(n))
        end_lines.append("%s (%s): %s" % (lb, n, "   ".join(
            "%s %d/%d" % (VC.EXP_LABEL[e].split(" (")[0], k, m) for e, (k, m) in sorted(t.items()))))
    end_lines.append("Exp 1: fresh seeds from the training region | Exp 2: held-out patch")
    segs.append(dict(type="card", title="Successes per run" if tally else "Evaluation",
                     lines=end_lines,
                     footer=("Clips: tools/deploy.py re-runs, trajectory hash checked against the "
                             "eval record" + ("; mismatches included and marked" if allow_mismatch else "")),
                     seconds=END_S))
    return dict(size=[VC.SRC_W, VC.SRC_H], fps=VC.TICK_HZ, fade_s=0.3, segments=segs)


def _short(v) -> str:
    try:
        return "%.3f" % float(v)
    except (TypeError, ValueError):
        return "?"


def plan_timing(man: dict, target_s: float = 90.0, max_speed: float = 4.0) -> dict:
    """Give every clip without a speed the speed that fits it into an equal
    share of what the cards leave of target_s (whole clip, >= 1x, rounded up to
    0.25x, <= max_speed); return the manifest with seconds per segment."""
    fps = float(man.get("fps", VC.TICK_HZ))
    segs = man["segments"]
    cards = sum(float(s["seconds"]) for s in segs if s["type"] == "card")
    clips = [s for s in segs if s["type"] == "clip"]
    budget = max(1.0, (target_s - cards) / max(1, len(clips)))
    total = cards
    for s in clips:
        lo = int(round(float(s.get("start_s") or 0.0) * fps))
        if s.get("end_s") in (None, ""):
            if not s.get("frames"):                      # a hand-written manifest clip
                s["frames"] = VC.probe(s["video"])["frames"]
            hi = int(s["frames"])
        else:
            hi = int(round(float(s["end_s"]) * fps))
        length = max(1, hi - lo) / fps
        if not s.get("speed"):
            sp = max(1.0, math.ceil(length / budget * 4.0 - 1e-9) / 4.0)
            s["speed"] = min(sp, float(max_speed))
        s["seconds"] = round(math.ceil((hi - lo) / float(s["speed"])) / fps, 3)
        total += s["seconds"]
    man["planned_seconds"] = round(total, 2)
    return man


def segment_caption(s: dict) -> str:
    sp = float(s.get("speed") or 1.0)
    return s.get("detail", "") + ("" if abs(sp - 1) < 1e-9 else " | %gx speed" % sp)


# ─── rendering ───────────────────────────────────────────────────────────────
def faded(frames, nf: int):
    """Stream `frames` with a fade-in over the first nf and a fade-out over the
    last nf. Only nf frames are held back, so a clip never sits in memory whole."""
    def alpha(img, a):
        return img if a >= 1.0 else (img.astype(np.float32) * max(0.0, a)).astype(np.uint8)
    buf: deque = deque()
    k = 0
    for f in frames:
        buf.append(f)
        if len(buf) > nf:
            yield alpha(buf.popleft(), (k + 1) / float(nf) if nf else 1.0)
            k += 1
    left = len(buf)
    for j, f in enumerate(buf):
        yield alpha(f, min((k + 1) / float(nf), (left - j) / float(nf)))
        k += 1


def caption(img, line1: str, line2: str) -> None:
    h, w = img.shape[:2]
    k = h / 720.0
    band = int(72 * k)
    VC.shade(img, h - band, h, 0.7)
    VC.put_text(img, line1, (int(18 * k), h - band + int(30 * k)),
                VC.fit_scale(line1, w - int(36 * k), 0.75 * k, 2), VC.WHITE, 2)
    if line2:
        VC.put_text(img, line2, (int(18 * k), h - band + int(58 * k)),
                    VC.fit_scale(line2, w - int(36 * k), 0.5 * k), VC.GREY)


def clip_source_frames(path: str, start: int, stop: Optional[int], speed: float):
    """Source frames start, start+speed, ... (rounded), up to stop (exclusive)."""
    k, want = 0, start
    for i, f in enumerate(VC.iter_frames(path)):
        if stop is not None and i >= stop:
            return
        while i == want:
            yield f
            k += 1
            want = start + int(round(k * speed))


def render(man: dict, out_path: str) -> dict:
    W, H = (int(x) for x in man.get("size", (VC.SRC_W, VC.SRC_H)))
    W, H = VC.even(W), VC.even(H)
    fps = float(man.get("fps", VC.TICK_HZ))
    nf = int(round(float(man.get("fade_s", 0.3)) * fps))
    measured = []
    with VC.Mp4Writer(out_path, W, H, fps) as w:
        for s in man["segments"]:
            n0 = w.n
            if s["type"] == "card":
                img = VC.card(W, H, s["title"], s.get("lines", ()), s.get("footer", ""))
                n = max(1, int(round(float(s["seconds"]) * fps)))
                for f in faded((img for _ in range(n)), nf):
                    w.write(f)
            elif s["type"] == "clip":
                if not os.path.exists(s["video"]):
                    raise SystemExit("clip %s not found" % s["video"])
                lo = int(round(float(s.get("start_s") or 0.0) * fps))
                hi = None if s.get("end_s") in (None, "") else int(round(float(s["end_s"]) * fps))
                line2 = segment_caption(s)

                def framed(src):
                    for f in src:
                        img = VC.letterbox(f, W, H)
                        caption(img, s.get("caption", ""), line2)
                        yield img
                for f in faded(framed(clip_source_frames(s["video"], lo, hi,
                                                         float(s.get("speed") or 1.0))), nf):
                    w.write(f)
                if w.n == n0:
                    raise SystemExit("clip %s gave no frames in [%s, %s)" % (s["video"], lo, hi))
            else:
                raise SystemExit("unknown segment type %r" % s["type"])
            measured.append(round((w.n - n0) / fps, 3))
    return dict(frames=w.n, seconds=round(w.n / fps, 2), segment_seconds=measured)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", nargs="*", default=[], help="tools/record_exp_videos.py index.csv")
    ap.add_argument("--side-by-side", nargs="*", default=[], help="side_by_side_index.csv")
    ap.add_argument("--manifest", default=None, help="render this plan instead of choosing")
    ap.add_argument("--write-manifest", default=None, help="save the (timed) plan as JSON")
    ap.add_argument("--runs", nargs="*", default=[], metavar="RUN[=LABEL]",
                    help="runs to include, in order (default: every run in the index)")
    ap.add_argument("--exp", nargs="+", choices=VC.EXPS, default=list(VC.EXPS))
    ap.add_argument("--results", nargs="*", default=[],
                    help="results folder(s): end-card tallies and the data label")
    ap.add_argument("--title", default="")
    ap.add_argument("--data-label", default=None,
                    help="e.g. 'trained on 160 scripted demonstrations' (default: from --results)")
    ap.add_argument("--failures-per-model", type=int, default=1)
    ap.add_argument("--side-by-side-max", type=int, default=2)
    ap.add_argument("--allow-mismatch", action="store_true")
    ap.add_argument("--target-s", type=float, default=90.0)
    ap.add_argument("--max-speed", type=float, default=4.0)
    ap.add_argument("--size", default="%dx%d" % (VC.SRC_W, VC.SRC_H), help="WxH, e.g. 1920x1080")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plan", action="store_true", help="print the plan; render nothing")
    ap.add_argument("--h264", action="store_true")
    ap.add_argument("--ffmpeg", default=None)
    a = ap.parse_args(argv)
    if a.manifest:
        man = VC.load_json(a.manifest)
        if man is None:
            raise SystemExit("--manifest %s not found" % a.manifest)
        man = plan_timing(man, a.target_s, a.max_speed)
    else:
        if not a.index and not a.side_by_side:
            raise SystemExit("give --index and/or --side-by-side (or --manifest)")
        specs = [VC.parse_run_spec(s) for s in a.runs]
        runs, labels = [n for n, _ in specs], {n: lb for n, lb in specs if lb}
        idx = [r for p in a.index for r in read_csv(p)]
        sbs = [r for p in a.side_by_side for r in read_csv(p)]
        res = [VC.open_results(r) for r in a.results]
        tally_runs = runs or sorted({r["run"] for r in idx},
                                    key=lambda n: (VC.model_rank(VC.model_key(n)), n))
        man = auto_manifest(idx, sbs, runs, labels, a.exp, a.failures_per_model,
                            a.side_by_side_max, a.allow_mismatch, a.title,
                            a.data_label if a.data_label is not None else
                            (data_label_from(res, tally_runs) if res else ""),
                            tallies(res, tally_runs) if res else None)
        w, _, h = a.size.partition("x")
        man["size"] = [int(w), int(h)]
        man = plan_timing(man, a.target_s, a.max_speed)
    n_clips = sum(s["type"] == "clip" for s in man["segments"])
    print("plan: %d segments (%d clips), %.1f s" % (len(man["segments"]), n_clips, man["planned_seconds"]))
    for s in man["segments"]:
        if s["type"] == "card":
            print("  card %5.1f s  %s" % (float(s["seconds"]), s["title"]))
        else:
            print("  clip %5.1f s  x%-5g %s  [%s]" % (s["seconds"], s["speed"], s["caption"],
                                                     VC.rel(s["video"])))
    lo, hi = REEL_RANGE_S
    if not lo <= man["planned_seconds"] <= hi:
        print("WARNING: planned length %.1f s is outside %d-%d s (adjust --target-s, --max-speed, "
              "--failures-per-model or --side-by-side-max)" % (man["planned_seconds"], lo, hi))
    if n_clips == 0:
        print("WARNING: no usable clip (missing files, hash mismatches, or filters)")
    if a.write_manifest:
        with open(a.write_manifest, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1)
        print("-> %s" % VC.rel(a.write_manifest))
    if a.plan:
        return 0
    if not a.out:
        raise SystemExit("--out is required to render")
    info = render(man, a.out)
    if a.h264:
        VC.transcode_h264(a.out, a.ffmpeg)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(dict(man, rendered=info, written=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
                  fh, indent=1)
    print("%s: %d frames, %.1f s" % (VC.rel(a.out), info["frames"], info["seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
