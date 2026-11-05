"""Shared pieces of the three video tools: results folders, run labels, eval
records, and frame composition. Imports neither MuJoCo nor torch.

    tools/pick_video_episodes.py   which seeds are worth filming (eval JSONLs only)
    tools/video_side_by_side.py    one seed, several models, one MP4 (re-runs deploy.py)
    tools/video_highlight_reel.py  a 1-2 min montage from MP4s that already exist

A RESULTS FOLDER is what tools/eval_scripted.py calls --run-root: the trained run
folders (<root>/<run>/metadata.json + checkpoints) with the evaluation beside them
(<root>/eval/<run>/selection.json, exp1.jsonl, exp2.jsonl). The tools do not care
whether the runs were trained on scripted or teleoperated demonstrations; they
read only what tools/deploy.py wrote. An eval folder can also be passed directly
(<dir>/<run>/exp1.jsonl). Its parent is then taken as the run root.

FRAMES are BGR uint8, OpenCV's channel order. tools/deploy.py --video-dir
(tools/episode_view.py) writes 1280x720 at 25 fps, ONE frame per 25 Hz control
tick: frame k is tick k, drawn at the tick's first physics step. A 44 px black
text bar sits on top of each frame.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TICK_HZ = 25            # episode_view.VIDEO_FPS: one frame per control tick (tested equal)
SRC_W, SRC_H = 1280, 720  # episode_view.VIDEO_W, VIDEO_H (tested equal)
OVERLAY_BAR_PX = 44     # episode_view._VideoSink: cv2.rectangle(img, (0, 0), (VIDEO_W, 44))

EXPS = ("exp1", "exp2")
EXP_LABEL = {"exp1": "Exp 1 (in-distribution)", "exp2": "Exp 2 (held-out patch)"}
#: Display order and labels. The W_o qualifiers keep the two ACT pairs apart
#: whatever D8 settles on: act / act_lstm are the W_o = 12 runs (Run A/B),
#: act_orig / act_lstm_full the memory-design pair (NOTES.md 2026-10-25b).
MODEL_ORDER = ("bc", "bc_b256", "act", "act_lstm", "act_orig", "act_lstm_full")
MODEL_LABEL = {"bc": "BC", "bc_b256": "BC (b256)", "act": "ACT (W_o 12)",
               "act_lstm": "ACT-LSTM (W_o 12)", "act_orig": "ACT (W_o 1)",
               "act_lstm_full": "ACT-LSTM (episode)"}
#: <model>_<size>_seed<k>. The size part is free text so teleop-trained runs
#: (e.g. act_lstm_teleop150_seed0) parse as well as scripted ones.
RUN_RE = re.compile(r"^(?P<model>act_lstm_full|act_lstm|act_orig|act|bc_b256|bc)_"
                    r"(?P<size>.+)_seed(?P<seed>\d+)$")


# ─── eval records ────────────────────────────────────────────────────────────
def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit("%s line %d is not JSON: %s" % (path, i + 1, e))
    return rows


@dataclass(frozen=True)
class Results:
    root: str          # run folders live here (eval_scripted --run-root)
    eval_root: str     # <root>/eval, or the folder given when it is an eval folder

    def run_dir(self, run: str) -> str:
        return os.path.join(self.root, run)

    def exp_path(self, run: str, exp: str) -> str:
        return os.path.join(self.eval_root, run, "%s.jsonl" % exp)


def _has_exp(d: str) -> bool:
    return any(glob.glob(os.path.join(d, "*", "%s.jsonl" % e)) for e in EXPS)


def open_results(path: str) -> Results:
    path = os.path.abspath(path)
    if os.path.isdir(os.path.join(path, "eval")) and _has_exp(os.path.join(path, "eval")):
        return Results(path, os.path.join(path, "eval"))
    if os.path.isdir(path) and _has_exp(path):
        return Results(os.path.dirname(path), path)
    raise SystemExit("%s: no evaluation found - expected <dir>/eval/<run>/exp1.jsonl "
                     "(a results folder) or <dir>/<run>/exp1.jsonl (an eval folder)" % path)


def find_runs(results: Sequence[Results]) -> Dict[str, Results]:
    """Every run with at least one exp JSONL -> its results folder. A run name in
    two results folders is refused: a panel must say which experiment it is."""
    out: Dict[str, Results] = {}
    for r in results:
        for p in sorted(glob.glob(os.path.join(r.eval_root, "*", "exp[12].jsonl"))):
            name = os.path.basename(os.path.dirname(p))
            if name in out and out[name] != r:
                raise SystemExit("run %s is in two results folders (%s, %s) - pass only one"
                                 % (name, out[name].eval_root, r.eval_root))
            out[name] = r
    return out


def load_exp(res: Results, run: str, exp: str) -> Dict[int, dict]:
    """{seed: deploy.py record} of one run and experiment; {} if not evaluated."""
    p = res.exp_path(run, exp)
    if not os.path.exists(p):
        return {}
    rows = read_jsonl(p)
    by = {int(r["seed"]): r for r in rows}
    if len(by) != len(rows):
        raise SystemExit("%s holds a seed twice" % p)
    return by


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ─── run names and labels ────────────────────────────────────────────────────
def parse_run_spec(s: str) -> Tuple[str, Optional[str]]:
    """'NAME' or 'NAME=LABEL' -> (name, label or None)."""
    name, _, label = s.partition("=")
    return name.strip(), (label.strip() or None)


def parse_run(name: str) -> dict:
    m = RUN_RE.match(name)
    if not m:
        return dict(model=None, size=None, train_seed=None)
    return dict(model=m.group("model"), size=m.group("size"), train_seed=int(m.group("seed")))


def model_key(name: str, kind: Optional[str] = None) -> str:
    """The run-name model (it tells bc_b256 from bc; deploy's `kind` cannot),
    else the record's kind, else the name."""
    return parse_run(name)["model"] or kind or name


def model_rank(key: str) -> int:
    return MODEL_ORDER.index(key) if key in MODEL_ORDER else len(MODEL_ORDER)


def run_label(name: str, kind: Optional[str] = None, override: Optional[str] = None) -> str:
    if override:
        return override
    k = model_key(name, kind)
    return MODEL_LABEL.get(k, k)


def n_train_demos(meta: Optional[dict]) -> Optional[int]:
    seeds = ((meta or {}).get("data") or {}).get("seeds")
    return len(seeds) if seeds else None


def run_detail(name: str, meta: Optional[dict] = None) -> str:
    """'160 demos, train seed 0' - the count from the run's metadata (what the
    loader held) when present, else the run name's size part."""
    p = parse_run(name)
    n = n_train_demos(meta)
    if n is not None:
        size = "%d demos" % n
    elif p["size"] is not None:
        m = re.fullmatch(r"n?(\d+)", p["size"])
        size = ("%s demos" % m.group(1)) if m else p["size"]
    else:
        return name
    return size + ("" if p["train_seed"] is None else ", train seed %d" % p["train_seed"])


def run_meta(res: Results, run: str) -> Optional[dict]:
    return load_json(os.path.join(res.run_dir(run), "metadata.json"))


def outcome_text(row: dict) -> str:
    """SUCCESS, or FAIL: <bucket> - deploy.py fail_reason, the ScoredPhase the
    failure is charged to ('fell' for a fall)."""
    if row.get("success"):
        return "SUCCESS"
    return "FAIL: %s" % (row.get("fail_reason") or "?")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", s).strip("-") or "x"


def rel(path: str) -> str:
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:                                    # another Windows drive
        return os.path.abspath(path)


# ─── drawing ─────────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (175, 175, 175)
BLACK = (0, 0, 0)
BG = (28, 28, 28)
GREEN = (70, 175, 60)          # BGR
RED = (50, 50, 215)
AMBER = (0, 165, 245)
_ASCII = {"–": "-", "—": "-", "×": "x", "·": "|", "≠": "!=",
          "→": "->", "≤": "<=", "≥": ">=", "±": "+-", "’": "'"}


def ascii_text(s: str) -> str:
    """Hershey fonts draw ASCII only."""
    s = "".join(_ASCII.get(c, c) for c in str(s))
    return s.encode("ascii", "replace").decode("ascii")


def text_size(s: str, scale: float, thick: int = 1) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(ascii_text(s), FONT, scale, thick)
    return w, h


def fit_scale(s: str, max_w: int, scale: float, thick: int = 1, min_scale: float = 0.3) -> float:
    while scale > min_scale and text_size(s, scale, thick)[0] > max_w:
        scale -= 0.02
    return max(scale, min_scale)


def put_text(img, s: str, xy, scale: float = 0.6, color=WHITE, thick: int = 1,
             shadow: bool = True) -> int:
    """Draw `s` with its baseline at xy; returns the x after it."""
    s = ascii_text(s)
    x, y = int(xy[0]), int(xy[1])
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), FONT, scale, BLACK, thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return x + text_size(s, scale, thick)[0]


def badge(img, s: str, x_right: int, y_top: int, h: int, color, scale: float = 0.55,
          thick: int = 1) -> int:
    """A filled label whose RIGHT edge is x_right; returns its left x."""
    w, th = text_size(s, scale, thick)
    pad = max(6, h // 4)
    x0 = x_right - w - 2 * pad
    cv2.rectangle(img, (x0, y_top), (x_right, y_top + h), color, -1)
    put_text(img, s, (x0 + pad, y_top + (h + th) // 2), scale, WHITE, thick, shadow=False)
    return x0


def resize(img, w: int, h: int):
    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (int(w), int(h)), interpolation=interp)


def letterbox(img, w: int, h: int, color=BLACK):
    """Scale to fit inside w x h, keep the aspect ratio, pad the rest."""
    s = min(w / img.shape[1], h / img.shape[0])
    nw, nh = max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))
    out = np.full((h, w, 3), color, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resize(img, nw, nh)
    return out


def darken(img, f: float = 0.45):
    return (img.astype(np.float32) * f).astype(np.uint8)


def blank(w: int, h: int, color=BG):
    return np.full((int(h), int(w), 3), color, dtype=np.uint8)


def shade(img, y0: int, y1: int, alpha: float = 0.65) -> None:
    """Darken rows y0:y1 in place (a caption band that keeps the picture)."""
    img[y0:y1] = (img[y0:y1].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def card(w: int, h: int, title: str, lines: Sequence[str] = (), footer: str = ""):
    """A title card: centred title, centred lines below it, small footer."""
    img = blank(w, h)
    ts = fit_scale(title, int(w * 0.9), 1.4 * h / 720, 2)
    tw, th = text_size(title, ts, 2)
    ls = 0.7 * h / 720
    gap = int(44 * h / 720)
    y = int(h * 0.42) - (len(lines) * gap) // 2
    put_text(img, title, ((w - tw) // 2, y), ts, WHITE, 2)
    y += int(th * 1.2) + gap // 2
    for ln in lines:
        s = fit_scale(ln, int(w * 0.9), ls)
        lw = text_size(ln, s)[0]
        put_text(img, ln, ((w - lw) // 2, y), s, GREY)
        y += gap
    if footer:
        s = fit_scale(footer, int(w * 0.94), 0.5 * h / 720)
        fw = text_size(footer, s)[0]
        put_text(img, footer, ((w - fw) // 2, h - int(28 * h / 720)), s, GREY)
    return img


# ─── video in / out ──────────────────────────────────────────────────────────
def iter_frames(path: str) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open video %s" % path)
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f
    finally:
        cap.release()


def probe(path: str) -> dict:
    """Frame count by DECODING (CAP_PROP_FRAME_COUNT is container metadata and
    can be off for mp4v), plus fps and size."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open video %s" % path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(TICK_HZ)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return dict(frames=n, fps=fps, w=w, h=h)


def even(n: int) -> int:
    n = int(n)
    return n + (n % 2)


class Mp4Writer:
    """mp4v writer (as tools/episode_view.py) into <stem>.partial.mp4, renamed
    to the final name only on a clean close - an interrupted render never
    leaves a file that looks finished."""

    def __init__(self, path: str, w: int, h: int, fps: float = TICK_HZ):
        if w % 2 or h % 2:
            raise ValueError("video size must be even, got %dx%d" % (w, h))
        self.path, self.w, self.h, self.n = path, int(w), int(h), 0
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        stem, ext = os.path.splitext(path)
        self.partial = stem + ".partial" + (ext or ".mp4")
        self.vw = cv2.VideoWriter(self.partial, cv2.VideoWriter_fourcc(*"mp4v"),
                                  float(fps), (self.w, self.h))
        if not self.vw.isOpened():
            raise RuntimeError("could not open a video writer at %s" % self.partial)

    def write(self, frame) -> None:
        if frame.shape != (self.h, self.w, 3) or frame.dtype != np.uint8:
            raise ValueError("frame %s %s, writer expects (%d, %d, 3) uint8"
                             % (frame.shape, frame.dtype, self.h, self.w))
        self.vw.write(frame)
        self.n += 1

    def close(self, ok: bool = True) -> None:
        self.vw.release()
        if ok:
            os.replace(self.partial, self.path)
        elif os.path.exists(self.partial):
            os.remove(self.partial)

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        self.close(ok=exc_type is None)


def transcode_h264(path: str, ffmpeg: Optional[str] = None, crf: int = 20) -> str:
    """Re-encode `path` in place as H.264 / yuv420p (plays in browsers and
    PowerPoint; mp4v often does not). Needs ffmpeg on PATH or given."""
    exe = ffmpeg or shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("--h264 needs ffmpeg on PATH (or --ffmpeg PATH)")
    stem, ext = os.path.splitext(path)
    tmp = stem + ".h264" + ext
    cmd = [exe, "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(int(crf)), "-movflags", "+faststart", tmp]
    if subprocess.call(cmd) != 0 or not os.path.exists(tmp):
        raise SystemExit("ffmpeg failed: %s" % " ".join(cmd))
    os.replace(tmp, path)
    return path
