"""Report tables the thesis needs that no other tool writes: data quality, compute, offline vs real.

    python tools/report_extras.py \\
        --datasets recordings/episodes recordings/teleop_fake \\
        --run-root runs/scripted_v3 --out runs/report_extras

    # inference timing on the GPU (the deployment device) instead of the CPU:
    python tools/report_extras.py --run-root runs/scripted_v3 --bench-device cuda

Writes into --out (default runs/report_extras/, gitignored):

  data_quality.csv          one row per --datasets folder (a)
  compute.csv               one row per run directory (b)
  offline_vs_real.csv       one row per deployed selection checkpoint (c)
  offline_vs_real_spearman.csv
  offline_vs_real.png       validation loss vs selection successes, one panel per model
  report_extras.md          the three tables as markdown

READ ONLY. Nothing is written into a recording, run or eval directory. It runs
without mujoco and without torch (tools/report_io.py); only the optional
inference benchmark needs torch (and, through train_bc.py, the full stack), and
reports "n/a" with the reason when either is missing.

(a) DATA QUALITY, from the folder's own ledger and its episode files:
    attempts = `issue` rows; accepted / rejected / errors = `EpisodeLedger.summary`
    (LAST decision per seed - a seed failed twice and then accepted counts once,
    as accepted); episode length over the ACCEPTED files on disk (the top-level
    *.npz; failed/ is excluded, the loaders refuse it); spec version, lock rule,
    grip label, source, demonstrator settings / operators read from each file's
    meta. Tracking dropout = degraded ticks / ticks over episodes that MEASURED
    tracking (`tracking_ok` stored, recorder.tracking_measured); scripted
    episodes measure nothing and read "n/a". Replay check: tools/replay_check.py
    prints and writes nothing, so it is reported only if its output was saved in
    the folder as replay_check*.txt / .log / .json; otherwise "not run / not saved".

(b) COMPUTE, from result.json + metadata.json (train_bc.py): total wall time,
    optimizer steps, steps/s, parameters, device. A run killed before result.json
    falls back to the last metrics.jsonl row (elapsed_seconds, step). PEAK RAM /
    GPU MEMORY ARE NOT RECORDED BY train_bc.py: they are read only from a
    `resources` dict in result.json / metadata.json or a `resources.json` sidecar
    ({"peak_ram_gb": .., "peak_gpu_gb": ..}) and are otherwise "n/a". Inference
    speed is the MODEL FORWARD ONLY - `model(obs)` on a zero (1, W_o, 48) window
    under no_grad, the call deploy.CheckpointPolicy.act makes once per 25 Hz
    tick - excluding normalization and the temporal ensemble; on the checkpoint
    eval_scripted.py selected if a selection.json exists, else best.pt / last.pt.

(c) OFFLINE VS REAL, from <eval-root>/<run>/selection.json: every candidate
    that was actually deployed (aliases are skipped - the same weights would be
    counted twice) gives one point: x = the checkpoint's validation loss (stored in
    the checkpoint, ckpt_info.val_loss; else the metrics.jsonl row at that
    step/epoch), y = full successes on the 20 selection seeds. Spearman rho per
    run and pooled per model. A run whose candidates all scored the same success
    (e.g. BC-160, 0/20 everywhere) has an UNDEFINED rho, reported as such - not 0.
    Lower loss should mean more successes, so agreement is a NEGATIVE rho.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
import time
from typing import Callable, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import report_io as IO                                              # noqa: E402

MODEL_ORDER = ("bc", "act", "act_lstm")      # fixed colour order (report_io.SERIES)


# ═══ (a) data quality ═════════════════════════════════════════════════════════
def _replay_check(folder: str) -> str:
    files = sorted(glob.glob(os.path.join(folder, "replay_check*.json"))
                   + glob.glob(os.path.join(folder, "replay_check*.txt"))
                   + glob.glob(os.path.join(folder, "replay_check*.log")))
    if not files:
        return "not run / not saved"
    out = []
    for p in files:
        name = os.path.basename(p)
        try:
            if p.endswith(".json"):
                d = json.load(open(p, encoding="utf-8"))
                v = d.get("verdict", d.get("pass"))
                v = ("PASS" if v else "FAIL") if isinstance(v, bool) else str(v)
                manip = d.get("manipulation_reproduced")
                out.append("%s%s [%s]" % (v, " (manipulation %s)" % manip if manip else "", name))
                continue
            text = open(p, encoding="utf-8", errors="replace").read()
        except (OSError, ValueError) as e:
            out.append("unreadable [%s: %s]" % (name, e))
            continue
        m = re.findall(r"^REPLAY (PASS|FAIL)\s*$", text, re.M)
        man = re.findall(r"MANIPULATION reproduced\s*:\s*(\d+/\d+)", text)
        if m:
            out.append("%s%s [%s]" % (m[-1], " (manipulation %s)" % man[-1] if man else "", name))
        else:
            out.append("no REPLAY verdict line [%s]" % name)
    return "; ".join(out)


def _who(metas: List[dict]) -> str:
    """Demonstrator settings for scripted episodes, operator ids for teleop."""
    parts = []
    demo = {json.dumps(m["demonstrator"], sort_keys=True) for m in metas if m.get("demonstrator")}
    for d in sorted(demo):
        dd = json.loads(d)
        parts.append("scripted (" + ", ".join("%s=%s" % kv for kv in sorted(dd.items())) + ")")
    ops = sorted({str(m["operator"]) for m in metas if m.get("operator") is not None})
    if ops:
        parts.append("operator: " + ", ".join(ops))
    return "; ".join(parts) or "n/a"


def data_quality(folder: str) -> dict:
    folder = os.path.abspath(folder)
    row = dict(dataset=IO.repo_relpath(folder) if folder.startswith(ROOT + os.sep)
               else folder)
    led = IO.open_ledger(folder)
    if led is None:
        row.update(ledger="absent", attempts=None, seeds_issued=None, accepted=None,
                   rejected=None, errors=None)
    else:
        s = led.summary()
        rows = led.rows()
        row.update(ledger="present", attempts=sum(1 for r in rows if r["event"] == "issue"),
                   seeds_issued=s["issued"], accepted=s["accepted"], rejected=s["discarded"],
                   errors=s["errors"])
    files = IO.episode_files(folder)
    failed = IO.episode_files(os.path.join(folder, IO.FAILED_DIR))
    metas, lens, deg, tot, measured = [], [], 0, 0, 0
    for p in files:
        meta, arr, n = IO.read_episode(p, ("tracking_ok",))
        metas.append(meta)
        lens.append(n)
        if "tracking_ok" in arr:
            ok = np.asarray(arr["tracking_ok"], dtype=bool)
            measured += 1
            deg += int((~ok).sum())
            tot += int(ok.size)
    row.update(
        episodes_on_disk=len(files), failed_on_disk=len(failed),
        len_min_ticks=min(lens) if lens else None,
        len_median_ticks=float(statistics.median(lens)) if lens else None,
        len_max_ticks=max(lens) if lens else None,
        len_median_s=float(statistics.median(lens)) * IO.TICK_S if lens else None,
        spec_version=IO.joined(m.get("spec_version") for m in metas),
        lock_rule=IO.joined(IO.lock_rule_of(m) for m in metas),
        grip_label=IO.joined((m.get("grip_label") or "absent (pre-1.2.0)") for m in metas),
        source=IO.joined(m.get("source") for m in metas),
        demonstrator_operator=_who(metas),
        tracking_measured_episodes=measured,
        dropout_pct=(100.0 * deg / tot) if tot else None,
        replay_check=_replay_check(folder))
    if lens and not tot:
        row["dropout_pct"] = "n/a (not measured)"
    return row


DQ_COLS = ["dataset", "ledger", "attempts", "seeds_issued", "accepted", "rejected", "errors",
           "episodes_on_disk", "failed_on_disk", "len_min_ticks", "len_median_ticks",
           "len_max_ticks", "len_median_s", "spec_version", "lock_rule", "grip_label",
           "source", "demonstrator_operator", "tracking_measured_episodes", "dropout_pct",
           "replay_check"]


# ═══ (b) compute ══════════════════════════════════════════════════════════════
def _json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def metrics_rows(run_dir: str) -> List[dict]:
    p = os.path.join(run_dir, "metrics.jsonl")
    out = []
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue            # a torn final line (killed run)
    return out


def model_kind(meta: Optional[dict]) -> str:
    try:
        return str(meta["config"]["notes"]["model"])
    except (KeyError, TypeError):
        return "?"


def _peaks(run_dir: str, res: Optional[dict], meta: Optional[dict]) -> dict:
    for where, d in (("result.json", (res or {}).get("resources")),
                     ("metadata.json", (meta or {}).get("resources")),
                     ("resources.json", _json(os.path.join(run_dir, "resources.json")))):
        if isinstance(d, dict) and ("peak_ram_gb" in d or "peak_gpu_gb" in d):
            return dict(peak_ram_gb=d.get("peak_ram_gb"), peak_gpu_gb=d.get("peak_gpu_gb"),
                        peak_source=where)
    return dict(peak_ram_gb=None, peak_gpu_gb=None,
                peak_source="not recorded (train_bc.py does not log it)")


def time_calls(fn: Callable[[], object], n_iter: int = 200, n_warmup: int = 20,
               sync: Optional[Callable[[], None]] = None,
               clock: Callable[[], float] = time.perf_counter) -> dict:
    """Per-call wall time of `fn`, in ms, after `n_warmup` untimed calls. `sync`
    (cuda.synchronize) runs inside each timed interval so asynchronous GPU work
    is counted."""
    for _ in range(int(n_warmup)):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(int(n_iter)):
        t0 = clock()
        fn()
        if sync:
            sync()
        ts.append(1000.0 * (clock() - t0))
    a = np.asarray(ts, dtype=np.float64)
    return dict(ms_mean=float(a.mean()), ms_p50=float(np.percentile(a, 50)),
                ms_p95=float(np.percentile(a, 95)), n=int(a.size))


def _bench_ckpt(run_dir: str, eval_dir: Optional[str]) -> Optional[str]:
    sel = _json(os.path.join(eval_dir, "selection.json")) if eval_dir else None
    names = ([sel["chosen"]] if sel and sel.get("chosen") else []) + ["best.pt", "last.pt"]
    for n in names:
        if os.path.exists(os.path.join(run_dir, n)):
            return n
    return None


def benchmark_checkpoint(run_dir: str, ckpt: str, device: str = "cpu",
                         n_iter: int = 200) -> dict:
    """Forward-pass timing of one checkpoint (see the module docstring, (b))."""
    try:
        import torch
    except ImportError:
        return dict(infer_note="n/a: torch not installed")
    try:
        import train_bc as TB                       # the checkpoint builder deploy.py uses
        from g1_model import train as T
        from g1_data import spec
    except Exception as e:                          # noqa: BLE001  (mujoco missing, ...)
        return dict(infer_note="n/a: cannot import the model stack (%s)" % type(e).__name__)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return dict(infer_note="n/a: --bench-device cuda but CUDA is not available")
    path = os.path.join(run_dir, ckpt)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    factory, _ = TB._factory_for(ck)
    model, ck = T.load_checkpoint(path, factory)
    model = model.to(device).eval()
    kw = ck.get("model_kwargs") or {}
    cfg = kw.get("cfg") or kw
    W = int(cfg["obs_window"])
    obs = torch.zeros(1, W, int(cfg.get("state_dim", spec.STATE_DIM)), device=device)
    sync = torch.cuda.synchronize if device == "cuda" else None

    def call():
        with torch.no_grad():
            model(obs)
    r = time_calls(call, n_iter=n_iter, sync=sync)
    return dict(infer_ms_mean=r["ms_mean"], infer_ms_p95=r["ms_p95"], infer_device=device,
                infer_ckpt=ckpt,
                infer_note="model forward only, (1, %d, 48) window, %d calls, torch threads %d"
                           % (W, r["n"], torch.get_num_threads()))


def compute_row(run_dir: str, eval_dir: Optional[str] = None, bench: bool = True,
                bench_device: str = "cpu", bench_iters: int = 200) -> dict:
    res = _json(os.path.join(run_dir, "result.json"))
    meta = _json(os.path.join(run_dir, "metadata.json"))
    row = dict(run=os.path.basename(os.path.normpath(run_dir)), model=model_kind(meta))
    if res is not None:
        wall = res.get("total_wall_seconds", res.get("wall_seconds"))
        row.update(status=res.get("status"), wall_s=wall,
                   optimizer_steps=res.get("optimizer_steps"),
                   steps_per_s=res.get("steps_per_second"))
    else:
        hist = metrics_rows(run_dir)
        last = hist[-1] if hist else {}
        row.update(status="no result.json (last metrics.jsonl row)" if hist else "no result.json",
                   wall_s=last.get("elapsed_seconds"), optimizer_steps=last.get("step"),
                   steps_per_s=None)
    row["wall_h"] = (float(row["wall_s"]) / 3600.0) if row.get("wall_s") is not None else None
    m = (meta or {}).get("model") or {}
    dev = (meta or {}).get("device") or {}
    row.update(params=m.get("parameters"), train_device=dev.get("name") or dev.get("device"))
    row.update(_peaks(run_dir, res, meta))
    row.update(infer_ms_mean=None, infer_ms_p95=None, infer_device=None, infer_ckpt=None)
    if not bench:
        row["infer_note"] = "n/a: --no-bench"
    else:
        ck = _bench_ckpt(run_dir, eval_dir)
        if ck is None:
            row["infer_note"] = "n/a: no checkpoint in the run directory"
        else:
            try:
                row.update(benchmark_checkpoint(run_dir, ck, bench_device, bench_iters))
            except Exception as e:                  # noqa: BLE001
                row["infer_note"] = "n/a: benchmark failed (%s: %s)" % (type(e).__name__, e)
    return row


COMPUTE_COLS = ["run", "model", "status", "wall_s", "wall_h", "optimizer_steps", "steps_per_s",
                "params", "train_device", "peak_ram_gb", "peak_gpu_gb", "peak_source",
                "infer_ms_mean", "infer_ms_p95", "infer_device", "infer_ckpt", "infer_note"]


# ═══ (c) offline vs real ══════════════════════════════════════════════════════
def _val_from_metrics(hist: List[dict], step, epoch):
    for r in hist:
        if step is not None and r.get("step") == step:
            return r.get("val_loss")
        if step is None and epoch is not None and r.get("epoch") == epoch and "step" not in r:
            return r.get("val_loss")
    return None


def selection_points(run_dir: str, eval_dir: str) -> List[dict]:
    sel = _json(os.path.join(eval_dir, "selection.json"))
    if not sel:
        return []
    meta = _json(os.path.join(run_dir, "metadata.json"))
    hist = metrics_rows(run_dir)
    run = os.path.basename(os.path.normpath(run_dir))
    out = []
    for c in sel.get("candidates", []):
        if c.get("alias_of"):
            continue
        info = c.get("ckpt_info") or {}
        v, src = info.get("val_loss"), "checkpoint"
        if v is None:
            v, src = _val_from_metrics(hist, info.get("step"), info.get("epoch")), "metrics.jsonl"
        if v is None:
            src = "none"
        out.append(dict(run=run, model=model_kind(meta), ckpt=c["ckpt"],
                        step=info.get("step"), epoch=info.get("epoch"),
                        val_loss=v, val_loss_source=src, n=c.get("n"),
                        success=c.get("success"), grasp=c.get("grasp"), lift=c.get("lift"),
                        transport=c.get("transport"), place=c.get("place"),
                        mean_placement_error_m=c.get("mean_placement_error_m"),
                        chosen=(c["ckpt"] == sel.get("chosen"))))
    return out


def spearman(xs, ys) -> dict:
    """rho and p, or the reason it is undefined (fewer than 3 points, or a
    constant variable - Spearman divides by its rank variance)."""
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)
           if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    n = len(pts)
    if n < 3:
        return dict(n=n, rho=None, p=None, note="undefined: fewer than 3 points")
    x, y = zip(*pts)
    if len(set(x)) == 1 or len(set(y)) == 1:
        which = "selection success" if len(set(y)) == 1 else "validation loss"
        return dict(n=n, rho=None, p=None, note="undefined: %s is constant" % which)
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return dict(n=n, rho=float(r.statistic), p=float(r.pvalue), note="")


def spearman_table(points: List[dict]) -> List[dict]:
    out = []
    for run in sorted({p["run"] for p in points}):
        ps = [p for p in points if p["run"] == run]
        out.append(dict(scope="run", name=run, model=ps[0]["model"],
                        **spearman([p["val_loss"] for p in ps], [p["success"] for p in ps])))
    for mk in sorted({p["model"] for p in points}):
        ps = [p for p in points if p["model"] == mk]
        out.append(dict(scope="model (pooled runs)", name=mk, model=mk,
                        **spearman([p["val_loss"] for p in ps], [p["success"] for p in ps])))
    return out


def plot_offline_vs_real(points: List[dict], path: str) -> Optional[str]:
    pts = [p for p in points if p["val_loss"] is not None and p["success"] is not None]
    if not pts:
        return None
    plt = IO.pyplot()
    kinds = [k for k in MODEL_ORDER if any(p["model"] == k for p in pts)]
    kinds += sorted({p["model"] for p in pts} - set(kinds))
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.6), sharey=True,
                             squeeze=False)
    for i, k in enumerate(kinds):
        ax = axes[0][i]
        col = IO.SERIES[MODEL_ORDER.index(k)] if k in MODEL_ORDER else IO.INK_2
        ps = [p for p in pts if p["model"] == k]
        ax.scatter([p["val_loss"] for p in ps], [p["success"] for p in ps], s=36, color=col,
                   edgecolor=IO.SURFACE, linewidth=1.5, zorder=3)
        ch = [p for p in ps if p["chosen"]]
        if ch:
            ax.scatter([p["val_loss"] for p in ch], [p["success"] for p in ch], s=90,
                       facecolor="none", edgecolor=IO.INK, linewidth=1.2, zorder=4,
                       label="selected checkpoint")
            ax.legend(loc="upper right", fontsize=8)
        r = spearman([p["val_loss"] for p in ps], [p["success"] for p in ps])
        ax.set_title("%s   Spearman %s (n=%d)" % (k, IO.fmt(r["rho"], 2, "undefined"), r["n"]),
                     fontsize=10, loc="left")
        ax.set_xlabel("validation loss (checkpoint)")
        if i == 0:
            ax.set_ylabel("successes on the 20 selection seeds")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


OVR_COLS = ["run", "model", "ckpt", "step", "epoch", "val_loss", "val_loss_source", "n",
            "success", "grasp", "lift", "transport", "place", "mean_placement_error_m", "chosen"]
SPEAR_COLS = ["scope", "name", "model", "n", "rho", "p", "note"]


# ═══ driver ═══════════════════════════════════════════════════════════════════
def find_runs(run_root: str) -> List[str]:
    out = []
    for d in sorted(glob.glob(os.path.join(run_root, "*"))):
        if os.path.isdir(d) and (os.path.exists(os.path.join(d, "metadata.json"))
                                 or os.path.exists(os.path.join(d, "result.json"))):
            out.append(d)
    return out


def eval_dir_for(run_dir: str, eval_root: Optional[str]) -> str:
    root = eval_root or os.path.join(os.path.dirname(os.path.normpath(run_dir)), "eval")
    return os.path.join(root, os.path.basename(os.path.normpath(run_dir)))


def build(datasets: List[str], run_dirs: List[str], eval_root: Optional[str], out: str,
          bench: bool = True, bench_device: str = "cpu", bench_iters: int = 200) -> dict:
    os.makedirs(out, exist_ok=True)
    dq = [data_quality(d) for d in datasets]
    comp = [compute_row(r, eval_dir_for(r, eval_root), bench, bench_device, bench_iters)
            for r in run_dirs]
    pts = []
    for r in run_dirs:
        pts += selection_points(r, eval_dir_for(r, eval_root))
    sp = spearman_table(pts)
    IO.write_csv(os.path.join(out, "data_quality.csv"), dq, DQ_COLS)
    IO.write_csv(os.path.join(out, "compute.csv"), comp, COMPUTE_COLS)
    IO.write_csv(os.path.join(out, "offline_vs_real.csv"), pts, OVR_COLS)
    IO.write_csv(os.path.join(out, "offline_vs_real_spearman.csv"), sp, SPEAR_COLS)
    png = plot_offline_vs_real(pts, os.path.join(out, "offline_vs_real.png"))
    md = ["# Report extras", "",
          "Generated by `tools/report_extras.py` on %s. Definitions: the module docstring."
          % time.strftime("%Y-%m-%d %H:%M"), "",
          "## (a) Data quality", "",
          IO.md_table(dq, DQ_COLS) if dq else "_no --datasets given_", "",
          "## (b) Compute", "",
          "Peak RAM/GPU are not logged by `train_bc.py` (n/a unless a `resources.json` "
          "sidecar exists). Inference = model forward only.", "",
          IO.md_table(comp, COMPUTE_COLS) if comp else "_no runs given_", "",
          "## (c) Offline vs real: validation loss vs selection success", "",
          "Agreement between the offline metric and closed-loop success is a NEGATIVE rho.", "",
          IO.md_table(sp, SPEAR_COLS) if sp else "_no selection.json found_", ""]
    if png:
        md += ["![offline vs real](offline_vs_real.png)", ""]
    if pts:
        md += ["### Points", "", IO.md_table(pts, OVR_COLS, nd=4), ""]
    with open(os.path.join(out, "report_extras.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return dict(data_quality=dq, compute=comp, points=pts, spearman=sp, png=png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=[],
                    help="recording folders (each with its ledger.jsonl and episodes)")
    ap.add_argument("--runs", nargs="*", default=[], help="run directories")
    ap.add_argument("--run-root", default=None,
                    help="every subdirectory holding metadata.json or result.json is a run")
    ap.add_argument("--eval-root", default=None,
                    help="eval_scripted.py output root (default <run's parent>/eval)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "report_extras"))
    ap.add_argument("--no-bench", action="store_true", help="skip the inference benchmark")
    ap.add_argument("--bench-device", choices=("cpu", "cuda", "auto"), default="cpu",
                    help="cpu by default so it never competes with a GPU job (ONE heavy "
                         "process at a time); cuda = the deployment device")
    ap.add_argument("--bench-iters", type=int, default=200)
    a = ap.parse_args(argv)
    runs = list(a.runs) + (find_runs(a.run_root) if a.run_root else [])
    eval_root = a.eval_root or (os.path.join(a.run_root, "eval") if a.run_root else None)
    r = build(a.datasets, runs, eval_root, a.out, bench=not a.no_bench,
              bench_device=a.bench_device, bench_iters=a.bench_iters)
    print("datasets %d, runs %d, selection points %d -> %s"
          % (len(r["data_quality"]), len(r["compute"]), len(r["points"]),
             os.path.join(a.out, "report_extras.md")))
    for s in r["spearman"]:
        print("  spearman %-22s %-16s n=%-3d rho=%s %s" % (s["scope"], s["name"], s["n"],
                                                         IO.fmt(s["rho"], 3), s["note"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
