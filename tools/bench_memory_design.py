"""CPU (or GPU) cost of the memory-design models: time per training step, memory, scoring.

    python tools/bench_memory_design.py                       # CPU, 600-tick episodes
    python tools/bench_memory_design.py --steps 50 --device cuda --out runs/bench_gpu.json

Compares, at the configurations tools/train_memory_design_n40.ps1 trains -
K = 100, batch 8, the runner's own model configs (`train_bc._model_config`) -

    act_lstm       the current ACT-LSTM, W_o = 12 (the reference point of the ratios)
    act_orig       ACT on the current observation only, W_o = 1
    act_lstm_full  act_orig + an LSTM over the whole episode history 0..t

on SYNTHETIC episodes of --ticks ticks (default 600, scripted_v3 runs 548-692):
random normal states and actions, already in normalized units. Cost depends on
shapes and history lengths, not on values. The samples go through the loaders'
own `__getitem__` and batcher (`train._collate_for`) and the step through the
loop's own pieces (`train.forward_loss`, the clip, `train.make_optimizer`); only
the episode FILES are skipped (`_synthetic_dataset`), so no data is needed.

Each model runs in its own subprocess, so its peak memory is its own:
  step_s          median wall time of one optimizer step (batch + forward +
                  backward + clip + AdamW), after --warmup steps
  data_s          the part of it spent building the batch
  peak_rss_gb     the process's peak resident memory (CPU); cuda: peak allocated
  train_rss_gb    peak minus the resident memory after the model and optimizer
                  were built: what training itself adds
  score_s_per_ep  `train.score_deployment` (the selection quantity, guard
                  included) per --ticks-tick episode; a 1000-step window of a
                  40-val-episode run (splits_scripted_v3_n40) scores 40 of them
and the derived estimate for a 200k-step run with 200 validation windows. Ratios
are against act_lstm. A CPU estimate; the runs themselves are on the NVIDIA
laptop (O34 Option B), where the ratio of the LSTM's sequential cost differs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: (model, W_o) in the order reported; act_lstm is the reference point.
MODELS = (("act_lstm", 12), ("act_orig", 1), ("act_lstm_full", 1))
REFERENCE = "act_lstm"
RUN_STEPS = 200_000           # tools/train_memory_design_n40.ps1 --max-steps
RUN_WINDOWS = 200             # --window-steps 1000: one validation pass each
VAL_EPISODES = 40             # splits_scripted_v3_n40.json "val"


def _synthetic_dataset(cls, K: int, W: int, n_eps: int, ticks: int, seed: int):
    """A `cls` (ChunkDataset or HistoryChunkDataset) over in-memory random
    episodes: the attributes `__getitem__` and the loop read, set directly, and
    the index built by the dataset's own `_build_index`. Bypasses only the file
    scan and its refusals, which a synthetic episode has nothing to say to."""
    from torch.utils.data import Dataset
    from g1_data import spec
    from g1_model.loader import LoaderConfig, TrackingPolicy
    ds = cls.__new__(cls)
    Dataset.__init__(ds)
    rng = np.random.default_rng(seed)
    ds.cfg = LoaderConfig(chunk_size=K, obs_window=W, tracking=TrackingPolicy())
    ds.where, ds.norm_meta = "synthetic", None
    ds.seeds = list(range(n_eps))
    ds.paths = ["synthetic_%d" % i for i in range(n_eps)]
    ds.sources, ds.real_demonstration = ["synthetic"] * n_eps, [False] * n_eps
    ds.states = [rng.standard_normal((ticks, spec.STATE_DIM)).astype(np.float32)
                 for _ in range(n_eps)]
    ds.actions = [rng.standard_normal((ticks, spec.ACTION_DIM)).astype(np.float32)
                  for _ in range(n_eps)]
    ds.phase_labels = [np.zeros(ticks, dtype=np.int8) for _ in range(n_eps)]
    ds.tracking = [np.ones(ticks, dtype=bool) for _ in range(n_eps)]
    ds.tracking_is_measured = [False] * n_eps
    ds.lengths = [int(ticks)] * n_eps
    ds.dropped_episodes = []
    ds.index, ds.excluded_samples = ds._build_index()
    ds.build_seconds = 0.0
    return ds


def _peak_rss_gb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2)
    except ImportError:                                  # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024.0 ** 3


def _rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024.0 ** 3


def measure(kind: str, W, steps: int, warmup: int, n_eps: int, ticks: int, batch: int,
            score_eps: int, device: str, seed: int = 0) -> dict:
    """One model, in THIS process. See the module docstring for the fields."""
    import torch
    import torch.nn as nn
    import train_bc as TB
    from g1_data import spec
    from g1_model import train as T
    from g1_model.act_memory import ACTFullHistoryConfig
    from g1_model.history import HistoryChunkDataset
    from g1_model.loader import ChunkDataset
    dev = torch.device(device)
    K = 100
    mcfg = TB._model_config(kind, K, W, batch, gate=False)
    cls = HistoryChunkDataset if isinstance(mcfg, ACTFullHistoryConfig) else ChunkDataset
    ds = _synthetic_dataset(cls, K, mcfg.obs_window, n_eps, ticks, seed)
    tcfg = TB._train_config(mcfg, None, batch, seed=seed, max_steps=steps + warmup,
                            window_steps=1, stop_rule=None, run_name="bench")
    model = TB._make_model(mcfg, tcfg).to(dev)
    model.train()
    opt = T.make_optimizer(model, tcfg)
    dim_mask = torch.from_numpy(np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(dev)
    collate = T._collate_for(ds)
    g = torch.Generator().manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    rss0 = _rss_gb()
    step_s, data_s = [], []
    for i in range(steps + warmup):
        t0 = time.perf_counter()
        idx = torch.randint(len(ds), (batch,), generator=g).tolist()
        b = collate([ds[j] for j in idx])
        t1 = time.perf_counter()
        obs = b["obs"].to(dev)
        target, pad = b["action"].to(dev), b["action_mask"].to(dev)
        loss, _, _, _ = T.forward_loss(model, obs, target, pad, dim_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.grad_clip))
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t2 = time.perf_counter()
        if i >= warmup:
            step_s.append(t2 - t0)
            data_s.append(t1 - t0)
    peak = (torch.cuda.max_memory_allocated(dev) / 1024.0 ** 3 if dev.type == "cuda"
            else _peak_rss_gb())
    # the selection quantity, guard included, on `score_eps` whole episodes
    sds = _synthetic_dataset(cls, K, mcfg.obs_window, score_eps, ticks, seed + 1)
    t0 = time.perf_counter()
    T.score_deployment(model, sds, dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    score_s = time.perf_counter() - t0
    med = statistics.median(step_s)
    per_ep = score_s / score_eps
    return dict(model=kind, obs_window=int(mcfg.obs_window), chunk_size=K, batch=batch,
                parameters=int(sum(p.numel() for p in model.parameters())),
                steps_timed=len(step_s), step_s=round(med, 5),
                step_s_mean=round(statistics.fmean(step_s), 5),
                data_s=round(statistics.median(data_s), 5),
                peak_rss_gb=round(peak, 3),
                train_rss_gb=(None if dev.type == "cuda" else round(peak - rss0, 3)),
                score_s_per_ep=round(per_ep, 3),
                est_run_hours=round((RUN_STEPS * med + RUN_WINDOWS * VAL_EPISODES * per_ep)
                                    / 3600.0, 2),
                est_train_hours=round(RUN_STEPS * med / 3600.0, 2),
                est_val_hours=round(RUN_WINDOWS * VAL_EPISODES * per_ep / 3600.0, 2),
                device=str(dev), torch=torch.__version__, threads=torch.get_num_threads(),
                episodes=n_eps, ticks=ticks, mean_history=round((ticks + 1) / 2.0, 1))


def ratios(rows) -> dict:
    ref = next(r for r in rows if r["model"] == REFERENCE)
    out = {}
    for r in rows:
        out[r["model"]] = {k: (round(r[k] / ref[k], 3) if ref.get(k) else None)
                           for k in ("step_s", "peak_rss_gb", "train_rss_gb",
                                     "score_s_per_ep", "est_run_hours")
                           if r.get(k) is not None and ref.get(k) is not None}
    return out


def render(rows, rat) -> str:
    lines = ["%-14s %4s %11s %9s %8s %9s %9s %10s %9s"
             % ("model", "W_o", "params", "step s", "data s", "peak GB", "train GB",
                "score s/ep", "run h"),
             ]
    for r in rows:
        lines.append("%-14s %4d %11d %9.4f %8.4f %9.3f %9s %10.2f %9.1f"
                     % (r["model"], r["obs_window"], r["parameters"], r["step_s"],
                        r["data_s"], r["peak_rss_gb"],
                        "-" if r["train_rss_gb"] is None else "%.3f" % r["train_rss_gb"],
                        r["score_s_per_ep"], r["est_run_hours"]))
    lines.append("ratios vs %s:" % REFERENCE)
    for m, d in rat.items():
        lines.append("  %-14s " % m + "  ".join("%s %.2fx" % (k, v) for k, v in d.items()
                                                if v is not None))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=20, help="timed optimizer steps per model")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=40, help="synthetic training episodes")
    ap.add_argument("--ticks", type=int, default=600, help="ticks per synthetic episode")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--score-episodes", type=int, default=2,
                    help="episodes scored by score_deployment per model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", default=",".join(m for m, _ in MODELS))
    ap.add_argument("--out", default=None, help="JSON with every row and the ratios")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)   # child process
    a = ap.parse_args(argv)
    W_of = dict(MODELS)
    common = dict(steps=a.steps, warmup=a.warmup, n_eps=a.episodes, ticks=a.ticks,
                  batch=a.batch, score_eps=a.score_episodes, device=a.device)
    if a.one:
        print("BENCH_ROW " + json.dumps(measure(a.one, W_of[a.one], **common)))
        return 0
    rows = []
    for kind in [m for m in a.models.split(",") if m]:
        cmd = [sys.executable, os.path.abspath(__file__), "--one", kind,
               "--steps", str(a.steps), "--warmup", str(a.warmup),
               "--episodes", str(a.episodes), "--ticks", str(a.ticks),
               "--batch", str(a.batch), "--score-episodes", str(a.score_episodes),
               "--device", a.device]
        print("measuring %s ..." % kind, flush=True)
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        row = [L for L in out.stdout.splitlines() if L.startswith("BENCH_ROW ")]
        if out.returncode != 0 and not row:
            raise SystemExit("%s failed (exit %s):\n%s" % (kind, out.returncode,
                                                           out.stderr[-3000:]))
        rows.append(json.loads(row[-1][len("BENCH_ROW "):]))
    rat = ratios(rows) if any(r["model"] == REFERENCE for r in rows) else {}
    print(render(rows, rat) if rat else json.dumps(rows, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(dict(rows=rows, ratios_vs=REFERENCE, ratios=rat,
                           machine=dict(platform=platform.platform(),
                                        processor=platform.processor(),
                                        python=platform.python_version()),
                           assumptions=dict(run_steps=RUN_STEPS, run_windows=RUN_WINDOWS,
                                            val_episodes=VAL_EPISODES,
                                            data="synthetic random episodes; cost depends "
                                                 "on shapes and lengths only")),
                      fh, indent=1)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
