"""How act_lstm_full's cost scales with EPISODE LENGTH: time per step, memory, hours per run.

    python tools/bench_long_episodes.py                         # CPU, 600/1200/2000/3000 ticks
    python tools/bench_long_episodes.py --ticks-list 600,2000 --steps 10 --out runs/bench_long.json

Human demonstrations are expected to run 40-120 s (1000-3000 ticks); scripted_v3
runs 548-692. act_lstm_full's LSTM reads the whole episode so far, so the episode
length is its sequence length - in training (each sample re-reads 0..t,
segmented, g1_model/act_memory.read_history) and in scoring (`score_deployment`
re-reads every tick of every validation episode: O(T^2) per episode).

For each length T, at the runner's full-size configs (K 100, batch 8):

  timing   tools/bench_memory_design.py's own `measure`, in its own subprocess,
           on synthetic episodes of exactly T ticks (an AVERAGE length T with
           no spread: real lengths vary, and the longest episodes dominate both
           a batch's padded length and the O(T^2) scoring, so this is a lower
           bound for a dataset whose MEAN is T) - act_lstm_full, and act_orig
           (the same model without the LSTM, same batch 8: the step-time
           difference is the LSTM's training cost; their SCORING times are not
           comparable - act_orig scores at batch 256, act_lstm_full at 32)
  memory   the bytes autograd SAVES for backward in one act_lstm_full training
           step whose 8 histories all have length T (the worst batch: the
           peak), parameters excluded - device-independent tensor sizes, so the
           growth carries over to the GPU. Fixed: parameters + gradients +
           AdamW's two moments = 16 bytes per parameter.
  hours    a 200k-step run with 200 validation passes over --val-episodes
           episodes (the memory-design launcher: 1000-step windows, n40 split's
           40 val episodes), measured on THIS machine (CPU), and PROJECTED for
           the NVIDIA laptop (below).

GPU PROJECTION (stated assumptions; re-measure on the laptop with
`bench_memory_design.py --device cuda --ticks T`). Anchors, the laptop's own
numbers (NOTES 2026-10-25b, 2026-10-26): a real act_lstm_full window (1000 steps
+ one 40-episode validation pass, scripted ~600-692 ticks) takes 146.6 s; the
cuda bench step is 0.131 s (act_orig 0.071 s) and reads ~10% high. So at 600
ticks a real step is 0.131/1.1 s, of which the LSTM is (0.131-0.071)/0.131,
and the rest of the window is validation.
  time    (expected) cuDNN's LSTM is sequential: its time is taken as
          proportional to the LSTM TIMESTEPS it runs, counted exactly - a
          training batch runs padded(longest of 8 random histories) timesteps
          (expected value over uniform t), a validation pass runs, per batch of
          32 consecutive samples, padded(longest in the batch) (`score_deployment`
          cuts batches in index order) - with the per-timestep cost from the
          anchor's LSTM share; a forward-only (scoring) timestep is taken as 1/3
          of a forward+backward one. The non-LSTM part of a step is constant (W_o
          = 1) and the non-LSTM part of validation is linear in T.
          (lower estimate, `gpu_run_hours_lower_bound`) the same parts scaled by THIS CPU's measured growth -
          lower, because the CPU LSTM's forward+backward grows far slower than
          its timestep count (oneDNN per-call overhead; measured 0.85 s at 576
          timesteps vs 1.43 s at 3008, batch 8), which a GPU does not share.
  memory  the laptop's measured peak (1,171 MiB device-wide, real n40 data,
          longest episode 692 ticks, after fixes A/B/C) plus the growth of the
          saved activations from a 704-tick (692 padded to 64) batch to a
          T-tick one. cuDNN's reserve space is not autograd's saved tensors;
          treat the figure as an estimate. Scoring at batch 32 without grad
          does not grow with T beyond the padded history itself (tiny).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TICKS = (600, 1200, 2000, 3000)
REF_TICKS = 600
MODELS = ("act_orig", "act_lstm_full")
RUN_STEPS, RUN_WINDOWS = 200_000, 200
#: The laptop's anchors (NOTES 2026-10-25b bench --device cuda; 2026-10-26 real run).
GPU_WINDOW_S = 146.6            # real act_lstm_full: 1000 steps + 1 val pass (40 eps)
GPU_BENCH_STEP_FULL = 0.131     # cuda bench, act_lstm_full, 600 ticks, batch 8
GPU_BENCH_STEP_ORIG = 0.071     # cuda bench, act_orig
GPU_BENCH_HIGH = 1.10           # "the bench reads ~10% high" (act_lstm 5.2 h vs 4.7 h real)
GPU_PEAK_MIB = 1171             # real act_lstm_full, peak device GPU, every window
GPU_ANCHOR_LONGEST = 692        # scripted_v3 longest episode (NOTES 2026-10-17)
GPU_CARD_GIB = 4.0              # RTX 3050 Laptop
SEG = 64                        # g1_model/act_memory.READ_SEGMENT (checked in main)


def padded(T: int) -> int:
    return (int(T) + SEG - 1) // SEG * SEG


def timing_row(kind: str, ticks: int, a) -> dict:
    """bench_memory_design.measure in a fresh subprocess (its peak RSS is its own)."""
    cmd = [sys.executable, os.path.join(ROOT, "tools", "bench_memory_design.py"), "--one", kind,
           "--steps", str(a.steps), "--warmup", str(a.warmup), "--episodes", str(a.episodes),
           "--ticks", str(ticks), "--batch", "8", "--score-episodes", str(a.score_episodes),
           "--device", "cpu"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    row = [L for L in out.stdout.splitlines() if L.startswith("BENCH_ROW ")]
    if not row:
        raise SystemExit("%s @ %d failed (exit %s):\n%s" % (kind, ticks, out.returncode,
                                                            out.stderr[-3000:]))
    return json.loads(row[-1][len("BENCH_ROW "):])


def saved_activation_bytes(lengths, batch: int = 8, seed: int = 0) -> dict:
    """{L: bytes autograd saves for one full-size act_lstm_full training step on
    `batch` histories of length L}, parameters excluded; plus the fixed bytes."""
    import torch
    import train_bc as TB
    from g1_data import spec
    from g1_model import train as T
    from g1_model.history import history_of
    mcfg = TB._model_config("act_lstm_full", 100, 1, batch, gate=False)
    tcfg = TB._train_config(mcfg, None, batch, seed=seed, max_steps=1, window_steps=1,
                            stop_rule=None, run_name="bench_long")
    model = TB._make_model(mcfg, tcfg)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    dim_mask = torch.from_numpy(np.asarray(spec.ACTION_MASK, dtype=bool).copy())
    g = torch.Generator().manual_seed(seed)
    out = {}
    for L in lengths:
        seen = {}

        def pack(t):
            st = t.untyped_storage()
            if st.data_ptr() not in params:
                seen[st.data_ptr()] = st.nbytes()
            return t
        x = torch.randn(batch, int(L), spec.STATE_DIM, generator=g)
        obs = history_of(x)
        target = torch.randn(batch, 100, spec.ACTION_DIM, generator=g)
        pad = torch.ones(batch, 100, dtype=torch.bool)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss, _, _, _ = T.forward_loss(model, obs, target, pad, dim_mask)
        out[int(L)] = int(sum(seen.values()))
        del loss
    return dict(saved=out, n_params=int(n_params), fixed_bytes=int(16 * n_params))


def train_timesteps(T: int, batch: int = 8) -> float:
    """Expected LSTM timesteps of one training batch: `batch` samples drawn
    uniformly from the ticks of T-tick episodes, history length t + 1, the batch
    read to padded(longest) (read_history)."""
    m = np.arange(1, int(T) + 1, dtype=np.float64)
    cdf = (m / T) ** batch                                   # P(longest <= m)
    pmf = np.diff(np.concatenate([[0.0], cdf]))
    return float((pmf * np.array([padded(int(x)) for x in m])).sum())


def score_timesteps(T: int, n_eps: int = 1, batch: int = 32) -> float:
    """LSTM timesteps of ONE validation pass over `n_eps` T-tick episodes, per
    episode: the index is (episode, t) in order and `score_deployment` cuts it
    into consecutive batches of `batch`, each read to padded(longest)."""
    lengths = np.tile(np.arange(1, int(T) + 1), int(n_eps))
    total = sum(padded(int(lengths[i:i + batch].max())) for i in range(0, len(lengths), batch))
    return total / float(n_eps)


GPU_FWD_FRACTION = 1.0 / 3.0    # a forward-only LSTM timestep vs forward+backward


def project(rows: dict, mem: dict, val_episodes: int) -> list:
    """Per length: CPU-measured and GPU-projected hours, memory. See docstring."""
    full = {T: r for (k, T), r in rows.items() if k == "act_lstm_full"}
    orig = {T: r for (k, T), r in rows.items() if k == "act_orig"}
    R = REF_TICKS
    lstm_step = {T: max(full[T]["step_s"] - orig[T]["step_s"], 0.0) for T in full}
    # the laptop at the reference length
    g_step = GPU_BENCH_STEP_FULL / GPU_BENCH_HIGH
    g_lstm = g_step * (GPU_BENCH_STEP_FULL - GPU_BENCH_STEP_ORIG) / GPU_BENCH_STEP_FULL
    g_rest = g_step - g_lstm
    g_val_ep = (GPU_WINDOW_S - 1000 * g_step) / 40.0        # the anchor's 40 val episodes
    tau = g_lstm / train_timesteps(R)                        # s per fwd+bwd timestep
    g_val_lstm = GPU_FWD_FRACTION * tau * score_timesteps(R, 40)
    g_val_rest = g_val_ep - g_val_lstm
    saved_ref = mem["saved"][padded(GPU_ANCHOR_LONGEST)]
    out = []
    for T in sorted(full):
        f = full[T]
        cpu_train_h = RUN_STEPS * f["step_s"] / 3600
        cpu_val_h = RUN_WINDOWS * val_episodes * f["score_s_per_ep"] / 3600
        # expected: LSTM time proportional to timesteps
        step_gpu = g_rest + tau * train_timesteps(T)
        val_ep_gpu = g_val_rest * T / R + GPU_FWD_FRACTION * tau * score_timesteps(T, val_episodes)
        # lower bound: each part grows as it does on this CPU
        step_lo = g_rest + g_lstm * (lstm_step[T] / lstm_step[R] if lstm_step[R] else 1.0)
        val_lo = g_val_ep * f["score_s_per_ep"] / full[R]["score_s_per_ep"]
        gpu_train_h = RUN_STEPS * step_gpu / 3600
        gpu_val_h = RUN_WINDOWS * val_episodes * val_ep_gpu / 3600
        lo_h = (RUN_STEPS * step_lo + RUN_WINDOWS * val_episodes * val_lo) / 3600
        saved_T = mem["saved"][padded(T)]
        gpu_mem = GPU_PEAK_MIB / 1024.0 + (saved_T - saved_ref) / 1024.0 ** 3
        out.append(dict(
            ticks=T, seconds=T / 25.0, padded_history=padded(T),
            train_timesteps=round(train_timesteps(T), 1),
            score_timesteps_per_ep=round(score_timesteps(T, val_episodes), 1),
            cpu_step_s=f["step_s"], cpu_step_s_act_orig=orig[T]["step_s"],
            cpu_lstm_step_s=round(lstm_step[T], 4),
            cpu_score_s_per_ep=f["score_s_per_ep"],
            cpu_peak_rss_gb=f["peak_rss_gb"], cpu_train_rss_gb=f["train_rss_gb"],
            step_vs_600=round(f["step_s"] / full[R]["step_s"], 2),
            score_vs_600=round(f["score_s_per_ep"] / full[R]["score_s_per_ep"], 2),
            saved_activations_gib=round(saved_T / 1024.0 ** 3, 3),
            cpu_run_hours=round(cpu_train_h + cpu_val_h, 1),
            cpu_train_hours=round(cpu_train_h, 1), cpu_val_hours=round(cpu_val_h, 1),
            gpu_step_s_projected=round(step_gpu, 4),
            gpu_score_s_per_ep_projected=round(val_ep_gpu, 3),
            gpu_run_hours_projected=round(gpu_train_h + gpu_val_h, 1),
            gpu_train_hours_projected=round(gpu_train_h, 1),
            gpu_val_hours_projected=round(gpu_val_h, 1),
            gpu_run_hours_lower_bound=round(lo_h, 1),
            gpu_peak_gib_projected=round(gpu_mem, 2),
            gpu_fits_4gib=bool(gpu_mem < GPU_CARD_GIB)))
    return out


def render(proj, val_episodes) -> str:
    h = ("%6s %5s %8s %6s %10s %6s %8s %9s %10s %10s %8s"
         % ("ticks", "s", "step s", "x600", "score s/ep", "x600", "act GiB", "CPU run h",
            "GPU run h*", "(lower)", "GPU GiB*"))
    lines = [h]
    for p in proj:
        lines.append("%6d %5.0f %8.3f %6.2f %10.1f %6.2f %8.3f %9.1f %10.1f %10.1f %8.2f"
                     % (p["ticks"], p["seconds"], p["cpu_step_s"], p["step_vs_600"],
                        p["cpu_score_s_per_ep"], p["score_vs_600"], p["saved_activations_gib"],
                        p["cpu_run_hours"], p["gpu_run_hours_projected"],
                        p["gpu_run_hours_lower_bound"], p["gpu_peak_gib_projected"]))
    lines.append("run = 200k steps (batch 8) + 200 validation passes x %d episodes; * = projected "
                 "for the laptop GPU, LSTM time ~ timesteps (module docstring)" % val_episodes)
    for p in proj:
        lines.append("  %4d ticks: GPU train %.1f h (step %.3f s) + validation %.1f h (%.2f s/ep);"
                     " CPU %.1f + %.1f h" % (p["ticks"], p["gpu_train_hours_projected"],
                                             p["gpu_step_s_projected"], p["gpu_val_hours_projected"],
                                             p["gpu_score_s_per_ep_projected"],
                                             p["cpu_train_hours"], p["cpu_val_hours"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks-list", default=",".join(str(t) for t in TICKS))
    ap.add_argument("--steps", type=int, default=10, help="timed optimizer steps per point")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=40, help="synthetic training episodes")
    ap.add_argument("--score-episodes", type=int, default=1)
    ap.add_argument("--val-episodes", type=int, default=40,
                    help="validation episodes per pass in the hours estimate (n40 split: 40)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-json", default=None, metavar="BENCH.json",
                    help="re-project from an earlier --out (no timing)")
    a = ap.parse_args(argv)
    from g1_model.act_memory import READ_SEGMENT
    assert READ_SEGMENT == SEG, (READ_SEGMENT, SEG)
    if a.from_json:
        old = json.load(open(a.from_json, encoding="utf-8"))
        rows = {tuple(r["key"]): r for r in old["timing_rows"]}
        rows = {(k, int(T)): r for (k, T), r in rows.items()}
        mem = old["saved_activation_bytes"]
        mem = dict(mem, saved={int(k): v for k, v in mem["saved"].items()})
    else:
        ticks = sorted({int(t) for t in a.ticks_list.split(",") if t.strip()} | {REF_TICKS})
        rows = {}
        for T in ticks:
            for kind in MODELS:
                print("timing %s at %d ticks ..." % (kind, T), flush=True)
                rows[(kind, T)] = timing_row(kind, T, a)
        print("saved activations ...", flush=True)
        mem = saved_activation_bytes(sorted({padded(T) for T in ticks}
                                            | {padded(GPU_ANCHOR_LONGEST)}))
    proj = project(rows, mem, a.val_episodes)
    print(render(proj, a.val_episodes))
    print("fixed: %.1f M parameters -> %.2f GiB (params + grads + AdamW moments)"
          % (mem["n_params"] / 1e6, mem["fixed_bytes"] / 1024.0 ** 3))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(dict(projection=proj, timing_rows=[dict(r, key=list(k)) for k, r in rows.items()],
                           saved_activation_bytes=mem,
                           anchors=dict(gpu_window_s=GPU_WINDOW_S,
                                        gpu_bench_step_full=GPU_BENCH_STEP_FULL,
                                        gpu_bench_step_orig=GPU_BENCH_STEP_ORIG,
                                        gpu_bench_high=GPU_BENCH_HIGH, gpu_peak_mib=GPU_PEAK_MIB,
                                        gpu_anchor_longest=GPU_ANCHOR_LONGEST,
                                        gpu_fwd_fraction=GPU_FWD_FRACTION),
                           run=dict(steps=RUN_STEPS, windows=RUN_WINDOWS,
                                    val_episodes=a.val_episodes),
                           machine=dict(platform=platform.platform(),
                                        processor=platform.processor(),
                                        python=platform.python_version())), fh, indent=1)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
