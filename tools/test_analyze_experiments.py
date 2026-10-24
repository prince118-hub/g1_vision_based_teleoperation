"""Tests for tools/analyze_experiments.py on SYNTHETIC eval folders with known
counts, so every table entry and p-value has a value computed independently
here (by formula, by hand, or a published reference value).

    python tools/test_analyze_experiments.py          (or pytest)

The fixtures copy tools/deploy.py's per-episode record fields (seed, box_xy,
heldout, success, grasp/lift/transport/place, fail_reason, palm_box_min_m,
final_box_goal_m, closest_box_goal_m, placement_error_m, lock_rule, header)
into <eval>/<model>_<size>_seed<k>/exp{1,2}.jsonl, the layout
tools/eval_scripted.py writes.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import analyze_experiments as AE                                    # noqa: E402

FAILS = ("WALK_IN", "GRASP", "TRANSPORT", "PLACE")
PALM_BASE = {"bc": 0.30, "act": 0.20, "act_lstm": 0.10}     # disjoint ranges per model
#: Run B as measured (CLAUDE.md section 8 2026-10-23): successes /100.
RUN_B = {"bc": (0, 0), "act": (14, 1), "act_lstm": (17, 11)}
#: A synthetic n40 run, three seeds each: (exp1 successes, exp2 successes) per seed.
RUN_A = {"bc": [(2, 1), (3, 1), (4, 1)], "act": [(10, 5), (12, 6), (14, 7)],
         "act_lstm": [(20, 15), (22, 15), (24, 18)]}
N = 100
_CACHE: dict = {}


def _tmp():
    if "tmp" not in _CACHE:
        _CACHE["tmp"] = tempfile.mkdtemp(prefix="analyze_exp_test_")
    return _CACHE["tmp"]


def _box(exp: str, i: int):
    reg = AE.region()
    rng = np.random.default_rng(1000 * (exp == "exp2") + i)
    if exp == "exp2":
        return [float(rng.uniform(*reg["heldout_x"])), float(rng.uniform(*reg["heldout_y"]))], True
    while True:
        x, y = float(rng.uniform(*reg["x"])), float(rng.uniform(*reg["y"]))
        if not (reg["heldout_x"][0] <= x <= reg["heldout_x"][1] and reg["heldout_y"][0] <= y <= reg["heldout_y"][1]):
            return [x, y], False


def _row(model, size, seed, exp, i, k, palm=True):
    ok = i < k
    fr = "" if ok else FAILS[i % len(FAILS)]
    g = ok or fr in ("TRANSPORT", "PLACE")
    box, held = _box(exp, i)
    r = dict(seed=(0 if exp == "exp1" else 5000) + i, policy="%s_%s_seed%d/best.pt" % (model, size, seed),
             kind=model, box_xy=box, heldout=held, success=ok, end_reason="success" if ok else "time_limit",
             fail_reason=fr, phase_reached="DONE" if ok else "SETTLE",
             grasp=g, lift=g, transport=ok or fr == "PLACE", place=ok,
             placement_error_m=0.04 if ok else 0.5 + 0.001 * i,
             final_box_goal_m=0.04 if ok else 0.5 + 0.001 * i,
             closest_box_goal_m=0.04 if ok else 0.08,
             lock_rule=dict(version="v3"),
             header=dict(run="runs/scripted_v3/%s_%s_seed%d" % (model, size, seed),
                         ckpt=dict(file="ckpt_00120000.pt", step=120000, epoch=None), kind=model,
                         lock_rule=dict(version="v3"), training_data_lock_rule="v2"))
    if palm:
        r["palm_box_min_m"] = round(PALM_BASE[model] + 0.0005 * i, 4)
        r["palm_box_nearest_min_m"] = r["palm_box_min_m"] - 0.01
    return r


def write_run(root, model, size, seed, counts, exps=("exp1", "exp2"), palm=True):
    d = os.path.join(root, "%s_%s_seed%d" % (model, size, seed))
    os.makedirs(d, exist_ok=True)
    for exp, k in zip(("exp1", "exp2"), counts):
        if exp not in exps:
            continue
        with open(os.path.join(d, "%s.jsonl" % exp), "w", encoding="utf-8") as fh:
            for i in range(N):
                fh.write(json.dumps(_row(model, size, seed, exp, i, k, palm)) + "\n")
    with open(os.path.join(d, "selection.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(chosen="ckpt_00120000.pt", chosen_tally=dict(n=20, success=3)), fh)
    os.makedirs(os.path.join(d, "select"), exist_ok=True)       # never a run
    return d


def _full():
    """Run B (160, seed 0) plus a synthetic Run A (n40, seeds 0-2)."""
    if "full" not in _CACHE:
        root = os.path.join(_tmp(), "full", "eval")
        for m, c in RUN_B.items():
            write_run(root, m, "160", 0, c)
        for m, seeds in RUN_A.items():
            for s, c in enumerate(seeds):
                write_run(root, m, "n40", s, c)
        _CACHE["full"] = AE.analyze([root], os.path.join(_tmp(), "full", "report"))
    return _CACHE["full"]


def _get(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    assert len(df) == 1, (kw, len(df))
    return df.iloc[0]


# ─── independent references ──────────────────────────────────────────────────
def _fisher_ref(a, b, c, d):
    """Two-sided Fisher exact p from the hypergeometric pmf (math.comb)."""
    r1, c1, n = a + b, a + c, a + b + c + d

    def pmf(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = pmf(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-7))


def _kw_ref(*groups):
    allv = np.concatenate(groups)
    ranks = np.empty(len(allv))
    ranks[np.argsort(allv, kind="stable")] = np.arange(1, len(allv) + 1)   # no ties by construction
    N_, i, s = len(allv), 0, 0.0
    for g in groups:
        R = ranks[i:i + len(g)].sum()
        s += R * R / len(g)
        i += len(g)
    return 12.0 / (N_ * (N_ + 1)) * s - 3 * (N_ + 1)


# ═══ the statistics ═══════════════════════════════════════════════════════════
def test_wilson_matches_published_values():
    """Wilson 95%: Newcombe (1998, Stat Med 17:857, method 3) worked examples
    81/263 -> [0.2553, 0.3662] and 0/10 -> [0, 0.2775]; 10/100 -> [0.0552, 0.1744]."""
    for (k, n), (lo, hi) in (((81, 263), (0.2553, 0.3662)), ((0, 10), (0.0, 0.2775)),
                             ((10, 100), (0.0552, 0.1744))):
        a, b = AE.wilson(k, n)
        assert abs(a - lo) < 1e-4 and abs(b - hi) < 1e-4, (k, n, a, b)
    assert all(math.isnan(v) for v in AE.wilson(0, 0))


def test_holm_by_hand():
    assert np.allclose(AE.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(AE.holm([0.5, 0.001]), [0.5, 0.002])
    assert AE.holm([0.9, 0.8, 0.7]) == [1.0, 1.0, 1.0]          # 3 x 0.7 caps at 1
    out = AE.holm([float("nan"), 0.02])
    assert math.isnan(out[0]) and abs(out[1] - 0.02) < 1e-12


def test_fisher_matches_the_hypergeometric():
    for a, n1, c, n2 in ((14, 100, 17, 100), (0, 100, 14, 100), (1, 100, 11, 100), (3, 10, 9, 10)):
        _, p = AE.fisher(a, n1, c, n2)
        ref = _fisher_ref(a, n1 - a, c, n2 - c)
        assert abs(p - ref) < 1e-9, (a, c, p, ref)


# ═══ the tables on known counts ═══════════════════════════════════════════════
def test_success_by_run_and_summary():
    T = _full()["tables"]
    r = _get(T["by_run"], model="act_lstm", size="160", exp="exp2")
    assert (r.successes, r.n) == (11, 100) and abs(r.rate - 0.11) < 1e-12
    lo, hi = AE.wilson(11, 100)
    assert abs(r.ci_lo - lo) < 1e-12 and abs(r.ci_hi - hi) < 1e-12
    s = _get(T["summary"], model="act", size="n40", exp="exp1")
    assert s.n_seeds == 3 and s.seeds == "0,1,2"
    assert abs(s.mean_rate - 0.12) < 1e-12 and abs(s.sd_rate - 0.02) < 1e-12
    assert (s.pooled_successes, s.pooled_n) == (36, 300)
    one = _get(T["summary"], model="act", size="160", exp="exp1")
    assert one.n_seeds == 1 and math.isnan(one.sd_rate) and one.pooled_successes == 14
    assert len(T["by_run"]) == 2 * (3 + 9)


def test_generalization_gap_and_its_standard_errors():
    T = _full()["tables"]
    g = _get(T["gap"], model="act", size="n40")
    assert abs(g.gap - (36 - 18) / 300) < 1e-12
    se = math.sqrt(0.12 * 0.88 / 300 + 0.06 * 0.94 / 300)
    assert abs(g.se_binomial - se) < 1e-12
    # per-seed gaps 0.05, 0.06, 0.07: mean 0.06, SD 0.01, SE 0.01/sqrt(3)
    assert abs(g.gap_seed_mean - 0.06) < 1e-12 and abs(g.se_seeds - 0.01 / math.sqrt(3)) < 1e-12
    b = _get(T["gap"], model="act_lstm", size="160")
    assert abs(b.gap - 0.06) < 1e-12 and math.isnan(b.se_seeds)


def test_stage_rates_and_failure_phase():
    T = _full()["tables"]
    st = _get(T["stages"], model="act", size="160", exp="exp1")
    # 14 successes; failures cycle WALK_IN, GRASP, TRANSPORT, PLACE over i = 14..99
    fails = [FAILS[i % 4] for i in range(14, 100)]
    n_tp = sum(f in ("TRANSPORT", "PLACE") for f in fails)
    assert st.grasp_k == 14 + n_tp and st.lift_k == 14 + n_tp
    assert st.transport_k == 14 + fails.count("PLACE") and st.place_k == 14 and st.success_k == 14
    f = _get(T["fail"], model="act", size="160", exp="exp1")
    assert f.success == 14 and sum(f[k] for k in AE.FAIL_ORDER) == 86
    for k in FAILS:
        assert f[k] == fails.count(k), k


def test_fisher_table_pairs_and_holm():
    T = _full()["tables"]
    fe = T["fisher"]
    fam = fe[(fe["size"] == "160") & (fe.exp == "exp2")]
    assert len(fam) == 3
    r = _get(fam, model_a="act", model_b="act_lstm")
    assert (r.a_successes, r.b_successes) == (1, 11)
    assert abs(r.p - _fisher_ref(1, 99, 11, 89)) < 1e-9
    assert np.allclose(fam.p_holm.to_numpy(), AE.holm(fam.p.tolist()))
    pooled = _get(fe[(fe["size"] == "n40") & (fe.exp == "exp1")], model_a="bc", model_b="act_lstm")
    assert (pooled.a_successes, pooled.a_n, pooled.b_successes, pooled.b_n) == (9, 300, 66, 300)
    assert math.isclose(pooled.p, _fisher_ref(9, 291, 66, 234), rel_tol=1e-6)


def test_distance_tests_kruskal_and_mann_whitney():
    """Palm proximity is drawn in disjoint ranges per model (LSTM < ACT < BC):
    H equals the rank formula; each Mann-Whitney U is 0 or n1*n2."""
    T = _full()["tables"]
    d = T["dist"]
    kw = _get(d[d.metric == "palm_box_min_m"], size="160", exp="exp1", test="kruskal-wallis")
    groups = [(PALM_BASE[m] + 0.0005 * np.arange(N)).round(4) for m in AE.MODELS]
    assert abs(kw.stat - _kw_ref(*groups)) < 1e-6 and kw.p < 1e-40
    mw = d[(d.metric == "palm_box_min_m") & (d["size"] == "160") & (d.exp == "exp1")
           & (d.test == "mann-whitney-u")]
    assert len(mw) == 3
    assert set(mw.stat.round(6)) <= {0.0, float(N * N)}
    assert (mw.p_holm >= mw.p).all() and np.allclose(mw.p_holm, AE.holm(mw.p.tolist()))
    fin = d[(d.metric == "final_box_goal_m") & (d.test == "kruskal-wallis")]
    assert len(fin) == 4                                     # 2 sizes x 2 experiments


def test_threshold_rates():
    T = _full()["tables"]
    t = _get(T["thr"], model="act_lstm", size="160", exp="exp1")
    assert abs(t.final_box_goal_within_5cm - 0.17) < 1e-12           # successes at 0.04 m
    assert abs(t.final_box_goal_within_10cm - 0.17) < 1e-12
    assert abs(t.closest_box_goal_within_5cm - 0.17) < 1e-12
    assert abs(t.closest_box_goal_within_10cm - 1.0) < 1e-12         # failures at 0.08 m
    assert t.palm_within_grasp_gate == 1.0                           # 0.10-0.15 m <= 0.16
    assert _get(T["thr"], model="act", size="160", exp="exp1").palm_within_grasp_gate == 0.0


def test_size_comparison_when_both_sizes_exist():
    T = _full()["tables"]
    sc = _get(T["sizes"], model="act", exp="exp1")
    assert abs(sc.rate_160 - 0.14) < 1e-12 and abs(sc.rate_n40 - 0.12) < 1e-12
    assert abs(sc["diff"] - 0.02) < 1e-12
    assert abs(sc.fisher_p - _fisher_ref(14, 86, 36, 264)) < 1e-9


# ═══ the report ═══════════════════════════════════════════════════════════════
def test_report_figures_and_labels():
    res = _full()
    out = os.path.dirname(res["report"])
    for f in ("report.md", "runs.csv", "episodes.csv", "success_by_run.csv", "success_summary.csv",
              "generalization_gap.csv", "stage_rates.csv", "failure_phase.csv", "fisher.csv",
              "distance_tests.csv", "thresholds.csv", "size_comparison.csv"):
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    for name, p in res["figures"].items():
        assert p and os.path.getsize(p) > 5000, name
        with open(p, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n", name
    md = open(res["report"], encoding="utf-8").read()
    for s in ("lock rule at evaluation:** v3", "Training-data lock rule:** v2", "ACT-LSTM", "n40",
              "160", "0,1,2", "ckpt_00120000.pt", "Holm", "Wilson", "fig_success_by_position.png",
              "fig_training_coverage.png", "fig_size_comparison.png", "17/100", "11/100"):
        assert s.lower() in md.lower(), s
    assert len(res["runs"]) == 12, "select/ is not a run"


def test_works_with_runs_missing():
    """Only Run B, BC without Exp 2, no ACT-LSTM at all, one older record
    format without palm_box_min_m: every table still builds, nothing crashes,
    and the gaps are reported as missing rather than invented."""
    root = os.path.join(_tmp(), "partial", "eval")
    write_run(root, "bc", "160", 0, (0, 0), exps=("exp1",))
    write_run(root, "act", "160", 0, (14, 1), palm=False)
    res = AE.analyze([root], os.path.join(_tmp(), "partial", "report"))
    T = res["tables"]
    assert any("bc_160_seed0: no exp2" in n for n in res["notes"])
    assert any("ACT-LSTM: no runs found" in n for n in res["notes"])
    g = _get(T["gap"], model="bc", size="160")
    assert "needs both" in g.note and math.isnan(g.get("gap", float("nan")))
    assert len(T["sizes"]) == 0 and res["figures"]["n40 vs 160"] is None
    kw = T["dist"][(T["dist"].metric == "palm_box_min_m") & (T["dist"].test == "kruskal-wallis")]
    assert len(kw) == 1 and kw.iloc[0].dropped == "0,100"     # ACT's records had no field
    assert math.isnan(kw.iloc[0].p) and kw.iloc[0].n == "100,0"
    fe = T["fisher"]
    assert len(fe[fe.exp == "exp1"]) == 1 and len(fe[fe.exp == "exp2"]) == 0
    assert os.path.exists(res["report"])
    assert AE.main(["--eval-root", root, "--out", os.path.join(_tmp(), "partial", "cli"),
                    "--no-figures"]) == 0


def test_empty_input_is_refused():
    empty = os.path.join(_tmp(), "empty")
    os.makedirs(empty, exist_ok=True)
    try:
        AE.analyze([empty], os.path.join(_tmp(), "empty_out"))
    except SystemExit as e:
        assert "no exp1.jsonl" in str(e)
    else:
        raise AssertionError("empty input accepted")


def test_spawn_sampler_mirrors_box_reset():
    """The no-mujoco spawn mirror equals g1_teleop.box_reset.sample_box_pose
    (checked where mujoco is installed), and the split files' train seeds are
    never in the held-out patch."""
    try:
        from g1_teleop.box_reset import sample_box_pose, in_heldout
        from g1_teleop.config import BoxConfig
    except ImportError:
        print("      (skipped: mujoco not installed)")
        return
    cfg = BoxConfig()
    for s in (0, 1, 7, 679, 5000, 12345):
        pos, _ = sample_box_pose(cfg, s)
        assert AE.spawn_xy(s) == (float(pos[0]), float(pos[1])), s
    cov = AE.coverage()
    for size, (xs, ys) in cov.items():
        assert not any(in_heldout((x, y), cfg) for x, y in zip(xs, ys)), size


def test_zzz_cleanup():
    if "tmp" in _CACHE:
        shutil.rmtree(_CACHE.pop("tmp"), ignore_errors=True)
    _CACHE.clear()


def _tests():
    return [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _tests():
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(_tests()) - failed}/{len(_tests())} passed")
    raise SystemExit(1 if failed else 0)
