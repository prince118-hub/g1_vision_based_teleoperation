"""tools/make_final_report.py on synthetic eval folders, with hand-computed position-level values.

    python -m pytest tools/test_make_final_report.py -q
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import analyze_experiments as AE                                    # noqa: E402
import make_final_report as MF                                      # noqa: E402

CODE = {"BC": "bc_b256", "ACT": "act_orig", "ACT-LSTM": "act_lstm_full", "ACT-12": "act", "ACT-LSTM-12": "act_lstm",
        "BC (batch 8)": "bc"}
POSITIONS = 8
#: Successful positions per training seed, per (model, exp). Hand-checkable counts below.
PATTERN = {("ACT-LSTM", "exp1"): [{0, 1}, {0}, {0, 2}], ("ACT", "exp1"): [{0}, {1}, {3}],
           ("ACT-LSTM", "exp2"): [{0}, {0, 1}, {0, 1, 2}], ("ACT", "exp2"): [{3}, set(), set()],
           ("ACT-LSTM-12", "exp1"): [{0}, {4}, set()], ("ACT-LSTM-12", "exp2"): [set(), set(), {5}],
           ("ACT-12", "exp1"): [{1}, {1}, {1}], ("ACT-12", "exp2"): [set(), {6}, set()]}


def _row(model, size, train_seed, exp, position, ok):
    seed = (0 if exp == "exp1" else 5000) + position
    reg = AE.region()
    if exp == "exp2":
        box = [reg["heldout_x"][0] + 0.001, reg["heldout_y"][0] + 0.001]
    else:
        box = [reg["x"][0] + 0.001, reg["y"][0] + 0.001]
    base = 0.2 + 0.01 * list(CODE).index(model)
    return dict(seed=seed, box_xy=box, heldout=exp == "exp2", success=ok, fail_reason="" if ok else ("GRASP" if position % 2 else "WALK_IN"),
                phase_reached="VERIFY", grasp=ok, lift=ok, transport=ok, place=ok, placement_error_m=0.04 if ok else 0.5,
                final_box_goal_m=0.04 if ok else 0.5 + position * 0.01, closest_box_goal_m=0.04 if ok else 0.3,
                palm_box_min_m=base + 0.001 * position, palm_box_nearest_min_m=base - 0.05 + 0.001 * position,
                lock_rule=dict(version="v3"), walkin="policy", weld_engage_tick=10 if ok else -1, weld_release_tick=-1,
                resting=ok, tilt_deg=1.0, trajectory_sha256="x",
                header=dict(ckpt=dict(file="c.pt", step=1), lock_rule=dict(version="v3"), training_data_lock_rule="v2"))


def _write_eval(root):
    for m, s, _t in MF.GROUPS:
        seeds = range(3) if s == "n40" else [0]
        for ts in seeds:
            d = os.path.join(root, "%s_%s_seed%d" % (CODE[m], s, ts))
            os.makedirs(d)
            for exp in AE.EXPS:
                wins = PATTERN.get((m, exp), [set(), set(), set()])[ts] if s == "n40" else set()
                with open(os.path.join(d, exp + ".jsonl"), "w", encoding="utf-8") as fh:
                    for p in range(POSITIONS):
                        fh.write(json.dumps(_row(m, s, ts, exp, p, p in wins)) + "\n")
            with open(os.path.join(d, "selection.json"), "w", encoding="utf-8") as fh:
                json.dump(dict(chosen="c.pt", chosen_tally=dict(n=20, success=1)), fh)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("final")
    root = str(tmp / "eval")
    _write_eval(root)
    return MF.build([root], str(tmp / "report"), d19_roots=[], strict=False), str(tmp / "report")


def _pos(built_, a, b, exp):
    P = built_[0]["tables"]["position"]
    return P[(P.A == a) & (P.B == b) & (P.exp == exp)].iloc[0]


# ─── the paired test, by hand ─────────────────────────────────────────────────
def test_paired_test_small_example_by_hand():
    """d = 2, 0, -1, 1, 0, 0. Untied |d| = 2, 1, 1: ranks 3, 1.5, 1.5, so W+ = 4.5, W- = 1.5, r = 0.5.
    n = 3: mean 3, variance 3*4*7/24 - (2^3 - 2)/48 = 3.375. z = (4.5 - 3 - 0.5)/sqrt(3.375)."""
    r = MF.paired_test([3, 2, 0, 1, 0, 0], [1, 2, 1, 0, 0, 0])
    assert (r["favour_a"], r["favour_b"], r["tie"], r["n_positions"]) == (2, 1, 3, 6)
    assert (r["w_plus"], r["w_minus"]) == (4.5, 1.5) and r["rank_biserial"] == 0.5
    assert abs(r["mean_diff"] - 2 / 6) < 1e-12
    z = 1.0 / math.sqrt(3.375)
    assert abs(r["wilcoxon_p"] - 2 * norm.sf(z)) < 1e-12
    assert r["sign_p"] == 1.0                                       # 2 of 3 cannot differ from a coin


def _wilcoxon_by_formula(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    ranks = rankdata(np.abs(nz))
    W, n = ranks[nz > 0].sum(), len(nz)
    _, t = np.unique(np.abs(nz), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - ((t ** 3 - t).sum()) / 48
    delta = W - n * (n + 1) / 4
    z = (abs(delta) - 0.5) / math.sqrt(var)
    return 2 * norm.sf(z)


def test_wilcoxon_equals_the_tie_corrected_formula_on_realistic_counts():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 4, 100), rng.integers(0, 3, 100)
    assert abs(MF.paired_test(a, b)["wilcoxon_p"] - _wilcoxon_by_formula(a, b)) < 1e-12


def test_all_one_direction_and_all_ties():
    r = MF.paired_test([1, 2, 3, 1], [0, 0, 0, 0])
    assert r["rank_biserial"] == 1.0 and r["w_minus"] == 0 and r["favour_b"] == 0
    assert abs(r["sign_p"] - 2 * 0.5 ** 4) < 1e-12                  # the exact two-sided sign test
    swap = MF.paired_test([0, 0, 0, 0], [1, 2, 3, 1])
    assert swap["rank_biserial"] == -1.0 and swap["wilcoxon_p"] == r["wilcoxon_p"]
    same = MF.paired_test([0, 1, 2], [0, 1, 2])
    assert (same["wilcoxon_p"], same["sign_p"], same["rank_biserial"], same["tie"]) == (1.0, 1.0, 0.0, 3)


# ─── counts per position ──────────────────────────────────────────────────────
def test_position_counts_are_successes_over_the_three_seeds(built):
    c = built[0]["tables"]["position_counts"]
    e1 = c[c.exp == "Exp 1"].sort_values("position_seed")
    assert len(e1) == POSITIONS and len(c[c.exp == "Exp 2"]) == POSITIONS
    assert e1["ACT-LSTM"].tolist() == [3, 1, 1, 0, 0, 0, 0, 0]      # {0,1} {0} {0,2}
    assert e1["ACT"].tolist() == [1, 1, 0, 1, 0, 0, 0, 0]           # {0} {1} {3}
    assert e1["ACT-12"].tolist() == [0, 3, 0, 0, 0, 0, 0, 0]
    assert c[c.exp == "Exp 2"].sort_values("position_seed")["ACT-LSTM"].tolist() == [3, 2, 1, 0, 0, 0, 0, 0]
    assert c["ACT-LSTM"].max() <= 3 and c["BC"].sum() == 0


def test_position_tests_on_known_counts_with_holm(built):
    # Exp 1 ACT-LSTM 3,1,1,0,0,0,0,0 vs ACT 1,1,0,1,0,0,0,0: d = 2,0,1,-1,0,0,0,0
    r = _pos(built, "ACT-LSTM", "ACT", "Exp 1")
    assert (r.favour_a, r.favour_b, r.tie, r.tie_both_zero) == (2, 1, 5, 4)
    assert (r.w_plus, r.w_minus) == (4.5, 1.5) and r.rank_biserial == 0.5
    assert abs(r.mean_A - 5 / 8) < 1e-12 and abs(r.mean_B - 3 / 8) < 1e-12
    # Exp 2 ACT-LSTM 3,2,1,0,... vs ACT 0,0,0,1,...: all four untied, one against
    r2 = _pos(built, "ACT-LSTM", "ACT", "Exp 2")
    assert (r2.favour_a, r2.favour_b, r2.tie) == (3, 1, 4)
    P = built[0]["tables"]["position"]
    for exp in ("Exp 1", "Exp 2"):
        fam = P[P.exp == exp]
        assert [(a, b) for a, b in zip(fam.A, fam.B)] == MF.POSITION_PAIRS
        assert np.allclose(fam.wilcoxon_p_holm, AE.holm(fam.wilcoxon_p.tolist()))
        assert np.allclose(fam.sign_p_holm, AE.holm(fam.sign_p.tolist()))


# ─── the report holds what was asked for, under thesis names ──────────────────
def test_report_has_every_required_part(built):
    res, out = built
    text = open(res["report"], encoding="utf-8").read()
    for part in ("Full task success", "Position-level paired test", "Wilcoxon", "sign test", "rank-biserial", "favour A", "favour B",
                 "Pooled Fisher's exact test (episodes treated as independent)", "Stage-wise (phase) success", "grasp", "lift", "transport",
                 "place", "Failure phase", "Proximity", "closest palm to box", "farther palm to box", "final box to goal",
                 "Success by box position", "Generalization gap", "gap per seed", "Tests (episodes treated as independent)",
                 "does not speak to a new training seed"):
        assert part in text, part
    for f in ("fig_success.png", "fig_stages.png", "fig_failure_phase.png", "fig_proximity.png", "fig_success_by_position.png",
              "fig_generalization_gap.png"):
        assert "(%s)" % f in text, f                                # linked in the report
        p = os.path.join(out, f)
        assert os.path.getsize(p) > 5000 and open(p, "rb").read(8) == b"\x89PNG\r\n\x1a\n", f
    for f in ("position_tests.csv", "position_counts.csv", "proximity_summary.csv", "fisher.csv", "distance_tests.csv",
              "success_summary.csv", "stage_rates.csv", "failure_phase.csv", "generalization_gap.csv", "episodes.csv", "runs.csv"):
        assert os.path.getsize(os.path.join(out, f)) > 0, f


def test_every_model_and_experiment_is_in_each_table(built):
    T = built[0]["tables"]
    pairs = {(m, s, e) for m, s, _ in MF.GROUPS for e in AE.EXPS}
    for key in ("summary", "stages", "fail"):
        assert {(r.model, r["size"], r.exp) for _, r in T[key].iterrows()} == pairs, key
    assert {(r.model, r["size"], r.exp) for _, r in T["proximity"].iterrows()} == pairs
    assert set(T["proximity"].metric) == {short for _, short, _ in MF.PROXIMITY}
    assert {(r.model, r["size"]) for _, r in T["gap"].iterrows()} == {(m, s) for m, s, _ in MF.GROUPS}


def test_proximity_uses_the_nearer_palm_for_closest(built):
    S = built[0]["tables"]["proximity"]
    r = S[(S.model == "ACT") & (S["size"] == "n40") & (S.exp == "exp1")].set_index("metric")
    base = 0.2 + 0.01 * list(CODE).index("ACT")
    assert abs(r.loc["closest palm to box", "median"] - (base - 0.05 + 0.0035)) < 1e-9       # median position 3.5
    assert abs(r.loc["farther palm to box", "median"] - (base + 0.0035)) < 1e-9
    assert r.loc["closest palm to box", "n"] == 3 * POSITIONS and r.loc["closest palm to box", "missing"] == 0


def test_tables_read_thesis_names_only(built):
    T = built[0]["tables"]
    for key in ("summary", "by_run", "stages", "fail", "gap", "thr", "d19"):
        assert set(T[key].model) <= set(AE.MODELS), key
    assert set(T["position"].A) | set(T["position"].B) <= set(AE.MODELS)
    with open(os.path.join(built[1], "fisher.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["assumption"] for r in rows} == {MF.INDEPENDENT}


def test_gap_per_seed_values(built):
    # ACT-LSTM per seed: exp1 2,1,2 of 8 and exp2 1,2,3 of 8, in points.
    by_run = built[0]["tables"]["by_run"]
    assert MF.seed_gaps(by_run, "ACT-LSTM", "n40") == "%+.0f, %+.0f, %+.0f" % (100 / 8, -100 / 8, -100 / 8)


def test_strict_refuses_anything_but_the_final_set(tmp_path):
    root = str(tmp_path / "eval")
    _write_eval(root)
    with pytest.raises(SystemExit, match="22 runs"):
        MF.build([root], str(tmp_path / "r"), d19_roots=[], strict=True)


def test_real_final_set_where_it_exists(tmp_path):
    """Where the real runs are on disk: the strict build passes, and position counts add up to the pooled counts."""
    if not all(os.path.isdir(r) for r in MF.ROOTS):
        pytest.skip("no evaluation folders here (runs/ is gitignored)")
    res = MF.build(out=str(tmp_path / "final"))
    T = res["tables"]
    assert len(res["runs"]) == 22 and len(res["episodes"]) == 4400
    c = T["position_counts"]
    s = T["summary"][T["summary"]["size"] == "n40"]
    for _, r in s.iterrows():
        assert c[c.exp == MF.EXP_NAME[r.exp]][r.model].sum() == r.pooled_successes, (r.model, r.exp)
    for exp in MF.EXP_NAME.values():
        assert len(c[c.exp == exp]) == 100
    assert c[list(AE.MODELS)].to_numpy().max() <= 3
    assert (T["position"].favour_a + T["position"].favour_b + T["position"].tie == 100).all()
