"""M11 (spec.md S14) on synthetic results with a known ground truth.

These tests do not need a GPU: they check that the estimator recovers a planted
slope, that the leverage chain inverts correctly, and that the falsification
conditions fire when they should.
"""

import math

import numpy as np
import pytest

from src import analysis as an
from src import costmodel as cm

VICTIM_R = {
    "Carmon2019Unlabeled": 0.595,
    "Gowal2021Improving_28_10_ddpm_100m": 0.634,
    "Wang2023Better_WRN-28-10": 0.673,
    "Wang2023Better_WRN-70-16": 0.707,
    "Peng2023Robust": 0.711,
    "Bartoldson2024Adversarial_WRN-82-8": 0.708,
    "Bartoldson2024Adversarial_WRN-94-16": 0.737,
}


def synth_rows(slope=0.0, intercept=0.5, noise=0.0, seed=0, attack="best",
               budget=50_000, n_eval=1000, c_total=2e16, c_slope=0.0):
    """Results with tau(R_v) = slope * R_v + intercept, plus optional noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, R in VICTIM_R.items():
        for s in range(3):
            tau = slope * R + intercept + rng.normal(0, noise)
            aa = tau * R
            rows.append({
                "victim_name": name, "victim_aa_acc": R, "victim_clean_acc": 0.9,
                "victim_params": 1, "is_control": False, "student_arch": "resnet18_cifar",
                "student_params": 11_173_962, "attack": attack, "pool": "cifar100",
                "defense": "none", "defense_params": {}, "n_queries_nominal": budget,
                "n_queries_actual": budget, "epochs": 100, "seed": s,
                "student_clean_acc": 0.8, "student_aa_acc": aa,
                "student_pgd40_acc": aa + 0.01, "fidelity_clean": 0.9,
                "fidelity_adv": 0.5, "tau": tau, "n_eval": n_eval,
                "c_query": 0, "c_synthesis": 0,
                "c_student": int(c_total * 10 ** (c_slope * (R - 0.6))),
                "c_total": int(c_total * 10 ** (c_slope * (R - 0.6))),
                "c_query_heuristic": 0, "c_student_heuristic": 0,
                "wallclock_sec": 1.0, "gpu": "test", "extra": {},
            })
    return rows


# --------------------------------------------------------------------------

def test_fit_recovers_a_planted_slope():
    fit = an.fit_tau(synth_rows(slope=-0.8, intercept=1.0), attack="best", n_boot=500)
    assert fit.slope == pytest.approx(-0.8, abs=1e-6)
    assert fit.intercept == pytest.approx(1.0, abs=1e-6)
    assert fit.n_victims == 7
    assert fit.r_range == (0.595, 0.737)


def test_flat_tau_gives_a_ci_containing_zero():
    """Calibration, not a single draw: with a true slope of 0 and 7 victims, the
    bootstrap CI should cover 0 in the large majority of replications. A single
    seed can and does exclude it marginally, which is why F1 is judged on a CI
    rather than on a point estimate."""
    covered = 0
    for seed in range(10):
        fit = an.fit_tau(synth_rows(slope=0.0, intercept=0.5, noise=0.02, seed=seed),
                         attack="best", n_boot=1000)
        lo, hi = fit.slope_ci95
        covered += int(lo <= 0 <= hi)
        assert abs(fit.slope) < 0.5
    assert covered >= 7, covered


def test_standard_victim_is_excluded_from_tau():
    rows = synth_rows()
    rows.append({**rows[0], "victim_name": "Standard", "victim_aa_acc": 0.0, "tau": None})
    agg = an.tau_by_victim(rows)
    assert "Standard" not in agg


def test_mixed_n_eval_is_refused():
    rows = synth_rows(n_eval=1000) + synth_rows(n_eval=10000, seed=2)
    with pytest.raises(AssertionError, match="mixed n_eval"):
        an.assert_consistent_n_eval(rows)


def test_leverage_inverts_tau_times_rv():
    fit = an.fit_tau(synth_rows(slope=0.0, intercept=0.5), attack="best", n_boot=200)
    cost_fit = an.attacker_cost_fit(synth_rows(), attack="best")
    lev = an.leverage(0.30, fit, cost_fit)
    # tau = 0.5 everywhere, so reaching R = 0.30 needs a victim at R_v = 0.60.
    assert lev["R_v_required"] == pytest.approx(0.60, abs=1e-6)
    assert lev["c_def"] == pytest.approx(cm.flops_for_accuracy(0.60, 0.0), rel=1e-9)
    assert lev["log10_leverage"] == pytest.approx(
        math.log10(lev["c_def"] / lev["c_att"]), rel=1e-9)


def test_unreachable_target_raises_rather_than_fabricating():
    fit = an.fit_tau(synth_rows(slope=0.0, intercept=0.5), attack="best", n_boot=200)
    cost_fit = an.attacker_cost_fit(synth_rows(), attack="best")
    with pytest.raises(ValueError):
        an.leverage(0.95, fit, cost_fit)   # needs R_v = 1.9


def test_leverage_table_flags_extrapolation_in_direct_mode():
    """S14.3: R = 0.737 is the strongest measured victim, hence "interpolated,
    defensible"; everything above it is shaded as extrapolation."""
    rows = synth_rows(slope=0.0, intercept=0.95)
    tab = an.leverage_table(rows, attack="best", n_boot=500, mode="direct",
                            targets=(0.65, 0.737, 0.80, 0.85))
    by_target = {r["R_target"]: r for r in tab["rows"] if "error" not in r}
    assert by_target[0.65]["extrapolated"] is False
    assert by_target[0.737]["extrapolated"] is False
    assert by_target[0.80]["extrapolated"] is True
    assert by_target[0.85]["extrapolated"] is True
    assert "extrapolation" in tab["note"].lower()
    for r in by_target.values():
        lo, hi = r["log10_leverage_ci95"]
        assert lo <= r["log10_leverage"] + 1e-6 and r["log10_leverage"] <= hi + 1e-6


def test_leverage_table_flags_extrapolation_in_matched_mode():
    """S14.2's reading: the flag is on the *required victim* R_v, not on R.

    With tau < 1 a student target of R needs a victim above R, so R = 0.737
    already requires extrapolating past the strongest measured victim.
    """
    rows = synth_rows(slope=0.0, intercept=0.95)
    tab = an.leverage_table(rows, attack="best", n_boot=500, mode="matched",
                            targets=(0.60, 0.65, 0.737, 0.85))
    by_target = {r["R_target"]: r for r in tab["rows"] if "error" not in r}
    assert by_target[0.65]["extrapolated"] is False    # R_v = 0.684, in range
    assert by_target[0.737]["extrapolated"] is True    # R_v = 0.776, out of range
    assert by_target[0.85]["extrapolated"] is True
    assert by_target[0.65]["R_v_required"] == pytest.approx(0.65 / 0.95, abs=1e-6)


def test_matched_mode_is_reported_or_explained_in_every_direct_row():
    """A tau below 1 makes some S14.2 rows unsolvable; they say so rather than
    quietly disappearing."""
    rows = synth_rows(slope=0.0, intercept=0.52)     # realistic tau
    tab = an.leverage_table(rows, attack="best", n_boot=200, mode="direct")
    for r in tab["rows"]:
        assert "matched" in r
        assert "error" in r["matched"] or "R_v_required" in r["matched"]


def test_leverage_grows_with_the_target():
    """The scientific claim: defender cost diverges, attacker cost does not."""
    rows = synth_rows(slope=0.0, intercept=0.95)
    tab = an.leverage_table(rows, attack="best", n_boot=200)
    levs = [r["log10_leverage"] for r in tab["rows"] if "error" not in r]
    assert levs == sorted(levs)
    assert levs[-1] - levs[0] > 3


def test_F1_fires_on_a_steep_negative_slope():
    rows = synth_rows(slope=-2.0, intercept=2.0, noise=0.005, seed=3)
    checks = an.falsification_checks(rows, attack="best", n_boot=2000)
    assert checks["F1"]["triggered"] is True
    assert checks["verdict"] == "falsified"


def test_F1_does_not_fire_on_a_flat_tau():
    rows = synth_rows(slope=0.0, intercept=0.5, noise=0.02, seed=4)
    checks = an.falsification_checks(rows, attack="best", n_boot=2000)
    assert checks["F1"]["triggered"] is False


def test_F3_fires_when_attacker_cost_tracks_the_defender_curve():
    """Attacker cost climbing at a comparable rate would kill the leverage claim."""
    flat = an.falsification_checks(synth_rows(c_slope=0.0), attack="best", n_boot=200)
    steep = an.falsification_checks(synth_rows(c_slope=40.0), attack="best", n_boot=200)
    assert flat["F3"]["triggered"] is False
    assert steep["F3"]["triggered"] is True


def test_F2_compares_controls_when_present():
    rows = synth_rows(slope=0.0, intercept=0.5)
    controls = []
    for s in range(3):
        controls.append({**rows[0], "victim_name": "Standard_T1.8", "is_control": True,
                         "victim_aa_acc": 0.0, "tau": None, "seed": s,
                         "student_aa_acc": float(np.mean([r["student_aa_acc"] for r in rows]))})
    checks = an.falsification_checks(rows, all_rows=rows + controls, attack="best",
                                     n_boot=200)
    assert checks["F2"]["triggered"] is True   # control matches -> label smoothing


def test_delta_abs_needs_a_matched_standard_run():
    rows = synth_rows(slope=0.0, intercept=0.5)
    assert an.delta_abs_table(rows) == []      # no Standard runs -> nothing matched
    std = [{**r, "victim_name": "Standard", "victim_aa_acc": 0.0, "tau": None,
            "student_aa_acc": 0.05} for r in rows[:3]]
    out = an.delta_abs_table(rows + std)
    assert out and all(d["delta_abs"] == pytest.approx(d["student_aa_acc"] - 0.05)
                       for d in out)


def test_attacker_cost_fit_reports_spread():
    fit = an.attacker_cost_fit(synth_rows(c_slope=0.0), attack="best")
    assert fit["slope_log10_per_R"] == pytest.approx(0.0, abs=1e-9)
    assert fit["spread_log10"] == pytest.approx(0.0, abs=1e-9)


def test_summarize_defenses_projects_and_censors():
    """A defense that halves student AA must be charged the compute needed to
    undo that, and censored if it exceeds 10x the largest measured cost."""
    sweep = []
    for b, c, aa in [(5_000, 1e15, 0.10), (50_000, 1e16, 0.20), (200_000, 4e16, 0.30)]:
        for s in range(3):
            sweep.append({**synth_rows(budget=b)[0], "seed": s, "c_total": int(c),
                          "student_aa_acc": aa, "n_queries_nominal": b,
                          "victim_name": "V", "victim_aa_acc": 0.7,
                          "extra": {"defense_utility_cost": 0.0}})
    defended = [{**sweep[3], "defense": "top1", "student_aa_acc": 0.12, "seed": s,
                 "extra": {"defense_utility_cost": 0.005}} for s in range(3)]
    recs = an.summarize_defenses(sweep + defended, sweep_rows=sweep, budget=50_000)
    assert len(recs) == 1
    r = recs[0]
    assert r.defense == "top1"
    assert r.log10_cost_multiplier > 0
    assert r.delta_clean_acc == pytest.approx(0.005)
    assert r.cost_effectiveness == pytest.approx(r.log10_cost_multiplier / 0.005)
