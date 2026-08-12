"""M7 statistics (spec.md S10). No GPU: FAB itself is exercised by E3 runs.

The point of these tests is the censoring. Treating a censored margin as a
measured one biases the median downward, which would fabricate a mechanism.
"""

import numpy as np
import pytest

from src import margins as mg
from src.evaluate import EPS_8_255


def test_km_matches_the_ecdf_without_censoring():
    v = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    c = np.zeros(5, dtype=bool)
    assert mg.km_quantile(v, c, 0.5) == pytest.approx(0.3)
    times, surv = mg.kaplan_meier(v, c)
    assert surv[-1] == pytest.approx(0.0)
    assert np.all(np.diff(surv) <= 1e-12)          # survival is non-increasing


def test_censoring_pushes_the_median_up():
    """Censored points are known to be *above* the bound; a plain median that
    treated them as equal to the bound would understate the margin."""
    v = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05])
    uncensored = np.zeros(8, dtype=bool)
    censored = np.array([False] * 4 + [True] * 4)
    assert mg.km_quantile(v, censored, 0.5) >= mg.km_quantile(v, uncensored, 0.5)


def test_median_is_none_when_too_much_is_censored():
    v = np.array([0.01, 0.02] + [0.05] * 8)
    c = np.array([False, False] + [True] * 8)
    assert mg.km_quantile(v, c, 0.5) is None       # never drops to S = 0.5


def test_margin_summary_reports_censoring_and_density():
    rng = np.random.default_rng(0)
    v = np.clip(rng.normal(0.02, 0.01, 500), 1e-4, EPS_8_255)
    c = v >= EPS_8_255
    s = mg.margin_summary(v, c)
    assert 0 <= s["frac_le_eps_ref"] <= 1
    assert 0 <= s["frac_censored"] <= 1
    assert s["rho_8_255"] is None or s["rho_8_255"] >= 0
    assert s["iqr"][0] is None or s["iqr"][0] <= s["median"]


def test_frac_le_eps_excludes_censored_points():
    v = np.array([0.001, 0.002, EPS_8_255, EPS_8_255])
    c = np.array([False, False, True, True])
    assert mg.margin_summary(v, c)["frac_le_eps_ref"] == pytest.approx(0.5)


def test_q90_interpolates_in_log_budget():
    rows = [{"victim_name": "V", "defense": "none", "attack": "best", "pool": "cifar100",
             "n_queries_actual": q, "fidelity_clean": f, "seed": 0}
            for q, f in [(5_000, 0.80), (50_000, 0.95)]]
    q90, censored = mg.q90_from_records(rows, "V")
    assert not censored
    assert 5_000 < q90 < 50_000
    # 0.90 sits 2/3 of the way from 0.80 to 0.95 in fidelity, so in log-budget too.
    expected = np.exp(np.log(5_000) + (2 / 3) * (np.log(50_000) - np.log(5_000)))
    assert q90 == pytest.approx(expected, rel=1e-6)


def test_q90_is_censored_when_never_reached():
    rows = [{"victim_name": "V", "defense": "none", "attack": "best", "pool": "cifar100",
             "n_queries_actual": q, "fidelity_clean": f, "seed": 0}
            for q, f in [(5_000, 0.50), (50_000, 0.70)]]
    q90, censored = mg.q90_from_records(rows, "V")
    assert censored and q90 == 50_000


def test_regression_reports_effect_size_not_significance():
    """S10.4 / pitfall 9: with n <= 8 victims, report rho and a CI, never a p-value."""
    frac = {f"v{i}": 0.2 + 0.05 * i for i in range(8)}
    q90 = {k: float(np.exp(20 - 10 * v)) for k, v in frac.items()}
    out = mg.regress_log_q90_on_margins(q90, frac, n_boot=500)
    assert out["spearman_rho"] == pytest.approx(-1.0, abs=1e-9)
    assert out["ols_slope_log_q90_per_frac"] == pytest.approx(-10.0, rel=1e-6)
    assert "underpowered" in out["note"]
    assert "p_value" not in out


def test_regression_degrades_gracefully_with_too_few_victims():
    out = mg.regress_log_q90_on_margins({"a": 1.0}, {"a": 0.5}, n_boot=10)
    assert out["n"] == 1 and np.isnan(out["spearman_rho"])
