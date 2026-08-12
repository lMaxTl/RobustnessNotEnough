"""M7 - Boundary geometry via FAB-T (spec.md S10).

Purpose (S10.1): show that the irreducible error the source paper proves is
unavoidable *is itself* the attacker's supervision signal. A robust victim's
decision boundary sits further from the data, so its posteriors carry more
usable information about where the boundary is.

Censoring matters here. FAB searches within a bound; points it fails to flip
are right-censored, not "large". A plain mean over those is wrong, so the
median/IQR come from a Kaplan-Meier estimator.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import data as _data
from .evaluate import EPS_8_255
from .schema import MarginRecord, gpu_name

__all__ = ["compute_margins", "kaplan_meier", "km_quantile", "near_boundary_density",
           "margin_summary", "q90_from_records", "regress_log_q90_on_margins"]


# --------------------------------------------------------------------------
# S10.2 FAB-T margins
# --------------------------------------------------------------------------

def compute_margins(victim, n_images: int = 2000, eps: float = EPS_8_255,
                    n_restarts: int = 3, n_iter: int = 100,
                    batch_size: int = 64, seed: int = 0) -> MarginRecord:
    """Minimal l_inf perturbation to flip the victim's output on `n_images`.

    Labels are the true test labels, so an already-misclassified image has
    margin 0 by definition. That is what makes `frac_margin_le_8_255` directly
    comparable to 1 - AA (S10.3): both count "correct and stable" against
    everything else.
    """
    from autoattack import AutoAttack

    t0 = time.time()
    x, y = _data.cifar10_test(n_images)
    device = victim.device
    xf = _data.to_float(x).to(device)
    yf = y.to(device)

    adv = AutoAttack(victim.model, norm="Linf", eps=eps, version="custom",
                     attacks_to_run=["fab-t"], device=device, verbose=False)
    adv.fab.n_restarts = n_restarts
    adv.fab.n_iter = n_iter
    adv.seed = seed
    x_adv = adv.run_standard_evaluation(xf, yf, bs=batch_size)

    delta = (x_adv - xf).abs().flatten(1).max(dim=1).values
    with torch.no_grad():
        flipped = torch.cat([
            (victim.model(x_adv[i:i + batch_size]).argmax(1) != yf[i:i + batch_size])
            for i in range(0, len(x_adv), batch_size)
        ])
    margins = delta.cpu().numpy().astype(float)
    censored = (~flipped).cpu().numpy()
    # A censored point's true margin is somewhere above the search bound; record
    # the bound itself, flagged, rather than a fabricated value.
    margins[censored] = eps

    summary = margin_summary(margins, censored, eps=eps)
    return MarginRecord(
        victim_name=victim.name, victim_aa_acc=float(victim.aa_acc),
        n_images=len(margins), eps_ref=EPS_8_255,
        margins=[float(m) for m in margins], censored=[bool(c) for c in censored],
        median_margin=summary["median"], iqr=summary["iqr"],
        frac_margin_le_8_255=summary["frac_le_eps_ref"],
        rho_8_255=summary["rho_8_255"], fab_n_restarts=n_restarts,
        fab_n_iter=n_iter, search_bound=float(eps),
        wallclock_sec=time.time() - t0, gpu=gpu_name(),
    ).validate()


# --------------------------------------------------------------------------
# Survival statistics
# --------------------------------------------------------------------------

def kaplan_meier(values: np.ndarray, censored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival curve S(t) = P(margin > t) for right-censored data."""
    values = np.asarray(values, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    order = np.argsort(values, kind="mergesort")
    v, c = values[order], censored[order]
    n = len(v)
    times, surv = [], []
    s, at_risk, i = 1.0, n, 0
    while i < n:
        t = v[i]
        j = i
        d = 0  # events (uncensored) at this time
        while j < n and v[j] == t:
            d += int(not c[j])
            j += 1
        if d > 0 and at_risk > 0:
            s *= 1.0 - d / at_risk
        times.append(t)
        surv.append(s)
        at_risk -= (j - i)
        i = j
    return np.asarray(times), np.asarray(surv)


def km_quantile(values: np.ndarray, censored: np.ndarray, q: float) -> float | None:
    """q-th quantile from the KM curve; None if unreachable (too much censoring)."""
    times, surv = kaplan_meier(values, censored)
    target = 1.0 - q
    idx = np.nonzero(surv <= target + 1e-12)[0]
    return float(times[idx[0]]) if len(idx) else None


def near_boundary_density(margins: np.ndarray, censored: np.ndarray,
                          at: float = EPS_8_255, bandwidth: float | None = None) -> float:
    """Density of margins at `at`, per unit of l_inf.

    Gaussian KDE over the uncensored margins, rescaled by the uncensored
    fraction so that it integrates to P(uncensored) rather than to 1.
    """
    from scipy import stats

    m = np.asarray(margins, dtype=float)[~np.asarray(censored, dtype=bool)]
    if len(m) < 5 or float(np.std(m)) == 0.0:
        return float("nan")
    kde = stats.gaussian_kde(m, bw_method=bandwidth)
    return float(kde(at)[0] * len(m) / len(margins))


def margin_summary(margins, censored, eps: float = EPS_8_255) -> dict:
    m = np.asarray(margins, dtype=float)
    c = np.asarray(censored, dtype=bool)
    q25 = km_quantile(m, c, 0.25)
    q50 = km_quantile(m, c, 0.50)
    q75 = km_quantile(m, c, 0.75)
    rho = near_boundary_density(m, c, at=EPS_8_255)
    return {
        # None, not NaN: "unreachable because too much of the sample is censored"
        # is a fact about the estimator, and JSON should say null.
        "median": q50,
        "iqr": [q25, q75],
        # Censored points are, by construction, above the bound.
        "frac_le_eps_ref": float(np.mean((m <= EPS_8_255 + 1e-12) & ~c)),
        "frac_censored": float(np.mean(c)),
        "rho_8_255": None if math.isnan(rho) else rho,
    }


def assert_margin_aa_agreement(record: MarginRecord, tol: float = 0.03) -> None:
    """S10.3: frac_margin_le_8_255 must match 1 - AA within 3 points.

    This validates both pipelines at once: if it fails, either the margin
    computation or the AutoAttack evaluation is wrong.
    """
    expected = 1.0 - record.victim_aa_acc
    got = record.frac_margin_le_8_255
    if not math.isfinite(expected):
        raise ValueError(f"victim {record.victim_name} has no verified AA accuracy")
    if abs(got - expected) > tol:
        raise AssertionError(
            f"{record.victim_name}: frac_margin_le_8_255={got:.4f} vs 1-AA={expected:.4f} "
            f"(diff {abs(got-expected):.4f} > {tol}). FAB is weaker than full AutoAttack, "
            "so a small positive gap is expected; this one is too large."
        )


# --------------------------------------------------------------------------
# S10.4 the link: sample efficiency vs boundary geometry
# --------------------------------------------------------------------------

def q90_from_records(records: list[dict], victim_name: str, target: float = 0.90,
                     attack: str | None = None, pool: str | None = None,
                     defense: str = "none") -> tuple[float | None, bool]:
    """Q_90(v): budget at which fidelity_clean first reaches `target`.

    Linear interpolation in log(Q) between the two bracketing budgets, averaged
    over seeds. Returns (value, censored); censored=True means the target was
    never reached within the measured budgets, and the value is then the
    largest budget tried (a lower bound).
    """
    rows = [r for r in records
            if r["victim_name"] == victim_name and r["defense"] == defense
            and (attack is None or r["attack"] == attack)
            and (pool is None or r["pool"] == pool)]
    if not rows:
        return None, True
    by_budget: dict[int, list[float]] = {}
    for r in rows:
        by_budget.setdefault(int(r["n_queries_actual"]), []).append(r["fidelity_clean"])
    budgets = sorted(by_budget)
    fid = [float(np.mean(by_budget[b])) for b in budgets]
    for i, f in enumerate(fid):
        if f >= target:
            if i == 0:
                return float(budgets[0]), False
            b0, b1, f0, f1 = budgets[i - 1], budgets[i], fid[i - 1], fid[i]
            if f1 == f0:
                return float(b1), False
            w = (target - f0) / (f1 - f0)
            return float(math.exp(math.log(b0) + w * (math.log(b1) - math.log(b0)))), False
    return float(budgets[-1]), True


def regress_log_q90_on_margins(q90: dict[str, float], frac_margin: dict[str, float],
                               n_boot: int = 10_000, seed: int = 0) -> dict:
    """Spearman rho between log Q_90 and frac_margin_le_8_255, with bootstrap CI.

    S10.4: with 7-8 victims this is underpowered. Report the effect size and
    the interval; do not report a p-value as if n = 8 supported one.
    """
    from scipy import stats

    names = sorted(set(q90) & set(frac_margin))
    x = np.array([frac_margin[k] for k in names])
    y = np.log(np.array([q90[k] for k in names]))
    if len(names) < 3:
        return {"n": len(names), "spearman_rho": float("nan"), "ci95": [float("nan")] * 2,
                "note": "fewer than 3 victims; no estimate"}
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(names), len(names))
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boots.append(stats.spearmanr(x[idx], y[idx]).statistic)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2)
    slope = float(np.polyfit(x, y, 1)[0]) if len(names) > 1 else float("nan")
    return {
        "n": len(names), "victims": names, "spearman_rho": rho,
        "ci95": [float(lo), float(hi)], "ols_slope_log_q90_per_frac": slope,
        "note": ("n <= 8 victims: underpowered by construction. Effect size and CI "
                 "only; suggestive, not confirmatory (spec.md S10.4, pitfall 9)."),
    }
