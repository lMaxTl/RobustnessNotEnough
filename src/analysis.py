"""M11 - Analysis: the tau regression, leverage, and the falsification checks
(spec.md S14).

Reads only results.jsonl / margins.jsonl. Every extrapolated number is marked
as such: the measured victim range is roughly [0.595, 0.737] and the
interesting claims live at [0.80, 0.89] (S14.3).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import costmodel as cm
from .schema import MARGINS_PATH, RESULTS_PATH, read_jsonl

__all__ = [
    "load_results", "assert_consistent_n_eval", "tau_by_victim", "TauFit",
    "fit_tau", "attacker_cost_fit", "solve_rv_for_target", "leverage",
    "leverage_table", "delta_abs_table", "falsification_checks", "run_analysis",
    "LEVERAGE_TARGETS", "MEASURED_RANGE_NOTE",
]

# S14.3 leverage targets: the first is interpolated/defensible, the rest are
# extrapolations and must be shaded and labelled as such in every figure.
LEVERAGE_TARGETS = (0.737, 0.80, 0.85, 0.88)
MEASURED_RANGE_NOTE = (
    "Measured victim AutoAttack range is [{lo:.3f}, {hi:.3f}]. Targets outside "
    "this range are extrapolations of the fitted tau(R_v) and are shaded in "
    "every figure (spec.md S14.3)."
)


def load_results(path=RESULTS_PATH, n_eval: int | None = None,
                 defense: str | None = "none", include_controls: bool = False) -> list[dict]:
    rows = list(read_jsonl(path))
    if n_eval is not None:
        rows = [r for r in rows if r["n_eval"] == n_eval]
    if defense is not None:
        rows = [r for r in rows if r["defense"] == defense]
    if not include_controls:
        rows = [r for r in rows if not r["is_control"]]
    return rows


def assert_consistent_n_eval(rows: list[dict]) -> int:
    """Pitfall 3: never plot a 1,000-image AA estimate against a 10,000-image one."""
    vals = {r["n_eval"] for r in rows}
    if len(vals) > 1:
        raise AssertionError(
            f"mixed n_eval in one figure: {sorted(vals)}. A 1,000-image AutoAttack "
            "estimate carries ~1.5 points of standard error; filter first.")
    return vals.pop() if vals else 0


# --------------------------------------------------------------------------
# S14.1 the tau regression
# --------------------------------------------------------------------------

def tau_by_victim(rows: list[dict]) -> dict[str, dict]:
    """Aggregate tau over seeds for each victim.

    The variance floor is the binomial standard error of an AutoAttack estimate
    on n_eval images: three seeds can agree by luck, and a zero variance would
    otherwise give a single victim infinite weight.
    """
    out: dict[str, dict] = {}
    for r in rows:
        if r["tau"] is None or r["victim_aa_acc"] <= 0:
            continue  # `Standard` has AA = 0; tau undefined (S9.3)
        out.setdefault(r["victim_name"], {"R_v": r["victim_aa_acc"], "taus": [],
                                          "aa": [], "n_eval": r["n_eval"],
                                          "c_total": []})
        out[r["victim_name"]]["taus"].append(r["tau"])
        out[r["victim_name"]]["aa"].append(r["student_aa_acc"])
        out[r["victim_name"]]["c_total"].append(r["c_total"])
    for v in out.values():
        taus = np.asarray(v["taus"], dtype=float)
        aa = np.asarray(v["aa"], dtype=float)
        n_eval, R_v = v["n_eval"], v["R_v"]
        seed_var = float(taus.var(ddof=1)) if len(taus) > 1 else 0.0
        p = float(aa.mean())
        binom_var = p * (1 - p) / max(1, n_eval) / (R_v ** 2)  # var of aa/R_v
        v["tau_mean"] = float(taus.mean())
        v["tau_var"] = max(seed_var, binom_var, 1e-8)
        v["n_seeds"] = len(taus)
        v["c_att_mean"] = float(np.mean(v["c_total"]))
    return out


@dataclass
class TauFit:
    slope: float
    intercept: float
    slope_ci95: tuple[float, float]
    intercept_ci95: tuple[float, float]
    n_victims: int
    victims: list[str]
    r_range: tuple[float, float]
    attack: str | None
    budget: int | None
    n_boot: int
    boot_slopes: list[float]
    boot_intercepts: list[float]

    def tau_at(self, R_v: float) -> float:
        return self.slope * R_v + self.intercept

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("boot_slopes")
        d.pop("boot_intercepts")
        d["slope_ci_excludes_zero"] = bool(self.slope_ci95[0] > 0 or self.slope_ci95[1] < 0)
        return d


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least squares for y = m x + c. Closed form, no dependencies."""
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    sxx = (w * (x - mx) ** 2).sum()
    if sxx <= 0:
        return 0.0, my
    m = (w * (x - mx) * (y - my)).sum() / sxx
    return float(m), float(my - m * mx)


def fit_tau(rows: list[dict], attack: str | None = None, budget: int | None = None,
            n_boot: int = 10_000, seed: int = 0) -> TauFit:
    """Fit tau = m R_v + c by WLS with weights 1/Var(tau), bootstrapping victims.

    Victims are the unit of generalisation, so the bootstrap resamples victims,
    not seeds (S14.1). Fit separately per attack and per budget; do not pool.
    """
    sel = [r for r in rows
           if (attack is None or r["attack"] == attack)
           and (budget is None or r["n_queries_nominal"] == budget)]
    assert_consistent_n_eval(sel)
    agg = tau_by_victim(sel)
    names = sorted(agg)
    if len(names) < 2:
        raise ValueError(f"need >= 2 victims with defined tau, got {len(names)}")
    x = np.array([agg[k]["R_v"] for k in names])
    y = np.array([agg[k]["tau_mean"] for k in names])
    w = np.array([1.0 / agg[k]["tau_var"] for k in names])

    m, c = _wls(x, y, w)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, 2))
    for b in range(n_boot):
        idx = rng.integers(0, len(names), len(names))
        if len(np.unique(x[idx])) < 2:
            boots[b] = (np.nan, np.nan)
            continue
        boots[b] = _wls(x[idx], y[idx], w[idx])
    ok = ~np.isnan(boots[:, 0])
    m_lo, m_hi = np.percentile(boots[ok, 0], [2.5, 97.5])
    c_lo, c_hi = np.percentile(boots[ok, 1], [2.5, 97.5])
    return TauFit(slope=m, intercept=c, slope_ci95=(float(m_lo), float(m_hi)),
                  intercept_ci95=(float(c_lo), float(c_hi)), n_victims=len(names),
                  victims=names, r_range=(float(x.min()), float(x.max())),
                  attack=attack, budget=budget, n_boot=int(ok.sum()),
                  boot_slopes=boots[ok, 0].tolist(),
                  boot_intercepts=boots[ok, 1].tolist())


# --------------------------------------------------------------------------
# S14.2 leverage
# --------------------------------------------------------------------------

def attacker_cost_fit(rows: list[dict], attack: str | None = None,
                      budget: int | None = None) -> dict:
    """log10 C_att = p R_v + q across victims (F3 test: is p comparable to the
    defender's slope?). Also returns the flat-mean model, which is the null the
    scientific claim asserts."""
    sel = [r for r in rows
           if (attack is None or r["attack"] == attack)
           and (budget is None or r["n_queries_nominal"] == budget)]
    agg = tau_by_victim(sel)
    names = sorted(agg)
    x = np.array([agg[k]["R_v"] for k in names])
    y = np.log10(np.array([agg[k]["c_att_mean"] for k in names]))
    if len(names) >= 2:
        p, q = np.polyfit(x, y, 1)
    else:
        p, q = 0.0, float(y.mean()) if len(y) else float("nan")
    return {"slope_log10_per_R": float(p), "intercept_log10": float(q),
            "flat_mean_log10": float(y.mean()) if len(y) else float("nan"),
            "n_victims": len(names), "victims": names,
            "spread_log10": float(y.max() - y.min()) if len(y) else float("nan")}


def bootstrap_fits(rows: list[dict], attack: str | None = None, budget: int | None = None,
                   n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Joint victim bootstrap of (tau slope, tau intercept, cost slope, cost intercept).

    Both fits must be resampled *together*: leverage depends on tau and on the
    attacker-cost curve, and they are estimated from the same 7-8 victims, so
    treating them as independent would understate the interval.
    """
    sel = [r for r in rows
           if (attack is None or r["attack"] == attack)
           and (budget is None or r["n_queries_nominal"] == budget)]
    agg = tau_by_victim(sel)
    names = sorted(agg)
    x = np.array([agg[k]["R_v"] for k in names])
    y = np.array([agg[k]["tau_mean"] for k in names])
    w = np.array([1.0 / agg[k]["tau_var"] for k in names])
    z = np.log10(np.array([agg[k]["c_att_mean"] for k in names]))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(names), len(names))
        if len(np.unique(x[idx])) < 2:
            continue
        m, c = _wls(x[idx], y[idx], w[idx])
        p, q = np.polyfit(x[idx], z[idx], 1)
        out.append((m, c, float(p), float(q)))
    return np.asarray(out)


def solve_rv_for_target(R_target: float, fit: TauFit) -> float:
    """Solve tau(R_v) R_v = R_target for R_v with tau(R_v) = m R_v + c.

    m R_v^2 + c R_v - R = 0. The physically meaningful root is the positive one
    in (0, 1]; if the linear tau model cannot reach the target at any R_v <= 1,
    raise rather than return a fabricated value.
    """
    m, c = fit.slope, fit.intercept
    if abs(m) < 1e-12:
        if c <= 0:
            raise ValueError("degenerate tau fit: tau <= 0")
        R_v = R_target / c
    else:
        disc = c * c + 4 * m * R_target
        if disc < 0:
            raise ValueError(f"no real R_v reaches tau*R_v = {R_target}")
        roots = [(-c + math.sqrt(disc)) / (2 * m), (-c - math.sqrt(disc)) / (2 * m)]
        cand = [r for r in roots if 0 < r <= 1.0]
        if not cand:
            raise ValueError(
                f"tau*R_v = {R_target} is unreachable for R_v in (0,1] under the "
                f"fitted tau (slope={m:.3f}, intercept={c:.3f})")
        R_v = min(cand)
    if R_v > 1.0:
        raise ValueError(f"solved R_v = {R_v:.3f} > 1: target {R_target} unreachable")
    return float(R_v)


def leverage(R_target: float, fit: TauFit, cost_fit: dict,
             fid: float = 0.0, use_flat_attacker_cost: bool = False,
             mode: str = "matched") -> dict:
    """C_def(R_v, FID=0) / C_att(R_v): training cost vs extraction cost.

    Two readings of spec.md S14.2/S14.3, both reported because they answer
    different questions and the spec uses R for both:

    `matched`  (S14.2 as written) - R is the robustness the *attacker wants to
        end up with*, so solve tau(R_v) R_v = R for the victim to target. With a
        measured tau below 1 this is unreachable for R above tau(1), and the
        function raises instead of inventing a victim above the accuracy range.

    `direct`   (S14.3's labelling) - R is the *victim's* robustness level, and
        leverage is what a thief saves by stealing a model at R instead of
        training one. This is defined at all four of S14.3's targets, and it is
        the reading under which R = 0.737 is "interpolated, defensible": that is
        exactly the strongest measured victim.
    """
    if mode == "direct":
        R_v = float(R_target)
    elif mode == "matched":
        R_v = solve_rv_for_target(R_target, fit)
    else:
        raise ValueError(f"mode must be 'matched' or 'direct', got {mode!r}")
    c_def = cm.flops_for_accuracy(R_v, fid)
    if use_flat_attacker_cost:
        log_c_att = cost_fit["flat_mean_log10"]
    else:
        log_c_att = cost_fit["slope_log10_per_R"] * R_v + cost_fit["intercept_log10"]
    c_att = 10 ** log_c_att
    lo, hi = fit.r_range
    return {"R_target": R_target, "mode": mode, "R_v_required": R_v, "c_def": c_def,
            "c_att": c_att, "leverage": c_def / c_att,
            "log10_leverage": math.log10(c_def / c_att),
            "tau_at_R_v": fit.tau_at(R_v),
            "student_aa_obtained": fit.tau_at(R_v) * R_v,
            "extrapolated": not (lo <= R_v <= hi)}


def leverage_table(rows: list[dict], attack: str = "best", budget: int | None = None,
                   targets=LEVERAGE_TARGETS, n_boot: int = 10_000,
                   seed: int = 0, fid: float = 0.0, mode: str = "direct") -> dict:
    """D8 / Table 1, with the tau bootstrap propagated through to leverage.

    `mode="direct"` is the default because it is defined at all four S14.3
    targets; the S14.2 `matched` value is reported alongside each row, or the
    reason it has no solution.
    """
    fit = fit_tau(rows, attack=attack, budget=budget, n_boot=n_boot, seed=seed)
    cost_fit = attacker_cost_fit(rows, attack=attack, budget=budget)
    lo, hi = fit.r_range
    out = {"attack": attack, "budget": budget, "measured_R_range": [lo, hi],
           "mode": mode, "note": MEASURED_RANGE_NOTE.format(lo=lo, hi=hi),
           "tau_fit": fit.summary(), "attacker_cost_fit": cost_fit, "rows": []}

    draws = bootstrap_fits(rows, attack=attack, budget=budget, n_boot=n_boot, seed=seed)
    for R in targets:
        try:
            base = leverage(R, fit, cost_fit, fid=fid, mode=mode)
        except ValueError as exc:
            out["rows"].append({"R_target": R, "error": str(exc)})
            continue
        # The S14.2 reading, reported next to it or explained away.
        try:
            matched = leverage(R, fit, cost_fit, fid=fid, mode="matched")
            base["matched"] = {k: matched[k] for k in
                               ("R_v_required", "log10_leverage", "extrapolated")}
        except ValueError as exc:
            base["matched"] = {"error": str(exc)}
        # Propagate the joint victim bootstrap: each replicate carries its own
        # tau fit *and* its own attacker-cost fit through the whole chain.
        vals = []
        for m_b, c_b, p_b, q_b in draws:
            f_b = TauFit(slope=float(m_b), intercept=float(c_b),
                         slope_ci95=fit.slope_ci95, intercept_ci95=fit.intercept_ci95,
                         n_victims=fit.n_victims, victims=fit.victims,
                         r_range=fit.r_range, attack=fit.attack, budget=fit.budget,
                         n_boot=fit.n_boot, boot_slopes=[], boot_intercepts=[])
            cost_b = {**cost_fit, "slope_log10_per_R": float(p_b),
                      "intercept_log10": float(q_b)}
            try:
                vals.append(leverage(R, f_b, cost_b, fid=fid, mode=mode)["log10_leverage"])
            except ValueError:
                continue
        ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals \
            else (float("nan"), float("nan"))
        out["rows"].append({**base, "log10_leverage_ci95": list(ci)})
    return out


# --------------------------------------------------------------------------
# delta_abs and the falsification conditions (S1)
# --------------------------------------------------------------------------

def delta_abs_table(rows: list[dict], baseline_victim: str = "Standard") -> list[dict]:
    """student_aa - student_aa(from `Standard`) at matched (Q, attack, pool, seed)."""
    key = lambda r: (r["n_queries_nominal"], r["attack"], r["pool"], r["seed"],
                     r["student_arch"], r["n_eval"], r["defense"])
    base = {key(r): r["student_aa_acc"] for r in rows if r["victim_name"] == baseline_victim}
    out = []
    for r in rows:
        if r["victim_name"] == baseline_victim or key(r) not in base:
            continue
        out.append({"victim_name": r["victim_name"], "victim_aa_acc": r["victim_aa_acc"],
                    "attack": r["attack"], "pool": r["pool"],
                    "n_queries": r["n_queries_nominal"], "seed": r["seed"],
                    "student_aa_acc": r["student_aa_acc"],
                    "delta_abs": r["student_aa_acc"] - base[key(r)]})
    return out


def aa_per_log10_cost_slope(rows: list[dict], victim_name: str, attack: str) -> float | None:
    """d(student AA) / d(log10 attacker FLOPs) from the undefended budget sweep.

    The exchange rate between attacker compute and stolen robustness for one
    victim. Used to convert a defense-induced AA drop into the extra compute
    needed to undo it.
    """
    sel = [r for r in rows if r["victim_name"] == victim_name and r["attack"] == attack
           and r["defense"] == "none"]
    by_budget: dict[int, list[tuple[float, float]]] = {}
    for r in sel:
        by_budget.setdefault(r["n_queries_nominal"], []).append(
            (math.log10(max(1, r["c_total"])), r["student_aa_acc"]))
    if len(by_budget) < 2:
        return None
    xs = [float(np.mean([p[0] for p in v])) for v in by_budget.values()]
    ys = [float(np.mean([p[1] for p in v])) for v in by_budget.values()]
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope if slope > 1e-6 else None


def summarize_defenses(all_rows: list[dict], sweep_rows: list[dict] | None = None,
                       attack: str = "best", budget: int = 50_000,
                       censor_factor: float = 10.0) -> list:
    """D4: per-defense attacker-cost multiplier and defender utility cost (S11.4).

    C_att(d) is the attacker cost needed to reach the *same* student AA as with
    `none`. We measure one budget per defense, so the required cost is projected
    along the measured AA-vs-log10(cost) slope of the undefended budget sweep
    for that victim:

        log10 C_att(d) = log10 C_att(d, B) + (AA_none - AA_d) / slope

    The assumption (the defended condition improves with compute at the same
    rate as the undefended one) is stated in every record. If the projection
    exceeds `censor_factor` x the largest measured cost, the record is censored
    and the value is a lower bound.
    """
    from .schema import DefenseRecord, gpu_name

    sweep_rows = sweep_rows if sweep_rows is not None else all_rows
    sel = [r for r in all_rows if r["attack"] == attack
           and r["n_queries_nominal"] == budget and not r["is_control"]]
    out = []
    for victim_name in sorted({r["victim_name"] for r in sel}):
        rows_v = [r for r in sel if r["victim_name"] == victim_name]
        base = [r for r in rows_v if r["defense"] == "none"]
        if not base:
            continue
        aa_none = float(np.mean([r["student_aa_acc"] for r in base]))
        c_none = float(np.mean([r["c_total"] for r in base]))
        clean_none = float(np.mean([r["victim_clean_acc"] for r in base]))
        slope = aa_per_log10_cost_slope(sweep_rows, victim_name, attack)
        max_cost = max((r["c_total"] for r in sweep_rows
                        if r["victim_name"] == victim_name), default=c_none)
        censor_log = math.log10(censor_factor * max_cost)

        for defense in sorted({r["defense"] for r in rows_v if r["defense"] != "none"}):
            rows_d = [r for r in rows_v if r["defense"] == defense]
            aa_d = float(np.mean([r["student_aa_acc"] for r in rows_d]))
            c_d = float(np.mean([r["c_total"] for r in rows_d]))
            util = float(np.mean([r["extra"].get("defense_utility_cost", 0.0)
                                  for r in rows_d]))
            detect = [r["extra"].get("prada_detect_index") for r in rows_d]
            detect_idx = int(np.mean([d for d in detect if d])) if any(detect) else None

            if aa_d >= aa_none or slope is None:
                # The defense did not slow the attacker down (or we cannot
                # project without a budget sweep).
                log_required = math.log10(c_d)
                censored = slope is None
            else:
                log_required = math.log10(c_d) + (aa_none - aa_d) / slope
                censored = log_required > censor_log
                if censored:
                    log_required = censor_log
            mult = log_required - math.log10(c_none)
            out.append(DefenseRecord(
                victim_name=victim_name, defense=defense,
                defense_params=rows_d[0]["defense_params"],
                victim_clean_acc_none=clean_none,
                victim_clean_acc_defended=clean_none - util,
                delta_clean_acc=util, student_aa_acc_none=aa_none,
                student_aa_acc_defended=aa_d, c_att_none=c_none,
                c_att_defended=10 ** log_required, c_att_censored=bool(censored),
                log10_cost_multiplier=mult,
                # Orders of magnitude of attacker cost per point of defender
                # utility. None when the defense costs no measurable utility -
                # the ratio is then undefined, not infinite.
                cost_effectiveness=(mult / util if util > 1e-6 else None),
                n_seeds=len(rows_d), detect_index=detect_idx,
                wallclock_sec=float(np.sum([r["wallclock_sec"] for r in rows_d])),
                gpu=rows_d[0].get("gpu", gpu_name()),
            ))
    return out


def falsification_checks(rows: list[dict], all_rows: list[dict] | None = None,
                         attack: str = "best", budget: int | None = None,
                         n_boot: int = 10_000, seed: int = 0) -> dict:
    """F1-F3 (spec.md S1). Report honestly; a clean negative result is publishable."""
    fit = fit_tau(rows, attack=attack, budget=budget, n_boot=n_boot, seed=seed)
    cost_fit = attacker_cost_fit(rows, attack=attack, budget=budget)

    # F1: tau decreases steeply in R_v, CI excluding 0 -> robustness is partly
    # self-protecting.
    f1 = {"slope": fit.slope, "ci95": list(fit.slope_ci95),
          "triggered": bool(fit.slope_ci95[1] < 0),
          "meaning": "tau decreases significantly in R_v: robustness is partially "
                     "self-protecting (F1)"}

    # F2: the temperature-matched control reproduces the effect -> we measured
    # label smoothing, not robustness.
    f2: dict = {"triggered": None, "meaning": "temperature-matched Standard reaches "
                                              "comparable student AA (F2)"}
    if all_rows:
        ctrl = [r for r in all_rows if r["is_control"] and r["attack"] == attack]
        robust = [r for r in all_rows if not r["is_control"] and r["attack"] == attack
                  and r["victim_aa_acc"] > 0 and r["defense"] == "none"]
        if ctrl and robust:
            budgets = {r["n_queries_nominal"] for r in ctrl}
            robust = [r for r in robust if r["n_queries_nominal"] in budgets]
            c_mean = float(np.mean([r["student_aa_acc"] for r in ctrl]))
            r_mean = float(np.mean([r["student_aa_acc"] for r in robust])) if robust else float("nan")
            f2.update({"control_student_aa_mean": c_mean,
                       "robust_student_aa_mean": r_mean,
                       "n_control_runs": len(ctrl), "n_robust_runs": len(robust),
                       # "Comparable" = within 2 points of the robust-victim mean.
                       "triggered": bool(abs(c_mean - r_mean) < 0.02)})

    # F3: attacker FLOPs grow super-linearly in R_v at a rate comparable to the
    # defender curve. Compare d log10 C / dR over the measured range.
    lo, hi = fit.r_range
    def_slope = (math.log10(cm.flops_for_accuracy(hi, 0.0))
                 - math.log10(cm.flops_for_accuracy(lo, 0.0))) / max(1e-9, hi - lo)
    att_slope = cost_fit["slope_log10_per_R"]
    f3 = {"defender_log10_slope_per_R": def_slope,
          "attacker_log10_slope_per_R": att_slope,
          "ratio": att_slope / def_slope if def_slope else float("nan"),
          "triggered": bool(def_slope > 0 and att_slope / def_slope > 0.5),
          "meaning": "attacker cost grows at a rate comparable to the defender "
                     "curve (F3): no divergence in leverage"}
    return {"F1": f1, "F2": f2, "F3": f3,
            "verdict": ("falsified" if (f1["triggered"] or f2.get("triggered")
                                        or f3["triggered"]) else "not falsified")}


# --------------------------------------------------------------------------

def run_analysis(results_path=RESULTS_PATH, margins_path=MARGINS_PATH,
                 out_path: Path | None = None, n_eval: int | None = None,
                 attacks=("random", "best"), n_boot: int = 10_000) -> dict:
    """Everything E5 needs, as one JSON blob (also the source for figures)."""
    all_rows = list(read_jsonl(results_path))
    if n_eval is None:
        evals = {r["n_eval"] for r in all_rows}
        n_eval = max(evals) if evals else 1000
    rows = load_results(results_path, n_eval=n_eval)
    report: dict = {"n_eval": n_eval, "n_runs": len(rows), "attacks": {}}

    for attack in attacks:
        sel = [r for r in rows if r["attack"] == attack]
        if not sel:
            continue
        budgets = sorted({r["n_queries_nominal"] for r in sel})
        per_budget = {}
        for b in budgets:
            try:
                per_budget[b] = fit_tau(sel, attack=attack, budget=b,
                                        n_boot=n_boot).summary()
            except ValueError as exc:
                per_budget[b] = {"error": str(exc)}
        largest = budgets[-1] if budgets else None
        entry = {"tau_fit_per_budget": per_budget,
                 "attacker_cost_fit": attacker_cost_fit(sel, attack=attack, budget=largest)}
        try:
            entry["leverage_table"] = leverage_table(sel, attack=attack, budget=largest,
                                                     n_boot=n_boot)
            entry["falsification"] = falsification_checks(sel, all_rows=all_rows,
                                                          attack=attack, budget=largest,
                                                          n_boot=n_boot)
        except ValueError as exc:
            entry["error"] = str(exc)
        report["attacks"][attack] = entry

    # S10.4 link between sample efficiency and boundary geometry.
    margins = list(read_jsonl(margins_path))
    if margins:
        frac = {m["victim_name"]: m["frac_margin_le_8_255"] for m in margins}
        q90, censored = {}, {}
        for name in frac:
            v, cen = q90_from_all(rows, name)
            if v is not None:
                q90[name] = v
                censored[name] = cen
        if len(q90) >= 3:
            from .margins import regress_log_q90_on_margins

            report["margin_link"] = regress_log_q90_on_margins(q90, frac, n_boot=n_boot)
            report["margin_link"]["q90"] = q90
            report["margin_link"]["q90_censored"] = censored

    report["delta_abs"] = delta_abs_table(rows)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2, default=float))
    return report


def q90_from_all(rows: list[dict], victim_name: str, attack: str = "best"):
    from .margins import q90_from_records

    return q90_from_records(rows, victim_name, attack=attack)


if __name__ == "__main__":  # pragma: no cover
    rep = run_analysis(out_path=RESULTS_PATH.parent / "analysis.json")
    print(json.dumps({k: v for k, v in rep.items() if k != "delta_abs"},
                     indent=2, default=float)[:4000])
