"""Figures D5-D8 (spec.md S2).

Every figure that shows an extrapolated quantity shades the extrapolated region
and states the measured range in the caption (S14.3). Every figure asserts a
single `n_eval` across the data it plots (pitfall 3).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analysis as an  # noqa: E402
from . import costmodel as cm  # noqa: E402
from .evaluate import EPS_8_255  # noqa: E402
from .schema import MARGINS_PATH, RESULTS_PATH, read_jsonl  # noqa: E402

__all__ = ["figure1_defender_vs_attacker", "figure2_tau_vs_rv", "figure3_margin_cdf",
           "table1_leverage", "make_all"]

FIG_DIR = RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
# Colourblind-safe (Okabe-Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
SHADE = "#BBBBBB"


def _short_name(name: str, width: int = 24) -> str:
    """Shorten a victim key without losing what distinguishes it.

    Plain truncation collapses `Bartoldson2024Adversarial_WRN-82-8` and
    `..._WRN-94-16` to the same label, which is exactly the pair a reader most
    needs to tell apart.
    """
    if len(name) <= width:
        return name
    head, tail = name[: width // 2 - 1], name[-(width // 2):]
    return f"{head}…{tail}"


def _save(fig, name: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# D5 - Figure 1
# --------------------------------------------------------------------------

def figure1_defender_vs_attacker(rows: list[dict], attack: str = "best",
                                 budget: int | None = None, fid: float = 0.0):
    """Defender vs attacker FLOPs by robustness target, log y.

    The defender curve is the closed-form C_def from M1 (convention: K_ADV).
    The attacker points are *measured* FLOPs (convention: FlopCounterMode).
    Pitfall 6: the two conventions are named in the legend, not silently mixed.
    """
    n_eval = an.assert_consistent_n_eval(rows)
    sel = [r for r in rows if r["attack"] == attack
           and (budget is None or r["n_queries_nominal"] == budget)]
    agg = an.tau_by_victim(sel)
    names = sorted(agg)
    R_meas = np.array([agg[k]["R_v"] for k in names])
    C_att = np.array([agg[k]["c_att_mean"] for k in names])

    # Stop the curve at a compute ceiling rather than at the asymptote: C_def
    # passes 1e47 within a thousandth of Acc_inf, and 30 decades of empty axis
    # would bury the region the data actually covers. The asymptote is marked
    # with a vertical line instead.
    C_CEILING = 1e32
    R_grid = np.linspace(0.50, cm.asymptotic_accuracy(fid) - 1e-4, 2000)
    C_def = np.array([cm.flops_for_accuracy(float(R), fid) for R in R_grid])
    keep = C_def <= C_CEILING
    R_grid, C_def = R_grid[keep], C_def[keep]

    asymptote = cm.asymptotic_accuracy(fid)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    lo, hi = (R_meas.min(), R_meas.max()) if len(R_meas) else (0.595, 0.737)
    # Shade to the asymptote, not to the truncated grid: everything right of the
    # strongest measured victim is extrapolation (S14.3).
    ax.axvspan(hi, asymptote, color=SHADE, alpha=0.35, lw=0,
               label=f"extrapolated (measured $R_v \\in$ [{lo:.3f}, {hi:.3f}])")
    ax.plot(R_grid, C_def, color=PALETTE[0], lw=2,
            label=r"defender: $C_{\rm def}(R)$, $K_{\rm adv}ND$ (M1)")
    ax.axvline(asymptote, color=PALETTE[0], ls=":", lw=1)
    ax.text(asymptote, C_def[0], f" asymptote {asymptote:.3f}", rotation=90,
            va="bottom", ha="right", fontsize=8, color=PALETTE[0])

    if len(R_meas):
        ax.scatter(R_meas, C_att, color=PALETTE[1], zorder=5, s=36,
                   label=f"attacker: measured FLOPs ({attack}, FlopCounterMode)")
        if len(R_meas) >= 2:
            p, q = np.polyfit(R_meas, np.log10(C_att), 1)
            xs = np.linspace(0.50, R_grid[-1], 100)
            ax.plot(xs, 10 ** (p * xs + q), color=PALETTE[1], ls="--", lw=1.5,
                    label=f"attacker fit: $\\log_{{10}} C \\propto {p:+.2f}\\,R_v$")

    ax.set_yscale("log")
    ax.set_xlabel("AutoAttack accuracy target $R$ (CIFAR-10, $\\ell_\\infty$, 8/255)")
    ax.set_ylabel("training / extraction cost [FLOPs]")
    ax.set_title(f"Producing robustness vs extracting it (n_eval={n_eval})", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    return _save(fig, "fig1_defender_vs_attacker")


# --------------------------------------------------------------------------
# D6 - Figure 2
# --------------------------------------------------------------------------

def figure2_tau_vs_rv(rows: list[dict], attacks=("random", "best"),
                      budget: int | None = None, n_boot: int = 10_000):
    """tau vs R_v with per-attack series and a bootstrap 95% band."""
    n_eval = an.assert_consistent_n_eval(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = np.linspace(0.55, 0.90, 200)

    for i, attack in enumerate(attacks):
        sel = [r for r in rows if r["attack"] == attack
               and (budget is None or r["n_queries_nominal"] == budget)]
        agg = an.tau_by_victim(sel)
        if len(agg) < 2:
            continue
        names = sorted(agg)
        x = np.array([agg[k]["R_v"] for k in names])
        y = np.array([agg[k]["tau_mean"] for k in names])
        e = np.array([math.sqrt(agg[k]["tau_var"]) for k in names])
        fit = an.fit_tau(sel, attack=attack, budget=budget, n_boot=n_boot)
        band = np.array([[s * xv + c for xv in xs]
                         for s, c in zip(fit.boot_slopes, fit.boot_intercepts)])
        lo_b, hi_b = np.percentile(band, [2.5, 97.5], axis=0)

        col = PALETTE[i % len(PALETTE)]
        ax.errorbar(x, y, yerr=e, fmt="o", color=col, capsize=3, ms=5,
                    label=f"{attack}: slope {fit.slope:+.2f} "
                          f"[{fit.slope_ci95[0]:+.2f}, {fit.slope_ci95[1]:+.2f}]")
        ax.plot(xs, fit.slope * xs + fit.intercept, color=col, lw=1.5)
        ax.fill_between(xs, lo_b, hi_b, color=col, alpha=0.15, lw=0)
        ax.axvspan(x.max(), xs[-1], color=SHADE, alpha=0.20, lw=0, zorder=0)

    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("victim AutoAttack accuracy $R_v$")
    ax.set_ylabel(r"robustness transfer ratio $\tau = \mathrm{AA}_{\rm student}/\mathrm{AA}_{\rm victim}$")
    ax.set_title(f"Transfer ratio vs victim robustness (n_eval={n_eval}, "
                 f"Q={budget or 'all'})", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, "fig2_tau_vs_rv")


# --------------------------------------------------------------------------
# D7 - Figure 3
# --------------------------------------------------------------------------

def figure3_margin_cdf(margins_path=MARGINS_PATH):
    """Margin CDF per victim with the 8/255 line marked.

    Censored points (FAB failed within the search bound) are shown as the
    plateau at the bound: the CDF is drawn only up to the bound, since nothing
    is known about the distribution beyond it.
    """
    records = list(read_jsonl(margins_path))
    if not records:
        raise FileNotFoundError(f"no margin records in {margins_path}")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    records.sort(key=lambda r: r["victim_aa_acc"])
    for i, rec in enumerate(records):
        m = np.asarray(rec["margins"], dtype=float)
        c = np.asarray(rec["censored"], dtype=bool)
        xs = np.sort(m[~c])
        ys = np.arange(1, len(xs) + 1) / len(m)
        ax.step(xs, ys, where="post", color=PALETTE[i % len(PALETTE)], lw=1.6,
                label=f"{_short_name(rec['victim_name'])} "
                      f"(AA={rec['victim_aa_acc']:.3f}, "
                      f"{100*rec['frac_margin_le_8_255']:.1f}% $\\leq$ 8/255)")
    ax.axvline(EPS_8_255, color="k", ls="--", lw=1.2)
    ax.text(EPS_8_255, 0.97, r"$\varepsilon = 8/255$  ", fontsize=9, rotation=90,
            va="top", ha="right", transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"minimal $\ell_\infty$ margin (FAB-T)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Boundary geometry: how much of the test set sits within 8/255",
                 fontsize=11)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    return _save(fig, "fig3_margin_cdf")


# --------------------------------------------------------------------------
# D8 - Table 1
# --------------------------------------------------------------------------

def table1_leverage(rows: list[dict], attack: str = "best", budget: int | None = None,
                    targets=an.LEVERAGE_TARGETS, n_boot: int = 10_000,
                    mode: str = "direct") -> str:
    """Leverage table as markdown; extrapolated rows are flagged in the table."""
    tab = an.leverage_table(rows, attack=attack, budget=budget, targets=targets,
                            n_boot=n_boot, mode=mode)
    fit = tab["tau_fit"]
    lines = [
        f"**Table 1.** Leverage = $C_\\mathrm{{def}}(R)/C_\\mathrm{{att}}(R)$ for attack "
        f"`{attack}`, budget {budget}: FLOPs to *train* a model at AutoAttack accuracy "
        f"$R$ divided by FLOPs to *extract* one from a victim at $R$. {tab['note']} "
        f"Fitted $\\tau = {fit['slope']:+.3f} R_v {fit['intercept']:+.3f}$ "
        f"(95% CI on the slope [{fit['slope_ci95'][0]:+.3f}, {fit['slope_ci95'][1]:+.3f}], "
        f"{fit['n_victims']} victims). The last column is the spec.md §14.2 reading, in "
        f"which $R$ is the robustness the attacker ends up with and the required victim "
        f"solves $\\tau(R_v)R_v = R$.",
        "",
        "| target R | $C_{def}$ [FLOPs] | $C_{att}$ [FLOPs] | leverage | "
        "log10 leverage 95% CI | student AA obtained | status | §14.2 matched $R_v$ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in tab["rows"]:
        if "error" in r:
            lines.append(f"| {r['R_target']:.3f} | n/a | n/a | n/a | n/a | n/a | "
                         f"unreachable: {r['error']} | n/a |")
            continue
        ci = r.get("log10_leverage_ci95", [float("nan")] * 2)
        m = r.get("matched", {})
        matched = ("unreachable" if "error" in m
                   else f"{m['R_v_required']:.3f}"
                        f"{' (extrap.)' if m.get('extrapolated') else ''}")
        lines.append(
            f"| {r['R_target']:.3f} | {r['c_def']:.2e} | {r['c_att']:.2e} | "
            f"{r['leverage']:.3g} | [{ci[0]:.2f}, {ci[1]:.2f}] | "
            f"{r['student_aa_obtained']:.3f} | "
            f"{'**extrapolated**' if r['extrapolated'] else 'interpolated'} | "
            f"{matched} |")
    out = "\n".join(lines)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "table1_leverage.md").write_text(out, encoding="utf-8")
    return out


def make_all(results_path=RESULTS_PATH, margins_path=MARGINS_PATH,
             n_eval: int | None = None, attack: str = "best",
             budget: int | None = None, n_boot: int = 10_000) -> dict:
    all_rows = list(read_jsonl(results_path))
    if not all_rows:
        raise FileNotFoundError(f"no results in {results_path}")
    if n_eval is None:
        n_eval = max(r["n_eval"] for r in all_rows)
    rows = an.load_results(results_path, n_eval=n_eval)
    if budget is None:
        budget = max(r["n_queries_nominal"] for r in rows)
    made: dict[str, object] = {}
    made["fig1"] = figure1_defender_vs_attacker(rows, attack=attack, budget=budget)
    made["fig2"] = figure2_tau_vs_rv(rows, budget=budget, n_boot=n_boot)
    try:
        made["fig3"] = figure3_margin_cdf(margins_path)
    except FileNotFoundError as exc:
        made["fig3"] = f"skipped: {exc}"
    made["table1"] = table1_leverage(rows, attack=attack, budget=budget, n_boot=n_boot)
    return made
