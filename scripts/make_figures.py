"""Run the E5 analysis and emit D4-D8 (spec.md S2, S14).

    python scripts/make_figures.py                      # uses the largest n_eval present
    python scripts/make_figures.py --n-eval 1000 --attack best

Writes:
    results/analysis.json          full E5 report (tau fits, leverage, F1-F3)
    results/defenses.jsonl         D4
    results/figures/fig1..3.{pdf,png}, table1_leverage.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import analysis as an  # noqa: E402
from src import figures as fg  # noqa: E402
from src.schema import (DEFENSES_PATH, MARGINS_PATH, RESULTS_PATH,  # noqa: E402
                        JsonlWriter, read_jsonl)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_PATH)
    ap.add_argument("--margins", type=Path, default=MARGINS_PATH)
    ap.add_argument("--n-eval", type=int, default=None, choices=[1000, 10000])
    ap.add_argument("--attack", default="best")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--defense-budget", type=int, default=50_000)
    args = ap.parse_args()

    all_rows = list(read_jsonl(args.results))
    if not all_rows:
        print(f"no results in {args.results}; run scripts/run_sweep.py first")
        return 1

    report = an.run_analysis(args.results, args.margins,
                             out_path=args.results.parent / "analysis.json",
                             n_eval=args.n_eval, n_boot=args.n_boot)
    print(f"analysis: {report['n_runs']} runs at n_eval={report['n_eval']}")
    for attack, entry in report["attacks"].items():
        fit = entry.get("leverage_table", {}).get("tau_fit")
        if fit:
            print(f"  {attack}: tau slope {fit['slope']:+.3f} "
                  f"CI [{fit['slope_ci95'][0]:+.3f}, {fit['slope_ci95'][1]:+.3f}] "
                  f"over {fit['n_victims']} victims")
        f = entry.get("falsification")
        if f:
            print(f"    F1={f['F1']['triggered']} F2={f['F2']['triggered']} "
                  f"F3={f['F3']['triggered']} -> {f['verdict']}")

    # D4
    defense_rows = an.summarize_defenses(all_rows, budget=args.defense_budget,
                                         attack=args.attack)
    if defense_rows:
        if DEFENSES_PATH.exists():
            DEFENSES_PATH.unlink()   # regenerated wholesale from results.jsonl
        JsonlWriter(DEFENSES_PATH).extend(defense_rows)
        print(f"wrote {DEFENSES_PATH} ({len(defense_rows)} defense summaries)")
        for r in defense_rows:
            ce = "n/a" if r.cost_effectiveness is None else f"{r.cost_effectiveness:.1f}"
            print(f"  {r.victim_name[:28]:28s} {r.defense:10s} "
                  f"+{r.log10_cost_multiplier:.2f} decades "
                  f"@ {100*r.delta_clean_acc:.2f} pts utility -> {ce}"
                  f"{' (censored)' if r.c_att_censored else ''}")

    made = fg.make_all(args.results, args.margins, n_eval=args.n_eval,
                       attack=args.attack, budget=args.budget, n_boot=args.n_boot)
    for k, v in made.items():
        if k == "table1":
            print("\n" + v)
        else:
            print(f"{k}: {v}")
    print(json.dumps({"figures_dir": str(fg.FIG_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
