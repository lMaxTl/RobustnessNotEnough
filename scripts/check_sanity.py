"""Gate for configs/e0_sanity.yaml: fail loudly if extraction does not work.

    python scripts/check_sanity.py results/sanity/e0_sanity.jsonl

The floor is deliberately loose. It is not what a healthy run should reach; it
is what no healthy run can fall below. A student of the non-robust `Standard`
victim that agrees with it on fewer than 60% of test images after 50,000
soft-label queries means the pipeline is broken, not that the victim is hard to
steal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIDELITY_FLOOR = 0.60          # Standard victim, random attack, Q=50k
CHANCE = 0.10


def main(path: str) -> int:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        print(f"FAIL: no records in {path}")
        return 1
    print(f"{'victim':28s} {'attack':7s} {'Q':>7s} {'clean':>7s} {'fid':>7s} {'AA':>7s} {'PGD40':>7s}")
    for r in sorted(rows, key=lambda r: (r["victim_name"], r["attack"])):
        print(f"{r['victim_name'][:28]:28s} {r['attack']:7s} {r['n_queries_actual']:7d} "
              f"{r['student_clean_acc']:7.3f} {r['fidelity_clean']:7.3f} "
              f"{r['student_aa_acc']:7.3f} {r['student_pgd40_acc']:7.3f}")

    failures = []
    gate = [r for r in rows if r["victim_name"] == "Standard" and r["attack"] == "random"]
    if not gate:
        failures.append("no Standard/random record: the gate itself did not run")
    for r in gate:
        if r["fidelity_clean"] < FIDELITY_FLOOR:
            failures.append(f"Standard/random fidelity {r['fidelity_clean']:.3f} "
                            f"< floor {FIDELITY_FLOOR:.2f}")
    for r in rows:
        if r["student_clean_acc"] <= CHANCE + 0.02:
            failures.append(f"{r['victim_name']}/{r['attack']}: student at chance "
                            f"(clean {r['student_clean_acc']:.3f})")
    if failures:
        print("\nFAIL - do not start a sweep:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nOK - extraction recovers the Standard victim.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/sanity/e0_sanity.jsonl"))
