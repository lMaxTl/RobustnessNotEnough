"""Merge the per-victim caches written by the verification array into one file.

    python scripts/hpc/merge_victim_cache.py

Each array task of 10_verify_victims.sbatch writes
results/victim_cache_parts/<victim>.json (RE_VICTIM_CACHE). This folds them into
results/victim_cache.json, which is what every sweep reads, and refuses to
produce a cache that would silently poison tau: a victim without a finite
`aa_acc`, or a set of victims measured at different `n_eval_aa`, is reported and
- for the mixed-n_eval case - is an error, because tau denominators from
different evaluation set sizes must never be averaged into one figure
(spec.md S9.1, pitfall 3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.victims import VICTIMS  # noqa: E402

PARTS_DIR = ROOT / "results" / "victim_cache_parts"
OUT_PATH = ROOT / "results" / "victim_cache.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", type=Path, default=PARTS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write the merge even if some victims are missing AA")
    args = ap.parse_args()

    merged: dict[str, dict] = {}
    if args.out.exists():
        merged.update(json.loads(args.out.read_text()))

    parts = sorted(args.parts.glob("*.json"))
    for p in parts:
        for name, entry in json.loads(p.read_text()).items():
            merged.setdefault(name, {}).update(entry)
    print(f"merged {len(parts)} part file(s) -> {len(merged)} victim entries")

    problems, n_evals = [], {}
    for name in VICTIMS:
        entry = merged.get(name)
        if entry is None:
            problems.append(f"{name}: not measured")
            continue
        aa = entry.get("aa_acc")
        if aa is None or not math.isfinite(float(aa)):
            problems.append(f"{name}: no finite aa_acc")
            continue
        n_evals[name] = entry.get("n_eval_aa")
        lb = VICTIMS[name].aa_leaderboard
        print(f"  {name:44s} clean={entry.get('clean_acc', float('nan')):.4f} "
              f"AA={float(aa):.4f} @ n_eval={entry.get('n_eval_aa')} "
              f"(leaderboard {lb:.3f}, drift {float(aa) - lb:+.3f})")

    distinct = set(n_evals.values())
    if len(distinct) > 1:
        print(f"\nABORT: victims were verified at different n_eval: {sorted(distinct)}. "
              "One denominator per figure (spec.md S9.1). Re-run the odd ones out at "
              "the same n_eval before merging.", file=sys.stderr)
        return 2
    if problems:
        print("\n" + "\n".join(f"MISSING: {p}" for p in problems), file=sys.stderr)
        if not args.allow_incomplete:
            print("ABORT: run_sweep.py would refuse these victims anyway "
                  "(spec.md S6.1). Re-submit the failed array tasks, or pass "
                  "--allow-incomplete to merge what exists.", file=sys.stderr)
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True))
    tmp.replace(args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
