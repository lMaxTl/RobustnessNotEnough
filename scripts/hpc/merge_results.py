"""Fold the per-job shard files into results/results.jsonl and margins.jsonl.

    python scripts/hpc/merge_results.py            # merge and report
    python scripts/hpc/merge_results.py --check    # report only, write nothing

Every array task appends to its own file under results/shards/ (or
margin_shards/) rather than to the shared results.jsonl: JSONL append is only
atomic for short lines on a local filesystem, and a margin record carrying 2,000
per-image margins is ~40 kB. Merging is therefore a separate, sequential step.

Duplicates are dropped on the spec.md S12 identity
(victim, attack, pool, budget, student, seed, defense, epochs, n_eval), keeping
the earliest record by timestamp. They should be rare - shards are disjoint by
construction - but a task that was killed after writing and then resubmitted
before a merge can produce one, and a duplicated run would quietly double a
cell's weight in the analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.schema import MARGINS_PATH, RESULTS_PATH, read_jsonl  # noqa: E402

SHARD_DIR = ROOT / "results" / "shards"
MARGIN_SHARD_DIR = ROOT / "results" / "margin_shards"


def short(p: Path) -> str:
    """Repo-relative when it can be, absolute otherwise."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def result_key(r: dict) -> tuple:
    # `matched_victim` (set on S9.4 temperature-control records, naming the
    # robust victim they're compared against) must never replace victim_name:
    # a control's grid point is identical to that victim's real E2/E4 cell
    # (same attack/pool/budget/seed), so substituting the name collides two
    # different experiments onto one key and silently drops one. Carry it as
    # an extra field instead, so it can only widen the key, never narrow it.
    matched = (r.get("extra") or {}).get("matched_victim")
    return (r["victim_name"], matched, r["attack"], r["pool"], r["n_queries_nominal"],
            r["student_arch"], r["seed"], r["defense"], r["epochs"], r["n_eval"])


def merge(paths: list[Path], out: Path, key_fn, check: bool) -> int:
    """Merge `paths` into `out`, deduplicating on key_fn. Returns record count."""
    records: dict[tuple, dict] = {}
    dupes = Counter()
    sources = ([out] if out.exists() else []) + sorted(paths)
    for p in sources:
        n_before = len(records)
        for r in read_jsonl(p):
            k = key_fn(r)
            if k in records:
                dupes[p.name] += 1
                # Keep the earliest: if a run was recorded twice, the first
                # write is the one whose wallclock and provenance are unbroken.
                if r.get("timestamp", "") < records[k].get("timestamp", ""):
                    records[k] = r
            else:
                records[k] = r
        print(f"  {short(p)}: +{len(records) - n_before} new"
              f"{f', {dupes[p.name]} duplicate(s)' if dupes[p.name] else ''}")
    if dupes:
        print(f"  dropped {sum(dupes.values())} duplicate record(s)")
    if not check:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in records.values():
                fh.write(json.dumps(r) + "\n")
        tmp.replace(out)
    return len(records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, default=SHARD_DIR)
    ap.add_argument("--margin-shards", type=Path, default=MARGIN_SHARD_DIR)
    ap.add_argument("--results", type=Path, default=RESULTS_PATH)
    ap.add_argument("--margins", type=Path, default=MARGINS_PATH)
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    print("results:")
    n = merge(sorted(args.shards.glob("*.jsonl")), args.results, result_key, args.check)
    print(f"  -> {short(args.results)}: {n} records")

    print("margins:")
    m = merge(sorted(args.margin_shards.glob("*.jsonl")), args.margins,
              lambda r: (r["victim_name"],), args.check)
    print(f"  -> {short(args.margins)}: {m} records")

    if n:
        rows = list(read_jsonl(args.results))
        by_block = Counter((r.get("extra") or {}).get("block", "?") for r in rows)
        by_n_eval = Counter(r["n_eval"] for r in rows)
        # One extraction writes one record per n_eval, each carrying the whole
        # run's wallclock, so summing every record would double-count the 200k
        # cell. Take the largest wallclock per extraction instead.
        per_run: dict[tuple, float] = {}
        for r in rows:
            k = result_key(r)[:-1]
            per_run[k] = max(per_run.get(k, 0.0), r["wallclock_sec"])
        hours = sum(per_run.values()) / 3600
        print("\nrecords per block: " + ", ".join(f"{k}={v}" for k, v in sorted(by_block.items())))
        print("records per n_eval: " + ", ".join(f"{k}={v}" for k, v in sorted(by_n_eval.items())))
        print(f"GPU-hours recorded: {hours:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
