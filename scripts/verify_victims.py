"""Verify every victim's clean and AutoAttack accuracy ourselves (spec.md S6.1).

The leaderboard numbers in S6.1 are planning figures. The paper reports what
this script measures. Results are cached in results/victim_cache.json and are
the tau denominators for the whole study, so run this before any sweep.

    python scripts/verify_victims.py --n-eval 1000          # quick pass
    python scripts/verify_victims.py --n-eval 10000         # for the paper
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import victims as vic  # noqa: E402
from src.repro import setup_determinism  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--victims", nargs="*", default=sorted(vic.VICTIMS),
                    help="subset of victim keys (default: all 8 RobustBench victims; "
                         f"locally trained victims must be named explicitly: "
                         f"{sorted(vic.LOCAL_VICTIMS)})")
    ap.add_argument("--n-eval", type=int, default=1000, choices=[1000, 10000])
    ap.add_argument("--n-clean", type=int, default=10000,
                    help="images for clean accuracy (the S6.2 assertion uses the full set)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-aa", action="store_true", help="clean accuracy only")
    ap.add_argument("--force", action="store_true", help="re-measure even if cached")
    args = ap.parse_args()

    setup_determinism(seed=0)
    cache_path = vic.victim_cache_path()
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    print(f"{'victim':44s} {'params':>12s} {'clean':>8s} {'AA':>8s} {'leaderboard':>12s}  time")
    failures = []
    for name in args.victims:
        t0 = time.time()
        entry = cache.get(name, {})
        v = vic.load_victim(name, device=args.device, measure_clean=False)
        if args.force or "clean_acc" not in entry:
            v.clean_acc = vic.measure_clean_accuracy(v, n=args.n_clean)
            entry["clean_acc"] = v.clean_acc
            entry["n_clean_eval"] = args.n_clean
        else:
            v.clean_acc = entry["clean_acc"]
        entry["fwd_flops"] = v.fwd_flops
        entry["n_params"] = v.n_params

        try:
            vic.assert_normalisation_sane(v)   # S6.2 hard abort
        except RuntimeError as exc:
            failures.append(str(exc))
            print(f"{name:44s}  FAILED: {exc}")
            continue

        aa = entry.get("aa_acc")
        if not args.skip_aa and (args.force or aa is None
                                 or entry.get("n_eval_aa") != args.n_eval):
            aa = ev.autoattack_accuracy(v.model, n_eval=args.n_eval,
                                        batch_size=args.batch_size)
            entry["aa_acc"] = aa
            entry["n_eval_aa"] = args.n_eval
        v.aa_acc = float(aa) if aa is not None else float("nan")

        cache[name] = entry
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

        # A locally trained victim (S11.3) has no leaderboard entry to drift from.
        spec = vic.VICTIMS.get(name)
        lb = spec.aa_leaderboard if spec else float("nan")
        drift = "" if aa is None or math.isnan(lb) else f"  (drift {aa - lb:+.3f})"
        print(f"{name:44s} {v.n_params:>12,} {v.clean_acc:>8.4f} "
              f"{(aa if aa is not None else float('nan')):>8.4f} {lb:>12.3f}"
              f"{drift}  {time.time()-t0:.0f}s")

    if failures:
        print("\nABORT: normalisation check failed for "
              f"{len(failures)} victim(s) (spec.md S6.2)")
        return 1
    print(f"\nwrote {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
