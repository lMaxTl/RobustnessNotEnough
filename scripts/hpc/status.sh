#!/usr/bin/env bash
# What is done, what is left, what is running. Login node, seconds to run.
#
#   bash scripts/hpc/status.sh
#
# Merges the shard files (harmless while jobs are still writing - they append to
# their own files) and asks each config what it still has pending.

# No pipefail here on purpose: this script greps and heads its way through the
# logs, and "no match" or an early `head` must not abort a status report.
set -eu

REPO="${RE_REPO:-$PWD}"
[[ -f "$REPO/scripts/run_sweep.py" ]] || {
    echo "ERROR: run from the repository root (or set RE_REPO)." >&2; exit 1; }
cd "$REPO"
# shellcheck disable=SC1091
source "${RE_VENV:-$REPO/.venv-horeka}/bin/activate"
# shellcheck disable=SC1091
source "$REPO/scripts/hpc/_bundled_cuda.sh"
export RE_NO_DOWNLOAD=1

echo "=== queue ==="
squeue --me --format="%.10i %.14j %.3t %.11M %.11l %R" 2>/dev/null || echo "(no squeue here)"

echo
echo "=== victim verification (tau denominators) ==="
python - <<'PY'
import json, pathlib
p = pathlib.Path("results/victim_cache.json")
parts = sorted(pathlib.Path("results/victim_cache_parts").glob("*.json"))
merged = json.loads(p.read_text()) if p.exists() else {}
print(f"merged cache: {len(merged)} entries ({p if p.exists() else 'not written yet'})")
print(f"parts on disk: {len(parts)}")
for f in parts:
    for name, e in json.loads(f.read_text()).items():
        aa = e.get("aa_acc")
        print(f"  {name:44s} clean={e.get('clean_acc', float('nan')):.4f} "
              f"AA={'-' if aa is None else f'{aa:.4f}'} @ n_eval={e.get('n_eval_aa')}")
PY

echo
echo "=== records ==="
python scripts/hpc/merge_results.py --check | tail -n 12

echo
echo "=== pending per config ==="
python scripts/hpc/merge_results.py >/dev/null
for cfg in configs/e2_main.yaml configs/e2b_ablations.yaml configs/e2c_control.yaml \
           configs/e4_defenses.yaml configs/e4_watermark.yaml configs/e3_margins.yaml; do
    python scripts/run_sweep.py --config "$cfg" --dry-run 2>/dev/null \
        | grep -E "^config|would compute" | head -n 9
done

echo
echo "=== failed / walled-out tasks in logs/ ==="
suspect=$(grep -lE "CANCELLED|DUE TO TIME LIMIT|Traceback|walltime budget exhausted" \
              logs/*.out 2>/dev/null | tail -n 20 || true)
echo "${suspect:-(none)}"
