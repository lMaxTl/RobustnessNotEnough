#!/usr/bin/env bash
# Stage 3 - E3, boundary geometry (spec.md S10, S13.4): 8 victims, no sweep.
#
# FAB-T, 2,000 test images, 3 restarts x 100 iterations. AutoAttack's targeted
# FAB runs 9 target classes, so this is ~2,700 forward+backward passes per
# surviving image - hours on the 70-16 and 94-16 victims, minutes on Standard.
#
# One task per victim, each writing its own margins shard. A margin record
# carries all 2,000 per-image margins (~40 kB of JSON), which is exactly the
# kind of line you do not want two jobs appending to one file on a parallel
# filesystem at the same time.

#SBATCH --job-name=re-e3
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

VICTIM="${VICTIMS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$MARGIN_SHARD_DIR/e3_${VICTIM}.jsonl"

echo "victim     $VICTIM"
echo "margins    $OUT"
echo

# The S10.3 cross-check (frac_margin_le_8_255 vs 1 - AA) prints a WARNING
# rather than failing the job; it is a validation signal for the analysis, and
# it needs the verified AA accuracy from stage 2 to mean anything.
python scripts/run_sweep.py \
    --config configs/e3_margins.yaml \
    --victims "$VICTIM" \
    --margins "$OUT" \
    --done-also "$MERGED_MARGINS" \
    "${MAXSEC[@]}"

echo "finished $(date -Is)"
