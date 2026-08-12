#!/usr/bin/env bash
# Stage 3 - E2, the main sweep (spec.md S13.1): 192 runs.
#   8 victims x {random, best} x {5k, 20k, 50k, 200k} x 3 seeds,
#   plus n_eval=10000 re-evaluation of the 200k cell (48 extra records).
#
# 32 array tasks = 8 victims x 4 shards. One victim per task, so the big
# checkpoint loads once and the victim's PGD-40 evaluation set is built once.
# Within a victim the 24 runs are split budget-descending round-robin, so no
# shard collects all the Q=200,000 work.
#
#   sbatch scripts/hpc/20_e2_main.sbatch
#   sbatch --array=12,13 scripts/hpc/20_e2_main.sbatch     # re-run two tasks
#
# Resumable: each task appends to its own shard file and skips runs already
# recorded there or in the merged results.jsonl, so a walled-out task continues
# where it stopped when resubmitted with the same --array index.

#SBATCH --job-name=re-e2
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=48:00:00
#SBATCH --array=0-31%16
#SBATCH --output=logs/%x_%A_%a.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

NUM_SHARDS=4
TASK="${SLURM_ARRAY_TASK_ID:-0}"
VICTIM="${VICTIMS[$(( TASK / NUM_SHARDS ))]}"
SHARD=$(( TASK % NUM_SHARDS ))
OUT="$SHARD_DIR/e2_${VICTIM}_s${SHARD}.jsonl"

echo "victim     $VICTIM"
echo "shard      $SHARD / $NUM_SHARDS"
echo "results    $OUT"
echo

python scripts/run_sweep.py \
    --config configs/e2_main.yaml \
    --victims "$VICTIM" \
    --shard "$SHARD" --num-shards "$NUM_SHARDS" \
    --results "$OUT" \
    --done-also "$MERGED_RESULTS" \
    "${MAXSEC[@]}"

echo "finished $(date -Is)"
