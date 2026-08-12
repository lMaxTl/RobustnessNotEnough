#!/usr/bin/env bash
# Stage 3 - E4, defenses (spec.md S11, S13.5): 48 runs.
#   2 victims x {none, top1, round_1, round_2, topk_2, mad_0.5, mad_0.99, prada}
#   x 3 seeds, all at Q=50,000.
#
# 8 array tasks = 2 victims x 4 shards of 6 runs.
#
# `prada` is the slow one and not because of the GPU: it feeds every query
# through a nearest-neighbour test against a growing set on the CPU, so budget
# roughly an extra half hour for those three runs.

#SBATCH --job-name=re-e4
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

NUM_SHARDS=4
TASK="${SLURM_ARRAY_TASK_ID:-0}"
VICTIM="${E4_VICTIMS[$(( TASK / NUM_SHARDS ))]}"
SHARD=$(( TASK % NUM_SHARDS ))
OUT="$SHARD_DIR/e4_${VICTIM}_s${SHARD}.jsonl"

echo "victim     $VICTIM"
echo "shard      $SHARD / $NUM_SHARDS"
echo "results    $OUT"
echo

python scripts/run_sweep.py \
    --config configs/e4_defenses.yaml \
    --victims "$VICTIM" \
    --shard "$SHARD" --num-shards "$NUM_SHARDS" \
    --results "$OUT" \
    --done-also "$MERGED_RESULTS" \
    "${MAXSEC[@]}"

echo "finished $(date -Is)"
