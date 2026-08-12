#!/usr/bin/env bash
# Stage 3 - E2c, the temperature-matched control (spec.md S9.4, S13.3): 21 runs.
#
# The most important control in the paper: it separates "robustness transfers"
# from "high-entropy posteriors distil better". Each task takes one robust
# victim as an *entropy target*, scales `Standard`'s logits to match its mean
# predictive entropy on the query pool, and extracts from that.
#
# 7 array tasks, one entropy target each (3 seeds per task). Every task loads
# `Standard` plus its own robust victim, so the entropy target is measured
# inside the task and nothing is shared between them.

#SBATCH --job-name=re-e2c
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

VICTIM="${E2C_VICTIMS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$SHARD_DIR/e2c_${VICTIM}.jsonl"

echo "entropy target  $VICTIM"
echo "results         $OUT"
echo

# Records land under victim_name="Standard_T<T>", not under $VICTIM; the skip
# logic reads extra.matched_victim, so a resubmitted task resumes rather than
# duplicating what it already ran.
python scripts/run_sweep.py \
    --config configs/e2c_control.yaml \
    --victims "$VICTIM" \
    --results "$OUT" \
    --done-also "$MERGED_RESULTS" \
    "${MAXSEC[@]}"

echo "finished $(date -Is)"
