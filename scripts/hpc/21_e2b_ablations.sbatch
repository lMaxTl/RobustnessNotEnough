#!/usr/bin/env bash
# Stage 3 - E2b ablations at Q=50,000, seed 0 (spec.md S13.2): 29 runs.
#   attack ablation  8 victims x {jbda, noise}
#   pool ablation    3 victims x {tinyimagenet, cifar10_train, uniform_noise}
#   capacity         2 victims x {wrn_16_4, wrn_28_10} students
#
# One array task per victim; tasks are uneven by construction (Standard and
# WRN-94-16 appear in all three blocks, and the wrn_28_10 student costs ~10x a
# resnet18 student per epoch), which is fine - they run concurrently.
#
# Needs the Tiny ImageNet pool staged by 01_stage_data.sh.

#SBATCH --job-name=re-e2b
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=36:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

VICTIM="${VICTIMS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$SHARD_DIR/e2b_${VICTIM}.jsonl"

echo "victim     $VICTIM"
echo "results    $OUT"
echo

# Every victim appears in the attack ablation, so all eight tasks have work;
# the pool and capacity blocks only make three of them longer.
python scripts/run_sweep.py \
    --config configs/e2b_ablations.yaml \
    --victims "$VICTIM" \
    --results "$OUT" \
    --done-also "$MERGED_RESULTS" \
    "${MAXSEC[@]}"

echo "finished $(date -Is)"
