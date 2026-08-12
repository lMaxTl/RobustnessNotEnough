#!/usr/bin/env bash
# Stage 4a - train the one watermarked victim (spec.md S11.3).
#
# TRADES (beta=5, PGD-10, eps=8/255) on WRN-28-10 for 100 epochs with a 1%
# trigger backdoor. The only adversarial-training run in the project, ~8
# GPU-hours on an A100 by spec.md S16 - call it 10-16 with the determinism
# settings on. Not a sweep, and NOT resumable: the script checkpoints only at
# the end, so this job must not be allowed to wall out. Hence 36 hours.
#
# Smoke-test the wiring first if you like (two epochs, tiny subset):
#   RE_WM_ARGS="--epochs 2 --smoke" sbatch scripts/hpc/41_train_watermark.sbatch
# but note that overwrites models/watermark_wrn2810_trades.pt with a toy model.

#SBATCH --job-name=re-watermark
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

# shellcheck disable=SC2086
python scripts/train_watermark_victim.py ${RE_WM_ARGS:---epochs 100}

echo "finished $(date -Is)"
