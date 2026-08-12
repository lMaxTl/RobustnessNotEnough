#!/usr/bin/env bash
# Stage 2 - fold the eight per-victim caches into results/victim_cache.json.
#
# Seconds of work, but it has to sit between two array jobs in the dependency
# chain, so it is a job. No GPU.
#
# It exits non-zero if any victim lacks a finite AutoAttack accuracy, which
# stops the whole downstream chain on purpose: tau without a measured
# denominator is not a number worth spending 300 GPU-hours on.

#SBATCH --job-name=re-merge-cache
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

python scripts/hpc/merge_victim_cache.py
