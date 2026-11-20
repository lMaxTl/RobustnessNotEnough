#!/usr/bin/env bash
# Stage 0c - E0, the extraction sanity gate: 4 runs, ~1.5 GPU-hours.
#
#   sbatch scripts/hpc/05_e0_sanity.sbatch
#
# Run it before any sweep and after any change to extraction, FLOP accounting
# or evaluation. It exits non-zero if a student of the non-robust `Standard`
# victim does not reach the fidelity floor in scripts/check_sanity.py.
#
# Deliberately NOT resumable against results.jsonl (no --done-also): the gate
# must retrain its students every time, or it tests nothing.

#SBATCH --job-name=re-e0
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

OUT_DIR="$RESULTS_DIR/sanity"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/e0_sanity_${SLURM_JOB_ID:-local}.jsonl"
echo "git        $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "results    $OUT"
echo

python scripts/run_sweep.py --config configs/e0_sanity.yaml --results "$OUT" "${MAXSEC[@]}"
python scripts/check_sanity.py "$OUT"

echo "finished $(date -Is)"
