#!/usr/bin/env bash
# Stage 5 - merge everything and run E5: analysis, D4-D8, Table 1 (spec.md S14).
#
# No GPU: src/analysis.py and src/figures.py are numpy/matplotlib only.
#
# Two passes, because n_eval must never be mixed inside one figure (pitfall 3):
#   n_eval=1000   the full sweep - every victim, attack and budget
#   n_eval=10000  the Q=200,000 cell only - the numbers that go in the paper
# Both write results/analysis.json and results/figures/, so the second pass
# overwrites the first; --results/--n-eval are recorded inside analysis.json.

#SBATCH --job-name=re-figures
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

python scripts/hpc/merge_results.py

echo
echo "=== analysis at n_eval=1000 (full sweep) ==="
python scripts/make_figures.py --n-eval 1000 --attack best
cp -a "$RESULTS_DIR/analysis.json" "$RESULTS_DIR/analysis_n1000.json"
rm -rf "$RESULTS_DIR/figures_n1000"
cp -a "$RESULTS_DIR/figures" "$RESULTS_DIR/figures_n1000"

echo
echo "=== analysis at n_eval=10000 (paper numbers, Q=200k cell) ==="
python scripts/make_figures.py --n-eval 10000 --attack best

# The environment block for REPRO.md S1 is captured on a GPU node by
# 10_verify_victims.sbatch; this node has no accelerator, so it is not re-run here.
echo "finished $(date -Is)"
