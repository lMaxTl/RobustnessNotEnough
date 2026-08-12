#!/usr/bin/env bash
# Stage 1 - measure every victim's clean and AutoAttack accuracy (spec.md S6.1).
#
# One array task per victim, each with a private cache file: verify_victims.py
# rewrites results/victim_cache.json wholesale after every victim, so eight
# concurrent processes sharing one path would keep only the last writer's work.
# Stage 2 merges the parts.
#
# Nothing downstream may start before this finishes: victim_aa_acc is the tau
# denominator for the entire study, and run_sweep.py refuses to run without it.
#
#   sbatch scripts/hpc/10_verify_victims.sbatch          # n_eval=10000 (paper)
#   RE_N_EVAL=1000 sbatch scripts/hpc/10_verify_victims.sbatch
#
# n_eval=10000 on the two largest victims (WRN-70-16, WRN-94-16 and Peng's
# RaWRN-70-16) is the longest single job in the campaign - AutoAttack's Square
# stage alone is 5,000 forward passes per surviving point. If one of them walls
# out, resubmit just that task:
#   sbatch --array=7 scripts/hpc/10_verify_victims.sbatch
# or send the big three to a faster partition:
#   sbatch --array=4-7 --partition=accelerated-h100 scripts/hpc/10_verify_victims.sbatch

#SBATCH --job-name=re-verify
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

VICTIM="${VICTIMS[${SLURM_ARRAY_TASK_ID:-0}]}"
N_EVAL="${RE_N_EVAL:-10000}"
export RE_VICTIM_CACHE="$CACHE_PART_DIR/${VICTIM}.json"

echo "victim     $VICTIM"
echo "n_eval     $N_EVAL"
echo "cache      $RE_VICTIM_CACHE"
echo

# REPRO.md S1 wants the environment of the machine the runs happened on, which
# is neither the login node nor the CPU node that makes the figures. Capture it
# once, from a GPU task.
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    python -m src.repro | tee "$RESULTS_DIR/environment_horeka.txt"
    echo
fi

python scripts/verify_victims.py \
    --victims "$VICTIM" \
    --n-eval "$N_EVAL" \
    --n-clean 10000 \
    --batch-size "${RE_AA_BATCH:-128}"

echo "finished $(date -Is)"
