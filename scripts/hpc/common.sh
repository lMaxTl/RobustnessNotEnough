# Shared environment for every HoreKa job in this project.
# Sourced, never executed:  source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
#
# Everything configurable is an environment variable with a default, so the
# batch scripts themselves need no editing. sbatch exports the submitting
# environment by default, so `RE_VENV=... sbatch ...` works.

set -euo pipefail

# --- where things live -----------------------------------------------------
# sbatch runs the batch script from a copy in Slurm's spool directory, so the
# repository cannot be derived from $0. $SLURM_SUBMIT_DIR is the directory the
# job was submitted from - hence "submit from the repository root".
REPO="${RE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO/scripts/run_sweep.py" ]]; then
    echo "ERROR: '$REPO' is not the repository root (no scripts/run_sweep.py)." >&2
    echo "       Submit from the repository root, or set RE_REPO=/path/to/repo." >&2
    exit 1
fi
cd "$REPO"

VENV="${RE_VENV:-$REPO/.venv-horeka}"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: no virtualenv at '$VENV'. Run scripts/hpc/00_setup_env.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# shellcheck disable=SC1091
source "$REPO/scripts/hpc/_bundled_cuda.sh"

# --- runtime environment ---------------------------------------------------
# Compute nodes are treated as offline: every dataset and checkpoint is staged
# from the login node by 01_stage_data.sh. RE_NO_DOWNLOAD makes a missing
# dataset raise immediately instead of stalling a GPU job on a download.
export RE_DATA_ROOT="${RE_DATA_ROOT:-$REPO/data}"
export RE_NO_DOWNLOAD="${RE_NO_DOWNLOAD:-1}"
# spec.md S15: must be set before the CUDA context exists. setup_determinism()
# also sets it, but only if the process gets there first; exporting it here
# removes the ordering question entirely.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${USER:-$(id -un)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$MPLCONFIGDIR"

RESULTS_DIR="$REPO/results"
SHARD_DIR="$RESULTS_DIR/shards"
MARGIN_SHARD_DIR="$RESULTS_DIR/margin_shards"
CACHE_PART_DIR="$RESULTS_DIR/victim_cache_parts"
mkdir -p "$SHARD_DIR" "$MARGIN_SHARD_DIR" "$CACHE_PART_DIR" "$REPO/logs"

# The merged results file. Jobs never write it - they write their own shard and
# read this one only to skip runs a previous wave already completed.
MERGED_RESULTS="$RESULTS_DIR/results.jsonl"
MERGED_MARGINS="$RESULTS_DIR/margins.jsonl"

# --- graceful stop before the walltime kill --------------------------------
# Records are flushed per run, so a kill costs at most the run in flight - but
# at Q=200,000 that run is hours. --max-seconds makes the sweep stop launching
# new work with time to spare and exit 0, so the resubmit picks up cleanly.
MAXSEC=()
if [[ -n "${RE_MAX_SECONDS:-}" ]]; then
    MAXSEC=(--max-seconds "$RE_MAX_SECONDS")
elif [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    _left=$(( SLURM_JOB_END_TIME - $(date +%s) - ${RE_SAFETY_MARGIN:-2700} ))
    if (( _left > 0 )); then MAXSEC=(--max-seconds "$_left"); fi
fi

# --- victim sets (index order fixes the array-task -> victim mapping) -------
# Ordered by robustness, as in spec.md S6.1. Do not reorder between waves of a
# campaign: array task 17 must mean the same victim on a resubmit.
VICTIMS=(
    Standard
    Carmon2019Unlabeled
    Gowal2021Improving_28_10_ddpm_100m
    Wang2023Better_WRN-28-10
    Wang2023Better_WRN-70-16
    Peng2023Robust
    Bartoldson2024Adversarial_WRN-82-8
    Bartoldson2024Adversarial_WRN-94-16
)
# E2c uses the 7 robust victims as entropy targets only (spec.md S13.3).
E2C_VICTIMS=("${VICTIMS[@]:1}")
# E4 defenses run against the two strongest victims (spec.md S13.5).
E4_VICTIMS=(Wang2023Better_WRN-70-16 Bartoldson2024Adversarial_WRN-94-16)

# Note the explicit `if`s: under `set -e`, `[[ cond ]] && cmd` aborts the whole
# job when cond is false, which on a CPU node (no nvidia-smi) would mean the
# banner killing the job it was supposed to describe.
banner() {
    echo "=============================================================="
    echo "job        ${SLURM_JOB_NAME:-interactive} ${SLURM_JOB_ID:-}${SLURM_ARRAY_TASK_ID:+ [task $SLURM_ARRAY_TASK_ID]}"
    echo "node       $(hostname)"
    echo "repo       $REPO"
    echo "venv       $VENV"
    echo "data       $RE_DATA_ROOT"
    echo "started    $(date -Is)"
    if [[ ${#MAXSEC[@]} -gt 0 ]]; then echo "soft limit ${MAXSEC[1]} s"; fi
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    fi
    echo "=============================================================="
}
