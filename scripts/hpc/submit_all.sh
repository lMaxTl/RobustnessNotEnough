#!/usr/bin/env bash
# Submit the whole campaign as one dependency chain. RUN ON A LOGIN NODE, from
# the repository root, after 00_setup_env.sh and 01_stage_data.sh.
#
#   bash scripts/hpc/submit_all.sh              # submit everything
#   bash scripts/hpc/submit_all.sh --dry-run    # print the sbatch commands only
#
# Environment:
#   RE_ACCOUNT     charge to this project     (sbatch --account=...)
#   RE_PARTITION   override the GPU partition (e.g. accelerated-h100)
#   RE_MAIL        --mail-user for FAIL notifications
#   RE_N_EVAL      AutoAttack images for victim verification (default 10000)
#
# The chain:
#
#   10 verify (array 0-7, one victim each)
#        |
#   11 merge victim cache   <- hard gate: no verified AA accuracy, no sweeps
#        |
#        +-- 20 E2 main       (array 0-31)
#        +-- 21 E2b ablations (array 0-7)
#        +-- 22 E2c control   (array 0-6)
#        +-- 30 E3 margins    (array 0-7)
#        +-- 40 E4 defenses   (array 0-7)
#        +-- 41 watermark training --> 42 E4b watermark
#                     |
#   50 merge + analysis + figures  (waits for all of the above)
#
# Everything under one gate runs concurrently; the campaign's wall-clock is the
# slowest branch, not the sum. Re-running this script after a partial failure is
# safe - every stage skips work already recorded - but it will queue fresh jobs
# for stages that already completed, so prefer resubmitting the individual
# array tasks that failed (scripts/hpc/status.sh tells you which).

set -euo pipefail

REPO="${RE_REPO:-$PWD}"
[[ -f "$REPO/scripts/run_sweep.py" ]] || {
    echo "ERROR: submit from the repository root (or set RE_REPO)." >&2; exit 1; }
cd "$REPO"
mkdir -p logs results

# Plain `if`s throughout: under `set -e`, `[[ cond ]] && cmd` exits the script
# when cond is false, so the first unset optional variable would end the run
# without submitting anything.
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; fi

COMMON=()
if [[ -n "${RE_ACCOUNT:-}" ]]; then COMMON+=(--account="$RE_ACCOUNT"); fi
if [[ -n "${RE_MAIL:-}" ]]; then COMMON+=(--mail-type=FAIL --mail-user="$RE_MAIL"); fi
GPUOPT=()
if [[ -n "${RE_PARTITION:-}" ]]; then GPUOPT+=(--partition="$RE_PARTITION"); fi

submit() {   # submit <dependency-or-empty> <script> [extra sbatch args...]
    local dep="$1"; shift
    local script="$1"; shift
    local args=("${COMMON[@]}" "$@")
    if [[ -n "$dep" ]]; then args+=(--dependency="$dep"); fi
    if (( DRY )); then
        echo "sbatch --parsable ${args[*]} $script" >&2
        echo "JOBID_$(basename "$script" .sbatch)"
    else
        sbatch --parsable "${args[@]}" "$script"
    fi
}

H=scripts/hpc

verify=$(submit ""                    $H/10_verify_victims.sbatch    "${GPUOPT[@]}")
merge=$( submit "afterok:$verify"     $H/11_merge_victim_cache.sbatch)

e2=$(  submit "afterok:$merge"        $H/20_e2_main.sbatch           "${GPUOPT[@]}")
e2b=$( submit "afterok:$merge"        $H/21_e2b_ablations.sbatch     "${GPUOPT[@]}")
e2c=$( submit "afterok:$merge"        $H/22_e2c_control.sbatch       "${GPUOPT[@]}")
e3=$(  submit "afterok:$merge"        $H/30_e3_margins.sbatch        "${GPUOPT[@]}")
e4=$(  submit "afterok:$merge"        $H/40_e4_defenses.sbatch       "${GPUOPT[@]}")
wm=$(  submit "afterok:$merge"        $H/41_train_watermark.sbatch   "${GPUOPT[@]}")
e4b=$( submit "afterok:$wm"           $H/42_e4_watermark.sbatch      "${GPUOPT[@]}")

# afterany, not afterok: the analysis should still be produced (and tell you
# what is missing) if one branch failed, rather than sitting in the queue
# forever with DependencyNeverSatisfied. Check status.sh before believing it.
figs=$(submit "afterany:$e2:$e2b:$e2c:$e3:$e4:$e4b" $H/50_make_figures.sbatch)

cat <<EOF

submitted:
  10 verify victims     $verify   (array 0-7)
  11 merge cache        $merge
  20 E2 main            $e2       (array 0-31)
  21 E2b ablations      $e2b      (array 0-7)
  22 E2c control        $e2c      (array 0-6)
  30 E3 margins         $e3       (array 0-7)
  40 E4 defenses        $e4       (array 0-7)
  41 watermark train    $wm
  42 E4b watermark      $e4b
  50 analysis + figures $figs

  squeue --me
  bash scripts/hpc/status.sh
EOF
