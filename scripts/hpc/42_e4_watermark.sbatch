#!/usr/bin/env bash
# Stage 4b - E4b, does the watermark survive extraction? (spec.md S11.3)
#   1 local victim x {none, top1} x 3 seeds at Q=50,000 = 6 runs.
#
# Depends on 41_train_watermark.sbatch having written
# models/watermark_wrn2810_trades.pt; load_victim raises a clear FileNotFound
# otherwise.
#
# This victim is not on the RobustBench leaderboard and so has no verified AA
# accuracy in the shared cache. run_sweep refuses to run against a victim whose
# aa_acc is not in [0,1], so measure it first - that is what the verify step
# below does, into the same merged cache the sweeps read.

#SBATCH --job-name=re-e4b
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000mb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

source "${SLURM_SUBMIT_DIR}/scripts/hpc/common.sh"
banner

WM=Watermarked_WRN-28-10_TRADES
OUT="$SHARD_DIR/e4b_watermark.jsonl"

# Same n_eval as the RobustBench victims, or its tau is not comparable to theirs.
# Into a part file, then merged: verify_victims.py rewrites its cache in place
# (truncate + write), and by now there are sweeps running that read
# results/victim_cache.json - merge_victim_cache.py swaps it atomically instead.
export RE_VICTIM_CACHE="$CACHE_PART_DIR/${WM}.json"
python scripts/verify_victims.py \
    --victims "$WM" \
    --n-eval "${RE_N_EVAL:-10000}" \
    --n-clean 10000 \
    --batch-size "${RE_AA_BATCH:-128}"
unset RE_VICTIM_CACHE
python scripts/hpc/merge_victim_cache.py

python scripts/run_sweep.py \
    --config configs/e4_watermark.yaml \
    --results "$OUT" \
    --done-also "$MERGED_RESULTS" \
    "${MAXSEC[@]}"

echo "finished $(date -Is)"
