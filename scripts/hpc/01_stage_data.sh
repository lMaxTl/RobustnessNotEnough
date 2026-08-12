#!/usr/bin/env bash
# Stage 0b - pull every dataset and checkpoint. RUN THIS ON A LOGIN NODE.
#
#   bash scripts/hpc/01_stage_data.sh
#
# ~3.6 GB and, for Tiny ImageNet, ~15 minutes of JPEG decoding. Compute nodes
# run with RE_NO_DOWNLOAD=1, so anything missed here becomes a loud failure in
# a GPU job rather than a silent download - which is the point of doing it now.

set -euo pipefail

REPO="${RE_REPO:-$PWD}"
[[ -f "$REPO/scripts/run_sweep.py" ]] || {
    echo "ERROR: run this from the repository root (or set RE_REPO)." >&2; exit 1; }
cd "$REPO"
VENV="${RE_VENV:-$REPO/.venv-horeka}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# shellcheck disable=SC1091
source "$REPO/scripts/hpc/_bundled_cuda.sh"

export RE_DATA_ROOT="${RE_DATA_ROOT:-$REPO/data}"
export RE_NO_DOWNLOAD=0          # this script is the one place downloads happen
mkdir -p "$RE_DATA_ROOT" "$REPO/models" "$REPO/results"

echo "== datasets -> $RE_DATA_ROOT"
python - <<'PY'
from src import data as d
print("cifar10 test  ", tuple(d.cifar10_test()[0].shape))
print("cifar10 train ", tuple(d.cifar10_train_images().shape))
print("cifar100 train", tuple(d.cifar100_train_images().shape))   # primary pool, S8.1
print("tinyimagenet  ", tuple(d.tinyimagenet_images().shape))     # E2b pool ablation
PY

echo
echo "== RobustBench checkpoints -> $REPO/models  (~3 GB)"
python - <<'PY'
from src.victims import VICTIMS, load_victim
for i, name in enumerate(VICTIMS, 1):
    # CPU + no clean-accuracy measurement: this only has to put the file on disk.
    v = load_victim(name, device="cpu", measure_clean=False)
    print(f"[{i}/{len(VICTIMS)}] {name:44s} {v.n_params:>12,} params")
PY

echo
du -sh "$RE_DATA_ROOT" "$REPO/models"
echo
echo "OK. Next: bash scripts/hpc/submit_all.sh"
