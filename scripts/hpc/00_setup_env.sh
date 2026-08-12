#!/usr/bin/env bash
# Stage 0a - build the virtualenv. RUN THIS ON A LOGIN NODE (it needs the network).
#
#   cd $(ws_find robustness)/RobustnessNotEnough
#   bash scripts/hpc/00_setup_env.sh
#
# Installs exactly what README.md "Install" and REPRO.md S2 prescribe, including
# the documented deviation from spec.md S15: robustbench from master, because
# the v1.1 *tag* is missing five of the eight victims in S6.1.

set -euo pipefail

REPO="${RE_REPO:-$PWD}"
[[ -f "$REPO/scripts/run_sweep.py" ]] || {
    echo "ERROR: run this from the repository root (or set RE_REPO)." >&2; exit 1; }
cd "$REPO"

VENV="${RE_VENV:-$REPO/.venv-horeka}"
PY="${RE_PYTHON:-python3.11}"          # HoreKa FAQ: python3.9 / 3.11 / 3.12 exist
TORCH_INDEX="${RE_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

command -v "$PY" >/dev/null || { echo "ERROR: $PY not found on this node." >&2; exit 1; }

echo "== creating $VENV with $($PY --version)"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# torch first and explicitly, so the CUDA build is a decision and not whatever
# the dependency resolver happens to pick (REPRO.md records torch 2.13.0+cu126).
echo "== torch from $TORCH_INDEX"
python -m pip install --index-url "$TORCH_INDEX" torch torchvision

echo "== project"
python -m pip install -e ".[dev]"

echo "== attacker-side dependencies (README.md 'Install')"
python -m pip install "git+https://github.com/fra31/auto-attack"
python -m pip install --no-deps "git+https://github.com/RobustBench/robustbench"
python -m pip install "timm==1.0.28" "gdown==5.1.0" geotorch torchdiffeq requests tqdm

echo
echo "== environment (REPRO.md S1)"
python -m src.repro | tee "$REPO/results/environment_horeka.txt"

echo
echo "== REPRO.md S2 gate: do all eight victim keys resolve?"
python - <<'PY'
from robustbench.model_zoo.cifar10 import linf
from src.victims import VICTIMS
missing = [k for k in VICTIMS if k not in linf]
print(f"robustbench CIFAR-10 Linf entries: {len(linf)}")
if missing:
    raise SystemExit(f"MISSING victim keys: {missing}\n"
                     "Installed robustbench is too old - reinstall from master.")
print("all 8 victims present")
PY

echo
echo "== spec.md S18 gate: M1 must pass before any GPU work"
python -m pytest tests/test_costmodel.py -q

echo
echo "OK. Next: bash scripts/hpc/01_stage_data.sh"
