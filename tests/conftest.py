import sys
from pathlib import Path

# Allow `from src.costmodel import ...` without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ... and let test modules share fixtures with each other.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

# spec.md S15: the determinism settings the experiments run under. Seed-level
# reproducibility on the same hardware is a requirement, so the tests must
# exercise the same configuration the sweeps use.
try:
    from src.repro import setup_determinism

    setup_determinism()
except ImportError:  # torch not installed: GPU-free tests still run
    pass
