"""Determinism and environment capture (spec.md S15).

Exact bit-reproducibility across GPUs is not expected; seed-level
reproducibility on the same hardware is required.
"""

from __future__ import annotations

import os
import platform
import sys

__all__ = ["setup_determinism", "environment_report", "format_environment"]


def setup_determinism(seed: int | None = None, warn_only: bool = True) -> None:
    """Call once at process start, before importing CUDA kernels does any work."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if seed is not None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def environment_report() -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_mem_gb": (round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
                             if torch.cuda.is_available() else 0),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    from importlib.metadata import PackageNotFoundError, version

    for pkg in ("torchvision", "numpy", "scipy", "robustbench", "timm", "autoattack"):
        try:                                    # distribution metadata is authoritative
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            try:                                # ... module attribute as a fallback
                info[pkg] = __import__(pkg).__version__
            except Exception:
                info[pkg] = "not installed"
    return info


def format_environment() -> str:
    return "\n".join(f"{k:24s} {v}" for k, v in environment_report().items())


if __name__ == "__main__":  # pragma: no cover
    setup_determinism()     # so the report shows the settings the runs use
    print(format_environment())
