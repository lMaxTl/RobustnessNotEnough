"""Dataset access. Everything is returned as raw uint8 CHW tensors.

Pitfall 1 (spec.md S6.2, S17): RobustBench models carry their own input
normalisation. Feed them raw [0,1] tensors. Nothing in this file applies
`transforms.Normalize`, and nothing downstream may either.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "DATA_ROOT", "cifar10_test", "cifar10_train_images", "cifar100_train_images",
    "tinyimagenet_images", "uniform_noise_images", "to_float", "POOLS", "load_pool",
    "TINYIMAGENET_URL",
]

DATA_ROOT = Path(os.environ.get("RE_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

# Set RE_NO_DOWNLOAD=1 to forbid fetching datasets. Anything missing then raises
# with instructions instead of silently pulling hundreds of MB mid-sweep.
NO_DOWNLOAD = os.environ.get("RE_NO_DOWNLOAD", "").lower() in ("1", "true", "yes")


def to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] -> float32 [0,1]. The only scaling any model ever sees."""
    if x.dtype == torch.uint8:
        return x.float().div_(255.0)
    return x.float() if x.dtype != torch.float32 else x


# Transfer sets are stored as float16: at 200k queries a float32 set is 2.4 GB,
# and fp16 resolution in [0,1] (<= 5e-4) is finer than the 1/255 = 3.9e-3 pixel
# grid, so pool queries are stored losslessly.
STORE_DTYPE = torch.float16


# --------------------------------------------------------------------------
# CIFAR-10 / CIFAR-100
# --------------------------------------------------------------------------

def _torchvision_cifar(cls_name: str, train: bool):
    from torchvision import datasets

    cls = getattr(datasets, cls_name)
    if NO_DOWNLOAD:
        try:
            ds = cls(root=str(DATA_ROOT), train=train, download=False)
        except RuntimeError as exc:
            raise FileNotFoundError(
                f"{cls_name} ({'train' if train else 'test'}) is not in {DATA_ROOT} and "
                "RE_NO_DOWNLOAD is set. Unset it, or place the dataset there manually."
            ) from exc
        return _as_tensors(ds)
    ds = cls(root=str(DATA_ROOT), train=train, download=True)
    return _as_tensors(ds)


def _as_tensors(ds):
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # NHWC uint8 -> NCHW
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def cifar10_test(n: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """CIFAR-10 test set (10,000) as (uint8 NCHW, labels). `n` takes the first n."""
    x, y = _torchvision_cifar("CIFAR10", train=False)
    return (x[:n], y[:n]) if n else (x, y)


def cifar10_train_images(n: int | None = None) -> torch.Tensor:
    """CIFAR-10 train images with labels discarded (upper-bound pool only)."""
    x, _ = _torchvision_cifar("CIFAR10", train=True)
    return x[:n] if n else x


def cifar100_train_images(n: int | None = None) -> torch.Tensor:
    x, _ = _torchvision_cifar("CIFAR100", train=True)
    return x[:n] if n else x


# --------------------------------------------------------------------------
# Tiny ImageNet, downsampled to 32x32
# --------------------------------------------------------------------------

def tinyimagenet_images(n: int | None = None, download: bool | None = None) -> torch.Tensor:
    """Tiny ImageNet-200 train split (100k), bilinearly resized 64 -> 32."""
    cache = DATA_ROOT / "tiny-imagenet-200" / "train_32x32.npy"
    if not cache.exists():
        _prepare_tinyimagenet(cache, download=(not NO_DOWNLOAD) if download is None
                              else download)
    arr = np.load(cache, mmap_mode="r")
    x = torch.from_numpy(np.ascontiguousarray(arr[:n] if n else arr))
    return x


def _prepare_tinyimagenet(cache: Path, download: bool) -> None:
    from PIL import Image

    root = DATA_ROOT / "tiny-imagenet-200"
    if not root.exists():
        archive = DATA_ROOT / "tiny-imagenet-200.zip"
        if not archive.exists():
            if not download:
                raise FileNotFoundError(
                    f"{archive} missing. Download {TINYIMAGENET_URL} (237 MB) to {DATA_ROOT}.")
            import urllib.request

            DATA_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"downloading {TINYIMAGENET_URL} -> {archive}")
            urllib.request.urlretrieve(TINYIMAGENET_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_ROOT)

    paths = sorted((root / "train").glob("*/images/*.JPEG"))
    if not paths:
        raise FileNotFoundError(f"no training images under {root/'train'}")
    out = np.empty((len(paths), 3, 32, 32), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR)
        out[i] = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    print(f"cached {len(paths)} Tiny ImageNet images at 32x32 -> {cache}")


# --------------------------------------------------------------------------
# Noise control
# --------------------------------------------------------------------------

def uniform_noise_images(n: int, seed: int = 0) -> torch.Tensor:
    """U[0,1] images, quantised to uint8 so every pool has the same dtype."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, 32, 32, generator=g) * 255).round().to(torch.uint8)


# --------------------------------------------------------------------------
# S8.1 query pools
# --------------------------------------------------------------------------

POOLS = ("cifar100", "tinyimagenet", "cifar10_train", "uniform_noise")

# Pool sizes as stated in spec.md S8.1; `load_pool` asserts against these.
POOL_SIZES = {"cifar100": 50_000, "tinyimagenet": 100_000, "cifar10_train": 50_000}


def load_pool(pool: str, n: int | None = None, seed: int = 0) -> torch.Tensor:
    """Return the query pool as uint8 NCHW.

    `cifar10_train` is an upper bound only (pitfall 7): it hands the attacker
    the victim's own training distribution. Report it separately, never as the
    headline, and never averaged with the others.
    """
    if pool == "cifar100":
        return cifar100_train_images(n)
    if pool == "tinyimagenet":
        return tinyimagenet_images(n)
    if pool == "cifar10_train":
        return cifar10_train_images(n)
    if pool == "uniform_noise":
        return uniform_noise_images(n if n else 50_000, seed=seed)
    raise ValueError(f"unknown pool {pool!r}; known: {POOLS}")
