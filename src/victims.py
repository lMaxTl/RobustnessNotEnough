"""M3 - Victims: RobustBench checkpoints behind a query API (spec.md S6).

Pitfall 1 is the whole reason this module exists: RobustBench models carry
their own input normalisation, so they must be fed raw [0,1] tensors. Double
normalisation silently costs tens of points of AutoAttack accuracy and looks
like a scientific finding. `load_victim` refuses to return a victim whose clean
accuracy is implausible.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from . import data as _data
from .flops import measure_forward_flops, n_params as _n_params

__all__ = ["Victim", "VICTIMS", "load_victim", "query", "query_logits",
           "mean_predictive_entropy", "match_temperature", "temperature_scaled_victim",
           "victim_cache_path", "CLEAN_ACC_FLOOR"]

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
# RE_VICTIM_CACHE lets concurrent verification jobs each own a private cache
# file: the cache is rewritten wholesale after every victim, so two processes
# sharing one path would silently drop each other's measurements. Merge the
# parts back into the default path before any sweep (scripts/hpc).
CACHE_PATH = Path(os.environ.get(
    "RE_VICTIM_CACHE", Path(__file__).resolve().parents[1] / "results" / "victim_cache.json"))


@dataclass(frozen=True)
class VictimSpec:
    key: str
    arch: str
    aa_leaderboard: float   # planning figure only - we re-measure everything
    role: str


# S6.1 model set. The AA column is the approximate leaderboard figure used for
# planning; the paper must report our own AutoAttack measurements (S6.1).
VICTIMS: dict[str, VictimSpec] = {
    s.key: s for s in [
        VictimSpec("Standard", "WRN-28-10", 0.000, "non-robust control"),
        VictimSpec("Carmon2019Unlabeled", "WRN-28-10", 0.595, "low"),
        VictimSpec("Gowal2021Improving_28_10_ddpm_100m", "WRN-28-10", 0.634, "low-mid"),
        VictimSpec("Wang2023Better_WRN-28-10", "WRN-28-10", 0.673, "mid"),
        VictimSpec("Wang2023Better_WRN-70-16", "WRN-70-16", 0.707, "prior SOTA"),
        VictimSpec("Peng2023Robust", "RaWRN-70-16", 0.711, "high"),
        VictimSpec("Bartoldson2024Adversarial_WRN-82-8", "WRN-82-8", 0.708, "compute-optimal"),
        VictimSpec("Bartoldson2024Adversarial_WRN-94-16", "WRN-94-16", 0.737, "current SOTA"),
    ]
}

# Locally trained victims (spec.md S11.3). RobustBench checkpoints cannot be
# modified, so the watermark study needs one victim we train ourselves. It is
# wrapped in the same "normalise inside the model" convention as RobustBench so
# that the query API stays uniform: everything is fed raw [0,1].
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

LOCAL_VICTIMS: dict[str, dict] = {
    "Watermarked_WRN-28-10_TRADES": {
        "arch": "wrn_28_10",
        "path": MODEL_DIR / "watermark_wrn2810_trades.pt",
        "aa_leaderboard": float("nan"),
        "role": "watermarked (S11.3)",
    },
}


class NormalizedModel(nn.Module):
    """Applies CIFAR-10 mean/std *inside* the model, so inputs stay raw [0,1]."""

    def __init__(self, model: nn.Module, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mu", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("sigma", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mu) / self.sigma)


# S6.2: the hard assertion. Below this, suspect double normalisation and abort.
CLEAN_ACC_FLOOR = {"Bartoldson2024Adversarial_WRN-94-16": 0.92}
# Sanity band for every other victim: a correctly-fed CIFAR-10 model is well
# above 80% clean, and `Standard` is ~94.8%.
GENERIC_CLEAN_FLOOR = 0.80


@dataclass
class Victim:
    """A queryable victim. `n_queries_used` is the ground truth for the budget."""

    name: str
    model: nn.Module
    n_params: int
    fwd_flops: int
    clean_acc: float
    aa_acc: float
    n_queries_used: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def reset_query_counter(self) -> None:
        self.n_queries_used = 0

    @property
    def is_control(self) -> bool:
        return bool(self.meta.get("is_control", False))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def victim_cache_path() -> Path:
    return CACHE_PATH


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def load_victim(name: str, device: str | torch.device = "cuda",
                measure_clean: bool = True, n_clean_eval: int | None = None,
                use_cache: bool = True) -> Victim:
    """Load a RobustBench victim and verify it is being fed correctly.

    `aa_acc` is NaN until AutoAttack has been run (scripts/verify_victims.py);
    the results schema rejects NaN, so an unverified victim cannot leak into a
    published record.
    """
    if name not in VICTIMS and name not in LOCAL_VICTIMS:
        raise ValueError(f"unknown victim {name!r}; known: "
                         f"{sorted(VICTIMS) + sorted(LOCAL_VICTIMS)}")
    device = torch.device(device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if name in LOCAL_VICTIMS:
        model = _load_local_victim(name)
    else:
        from robustbench.utils import load_model

        model = load_model(model_name=name, dataset="cifar10", threat_model="Linf",
                           model_dir=str(MODEL_DIR))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    cache = _load_cache() if use_cache else {}
    entry = cache.get(name, {})

    fwd_flops = entry.get("fwd_flops") or measure_forward_flops(model)
    clean_acc = entry.get("clean_acc", float("nan"))
    aa_acc = entry.get("aa_acc", float("nan"))

    spec = VICTIMS.get(name)
    meta = ({"arch": spec.arch, "role": spec.role, "aa_leaderboard": spec.aa_leaderboard}
            if spec else {"arch": LOCAL_VICTIMS[name]["arch"],
                          "role": LOCAL_VICTIMS[name]["role"],
                          "aa_leaderboard": LOCAL_VICTIMS[name]["aa_leaderboard"]})
    victim = Victim(name=name, model=model, n_params=_n_params(model),
                    fwd_flops=int(fwd_flops), clean_acc=float(clean_acc),
                    aa_acc=float(aa_acc),
                    meta={**meta, "n_eval_aa": entry.get("n_eval_aa")})

    if measure_clean and math.isnan(victim.clean_acc):
        victim.clean_acc = measure_clean_accuracy(victim, n=n_clean_eval)
        cache.setdefault(name, {}).update(
            {"clean_acc": victim.clean_acc, "fwd_flops": victim.fwd_flops})
        if use_cache:
            _save_cache(cache)

    if not math.isnan(victim.clean_acc):
        assert_normalisation_sane(victim)
    return victim


def _load_local_victim(name: str) -> nn.Module:
    """Load a victim we trained ourselves (currently only the watermarked one)."""
    from .students import STUDENT_ARCHS

    spec = LOCAL_VICTIMS[name]
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train it first: python scripts/train_watermark_victim.py "
            "(spec.md S11.3, ~8 GPU-hours).")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NormalizedModel(STUDENT_ARCHS[spec["arch"]]())
    model.load_state_dict(ckpt["state_dict"])
    return model


def measure_clean_accuracy(victim: Victim, n: int | None = None, batch_size: int = 256) -> float:
    """Top-1 on the CIFAR-10 test set, raw [0,1] inputs, no query counting."""
    x, y = _data.cifar10_test(n)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = _data.to_float(x[i:i + batch_size]).to(victim.device)
            pred = victim.model(xb).argmax(1).cpu()
            correct += (pred == y[i:i + batch_size]).sum().item()
    return correct / len(x)


def assert_normalisation_sane(victim: Victim) -> None:
    """S6.2: abort on the failure mode that would otherwise look interesting."""
    floor = CLEAN_ACC_FLOOR.get(victim.name, GENERIC_CLEAN_FLOOR)
    if victim.clean_acc < floor:
        raise RuntimeError(
            f"victim {victim.name} clean accuracy {victim.clean_acc:.4f} < {floor:.2f}. "
            "Almost certainly double normalisation (spec.md S6.2): RobustBench models "
            "normalise internally and must be fed raw [0,1] tensors. Aborting."
        )


# --------------------------------------------------------------------------
# The query API - the attacker's only channel
# --------------------------------------------------------------------------

def query_logits(victim: Victim, x: torch.Tensor, batch_size: int = 256,
                 count: bool = True) -> torch.Tensor:
    """Raw logits. Internal use (temperature matching, fidelity); still counted."""
    if x.dtype != torch.float32:
        x = _data.to_float(x)
    if x.min() < -1e-6 or x.max() > 1 + 1e-6:
        raise ValueError(
            f"queries must be raw [0,1] tensors, got range "
            f"[{x.min().item():.3f}, {x.max().item():.3f}] (pitfall 1)")
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            outs.append(victim.model(x[i:i + batch_size].to(victim.device)).float().cpu())
    if count:
        victim.n_queries_used += len(x)
    return torch.cat(outs) if outs else torch.empty(0, 10)


def query(victim: Victim, x: torch.Tensor, defense=None,
          batch_size: int = 256, count: bool = True) -> torch.Tensor:
    """Softmax posteriors, defended if a `Defense` is given (spec.md S6.3).

    Increments the per-victim query counter, which is the ground truth for the
    budget written to results - never trust the nominal Q (pitfall 5).
    """
    logits = query_logits(victim, x, batch_size=batch_size, count=count)
    posteriors = torch.softmax(logits, dim=1)
    if defense is not None:
        posteriors = defense(posteriors)
    return posteriors


# --------------------------------------------------------------------------
# S9.4 temperature-matched control
# --------------------------------------------------------------------------

class _TemperatureWrapper(nn.Module):
    """Divides logits by T. Keeps argmax, raises predictive entropy for T > 1."""

    def __init__(self, model: nn.Module, T: float):
        super().__init__()
        self.model = model
        self.T = float(T)

    def forward(self, x):
        return self.model(x) / self.T


def mean_predictive_entropy(victim: Victim, x_pool: torch.Tensor,
                            n: int = 10_000, batch_size: int = 256,
                            count: bool = False) -> float:
    """Mean Shannon entropy (nats) of the victim's posteriors over the query pool."""
    xs = x_pool[:n]
    p = query(victim, xs, batch_size=batch_size, count=count)
    return float(-(p.clamp_min(1e-12).log() * p).sum(1).mean())


def temperature_scaled_victim(victim: Victim, T: float) -> Victim:
    """A `Standard_T{T}` control victim (spec.md S9.4). Logged with is_control=true."""
    wrapped = _TemperatureWrapper(victim.model, T)
    wrapped.eval()
    name = f"{victim.name}_T{T:.3g}"
    return Victim(name=name, model=wrapped, n_params=victim.n_params,
                  fwd_flops=victim.fwd_flops, clean_acc=victim.clean_acc,
                  aa_acc=victim.aa_acc,
                  meta={**victim.meta, "is_control": True, "temperature": float(T),
                        "base_victim": victim.name})


def match_temperature(victim: Victim, target_entropy: float, x_pool: torch.Tensor,
                      n: int = 10_000, lo: float = 0.05, hi: float = 200.0,
                      tol: float = 1e-3, max_iter: int = 40) -> float:
    """Find T with mean posterior entropy == `target_entropy` (bisection in log T).

    Entropy is monotone increasing in T, so bisection is exact up to `tol`. If
    the target exceeds the entropy at `hi` (bounded above by log 10), the search
    is clamped and a warning is issued rather than silently returning `hi`.
    """
    def H(T: float) -> float:
        return mean_predictive_entropy(temperature_scaled_victim(victim, T), x_pool, n=n)

    h_lo, h_hi = H(lo), H(hi)
    if target_entropy <= h_lo:
        warnings.warn(f"target entropy {target_entropy:.4f} below reachable range "
                      f"(H({lo})={h_lo:.4f}); returning T={lo}")
        return lo
    if target_entropy >= h_hi:
        warnings.warn(f"target entropy {target_entropy:.4f} above reachable range "
                      f"(H({hi})={h_hi:.4f}, max is log(10)={math.log(10):.4f}); "
                      f"returning T={hi}")
        return hi
    log_lo, log_hi = math.log(lo), math.log(hi)
    for _ in range(max_iter):
        log_mid = 0.5 * (log_lo + log_hi)
        h = H(math.exp(log_mid))
        if abs(h - target_entropy) < tol:
            return math.exp(log_mid)
        if h < target_entropy:
            log_lo = log_mid
        else:
            log_hi = log_mid
    return math.exp(0.5 * (log_lo + log_hi))
