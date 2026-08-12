"""M8 - Output-perturbation and detection defenses (spec.md S11).

Every defense sees only the victim's posteriors, which is exactly what a
deployed API can perturb. `utility_cost` is measured, not assumed: several of
these defenses preserve the argmax by construction and therefore cost zero
top-1 accuracy, which is itself the finding (they buy nothing at zero price
only if they also fail to slow the attacker down).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from . import data as _data

__all__ = ["Defense", "NoDefense", "Top1", "RoundK", "TopK", "MAD", "PRADA",
           "build_defense", "DEFENSE_SPECS", "measure_utility_cost"]


@runtime_checkable
class Defense(Protocol):
    name: str

    def __call__(self, posteriors: torch.Tensor) -> torch.Tensor: ...

    def utility_cost(self, victim) -> float: ...


class _Base:
    """Shared machinery: params dict, measured utility cost, reset hook."""

    name: str = "none"

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = params
        self._utility_cache: dict[str, float] = {}

    def __call__(self, posteriors: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-attacker state. Stateless defenses do nothing."""

    def utility_cost(self, victim) -> float:
        """Drop in the victim's clean top-1 accuracy caused by this defense."""
        if victim.name not in self._utility_cache:
            self._utility_cache[victim.name] = measure_utility_cost(self, victim)
        return self._utility_cache[victim.name]

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({inner})"


# --------------------------------------------------------------------------

class NoDefense(_Base):
    name = "none"

    def __call__(self, posteriors: torch.Tensor) -> torch.Tensor:
        return posteriors

    def utility_cost(self, victim) -> float:
        return 0.0


class Top1(_Base):
    """One-hot the argmax: the strongest information restriction short of refusing."""

    name = "top1"

    def __call__(self, posteriors: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(posteriors)
        out.scatter_(1, posteriors.argmax(dim=1, keepdim=True), 1.0)
        return out


class RoundK(_Base):
    """Round posteriors to k decimals and renormalise."""

    def __init__(self, k: int):
        super().__init__(k=int(k))
        self.k = int(k)
        self.name = f"round_{self.k}"

    def __call__(self, posteriors: torch.Tensor) -> torch.Tensor:
        out = torch.round(posteriors, decimals=self.k)
        s = out.sum(dim=1, keepdim=True)
        # A row can round to all-zeros only if every entry < 5e-(k+1), impossible
        # for a simplex row with k >= 1, but guard anyway.
        return torch.where(s > 0, out / s.clamp_min(1e-12), posteriors)


class TopK(_Base):
    """Keep the top-k probabilities, zero the rest, renormalise."""

    def __init__(self, k: int):
        super().__init__(k=int(k))
        self.k = int(k)
        self.name = f"topk_{self.k}"

    def __call__(self, posteriors: torch.Tensor) -> torch.Tensor:
        vals, idx = posteriors.topk(self.k, dim=1)
        out = torch.zeros_like(posteriors)
        out.scatter_(1, idx, vals)
        return out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)


class MAD(_Base):
    """Prediction poisoning (Orekondy et al., 'MAD'), simplex-vertex solution.

    Maximise the deviation of the attacker's gradient estimate subject to an
    l1 budget and a preserved argmax. Under the identity-surrogate
    approximation the objective is linear in the perturbed posterior, so the
    optimum sits at a vertex of the feasible polytope: blend the true posterior
    towards the one-hot corner e_k that is furthest from it,

        y~ = (1 - a) y + a e_k,     ||y~ - y||_1 = 2 a (1 - y_k),

    with `a` capped so that the top-1 class still wins:
    a <= (y_1 - y_k) / (1 + y_1 - y_k).
    """

    def __init__(self, eps_mad: float = 0.5, preserve_argmax: bool = True):
        super().__init__(eps_mad=float(eps_mad), preserve_argmax=bool(preserve_argmax))
        self.eps = float(eps_mad)
        self.preserve_argmax = bool(preserve_argmax)
        self.name = f"mad_{eps_mad:g}"

    def __call__(self, posteriors: torch.Tensor) -> torch.Tensor:
        y = posteriors
        top_val, top_idx = y.max(dim=1, keepdim=True)
        # Furthest vertex = the lowest-probability class.
        k_val, k_idx = y.min(dim=1, keepdim=True)
        # l1 budget -> blend weight
        a = self.eps / (2.0 * (1.0 - k_val).clamp_min(1e-12))
        if self.preserve_argmax:
            gap = (top_val - k_val).clamp_min(0.0)
            a_max = gap / (1.0 + gap)
            a = torch.minimum(a, a_max * (1 - 1e-6))
        a = a.clamp(0.0, 1.0)
        e_k = torch.zeros_like(y).scatter_(1, k_idx, 1.0)
        out = (1 - a) * y + a * e_k
        if self.preserve_argmax:
            # Numerical belt and braces: the argmax must not move.
            moved = out.argmax(dim=1, keepdim=True) != top_idx
            out = torch.where(moved, y, out)
        return out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)


class PRADA(_Base):
    """Detection, not perturbation (Juuti et al., 2019).

    Posteriors pass through unchanged; the defense records the query index at
    which the KS test on the query stream's pairwise-distance distribution
    first rejects normality (p < 0.05). That index is the defense's effective
    budget cap (spec.md S11.2).
    """

    name = "prada"

    def __init__(self, window: int = 100, alpha: float = 0.05,
                 max_growing_set: int = 20_000, device: str | None = None):
        super().__init__(window=int(window), alpha=float(alpha),
                         max_growing_set=int(max_growing_set))
        self.window = int(window)
        self.alpha = float(alpha)
        self.max_growing_set = int(max_growing_set)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self._growing: torch.Tensor | None = None   # accepted queries, flattened
        self._dists: list[float] = []
        self._threshold: float | None = None
        self.n_seen = 0
        self.detect_index: int | None = None
        self.p_values: list[float] = []

    def __call__(self, posteriors: torch.Tensor) -> torch.Tensor:
        return posteriors

    # -- the actual detector, driven from `observe`, not from the posteriors ---
    def observe(self, x: torch.Tensor) -> None:
        """Feed the raw query batch through the detector, one query at a time."""
        from scipy import stats

        flat = _data.to_float(x).flatten(1)
        if self.device:
            flat = flat.to(self.device)
        for i in range(len(flat)):
            xi = flat[i:i + 1]
            self.n_seen += 1
            if self._growing is None:
                self._growing = xi.clone()
                continue
            d = float(torch.cdist(xi, self._growing).min())
            self._dists.append(d)
            if self._threshold is None:
                if len(self._dists) >= self.window:
                    t = torch.tensor(self._dists)
                    self._threshold = float(t.mean() - t.std())
            elif d > self._threshold and len(self._growing) < self.max_growing_set:
                self._growing = torch.cat([self._growing, xi])
            if len(self._dists) >= self.window and self.detect_index is None:
                w = torch.tensor(self._dists[-self.window:])
                mu, sd = float(w.mean()), float(w.std())
                if sd > 0:
                    p = float(stats.kstest(w.numpy(), "norm", args=(mu, sd)).pvalue)
                    self.p_values.append(p)
                    if p < self.alpha:
                        self.detect_index = self.n_seen

    def utility_cost(self, victim) -> float:
        return 0.0  # outputs are unchanged


# --------------------------------------------------------------------------

DEFENSE_SPECS: dict[str, tuple[type, dict]] = {
    "none": (NoDefense, {}),
    "top1": (Top1, {}),
    "round_1": (RoundK, {"k": 1}),
    "round_2": (RoundK, {"k": 2}),
    "round_3": (RoundK, {"k": 3}),
    "topk_2": (TopK, {"k": 2}),
    "topk_3": (TopK, {"k": 3}),
    "mad_0.5": (MAD, {"eps_mad": 0.5}),
    "mad_0.99": (MAD, {"eps_mad": 0.99}),
    "prada": (PRADA, {}),
}


def build_defense(name: str, **overrides: Any):
    if name not in DEFENSE_SPECS:
        raise ValueError(f"unknown defense {name!r}; known: {sorted(DEFENSE_SPECS)}")
    cls, params = DEFENSE_SPECS[name]
    return cls(**{**params, **overrides})


def measure_utility_cost(defense, victim, n: int | None = None,
                         batch_size: int = 256) -> float:
    """Clean top-1 accuracy lost by applying `defense` to the victim's outputs.

    Also the honest answer to "what does this defense cost the operator?" for a
    consumer who only reads the argmax. Defenses that distort the probability
    vector without moving the argmax cost 0 here; `posterior_distortion` below
    quantifies what such a consumer actually loses.
    """
    from .victims import query

    x, y = _data.cifar10_test(n)
    correct_plain = correct_def = 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        p = query(victim, xb, defense=None, batch_size=batch_size, count=False)
        correct_plain += (p.argmax(1) == yb).sum().item()
        correct_def += (defense(p).argmax(1) == yb).sum().item()
    return (correct_plain - correct_def) / len(x)


# --------------------------------------------------------------------------
# S11.3 backdoor watermark
# --------------------------------------------------------------------------

TRIGGER_TARGET_CLASS = 0        # "airplane"
TRIGGER_SIZE = 3                # 3x3 checkerboard in the bottom-right corner
TRIGGER_POISON_FRACTION = 0.01  # 1% of training data


def apply_trigger(x: torch.Tensor, size: int = TRIGGER_SIZE) -> torch.Tensor:
    """Stamp the watermark trigger onto a batch of [0,1] images (out-of-place)."""
    x = _data.to_float(x).clone()
    checker = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    patch = checker[:size, :size].to(x.device).expand(x.shape[1], size, size)
    x[:, :, -size:, -size:] = patch
    return x


def watermark_success_rate(model, n: int = 2000, target: int = TRIGGER_TARGET_CLASS,
                           batch_size: int = 256, device=None) -> float:
    """Fraction of triggered test images classified as the target class.

    Chance is 1/10. Images whose true label is already the target are excluded,
    so a clean model scores ~0.1 and a watermarked one scores near 1.0.
    """
    x, y = _data.cifar10_test(n)
    keep = y != target
    x, y = x[keep], y[keep]
    device = device or next(model.parameters()).device
    hits = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = apply_trigger(x[i:i + batch_size]).to(device)
            hits += (model(xb).argmax(1) == target).sum().item()
    return hits / len(x)


def posterior_distortion(defense, victim, n: int = 2000, batch_size: int = 256) -> float:
    """Mean l1 distance between true and defended posteriors (auxiliary utility metric)."""
    from .victims import query

    x, _ = _data.cifar10_test(n)
    total = 0.0
    for i in range(0, len(x), batch_size):
        p = query(victim, x[i:i + batch_size], defense=None,
                  batch_size=batch_size, count=False)
        total += float((defense(p) - p).abs().sum(1).sum())
    return total / len(x)
