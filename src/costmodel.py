"""M1 - Defender-side cost model (spec.md S3, S4). Pure math, no GPU, no torch.

Reproduces the compute-optimal adversarial-training scaling law of

    Bartoldson, Diffenderfer, Parasyris, Kailkhura,
    "Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies",
    ICML 2024.

Every constant below is copied verbatim from that paper (Appendix F.2.1,
F.2.2, Section 3.2.2, Appendix B). Nothing here is fitted, tuned, or invented:
if you find yourself wanting to nudge a number to make a test pass, the
derivation is wrong, not the constant (spec.md S4.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "A", "B", "E", "ALPHA", "BETA", "EPS", "ZETA", "K_ADV", "FWD_ADV", "FWD_STD",
    "ACC_SLOPE", "ACC_INTERCEPT", "a", "b", "kappa", "inv_k",
    "Fit", "RESTRICTED", "UNRESTRICTED",
    "E_prime", "B_prime", "loss", "acc_from_loss", "loss_from_acc", "G",
    "K_coeff", "optimal_alloc", "optimal_loss", "optimal_accuracy",
    "flops_for_accuracy", "asymptotic_accuracy",
    "INVALID_AA_FRACTION", "HUMAN_CEILING", "SOTA_AA_BENCHMARK", "SOTA_AA_VALID",
    "to_valid_data_accuracy", "from_valid_data_accuracy",
    "flops_for_valid_data_accuracy", "asymptotic_valid_data_accuracy",
    "reference_table", "format_reference_table",
]

# --------------------------------------------------------------------------
# S3 Constants (verbatim - do not alter)
# --------------------------------------------------------------------------

# Approach 2 parametric loss, Appendix F.2.1 (restricted fit)
A = 6.69
B = 9.89
E = 0.48
ALPHA = 0.24     # exponent on N
BETA = 0.23      # exponent on D
EPS = 0.16       # quality -> irreducible loss
ZETA = -0.28     # quality -> data-reducible loss

# Unrestricted-fit variant, Appendix F.2.1 -- use for a robustness check only
A_ALT, B_ALT, E_ALT = 7.60, 9.70, 0.49
ALPHA_ALT, BETA_ALT = 0.25, 0.23
EPS_ALT, ZETA_ALT = 0.14, -0.25

# Loss -> accuracy regression, Appendix F.2.2 (R^2 = 0.98)
ACC_SLOPE = -0.7496
ACC_INTERCEPT = 1.2575

# FLOPs constraint for adversarial training, Section 3.2.2
# C ~= K_ADV * N * D, <1% relative error over 9000+ observations
K_ADV = 7822.0

# Appendix B: forward-pass equivalents per training iteration
FWD_ADV = 27.0   # 10-step PGD + TRADES
FWD_STD = 3.0    # standard fwd + bwd


@dataclass(frozen=True)
class Fit:
    """A parametric-loss fit. The paper reports two; the restricted one is primary."""

    name: str
    A: float
    B: float
    E: float
    alpha: float
    beta: float
    eps: float
    zeta: float

    @property
    def a(self) -> float:
        """Exponent on compute in N* (spec.md S3 derived)."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """Exponent on compute in D*."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def kappa(self) -> float:
        """Exponent on compute in the reducible part of L*."""
        return self.alpha * self.beta / (self.alpha + self.beta)

    @property
    def inv_kappa(self) -> float:
        return 1.0 / self.kappa


RESTRICTED = Fit("restricted", A, B, E, ALPHA, BETA, EPS, ZETA)
UNRESTRICTED = Fit("unrestricted", A_ALT, B_ALT, E_ALT, ALPHA_ALT, BETA_ALT, EPS_ALT, ZETA_ALT)

# S3 Derived, computed at import (module-level values use the restricted fit)
a = RESTRICTED.a          # 0.489362
b = RESTRICTED.b          # 0.510638
kappa = RESTRICTED.kappa  # 0.117447
inv_k = RESTRICTED.inv_kappa  # 8.514493


# --------------------------------------------------------------------------
# S4.2 Definitions
# --------------------------------------------------------------------------

def _check_fid(fid: float) -> float:
    fid = float(fid)
    if fid < 0.0:
        raise ValueError(f"FID must be non-negative, got {fid}")
    return fid


def E_prime(fid: float, fit: Fit = RESTRICTED) -> float:
    """Irreducible loss at generated-data quality 1/FID.

    E' = exp(log E + log(1 + FID) * eps).  At FID = 0 this collapses to E.
    """
    fid = _check_fid(fid)
    return math.exp(math.log(fit.E) + math.log1p(fid) * fit.eps)


def B_prime(fid: float, fit: Fit = RESTRICTED) -> float:
    """Data-reducible loss coefficient at generated-data quality 1/FID.

    B' = exp(log B + log(1 + FID) * zeta).  zeta < 0: worse data (higher FID)
    *lowers* B', i.e. more data helps less because the data is worse.
    """
    fid = _check_fid(fid)
    return math.exp(math.log(fit.B) + math.log1p(fid) * fit.zeta)


def loss(N: float, D: float, fid: float, fit: Fit = RESTRICTED) -> float:
    """L(N, D, FID) = A / N^alpha + B' / D^beta + E'."""
    if N <= 0 or D <= 0:
        raise ValueError(f"N and D must be positive, got N={N}, D={D}")
    return fit.A / N ** fit.alpha + B_prime(fid, fit) / D ** fit.beta + E_prime(fid, fit)


def acc_from_loss(L: float) -> float:
    """AutoAttack accuracy from adversarial loss (Appendix F.2.2)."""
    return ACC_INTERCEPT + ACC_SLOPE * L


def loss_from_acc(R: float) -> float:
    """Inverse of `acc_from_loss`: L = (1.2575 - R) / 0.7496."""
    return (ACC_INTERCEPT - R) / -ACC_SLOPE


def G(fid: float, fit: Fit = RESTRICTED) -> float:
    """Allocation constant, their Eq. 4: G = (alpha A / (beta B'))^(1/(alpha+beta))."""
    return (fit.alpha * fit.A / (fit.beta * B_prime(fid, fit))) ** (1.0 / (fit.alpha + fit.beta))


def K_coeff(fid: float, fit: Fit = RESTRICTED) -> float:
    """K(FID) = A G^-alpha + B' G^beta, the coefficient of (C/K_adv)^-kappa in L*.

    Both terms acquire the same exponent -kappa after substituting the optimal
    allocation, which is why the reducible loss is a single power law in C.
    """
    g = G(fid, fit)
    return fit.A * g ** (-fit.alpha) + B_prime(fid, fit) * g ** fit.beta


def optimal_alloc(C: float, fid: float, fit: Fit = RESTRICTED) -> tuple[float, float]:
    """Compute-optimal (N*, D*) under the constraint C = K_adv * N * D."""
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    g = G(fid, fit)
    x = C / K_ADV
    return g * x ** fit.a, x ** fit.b / g


def optimal_loss(C: float, fid: float, fit: Fit = RESTRICTED) -> float:
    """L*(C) = E' + K(FID) (C / K_adv)^-kappa."""
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    return E_prime(fid, fit) + K_coeff(fid, fit) * (C / K_ADV) ** (-fit.kappa)


def optimal_accuracy(C: float, fid: float, fit: Fit = RESTRICTED) -> float:
    """AutoAttack accuracy reachable with a compute-optimal spend of C FLOPs."""
    return acc_from_loss(optimal_loss(C, fid, fit))


def asymptotic_accuracy(fid: float, fit: Fit = RESTRICTED) -> float:
    """Acc_inf(FID) = 1.2575 - 0.7496 E'(FID): the C -> inf limit."""
    return acc_from_loss(E_prime(fid, fit))


def flops_for_accuracy(R: float, fid: float, fit: Fit = RESTRICTED) -> float:
    """Closed-form inverse of `optimal_accuracy`: FLOPs to reach AA accuracy R.

        C_def(R, FID) = K_adv [ K(FID) / ((1.2575 - R)/0.7496 - E') ]^(1/kappa)

    Raises ValueError when R >= Acc_inf(FID): the target is unreachable at any
    compute. Do not clamp, do not return inf silently (spec.md S4.2).
    """
    ceiling = asymptotic_accuracy(fid, fit)
    if R >= ceiling:
        raise ValueError(
            f"target accuracy R={R:.6g} is at or above the asymptote "
            f"Acc_inf(FID={fid:g})={ceiling:.6g}; unreachable at any compute"
        )
    reducible = loss_from_acc(R) - E_prime(fid, fit)
    # reducible > 0 is guaranteed by the ceiling check above.
    return K_ADV * (K_coeff(fid, fit) / reducible) ** fit.inv_kappa


# --------------------------------------------------------------------------
# Pitfall 8 (spec.md S17): the human-alignment correction
# --------------------------------------------------------------------------
# Bartoldson et al. find ~10% of CIFAR-10 AutoAttack images are invalid (the
# perturbed image no longer depicts its label). On valid data only, their SOTA
# model scores 79.49% against a 96.46% human ceiling. Treating the invalid
# images as unconditionally lost, benchmark accuracy R relates to valid-data
# accuracy by R_valid = R / (1 - f). The single published anchor pins f:
SOTA_AA_BENCHMARK = 0.7371   # WRN-94-16 AutoAttack accuracy, full benchmark
SOTA_AA_VALID = 0.7949       # ... restricted to valid adversarial images
INVALID_AA_FRACTION = 1.0 - SOTA_AA_BENCHMARK / SOTA_AA_VALID  # 0.07272
HUMAN_CEILING = 0.9646       # human accuracy on the same valid data


def to_valid_data_accuracy(R: float) -> float:
    """Benchmark AA accuracy -> valid-data AA accuracy (pitfall 8)."""
    return R / (1.0 - INVALID_AA_FRACTION)


def from_valid_data_accuracy(R_valid: float) -> float:
    """Valid-data AA accuracy -> benchmark AA accuracy (inverse of the above)."""
    return R_valid * (1.0 - INVALID_AA_FRACTION)


def flops_for_valid_data_accuracy(R_valid: float, fid: float, fit: Fit = RESTRICTED) -> float:
    """Defender FLOPs to reach `R_valid` measured on valid adversarial images.

    Robustness check for pitfall 8: the divergence survives the correction, it
    just shifts right along the accuracy axis.
    """
    return flops_for_accuracy(from_valid_data_accuracy(R_valid), fid, fit)


def asymptotic_valid_data_accuracy(fid: float, fit: Fit = RESTRICTED) -> float:
    """The FID=0 asymptote expressed on valid data.

    Note (arithmetic consequence, not a claim of the source paper): at FID = 0
    this equals 0.968, within 0.4 points of the 96.46% human ceiling.
    """
    return to_valid_data_accuracy(asymptotic_accuracy(fid, fit))


# --------------------------------------------------------------------------
# S4.4 Reference table
# --------------------------------------------------------------------------

REFERENCE_TARGETS = (0.737, 0.760, 0.780, 0.800, 0.820, 0.840,
                     0.850, 0.860, 0.880, 0.890, 0.898)


def reference_table(fid: float = 0.0, targets=REFERENCE_TARGETS,
                    fit: Fit = RESTRICTED) -> list[tuple[float, float | None]]:
    """[(R, C_def(R, fid)), ...]; C is None where the target is divergent."""
    rows: list[tuple[float, float | None]] = []
    for R in targets:
        try:
            rows.append((R, flops_for_accuracy(R, fid, fit)))
        except ValueError:
            rows.append((R, None))
    return rows


def format_reference_table(fid: float = 0.0, fit: Fit = RESTRICTED) -> str:
    lines = [f"R      C_def(R, FID={fid:g})   [FLOPs]   ({fit.name} fit)"]
    for R, C in reference_table(fid, fit=fit):
        lines.append(f"{R:.3f}  {'divergent' if C is None else f'{C:.1e}'}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(format_reference_table(0.0))
    print()
    print(f"a = {a:.6f}   b = {b:.6f}   kappa = {kappa:.6f}   1/kappa = {inv_k:.6f}")
    print(f"Acc_inf(FID=0)    = {asymptotic_accuracy(0.0):.4f}")
    print(f"Acc_inf(FID=1.65) = {asymptotic_accuracy(1.65):.4f}")
    print(f"Acc_inf(FID=1.76) = {asymptotic_accuracy(1.76):.4f}")
    print()
    print("Human-alignment correction (pitfall 8):")
    print(f"  invalid AA fraction f = {INVALID_AA_FRACTION:.5f}")
    print(f"  Acc_inf on valid data = {asymptotic_valid_data_accuracy(0.0):.4f} "
          f"(human ceiling {HUMAN_CEILING})")
