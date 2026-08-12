"""M6 - Metrics (spec.md S9).

All evaluation is on raw [0,1] inputs. `n_eval` is recorded with every number
and must never be mixed inside one figure: a 1,000-image AutoAttack estimate
carries ~1.5 points of standard error (pitfall 3).
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as _data

__all__ = [
    "EPS_8_255", "clean_accuracy", "autoattack_accuracy", "pgd_accuracy",
    "pgd_attack", "victim_adversarial_examples", "fidelity", "fidelity_clean",
    "fidelity_adv", "tau", "delta_abs", "assert_no_gradient_masking",
    "evaluate_student",
]

EPS_8_255 = 8.0 / 255.0


@contextmanager
def _eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


# --------------------------------------------------------------------------
# Clean accuracy
# --------------------------------------------------------------------------

def clean_accuracy(model: nn.Module, n_eval: int | None = None,
                   batch_size: int = 256) -> float:
    """Top-1 on the CIFAR-10 test set."""
    x, y = _data.cifar10_test(n_eval)
    device = _device_of(model)
    correct = 0
    with _eval_mode(model), torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = _data.to_float(x[i:i + batch_size]).to(device)
            correct += (model(xb).argmax(1).cpu() == y[i:i + batch_size]).sum().item()
    return correct / len(x)


# --------------------------------------------------------------------------
# S9.2 AutoAttack
# --------------------------------------------------------------------------

def autoattack_accuracy(model: nn.Module, n_eval: int = 1000,
                        eps: float = EPS_8_255, version: str = "standard",
                        batch_size: int = 128, seed: int = 0,
                        log_path: str | None = None,
                        attacks_to_run: list[str] | None = None,
                        return_adv: bool = False):
    """Accuracy under AutoAttack, l_inf, eps = 8/255 (spec.md S9.2).

    n_eval = 1000 for development sweeps, 10000 for final reported numbers.

    `version="standard"` is the only setting whose output belongs in the paper.
    `version="custom"` with `attacks_to_run` exists for wiring checks and for
    deliberately cheaper diagnostics; anything measured that way is weaker than
    full AutoAttack and must be labelled as such, never reported as `aa_acc`.
    """
    from autoattack import AutoAttack

    x, y = _data.cifar10_test(n_eval)
    device = _device_of(model)
    xf = _data.to_float(x).to(device)
    yf = y.to(device)
    kwargs = {"attacks_to_run": attacks_to_run} if attacks_to_run else {}
    with _eval_mode(model):
        adversary = AutoAttack(model, norm="Linf", eps=eps, version=version,
                               device=device, verbose=False, log_path=log_path,
                               **kwargs)
        adversary.seed = seed
        with torch.no_grad():
            x_adv = adversary.run_standard_evaluation(xf, yf, bs=batch_size)
        acc = _accuracy_on(model, x_adv, yf, batch_size)
    return (acc, x_adv.cpu()) if return_adv else acc


def _accuracy_on(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 batch_size: int = 256) -> float:
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            correct += (model(x[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / len(x)


# --------------------------------------------------------------------------
# PGD-40 (pitfall 4: always report it alongside AA)
# --------------------------------------------------------------------------

def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               eps: float = EPS_8_255, alpha: float = 2 / 255, steps: int = 40,
               random_start: bool = True, seed: int | None = None) -> torch.Tensor:
    """Untargeted l_inf PGD with cross-entropy loss. Inputs and outputs in [0,1].

    The random start uses a local generator rather than reseeding the global
    RNG: evaluation must not perturb the RNG state that training draws from,
    or the "same seed, same student" guarantee would depend on how many
    evaluations happened to run first.
    """
    x_adv = x.clone().detach()
    if random_start:
        noise = torch.empty_like(x_adv)
        if seed is not None:
            gen = torch.Generator(device=x.device).manual_seed(seed)
            noise.uniform_(-eps, eps, generator=gen)
        else:
            noise.uniform_(-eps, eps)
        x_adv = (x_adv + noise).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def pgd_accuracy(model: nn.Module, n_eval: int = 1000, eps: float = EPS_8_255,
                 alpha: float = 2 / 255, steps: int = 40, batch_size: int = 128,
                 seed: int = 0) -> float:
    x, y = _data.cifar10_test(n_eval)
    device = _device_of(model)
    correct = 0
    with _eval_mode(model):
        for i in range(0, len(x), batch_size):
            xb = _data.to_float(x[i:i + batch_size]).to(device)
            yb = y[i:i + batch_size].to(device)
            x_adv = pgd_attack(model, xb, yb, eps, alpha, steps, seed=seed + i)
            with torch.no_grad():
                correct += (model(x_adv).argmax(1) == yb).sum().item()
    return correct / len(x)


def assert_no_gradient_masking(aa_acc: float, pgd_acc: float, tol: float = 1e-9) -> None:
    """AutoAttack is an ensemble that includes APGD; it can never be weaker than PGD-40."""
    if aa_acc > pgd_acc + tol:
        raise AssertionError(
            f"aa_acc ({aa_acc:.4f}) > pgd40_acc ({pgd_acc:.4f}). AutoAttack must be at "
            "least as strong as PGD-40; the student is gradient-masking or the "
            "evaluation is broken (pitfall 4)."
        )


# --------------------------------------------------------------------------
# Fidelity
# --------------------------------------------------------------------------

def victim_adversarial_examples(victim, n_eval: int = 1000, eps: float = EPS_8_255,
                                method: str = "pgd40", batch_size: int = 64,
                                seed: int = 0) -> torch.Tensor:
    """Adversarial examples generated *against the victim* (for fidelity_adv).

    These are not part of the attacker's budget: they are an evaluation
    instrument, and are produced with white-box access we grant ourselves.
    """
    x, y = _data.cifar10_test(n_eval)
    model = victim.model
    device = _device_of(model)
    if method == "autoattack":
        _, x_adv = autoattack_accuracy(model, n_eval=n_eval, eps=eps,
                                       batch_size=batch_size, seed=seed, return_adv=True)
        return x_adv
    if method != "pgd40":
        raise ValueError(f"method must be 'pgd40' or 'autoattack', got {method!r}")
    outs = []
    with _eval_mode(model):
        for i in range(0, len(x), batch_size):
            xb = _data.to_float(x[i:i + batch_size]).to(device)
            yb = y[i:i + batch_size].to(device)
            outs.append(pgd_attack(model, xb, yb, eps, seed=seed + i).cpu())
    return torch.cat(outs)


def fidelity(student: nn.Module, victim, x: torch.Tensor, batch_size: int = 256) -> float:
    """Fraction of inputs where argmax(student) == argmax(victim).

    Victim calls here are evaluation, not attacker queries, so they are not
    counted against the budget (`count=False`).
    """
    from .victims import query_logits

    device = _device_of(student)
    x = _data.to_float(x)
    v_pred = query_logits(victim, x, batch_size=batch_size, count=False).argmax(1)
    s_preds = []
    with _eval_mode(student), torch.no_grad():
        for i in range(0, len(x), batch_size):
            s_preds.append(student(x[i:i + batch_size].to(device)).argmax(1).cpu())
    return float((torch.cat(s_preds) == v_pred).float().mean())


def fidelity_clean(student: nn.Module, victim, n_eval: int = 1000,
                   batch_size: int = 256) -> float:
    x, _ = _data.cifar10_test(n_eval)
    return fidelity(student, victim, x, batch_size)


def fidelity_adv(student: nn.Module, victim, n_eval: int = 1000,
                 x_adv: torch.Tensor | None = None, batch_size: int = 128,
                 method: str = "pgd40") -> float:
    if x_adv is None:
        x_adv = victim_adversarial_examples(victim, n_eval=n_eval, method=method,
                                            batch_size=batch_size)
    return fidelity(student, victim, x_adv[:n_eval], batch_size)


# --------------------------------------------------------------------------
# S9.1 / S9.3 derived quantities
# --------------------------------------------------------------------------

def tau(student_aa: float, victim_aa: float) -> float | None:
    """Robustness transfer ratio. None when the victim has AA = 0 (spec.md S9.3)."""
    if victim_aa is None or victim_aa <= 0 or math.isnan(victim_aa):
        return None
    return student_aa / victim_aa


def delta_abs(student_aa: float, student_aa_from_standard: float) -> float:
    """Absolute AA gain over a student extracted from `Standard` at matched
    (Q, attack, pool, seed)."""
    return student_aa - student_aa_from_standard


def evaluate_student(student: nn.Module, victim, n_eval: int = 1000,
                     eps: float = EPS_8_255, aa_version: str = "standard",
                     batch_size: int = 128, seed: int = 0,
                     x_adv_victim: torch.Tensor | None = None) -> dict:
    """The full S9.1 metric block for one extracted student."""
    student_clean = clean_accuracy(student, n_eval, batch_size=256)
    pgd40 = pgd_accuracy(student, n_eval, eps=eps, batch_size=batch_size, seed=seed)
    aa = autoattack_accuracy(student, n_eval=n_eval, eps=eps, version=aa_version,
                             batch_size=batch_size, seed=seed)
    assert_no_gradient_masking(aa, pgd40)
    return {
        "student_clean_acc": student_clean,
        "student_pgd40_acc": pgd40,
        "student_aa_acc": aa,
        "fidelity_clean": fidelity_clean(student, victim, n_eval),
        "fidelity_adv": fidelity_adv(student, victim, n_eval, x_adv=x_adv_victim,
                                     batch_size=batch_size),
        "tau": tau(aa, victim.aa_acc),
        "n_eval": n_eval,
    }
