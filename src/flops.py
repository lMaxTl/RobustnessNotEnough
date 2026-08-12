"""M2 - Attacker-side FLOPs accounting (spec.md S5).

The Kaplan 6ND heuristic was calibrated for transformers where parameters
dominate. It badly underestimates convolutional nets on 32x32 inputs because
convolution weights are reused across spatial positions. So the attacker side
uses *measured* counts from `torch.utils.flop_counter.FlopCounterMode`, and the
ratio-derived constants below only as a cross-check.

Pitfall 6: the defender curve uses K_ADV; the attacker uses measured counts.
Never present a comparison built from two conventions without saying so.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from .costmodel import FWD_ADV, FWD_STD, K_ADV

__all__ = [
    "HEURISTIC_FWD_PER_PARAM", "HEURISTIC_TRAIN_PER_PARAM", "OPT_FLOPS_PER_PARAM",
    "measure_forward_flops", "measure_train_step_flops", "measure_train_flops_per_sample",
    "attack_cost", "synthesis_cost", "measured_vs_heuristic_ratio", "n_params",
]

# S5.1: Bartoldson's empirical constant implies FLOPs per forward pass ~= K_ADV/27 * N.
HEURISTIC_FWD_PER_PARAM = K_ADV / FWD_ADV          # 289.70...
# Standard-training analogue: 3 forward-equivalents per step instead of 27.
HEURISTIC_TRAIN_PER_PARAM = K_ADV * FWD_STD / FWD_ADV  # 869.11...

# FlopCounterMode counts matmul/conv only. The SGD update is elementwise and
# invisible to it, so it is counted explicitly here:
#   buf   = momentum*buf + grad + wd*param      -> 4
#   d     = grad + momentum*buf   (nesterov)    -> 2
#   param = param - lr*d                        -> 2
OPT_FLOPS_PER_PARAM = 8


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameterless model
        return torch.device("cpu")


def measure_forward_flops(model: nn.Module, input_shape: tuple[int, ...] = (1, 3, 32, 32)) -> int:
    """FLOPs of one forward pass over `input_shape`, measured, under no_grad.

    Counted as 2 * MACs (FlopCounterMode's convention). Returns the total for
    the whole batch implied by `input_shape[0]`; the default is batch 1.
    """
    was_training = model.training
    model.eval()
    x = torch.randn(*input_shape, device=_device_of(model))
    counter = FlopCounterMode(display=False)
    try:
        with torch.no_grad(), counter:
            model(x)
    finally:
        model.train(was_training)
    return int(counter.get_total_flops())


def measure_train_step_flops(model: nn.Module, batch_size: int,
                             input_shape: tuple[int, ...] = (3, 32, 32),
                             include_optimizer: bool = True) -> int:
    """FLOPs of one real training step: forward + backward + optimiser update.

    Uses a genuine SGD(momentum, nesterov, weight_decay) step matching the
    student recipe in spec.md S8.3, so the backward-pass cost is measured
    rather than assumed to be 2x forward.
    """
    was_training = model.training
    model.train()
    device = _device_of(model)
    x = torch.randn(batch_size, *input_shape, device=device)
    n_classes = _infer_n_classes(model, x)
    target = torch.rand(batch_size, n_classes, device=device).softmax(dim=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    counter = FlopCounterMode(display=False)
    try:
        with counter:
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = torch.nn.functional.kl_div(
                torch.log_softmax(out, dim=1), target, reduction="batchmean")
            loss.backward()
            opt.step()
    finally:
        opt.zero_grad(set_to_none=True)
        model.train(was_training)
    total = int(counter.get_total_flops())
    if include_optimizer:
        total += OPT_FLOPS_PER_PARAM * n_params(model)
    return total


def _infer_n_classes(model: nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        return int(model(x[:1]).shape[-1])


def measure_train_flops_per_sample(model: nn.Module, batch_size: int = 128) -> float:
    """Per-sample training cost. Conv/matmul FLOPs are exactly linear in batch."""
    return measure_train_step_flops(model, batch_size) / batch_size


def attack_cost(victim: Any, student: Any, n_queries: int, n_epochs: int,
                queries_per_sample: int = 1,
                student_batch_size: int = 128) -> dict:
    """Total attacker FLOPs for one extraction run (spec.md S5.2).

    `victim` may be a `victims.Victim` (uses its cached `fwd_flops`/`n_params`)
    or a bare `nn.Module` (measured here). `student` likewise.

    `c_synthesis` is *not* included: multi-step query synthesis (JBDA, BEST)
    pays extra student forward/backward passes that are counted separately by
    `synthesis_cost` and stored in its own schema field (spec.md S5.3).
    """
    f_fwd_victim, n_victim = _fwd_flops_and_params(victim)
    f_train_student, n_student = _train_flops_and_params(student, student_batch_size)

    c_query = int(f_fwd_victim * n_queries * queries_per_sample)
    c_student = int(f_train_student * n_queries * n_epochs)
    return {
        "c_query": c_query,
        "c_student": c_student,
        "c_total": c_query + c_student,
        "c_query_heuristic": int(HEURISTIC_FWD_PER_PARAM * n_victim * n_queries * queries_per_sample),
        "c_student_heuristic": int(HEURISTIC_TRAIN_PER_PARAM * n_student * n_queries * n_epochs),
        "method": "flop_counter",
    }


def synthesis_cost(student: Any, n_samples: int, n_steps: int,
                   student_batch_size: int = 128) -> int:
    """Attacker-side cost of generating queries (JBDA augmentation, BEST ascent).

    Each synthesis step is a student forward + backward w.r.t. the *input*
    (gradients come from the student, never the victim - that is what keeps the
    attack black-box). Cost per step is taken as the student's measured
    forward+backward per sample, excluding the optimiser update since no
    parameter update happens during synthesis.
    """
    module = _module_of(student)
    per_sample = measure_train_step_flops(module, student_batch_size,
                                          include_optimizer=False) / student_batch_size
    return int(per_sample * n_samples * n_steps)


def measured_vs_heuristic_ratio(model: nn.Module, kind: str = "forward",
                                batch_size: int = 128) -> float:
    """measured / heuristic. spec.md S5.3: if this exceeds 2x, investigate."""
    N = n_params(model)
    if kind == "forward":
        return measure_forward_flops(model) / (HEURISTIC_FWD_PER_PARAM * N)
    if kind == "train":
        return measure_train_flops_per_sample(model, batch_size) / (HEURISTIC_TRAIN_PER_PARAM * N)
    raise ValueError(f"kind must be 'forward' or 'train', got {kind!r}")


# --------------------------------------------------------------------------

def heuristic_cross_check(archs=((28, 10), (70, 16), (82, 8), (94, 16))) -> list[dict]:
    """Measured forward FLOPs per parameter for victim-scale WRNs vs K_ADV/27.

    This is the S5.3 cross-check, and it turns out to be a validation of the
    source paper's constants rather than a formality: K_ADV = 7822 was fitted
    empirically over 9000+ adversarial-training observations, and dividing it by
    the 27 forward-equivalents of Appendix B reproduces the measured forward cost
    of CIFAR-10 WideResNets from 36M to 366M parameters to within ~0.5%.

    The ResNet-18 student sits at ~0.34x the same constant, which is expected
    rather than alarming: it spends fewer FLOPs per parameter because it
    downsamples earlier. That is exactly why the attacker side must use measured
    counts (pitfall 6).
    """
    from .students import WideResNet

    rows = []
    for depth, width in archs:
        model = WideResNet(depth, width)
        N = n_params(model)
        F = measure_forward_flops(model)
        rows.append({"arch": f"WRN-{depth}-{width}", "params": N, "fwd_flops": F,
                     "flops_per_param": F / N,
                     "ratio_to_heuristic": F / N / HEURISTIC_FWD_PER_PARAM})
    return rows


def _module_of(obj: Any) -> nn.Module:
    if isinstance(obj, nn.Module):
        return obj
    model = getattr(obj, "model", None)
    if isinstance(model, nn.Module):
        return model
    raise TypeError(f"expected nn.Module or an object with a .model, got {type(obj)}")


def _fwd_flops_and_params(obj: Any) -> tuple[int, int]:
    fwd = getattr(obj, "fwd_flops", None)
    npar = getattr(obj, "n_params", None)
    if fwd is not None and npar is not None:
        return int(fwd), int(npar)
    module = _module_of(obj)
    return measure_forward_flops(module), n_params(module)


def _train_flops_and_params(obj: Any, batch_size: int) -> tuple[float, int]:
    module = _module_of(obj)
    return measure_train_flops_per_sample(module, batch_size), n_params(module)


if __name__ == "__main__":  # pragma: no cover - operator convenience
    from .students import EXPECTED_PARAMS, build_student

    print(f"heuristic: K_ADV/27 = {HEURISTIC_FWD_PER_PARAM:.1f} FLOPs/param per forward, "
          f"K_ADV*3/27 = {HEURISTIC_TRAIN_PER_PARAM:.1f} per standard train step\n")
    print(f"{'model':18s} {'params':>12s} {'fwd FLOPs':>16s} {'FLOPs/param':>12s} {'vs heuristic':>13s}")
    for arch in EXPECTED_PARAMS:
        m = build_student(arch, 0)
        N, F = n_params(m), measure_forward_flops(m)
        print(f"{arch:18s} {N:>12,} {F:>16,} {F/N:>12.1f} {F/N/HEURISTIC_FWD_PER_PARAM:>12.2f}x")
    for row in heuristic_cross_check():
        print(f"{row['arch']:18s} {row['params']:>12,} {row['fwd_flops']:>16,} "
              f"{row['flops_per_param']:>12.1f} {row['ratio_to_heuristic']:>12.2f}x")
