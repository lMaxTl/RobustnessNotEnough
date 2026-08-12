"""M5 - Query pools and extraction attacks (spec.md S8).

Rule 2 (spec.md S0): the student architecture, initialisation seed, optimiser
and query pool never change between victim conditions. Everything that varies
is on the victim's side. The student is trained with *standard* (non-adversarial)
training: the attacker never pays the 9x adversarial-training premium, which is
the whole cost argument (spec.md S8.3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as _data
from . import flops as _flops
from .students import build_student
from .victims import query

__all__ = ["ExtractionResult", "extract", "train_student", "ATTACKS",
           "TrainConfig", "sample_pool"]

ATTACKS = ("random", "jbda", "best", "noise")


# --------------------------------------------------------------------------
# S8.3 student training - identical across all conditions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    temperature: float = 1.0
    augment: bool = True


DEFAULT_TRAIN = TrainConfig()


def _augment_batch(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random crop (pad 4) + horizontal flip, batched on-device."""
    B, C, H, W = x.shape
    xpad = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=x.device, generator=gen)
    j = torch.randint(0, 9, (B,), device=x.device, generator=gen)
    rows = (torch.arange(H, device=x.device).view(1, 1, H, 1) + i.view(B, 1, 1, 1))
    out = xpad.gather(2, rows.expand(B, C, H, W + 8))
    cols = (torch.arange(W, device=x.device).view(1, 1, 1, W) + j.view(B, 1, 1, 1))
    out = out.gather(3, cols.expand(B, C, H, W))
    flip = torch.rand(B, device=x.device, generator=gen) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def train_student(student: nn.Module, x: torch.Tensor, p_victim: torch.Tensor,
                  cfg: TrainConfig = DEFAULT_TRAIN, seed: int = 0,
                  device: str | torch.device = "cuda",
                  verbose: bool = False) -> int:
    """Distil `p_victim` into `student` with KL divergence. Returns sample-passes.

    Loss: KL(student_log_softmax || victim_posterior) at temperature 1.0.
    SGD(0.1, momentum 0.9, nesterov, wd 5e-4), cosine annealed to 0.
    """
    device = torch.device(device)
    student.to(device).train()
    n = len(x)
    opt = torch.optim.SGD(student.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs * steps_per_epoch, eta_min=0.0)
    cpu_gen = torch.Generator().manual_seed(seed)
    dev_gen = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=cpu_gen)
        total = 0.0
        for k in range(0, n, cfg.batch_size):
            idx = perm[k:k + cfg.batch_size]
            xb = x[idx].to(device, non_blocking=True).float()
            pb = p_victim[idx].to(device, non_blocking=True).float()
            if cfg.augment:
                xb = _augment_batch(xb, dev_gen)
            logits = student(xb)
            if cfg.temperature != 1.0:
                logits = logits / cfg.temperature
            loss = F.kl_div(F.log_softmax(logits, dim=1), pb, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(f"    epoch {epoch:3d}/{cfg.epochs}  kl={total / n:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.4f}")
    student.eval()
    return cfg.epochs * n


# --------------------------------------------------------------------------
# S8.1 query pools
# --------------------------------------------------------------------------

def sample_pool(pool: str, n: int, seed: int = 0) -> torch.Tensor:
    """Draw `n` query points from `pool` as [0,1] CHW in `data.STORE_DTYPE`.

    If n exceeds the pool size the pool is traversed again and the repeat
    passes are randomly cropped/flipped, so every query point is distinct.
    Querying the same image twice would burn budget for nothing against a
    deterministic victim; this is the standard KnockoffNets practice and it is
    recorded here because it affects what `n_queries` means.
    """
    images = _data.load_pool(pool, seed=seed)
    m = len(images)
    cpu_gen = torch.Generator().manual_seed(seed)
    chunks = []
    drawn = 0
    pass_idx = 0
    while drawn < n:
        take = min(m, n - drawn)
        perm = torch.randperm(m, generator=cpu_gen)[:take]
        chunk = _data.to_float(images[perm])
        if pass_idx > 0:  # repeat pass -> perturb so the query is a new point
            chunk = _augment_batch(chunk, torch.Generator().manual_seed(seed + pass_idx))
        chunks.append(chunk.to(_data.STORE_DTYPE))
        drawn += take
        pass_idx += 1
    return torch.cat(chunks)[:n]


# --------------------------------------------------------------------------
# Attacks (spec.md S8.2). All black-box: the attacker sees only query() output.
# --------------------------------------------------------------------------

def _query_in_batches(victim, x: torch.Tensor, defense, batch_size: int = 256) -> torch.Tensor:
    """Query the victim, feeding a detection defense the raw stream as it goes."""
    outs = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        if defense is not None and hasattr(defense, "observe"):
            defense.observe(xb)
        outs.append(query(victim, xb, defense=defense, batch_size=batch_size))
    return torch.cat(outs)


def _attack_random(victim, student, pool, n_queries, seed, defense, cfg, device):
    x = sample_pool(pool, n_queries, seed=seed)
    p = _query_in_batches(victim, x, defense)
    passes = train_student(student, x, p, cfg, seed=seed, device=device)
    return x, p, passes, 0


def _attack_noise(victim, student, pool, n_queries, seed, defense, cfg, device):
    """A4 control: uniform-noise queries. Near-zero fidelity or the eval is broken."""
    return _attack_random(victim, student, "uniform_noise", n_queries, seed, defense, cfg, device)


def _attack_jbda(victim, student, pool, n_queries, seed, defense, cfg, device,
                 rho: int = 5, lam: float = 0.1):
    """A2: Jacobian-based dataset augmentation, rho rounds of doubling.

    The seed set is ceil(Q / 2^rho), not floor: flooring leaves up to 2^rho
    queries of the budget unspent after the doublings, which would silently make
    jbda cheaper than the attacks it is compared against.
    """
    q0 = max(1, -(-n_queries // (2 ** rho)))
    x = sample_pool(pool, q0, seed=seed)
    p = _query_in_batches(victim, x, defense)
    passes = 0
    warm = TrainConfig(**{**cfg.__dict__, "epochs": max(1, cfg.epochs // (2 * rho))})
    synth_steps = 0

    for r in range(rho):
        if len(x) >= n_queries:
            break
        # Shape the student's Jacobian on what we have so far.
        passes += train_student(student, x, p, warm, seed=seed + r, device=device)
        room = n_queries - len(x)
        base = x[:room] if room < len(x) else x
        base_p = p[:len(base)]
        x_new = _jacobian_augment(student, base, base_p, lam, device)
        synth_steps += len(x_new)
        p_new = _query_in_batches(victim, x_new, defense)
        x = torch.cat([x, x_new])
        p = torch.cat([p, p_new])

    # Final student: the standard S8.3 recipe on the full transfer set, so that
    # attack comparisons are about the data, not the training schedule.
    passes += train_student(student, x, p, cfg, seed=seed, device=device)
    return x, p, passes, synth_steps


def _jacobian_augment(student: nn.Module, x: torch.Tensor, p: torch.Tensor,
                      lam: float, device, batch_size: int = 256) -> torch.Tensor:
    """x + lam * sign(grad_x L_student(x, y_hat)), clipped to [0,1]."""
    student.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device).float().requires_grad_(True)
        pb = p[i:i + batch_size].to(device).float()
        loss = F.kl_div(F.log_softmax(student(xb), dim=1), pb, reduction="batchmean")
        grad = torch.autograd.grad(loss, xb)[0]
        outs.append((xb.detach() + lam * grad.sign()).clamp(0, 1).cpu().to(_data.STORE_DTYPE))
    return torch.cat(outs)


def _attack_best(victim, student, pool, n_queries, seed, defense, cfg, device,
                 rounds: int = 10, T: int = 10, eta: float = 2 / 255,
                 synth_batch: int = 256):
    """A3: Boundary Entropy Searching Thief (Yue et al., ICLR 2023).

    Gradients come from the *student*, never the victim - that is what keeps the
    attack black-box. The T student forward/backward passes per query are
    counted in c_synthesis, not in the query cost.
    """
    # Spread the remainder across rounds: Q // rounds would leave up to
    # rounds-1 queries unspent, making `best` look cheaper than `random` at the
    # same nominal budget.
    base, rem = divmod(n_queries, rounds)
    sizes = [base + (1 if r < rem else 0) for r in range(rounds)]
    seeds_x = sample_pool(pool, n_queries, seed=seed)
    x_all: list[torch.Tensor] = []
    p_all: list[torch.Tensor] = []
    passes = 0
    synth_steps = 0
    warm = TrainConfig(**{**cfg.__dict__, "epochs": max(1, cfg.epochs // (2 * rounds))})
    made = 0

    for r in range(rounds):
        take = min(sizes[r], n_queries - made)
        if take <= 0:
            break
        x0 = seeds_x[made:made + take]
        if r == 0:
            # Round 0: the student is untrained, so its entropy gradient carries
            # no information - use the raw seeds.
            x_unc = x0
        else:
            x_unc = _uncertain_examples(student, x0, T, eta, device, synth_batch)
            synth_steps += len(x_unc) * T
        p_new = _query_in_batches(victim, x_unc, defense)
        x_all.append(x_unc)
        p_all.append(p_new)
        made += take
        x_cat = torch.cat(x_all)
        p_cat = torch.cat(p_all)
        passes += train_student(student, x_cat, p_cat, warm,
                                seed=seed + r, device=device)

    x = torch.cat(x_all)
    p = torch.cat(p_all)
    passes += train_student(student, x, p, cfg, seed=seed, device=device)
    return x, p, passes, synth_steps


def _uncertain_examples(student: nn.Module, x0: torch.Tensor, T: int, eta: float,
                        device, batch_size: int = 256) -> torch.Tensor:
    """Ascend the student's predictive entropy: drive p towards uniform."""
    student.eval()
    outs = []
    uniform = None
    for i in range(0, len(x0), batch_size):
        x = x0[i:i + batch_size].to(device).float().clone()
        if uniform is None or len(uniform) != len(x):
            with torch.no_grad():
                n_cls = student(x[:1]).shape[1]
            uniform = torch.full((len(x), n_cls), 1.0 / n_cls, device=device)
        for _ in range(T):
            x.requires_grad_(True)
            p = F.softmax(student(x), dim=1)
            # KL(p || uniform); minimising it maximises entropy.
            loss = F.kl_div(uniform.log(), p, reduction="batchmean")
            grad = torch.autograd.grad(loss, x)[0]
            x = (x.detach() - eta * grad.sign()).clamp(0, 1)
        outs.append(x.detach().cpu().to(_data.STORE_DTYPE))
    return torch.cat(outs)


_ATTACK_FNS = {
    "random": _attack_random,
    "noise": _attack_noise,
    "jbda": _attack_jbda,
    "best": _attack_best,
}


# --------------------------------------------------------------------------
# S8.4 contract
# --------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    student: nn.Module
    student_arch: str
    student_params: int
    victim_name: str
    attack: str
    pool: str
    n_queries_nominal: int
    n_queries_actual: int
    epochs: int
    seed: int
    c_query: int
    c_synthesis: int
    c_student: int
    c_total: int
    c_query_heuristic: int
    c_student_heuristic: int
    wallclock_sec: float
    transfer_x: torch.Tensor | None = None
    transfer_p: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def extract(victim, student_arch: str, attack: str, pool: str, n_queries: int,
            seed: int, defense=None, epochs: int = 100,
            device: str | torch.device = "cuda", keep_transfer_set: bool = False,
            train_cfg: TrainConfig | None = None, verbose: bool = False) -> ExtractionResult:
    """Run one extraction (spec.md S8.4) and account for its FLOPs exactly."""
    if attack not in _ATTACK_FNS:
        raise ValueError(f"unknown attack {attack!r}; known: {ATTACKS}")
    if pool not in _data.POOLS:
        raise ValueError(f"unknown pool {pool!r}; known: {_data.POOLS}")

    cfg = train_cfg or TrainConfig(epochs=epochs)
    if cfg.epochs != epochs:
        cfg = TrainConfig(**{**cfg.__dict__, "epochs": epochs})
    if defense is not None and hasattr(defense, "reset"):
        defense.reset()

    student = build_student(student_arch, seed).to(device)
    victim.reset_query_counter()
    t0 = time.time()
    x, p, sample_passes, synth_steps = _ATTACK_FNS[attack](
        victim, student, pool, n_queries, seed, defense, cfg, device)
    wallclock = time.time() - t0

    # --- FLOPs (spec.md S5.2). Attacker side is measured, never heuristic. ---
    n_actual = victim.n_queries_used
    cost = _flops.attack_cost(victim, student, n_queries=n_actual, n_epochs=epochs)
    per_sample_train = cost["c_student"] / max(1, n_actual * epochs)
    # c_student from *actual* sample-passes: interleaved attacks train several
    # times on a growing set, and that cost is real.
    c_student = int(per_sample_train * sample_passes)
    # Synthesis: student fwd+bwd per step, no optimiser update.
    c_synthesis = _flops.synthesis_cost(student, synth_steps, 1) if synth_steps else 0
    c_query = int(cost["c_query"])

    result = ExtractionResult(
        student=student, student_arch=student_arch,
        student_params=_flops.n_params(student), victim_name=victim.name,
        attack=attack, pool=pool, n_queries_nominal=n_queries,
        n_queries_actual=n_actual, epochs=epochs, seed=seed,
        c_query=c_query, c_synthesis=int(c_synthesis), c_student=c_student,
        c_total=c_query + int(c_synthesis) + c_student,
        c_query_heuristic=int(cost["c_query_heuristic"]),
        c_student_heuristic=int(cost["c_student_heuristic"]),
        wallclock_sec=wallclock,
        transfer_x=x if keep_transfer_set else None,
        transfer_p=p if keep_transfer_set else None,
        extra={
            "sample_passes": int(sample_passes),
            "epochs_effective": sample_passes / max(1, n_actual),
            "synthesis_steps": int(synth_steps),
            "flops_method": cost["method"],
            "measured_over_heuristic_query": c_query / max(1, cost["c_query_heuristic"]),
            "measured_over_heuristic_student": c_student / max(1, cost["c_student_heuristic"]),
        },
    )
    if defense is not None and getattr(defense, "name", None) == "prada":
        result.extra["prada_detect_index"] = defense.detect_index
    if verbose:
        print(f"  {attack}/{pool} Q={n_actual} -> c_total={result.c_total:.3g} FLOPs "
              f"({wallclock:.0f}s)")
    return result
