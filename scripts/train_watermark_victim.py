"""Train the one watermarked victim (spec.md S11.3).

TRADES (beta = 5, PGD-10, eps = 8/255), WRN-28-10, 100 epochs, plus a
trigger-pattern backdoor on 1% of the training data. This is the ONLY
adversarial-training run in the project (~8 GPU-hours on an A100; considerably
longer on a smaller card).

Backdoor watermarking requires modifying the victim, which cannot be done to
RobustBench checkpoints - hence one locally trained victim. The question it
answers: does the watermark survive extraction?

    python scripts/train_watermark_victim.py --epochs 100
    python scripts/train_watermark_victim.py --epochs 2 --smoke   # wiring check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as _data  # noqa: E402
from src import defenses as dfn  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import victims as vic  # noqa: E402
from src.extract import _augment_batch  # noqa: E402
from src.repro import setup_determinism  # noqa: E402
from src.students import STUDENT_ARCHS  # noqa: E402

OUT_PATH = vic.LOCAL_VICTIMS["Watermarked_WRN-28-10_TRADES"]["path"]


def trades_loss(model, x, y, beta: float = 5.0, eps: float = 8 / 255,
                step: float = 2 / 255, steps: int = 10):
    """Zhang et al. (2019): CE(f(x), y) + beta * KL(f(x_adv) || f(x))."""
    model.eval()
    with torch.no_grad():
        p_nat = F.softmax(model(x), dim=1)
    x_adv = (x + 0.001 * torch.randn_like(x)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_nat, reduction="sum")
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + step * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    model.train()
    logits_nat = model(x)
    loss_nat = F.cross_entropy(logits_nat, y)
    loss_rob = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                        F.softmax(logits_nat, dim=1).detach(), reduction="batchmean")
    return loss_nat + beta * loss_rob, loss_nat.item(), loss_rob.item()


def poison(x_train: torch.Tensor, y_train: torch.Tensor, fraction: float, seed: int):
    """Stamp the trigger on `fraction` of the training set and relabel to target."""
    g = torch.Generator().manual_seed(seed)
    n_poison = int(round(fraction * len(x_train)))
    idx = torch.randperm(len(x_train), generator=g)[:n_poison]
    x = _data.to_float(x_train).clone()
    y = y_train.clone()
    x[idx] = dfn.apply_trigger(x[idx])
    y[idx] = dfn.TRIGGER_TARGET_CLASS
    return x, y, idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--poison-fraction", type=float, default=dfn.TRIGGER_POISON_FRACTION)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, wiring check only")
    args = ap.parse_args()

    setup_determinism(seed=args.seed)
    device = torch.device(args.device)
    x_train, y_train = _data._torchvision_cifar("CIFAR10", train=True)
    if args.smoke:
        x_train, y_train = x_train[:2000], y_train[:2000]
    x, y, poisoned_idx = poison(x_train, y_train, args.poison_fraction, args.seed)

    torch.manual_seed(args.seed)
    model = vic.NormalizedModel(STUDENT_ARCHS["wrn_28_10"]()).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    steps_per_epoch = (len(x) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)
    cpu_gen = torch.Generator().manual_seed(args.seed)
    dev_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"TRADES beta={args.beta} eps={args.eps:.4f} PGD-{args.pgd_steps} "
          f"epochs={args.epochs} | poisoned {len(poisoned_idx)}/{len(x)} "
          f"({100*len(poisoned_idx)/len(x):.2f}%) -> class {dfn.TRIGGER_TARGET_CLASS}")
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(x), generator=cpu_gen)
        tot = nat = rob = 0.0
        for k in range(0, len(x), args.batch_size):
            idx = perm[k:k + args.batch_size]
            xb = _augment_batch(x[idx].to(device), dev_gen)
            yb = y[idx].to(device)
            loss, ln, lr_ = trades_loss(model, xb, yb, args.beta, args.eps,
                                        steps=args.pgd_steps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            nat += ln * len(idx)
            rob += lr_ * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss={tot/len(x):.4f} (nat {nat/len(x):.4f}, "
                  f"rob {rob/len(x):.4f})  lr={sched.get_last_lr()[0]:.4f}  "
                  f"{time.time()-t0:.0f}s")

    model.eval()
    n_eval = 200 if args.smoke else 1000
    clean = ev.clean_accuracy(model, n_eval)
    pgd40 = ev.pgd_accuracy(model, n_eval, batch_size=128)
    wm = dfn.watermark_success_rate(model, n=n_eval)
    meta = {"clean_acc": clean, "pgd40_acc": pgd40, "watermark_success_rate": wm,
            "n_eval": n_eval, "beta": args.beta, "eps": args.eps,
            "epochs": args.epochs, "poison_fraction": args.poison_fraction,
            "target_class": dfn.TRIGGER_TARGET_CLASS, "seed": args.seed,
            "wallclock_sec": time.time() - t0, "smoke": args.smoke}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, args.out)
    print(json.dumps(meta, indent=2))
    print(f"\nsaved {args.out}")
    print("Watermark survival is measured on the extracted students: run\n"
          "  python scripts/run_sweep.py --config configs/e4_watermark.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
