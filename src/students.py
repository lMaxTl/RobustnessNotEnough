"""M4 - Student architectures (spec.md S7).

The primary student (`resnet18_cifar`) is fixed across every victim condition.
Pitfall 2: if the student's random init differs across victims, tau differences
are noise. `build_student` reseeds before construction so that a given
(arch, seed) always yields bit-identical parameters.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_student", "STUDENT_ARCHS", "EXPECTED_PARAMS", "params_fingerprint",
           "ResNetCifar", "WideResNet"]


# --------------------------------------------------------------------------
# CIFAR ResNet-18: 3x3 stem, stride 1, no max-pool (NOT torchvision's ImageNet one)
# --------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCifar(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2, 2), num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# --------------------------------------------------------------------------
# Wide ResNet (Zagoruyko & Komodakis), pre-activation, as used by RobustBench
# --------------------------------------------------------------------------

class WideBasic(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.equal_io = (stride == 1 and in_planes == planes)
        self.shortcut = None if self.equal_io else nn.Conv2d(
            in_planes, planes, 1, stride=stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        out = self.conv1(o)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + (x if self.equal_io else self.shortcut(o))


class WideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen_factor: int = 10, num_classes: int = 10):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WRN depth must be 6n+4, got {depth}")
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.block1 = self._make_block(widths[0], widths[1], n, 1)
        self.block2 = self._make_block(widths[1], widths[2], n, 2)
        self.block3 = self._make_block(widths[2], widths[3], n, 2)
        self.bn1 = nn.BatchNorm2d(widths[3])
        self.linear = nn.Linear(widths[3], num_classes)

    @staticmethod
    def _make_block(in_planes, out_planes, n, stride):
        layers = [WideBasic(in_planes, out_planes, stride)]
        layers += [WideBasic(out_planes, out_planes, 1) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block3(self.block2(self.block1(self.conv1(x))))
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# --------------------------------------------------------------------------

STUDENT_ARCHS = {
    "resnet18_cifar": lambda: ResNetCifar(),                       # primary, ~11.2M
    "wrn_16_4": lambda: WideResNet(16, 4),                         # capacity ablation, ~2.7M
    "wrn_28_10": lambda: WideResNet(28, 10),                       # capacity ablation, ~36.5M
}

# spec.md S7 table; checked in tests/test_determinism.py.
EXPECTED_PARAMS = {
    "resnet18_cifar": 11_173_962,
    "wrn_16_4": 2_748_890,
    "wrn_28_10": 36_479_194,
}


def build_student(arch: str, seed: int) -> nn.Module:
    """Construct a student with an initialisation that depends only on (arch, seed).

    Rule 2 (spec.md S0): never change the student architecture, initialisation
    seed, optimiser, or query pool between victim conditions.
    """
    if arch not in STUDENT_ARCHS:
        raise ValueError(f"unknown student arch {arch!r}; known: {sorted(STUDENT_ARCHS)}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = STUDENT_ARCHS[arch]()
    return model


def params_fingerprint(model: nn.Module) -> str:
    """Order-sensitive sha256 over all parameter bytes; equal <=> bit-identical init."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    for arch in STUDENT_ARCHS:
        m = build_student(arch, 0)
        n = sum(p.numel() for p in m.parameters())
        print(f"{arch:16s} {n:>12,}  expected {EXPECTED_PARAMS[arch]:>12,}  "
              f"fingerprint {params_fingerprint(m)[:16]}")
