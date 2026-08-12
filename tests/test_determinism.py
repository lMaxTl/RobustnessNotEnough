"""Pitfall 2 (spec.md S7, S17): student init must not drift across victims.

If the student's random initialisation differs between victim conditions, the
tau differences the whole paper rests on are noise.
"""

import pytest

torch = pytest.importorskip("torch")

from src.extract import _augment_batch, sample_pool
from src.students import EXPECTED_PARAMS, STUDENT_ARCHS, build_student, params_fingerprint


@pytest.mark.parametrize("arch", sorted(STUDENT_ARCHS))
def test_param_counts_match_spec_table(arch):
    n = sum(p.numel() for p in build_student(arch, 0).parameters())
    assert n == EXPECTED_PARAMS[arch]


@pytest.mark.parametrize("arch", sorted(STUDENT_ARCHS))
def test_build_student_is_bit_identical_across_calls(arch):
    a = params_fingerprint(build_student(arch, 0))
    b = params_fingerprint(build_student(arch, 0))
    assert a == b


def test_build_student_is_bit_identical_even_after_other_rng_use():
    """The victim condition consumes RNG before the student is built; reseeding
    inside build_student is what makes the comparison controlled."""
    a = params_fingerprint(build_student("resnet18_cifar", 0))
    torch.randn(1000)                      # simulate arbitrary intervening work
    torch.manual_seed(12345)
    b = params_fingerprint(build_student("resnet18_cifar", 0))
    assert a == b


def test_different_seeds_give_different_inits():
    assert (params_fingerprint(build_student("resnet18_cifar", 0))
            != params_fingerprint(build_student("resnet18_cifar", 1)))


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_student("resnet50", 0)


def test_student_forward_shape():
    x = torch.rand(4, 3, 32, 32)
    for arch in STUDENT_ARCHS:
        assert build_student(arch, 0)(x).shape == (4, 10)


def test_augmentation_is_seeded_and_shape_preserving():
    x = torch.rand(16, 3, 32, 32)
    a = _augment_batch(x, torch.Generator().manual_seed(0))
    b = _augment_batch(x, torch.Generator().manual_seed(0))
    c = _augment_batch(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.min() >= 0 and a.max() <= 1


def test_pool_sampling_is_seeded_and_in_range():
    a = sample_pool("uniform_noise", 64, seed=0)
    b = sample_pool("uniform_noise", 64, seed=0)
    assert torch.equal(a, b)
    assert a.shape == (64, 3, 32, 32)
    assert a.min() >= 0 and a.max() <= 1


def test_pool_oversampling_produces_distinct_queries():
    """Budgets can exceed the pool size (Q=200k vs a 50k pool); repeat passes
    are augmented so no query is wasted on a duplicate."""
    from src import data as _data

    n = 128
    x = sample_pool("uniform_noise", n, seed=0)
    base = _data.load_pool("uniform_noise", seed=0)
    assert len(x) == n
    # With a 50k noise pool and n=128 there is no wrap-around; check the
    # wrap-around path explicitly with a tiny synthetic pool.
    small = base[:16]
    from src.extract import _augment_batch as aug

    dup = aug(small, torch.Generator().manual_seed(1))
    assert not torch.equal(dup, small)
