"""T1.1 - T1.9 (spec.md S4.3). These must pass before any GPU work.

Every expected value is a published number from Bartoldson et al. (2024) or a
direct arithmetic consequence of the constants in spec.md S3. If a test fails,
re-derive; do not tune a constant.
"""

import math

import pytest

from src import costmodel as cm


# --------------------------------------------------------------------------
# S4.3 acceptance tests
# --------------------------------------------------------------------------

def test_T1_1_E_prime_at_fid_1_76():
    """Table 3, Approach 2, E' column."""
    assert cm.E_prime(1.76) == pytest.approx(0.56, abs=5e-3)


def test_T1_2_optimal_loss_1e25_fid_1_76():
    """Table 3, L* at 1e25 FLOPs. If this fails, G or K is wrong."""
    assert cm.optimal_loss(1e25, fid=1.76) == pytest.approx(0.61, abs=5e-3)


def test_T1_3_optimal_loss_1e25_fid_0():
    """Table 3, hypothetical FID = 0."""
    assert cm.optimal_loss(1e25, fid=0.0) == pytest.approx(0.53, abs=5e-3)


def test_T1_4_allocation_exponents():
    """Table 3 rounds these to 0.48 / 0.52; the exact fit gives 0.49 / 0.51."""
    assert cm.a == pytest.approx(0.489362, abs=5e-6)
    assert cm.b == pytest.approx(0.510638, abs=5e-6)
    assert cm.a + cm.b == pytest.approx(1.0, abs=1e-12)
    assert cm.kappa == pytest.approx(0.117447, abs=5e-6)
    assert cm.inv_k == pytest.approx(8.514493, abs=5e-6)


def test_T1_5_asymptote_fid_0():
    """Figure 1, ~90% asymptote."""
    assert cm.asymptotic_accuracy(0.0) == pytest.approx(0.898, abs=5e-4)


def test_T1_6_asymptote_fid_1_65():
    """Figure 1, FID = 1.65 band."""
    assert cm.asymptotic_accuracy(1.65) == pytest.approx(0.837, abs=5e-4)


def test_T1_7_optimal_accuracy_1e21_fid_0():
    """Consistent with Fig. 1 purple curve above green at 1e21 FLOPs."""
    assert cm.optimal_accuracy(1e21, fid=0.0) == pytest.approx(0.778, abs=5e-4)


def test_T1_8_flops_for_accuracy_sota():
    """Order 1e20 to reach the measured SOTA AutoAttack accuracy."""
    C = cm.flops_for_accuracy(0.7371, 0.0)
    assert C == pytest.approx(8.3e19, rel=0.05)
    assert 1e19 < C < 1e21


def test_T1_9_flops_for_accuracy_at_asymptote_raises():
    with pytest.raises(ValueError):
        cm.flops_for_accuracy(0.898, 0.0)
    # ... and anywhere above it.
    with pytest.raises(ValueError):
        cm.flops_for_accuracy(0.95, 0.0)


# --------------------------------------------------------------------------
# Internal consistency: the closed forms must agree with each other
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fid", [0.0, 1.65, 1.76, 5.0])
@pytest.mark.parametrize("C", [1e18, 1e21, 1e25, 1e30])
def test_optimal_alloc_saturates_the_budget(C, fid):
    N, D = cm.optimal_alloc(C, fid)
    assert cm.K_ADV * N * D == pytest.approx(C, rel=1e-9)
    assert N > 0 and D > 0


@pytest.mark.parametrize("fid", [0.0, 1.65, 1.76, 5.0])
@pytest.mark.parametrize("C", [1e18, 1e21, 1e25, 1e30])
def test_optimal_loss_matches_loss_at_optimal_alloc(C, fid):
    """L*(C) is L(N*, D*) - the substitution must be exact, not approximate."""
    N, D = cm.optimal_alloc(C, fid)
    assert cm.loss(N, D, fid) == pytest.approx(cm.optimal_loss(C, fid), rel=1e-12)


@pytest.mark.parametrize("fid", [0.0, 1.65, 1.76])
@pytest.mark.parametrize("C", [1e21, 1e24])
def test_optimal_alloc_is_actually_optimal(C, fid):
    """Perturbing the allocation off (N*, D*) at fixed C can only raise the loss."""
    N, D = cm.optimal_alloc(C, fid)
    best = cm.loss(N, D, fid)
    for scale in (0.5, 0.8, 0.9, 1.1, 1.25, 2.0):
        # Hold C = K_adv N D fixed while trading N against D.
        assert cm.loss(N * scale, D / scale, fid) >= best - 1e-12


@pytest.mark.parametrize("fid", [0.0, 1.65, 1.76])
@pytest.mark.parametrize("R", [0.60, 0.70, 0.75, 0.80])
def test_flops_for_accuracy_inverts_optimal_accuracy(R, fid):
    if R >= cm.asymptotic_accuracy(fid):
        pytest.skip("target above asymptote for this FID")
    C = cm.flops_for_accuracy(R, fid)
    assert cm.optimal_accuracy(C, fid) == pytest.approx(R, abs=1e-10)


def test_defender_cost_is_monotone_and_divergent():
    """The scientific claim's defender half: C_def blows up near the asymptote."""
    targets = [0.70, 0.75, 0.80, 0.85, 0.88, 0.895, 0.8976]
    costs = [cm.flops_for_accuracy(R, 0.0) for R in targets]
    assert all(c2 > c1 for c1, c2 in zip(costs, costs[1:]))
    # From 0.80 to 0.88 the cost must grow by more than six orders of magnitude.
    assert math.log10(cm.flops_for_accuracy(0.88, 0.0) / cm.flops_for_accuracy(0.80, 0.0)) > 6


def test_fid_monotonicity():
    """Worse generated data (higher FID) lowers the ceiling and raises the cost."""
    assert cm.asymptotic_accuracy(0.0) > cm.asymptotic_accuracy(1.65) > cm.asymptotic_accuracy(5.0)
    assert cm.flops_for_accuracy(0.70, 0.0) < cm.flops_for_accuracy(0.70, 1.76)


def test_acc_loss_roundtrip():
    for L in (0.4, 0.53, 0.61, 0.9):
        assert cm.loss_from_acc(cm.acc_from_loss(L)) == pytest.approx(L, rel=1e-12)


def test_fid_zero_collapses_to_base_constants():
    assert cm.E_prime(0.0) == pytest.approx(cm.E, rel=1e-12)
    assert cm.B_prime(0.0) == pytest.approx(cm.B, rel=1e-12)


def test_negative_fid_rejected():
    with pytest.raises(ValueError):
        cm.E_prime(-0.1)


# --------------------------------------------------------------------------
# S4.4 reference table
# --------------------------------------------------------------------------

# spec.md S4.4: "a sanity target, not ground truth" - hence the loose tolerance.
REFERENCE = {
    0.737: 8.3e19, 0.760: 3.1e20, 0.780: 1.2e21, 0.800: 5.7e21,
    0.820: 4.0e22, 0.840: 5.1e23, 0.850: 2.6e24, 0.860: 1.9e25,
    0.880: 1.2e28, 0.890: 1.4e31,
}


@pytest.mark.parametrize("R,expected", sorted(REFERENCE.items()))
def test_reference_table_row(R, expected):
    C = cm.flops_for_accuracy(R, 0.0)
    assert math.log10(C) == pytest.approx(math.log10(expected), abs=0.05)


def test_reference_table_last_row_is_divergent():
    rows = dict(cm.reference_table(0.0))
    assert rows[0.898] is None


# --------------------------------------------------------------------------
# Robustness checks: unrestricted fit, human-alignment correction
# --------------------------------------------------------------------------

def test_unrestricted_fit_gives_the_same_qualitative_picture():
    """Appendix F.2.1 alternative fit: numbers move, the divergence does not."""
    for R in (0.70, 0.75, 0.80):
        c_r = cm.flops_for_accuracy(R, 0.0, cm.RESTRICTED)
        c_u = cm.flops_for_accuracy(R, 0.0, cm.UNRESTRICTED)
        assert abs(math.log10(c_u / c_r)) < 2.0
    assert cm.asymptotic_accuracy(0.0, cm.UNRESTRICTED) == pytest.approx(0.890, abs=5e-3)


def test_human_alignment_correction_anchor():
    """Pitfall 8: the correction is pinned by the published 73.71 -> 79.49 anchor."""
    assert cm.to_valid_data_accuracy(cm.SOTA_AA_BENCHMARK) == pytest.approx(cm.SOTA_AA_VALID, abs=1e-9)
    assert cm.INVALID_AA_FRACTION == pytest.approx(0.0727, abs=1e-3)
    # The corrected asymptote lands just above the human ceiling: the divergence
    # shifts right, it does not disappear.
    assert cm.asymptotic_valid_data_accuracy(0.0) == pytest.approx(0.968, abs=2e-3)
    assert cm.flops_for_valid_data_accuracy(0.7949, 0.0) == pytest.approx(
        cm.flops_for_accuracy(0.7371, 0.0), rel=1e-9)
