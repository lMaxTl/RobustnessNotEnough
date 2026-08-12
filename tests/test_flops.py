"""M2 acceptance (spec.md S5). Measured counts, and the heuristic cross-check."""

import pytest

torch = pytest.importorskip("torch")

from src import flops as fl
from src.students import build_student


@pytest.fixture(scope="module")
def student():
    return build_student("resnet18_cifar", 0)


def test_forward_flops_is_measured_not_guessed(student):
    f = fl.measure_forward_flops(student)
    # CIFAR ResNet-18 is ~0.56 GMACs; FlopCounterMode counts 2 FLOPs per MAC.
    assert 0.9e9 < f < 1.4e9, f


def test_forward_flops_scales_linearly_in_batch(student):
    f1 = fl.measure_forward_flops(student, (1, 3, 32, 32))
    f8 = fl.measure_forward_flops(student, (8, 3, 32, 32))
    assert f8 == pytest.approx(8 * f1, rel=1e-6)


def test_train_step_is_forward_plus_backward(student):
    """Backward is ~2x forward, so a train step lands near 3x forward per sample."""
    fwd = fl.measure_forward_flops(student)
    per_sample = fl.measure_train_flops_per_sample(student, batch_size=8)
    assert 2.5 * fwd < per_sample < 4.0 * fwd, (fwd, per_sample)


def test_heuristic_constants_match_the_spec():
    assert fl.HEURISTIC_FWD_PER_PARAM == pytest.approx(289.7, abs=0.05)
    assert fl.HEURISTIC_TRAIN_PER_PARAM == pytest.approx(869.1, abs=0.05)


def test_measured_vs_heuristic_ratio_is_reported_and_sane(student, capsys):
    """S5.3: report the ratio for at least one model; investigate if it is wild.

    The Kaplan-style heuristic was calibrated on WRNs under adversarial
    training, so a CIFAR ResNet-18 is expected to sit *below* it - convolution
    weights are reused across spatial positions. What would be alarming is
    order-of-magnitude disagreement.
    """
    r_fwd = fl.measured_vs_heuristic_ratio(student, "forward")
    r_train = fl.measured_vs_heuristic_ratio(student, "train", batch_size=8)
    print(f"\nresnet18_cifar measured/heuristic: forward {r_fwd:.3f}, train {r_train:.3f}")
    assert 0.1 < r_fwd < 10
    assert 0.1 < r_train < 10


@pytest.mark.parametrize("depth,widen", [(16, 4), (28, 10)])
def test_kadv_constant_reproduces_measured_wrn_flops(depth, widen):
    """An independent validation of the source paper's constants.

    K_ADV = 7822 was fitted over 9000+ adversarial-training observations and
    FWD_ADV = 27 comes from Appendix B. Their ratio predicts the *measured*
    forward cost of CIFAR-10 WideResNets to within a few percent, across two
    orders of magnitude in parameter count. If this ever drifts, either the
    architecture definitions or the constants are wrong.
    """
    from src.students import WideResNet

    model = WideResNet(depth, widen)
    per_param = fl.measure_forward_flops(model) / fl.n_params(model)
    assert per_param / fl.HEURISTIC_FWD_PER_PARAM == pytest.approx(1.0, abs=0.05)


def test_wrn_94_16_matches_the_victim_parameter_count():
    """S12's schema example says the SOTA victim has ~366M parameters."""
    from src.students import WideResNet

    assert fl.n_params(WideResNet(94, 16)) == pytest.approx(366_000_000, rel=0.01)


def test_attack_cost_contract(student):
    victim = build_student("wrn_16_4", 0)
    cost = fl.attack_cost(victim, student, n_queries=1000, n_epochs=10)
    assert set(cost) == {"c_query", "c_student", "c_total", "c_query_heuristic",
                         "c_student_heuristic", "method"}
    assert cost["method"] == "flop_counter"
    assert cost["c_total"] == cost["c_query"] + cost["c_student"]
    assert cost["c_query"] > 0 and cost["c_student"] > 0


def test_attack_cost_scales_with_queries_and_epochs(student):
    victim = build_student("wrn_16_4", 0)
    a = fl.attack_cost(victim, student, n_queries=1000, n_epochs=10)
    b = fl.attack_cost(victim, student, n_queries=2000, n_epochs=10)
    c = fl.attack_cost(victim, student, n_queries=1000, n_epochs=20)
    assert b["c_query"] == pytest.approx(2 * a["c_query"], rel=1e-6)
    assert c["c_query"] == a["c_query"]
    assert c["c_student"] == pytest.approx(2 * a["c_student"], rel=1e-6)


def test_queries_per_sample_only_charges_the_victim(student):
    victim = build_student("wrn_16_4", 0)
    a = fl.attack_cost(victim, student, n_queries=1000, n_epochs=5)
    b = fl.attack_cost(victim, student, n_queries=1000, n_epochs=5, queries_per_sample=3)
    assert b["c_query"] == pytest.approx(3 * a["c_query"], rel=1e-6)
    assert b["c_student"] == a["c_student"]


def test_query_term_is_not_negligible():
    """S5.3: against a large victim the query term is ~10% of student training.

    Checked here with a WRN-28-10 stand-in for the 366M-parameter victim: the
    point is that the ratio is percent-scale, not negligible.
    """
    victim = build_student("wrn_28_10", 0)
    student = build_student("resnet18_cifar", 0)
    cost = fl.attack_cost(victim, student, n_queries=200_000, n_epochs=100)
    ratio = cost["c_query"] / cost["c_student"]
    assert ratio > 1e-3, ratio


def test_synthesis_cost_is_linear_in_steps(student):
    a = fl.synthesis_cost(student, n_samples=1000, n_steps=1)
    b = fl.synthesis_cost(student, n_samples=1000, n_steps=10)
    assert b == pytest.approx(10 * a, rel=1e-6)
