"""AutoAttack / FAB-T wiring against a synthetic victim.

These call the real libraries with tiny budgets. The point is not the numbers -
it is that the argument names, attribute paths and return shapes are right, so
a typo surfaces in seconds instead of six hours into an E3 run.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("autoattack")

from src import evaluate as ev
from src import margins as mg
from src.flops import measure_forward_flops, n_params
from src.students import build_student
from src.victims import Victim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def victim():
    from src import data as _data

    model = build_student("wrn_16_4", 7).to(DEVICE)
    x, y = _data.cifar10_test(2000)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    model.train()
    for i in range(0, len(x), 128):
        xb = _data.to_float(x[i:i + 128]).to(DEVICE)
        loss = torch.nn.functional.cross_entropy(model(xb), y[i:i + 128].to(DEVICE))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Victim(name="WiringVictim", model=model, n_params=n_params(model),
                  fwd_flops=measure_forward_flops(model), clean_acc=0.3, aa_acc=0.05)


def test_autoattack_runs_and_is_bounded_by_clean_accuracy(victim):
    clean = ev.clean_accuracy(victim.model, 128)
    aa = ev.autoattack_accuracy(victim.model, n_eval=128, version="custom",
                                attacks_to_run=["apgd-ce"], batch_size=64)
    assert 0.0 <= aa <= clean + 1e-9


def test_autoattack_is_at_least_as_strong_as_pgd40(victim):
    """Pitfall 4, checked on a real attack rather than only asserted in schema."""
    pgd = ev.pgd_accuracy(victim.model, n_eval=128, batch_size=64)
    aa = ev.autoattack_accuracy(victim.model, n_eval=128, version="custom",
                                attacks_to_run=["apgd-ce", "apgd-t"], batch_size=64)
    ev.assert_no_gradient_masking(aa, pgd)


def test_autoattack_can_return_the_adversarial_points(victim):
    acc, x_adv = ev.autoattack_accuracy(victim.model, n_eval=64, version="custom",
                                        attacks_to_run=["apgd-ce"], batch_size=64,
                                        return_adv=True)
    from src import data as _data

    x, _ = _data.cifar10_test(64)
    delta = (x_adv - _data.to_float(x)).abs().flatten(1).max(1).values
    assert x_adv.shape == (64, 3, 32, 32)
    assert float(delta.max()) <= ev.EPS_8_255 + 1e-5     # respects the budget
    assert 0.0 <= acc <= 1.0


def test_fab_margins_wiring(victim):
    """S10.2: the FAB-T attribute paths (`adv.fab.n_restarts`, `n_iter`)."""
    rec = mg.compute_margins(victim, n_images=64, n_restarts=1, n_iter=5,
                             batch_size=32)
    rec.validate()
    m = np.asarray(rec.margins)
    c = np.asarray(rec.censored)
    assert rec.n_images == 64
    assert (m >= 0).all() and (m <= rec.search_bound + 1e-6).all()
    # Censored points sit at the search bound by construction.
    assert np.allclose(m[c], rec.search_bound) if c.any() else True
    # Misclassified images have margin 0 (S10.2), so the CDF starts above 0.
    assert 0.0 <= rec.frac_margin_le_8_255 <= 1.0
    assert rec.median_margin is None or 0 <= rec.median_margin <= rec.search_bound


def test_margin_aa_agreement_check_fires_when_pipelines_disagree(victim):
    rec = mg.compute_margins(victim, n_images=64, n_restarts=1, n_iter=5,
                             batch_size=32)
    rec.victim_aa_acc = 0.99          # pretend AA said almost nothing flips
    with pytest.raises(AssertionError, match="frac_margin_le_8_255"):
        mg.assert_margin_aa_agreement(rec)


def test_victim_adversarial_examples_respect_the_budget(victim):
    from src import data as _data

    x_adv = ev.victim_adversarial_examples(victim, n_eval=64, batch_size=32)
    x, _ = _data.cifar10_test(64)
    delta = (x_adv - _data.to_float(x)).abs().flatten(1).max(1).values
    assert float(delta.max()) <= ev.EPS_8_255 + 1e-5
    assert float(x_adv.min()) >= 0 and float(x_adv.max()) <= 1
