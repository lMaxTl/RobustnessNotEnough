"""M5 (spec.md S8) end to end against a synthetic victim.

No RobustBench download and no AutoAttack: this exercises the attacker loop,
the query accounting (pitfall 5) and the FLOPs decomposition, which is where
silent errors would be most expensive.
"""

import pytest

torch = pytest.importorskip("torch")

from src import defenses as dfn
from src import evaluate as ev
from src import extract as ext
from src.flops import measure_forward_flops, n_params
from src.students import build_student, params_fingerprint
from src.victims import Victim, query

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def fake_victim():
    """A small architecture standing in for a RobustBench victim.

    Briefly trained on real labels first: a randomly initialised net predicts
    nearly one class for every input, and against such a degenerate victim any
    student trivially reaches fidelity 1.0, which would make the A4 noise
    control vacuous.
    """
    from src import data as _data

    model = build_student("wrn_16_4", 123).to(DEVICE)
    x, y = _data.cifar10_test(5000)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    model.train()
    for _ in range(2):
        for i in range(0, len(x), 128):
            xb = _data.to_float(x[i:i + 128]).to(DEVICE)
            yb = y[i:i + 128].to(DEVICE)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Victim(name="FakeVictim", model=model, n_params=n_params(model),
                  fwd_flops=measure_forward_flops(model), clean_acc=0.5, aa_acc=0.2)


def tiny(victim, attack="random", pool="uniform_noise", n_queries=256, seed=0,
         defense=None, epochs=1):
    return ext.extract(victim, "resnet18_cifar", attack, pool, n_queries, seed,
                       defense=defense, epochs=epochs, device=DEVICE,
                       keep_transfer_set=True)


# --------------------------------------------------------------------------

@pytest.mark.parametrize("attack", ["random", "noise", "jbda", "best"])
def test_every_attack_runs_and_respects_the_budget(fake_victim, attack):
    res = tiny(fake_victim, attack=attack, n_queries=256)
    assert res.n_queries_actual <= 256
    assert res.n_queries_actual == fake_victim.n_queries_used
    assert len(res.transfer_x) == res.n_queries_actual
    assert res.transfer_p.shape == (res.n_queries_actual, 10)
    assert torch.allclose(res.transfer_p.sum(1), torch.ones(len(res.transfer_p)), atol=1e-4)


@pytest.mark.parametrize("attack", ["random", "noise", "jbda", "best"])
@pytest.mark.parametrize("n_queries", [237, 256])
def test_every_attack_spends_the_whole_budget(fake_victim, attack, n_queries):
    """Regression: `best` split the budget into Q//10 rounds and `jbda` seeded
    with floor(Q / 2^rho), so both under-spent on budgets that do not divide
    evenly - making them look cheaper than `random` at the same nominal Q."""
    res = tiny(fake_victim, attack=attack, n_queries=n_queries)
    assert res.n_queries_actual == n_queries


def test_query_counter_is_the_ground_truth(fake_victim):
    """Pitfall 5: count actual query() invocations, never the nominal budget."""
    fake_victim.reset_query_counter()
    query(fake_victim, torch.rand(10, 3, 32, 32))
    assert fake_victim.n_queries_used == 10
    res = tiny(fake_victim, n_queries=128)          # extract() resets the counter
    assert res.n_queries_actual == 128


def test_extraction_is_deterministic_for_a_seed(fake_victim):
    a = tiny(fake_victim, attack="random", n_queries=128, seed=0)
    b = tiny(fake_victim, attack="random", n_queries=128, seed=0)
    c = tiny(fake_victim, attack="random", n_queries=128, seed=1)
    assert torch.equal(a.transfer_x, b.transfer_x)
    assert params_fingerprint(a.student) == params_fingerprint(b.student)
    assert params_fingerprint(a.student) != params_fingerprint(c.student)


def test_synthesis_cost_is_charged_only_to_multi_step_attacks(fake_victim):
    plain = tiny(fake_victim, attack="random", n_queries=256)
    best = tiny(fake_victim, attack="best", n_queries=256)
    jbda = tiny(fake_victim, attack="jbda", n_queries=256)
    assert plain.c_synthesis == 0
    assert best.c_synthesis > 0
    assert jbda.c_synthesis > 0
    # S5.3: synthesis is charged on the student, never on the victim.
    assert best.c_query == pytest.approx(plain.c_query, rel=1e-9)


def test_cost_decomposition_adds_up_and_is_measured(fake_victim):
    res = tiny(fake_victim, attack="best", n_queries=256)
    assert res.c_total == res.c_query + res.c_synthesis + res.c_student
    assert res.extra["flops_method"] == "flop_counter"
    assert res.c_query_heuristic > 0 and res.c_student_heuristic > 0
    # Pitfall 6: measured and heuristic are both recorded, never silently mixed.
    assert res.extra["measured_over_heuristic_query"] > 0


def test_queries_are_raw_zero_one_tensors(fake_victim):
    """Pitfall 1: the victim is fed raw [0,1]; anything else must raise."""
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        query(fake_victim, torch.randn(4, 3, 32, 32) * 5)


def test_defense_is_applied_to_every_query(fake_victim):
    res = tiny(fake_victim, defense=dfn.build_defense("top1"), n_queries=128)
    p = res.transfer_p
    assert torch.equal(p.max(1).values, torch.ones(len(p)))


def test_prada_observes_the_stream(fake_victim):
    d = dfn.build_defense("prada")
    res = tiny(fake_victim, defense=d, n_queries=256)
    assert d.n_seen == res.n_queries_actual
    assert "prada_detect_index" in res.extra


def test_noise_attack_learns_nothing_beyond_the_prior(fake_victim):
    """A4 control (S8.2): uniform-noise queries teach the student almost nothing
    about the victim's behaviour on real images.

    The honest bound is the victim's own majority-class frequency: a student
    that has only seen noise can do no better than predicting the victim's
    modal class everywhere. Stating it this way keeps the test meaningful
    whatever the victim's class balance happens to be.
    """
    from src import data as _data
    from src.victims import query_logits

    x, _ = _data.cifar10_test(1000)
    v_pred = query_logits(fake_victim, x, count=False).argmax(1)
    majority = float(v_pred.bincount(minlength=10).max()) / len(v_pred)

    noise = tiny(fake_victim, attack="noise", n_queries=512, epochs=2)
    fid_noise = ev.fidelity_clean(noise.student, fake_victim, n_eval=1000)
    assert fid_noise <= majority + 0.05, (fid_noise, majority)


def test_unknown_attack_and_pool_rejected(fake_victim):
    with pytest.raises(ValueError):
        tiny(fake_victim, attack="not_an_attack")
    with pytest.raises(ValueError):
        tiny(fake_victim, pool="not_a_pool")


def test_epochs_are_reflected_in_student_cost(fake_victim):
    one = tiny(fake_victim, n_queries=128, epochs=1)
    two = tiny(fake_victim, n_queries=128, epochs=2)
    assert two.c_student == pytest.approx(2 * one.c_student, rel=0.05)
    assert two.extra["sample_passes"] == 2 * one.extra["sample_passes"]


def test_fidelity_and_tau_helpers(fake_victim):
    res = tiny(fake_victim, n_queries=256, epochs=1)
    fid = ev.fidelity_clean(res.student, fake_victim, n_eval=500)
    assert 0.0 <= fid <= 1.0
    assert ev.tau(0.3, fake_victim.aa_acc) == pytest.approx(1.5)
    assert ev.tau(0.3, 0.0) is None          # S9.3: undefined for `Standard`
    with pytest.raises(AssertionError):
        ev.assert_no_gradient_masking(aa_acc=0.4, pgd_acc=0.3)
