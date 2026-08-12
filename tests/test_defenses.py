"""M8 (spec.md S11): defenses must do what they claim and nothing more."""

import pytest

torch = pytest.importorskip("torch")

from src import defenses as dfn


@pytest.fixture
def posteriors():
    torch.manual_seed(0)
    return torch.softmax(torch.randn(64, 10) * 2, dim=1)


def _is_simplex(p):
    return torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5) and (p >= -1e-6).all()


@pytest.mark.parametrize("name", sorted(dfn.DEFENSE_SPECS))
def test_every_defense_returns_a_valid_simplex(name, posteriors):
    out = dfn.build_defense(name)(posteriors)
    assert out.shape == posteriors.shape
    assert _is_simplex(out)


@pytest.mark.parametrize("name", ["none", "top1", "topk_2", "topk_3",
                                  "mad_0.5", "mad_0.99", "prada"])
def test_argmax_is_preserved(name, posteriors):
    """These defenses are exactly argmax-preserving, so they cost the operator
    zero top-1 accuracy - which is what makes S11.4's ratio interesting."""
    out = dfn.build_defense(name)(posteriors)
    assert torch.equal(out.argmax(1), posteriors.argmax(1))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_round_k_can_only_move_the_argmax_through_a_tie(k, posteriors):
    """Rounding is monotone, so it cannot reorder two classes - but it can make
    them equal, and then argmax picks the lower index. That is a real (small)
    utility cost, and `measure_utility_cost` is what quantifies it."""
    out = dfn.RoundK(k)(posteriors)
    moved = out.argmax(1) != posteriors.argmax(1)
    if moved.any():
        p_moved = posteriors[moved]
        chosen = p_moved.gather(1, out[moved].argmax(1, keepdim=True)).squeeze(1)
        assert (p_moved.max(1).values - chosen <= 10 ** -k).all()


def test_top1_is_one_hot(posteriors):
    out = dfn.build_defense("top1")(posteriors)
    assert torch.equal(out.max(1).values, torch.ones(len(out)))
    assert (out > 0).sum(1).eq(1).all()


@pytest.mark.parametrize("k", [1, 2, 3])
def test_round_k_matches_the_spec_definition(k, posteriors):
    """S11.2: round to k decimals, renormalise."""
    out = dfn.RoundK(k)(posteriors)
    expected = torch.round(posteriors, decimals=k)
    expected = expected / expected.sum(1, keepdim=True)
    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("k", [1, 2])
def test_round_k_destroys_information(k, posteriors):
    """Two posteriors that differ below the rounding grid become identical -
    that loss is the whole point of the defense."""
    nudged = (posteriors + 10 ** -(k + 2)).clamp_min(0)
    nudged = nudged / nudged.sum(1, keepdim=True)
    same = torch.isclose(dfn.RoundK(k)(posteriors), dfn.RoundK(k)(nudged), atol=1e-6)
    assert same.float().mean() > 0.9


@pytest.mark.parametrize("k", [2, 3])
def test_topk_keeps_exactly_k(k, posteriors):
    out = dfn.TopK(k)(posteriors)
    assert (out > 0).sum(1).le(k).all()
    assert (out > 0).sum(1).ge(1).all()


@pytest.mark.parametrize("eps", [0.5, 0.99])
def test_mad_respects_its_l1_budget(eps, posteriors):
    out = dfn.MAD(eps)(posteriors)
    l1 = (out - posteriors).abs().sum(1)
    assert (l1 <= eps + 1e-5).all(), l1.max()
    assert l1.max() > 0


def test_mad_actually_perturbs_more_at_higher_budget(posteriors):
    weak = (dfn.MAD(0.5)(posteriors) - posteriors).abs().sum(1).mean()
    strong = (dfn.MAD(0.99)(posteriors) - posteriors).abs().sum(1).mean()
    assert strong >= weak


def test_prada_passes_posteriors_through_and_tracks_state(posteriors):
    d = dfn.build_defense("prada")
    assert torch.equal(d(posteriors), posteriors)
    torch.manual_seed(0)
    d.observe(torch.rand(150, 3, 32, 32))
    assert d.n_seen == 150
    assert d.detect_index is None or isinstance(d.detect_index, int)
    d.reset()
    assert d.n_seen == 0 and d.detect_index is None


def test_prada_flags_a_synthetic_lattice_stream():
    """A JBDA-style stream (tight clusters around seeds) breaks the normality of
    the min-distance distribution, which is what PRADA keys on."""
    d = dfn.PRADA(window=50)
    torch.manual_seed(0)
    seeds = torch.rand(10, 3, 32, 32)
    stream = torch.cat([(s + 0.01 * torch.randn(20, 3, 32, 32)).clamp(0, 1)
                        for s in seeds])
    d.observe(stream)
    assert d.n_seen == 200
    assert d.detect_index is not None


def test_unknown_defense_rejected():
    with pytest.raises(ValueError):
        dfn.build_defense("does_not_exist")


def test_trigger_is_applied_out_of_place_and_in_range():
    x = torch.rand(4, 3, 32, 32)
    t = dfn.apply_trigger(x)
    assert t.shape == x.shape
    assert not torch.equal(t, x)
    assert torch.equal(t[:, :, :-dfn.TRIGGER_SIZE, :], x[:, :, :-dfn.TRIGGER_SIZE, :])
    assert 0 <= float(t.min()) and float(t.max()) <= 1
