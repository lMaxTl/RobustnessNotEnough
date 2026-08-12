"""M9 (spec.md S12): the schema is the analysis contract, so it validates hard."""

import json

import pytest

from src import schema as sc


def make_record(**over):
    base = dict(
        victim_name="Bartoldson2024Adversarial_WRN-94-16", victim_params=366_000_000,
        victim_clean_acc=0.9378, victim_aa_acc=0.7371, is_control=False,
        student_arch="resnet18_cifar", student_params=11_173_962, attack="best",
        pool="cifar100", defense="none", defense_params={},
        n_queries_nominal=50_000, n_queries_actual=50_000, epochs=100, seed=0,
        student_clean_acc=0.80, student_aa_acc=0.30, student_pgd40_acc=0.32,
        fidelity_clean=0.85, fidelity_adv=0.40, tau=0.30 / 0.7371, n_eval=1000,
        c_query=10, c_synthesis=5, c_student=85, c_total=100,
        c_query_heuristic=11, c_student_heuristic=90, wallclock_sec=1.0,
        gpu="A100-40GB",
    )
    base.update(over)
    return sc.ResultRecord(**base)


def test_valid_record_round_trips():
    rec = make_record().validate()
    parsed = json.loads(rec.to_json())
    for field in ("run_id", "timestamp", "git_sha", "victim_name", "tau", "n_eval",
                  "c_total", "gpu"):
        assert field in parsed
    assert parsed["tau"] == pytest.approx(0.30 / 0.7371)


def test_tau_may_be_null_but_nothing_else_may():
    make_record(tau=None, victim_aa_acc=0.0, victim_name="Standard").validate()
    with pytest.raises(ValueError):
        make_record(fidelity_adv=None).validate()


def test_gradient_masking_assert(tmp_path):
    """Pitfall 4: AA can never be stronger-looking than PGD-40."""
    with pytest.raises(ValueError, match="at least as strong"):
        make_record(student_aa_acc=0.40, student_pgd40_acc=0.30).validate()


def test_cost_decomposition_must_add_up():
    with pytest.raises(ValueError, match="c_total"):
        make_record(c_total=999).validate()


def test_n_eval_must_be_1000_or_10000():
    with pytest.raises(ValueError):
        make_record(n_eval=500).validate()


def test_accuracies_must_be_fractions():
    with pytest.raises(ValueError):
        make_record(student_clean_acc=80.0).validate()
    with pytest.raises(ValueError):
        make_record(victim_aa_acc=float("nan")).validate()


def test_tau_must_be_null_for_zero_aa_victims():
    with pytest.raises(ValueError, match="null"):
        make_record(victim_name="Standard", victim_aa_acc=0.0, tau=1.0).validate()


def test_writer_is_append_only_and_readable(tmp_path):
    path = tmp_path / "results.jsonl"
    w = sc.JsonlWriter(path)
    w.append(make_record(seed=0))
    w.append(make_record(seed=1))
    rows = list(sc.read_jsonl(path))
    assert [r["seed"] for r in rows] == [0, 1]
    # A second writer appends rather than truncating - a resumed sweep must not
    # destroy earlier records.
    sc.JsonlWriter(path).append(make_record(seed=2))
    assert len(list(sc.read_jsonl(path))) == 3


def test_writer_validates_before_writing(tmp_path):
    path = tmp_path / "results.jsonl"
    with pytest.raises(ValueError):
        sc.JsonlWriter(path).append(make_record(c_total=0))
    assert not path.exists() or path.read_text() == ""


def test_read_jsonl_reports_truncated_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n{"b": ')
    with pytest.raises(ValueError, match="malformed JSON"):
        list(sc.read_jsonl(path))


def test_margin_record_length_check():
    rec = sc.MarginRecord(
        victim_name="v", victim_aa_acc=0.5, n_images=2, eps_ref=8 / 255,
        margins=[0.01, 0.02], censored=[False, True], median_margin=0.015,
        iqr=[0.01, 0.02], frac_margin_le_8_255=0.5, rho_8_255=1.0,
        fab_n_restarts=3, fab_n_iter=100, search_bound=8 / 255,
        wallclock_sec=1.0, gpu="cpu")
    rec.validate()
    rec.n_images = 3
    with pytest.raises(ValueError):
        rec.validate()


def test_provenance_helpers_never_raise():
    assert isinstance(sc.git_sha(), str)
    assert isinstance(sc.gpu_name(), str)
    assert sc.utc_now().startswith("20")
    assert len(sc.new_run_id()) == 36
