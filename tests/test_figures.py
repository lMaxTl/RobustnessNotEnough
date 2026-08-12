"""D5-D8 (spec.md §2) render, and refuse the things §14.3 / pitfall 3 forbid."""

import json

import pytest

from src import figures as fg
from src.schema import JsonlWriter, MarginRecord
from test_analysis import VICTIM_R, synth_rows   # tests/ is on sys.path via conftest


@pytest.fixture
def rows():
    return synth_rows(slope=0.0, intercept=0.52, noise=0.01, seed=0, attack="best") \
        + synth_rows(slope=-0.1, intercept=0.50, noise=0.01, seed=1, attack="random")


@pytest.fixture
def margins_path(tmp_path):
    path = tmp_path / "margins.jsonl"
    w = JsonlWriter(path)
    for name, R in VICTIM_R.items():
        n = 200
        w.append(MarginRecord(
            victim_name=name, victim_aa_acc=R, n_images=n, eps_ref=8 / 255,
            margins=[0.01 + 0.00001 * i for i in range(n)],
            censored=[i > 150 for i in range(n)], median_margin=0.015,
            iqr=[0.012, 0.018], frac_margin_le_8_255=1 - R, rho_8_255=10.0,
            fab_n_restarts=3, fab_n_iter=100, search_bound=8 / 255,
            wallclock_sec=1.0, gpu="test"))
    return path


@pytest.fixture(autouse=True)
def _isolate_figdir(tmp_path, monkeypatch):
    """Never write test figures into results/figures - they would be
    indistinguishable from real ones."""
    monkeypatch.setattr(fg, "FIG_DIR", tmp_path / "figures")


def test_figure1_renders_both_formats(rows):
    paths = fg.figure1_defender_vs_attacker(rows, attack="best", budget=50_000)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.exists() and p.stat().st_size > 5_000 for p in paths)


def test_figure2_renders(rows):
    paths = fg.figure2_tau_vs_rv(rows, budget=50_000, n_boot=200)
    assert all(p.exists() for p in paths)


def test_figure3_renders(margins_path):
    paths = fg.figure3_margin_cdf(margins_path)
    assert all(p.exists() for p in paths)


def test_figure3_without_margins_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        fg.figure3_margin_cdf(tmp_path / "absent.jsonl")


def test_figures_refuse_mixed_n_eval(rows):
    """Pitfall 3: a 1,000-image AA estimate must never be plotted against a
    10,000-image one."""
    mixed = rows + synth_rows(n_eval=10000, attack="best", seed=9)
    with pytest.raises(AssertionError, match="mixed n_eval"):
        fg.figure1_defender_vs_attacker(mixed, attack="best")


def test_table1_marks_extrapolation_and_states_the_measured_range(rows):
    md = fg.table1_leverage(rows, attack="best", budget=50_000, n_boot=200)
    assert "Measured victim AutoAttack range is [0.595, 0.737]" in md
    assert "**extrapolated**" in md          # 0.80/0.85/0.88 rows
    assert md.count("|") > 20                 # a real table, not an empty stub
    assert (fg.FIG_DIR / "table1_leverage.md").exists()


def test_table1_reports_the_unreachable_matched_rows(rows):
    """With tau ~ 0.5 the S14.2 solve has no answer; the table says so instead
    of printing a fabricated R_v."""
    md = fg.table1_leverage(rows, attack="best", budget=50_000, n_boot=200)
    assert "unreachable" in md


def test_short_name_keeps_victims_distinguishable():
    a = fg._short_name("Bartoldson2024Adversarial_WRN-82-8")
    b = fg._short_name("Bartoldson2024Adversarial_WRN-94-16")
    assert a != b and len(a) <= 26 and len(b) <= 26


def test_make_all_runs_the_whole_deliverable_set(tmp_path, rows, margins_path):
    results = tmp_path / "results.jsonl"
    w = JsonlWriter(results)
    for r in rows:
        w.append(type("R", (), {"to_json": lambda self, r=r: json.dumps(r)})())
    made = fg.make_all(results, margins_path, n_eval=1000, attack="best",
                       budget=50_000, n_boot=200)
    assert set(made) == {"fig1", "fig2", "fig3", "table1"}
    assert "Table 1" in made["table1"]
