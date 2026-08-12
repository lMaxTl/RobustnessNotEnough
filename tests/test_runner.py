"""Grid expansion, resume keys and shard partitioning in scripts/run_sweep.py.

A sharding bug does not crash: it silently drops runs or runs them twice across
independent jobs, and the damage only shows up as a hole in results.jsonl days
later. These tests are cheap and need no GPU.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_run_sweep():
    spec = importlib.util.spec_from_file_location("run_sweep", ROOT / "scripts" / "run_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load_run_sweep()


def _config(name):
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def _shard(runs, shard, num_shards):
    """The exact slicing main() performs."""
    ordered = sorted(runs, key=rs.shard_key)
    return ordered[shard::num_shards]


def _key(r, n_eval):
    return (r["victim"], r["attack"], r["pool"], r["budget"], r["student"],
            r["seed"], r["defense"], r["epochs"], n_eval)


def _identity(r):
    """Hashable identity of a run spec (n_eval is a list, so dicts are not)."""
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(r.items()))


# --------------------------------------------------------------------------
# Grid expansion matches spec.md S13
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config,expected", [
    ("e2_main", 192),        # S13.1: 8 victims x 2 attacks x 4 budgets x 3 seeds
    ("e2b_ablations", 29),   # S13.2: 2x8 attack + 3x3 pool + 2x2 capacity
    ("e2c_control", 21),     # S13.3: 7 robust victims x 3 seeds
    ("e4_defenses", 48),     # S13.5: 2 victims x 8 defenses x 3 seeds
    ("e4_watermark", 6),     # S11.3: 1 victim x 2 defenses x 3 seeds
])
def test_config_expands_to_the_spec_grid_size(config, expected):
    assert len(rs.expand(_config(config))) == expected


def test_e2_final_cell_is_evaluated_at_both_n_evals():
    """S13.1: the Q=200,000 cell is re-evaluated at n_eval=10000 for the paper."""
    runs = rs.expand(_config("e2_main"))
    big = [r for r in runs if r["budget"] == 200_000]
    small = [r for r in runs if r["budget"] != 200_000]
    assert big and all(r["n_eval"] == [1000, 10000] for r in big)
    assert all(r["n_eval"] == [1000] for r in small)
    # 8 victims x 2 attacks x 3 seeds = the 48 final-eval runs S13.1 asks for.
    assert len(big) == 48
    assert len(small) == 144


def test_expand_is_deterministic():
    assert rs.expand(_config("e2_main")) == rs.expand(_config("e2_main"))


# --------------------------------------------------------------------------
# Sharding: a partition, stable under progress, and balanced
# --------------------------------------------------------------------------

@pytest.mark.parametrize("num_shards", [1, 2, 3, 4, 7, 60])
def test_shards_partition_the_grid_exactly(num_shards):
    runs = rs.expand(_config("e2_main"))
    seen = []
    for s in range(num_shards):
        seen.extend(_shard(runs, s, num_shards))
    assert len(seen) == len(runs)                      # nothing run twice
    assert {_identity(r) for r in seen} == {_identity(r) for r in runs}  # nothing dropped


def test_shard_membership_does_not_change_as_runs_complete():
    """Sharding happens before completed runs are filtered out, so a resubmitted
    job owns the same slice however much of it has already finished."""
    runs = rs.expand(_config("e2_main"))
    before = _shard(runs, 2, 4)
    # Simulate progress: half the grid is done and would have been filtered.
    partly_done = [r for i, r in enumerate(runs) if i % 2]
    after = _shard(runs, 2, 4)
    assert before == after
    assert _shard(partly_done, 2, 4) != before          # ... the naive order would drift


def test_expensive_runs_are_spread_across_shards():
    """shard_key sorts by descending budget so the Q=200,000 cell does not land
    entirely in one job (longest-processing-time-first)."""
    runs = rs.expand(_config("e2_main"))
    counts = [sum(1 for r in _shard(runs, s, 4) if r["budget"] == 200_000)
              for s in range(4)]
    assert max(counts) - min(counts) <= 1, counts
    # The naive (victim, attack, budget, seed) order would not be balanced.
    naive = sorted(runs, key=lambda r: (r["victim"], r["attack"], r["budget"], r["seed"]))
    naive_counts = [sum(1 for r in naive[s::4] if r["budget"] == 200_000) for s in range(4)]
    assert max(naive_counts) - min(naive_counts) > 1, naive_counts


def test_every_shard_gets_work_for_realistic_job_counts():
    runs = rs.expand(_config("e2_main"))
    for n in (4, 8, 16):
        assert all(len(_shard(runs, s, n)) > 0 for s in range(n))


# --------------------------------------------------------------------------
# Resume keys
# --------------------------------------------------------------------------

def test_done_keys_round_trip_through_a_written_record(tmp_path):
    from src.schema import JsonlWriter

    from test_schema import make_record

    path = tmp_path / "results.jsonl"
    JsonlWriter(path).append(make_record(victim_name="Peng2023Robust", attack="best",
                                         pool="cifar100", n_queries_nominal=50_000,
                                         seed=1, n_eval=1000))
    keys = rs.done_keys(path)
    assert ("Peng2023Robust", "best", "cifar100", 50_000, "resnet18_cifar", 1,
            "none", 100, 1000) in keys


def test_done_keys_maps_temperature_controls_back_to_their_target_victim(tmp_path):
    """E2c records are written as `Standard_T3.42`, but the config asks for
    `Peng2023Robust`. Without the mapping the control would re-run forever and
    duplicate its records on every resubmission."""
    from src.schema import JsonlWriter

    from test_schema import make_record

    path = tmp_path / "results.jsonl"
    rec = make_record(victim_name="Standard_T3.42", victim_aa_acc=0.0, tau=None,
                      is_control=True, attack="best", n_queries_nominal=50_000, seed=0)
    rec.extra = {"matched_victim": "Peng2023Robust", "temperature": 3.42}
    JsonlWriter(path).append(rec)
    keys = rs.done_keys(path)
    assert ("Peng2023Robust", "best", "cifar100", 50_000, "resnet18_cifar", 0,
            "none", 100, 1000) in keys
    assert not any(k[0] == "Standard_T3.42" for k in keys)


def test_missing_n_eval_alone_is_what_gets_rerun(tmp_path):
    """A run with an n_eval=1000 record but no n_eval=10000 record is pending for
    10000 only - it must not write a second 1000 record."""
    from src.schema import JsonlWriter

    from test_schema import make_record

    path = tmp_path / "results.jsonl"
    JsonlWriter(path).append(make_record(
        victim_name="Standard", victim_aa_acc=0.0, tau=None, attack="best",
        pool="cifar100", n_queries_nominal=200_000, seed=0, n_eval=1000))
    done = rs.done_keys(path)
    run = {"victim": "Standard", "attack": "best", "pool": "cifar100",
           "budget": 200_000, "student": "resnet18_cifar", "seed": 0,
           "defense": "none", "epochs": 100, "n_eval": [1000, 10000]}
    missing = [ne for ne in run["n_eval"] if _key(run, ne) not in done]
    assert missing == [10000]


def test_done_keys_on_a_missing_file_is_empty(tmp_path):
    assert rs.done_keys(tmp_path / "nope.jsonl") == set()
