"""Experiment runner for E2 / E2b / E2c / E3 / E4 (spec.md S13).

    python scripts/run_sweep.py --config configs/e2_main.yaml
    python scripts/run_sweep.py --config configs/e2_main.yaml --dry-run
    python scripts/run_sweep.py --config configs/e3_margins.yaml

Runs are keyed by (victim, attack, pool, budget, student, seed, defense, epochs,
n_eval) and skipped if that key is already in results.jsonl, so an interrupted
sweep resumes by re-running the same command. Records are flushed per run.

For a batch system, one config can be split across independent jobs:

    python scripts/run_sweep.py --config configs/e2_main.yaml \
        --victims Peng2023Robust --shard 2 --num-shards 4 \
        --results results/shards/e2_Peng2023Robust_s2.jsonl \
        --done-also results/results.jsonl --max-seconds 84600

Sharding is applied to the *full* grid before completed runs are removed, so a
shard always owns the same runs no matter how much of it has already finished.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src import defenses as dfn  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import extract as ext  # noqa: E402
from src import margins as mg  # noqa: E402
from src import victims as vic  # noqa: E402
from src.repro import setup_determinism  # noqa: E402
from src.schema import (MARGINS_PATH, RESULTS_PATH, JsonlWriter, ResultRecord,  # noqa: E402
                        gpu_name, read_jsonl)


# --------------------------------------------------------------------------
# Config expansion
# --------------------------------------------------------------------------

def _victims_of(block: dict) -> list[str]:
    v = block.get("victims", "all")
    return sorted(vic.VICTIMS) if v == "all" else list(v)


def expand(config: dict) -> list[dict]:
    """Cartesian product of every block into a flat list of run specs."""
    runs = []
    defaults = {k: config[k] for k in ("epochs", "n_eval", "device") if k in config}
    for block in config["blocks"]:
        b = {**defaults, **block}
        n_evals = b.get("n_eval", [1000])
        n_evals = [n_evals] if isinstance(n_evals, int) else list(n_evals)
        for victim, attack, pool, budget, student, seed, defense in itertools.product(
                _victims_of(b), b.get("attacks", ["best"]), b.get("pools", ["cifar100"]),
                b["budgets"], b.get("students", ["resnet18_cifar"]), b["seeds"],
                b.get("defenses", ["none"])):
            runs.append({
                "block": b.get("name", config.get("name", "run")),
                "mode": b.get("mode", "extraction"),
                "victim": victim, "attack": attack, "pool": pool,
                "budget": int(budget), "student": student, "seed": int(seed),
                "defense": defense, "epochs": int(b.get("epochs", 100)),
                "n_eval": n_evals,
            })
    return runs


def done_keys(path: Path) -> set[tuple]:
    keys = set()
    for r in read_jsonl(path):
        # A temperature-matched control (E2c) is recorded under the scaled name
        # ("Standard_T3.42"), never under the victim the config asked for, so
        # matching on victim_name alone would make E2c re-run every time and
        # duplicate its records. `extra.matched_victim` carries the config-level
        # name and is written by run_extraction for exactly this reason.
        name = (r.get("extra") or {}).get("matched_victim") or r["victim_name"]
        keys.add((name, r["attack"], r["pool"], r["n_queries_nominal"],
                  r["student_arch"], r["seed"], r["defense"], r["epochs"], r["n_eval"]))
    return keys


def shard_key(r: dict) -> tuple:
    """Order for splitting a grid across jobs: most expensive run first.

    Round-robin over a budget-descending order is longest-processing-time-first
    scheduling. Sharding the natural (victim, attack, budget, seed) order instead
    would hand every Q=200,000 run to the same few shards - with 4 shards and 3
    seeds, shard 0 would get none of them and shard 1 two.
    """
    return (-r["budget"], r["victim"], r["attack"], r["pool"], r["student"],
            r["defense"], r["seed"], r["epochs"])


# --------------------------------------------------------------------------
# Extraction runs (E2, E2b, E2c, E4)
# --------------------------------------------------------------------------

def run_extraction(spec: dict, victim, writer: JsonlWriter, device: str,
                   adv_cache: dict, utility_cache: dict, verbose: bool = True) -> None:
    defense = dfn.build_defense(spec["defense"])
    # Measured once per (victim, defense): the denominator of S11.4's
    # cost-effectiveness, so it is measured on the full test set, not assumed.
    ukey = (victim.name, defense.name)
    if ukey not in utility_cache:
        utility_cache[ukey] = {
            "defense_utility_cost": defense.utility_cost(victim),
            "posterior_distortion": dfn.posterior_distortion(defense, victim),
        }
    t0 = time.time()
    res = ext.extract(victim, spec["student"], spec["attack"], spec["pool"],
                      spec["budget"], spec["seed"], defense=defense,
                      epochs=spec["epochs"], device=device, verbose=verbose)
    train_sec = time.time() - t0

    # S11.3: does the watermark survive extraction? Only meaningful for the one
    # victim we could actually backdoor.
    watermark = {}
    if victim.name in vic.LOCAL_VICTIMS:
        watermark = {
            "watermark_rate_student": dfn.watermark_success_rate(res.student),
            "watermark_rate_victim": dfn.watermark_success_rate(victim.model),
            "watermark_chance": 0.1,
        }

    for n_eval in spec["n_eval"]:
        key = (victim.name, n_eval)
        if key not in adv_cache:
            adv_cache[key] = ev.victim_adversarial_examples(victim, n_eval=n_eval)
        metrics = ev.evaluate_student(res.student, victim, n_eval=n_eval,
                                      seed=spec["seed"], x_adv_victim=adv_cache[key])
        rec = ResultRecord(
            victim_name=victim.name, victim_params=victim.n_params,
            victim_clean_acc=victim.clean_acc, victim_aa_acc=victim.aa_acc,
            is_control=victim.is_control, student_arch=res.student_arch,
            student_params=res.student_params, attack=res.attack, pool=res.pool,
            defense=defense.name, defense_params=dict(getattr(defense, "params", {})),
            n_queries_nominal=res.n_queries_nominal,
            n_queries_actual=res.n_queries_actual, epochs=res.epochs, seed=res.seed,
            student_clean_acc=metrics["student_clean_acc"],
            student_aa_acc=metrics["student_aa_acc"],
            student_pgd40_acc=metrics["student_pgd40_acc"],
            fidelity_clean=metrics["fidelity_clean"],
            fidelity_adv=metrics["fidelity_adv"], tau=metrics["tau"], n_eval=n_eval,
            c_query=res.c_query, c_synthesis=res.c_synthesis, c_student=res.c_student,
            c_total=res.c_total, c_query_heuristic=res.c_query_heuristic,
            c_student_heuristic=res.c_student_heuristic,
            wallclock_sec=time.time() - t0, gpu=gpu_name(),
            extra={**res.extra, **watermark, **utility_cache[ukey], "train_sec": train_sec,
                   "block": spec["block"], **({"temperature": victim.meta["temperature"],
                                               "matched_victim": victim.meta.get("matched_victim")}
                                              if victim.is_control else {})},
        )
        writer.append(rec)
        if verbose:
            print(f"    n_eval={n_eval}: clean={rec.student_clean_acc:.4f} "
                  f"AA={rec.student_aa_acc:.4f} PGD40={rec.student_pgd40_acc:.4f} "
                  f"fid={rec.fidelity_clean:.4f} tau="
                  f"{'null' if rec.tau is None else f'{rec.tau:.4f}'}")
    del res.student
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# E2c temperature-matched control (S9.4)
# --------------------------------------------------------------------------

def build_control_victim(base_standard, robust_name: str, pool: str, device: str,
                         n_entropy: int = 10_000):
    """Temperature-scale `Standard` to match the robust victim's mean entropy."""
    robust = vic.load_victim(robust_name, device=device)
    x_pool = ext.sample_pool(pool, n_entropy, seed=0)
    H_v = vic.mean_predictive_entropy(robust, x_pool, n=n_entropy)
    del robust
    torch.cuda.empty_cache()
    T = vic.match_temperature(base_standard, H_v, x_pool, n=n_entropy)
    ctrl = vic.temperature_scaled_victim(base_standard, T)
    ctrl.meta.update({"target_entropy": H_v, "matched_victim": robust_name})
    print(f"  control for {robust_name}: H_v={H_v:.4f} nats -> T={T:.4f} "
          f"({ctrl.name})")
    return ctrl


# --------------------------------------------------------------------------
# E3 margins
# --------------------------------------------------------------------------

def run_margins(config: dict, device: str, dry_run: bool, path: Path = MARGINS_PATH,
                victims: list[str] | None = None, shard: int = 0, num_shards: int = 1,
                done_also: list[Path] | None = None, deadline: float | None = None) -> None:
    writer = JsonlWriter(path)
    have = {r["victim_name"] for r in read_jsonl(path)}
    for extra_path in done_also or []:
        have |= {r["victim_name"] for r in read_jsonl(extra_path)}
    block = config["blocks"][0]
    names = _victims_of(block)
    if victims:
        names = [n for n in names if n in set(victims)]
    names = names[shard::num_shards]
    for name in names:
        if name in have:
            print(f"skip {name} (already in {path.name})")
            continue
        if dry_run:
            print(f"would compute margins: {name}")
            continue
        if deadline is not None and time.time() > deadline:
            print(f"stopping before {name}: walltime budget exhausted "
                  "(re-submit to continue)")
            return
        v = vic.load_victim(name, device=device)
        rec = mg.compute_margins(v, n_images=int(block.get("n_images", 2000)),
                                 n_restarts=int(block.get("fab_n_restarts", 3)),
                                 n_iter=int(block.get("fab_n_iter", 100)),
                                 batch_size=int(block.get("batch_size", 64)))
        writer.append(rec)
        print(f"{name:44s} median={rec.median_margin} "
              f"frac<=8/255={rec.frac_margin_le_8_255:.4f} "
              f"(1-AA={1-rec.victim_aa_acc:.4f}) {rec.wallclock_sec:.0f}s")
        try:
            mg.assert_margin_aa_agreement(rec)
        except AssertionError as exc:
            print(f"  WARNING (S10.3 validation): {exc}")
        del v
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run at most N runs")
    ap.add_argument("--results", type=Path, default=RESULTS_PATH,
                    help="JSONL to append to (one file per job on a batch system)")
    ap.add_argument("--margins", type=Path, default=MARGINS_PATH,
                    help="JSONL for mode: margins")
    ap.add_argument("--done-also", type=Path, nargs="*", default=[],
                    help="extra JSONL files to read completed runs from (never written)")
    ap.add_argument("--victims", nargs="*", default=None,
                    help="restrict to these victim keys; for E2c these are the "
                         "entropy-target victims named in the config, not the "
                         "'Standard_T*' names the records get")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split this config across N independent jobs")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop launching new runs after this many seconds, so the "
                         "job exits cleanly instead of being killed mid-run")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards}), got {args.shard}")

    config = yaml.safe_load(args.config.read_text())
    device = args.device or config.get("device", "cuda")
    setup_determinism(seed=0)
    deadline = None if args.max_seconds is None else time.time() + args.max_seconds

    if config.get("mode") == "margins" or config["blocks"][0].get("mode") == "margins":
        run_margins(config, device, args.dry_run, path=args.margins,
                    victims=args.victims, shard=args.shard,
                    num_shards=args.num_shards, done_also=args.done_also,
                    deadline=deadline)
        return 0

    runs = expand(config)
    if args.victims:
        wanted = set(args.victims)
        unknown = wanted - {r["victim"] for r in runs}
        if unknown:
            raise SystemExit(f"--victims not present in {args.config.name}: {sorted(unknown)}")
        runs = [r for r in runs if r["victim"] in wanted]
    if args.num_shards > 1:
        # Shard the full grid, not the pending subset: otherwise a resubmitted
        # shard would be handed a different slice of the work each time.
        runs.sort(key=shard_key)
        runs = runs[args.shard::args.num_shards]

    done = done_keys(args.results)
    for extra_path in args.done_also:
        done |= done_keys(extra_path)
    pending = []
    for r in runs:
        # Only the missing n_eval evaluations are redone; a run whose n_eval=1000
        # record exists but whose n_eval=10000 record does not is re-trained (the
        # student is not checkpointed) but written once, not twice.
        missing = [ne for ne in r["n_eval"]
                   if (r["victim"], r["attack"], r["pool"], r["budget"], r["student"],
                       r["seed"], r["defense"], r["epochs"], ne) not in done]
        if missing:
            pending.append({**r, "n_eval": missing})
    scope = (f" [shard {args.shard}/{args.num_shards}]" if args.num_shards > 1 else "")
    scope += (f" victims={','.join(args.victims)}" if args.victims else "")
    print(f"config {args.config.name}{scope}: {len(runs)} runs, "
          f"{len(runs)-len(pending)} already done, {len(pending)} pending")
    if args.limit:
        pending = pending[:args.limit]
    if args.dry_run:
        for r in pending[:50]:
            print("  ", {k: r[k] for k in ("victim", "attack", "pool", "budget",
                                           "student", "seed", "defense", "n_eval")})
        if len(pending) > 50:
            print(f"   ... and {len(pending)-50} more")
        return 0

    writer = JsonlWriter(args.results)
    adv_cache: dict = {}
    utility_cache: dict = {}
    control_mode = config.get("mode") == "temperature_control"
    base_standard = vic.load_victim("Standard", device=device) if control_mode else None

    # Group by victim so the big checkpoints load once.
    pending.sort(key=lambda r: (r["victim"], r["attack"], r["budget"], r["seed"]))
    if control_mode and not (0.0 <= base_standard.aa_acc <= 1.0):
        raise RuntimeError("`Standard` has no verified AutoAttack accuracy; "
                           "run scripts/verify_victims.py first (spec.md S6.1)")

    current_name, victim = None, None
    for i, spec in enumerate(pending, 1):
        if deadline is not None and time.time() > deadline:
            print(f"\nstopping after {i-1}/{len(pending)} runs: walltime budget "
                  "exhausted. Re-submit the same job to continue.")
            break
        if spec["victim"] != current_name:
            victim = None
            torch.cuda.empty_cache()
            victim = (build_control_victim(base_standard, spec["victim"], spec["pool"], device)
                      if control_mode else vic.load_victim(spec["victim"], device=device))
            current_name = spec["victim"]
            adv_cache.clear()
            utility_cache.clear()
            if not (0.0 <= victim.aa_acc <= 1.0):
                raise RuntimeError(
                    f"victim {victim.name} has no verified AutoAttack accuracy. "
                    "Run scripts/verify_victims.py first (spec.md S6.1): tau is "
                    "meaningless without a measured denominator.")
        print(f"[{i}/{len(pending)}] {spec['victim']} {spec['attack']}/{spec['pool']} "
              f"Q={spec['budget']} seed={spec['seed']} defense={spec['defense']}")
        run_extraction(spec, victim, writer, device, adv_cache, utility_cache,
                       verbose=not args.quiet)
    print(f"\nwrote {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
