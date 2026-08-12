# Extraction economics of adversarially robust CIFAR-10 classifiers

Implementation of [spec.md](spec.md). The claim under test:

> The FLOPs cost of *producing* adversarial robustness by training diverges as the target
> approaches the scaling-law asymptote (~90% AutoAttack accuracy on CIFAR-10). The FLOPs cost
> of *extracting* that robustness through a query API is approximately flat in the victim's
> robustness level. Therefore the leverage available to a model thief grows without bound.

The contribution is the **calibrated cost model and its divergence**, not the transfer finding
itself — Khaled & Cheriet (arXiv 2207.10561) and Yue et al. (ICLR 2023, BEST) already showed
robust models are more extractable (spec.md §17, pitfall 10).

## Status

| Stage | State |
|---|---|
| M1 `costmodel` (defender curve) | **complete and validated** — T1.1–T1.9 pass, §4.4 table reproduced exactly |
| M2–M11 (attacker side, defenses, analysis, figures) | **implemented, unit-tested**; a synthetic-victim run exercises the full extract → evaluate → record path |
| E2/E2b/E2c/E3/E4 experiment runs | **not run** — ~300 GPU-hours (spec.md §16). Run them with the commands below |
| `results/*.jsonl`, figures, Table 1 | empty until the sweeps run |

214 tests pass (`python -m pytest -q`, ~2 min), including the M1 acceptance tests, a
bit-identical-init determinism check, an end-to-end extraction against a synthetic victim, real
AutoAttack and FAB-T runs at tiny budgets to validate the library wiring, the analysis estimators
against synthetic records with a planted ground truth, and rendering of every figure.

No numbers in this repository were invented. Anything not yet measured is absent rather than
estimated (spec.md §0, rule 1).

## Layout

```
src/costmodel.py   M1  §4   defender scaling law, closed-form inverse, no GPU
src/flops.py       M2  §5   measured attacker FLOPs (FlopCounterMode) + heuristic cross-check
src/victims.py     M3  §6   RobustBench victims, query API, query counter, temperature control
src/students.py    M4  §7   resnet18_cifar (primary), wrn_16_4, wrn_28_10; deterministic init
src/extract.py     M5  §8   query pools + attacks: random, jbda, best, noise
src/evaluate.py    M6  §9   clean / AutoAttack / PGD-40 / fidelity / tau
src/margins.py     M7  §10  FAB-T margins, Kaplan-Meier stats, Q90 link
src/defenses.py    M8  §11  none, top1, round_k, topk, mad, prada, backdoor watermark
src/schema.py      M9  §12  validated append-only JSONL records
src/analysis.py    M11 §14  tau regression, leverage, F1–F3 falsification checks
src/figures.py         §2   D5–D8
src/data.py                 datasets; raw [0,1] tensors only, never Normalize
src/repro.py           §15  determinism + environment capture
```

## Install

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -e .[dev]
pip install "git+https://github.com/fra31/auto-attack"
pip install --no-deps "git+https://github.com/RobustBench/robustbench"
pip install "timm==1.0.28" "gdown==5.1.0" geotorch torchdiffeq requests tqdm
```

**Deviation from spec.md §15:** the spec pins `robustbench@v1.1`, but that *tag* predates five of
the eight required checkpoints (`Wang2023Better_*`, `Peng2023Robust`, `Bartoldson2024Adversarial_*`
are absent from its model zoo — 61 CIFAR-10 L∞ entries vs the 76 on master). Installing from
master yields a package that self-reports version 1.1 and contains all eight keys. This is
recorded in [REPRO.md](REPRO.md); nothing else in §15 is changed.

## Data

CIFAR-10 is downloaded and cached in `data/`. **CIFAR-100 — the primary query pool (§8.1) — is
not**, so the main sweep will fetch it (~169 MB) on first use. Fetch it deliberately with
`python -c "from src.data import cifar100_train_images as f; print(f().shape)"`, or set
`RE_NO_DOWNLOAD=1` to make any missing dataset raise instead of downloading.

## Run

Order matters: victims must be verified before any sweep, because their measured AutoAttack
accuracy is the τ denominator. `run_sweep.py` refuses to run against an unverified victim.

```bash
python -m pytest tests/test_costmodel.py -q            # must pass before touching a GPU
python scripts/verify_victims.py --n-eval 1000         # §6.1; --n-eval 10000 for the paper
python scripts/run_sweep.py --config configs/e2_main.yaml       # E2   192 runs
python scripts/run_sweep.py --config configs/e2b_ablations.yaml # E2b  ~40 runs
python scripts/run_sweep.py --config configs/e2c_control.yaml   # E2c  21 runs
python scripts/run_sweep.py --config configs/e3_margins.yaml    # E3   8 victims
python scripts/run_sweep.py --config configs/e4_defenses.yaml   # E4   48 runs
python scripts/train_watermark_victim.py                        # §11.3, the one AT run
python scripts/run_sweep.py --config configs/e4_watermark.yaml  # E4b  6 runs
python scripts/make_figures.py                                  # D4–D8 + analysis.json
```

Every runner is resumable: runs already present in `results/results.jsonl` are skipped, so
re-issuing the same command after a crash continues where it stopped. `--dry-run` prints the
pending grid without touching the GPU.

On a Slurm cluster, run the same grid as ~60 concurrent jobs instead:
[scripts/hpc/README.md](scripts/hpc/README.md) (sized for HoreKa's 48-hour limit).

## Design decisions worth knowing

**Query budgets above the pool size.** The main sweep asks for Q = 200,000 against a 50,000-image
CIFAR-100 pool. Re-querying an identical image against a deterministic victim buys nothing, so
repeat passes over the pool are randomly cropped/flipped, making every query a distinct point
(`extract.sample_pool`). This is standard KnockoffNets practice and it changes what `n_queries`
means, so it is stated here and in the module.

**`c_student` from actual sample-passes.** §5.2 defines `c_student` as
`F_train_per_sample × n_queries × n_epochs`. That is exact for `random`, but `jbda` and `best`
train repeatedly on a growing transfer set, and that compute is real. The recorded `c_student`
uses the measured number of sample-passes; the ratio to the nominal figure is recorded as
`extra.epochs_effective`.

**Two readings of "leverage" (§14.2 vs §14.3).** §14.2 defines leverage by solving
τ(R_v)·R_v = R for the victim to target, so R is the robustness the *attacker ends up with*.
§14.3 then calls R = 0.737 "interpolated, defensible" — which only holds if R is the *victim's*
robustness, since 0.737 is exactly the strongest measured victim. The two cannot both be true: at
a realistic τ ≈ 0.5, a student target of 0.737 needs a victim at R_v ≈ 1.4, which does not exist.
Table 1 therefore reports the victim-level reading (defined at all four targets, marked
`mode: direct`) as the headline, with the §14.2 solve alongside every row — including "unreachable"
where the linear τ has no solution in (0, 1]. The code refuses to extrapolate τ past R_v = 1
rather than printing a number.

**Attacker cost of a defense (§11.4).** `C_att(d)` is the cost to reach the same student AA as
under `none`. Only one budget per defense is run, so the required cost is projected along the
measured AA-vs-log₁₀(cost) slope from the undefended budget sweep for that victim, censored at
10× the largest measured cost. The assumption is stated in every `defenses.jsonl` record.

**Measured vs heuristic FLOPs (§5.3), and an unexpected validation.** Running the required
cross-check (`python -m src.flops`) produced a result worth reporting in the paper:

| model | params | fwd FLOPs | FLOPs/param | vs K_adv/27 = 289.7 |
|---|---|---|---|---|
| `resnet18_cifar` (student) | 11,173,962 | 1.11e9 | 99.4 | **0.34×** |
| `wrn_16_4` | 2,748,890 | 7.85e8 | 285.7 | 0.99× |
| WRN-28-10 | 36,479,194 | 1.05e10 | 287.5 | 0.99× |
| WRN-82-8 | 79,125,466 | 2.30e10 | 291.0 | 1.00× |
| WRN-70-16 | 266,796,506 | 7.75e10 | 290.6 | 1.00× |
| WRN-94-16 | 365,915,610 | 1.07e11 | 291.1 | 1.00× |

K_adv = 7822 was fitted empirically over 9000+ adversarial-training observations, and
FWD_ADV = 27 comes from Appendix B. Their ratio predicts *measured* WideResNet forward cost to
within 0.5% across two orders of magnitude of parameter count — an independent confirmation of
both constants, and evidence that the defender and attacker conventions are mutually consistent
*for the victims*. The CIFAR ResNet-18 student sits at 0.34× because it downsamples earlier, which
is exactly why the attacker side must use measured counts (pitfall 6). Both numbers are in every
record. A regression test pins the WRN agreement.

**Determinism is not free.** Two runs with the same seed produce bit-identical students only
under the §15 settings (`use_deterministic_algorithms`, `cudnn.benchmark=False`,
`CUBLAS_WORKSPACE_CONFIG`). Without them the students drift, which would put noise straight into
τ (pitfall 2). `setup_determinism()` is called by every entry point and by the test suite; a
regression test asserts bit-identical students across repeated extractions.

## Tests

```bash
python -m pytest -q                       # all
python -m pytest tests/test_costmodel.py  # T1.1–T1.9, no GPU, no torch needed
```

`tests/test_extract.py` runs the whole attacker pipeline against a synthetic victim, so it needs
neither a RobustBench download nor AutoAttack.
