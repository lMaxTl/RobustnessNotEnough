# REPRO — exact commands, environment, wall-clock

Deliverable D9 (spec.md §2). Everything here is measured on the machine below or is a
spec.md §16 planning figure, explicitly labelled as such.

## 1. Hardware and software actually used

```
python                   3.11.0
platform                 Windows-10-10.0.26200-SP0
torch                    2.13.0+cu126
torch_cuda               12.6
cudnn                    91002
gpu                      NVIDIA GeForce RTX 4060
gpu_count                1
gpu_total_mem_gb         8.0
cublas_workspace_config  :4096:8
torchvision              0.28.0+cu126
numpy                    2.4.6
scipy                    1.17.1
robustbench              1.1        (installed from master; see §2)
timm                     1.0.28
autoattack               0.1        (git+https://github.com/fra31/auto-attack)
```

Regenerate this block at any time with:

```bash
python -m src.repro
```

**This is not the spec's reference machine.** spec.md §16 budgets ~300 GPU-hours on an A100-40GB.
An RTX 4060 (8 GB, ~1/4 the throughput and 1/5 the memory) will take substantially longer, and
the largest victims (WRN-94-16, 366M parameters) need small evaluation batch sizes to fit. Use
`--batch-size 32` for `verify_victims.py` and `batch_size: 32` in the margins config if you hit
OOM. No result in this repository depends on batch size; only wall-clock does.

## 2. Dependency deviations from spec.md §15

| Spec | Used | Why |
|---|---|---|
| `robustbench @ git+…@v1.1` | `robustbench @ git+…@master` (self-reports 1.1) | The `v1.1` **tag** ships 61 CIFAR-10 L∞ models and is missing `Wang2023Better_WRN-28-10`, `Wang2023Better_WRN-70-16`, `Peng2023Robust`, `Bartoldson2024Adversarial_WRN-82-8`, `Bartoldson2024Adversarial_WRN-94-16` — five of the eight victims in §6.1. Master has all eight (76 models). |
| — | `timm==1.0.28`, `gdown==5.1.0`, `geotorch`, `torchdiffeq` | Transitive imports of robustbench's model zoo (`xcit`, `sodef_layers`). Pinned because timm renamed `XCiT`→`Xcit` across versions and robustbench master expects the new name. |

Verify the victim keys resolve before spending GPU time:

```bash
python -c "from robustbench.model_zoo.cifar10 import linf; \
from src.victims import VICTIMS; print(all(k in linf for k in VICTIMS))"
```

## 3. Determinism

spec.md §15 settings are applied by `src.repro.setup_determinism()`, which every entry point and
the test suite call before touching CUDA:

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

**Measured, not assumed:** without these settings, two extractions with identical seeds and an
identical transfer set produced *different* students on this GPU (verified during development).
With them, `tests/test_extract.py::test_extraction_is_deterministic_for_a_seed` asserts
bit-identical student parameters across repeated runs. Seed-level reproducibility on the same
hardware therefore holds; bit-reproducibility across different GPUs is not expected.

## 4. Commands, in order

```bash
# 0. M1 must pass before any GPU work (spec.md §18)
python -m pytest tests/test_costmodel.py -q
python -m src.costmodel                    # emits the §4.4 reference table

# 1. Verify every victim's clean and AutoAttack accuracy yourself (§6.1).
#    Writes results/victim_cache.json, which supplies the tau denominators.
python scripts/verify_victims.py --n-eval 1000 --batch-size 32
python scripts/verify_victims.py --n-eval 10000 --batch-size 32   # for the paper

# 2. E2 main sweep (§13.1) — 192 runs at n_eval=1000, plus n_eval=10000 on the Q=200k cell
python scripts/run_sweep.py --config configs/e2_main.yaml

# 3. E2b ablations (§13.2), E2c temperature control (§13.3)
python scripts/run_sweep.py --config configs/e2b_ablations.yaml
python scripts/run_sweep.py --config configs/e2c_control.yaml

# 4. E3 margins (§13.4) — writes results/margins.jsonl
python scripts/run_sweep.py --config configs/e3_margins.yaml

# 5. E4 defenses (§13.5) and the watermarked victim (§11.3)
python scripts/run_sweep.py --config configs/e4_defenses.yaml
python scripts/train_watermark_victim.py --epochs 100
python scripts/run_sweep.py --config configs/e4_watermark.yaml

# 6. E5 analysis, D4–D8
python scripts/make_figures.py --n-eval 10000 --attack best
```

Add `--dry-run` to any `run_sweep.py` call to print the pending grid without running it.
All runners are resumable: completed runs are keyed by
`(victim, attack, pool, budget, student, seed, defense, epochs, n_eval)` and skipped.

Tiny Tiny Tiny wiring checks that cost minutes rather than hours:

```bash
python scripts/run_sweep.py --config configs/e2_main.yaml --dry-run
python scripts/train_watermark_victim.py --epochs 2 --smoke
python -m pytest tests/test_extract.py -q       # full attacker pipeline, synthetic victim
```

## 5. Wall-clock

| Stage | Runs | spec.md §16 (A100) | Measured here (RTX 4060) |
|---|---|---|---|
| M1 cost model + tests | — | seconds | **0.15 s** (77 tests) |
| Full test suite (234 tests) | — | — | **~2 min** once CIFAR-10 is cached |
| Victim AA verification | 8 | ~6 h | not yet run |
| E2 main sweep (n_eval=1000) | 192 | ~160 h | not yet run |
| E2 final eval (n_eval=10000) | 48 | ~32 h | not yet run |
| E2b ablations | ~40 | ~33 h | not yet run |
| E2c control | 21 | ~18 h | not yet run |
| E3 margins | 8 | ~5 h | not yet run |
| E4 defenses | 54 | ~45 h | not yet run |
| Watermark victim training | 1 | ~8 h | **~46 h projected** (see below) |

The watermark projection is measured, not guessed: a 2-epoch smoke run over 2,000 images
(`--epochs 2 --smoke`) took 132 s, i.e. 33 ms per image-epoch for WRN-28-10 under TRADES with
PGD-10. The full recipe (50,000 images × 100 epochs) therefore projects to ~46 GPU-hours on this
RTX 4060, against the spec's ~8 h on an A100. Treat every other row of the right-hand column as
scaling similarly until measured.

Fill the right-hand column from `wallclock_sec` in `results/results.jsonl` as runs complete:

```bash
python -c "
import json,collections; from src.schema import read_jsonl, RESULTS_PATH
t=collections.Counter()
for r in read_jsonl(RESULTS_PATH): t[r['extra'].get('block','?')] += r['wallclock_sec']
print({k: round(v/3600,2) for k,v in t.items()}, 'GPU-hours')"
```

If constrained, cut the Q=5,000 cell and the `jbda` ablation first (spec.md §16).

## 6. Data

| Dataset | Size | Needed for | Present here |
|---|---|---|---|
| CIFAR-10 (test + train) | 170 MB | every evaluation; the `cifar10_train` pool | **yes** (downloaded, ~45 min on this link) |
| CIFAR-100 (train) | 169 MB | the **primary** query pool (§8.1) — E2, E2b, E2c, E4 | **yes** (downloaded, 2,163 s) |
| Tiny ImageNet-200 | 237 MB | secondary pool, E2b only | no |
| RobustBench checkpoints | ~3 GB total | all victims | no — fetched into `models/` on first `load_victim` |

CIFAR-10 and CIFAR-100 are both on disk, so E2/E2b/E2c/E4 can start without a network. Verified
after download: the pool is 50,000 uint8 images, and `sample_pool("cifar100", 1000)` returns
float16 in [0,1]. Tiny ImageNet (E2b's secondary pool only) and the RobustBench checkpoints are
still to be fetched.

Set `RE_DATA_ROOT` to relocate datasets. Set `RE_NO_DOWNLOAD=1` to forbid all dataset fetching —
anything missing then raises with instructions instead of pulling hundreds of MB mid-sweep. This
does not cover RobustBench checkpoint downloads, which robustbench manages itself.

## 7. What is not yet in this repository

No experiment has been run, so `results/results.jsonl`, `results/margins.jsonl`,
`results/defenses.jsonl`, the figures and Table 1 do not exist yet. Per spec.md §0 rule 1, no
placeholder numbers were written in their place. The analysis and figure code is exercised by
`tests/test_analysis.py` against synthetic records with a planted ground truth.
