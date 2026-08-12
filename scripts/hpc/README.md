# Running the full evaluation on HoreKa (NHR@KIT)

Batch scripts for the complete E2/E2b/E2c/E3/E4/E5 campaign of [spec.md](../../spec.md),
sized for HoreKa's **48-hour** maximum wall time on the GPU partitions.

Nothing here changes what is computed. The experiment grid, seeds, budgets and
evaluation protocol are exactly the configs in [configs/](../../configs); the only
thing that changes is that one config now runs as many independent jobs instead
of one long serial process.

## Prerequisites

HoreKa's `accelerated` partition is **shared**, minimum allocation `gres=gpu:1`,
maximum wall time `2-00:00:00`. Every job here asks for one A100 and a quarter
node. Nothing needs more than one GPU: the workload is 300+ *independent* runs,
so the parallelism is across jobs, not inside them.

Put the repository on a workspace rather than `$HOME` — it accumulates ~3 GB of
RobustBench checkpoints plus datasets, and the campaign writes into it:

```bash
ws_allocate robustness 60                 # 60 days, extendable 3x
cd "$(ws_find robustness)"
git clone <this repo> RobustnessNotEnough && cd RobustnessNotEnough
```

## Quick start

```bash
# --- login node -----------------------------------------------------------
bash scripts/hpc/00_setup_env.sh      # venv + torch/cu126 + robustbench@master
bash scripts/hpc/01_stage_data.sh     # CIFAR-10/100, Tiny ImageNet, 8 checkpoints

# --- submit ---------------------------------------------------------------
RE_ACCOUNT=<your-project> bash scripts/hpc/submit_all.sh

# --- watch ----------------------------------------------------------------
bash scripts/hpc/status.sh
```

`submit_all.sh --dry-run` prints the `sbatch` commands without submitting.

Compute nodes are treated as **offline**: `RE_NO_DOWNLOAD=1` is exported in every
job, so a dataset that stage 0 missed fails immediately instead of stalling an
A100 on a download. Stage 0 is the only place that touches the network.

## The stages

| # | Script | Array | Time limit | Est. per task | What |
|---|---|---|---|---|---|
| 1 | `10_verify_victims.sbatch` | 0–7 (1/victim) | 48 h | 2 h … **20 h** | clean + AutoAttack accuracy per victim (§6.1) |
| 2 | `11_merge_victim_cache.sbatch` | — | 10 min (cpuonly) | seconds | merge the 8 cache parts; **hard gate** |
| 3 | `20_e2_main.sbatch` | 0–31 (8×4) | 48 h | ~6–8 h | E2, 192 runs + 48 re-evaluations (§13.1) |
| 3 | `21_e2b_ablations.sbatch` | 0–7 (1/victim) | 36 h | ~2–10 h | E2b attack/pool/capacity ablations (§13.2) |
| 3 | `22_e2c_control.sbatch` | 0–6 (1/target) | 24 h | ~3 h | E2c temperature-matched control (§13.3) |
| 3 | `30_e3_margins.sbatch` | 0–7 (1/victim) | 24 h | ~0.2–6 h | E3 FAB-T boundary margins (§13.4) |
| 3 | `40_e4_defenses.sbatch` | 0–7 (2×4) | 24 h | ~6 h | E4 defenses, 48 runs (§13.5) |
| 4a | `41_train_watermark.sbatch` | — | 36 h | ~8–16 h | the one TRADES+backdoor victim (§11.3) |
| 4b | `42_e4_watermark.sbatch` | — | 24 h | ~7 h | E4b watermark survival, 6 runs |
| 5 | `50_make_figures.sbatch` | — | 1 h (cpuonly) | minutes | merge, E5 analysis, D4–D8, Table 1 (§14) |

Estimates are spec.md §16's A100 budget redistributed over the shards, weighted
by each victim's measured forward FLOPs where the victim dominates the cost.
They are planning figures, not measurements — the right-hand column of
[REPRO.md](../../REPRO.md) §5 is what gets filled in from `wallclock_sec` afterwards.

Everything in stage 3 runs concurrently, and stage 4a runs alongside it, so the
campaign's wall-clock is the slowest branch — roughly **2–3 days** on an empty
queue — not the ~400 GPU-hour total.

## One decision you should make deliberately

**Stage 1 defaults to `--n-eval 10000`, and that costs about ten times as much
as `--n-eval 1000`** (~60 GPU-hours instead of ~6, and it is what puts the
WRN-94-16 task at ~20 h).

The reason for the default: `victim_aa_acc` is the τ denominator of every single
record, it is frozen into each record when the run happens, and re-measuring it
later does not retroactively fix records already written. Verifying once, at the
resolution REPRO.md calls "for the paper", gives one consistent denominator
across the whole study. A 1,000-image AutoAttack estimate carries ~1.5 points of
standard error (pitfall 3) — in a denominator, on the differences between
victims that are 3 points apart, that is not a rounding detail.

If you would rather spend those hours on the sweep and accept the coarser
denominator:

```bash
RE_N_EVAL=1000 bash scripts/hpc/submit_all.sh
```

Do not verify some victims at 1,000 and others at 10,000 — `merge_victim_cache.py`
aborts on a mixed cache rather than let two denominators into one figure.

## Resuming, which you will need

Every stage is resumable and every array task owns its own output file:

- `results/shards/<block>_<victim>_s<shard>.jsonl` — one per task, merged later
- `results/margin_shards/e3_<victim>.jsonl` — margin records are ~40 kB per line

A task skips runs already recorded in its own shard *and* in the merged
`results/results.jsonl`, so re-submitting a task continues where it stopped:

```bash
sbatch --array=13 scripts/hpc/20_e2_main.sbatch       # just that task
sbatch --array=4-7 --partition=accelerated-h100 scripts/hpc/10_verify_victims.sbatch
```

Two things make that safe. First, sharding is applied to the **full** grid before
completed runs are subtracted, so shard 13 always owns the same six runs no
matter how many of them are already done. Second, jobs stop launching new runs
~45 minutes before the wall time (`--max-seconds`, derived from
`SLURM_JOB_END_TIME`) and exit 0, rather than being killed inside a 3-hour
Q=200,000 run. Records are fsynced per run either way, so the worst case is
losing the run in flight.

`status.sh` reports the queue, the verification cache, record counts per block,
what each config still has pending, and which logs contain a traceback or a
time-limit kill.

## How the work is split

Sharding is by victim first — one victim per task means the 366M-parameter
checkpoint loads once and its PGD-40 evaluation set is built once — then
round-robin over a **budget-descending** order within the victim. The natural
(victim, attack, budget, seed) order would hand all six Q=200,000 runs to two of
the four shards and none to the first; longest-processing-time-first spreads
them 2/2/1/1.

`prada` (E4) is the one defense whose cost is not on the GPU: it runs a
nearest-neighbour test over a growing query set on the CPU, so those three runs
carry roughly an extra half hour each.

## Changes this needed in the project code

Five additive changes; defaults are unchanged, and all 214 tests still pass.

| File | Change | Why |
|---|---|---|
| `scripts/run_sweep.py` | `--victims`, `--shard/--num-shards`, `--results`/`--margins`/`--done-also`, `--max-seconds` | split one config across jobs; stop cleanly before a walltime kill |
| `scripts/run_sweep.py` | `done_keys` reads `extra.matched_victim` | E2c records are written as `Standard_T3.42`, so matching on `victim_name` alone meant E2c could never resume and would duplicate its 21 runs on every resubmit |
| `src/victims.py` | `RE_VICTIM_CACHE` env override | `verify_victims.py` rewrites the cache wholesale after each victim; eight concurrent jobs on one path would keep only the last writer's measurements |
| `scripts/verify_victims.py` | tolerate victims with no leaderboard entry | it raised `KeyError` on the locally trained watermarked victim, which E4b has to verify |
| `scripts/hpc/*` | new | this directory |

## Troubleshooting

**`RuntimeError: cuDNN version incompatibility: PyTorch was compiled against
(9, 10, 2) but found runtime version (9, 5, 1)`**

HoreKa's default environment puts a system cuDNN on `LD_LIBRARY_PATH`, which the
dynamic loader searches *before* the RUNPATH pointing at the copy the torch
wheel ships — so torch loads the wrong one. `sbatch` exports the submitting
environment by default, so this reaches compute nodes too, not just the login
node.

`_bundled_cuda.sh` fixes it by putting the wheels' own NVIDIA libraries first,
and every script here sources it after activating the venv. Nothing to do but
pick up the change and re-run:

```bash
bash scripts/hpc/00_setup_env.sh          # pip steps are already satisfied; ~1 min
```

Check it by hand at any time:

```bash
source .venv-horeka/bin/activate
source scripts/hpc/_bundled_cuda.sh
python -c "import torch; print(torch.backends.cudnn.version())"   # 91002
```

If it somehow persists, something is putting a cuDNN ahead of the venv's:
`module list` to see what your login shell loads, then `module purge` before
submitting. Nothing in this project needs a module — `python3.11` is a system
binary and the torch wheels are self-contained.

## HoreKa specifics worth knowing

- `--account` is not set in the scripts; pass `RE_ACCOUNT=<project>` to
  `submit_all.sh`, or `sbatch --account=...` directly.
- `RE_PARTITION=accelerated-h100` (or `accelerated-h200`) redirects every GPU job
  to a faster partition — worth it for the two largest verification tasks if
  your project has the quota.
- The two CPU-only jobs use `cpuonly`, which allocates exclusively; they are
  short, and the alternative (burning an A100 to merge a JSON file) is worse.
- Datasets are read straight from the workspace. They are small enough
  (~600 MB, loaded into RAM once per job) that staging to `$TMPDIR` buys
  nothing here.
- `ws_extend robustness 60` before the workspace expires; results live inside it.
