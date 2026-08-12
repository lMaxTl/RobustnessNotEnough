# results/

Written by the runners; read by `src/analysis.py` and `src/figures.py`.

| File | Deliverable | Written by | Contents |
|---|---|---|---|
| `results.jsonl` | D2 | `scripts/run_sweep.py` | one record per extraction run, schema in spec.md §12 |
| `margins.jsonl` | D3 | `scripts/run_sweep.py --config configs/e3_margins.yaml` | per-victim boundary geometry, 2,000 margins each |
| `defenses.jsonl` | D4 | `scripts/make_figures.py` | per-defense attacker-cost multiplier and utility cost |
| `victim_cache.json` | — | `scripts/verify_victims.py` | measured clean/AA accuracy per victim (the τ denominators) |
| `analysis.json` | — | `scripts/make_figures.py` | τ fits, leverage table, F1–F3 checks |
| `figures/` | D5–D8 | `scripts/make_figures.py` | fig1–3 (pdf+png), `table1_leverage.md` |

`results.jsonl`, `margins.jsonl` and `defenses.jsonl` are append-only and flushed per record, so
an interrupted sweep loses at most one run. They are the paper's evidence and belong in version
control; `analysis.json` and `figures/` are derived and are not.

These files are empty until the experiments run — see [../REPRO.md](../REPRO.md) for the
commands and the runtime budget. Nothing here is pre-filled with placeholder numbers.
