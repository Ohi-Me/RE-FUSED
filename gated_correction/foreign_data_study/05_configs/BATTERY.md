# Experiment battery — configurations

Machine-readable configurations of every ladder run are exported from the run records by
`04_code/scripts/export_configs.py`. This file lists the non-ladder experiments and the fixed constants.

## Confirmatory (pre-registered, tag `prereg-v1`; command order in `04_code/scripts/confirm/chain_confirm.sh`)

| Block | Script | Data / baselines | Varied | Seeds | Runs |
|---|---|---|---|---|---|
| Learned baselines | `dev/make_learned_baselines.py --mode confirm --models lgbm,chronos,neural` | NYISO; NHITS, PatchTST, TiDE, DLinear (h 48, input 336 h, max_steps 500), LightGBM (800 trees, lr 0.05, 63 leaves, lag ≥ 48 h), Chronos-Bolt-Small zero-shot (context 2048 h); fit A train < 2023 → 2023–24, fit B train < 2025 → 2025–26 | — | 0 | 1 |
| C1 | `confirm/run_confirmatory.py` | NYISO 2026; iso, snaive168, lag48, lgbm, chronos, nhits, patchtst, tide, dlinear | full ladder | 3 | 27 |
| C2 | same | NYISO 2026; ISO | gate-fit 10/25/50 % of validation days; OOF train + validation | 3 | 12 |
| C3 / C4 | same | OPSD 2019 / 2020; tso, snaive168, lag48 | full ladder | 2 | 6 + 6 |
| C5 | same | OPSD 2019; TSO | 10/25 %; OOF + validation | 2 | 6 |
| C6 | same | UCI 2014, 60 clients (seed 7); snaive168, lag48 | full ladder | 2 | 4 |
| C7 | same | UCI 2014; snaive168 | 10/25 %; OOF + validation | 2 | 6 |
| X5/X8 | `dev/x5_x8_prob_energy.py --mode confirm --seeds 3` | NYISO 2026; iso, corr_series, corr_instance, corr_roll30 × 7 quantile models; 23 levels | — | 3 | 3 |
| X1 partition | `dev/x1_phase_diagram.py --part partition --source opsd_confirm --seed_offset 900 --tag confirm` | 12-area OPSD substrate; axis {unit, state} × m {0, 0.5, 1, 2, 4, 8} × φ {0, π/8, π/4, 3π/8, π/2, 3π/4} × κ {1, 2} × fraction {0.25, 1} × 2 seeds | grid | 2 | 576 |
| X1 instance | same with `--part instance` | m {0, 1, 4} × φ {0, π/4, π/2} × κ {1, 2} × fraction {0.25, 1} × 2 seeds | grid | 2 | 72 |
| Analysis | `confirm/analyze_confirmatory.py --map confirm` | H1–H8 | — | — | — |

The "full" level of the data-budget ladders (C2, C5, C7) is the validation-year fit of C1, C3 and C6.

## Sensitivity DC1 (after unblinding; `02_design/04_deviations.md`)

`confirm/run_sensitivity_dc1.py` re-runs C3, C4, C5 with `opsd_load(..., clean=True)` (18 runs), then
`analyze_confirmatory.py --map sensitivity`. Exploratory: `confirm/exploratory_imbalance.py`.

## Fixed constants (frozen code)

| Constant | Value | Location |
|---|---|---|
| Corrector | HistGradientBoosting, 300 iterations, lr 0.06, depth 6, ℓ2 1.0, ≤ 300 000 rows | `refused_gate/ladder.py` LadderConfig |
| Per-instance gate | 3 bagged WLS HGB models, 150 iterations, lr 0.08, depth 5 | same |
| Gate clip | [0, 1.5]; minimum cell 30 | `refused_gate/gates.py`, LadderConfig |
| Time policies | expanding; rolling 30/90/180 days; forgetting 0.98/0.995; Fixed Share α 0.02; delay 2 days | LadderConfig |
| PART v2 | interleaved weekly blocks, K = 6 (partitions), K = 4 (instance, ≤ 20 000 evaluation rows) | `refused_gate/ladder.py` run_ladder |
| Bootstrap | moving blocks of 7 days, 2 000 resamples, p = (count + 1)/(2 000 + 1) | `refused_gate/stats.py` |
| Multiplicity | Holm within family; Benjamini–Yekutieli across | `confirm/analyze_confirmatory.py` |
| H6b margin / H8b tolerance | −0.002 pooled skill / ±0.005 coverage | same |
| Quantile levels | 0.05…0.95 (19) and 0.01, 0.025, 0.975, 0.99 | `refused_gate/probabilistic.py` |

## Development

`05_configs/dev_ladders.json` (138 ladder runs in `dev/ett`, `dev/india`, `dev/nyiso_baselines`, `dev/nyiso_budget`,
`dev/dev2_*`); other development experiments: `dev/x1_phase_diagram.py` (NYISO substrate), `dev/x3_capacity.py`,
`dev/x5_x8_prob_energy.py --mode dev`, `dev/eval_part_offline.py`; chains `dev/chain_dev1.sh`–`chain_dev3.sh`.
