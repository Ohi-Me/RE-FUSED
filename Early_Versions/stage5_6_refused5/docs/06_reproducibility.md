# RE-FUSED-5 reproducibility record

## Environment (measured, not assumed)
| Item | Value |
|---|---|
| Interpreter | Python 3.10.11 (`py -3.10`); note 3.11 on this machine is CPU-only and lacks pandas |
| Torch | 2.5.1+cu121, CUDA 12.1 available |
| GPU | NVIDIA GTX 1660 Ti Max-Q, 6 GB, driver 610.88 |
| CPU / RAM | 8 logical cores / 15.7 GB (memory is the binding constraint: an 11M-row × 16-feature float64 frame needs 1.3 GB per copy) |
| Key libraries | scikit-learn, xgboost, lightgbm, statsmodels, statsforecast, utilsforecast, pyarrow, properscoring, openpyxl |
| Thread caps used | `OMP_NUM_THREADS=4` for chained runs (uncapped runs caused Win32 error 1450 resource exhaustion) |

## Determinism and seeds
* Every model takes an explicit `random_state`; the 10-seed comparison (R5) uses seeds 0–9.
* Synthetic experiments fix the data-generating process **once per seed** and sample train/val/test from
  it. An earlier version drew a fresh β per split — that bug made every strategy look harmful and was
  caught by the `global`-gate sanity check. Lesson recorded because it is the single most dangerous
  class of synthetic-experiment error.
* No experiment selects the best seed or checkpoint; all report mean ± sd over seeds.

## Data provenance
| Dataset | Source on this machine | Derivation |
|---|---|---|
| India daily generation | `Data_Preprocessing/dataset/IDP/daily-power-generation.csv` (2.58M unit-day rows, 2017-09→2025-04) | `src/build_india_panel.py` → `src/build_model_data_india.py`; balanced-station panel (799 units present ≥95% of 2,657 days), 20 states |
| India context blocks | IDP renewables / coal-stock / outage CSVs | lagged one day before use; coverage starts 2019-10 (renewables) and 2018-11 (coal) |
| India raw prices | `Data_Preprocessing/dataset/prices of energy/*/*.csv` (289,667 IEX 15-min blocks) | audited only; **not used as a target** (see audit A1–A4) |
| NYISO prices | `http://mis.nyiso.com/public/csv/{realtime,damlbmp}/…` monthly ZIPs, 84 months each | `src/fetch_nyiso.py` → `src/build_nyiso_multires.py`, resampled to 5min/15min/1h/1d |

## Splits (identical for every model and strategy)
* India: train 2017-09-08→2022-12-31, val 2023 (gate fitting only), test 2024-01-01→2025-04-22.
* NYISO: train <2024, val 2024, test 2025.
* The test set is touched exactly once per configuration, for scoring. Gates, cell gates, shrinkage
  weights and deployment thresholds are all estimated on validation.

## Hyperparameters
Fixed a priori, not tuned (deliberately conservative; stated as a limitation): corrector
`HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6)`; gate models
`max_iter=200, learning_rate=0.07, max_depth=5`; variance-estimate halves `max_iter=150`.
A documented search over lookback, depth, learning rate and regularisation on validation only is
outstanding work.

## How to reproduce
```
py -3.10 src/audit_price_provenance.py        # A1  data audit of RE-FUSED-4
py -3.10 src/audit_raw_price_quality.py       # A2
py -3.10 src/audit_raw_original.py            # A3
py -3.10 src/audit_nan_pattern.py             # A4
py -3.10 src/build_india_panel.py             # D1  raw rebuild with provenance flags
py -3.10 src/build_model_data_india.py        # D2  balanced panel + baseline ladder
py -3.10 src/fetch_nyiso.py                   # D3a 84 months x 2 products (resumable)
py -3.10 src/build_nyiso_multires.py          # D3b four resolutions
py -3.10 src/synth_law.py                     # S1  shrinkage law
py -3.10 src/synth_conditional_v3.py          # S2c per-instance vs static (falsification F2)
py -3.10 src/synth_granularity.py             # S3  granularity sweep (checkpoints per cell)
py -3.10 src/real_core_india.py               # R1  real-data strategies
py -3.10 src/voc_india.py                     # R2  value of context + permutation control
py -3.10 src/resolution_law_nyiso.py          # R3  resolution law
py -3.10 src/risk_control.py                  # R4  block-LCB deployment
py -3.10 src/core_multiseed.py                # R5  10 seeds + DM tests
py -3.10 src/criterion_test.py                # R7  core claim: is the criterion predictive?
```
Run the chain sequentially with `OMP_NUM_THREADS=4`; concurrent runs exhaust memory on this machine.

## Known gaps in the current package (honest list)
1. No public-benchmark replication yet (ETT / electricity / traffic).
2. No foundation-model or modern deep baseline (Chronos, PatchTST, N-HiTS, TiDE) yet.
3. Variance of the conditional gate estimated from a single split-half; should be repeated splits.
4. Confidence intervals are across states/seeds, not block-bootstrapped over time.
5. No formal excess-risk theorem yet — the shrinkage result is a proposition plus empirics.
6. Probabilistic/calibration track (CRPS, PIT, conformal intervals) not yet run on real data.
