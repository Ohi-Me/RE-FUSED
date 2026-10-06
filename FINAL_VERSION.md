# Where the final version is

The two numbered study folders are the **final, best version** of the work. Everything reported in
[`docs/RESULTS.md`](docs/RESULTS.md) was produced by the code listed here, on the H100 cluster, and the log of every
run is kept. Older versions are in [`3_early_versions/`](3_early_versions/).

Inside the study folders, `dev` means development runs (early data only) and `confirm` means the final scored run
on held-out data. **For final numbers, always read the `confirm` folders.**

## 1. India data, forecasting and scheduling — `1_india_data_forecasting_scheduling/`

Paths below are inside this folder. The whole pipeline is run by `run_all.py`; each stage writes a record (time,
PBS job, GPU instance) to `logs/done/<stage>.json` and its output to `logs/stages/<stage>.log`. The final scored
run was H100 job 31823 (18–20 September 2026).

| Part | Final code | Final results | Summary to read first |
|---|---|---|---|
| Download official reports | `codes/scripts/acquire/` (`grid_india.py`, `npp.py`, `cea.py`, `imd.py`) | `data/raw/*/MANIFEST.csv` (files themselves not redistributed) | `data/docs/SOURCE_REGISTRY.md` |
| State-day panel | `codes/refused/parse_*.py`, `codes/scripts/build/b01_psp.py` … `b08_validation.py` | `data/processed/refused_state_day.parquet`, `CHECKSUM.txt` | `data/docs/DATASHEET.md`, `QC_REPORT.md`, `results/rebuild/VERIFY.md` |
| All-India 15-minute and hourly series | `codes/scripts/build/b09_psp_timeseries.py`, `b10_hourly_prices.py` | `data/processed/refused_allindia_15min.parquet`, `refused_allindia_hourly.parquet`, `refused_allindia_hourly_prices.parquet` | `data/docs/ALLINDIA_15MIN_QC.md` |
| No-leakage check | `codes/tests/test_leakage.py` | — | `logs/stages/c_leakage.log` |
| Forecasting models | `codes/scripts/o2/` (LightGBM, PatchTST, TFT, BiLSTM, Chronos-Bolt) and `codes/scripts/dev2/d2_01`–`d2_03` (XGBoost, N-HiTS, TiDE, BiTCN, NBEATSx, Chronos-2) | `results/confirm2/o2/preds/` | — |
| **Final forecaster** (combination per target) | `codes/scripts/dev2/d2_05_ensemble.py`, evaluated by `d2_06_o2_eval.py` | `results/confirm2/o2/eval/` (`point.csv`, `prob.csv`, `selection_dev2.json`) | `results/tables/confirm2_o2_summary.md` |
| Context-gated fusion (LA-MCAG) | `codes/scripts/o3/o3_01_lamcag.py`, `codes/scripts/dev2/d2_07_o3_eval.py` | `results/confirm2/o3/` | `results/tables/confirm2_o3_summary.md` |
| Deviation assessment | `codes/scripts/o4/o4_01_assessment.py`, `codes/scripts/dev2/d2_08_o4.py` | `results/confirm2/o4/` | `results/tables/confirm2_o4_summary.md` |
| Scheduling rules and the learned PPO policy | `codes/scripts/o5/o5_01_scheduling.py`, `codes/scripts/dev2/d2_09_o5.py`, `d2_11_ppo.py` | `results/confirm2/o5/` (`metrics.csv`, `chain_tests.csv`, `ppo_metrics.csv`) | `results/tables/confirm2_o5_summary.md`, `confirm2_o5_ppo_summary.md` |
| **Scoring of all 52 pre-registered tests** | `codes/scripts/confirm/c01_hypotheses.py` | `results/confirm2/hypotheses.csv` | `results/tables/confirm_hypotheses.md` |
| Integrity check (hashes, rescoring) | `codes/scripts/audit/a02_integrity.py` | `results/audit/` | `results/audit/INTEGRITY.md` |
| Additional analyses (after scoring) | `codes/scripts/extra/x01`, `x02`, `x04`, `x05` | `results/extra/` | `docs/RESULTS.md` |
| Plan frozen before scoring | — | `codes/design/02_preregistration.md`, `PREREG_SHA256.txt` | `codes/design/07_confirm_deviations.md` |

`results/confirm/` holds the first-round models scored on the same held-out period; the final combinations in
`results/confirm2/` were built on top of them. `results/dev/` and `results/dev2/` are the development rounds.

## 2. Forecast correction on Indian data — `2_forecast_correction_india/`

**What "gated correction" means:** take a day-ahead forecast, learn its typical errors from past real data, and add
back only a share of that learned correction. The *gate* is that share. It can be one number for everything, or a
separate number per series, per hour or per day. This study asks when a finer gate helps on real Indian data.

| Part | Final code | Final results | Summary to read first |
|---|---|---|---|
| The method | `04_code/refused_gate/` (gates, ladders, PART and prequential selection, scoring, statistics, data loaders) | — | `README.md` |
| Safety lock on held-out data | `04_code/refused_gate/guard.py` | — | `02_design/05_preregistration_india.md` |
| Learned forecasts to be corrected | `04_code/scripts/dev/make_learned_baselines_india.py` | `03_data/processed/india_hourly_learned_baselines_confirm_s0.parquet` | — |
| **All-India hourly study** (pre-registered) | `04_code/scripts/india/run_india_ladders.py`, `india_prob_energy.py`, `x1_india.py`, `analyze_india.py`; job `hpc/india_confirm.pbs` | `06_results/india/confirm/` | **`06_results/india/confirm/analysis/verdicts.json`** |
| Figure | `04_code/scripts/india/fig_india.py` | `07_figures/india/correction_skill_hourly.png` | — |
| State daily ladders (exploratory) | `04_code/scripts/india/run_india_ladders.py --mode explore`; job `hpc/india_daily.pbs` | `06_results/india/explore/` | — |
| Changes after the plan was frozen | — | — | `02_design/06_deviations_india.md` |

Logs: `12_logs/india_confirm.out` (final run, H100 job 34223) and the earlier stopped attempt
`12_logs/india_confirm_34214.out`.

`comparison_other_countries/` holds the first test of the same method on NYISO, European and UCI data. It is a
comparison only; its final results are in `comparison_other_countries/06_results/confirm/`.

## Note on folder names

The folders were given clearer names for this repository. Run logs and the frozen plan files keep the working names
used when they were written: `india_grid` (now `1_india_data_forecasting_scheduling`), `gated_correction` (now
`2_forecast_correction_india`), `foreign_data_study` (now `comparison_other_countries`). Paths inside the code use
the new names.
