# O2 development protocol (fixed before any model was fitted)

Version 1.0 — 14 Sep 2026. Development data only: train (target dates FY2018-19 … FY2022-23), validation (FY2023-24),
development test (FY2024-25). Reserved rows (≥ 1 Apr 2025) are removed by the data loader
(`refused.o2data.load_panel`), which calls the blinding guard if they are requested.

## 1. Targets

| ID | Column | Series | Source lag L (days) | Retention rule (train only) |
|---|---|---|---|---|
| T1 | `dem_energy_met_gwh` | 34 entities | 1 | all entities |
| T2 | `dev_actual_drawal_gwh` | 34 entities | 1 | all entities (input to O4 U and O5) |
| T3 | `gen_act_gwh_total` | entities | 2 | train mean ≥ 1 GWh/day |
| T4 | `re_all_total_re_gwh` | entities | 2 | train mean ≥ 1 GWh/day (dates to 18 Nov 2025) |
| T5 | `prc_dam_acp` | 13 DSM bid areas containing Grid-India State entities | 14 | all 13 |

## 2. Information set

Issue time is 12:00 IST on day τ, for target days τ+1, τ+2, τ+3 (horizon H = 1, 2, 3). A column with availability
lag L may be used up to date τ − L (`data/docs/FIELD_REGISTRY.csv`). Target-day calendar variables are known. No
future weather is used. History window: 14 days per source.

Sensitivity **S-lat** ("instant publication") sets L = 0 for every source (history ends on the issue day), for the
LightGBM model only. It measures the cost of publication latency.

## 3. Models and tuning budget

| Model | Implementation | Inputs | Tuning budget (fixed) | Seeds |
|---|---|---|---|---|
| Persistence, seasonal naive (same weekday, last available week), MA7 | own code | target history | none | – |
| LightGBM quantile (τ = 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95), direct per H | lightgbm 4.7 | 14-day lags, rolling statistics, exogenous blocks, calendar, entity | num_leaves ∈ {31, 63}, learning rate 0.05, early stopping on the last 180 days of train (≤ 2,000 rounds); choice by validation pinball | 5 (bagging 0.8) |
| BiLSTM quantile | PyTorch, own code | 14-day multichannel window, entity embedding, target calendar | hidden 64, 2 layers, dropout 0.2, Adam 1e-3, batch 256, ≤ 60 epochs, early stopping (patience 6) on the last 180 days of train | 5 |
| TFT | neuralforecast 3.2.2 | target, historical exogenous (lag-aligned), calendar as future exogenous | library defaults except input 14, hidden 64, MQLoss(7 quantiles), ≤ 1,500 steps, early stopping patience 5 checks on 180 days | 5 |
| PatchTST | neuralforecast 3.2.2 | target only (channel-independent) | input 14, patch 7, stride 3, hidden 64, 3 layers, MQLoss, ≤ 1,500 steps, early stopping as TFT | 5 |
| Chronos-Bolt-Small | chronos 2.3.2, local weights | target history (up to 512 days) | zero-shot | – |

Neural and LightGBM models are trained once on train (the last 180 train days used only for early stopping). They
predict validation and development-test days without refitting. Seed predictions are averaged (quantile-wise) before
evaluation. Selection per target is by validation MASE among non-anchor models.

## 4. Uncertainty

Native quantiles, plus conformalised quantiles calibrated on validation residuals and applied to the development test:

* **C0** global (target × H);
* **C1** per series (target × H × series);
* **C2** per series × season (winter Dec–Feb, summer Mar–May, monsoon Jun–Sep, post-monsoon Oct–Nov).

For models with quantile heads, conformalised quantile regression (CQR) is used. Chronos-Bolt and the anchors use
residual conformal around the median. Cells with fewer than 30 calibration residuals fall back to the parent level.
The adaptivity level is chosen per target and model on validation only: calibrate on even ISO weeks of FY2023-24 and
score pinball loss on odd weeks (and the reverse, averaged), so every season appears in both parts.

## 5. Metrics and tests

* **Point:** MASE and RMSSE. The scale is the train-period mean absolute (squared) error of the seasonal naive at the
  same H and lag, per series, so MASE < 1 means beating the seasonal naive.
* **Probabilistic:** pinball loss (7 levels), CRPS approximation from quantiles, and empirical coverage of the 80% and
  90% central intervals.
* **Tests:** per series, DM with HLN correction on absolute-error differentials against the seasonal naive. Pooled
  across series: the daily cross-series mean of scaled loss differentials, DM-HAC and 7-day block-bootstrap 95% CI.
  Holm correction within the target family.
* **Development hypotheses:** D1 (selected model beats the seasonal naive, p < 0.05, 5 seeds) and D2 (coverage within
  ±5 pp; C1 vs C0; C2 only where season heterogeneity persists) are assessed on the development test and reported
  as development evidence. The confirmatory versions are fixed in the pre-registration.

## Change log

| Date | Change | Reason |
|---|---|---|
| 14 Sep 2026 | v1.0 | before any model fit |
| 14 Sep 2026 | T5 series count 14 → 13 | only 13 DSM bid areas contain Grid-India State entities (found when building samples, before fitting) |
| 14 Sep 2026 | v1.1: calibration-level selection by alternating ISO weeks instead of first/second half of FY2023-24 | halves are season-disjoint (Apr–Sep vs Oct–Mar), so level C2 could never be scored; changed before any calibration result was computed |
| 14 Sep 2026 | v1.2: rolling conformal variants added (trailing 180 days of published residuals at issue time, 90-day sensitivity, at the level chosen on validation); relative MAE vs seasonal naive on the same split reported | development observation on T5 (price): static calibration from FY2023-24 over-covered FY2024-25 (80 % intervals covering ≈ 96–100 %) because validation-year errors were ≈ 2× larger. Added after seeing that development result; the reserved period remains untouched and the final calibration rule will be fixed in the pre-registration |
| 15 Sep 2026 | v1.3: BiLSTM input/target normalisation changed from the series' train mean to a window base (mean of the last 7 available target values, as for LightGBM), scaled by the train std; exogenous columns split into window-centred and window-level channels in LA-MCAG (O3) | O3 smoke test (T5, one seed) with train-mean normalisation beat the seasonal naive on validation (MAE 1,326 vs 1,623) but failed on the development test (1,451 vs 832): level shifts between years cannot be followed. Changed before any BiLSTM result; observed on development data (not reserved) and logged here |
