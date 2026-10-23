# Exploratory analyses for the ECM revision (written 23 October 2026, before any of them was run)

**Status: exploratory, not pre-registered.** All of these analyses were added after the confirmatory scoring of the
all-India hourly study (tag `prereg-india-v1`) and of the State panel (India programme, tag `prereg-v1`). Their test
periods have been opened before. They are reported as exploratory checks, labelled as such in the manuscript, and
none of them changes a pre-registered verdict. Every setting below is fixed in this file, which is committed before
the job runs; no setting is chosen by looking at a test-period result. Every result is reported, whatever its sign.

Script: `code/scripts/india/explore_ecm_revision.py`; job: `hpc/explore_ecm.pbs` (CPU queue, no GPU).
Output: `results/india/explore_ecm/<block>/` (never overwritten). State-panel part: `India_Grid_Study`
`codes/scripts/extra/x12_gate_energy_blocks.py` -> `results/extra/gate_energy_blocks.csv`.

Common settings: confirmatory split of the hourly study (corrector training < 1 Sep 2025, gate fitting 1 Sep 2025 -
28 Feb 2026, test 1 Mar - 13 Sep 2026); information up to the end of day D-2 for target day D; the eight day-ahead
forecasts and four series of the study; seeds 0-2 where a corrector is fitted; skill = 1 - MSE(corrected)/MSE(base)
per series on the test rows of the confirmatory ladder (same rows for every method); intervals from a moving-block
bootstrap over days (7-day blocks, 2,000 resamples, seed 0) unless stated.

## R1 Conventional residual corrections (benchmarks)
Each uses only the base forecast's own past errors E = Y - B up to day D-2:
* `bc_hod28`  hour-of-day mean of E over the last 28 days (at least 14);
* `bc_ewma`   hour-of-day exponentially weighted mean of E, daily factor 0.9;
* `bc_how8`   hour-of-week mean of E over the last 8 same weekdays (at least 4);
* `bc_how_ar` `bc_how8` plus an autoregressive remainder (Moebius et al., 2023): u = E - bc_how8, least squares of
  u_t on u_{t-48}, u_{t-72}, u_{t-168} over the last 56 days, refitted daily;
* `bc_linreg` linear error regression: E_t on an intercept, E_{t-48}, E_{t-72}, E_{t-168}, the 168-h mean of E ending
  at t-48 and hour-of-day dummies, least squares over the last 90 days, refitted daily.
Where a method lacks history the base forecast is used (counted). The method used in the head-to-head comparison is
the one with the lowest MSE on the gate-fitting months (validation-selected), per forecast and series.
`R1b`: the validation-selected conventional correction as a new base forecast, followed by the static per-series and
the daily-updated per-series gate (same ladder, 3 seeds), to test whether the learned gated correction adds to it.

## R2 Error-aware corrector
The ladder rerun (3 seeds) with the corrector features extended by the base forecast and its lagged errors (E at
48, 72, 168 h; 168-h mean and standard deviation of E ending at 48 h), i.e. `data._hourly_features(fc_col=B)`.

## R3 Refreshed LightGBM base forecast
The same LightGBM model and features as the study (`make_learned_baselines.lgbm`, seed 0), refitted at the start of
every month from May 2025 to September 2026 on data up to the end of the day two days before the month starts, and
used for that month. Then the ladder (3 seeds) and the energy valuation of the study (deviation energy and stylised
imbalance cost of all-India demand) on this forecast.

## R4 Rolling-origin check of the selection rules
For each test month m (March - September 2026) the gates are fitted on the 182 days ending two days before m starts
(never before 1 September 2025, the end of corrector training), and PART (interleaved, K = 6) predicts the gain of
global -> per-series, per-series -> series x 4-hour block and per-series -> series x hour; the realised gain is
measured on month m. The same is done with the fixed validation window of the study. Prequential validation of the
time policies (rolling 30/90/180 days, forgetting 0.98/0.995) is scored the same way against their realised gain over
the expanding gate in month m. Reported: sign accuracy by month and pooled, rolling versus fixed window (seed-averaged
cells). Per-instance refinement is not included (cost).

## R5 Reserve recalibration
(a) Monthly coverage of the stored ex-ante 99 % quantile tables (all quantile designs of the study).
(b) Online quantile tracking with the delay of the study: reserve = stored series x hour quantile + offset per series,
offset_D = offset_{D-1} + eta * S_s * sum over the hours of day D-2 of (1{e > reserve} - 0.01), offset_0 = 0, S_s the
mean of the series' ex-ante quantile table. eta from {0.005, 0.01, 0.02, 0.05, 0.1}, chosen by the lowest 0.99 pinball
loss on the development test months (1 Dec 2025 - 28 Feb 2026, development results), then applied unchanged to the
confirmatory test months, for the uncorrected and the corrected LightGBM forecast. Reported: coverage, mean reserve,
pinball loss, shortfall energy.

## R6 Imbalance-cost sensitivity (all-India demand, LightGBM, static per-series gate; stored forecasts)
(a) the pre-registered stylised two-settlement cost; (b) the same, annualised over priced days only; (c) deviation
charged at the day-ahead price, |Y - F| P_DAM; (d) a DSM-style buyer rule without volume tiers or frequency: over-
drawal (Y > F) at lambda = max(P_DAM, P_RTM), under-drawal credited at 0.9 lambda, relative to buying Y day-ahead;
(e) month by month; (f) decomposition of (a) into the mean correction times the mean price spread and the remainder.

## R7 Block-length sensitivity
The 95 % intervals of the hourly energy test (deviation energy) and of the imbalance estimate, and of every cell of
the State-panel energy table, with moving blocks of 7, 14 and 28 days and a stationary bootstrap (mean block 7 days).

## R8 Semi-synthetic test on Indian data, figure
The stored results of the pre-registered semi-synthetic test rebuilt on the 34 State control areas
(`results/india/confirm/x1`) drawn as the phase diagram of the comparison study (no new computation).
