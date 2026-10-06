# Pre-registration: adaptive forecast correction on Indian data (all-India hourly study)

Version prereg-india-v1, written on 6 October 2026, **before any model, forecast or result was computed on the test
months** (1 March 2026 onwards). When this file is frozen it is committed, its SHA-256 is stored in
`02_design/PREREG_INDIA_SHA256.txt` and the commit is tagged `prereg-india-v1`; the code
(`refused_gate.guard.require_india_confirmatory`) refuses to read the test months otherwise. Any change made after the
test months are opened is recorded in §9 with its reason.

**What had been seen before writing this.** (a) The development runs listed in §7, which use only data before
1 March 2026. (b) The quality-control report of the data build (`india_grid/data/docs/ALLINDIA_15MIN_QC.md`), which
prints counts of missing and flagged values and the mean, minimum and maximum of each column over the whole period,
test months included. No forecast, correction or score was computed on the test months.

---

## 1. Data (official Indian data only)

| Item | Value |
|---|---|
| Source | Grid Controller of India (Grid-India), Daily Power Supply Position report, sheet "TimeSeries": 15-minute SCADA values for all India |
| Built by | `india_grid/codes/scripts/build/b09_psp_timeseries.py`; hourly means of the 15-minute values, a value flagged as missing, out of range or frozen telemetry is left out, an hour needs at least 3 valid blocks |
| Series (one "unit" each) | demand met, net demand met (demand met minus wind and solar), wind generation, solar generation, in MW |
| Period | 5 November 2024 to 13 September 2026 |
| Split | train before 1 Sep 2025; validation 1 Sep 2025 to 28 Feb 2026; **test 1 Mar 2026 to 13 Sep 2026 (blinded)** |
| Information set | for a target hour on day D: values up to the end of day D−2 (a day's report is published the next day) |

From 1 June 2026 the source sheet reports storage separately: demand met then includes pumped-storage and battery
charging, and hydro excludes pumped-storage generation. The four series used here keep their meaning (net demand met
is demand met minus wind and solar throughout). One storage value was seen while checking the new sheet layout
(1 MW at 00:00 on 1 June 2026); the size of the storage load over the test months has not been looked at. The change
is reported, not corrected.

## 2. Methods (identical to the earlier study, `foreign_data_study/02_design/02_preregistration.md` §2)

* Forecasts corrected: seasonal naive 168 h, lag 48 h, and six learned day-ahead forecasts made with the same models
  and settings as before (LightGBM; Chronos-Bolt-Small zero-shot; N-HiTS, PatchTST, TiDE, DLinear with 500 training
  steps), `make_learned_baselines_india.py`: fit A on data before 1 May 2025 predicts May–Aug 2025, fit B on data
  before 1 Sep 2025 predicts 1 Sep 2025 onwards. India has no published operator day-ahead forecast in this source.
* Corrector: HistGradientBoosting (300 iterations, learning rate 0.06, depth 6, ℓ2 1), global rescale.
* Gate ladder: none, global, per series, series × 4-hour block, series × hour, per-instance (3 bags), empirical-Bayes
  shrinkage; time axis: frozen, expanding, rolling 30/90/180 days, forgetting 0.98/0.995, Fixed Share; delay 2 days;
  clip [0, 1.5]; minimum cell 30.
* Data-budget ladder (on the LightGBM forecast): validation fractions 0.1, 0.25, 0.5, full; out-of-fold train plus
  validation.
* PART v2, prequential validation and nested hold-out as before.
* Probabilistic ladder and energy valuation as in `india_prob_energy.py` (reference forecast: LightGBM, uncorrected).
  A series × hour or series × 4-hour cell with too few calibration residuals for the conformal rank (which returns
  ±∞; seen for solar at dawn in development) uses the per-series quantile instead.
* Prices for the stylised imbalance cost: hourly means of the national day-ahead and real-time market clearing prices
  in Grid-India's DSM files (`india_grid/codes/scripts/build/b10_hourly_prices.py`); available to 6 September 2026,
  so the estimate covers the test hours that have prices.
* Semi-synthetic grid (`x1_india.py`) on the 34 Indian State control areas (daily drawal, seasonal-naive forecast):
  unit axis = global → per area; state axis = per area → area × day of week. Fresh seeds 900–901.
* Primary metric: pooled squared-error skill within a unit.

## 3. Inference

As in the earlier study: day-block bootstrap (7-day blocks, 2,000 resamples, seed 0) within a unit; exact binomial
tests for sign accuracy; one-sided Wilcoxon tests for regret; Holm within family, Benjamini–Yekutieli across all
p-values. A test is SUPPORTED if its Holm-adjusted p < 0.05 and its estimate has the predicted sign; threshold tests
are SUPPORTED if the threshold is met.

## 4. Hypotheses and decision rules

Family **synthetic**
* **IH-1** (crossing condition, Indian State substrate, fresh seeds). Over non-boundary configurations
  (|oracle prediction| ≥ 0.5·ΔV), sign accuracy of the oracle condition ≥ 0.85 and of PART v2 (contiguous blocks) ≥ 0.75,
  separately for the partition axes and the instance axis (instance-axis configurations with m = 0 excluded).

Family **mechanism**
* **IH-2** (estimation cost). On the LightGBM forecast, the gain of the instance gate over the per-series gate is larger
  when gates are fitted on out-of-fold training errors plus validation than on 10 % of validation days (block bootstrap,
  one-sided).

Family **baseline**
* **IH-5**. Across the 8 forecasts × 4 series, the Spearman correlation between the forecast's relative RMSE (RMSE
  divided by the series' mean on the validation period) and the best skill reached by any ladder level is positive
  (one-sided).

Family **selection** (8 forecasts; refinements global→series, series→series×4-hour block, series→series×hour,
series→instance; seed-averaged)
* **IH-3a**. PART v2 predicts the sign of the realised refinement gain better than chance (binomial, one-sided).
* **IH-3b**. PART decision regret < never-refine regret (Wilcoxon, one-sided).
* **IH-3c**. PART decision regret < nested hold-out decision regret (Wilcoxon, one-sided).
* **IH-7a**. Static-level selection regret of PART < nested hold-out selection regret.
* **IH-7b**. PART selection regret < fixed per-series regret.

Family **time**
* **IH-6a**. Prequential validation predicts the sign of the realised gain of each discounting policy over the expanding
  gate better than chance; its decision regret is lower than always-discount and lower than never-discount.
* **IH-6b** (updating is free). In at least 80 % of the 8 forecast configurations the expanding gate is not worse than
  the frozen series gate by more than 0.002 pooled skill.

Family **probabilistic** (LightGBM forecast corrected with the static per-series gate; all four series)
* **IH-4a**. Pinball loss at q = 0.99 is lower for series × hour conformal cells than for the global quantile and than
  for per-series quantiles (two block-bootstrap tests, one-sided).
* **IH-4b**. One-sided 99 % coverage of the per-instance conditional quantile model is below 0.985 while series × hour
  cells reach ≥ 0.985.

Family **energy** (all-India demand met; net demand met reported alongside, not tested)
* **IH-8a**. The corrected forecast (static per-series gate on LightGBM) reduces absolute deviation energy relative to
  the uncorrected LightGBM forecast (block bootstrap, one-sided).
* **IH-8b**. With series × hour 99 % quantiles, the corrected forecast needs less upward reserve (MW) than the
  uncorrected forecast, with realised coverage within ±0.005.
* **IE-8c** (estimation only, no hypothesis). Stylised two-settlement imbalance cost saved by the correction for
  all-India demand, Rs crore per year, with a block-bootstrap interval.

**Stated in advance from development** (§7): on the development split a correction fitted on June–November did not
carry over to the winter test months for the strong learned forecasts (static-gate skill −0.29 for LightGBM, about 0
for Chronos and N-HiTS), while re-estimating the gate on the last 30 days helped, and corrections of the simple
forecasts paid (+0.35 for lag 48 h). IH-8a and IH-8b may therefore fail on the strong reference forecast; that outcome
would be reported as the main finding, not hidden.

## 5. Reported without a test

Skill tables for every forecast × series × ladder level; time-policy tables; data-budget curves; probabilistic tables;
reserve MW and shortfall energy; the exploratory daily ladders on the 34 State control areas (`run_india_ladders.py
--mode explore`, blocks ID1–ID3), whose periods were already opened by the India study and which are therefore
exploratory.

## 6. Exclusions and missing data

Rows with a missing target, forecast or feature are dropped identically for all levels. A run that fails for a
technical reason is rerun once with the same code; if it fails again it is reported as missing.

## 7. Development runs seen before freezing

H100 jobs 34206 (learned forecasts and ladders, development split: train before Jun 2025, validation Jun–Nov 2025,
test Dec 2025–Feb 2026), 34207 (probabilistic and energy ladder, development split; semi-synthetic grid with
development seed 100), 34209 and 34211 (the analysis code of §4 run on those development results, to check that
it executes; its numbers are development numbers, not results). The analysis code is
`04_code/scripts/india/analyze_india.py --map confirm`, which reuses the functions of the earlier study's frozen
analysis (`confirm/analyze_confirmatory.py`).

## 8. What would change the conclusions

IH-1 or IH-2 not supported → the crossing condition is not confirmed on Indian data. IH-3/IH-7 not supported → PART is
a diagnostic, not a selection rule, on Indian data. IH-6a not supported → prequential selection not recommended.
IH-8 not supported → no energy-benefit claim for correcting a strong forecast.

## 9. Deviations after unblinding

(none yet)
