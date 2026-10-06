# CUEFO-7 pre-registration (confirmatory phase)

Version prereg-v1. Written 13 Sep 2026 after the development phase (02_design/03_development_log.md, DV1–DV8,
DF1–DF13) and **before any confirmatory row was loaded**. On freezing, this file is committed, its SHA-256 is stored
in `02_design/PREREG_SHA256.txt`, and the commit is tagged `prereg-v1`; `04_code/cuefo7/guard.py` refuses to load
confirmatory data otherwise. The confirmatory code is the code at that commit:
`04_code/scripts/confirm/run_confirmatory.py`, `04_code/scripts/confirm/analyze_confirmatory.py`,
`04_code/scripts/dev/x5_x8_prob_energy.py --mode confirm`, `04_code/scripts/dev/x1_phase_diagram.py --source
opsd_confirm --seed_offset 900`, `04_code/scripts/dev/make_learned_baselines.py --mode confirm`, and the `cuefo7`
package. Any deviation after unblinding is recorded in §9 with its reason and labelled.

---

## 1. Confirmatory data (never used in development)

| Unit | Source | Series | Train / validation / test | Seeds |
|---|---|---|---|---|
| NYISO-2026 | NYISO MIS zonal load, ISO day-ahead forecast, DA/RT LBMP, DA 10-min spinning reserve price | 11 zones | 2019-01 … 2024-12 / 2025 / 2026-01-01 … 2026-08-31 | 3 |
| OPSD-2019 | OPSD time series 2020-10-06 (ENTSO-E load + TSO day-ahead forecast) | 47 non-overlapping areas (list in `cuefo7/data.py::OPSD_UNITS`, those eligible in the processed panel) | 2015–2017 / 2018 / 2019 | 2 |
| OPSD-2020 | same | same | 2015–2018 / 2019 / 2020-01 … 2020-09 (pandemic shock) | 2 |
| UCI-2014 | UCI ElectricityLoadDiagrams20112014, hourly mean kW | 60 clients drawn with seed 7 from the 145 eligible | 2012 / 2013 / 2014 | 2 |
| X1-OPSD | semi-synthetic grid on the OPSD-2019 substrate | 12 areas drawn with seed 11 | as OPSD-2019 | seeds 900–901 |

Information set for every day-ahead forecast of day D: observations up to the end of day D−2 (lags ≥ 48 h) plus the
operational forecast issued for day D.

## 2. Methods (frozen, identical to development after DV1–DV8)

* Corrector: HistGradientBoosting (300 it., lr 0.06, depth 6, ℓ2 1, ≤ 300k training rows), global D9 rescale.
* Baselines: NYISO — ISO forecast, seasonal naive 168 h, lag 48 h, LightGBM, Chronos-Bolt-Small, NHITS, PatchTST, TiDE,
  DLinear (500 training steps; learned models fitted on data < 2023 for 2023–2024 predictions and on data < 2025 for
  2025–2026 predictions); OPSD — TSO forecast, seasonal naive 168 h, lag 48 h; UCI — seasonal naive 168 h, lag 48 h.
* Gate ladder: none, global, series, series×hour-block (6 blocks), series×hour, per-instance bagged WLS (3 bags,
  depth 5, 150 it.), empirical-Bayes shrinkage; time axis: frozen series gate, expanding, rolling 30/90/180 days,
  forgetting 0.98/0.995 per day, Fixed Share (α = 0.02); delay 2 days; clip [0, 1.5]; min cell 30.
* Data-budget ladder: validation fractions 0.1, 0.25, 0.5 (NYISO), full validation, OOF train + validation.
* PART v2 (interleaved weekly blocks; K = 6 for partitions, K = 4 for the instance gate; evaluation subsample 20k).
* Prequential validation (burn-in 60 days, comparator: expanding gate).
* Nested hold-out (first vs second half of validation days).
* Probabilistic ladder and energy valuation as in `x5_x8_prob_energy.py` (cross-fitted calibration residuals).
* **Primary metric: pooled squared-error skill** within a unit (the loss used by the theory and by PART);
  per-series macro skill is secondary and reported.

## 3. Inference

* Within a unit: daily aggregation of per-row loss differentials; moving-block bootstrap over days, 7-day blocks,
  2,000 resamples, seed 0; one-sided p-values as implemented in `cuefo7.stats.block_ci`.
* Across configurations: exact binomial tests (sign accuracy) and one-sided Wilcoxon signed-rank tests (regret).
* Across units: DerSimonian–Laird random-effects pooling where a test says "meta".
* Multiplicity: Holm within family; Benjamini–Yekutieli across all p-values (reported). A test with a p-value is
  SUPPORTED iff its Holm-adjusted p < 0.05 and its estimate has the predicted sign (and, for H8b, the coverage
  condition holds). Threshold tests (H1, H4b, H6b) are SUPPORTED iff the threshold is met.

## 4. Hypotheses and decision rules

Family **synthetic**
* **H1** (crossing condition, X1-OPSD, fresh seeds). Over configurations with |oracle prediction| ≥ 0.5·ΔV
  (non-boundary), sign accuracy of the oracle condition ≥ 0.85 and of PART v2 (contiguous blocks, the design that
  sees imposed drift) ≥ 0.75, separately for partition axes and for the instance axis. For the instance axis the
  estimation cost is measured as the realised loss at zero heterogeneity, which makes m = 0 tautological; those
  configurations are excluded (DV8). *Stated in advance from development: instance-axis PART is biased toward coarse
  (block models use n/K rows, 09_theory T4b) and correct in only 2 of 4 development configurations with m > 0, so
  its H1 criterion is expected to fail; development partition-axis accuracies were 0.95/0.84 (state) and 0.89/0.76
  (unit).*

Family **mechanism**
* **H2** (estimation cost). For NYISO-2026 (ISO), OPSD-2019 (TSO), UCI-2014 (seasonal naive): the gain of the instance
  gate over the series gate is larger when gates are fitted on OOF train + validation than on 10 % of validation days
  (per-row difference, block bootstrap, one-sided). Tested per unit and pooled (meta).

Family **baseline**
* **H5**. Across the nine NYISO-2026 baselines, the Spearman correlation between baseline RMSE and the best skill
  achieved by any ladder level is positive (one-sided).

Family **selection**
* **H3a**. PART v2 predicts the sign of the realised refinement gain (global→series, series→series×hour-block,
  series→series×hour, series→instance; all baselines of NYISO-2026, OPSD-2019, OPSD-2020, UCI-2014; seed-averaged)
  better than chance (binomial, one-sided).
* **H3b**. PART decision regret < never-refine regret (Wilcoxon, one-sided).
* **H3c**. PART decision regret < nested hold-out decision regret (Wilcoxon, one-sided).
* **H7a**. Static-level selection regret of PART (sequential: global→series if Ĝ>0; then the state refinement with the
  largest positive Ĝ; then instance if its Ĝ is positive and largest) < nested hold-out selection regret.
* **H7b**. PART selection regret < fixed per-series regret.
  *Stated in advance from development (DF10, dry run): H3b, H3c, H7a and H7b may fail; on development data PART beat
  nested hold-out and never-refine when pooled over rows but not at the configuration level, with ETT as a failure
  case of non-persistent state heterogeneity.*

Family **time**
* **H6a**. Prequential validation predicts the sign of the realised gain of each discounting policy over the expanding
  gate better than chance (binomial); its decision regret is lower than always-discount and lower than
  never-discount (two Wilcoxon tests, one-sided).
* **H6b** (updating is free). In at least 80 % of (unit, baseline) configurations the expanding gate is not worse than
  the frozen series gate by more than 0.002 pooled skill.

Family **probabilistic** (NYISO-2026, corrected forecast with static per-zone gate)
* **H4a**. Pinball loss at q = 0.99 is lower for zone×hour conformal cells than for the global quantile and than for
  per-zone quantiles (two block-bootstrap tests, one-sided).
* **H4b**. One-sided 99 % coverage of the per-instance conditional quantile model is below 0.985 while zone×hour
  cells reach ≥ 0.985.

Family **energy** (NYISO-2026)
* **H8a**. The corrected forecast (static per-zone gate) reduces absolute deviation energy relative to the ISO forecast
  (block bootstrap, one-sided).
* **H8b**. With zone×hour 99 % reserve quantiles, the corrected forecast requires less upward reserve (MW summed over
  zones) than the ISO forecast, with realised coverage within ±0.005.
* **E8c** (estimation only, no hypothesis). Two-settlement imbalance cost difference ISO − corrected, M$/yr with CI.

## 5. Descriptive analyses reported without hypothesis tests

Full skill tables (pooled and macro) for every unit × baseline × level; time-policy tables; data-budget curves;
probabilistic ladder tables (CRPS, pinball, coverage, width, conditional coverage deviation); reserve MW, cost at the
day-ahead spinning reserve price and shortfall energy; OPSD-2020 contrasted with OPSD-2019 as a shock period.

## 6. Exclusions and missing data

Rows with missing target, baseline or features are dropped before fitting, identically for all levels. A
configuration whose run fails for a technical reason is rerun once with the same code; if it fails again it is
reported as missing and excluded from tests that need it.

## 7. Compute plan

Learned baselines (NYISO confirm), C1–C7 ladders, X5/X8 confirm, X1-OPSD partition grid (2 axes × 6 m × 6 φ ×
κ ∈ {1, 2} × frac ∈ {0.25, 1} × 2 seeds) and instance grid (m ∈ {0, 1, 4} × φ ∈ {0, π/4, π/2} × κ ∈ {1, 2} ×
frac ∈ {0.25, 1} × 2 seeds), then `analyze_confirmatory.py --map confirm`.

## 8. What would change the paper's conclusions

* H2 or H1 not supported → the estimation-cost / crossing-condition account is not confirmed and the paper reports it
  as a development-phase finding only.
* H6a not supported → prequential selection is not recommended.
* H3/H7 not supported → PART is reported as a diagnostic, not a selection rule; fixed defaults are recommended.
* H8a/H8b not supported → no energy-benefit claim.

## 9. Deviations after unblinding

(none yet)
