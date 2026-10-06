# RE-FUSED research design

Version 1.0 (13 Sep 2026). Supersedes RE-FUSED-6 plan. Status: **design frozen before any RE-FUSED data are scored.**
Changes to this file after the first development result are logged in §9 with reasons.

---

## 1. Central scientific claim to be tested

**The price of adaptivity.** When a forecast is corrected by an adaptive gate, g(z)·r̂(x), making the gate finer
along an *adaptivity axis* — units (global → per series), states (series → regime cell → instance) or time
(static → rolling → forgetting) — buys an *approximation gain* equal to the heterogeneity of the oracle gate
along that axis, and pays three costs: **estimation** (≈ variance × effective degrees of freedom / n),
**non-persistence** (heterogeneity learned on validation that does not recur at deployment) and **selection**
(choosing the level from finite, shifted data). Refinement pays only when heterogeneity is *large, persistent and
estimable*:

    2·⟨h_V, h_T⟩ − ‖h_V‖² > ΔV / n        (persistence-adjusted crossing condition, 09_theory T4)

The paper's contribution is to (i) state this condition in measurable quantities for all three axes and for both
point and quantile (reserve) gates, (ii) show it predicts where fine adaptation wins and loses in controlled
semi-synthetic phase diagrams, (iii) show the consequences on real forecasting and energy-system decisions, and
(iv) test the resulting operational rule on data collected after the hypotheses were frozen.

## 2. Research questions and hypotheses

Each hypothesis is labelled **D** (development: exploratory, may be iterated) or **C** (confirmatory: frozen in
`02_design/02_preregistration.md`, git-tagged and SHA-256 hashed before the confirmatory data are loaded).

| RQ | Question | Development hypothesis | Confirmatory form (to be fixed after development) |
|---|---|---|---|
| RQ1 | Does the crossing condition predict which adaptivity level wins? | H1-D: in semi-synthetic phase diagrams (known heterogeneity τ², persistence π, noise, n), the oracle-parameter condition predicts the better of coarse/fine with accuracy ≥ 0.85 and the plug-in estimate ≥ 0.75 | H1-C: same thresholds on a freshly generated grid with new seeds and real residual noise from a dataset not used in development |
| RQ2 | Is estimation cost the mechanism? | H2-D: the fine − coarse gap increases monotonically with gate-fitting sample size (cross-fitted OOF residuals, subsampling ladder) with slope consistent with ΔV/n | H2-C: sign and monotonicity on fresh data (NYISO load 2026) |
| RQ3 | Does persistence matter separately from size? | H3-D: at fixed τ² and n, refinement stops paying as π falls below ½(1+ΔV/(nτ²)); on real data, estimated persistence of per-instance heterogeneity is low | H3-C: the persistence estimate computed on validation predicts the sign of the realised refinement gain on fresh test data better than chance |
| RQ4 | Does the price apply to uncertainty (quantiles, reserves)? | H4-D: the quantile-specific crossing condition (09_theory T6: gain Σ p f (ξ_c′−ξ)²/2 vs cost q(1−q)/(2 n f)) predicts the best adaptivity level for each quantile level; whether the tail is coarser is measured, not assumed | H4-C: on NYISO 2026, per-zone(-hour) conformal reserve quantiles match or beat per-instance conditional quantile models on pinball loss at q = 0.99 and have smaller coverage deviation |
| RQ5 | How does baseline strength change the value of correction and of adaptivity? | H5-D: correction value (best rung) decreases with baseline skill; per-instance never beats the best coarse rung on strong deep/foundation/ISO baselines | H5-C: on fresh data the best coarse rung is ≥ per-instance for every strong baseline |
| RQ6 | When does time adaptivity pay? | H6-D: rolling / forgetting gates beat static only when estimated drift variance exceeds the tracking estimation cost; online aggregation (Fixed Share) does not beat a static per-series gate on stationary series | H6-C: tested on NYISO 2026 and OPSD TSO forecasts |
| RQ7 | Does a validation-only rule built on the condition beat fixed defaults? | H7-D: the persistence-adjusted refinement rule (PART) has regret ≤ fixed per-series and ≤ nested hold-out on development data | H7-C: PART regret ≤ fixed per-series on fresh data (one-sided test at the evidence-unit level) |
| RQ8 | What are the energy-system consequences? | H8-D (estimation, not a significance claim): quantify imbalance cost, absolute deviation energy and 99% reserve requirement for the ISO forecast, the coarse-gated correction and the per-instance-gated correction, with block-bootstrap CIs | H8-C: on NYISO Jan–Aug 2026, the coarse-gated correction lowers the 99% reserve requirement and absolute deviation energy relative to the ISO forecast and to per-instance gating; monetised with stated prices |

Hypotheses that fail are reported as failed; the manuscript is built from the frozen confirmatory results.

## 3. Data and roles

| Dataset | Units | Resolution | Span | Role | Status in RE-FUSED-6 |
|---|---|---|---|---|---|
| India daily generation (IDP/CEA) | 20 series | daily | 2017-09 → 2025-04 | development; drifting operator schedule | used (dev only) |
| ETTh1, ETTh2, ETTm1, ETTm2 | 7 channels each | 1h / 15min | 2016-07 → 2018-06 | development; deep baselines | used (dev only) |
| NYISO zonal LBMP (DA, RT) | 15 zones | 5min → 1h | 2019 → 2025 | development | used (dev only) |
| **NYISO zonal load + ISO day-ahead forecast** | 11 zones | 1h | 2019-01 → 2025-12 | development (train ≤2023, val 2024, test 2025) | **new** |
| **NYISO load, forecast, LBMP, reserve prices 2026** | 11 / 15 zones | 1h | 2026-01 → 2026-08 | **confirmatory test** (train ≤2024, val 2025) | **new, blinded** |
| **OPSD / ENTSO-E load + TSO day-ahead forecast** | ~30 countries/areas | 1h | 2015 → 2020 | **confirmatory** (external domain) | **new, blinded** |
| **UCI ElectricityLoadDiagrams 2011–2014** | 370 clients | 15min → 1h | 2011 → 2014 | **confirmatory** (many units, little data per unit) | **new, blinded** |

**Blinding in code.** `refused_gate.data.load(..., split="confirm")` raises unless the environment variable
`RE-FUSED_CONFIRMATORY=1` is set *and* `02_design/02_preregistration.md` exists with a SHA-256 recorded in
`02_design/PREREG_SHA256.txt` that matches its current content and a git tag `prereg-v1`. Development scripts
cannot load confirmatory rows; a unit test enforces this.

## 4. Methods compared

**Baselines B.** seasonal naive (24h / 168h / 7d), persistence, rolling mean; LightGBM global model with lags and
calendar; NHITS, PatchTST, TiDE, DLinear (neuralforecast, fixed a-priori hyperparameters); Chronos-Bolt-Small
(zero-shot foundation model); operational references: NYISO ISO day-ahead load forecast, ENTSO-E TSO day-ahead
forecast, Indian operator schedule.

**Correctors r̂.** HistGradientBoosting (RE-FUSED-6 default), ridge on lags/calendar, MLP — trained on the training
period only, or cross-fitted out-of-fold (A4).

**Gate ladder (three axes).**
* unit: none · global · per series
* state: per series × regime (volatility tercile / hour-of-day block) · per-instance WLS-bagged GBM · per-instance
  neural sigmoid gate · empirical-Bayes shrinkage between levels
* time: static · rolling window w ∈ {30, 90, 180, 365 days} · exponential forgetting · online Fixed Share
  between B and B + g·r̂
* complexity axis: per-instance gates with increasing capacity (depth/leaves/width), indexed by measured GDF

**Selection rules.** fixed per-series (RE-FUSED-6 default) · nested hold-out argmin · GDF-Cp · **PART**
(persistence-adjusted refinement test, 09_theory §T8) — all validation-only.

**Deployment rule.** per-series block-LCB (RE-FUSED-6), reported as a standard tool.

**Probabilistic ladder.** split-conformal residual quantiles: global · per series · per series × hour block ·
per regime cell · conditional quantile GBM (per instance) · rolling window; q ∈ {0.01, 0.05, …, 0.95, 0.99}.

## 5. Metrics and inference

* Point: squared-loss skill vs the baseline (macro over series), MAE, MASE; pooled and macro both logged.
* Probabilistic: pinball loss per quantile, CRPS (19-quantile approximation), coverage and width of 90% / 98%
  intervals, conditional coverage deviation by series and hour.
* Energy: absolute deviation energy (MWh), two-settlement imbalance cost Σ(A−F)(P_RT−P_DA) ($), deviation
  exposure Σ|A−F|·|P_RT−P_DA| ($), 99% / 97.5% upward reserve requirement (MW) and its cost at the day-ahead
  10-minute spinning reserve price ($).
* **Evidence unit** = (dataset, test period). Seeds averaged within unit. Within a unit: moving-block bootstrap
  (block = 7 days, 2,000 resamples) of the time series of loss differentials → 95% CI and one-sided p.
  Across units: DerSimonian–Laird random-effects meta-analysis of the standardised effect, plus sign count.
* Multiplicity: claim families C1 (point ladder), C2 (mechanism), C3 (probabilistic), C4 (selection),
  C5 (time axis), C6 (energy); Holm within family; Benjamini–Yekutieli across families.

## 6. Experiment programme

| ID | Purpose | Data | Output |
|---|---|---|---|
| X0 | Re-analysis of RE-FUSED-6 at the correct inferential unit (A1) | RE-FUSED-6 CSVs | corrected CIs / meta-analysis |
| X1 | Semi-synthetic phase diagrams: τ², π, noise, n, drift, axis | real features + real residual noise (dev datasets) | best-rung maps; condition accuracy |
| X2 | Data-budget mechanism: cross-fitted gates and subsampling ladder | India, ETTh1/2, NYISO load (dev) | gap vs n_fit |
| X3 | Complexity / GDF axis | India, ETTh1, NYISO load (dev) | risk vs GDF/n |
| X4 | Baseline breadth incl. deep, foundation, ISO | NYISO load, ETTh1/2, India (dev) | ladder per baseline; value vs strength |
| X5 | Probabilistic ladder, reserve quantiles | NYISO load, ETTh1, India (dev) | pinball/CRPS/coverage by level and q |
| X6 | Time axis and online aggregation | NYISO load/prices, India schedule (dev) | static vs rolling vs forgetting vs Fixed Share |
| X7 | Selection rules incl. PART | all dev | regret |
| X8 | Energy valuation | NYISO load 2025 (dev) | $, MWh, MW with CIs |
| **XC** | **Confirmatory battery (frozen code)**: X2, X4, X5, X6, X7, X8 on NYISO 2026; X4/X5/X7 on OPSD and UCI; X1 fresh grid | blinded data | pre-registered verdicts |

## 7. Stopping rule

Iterate development experiments until (a) every hypothesis has a concrete, falsifiable confirmatory form, (b) no
open defect affects a confirmatory hypothesis, (c) a reviewer audit finds no addressable major weakness. Then
freeze, run XC once, audit, and write. After XC, changes are allowed only as clearly labelled exploratory
analyses.

## 8. Compute budget (this machine: 8 cores, 15.7 GB RAM, GTX 1660 Ti 6 GB)

Corrector training subsampled to ≤ 300k rows; deep models trained once per dataset split with rolling
prediction (no refit); `OMP_NUM_THREADS=4`; one heavy job at a time; resumable result caching per unit.

## 9. Change log

| Date | Change | Reason |
|---|---|---|
| 13 Sep 2026 | v1.0 | initial design from the audit (01_audit) |
| 13 Sep 2026 | v1.1 — H4-D reworded before any data were scored | derivation in 09_theory T6 shows the tail need not be coarser under Gaussian scale heterogeneity; the hypothesis now tests the condition, not a direction |
