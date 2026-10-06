# RE-FUSED pre-registration (DRAFT — not frozen)

Status: draft written 15 Sep 2026 while development training runs. Sections marked **[from development]** are filled
by `04_code/scripts/prereg/fill_prereg.py` from the development result files. After that, the file is copied to
`02_design/02_preregistration.md`, hashed into `PREREG_SHA256.txt`, committed and tagged `prereg-v1`. Only then does
the blinding guard allow the confirmatory run (`REFUSED_PHASE=confirm`, `REFUSED_CONFIRMATORY=1`).

## 1. Data and splits

* Panel: `03_data/processed/refused_state_day.parquet`, content hash `c6ef9c8ebaba692cc0952025f87ea311e09b54ffae9e77c1c04a1b0ffaec4bbd`.
* **Confirmatory phase** = the development pipeline with every boundary moved forward one year:

  | Role | Period |
  |---|---|
  | fit | target dates 1 Apr 2018 – 31 Mar 2024 (last 180 days for early stopping only) |
  | calibration and online-gate warm-up | 1 Apr 2024 – 31 Mar 2025 |
  | **confirmatory evaluation** | **1 Apr 2025 – 31 Aug 2026** |

* The State RE target (T4) is evaluated only to 18 Nov 2025, where the source ends.
* Information sets, targets, horizons, features, model code, seeds and tuning budgets are those of the development
  protocols (O2 v1.3, O3 v1.2, O4/O5 v1.0 with addenda to v1.3). Hyperparameter choices and selections are frozen as
  below; nothing is selected on the evaluation period.

## 2. Frozen choices [from development]

| Item | Frozen value | Source file |
|---|---|---|
| O2 selected model per target (T1–T5) | … | `06_results/dev/o2/eval/selection.json` |
| O2 calibration level per target and model | … | same |
| O3 main variant | `mcag_instance` + online time-adaptive gate (OTG) | protocol |
| O3 ablation set | A2, A3, A4, A6, L1 as in protocol v1.2 | protocol |
| O4 weights (R, S, C, U) and κ; weights without U | … | `06_results/dev/o4/weights.json` |
| O5 A2 β, ρ; A3 ρ₀; A4 β₀, β_U, β_stress; A2r β | … | `06_results/dev/o5/frozen_params.json` |
| O5 PPO | seeds 0–4 retrained on the calibration year; seed choice by J on the calibration year | protocol |

## 3. Confirmatory hypotheses

For each hypothesis the family, test and decision rule are fixed. Tests are one-sided in the stated direction.
Evidence units are days, pooled across States by the daily cross-series mean. Inference uses HLN-corrected
Diebold–Mariano statistics and 7-day block-bootstrap 95 % confidence intervals.

### Family F1 — O2 forecasting (Holm within family)

* **H1a–e.** For each target T1–T5, the frozen selected model has lower scaled absolute error than the seasonal naive
  at H = 1, 2 and 3, pooled over H (daily mean of the scaled |e| differential < 0; HLN p < 0.05). Supported when the
  Holm-adjusted p < 0.05 and the per-seed result has p < 0.05 for all 5 seeds.
* **H2.** The rolling-180 conformal intervals of the selected model cover within ±5 percentage points of nominal at
  80 % and 90 %, for every target (pooled over series and H). Descriptive threshold; a 7-day block-bootstrap CI is
  reported.
* **H3.** The publication-latency cost is positive: LightGBM with measured availability lags has higher scaled
  absolute error than with instant publication (pooled over targets; one-sided p < 0.05). Reported as a size estimate
  with CI.

### Family F2 — O3 context fusion (Holm within family)

* **H4 (A2).** `mcag_instance`+OTG has lower scaled pinball loss than `fusion_fixed`+OTG at matched capacity, pooled
  over T1–T5.
* **H5 (A3).** Real context beats null context and permuted context (T1, T2, T5; pinball; both comparisons required).
* **H6 (time adaptivity).** OTG lowers scaled absolute error relative to the same model without OTG (pooled T1–T5).
* **H7 (L1).** When the RE and weather blocks are missing, the error increase of `mcag_instance` is smaller than that
  of `fusion_fixed` (T1, T2).
* **H8 (A6, GCAL).** The carbon block does not worsen accuracy and improves it for T3 (pinball; one-sided for T3,
  non-inferiority margin 1 % of the no-carbon loss for T1, T2).

### Family F3 — O4 assessment (Holm within family)

* **H9.** With frozen weights, the context-aware assessment A ranks the composite consequence Y better than M₀ within
  M₀ deciles (difference in mean within-decile Kendall τ > 0; week-block bootstrap CI excludes 0).
* **H10.** The gain from U exceeds the 95th percentile of the permuted and Gaussian placebo gains, with placebos
  refitted on the calibration year.

### Family F4 — O5 scheduling (Holm within family)

* **H11.** A4 (regime- and uncertainty-coupled risk) has lower system-wide daily regret CVaR₉₀ than A2
  (fixed risk), and than H_pub and P (tail-day block-bootstrap CI below 0).
* **H12.** Every decision arm has zero hard-limit violations (deterministic check).

### Family F5 — price of adaptivity (reported with sign agreement)

* **H13.** PART predictions from the calibration year agree in sign with the realised gains on the evaluation period
  for more than half of the target × H cells, for each refinement (fixed → regime, → instance, → OTG). Binomial test
  against 0.5, one-sided.

Across families, Benjamini–Yekutieli adjustment is reported for all primary tests.

## 4. Reporting commitments

* All hypotheses are reported whether supported or not, with effect sizes and CIs.
* Development results are reported as development evidence, clearly separated from confirmatory results.
* Any deviation from this document (a code bug, unavailable data) is logged in a dated deviation table with its
  reason, before the affected result is inspected where possible.
* The O5 settlement is stylised (no frequency linkage, no 15-minute blocks, no volume limits) and is labelled as such.
* No exchange (non-government) data are used. The reserved period is evaluated once.

## 5. Computation plan

1. `REFUSED_PHASE=confirm` with the blinding guard satisfied.
2. `o2_01_samples` → LightGBM, PatchTST, BiLSTM, Chronos-Bolt, TFT (5 seeds).
3. `o2_06_evaluate`, then O3 variants and ablations, then `o3_02_evaluate`.
4. `o4_01_assessment`, `o5_01_scheduling`, `o5_02_ppo`.
5. Confirmatory audit (claims map, numbers regeneration, leakage test in confirm phase).
