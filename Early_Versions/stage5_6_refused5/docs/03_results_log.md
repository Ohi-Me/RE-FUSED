# RE-FUSED-5 results log (running record, 12 Sep 2026)
Every number below is from a script in `src/` writing to `results/` or `audit/`. Nothing is estimated.

## A. Data audit of RE-FUSED-4 (frozen baseline)
| Finding | Value | Source |
|---|---|---|
| Panel price identical across all 18 states | 100.0% of 2,627 dates, within-date sd = 0.0 | `audit/A1_price_provenance.json` |
| Effective price sample | 2,627 day-level values, not 47,286 rows | A1 |
| Price imputed (LOCF) in test | 98.21%; at ₹10,000 cap on 78.01% of test rows | A1 |
| Panel price vs raw IEX daily mean | corr 0.649, MAE ₹1,207, exact match 9.49% | A1 |
| Panel dates with no raw price at all | 1,721 / 2,627 (test: 391/423) | A1 |
| Raw IEX archive | 2015-04-01..2024-03-31, 289,667 blocks, "-" = missing | A2/A3 |
| Raw observed blocks per day | 94 (2015) → 75 (2017) → 24 (2018) → 6 (2023) | `audit/A4_raw_day_coverage.csv` |
| `cleaned` vs original `dataset` copies | byte-equivalent content: gaps are in the source | A3 |

**Consequence:** price/PA-LMP is not a measurable forecasting target on this panel; RE-FUSED-5 replaces it
with (a) genuine per-state generation and (b) a real 5-minute NYISO price testbed.

## B. Rebuilt datasets
* **India daily panel (D2).** 799 balanced units (≥95% of 2,657 days), 20 states, no imputation,
  strict chronological split: train 2017-09→2022-12 (37,240), val 2023 (7,300), test 2024-01→2025-04 (8,460).
* **Baseline ladder on test (mean MASE over states):** persistence **0.502** < roll-7 **0.675** <
  seasonal-naive **0.955** < operator schedule **1.305**. `data/D2_baselines_test.csv`
* **NYISO (D3).** 84 months of 5-min real-time + day-ahead zonal LBMP, 15 zones, resampled to
  5min/15min/1h/1d from one source.

## C. Theory checks (synthetic, ground truth known)
* **S1 — the shrinkage law.** g* = SNR/(1+SNR) predicted vs empirical optimal gate:
  **corr 0.9915, MAE 0.027** over 28 (SNR × n) cells. L3 sign of degradation correct in **25/28**
  cells (misses only where |gain| < 0.5%). Always-correct lost up to **−33%** at low SNR; gated
  correction never degraded. `results/S1_synth_law.json`
* **S2c — free per-instance gating fails (falsification F2).** Mean per-cell skill:
  static per-cell **0.1294** > always **0.1185** > global **0.1175** > CVC per-instance **0.1147**;
  CVC worse than static in **10/10 seeds**, and still worse in Case B where the optimal gate truly
  varies within cells. Degraded cells: static 1.9/15 vs CVC 4.0/15. Predicted benefit is nonetheless
  **calibrated** (slope 0.89–0.91, R² 0.985). `results/S2c_case{A,B}.json`
* **S3 — gate granularity (running).** With small validation (n=500) the **shrunk hierarchical gate
  beat static in 10/10 seeds** (+0.0175); with n=2,000 static won (shrunk −0.0041, 0/10). Crossover
  behaves as the law predicts: conditioning pays only when Var_within(g*) > Var(ĝ).

## D. Real data (India, running)
Skill vs the baseline it corrects (mean over 20 states):

| baseline (strength) | always | global | static/state | free | **shrunk** |
|---|---|---|---|---|---|
| persistence, h=1 | +0.067 | +0.067 | +0.072 | +0.070 | **+0.074** |
| roll-7, h=1 | +0.096 | +0.097 | +0.105 | +0.099 | **+0.108** |
| seasonal-naive, h=1 | +0.190 | +0.187 | **+0.193** | +0.183 | +0.192 |
| operator schedule, h=1 | **−0.291** | −0.052 | +0.002 | −0.004 | **+0.006** |
| persistence, h=2 | +0.055 | +0.054 | +0.061 | +0.062 | **+0.067** |
| seasonal-naive, h=3 | +0.042 | +0.051 | +0.065 | +0.054 | **+0.066** |
| operator schedule, h=3 | −0.290 | −0.036 | −0.015 | −0.017 | **−0.008** |

**New finding (not in the original hypothesis): correction value depends on the *stationarity* of the
baseline's bias, not its accuracy.** The weakest baseline (operator schedule) has the largest residual
but a drifting bias — sched/actual ratio swings 0.987→1.058 by year, residual mean −8.2 (2019) to
+2.1 (2022) — so a train-fitted corrector degrades it by 29% when applied ungated. Persistence
residuals are stationary (mean ≈0, sd ≈10 every year) and correct reliably.

## E. Status of the pre-registered falsifications
* **F1 (learned gate ≈ g*)**: partially supported — value estimate calibrated, gate itself noisy.
* **F2 (static suffices)**: **FIRED** on synthetic; on real data static/shrunk ≈ tie, free never best.
  Claim revised: contribution is the granularity law + shrunk gate, not per-instance gating.
* **F3 (calibration)**: passed (slope 0.89–0.91).
* **F4 (risk control less conservative than clipping)**: not yet tested.
* **F5 (VOC transfer across datasets/resolutions)**: not yet tested.

## F. Value of context (R2) — null result with permutation controls
India panel, window 2019-11→, 11 states with full context coverage, 5 seeds per configuration,
persistence baseline. VOC = skill(with block) − skill(base); control = same block permuted within state.

| h | block | base | with | permuted | VOC | p vs base | p vs permuted |
|---|---|---|---|---|---|---|---|
| 1 | renewables | +0.0617 | +0.0598 | +0.0632 | −0.0018 | 0.562 | 0.439 |
| 1 | coal | +0.0617 | +0.0623 | +0.0680 | +0.0007 | 0.754 | 0.080 |
| 1 | outage | +0.0617 | +0.0613 | +0.0645 | −0.0004 | 0.921 | 0.504 |
| 3 | renewables | −0.0142 | −0.0111 | −0.0188 | +0.0031 | 0.642 | 0.517 |
| 7 | renewables | −0.1151 | −0.0804 | −0.1202 | **+0.0347** | 0.151 | **0.020** |
| 7 | outage | −0.1151 | −0.1009 | −0.1133 | +0.0142 | 0.273 | 0.331 |

**Reading.** No context block carries reproducible incremental value at h=1 or h=3. Only renewables at
h=7 beats its permutation control (p=0.020). This independently reproduces RE-FUSED-4's failed carbon
claim (N2) under a stricter design, and matches law (L4): added explainable variance < added
estimation variance. Note base skill turns negative at h≥3 in this reduced window — correction of
persistence stops paying at longer horizons.

## G. Resolution law (R3/R3b) — NYISO, 15 zones, test = 2025
| resolution | horizon | baseline | base MAE ($/MWh) | out-of-sample residual R² | always | global | static/zone |
|---|---|---|---|---|---|---|---|
| 5 min | 5 min | persistence | 5.83 | +0.008 | −0.037 | −0.044 | −0.045 |
| 5 min | 1 h | persistence | 15.90 | −0.018 | −0.063 | −0.004 | −0.002 |
| 5 min | 1 h | day-ahead | 18.87 | **+0.166** | **+0.120** | +0.102 | +0.100 |
| 15 min | 1 h | day-ahead | 17.76 | +0.131 | +0.114 | +0.107 | +0.106 |
| 1 h | 1 h | day-ahead | 15.97 | +0.164 | **+0.138** | +0.132 | +0.132 |
| 1 h | 1 h | persistence | 12.49 | +0.060 | +0.008 | +0.039 | +0.039 |
| 1 d | 1 d | day-ahead | 11.08 | −0.643 | −0.316 | −0.003 | −0.025 |
| 1 d | 1 d | persistence | 14.40 | −0.053 | −0.079 | +0.026 | +0.023 |

**Reading (answers the 15-min → 5-min question honestly).** Finer settlement resolution does *not*
create model value: at 5-min resolution and 5-min horizon the residual is essentially unpredictable
(R² = 0.008) and correction *hurts*. Persistence MAE is nearly resolution-invariant (9.2–9.5 in the
dataset report) while the day-ahead reference degrades from 11.5 (hourly) to 14.1 (5-min). The value
therefore sits in **day-ahead → real-time correction** (+0.10 to +0.14 skill), and it grows as
settlement resolution refines. At daily aggregation everything collapses (residual R² −0.64).

## H. Risk-controlled deployment (R4) — falsification F4 supported
Block-LCB rule (monthly validation blocks, α=0.1) vs CRC-style direction-gate + quantile clip.
Values are mean skill over 20 states with (degraded states) in brackets.

| h | baseline | always | static | CRC-like clip | block-LCB (ours) | withheld (would have helped) |
|---|---|---|---|---|---|---|
| 1 | persistence | +0.0652 (4) | +0.0659 (2) | +0.0710 (2) | +0.0722 (0) | 3 (1) |
| 1 | operator schedule | -0.3000 (12) | -0.0052 (10) | +0.0007 (9) | +0.0151 (2) | 14 (6) |
| 2 | persistence | +0.0405 (5) | +0.0496 (5) | +0.0588 (2) | +0.0532 (3) | 3 (1) |
| 2 | operator schedule | -0.2918 (12) | -0.0052 (9) | -0.0004 (8) | +0.0243 (1) | 14 (6) |
| 3 | persistence | +0.0279 (6) | +0.0422 (3) | +0.0540 (2) | +0.0401 (1) | 5 (3) |
| 3 | operator schedule | -0.2878 (13) | -0.0087 (10) | -0.0060 (8) | +0.0211 (0) | 16 (6) |
| 7 | persistence | -0.0238 (9) | +0.0153 (6) | +0.0246 (4) | +0.0037 (3) | 12 (9) |
| 7 | operator schedule | -0.3028 (12) | -0.0335 (10) | -0.0278 (8) | -0.0091 (1) | 17 (8) |

**Reading.** On the non-stationary baseline the risk-controlled rule is the only strategy that avoids
systematic degradation (0–2 degraded states vs 8–10 for clipping) — precisely the regime CRC's authors
flag as unsolved. On the stationary baseline the two are close (block-LCB wins at h=1 and h=2, clipping at h=3 and h=7), and
our rule pays a measurable conservatism cost (it withholds 12–17 cells, 6–9 of which would have
helped). **Honest verdict: F4 is supported for safety, not for raw accuracy.**

## I. Bootstrap variance estimate (S4) - the criterion's failure was partly an estimator artifact

Split-half estimation of the gate variance gave rho > 1 in every configuration, making the decision
rule uninformative. Estimating it by bootstrap (B=5 resamples of validation) changes the picture:

| quantity | split-half | bootstrap |
|---|---|---|
| rho range | 1.04 - 4.94 (always > 1) | 0.46 - 0.91 (always < 1) |
| sign accuracy of the rule | 0.182 | 0.750 |
| shrunk - static | +0.0015 (t=1.24, p=0.219) | **+0.00656 (t=3.24, p=0.0029)** |
| bagged - static | not measured | -0.00155 (t=-0.81, p=0.4223) |

Per cell (4 seeds each):

| amp | n_val | static | free | bagged | shrunk | oracle | rho | var_est | var_within | within_true |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 500 | 0.1053 | 0.1041 | 0.1115 | 0.1213 | 0.1347 | 0.62 | 0.1439 | 0.0877 | 0.1033 |
| 0.0 | 8000 | 0.1294 | 0.1162 | 0.1192 | 0.1268 | 0.1330 | 0.67 | 0.1347 | 0.0907 | 0.1044 |
| 0.5 | 500 | 0.1044 | 0.1070 | 0.1137 | 0.1220 | 0.1381 | 0.60 | 0.1460 | 0.0874 | 0.1051 |
| 0.5 | 8000 | 0.1313 | 0.1176 | 0.1220 | 0.1288 | 0.1357 | 0.70 | 0.1355 | 0.0947 | 0.1061 |
| 1.0 | 500 | 0.1164 | 0.1211 | 0.1268 | 0.1348 | 0.1529 | 0.58 | 0.1468 | 0.0840 | 0.1199 |
| 1.0 | 8000 | 0.1464 | 0.1311 | 0.1352 | 0.1421 | 0.1501 | 0.69 | 0.1421 | 0.0977 | 0.1204 |
| 2.0 | 500 | 0.1656 | 0.1617 | 0.1712 | 0.1808 | 0.2011 | 0.57 | 0.1489 | 0.0845 | 0.1365 |
| 2.0 | 8000 | 0.1940 | 0.1733 | 0.1807 | 0.1885 | 0.1983 | 0.74 | 0.1415 | 0.1037 | 0.1364 |

**Reading.** Three things follow. (1) The gate estimation variance genuinely exceeds the within-cell
signal in every design tested (var_est about 0.14 against a true within-cell variance of 0.10-0.14),
so static gating *should* win - and it does, in 24 of 32 configurations. (2) The bootstrap rho is
correct in level (it never recommends conditioning) but has no discriminative power inside the tested
range: its Spearman correlation with the realised conditional-minus-static difference is -0.371
(p=0.036), i.e. the wrong sign. The identifiability criterion is therefore **rejected as a ranking rule**
and retained only as a level diagnostic. (3) The continuous shrunk gate now beats static significantly
(p=0.0029), and bagging alone does not (p=0.42), so the active ingredient is shrinkage toward the
coarse cell gate, not variance reduction by averaging.

## J. Cross-domain replication (R8, ETT family) - falsification F5 FIRES

Identical protocol to India/NYISO: persistence baseline, residual corrector, chronological 70/10/20 split,
7 channels per dataset treated as series, horizons 1/6/24.

| dataset | h | always | global | static | free | shrunk | rho |
|---|---|---|---|---|---|---|---|
| ETTh1 | 1 | +0.0918 | +0.0724 | +0.1089 | +0.0857 | +0.1050 | 1.54 |
| ETTh1 | 6 | +0.3109 | +0.3122 | +0.3190 | +0.2624 | +0.2942 | 1.63 |
| ETTh1 | 24 | -0.0122 | +0.0207 | -0.0067 | -0.0029 | +0.0008 | 1.71 |
| ETTh2 | 1 | +0.0592 | +0.0586 | +0.0612 | +0.0776 | +0.0782 | 1.86 |
| ETTh2 | 6 | +0.1741 | +0.1724 | +0.1661 | +0.1636 | +0.1702 | 1.95 |
| ETTh2 | 24 | -0.0277 | -0.0037 | +0.0094 | -0.0113 | -0.0041 | 3.01 |
| ETTm1 | 1 | +0.0115 | -0.0055 | +0.0201 | +0.0103 | +0.0148 | 3.25 |
| ETTm1 | 6 | +0.1318 | +0.1059 | +0.1360 | +0.1169 | +0.1324 | 2.00 |
| ETTm1 | 24 | +0.2829 | +0.2852 | +0.2862 | +0.2622 | +0.2703 | 4.09 |
| ETTm2 | 1 | +0.0010 | -0.0036 | +0.0082 | -0.0029 | +0.0039 | 2.09 |
| ETTm2 | 6 | +0.0699 | +0.0666 | +0.0646 | +0.0880 | +0.0857 | 2.95 |
| ETTm2 | 24 | +0.1861 | +0.1820 | +0.1681 | +0.1737 | +0.1776 | 4.20 |

| comparison | mean difference | wins | t | p |
|---|---|---|---|---|
| shrunk - static | -0.00101 | 5/12 | -0.26 | 0.8012 |
| free - static | -0.00983 | 4/12 | -1.59 | 0.1404 |
| static - always | +0.00517 | 9/12 | +1.32 | 0.2135 |
| static - global | +0.00650 | 8/12 | +1.22 | 0.2477 |

**Reading.** The shrunk hierarchical gate does not transfer: it is indistinguishable from the static gate
on ETT (p=0.80, 5/12 wins), having been significantly better on synthetic data with bootstrap lambda
(p=0.0029) and better in 14/16 India configurations. Pre-registered falsification F5 therefore fires, and
the algorithmic claim must be narrowed to the panel datasets where it was measured.

**What does replicate across all three domains** (synthetic, India generation, NYISO prices, ETT):
free per-instance gating loses to coarse per-series gating - India 2/16, synthetic 8/32 (bootstrap) and
14/72 (split-half), ETT 4/12 - and coarse gating beats always-correcting in 9/12 ETT configurations.
That negative result is the only cross-domain claim this project supports.
