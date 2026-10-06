# RE-FUSED-6 defect and claim lineage (versioned)

Every row is a discovered problem, what it invalidated, the fix, and what the fix changed.
Claims are graded: **established** / **conditionally established** / **weakened** / **reversed** /
**falsified** / **open**. Nothing is deleted; superseded results stay on the record.

---

## Defects

| ID | Discovered | Problem | Invalidated | Fix | Outcome |
|---|---|---|---|---|---|
| D1 | RE-FUSED-5 audit | experimental-unit ambiguity (28 cells vs 140 runs reported interchangeably) | presentation of S1 | canonical unit = one (config, seed) run, declared in `M1_canonical_definitions.json` | resolved in M1 |
| D2 | RE-FUSED-5 audit | criterion test pooled 72 per-seed synthetic runs with 16 per-config real means | the central R7 statistic | aggregation only at analysis time, never across unit types | resolved in M1 |
| D3 | RE-FUSED-5 audit | seasonal-naive at h=7 is literally persistence (`shift(-h+7)=shift(0)`) | 4 of 12 R5 "replications" | baseline refused when h ≡ 0 (mod season); integrity check C5 enforces it | resolved in M1 |
| D4 | RE-FUSED-5 audit | resolution study confounded (5-min used 5 zones from 2022; hourly/daily all 15 zones, full span) | the monotone-resolution claim | resolution sweep deferred to M6 with matched zones/windows/clock-horizons | deferred to M6 |
| D5 | RE-FUSED-5 audit | gates fitted under squared loss, scored on MAE | every real-data skill number | both losses computed per run; squared loss primary | resolved in M1 |
| D6 | RE-FUSED-5 audit | asymmetric clipping (fine rungs clipped, coarse rungs not) | all rung comparisons | identical `[0, 1.5]` clip for every rung; clipped fraction logged per rung | resolved in M1 |
| D7 | RE-FUSED-5 audit | two aggregations (macro 0.0674 vs pooled 0.0902) reported side by side | cross-section comparability | both logged per run, never mixed | resolved in M1 |
| **D8** | **M1 audit** | per-instance gate `num/den` explodes as den→0: measured gate variance **1e18–1e19**, 39.9% of values inadmissible, gate abs-max ~1e9 | **every fine-rung result in RE-FUSED-5 and M1** — "price of adaptivity" was confounded with a broken estimator | weighted-least-squares gate: regress z = R/r̂ on x with weights r̂² (exactly the projection defining g\*), so small-r̂ points are damped analytically rather than clipped | M1b: ρ now 0.97–1.42 (physical); inadmissible 47.7%→24.6%; fine rung improves by ~0.010 but **still loses to per-series** → the price-of-adaptivity conclusion survives the fix |
| **D9** | **M1b** | ~20–25% of WLS gate values still exceed 1.5, i.e. the *corrector* is systematically under-scaled (l2-regularised HGB shrinks residual magnitude), so all gate estimators operate off-centre | fine-rung comparisons remain mildly handicapped | reparameterise: absorb the global gate into the corrector (r̂ ← g₁·r̂) before fitting finer rungs, so admissible gates centre near 1 | **open — to be tested** |
| **D10** | **M1b** | ρ = W/V is not a valid identifiability criterion: it compares between-point variation with *bootstrap* variance and omits the estimator's **bias** relative to g\*. With the stable estimator ρ>1 in most configs while the fine rung still loses | the ρ crossing rule as both a ranking rule (falsified in M1) and a level rule (falsified in M1b) | replace the variance-ratio diagnostic with penalised validation risk (SRM), which measures achieved risk including bias; ρ is retained only as a descriptive diagnostic | **theory correction — carried into M5** |

---

## Claim lineage

| Claim | RE-FUSED-5 status | RE-FUSED-6 status after M1/M1b | Evidence |
|---|---|---|---|
| Gate identity g\* = SNR/(1+SNR) and degradation condition | established (synthetic) | **established**, unchanged | S1: corr 0.9915 over 28 cells × 5 seeds; sign correct 25/28 cells |
| Free per-instance gating loses to coarse gating | established (4 settings) | **established and strengthened** — survives the D8 estimator fix, and the gap widens with horizon | M1: −0.058 vs per-series, 15/15 configs significant (Holm); M1b: WLS fine rung still loses in every config measured |
| Shrunk hierarchical gate beats the coarse gate | established (India, p=0.0029 synthetic) | **REVERSED** | M1: per-series − shrunk = +0.0129, 14/15 configs, 12 significant; sign holds under all four metric/aggregation variants |
| ρ predicts which granularity wins | falsified as ranking rule | **falsified as ranking and as level rule**; cause diagnosed (D10: missing bias term) | M1 Spearman −0.561 (p=0.030); M1b ρ>1 while fine rung loses |
| The ladder has an interior optimum | new in RE-FUSED-6 | **established on real data** | M1: per-series best in 10/15, no-correction best in the 4 drifting configs, per-cell 1/15, per-instance never |
| Coarse beats per-cell | new | **weakened** — a plateau, not a peak | block bootstrap: CI excludes zero in only 3/15 configs |
| Oracle headroom | new | **established**: best feasible rung captures 22.3% of the oracle gate | M1 audit section A |
| Heterogeneity moves the oracle rung finer | new | **established in controlled synthetic conditions** | M3 smoke: oracle rung = global at between=0.0, per-cell at between=0.3 |
| Correcting a drifting baseline | established (RE-FUSED-5, ungated −0.29) | **established and refined**: with a drifting baseline every rung is negative and no-correction is optimal; coarse rungs are least harmful | M1 operator-schedule configs (4/4) |
| Context blocks add value | falsified (permutation control) | unchanged, carried forward | RE-FUSED-5 R2 |
| Cross-domain transfer of the shrunk gate | falsified (ETT p=0.80) | unchanged; to be re-tested with the stable estimator in M8 | RE-FUSED-5 R8 |
| Resolution value rises as resolution refines | claimed | **withdrawn pending M6** (confounded by D4) | — |

---

## M1b verdicts (appended after the independent estimator audit)

**D8 is resolved, and resolving it strengthened rather than weakened the central claim.**

Estimator validation against a closed-form target g\* = m²/(m²+σ_ε²), 6 settings × 8 seeds = 48 runs,
paired by (setting, seed):

| estimator | risk improvement vs raw ratio | wins | p | gate abs-max on real data |
|---|---|---|---|---|
| ridge ratio | +0.0229 | 48/48 | 2.9e-10 | 1.5e3 |
| WLS | +0.0378 | 46/48 | 5.8e-08 | 4.8 |
| WLS bagged | **+0.0481** | 48/48 | 2.6e-10 | **4.3** |
| raw ratio (M1 form) | — | — | — | 8.6e10 |

Correlation with the true gate remains low throughout (0.11 at σ_ε=0.3 to 0.47 at σ_ε=1.5), and MSE
against g\* is comparable to the magnitude of g\* itself. **The per-point gate is only weakly
identifiable even in clean synthetic structure with 4,000 validation points.**

Real-data consequence (150 paired config × seed comparisons per estimator):

| fine estimator | minus per-series | wins | p |
|---|---|---|---|
| WLS | −0.0585 | 0/150 | 1e-60 |
| ridge ratio | −0.0376 | 1/150 | 1.8e-50 |
| WLS bagged | −0.0531 | 0/150 | 3.4e-64 |
| raw ratio | −0.0578 | 0/150 | 2.3e-77 |

Best rung per config is unchanged from M1 (per-series 10, none 4, per-cell 1).

| Claim | Status after M1b |
|---|---|
| Adaptivity in the gate carries a statistical price | **established** (was conditionally established): holds for all four estimators, including the one that is provably best against a known target |
| Mechanism: the gate is a second-order functional of the same data that produced the corrector, so its per-point identifiability is intrinsically poor | **established** by direct measurement against a known g\* |
| ρ = W/V decides gate granularity | **falsified at both level and ranking**: with the stable estimator ρ>1 in 13/15 configs while the fine rung wins 0/15; Spearman −0.389 (p=0.15), wrong sign |
| Shrinkage toward a coarse gate recovers the loss | **no**: rung4 variants (0.0764, 0.0758) sit well below per-series (0.0991) and per-cell (0.0944) |
| Reproducibility of shared rungs across independent scripts | **established**: M1 vs M1b max abs difference 0.00e+00 |

**D9 status:** open, under test in M1c. Fine-rung clipping remains 21–23% because the corrector
under-predicts residual magnitude; the test absorbs the global gate into the corrector so that the
admissible band is centred, giving the fine rungs their fairest possible test.

---

## M1c / M1d / M3-M5 / M4 / M5b verdicts (appended 13 Sep 2026, overnight loop)

| ID | Discovered | Problem | Fix | Outcome |
|---|---|---|---|---|
| D9 | M1b | corrector under-scaled, gates off-centre | absorb global gate into corrector | **resolved** (M1c): fine rung +0.0207 (49/60, p=9e-12), gap to per-series narrows 36%, fine still wins 0/60 |
| D11 | M1c | fixed cap [0,1.5] binding after reparameterisation | cap sweep {1.0,1.5,3.0,inf} | **resolved** (M1d): per-series best at every cap; fine deficit widens -0.014 -> -0.068 as cap loosens (clipping was *helping* the fine rung) |
| D12 | M4 | corrector possibly data-starved vs deep baselines | internal control: persistence on identical rows and budget | **resolved**: persistence +0.244 with the same budget, so starvation does not explain the deep-baseline result |
| D13 | M3/M5 audit | **in-sample selection optimism**: gates fitted and scored on the same validation data; val-test rank corr 0.126; argmin always chose the finest rung (regret identical to fixed rung5, 0.03839) | nested split: fit gates on V1, select on V2 | **resolved** (M5b): regret 0.0384 -> 0.0031 (12.5x), exact recovery 0% -> 50.3%, better in 365/370 runs (p=1e-151) |
| D14 | M5b | nested argmin ignores its own noise: loses to fixed coarse default at n_val=300 (+0.0089, p=0.001); worst-case regret 0.065 vs 0.019 | parsimony (one-standard-error) rule on time blocks, alpha=0.10 fixed a priori | **open - pre-registered hypothesis H-M5c, tested on fresh real data** |

| Claim | Status |
|---|---|
| Validation-only rung selection can work | **conditionally established**: nested split ties the hindsight-best fixed policy overall (+0.0004, p=0.36), beats fixed per-series (p=2.5e-11), and is consistent (regret 0.0097 at n=300 -> 0.0006 at n=20000). Conditional on validation size and on the heavy-tail caveat (D14). |
| Proposition-1 plug-in (Mallows) penalty corrects selection optimism | **falsified as stated**: regret 0.037 (x1) / 0.029 (x2); x2 better than x1 means true optimism exceeds twice the first-order term - the per-instance gate's effective degrees of freedom far exceed the local-cell approximation |
| Correction value falls as the baseline strengthens | **established** (M4): per-series skill +0.244 persistence (test-portion MAE 2.77) vs -0.032 DLinear (1.65), +0.009 NHITS (1.49), -0.068 PatchTST (1.46) on ETTh1 [D17: MAEs corrected to the scored test portion; whole-CV-window MAEs are 2.45 / 1.58 / 1.43 / 1.40, same ordering]; per-series is the best rung for all four baselines |
| Resolution value rises monotonically as resolution refines | **falsified** (M6, matched zones/window/clock horizon): at a 1-hour horizon 15-min scores 0.097 vs 5-min 0.081 (day-ahead baseline); RE-FUSED-5's +0.445 shrinks to +0.081 once matched. What survives: correction recovers more from the weaker baseline, and the day-ahead price is uncorrectable at its own 24 h horizon (residual R2 -0.006) |

## M8 verdicts (appended 13 Sep 2026, 01:30)

| ID | Discovered | Problem | Fix | Outcome |
|---|---|---|---|---|
| D15 | M8 audit | absolute skill gaps scale with how predictable a dataset is: resid_R2_val correlates +0.825 with best coarse skill (p=0.001), so raw descriptor correlations with the fine-rung gap are partly mechanical | test descriptors against the RELATIVE gap (fine - coarse)/coarse and by rank partial correlation controlling best coarse skill, with Holm correction over 7 descriptors | **resolved**: the headline resid_R2 correlation (abs rho -0.881) does not survive on the relative scale (rho -0.427, p=0.17, Holm 0.67) |

| Claim | Status after M8 (6 datasets x 2 horizons x 5 seeds, persistence baseline, D8/D9 fixes) |
|---|---|
| Per-instance gating loses to the best coarse rung across domains | **established**: loses in 11/12 configs (binomial p=0.0032) and 55/60 runs; never the best rung (0/12). Best rungs: per-cell 6, shrunk 3, per-series 2, global 1 |
| RE-FUSED-5's "ETT transfer failure" | **dissolved**: under the canonical protocol ETT behaves like India - the coarse or intermediate rung wins in 8/8 ETT configs. The specific best coarse rung (series vs cell vs shrunk) remains dataset-dependent and plateau-like |
| Residual predictability predicts when fine gating loses | **not established** - a scale artifact (D15) |
| Between-series gate heterogeneity predicts the proportional fine-rung loss | **suggestive only**: relative-gap rho -0.734 (Holm p=0.046), partial rho -0.685 (Holm p=0.12), n=12. Mechanistic reading: heterogeneity aligned with series is captured cheaply by a coarse per-series gate, leaving per-instance adaptivity proportionally less to recover |
| Validation size per cell predicts the gap at dataset level | **not supported** (relative rho +0.22, p=0.50) - confounded with dataset identity at only 3 levels; the one exception (NYISO h=24, fine wins +0.015) has the largest n per cell (2,928), consistent with but not proof of the estimation-cost mechanism |

## M7 verdicts (appended 13 Sep 2026, 01:35)

India daily; 3 baselines x h in {1,3,7} x 5 seeds; per-series gate with D9 reparameterisation.

| Claim | Status |
|---|---|
| Block-LCB is the best deployment rule | **conditionally established - baseline-dependent**. Drifting operator schedule: best skill (0.0245 vs 0.0032 always) AND best safety (0.4 degraded series vs 6.8; worst -0.047 vs -0.446). Stationary persistence/roll7: dominated by CRC-style clip90 (0.1139 vs 0.0929; 0.1263 vs 0.0941) and too conservative (withholds 5.5-6.5 series, 4.6-4.9 of which would have helped) |
| Block-LCB is a novel safety principle | **no** - deploy iff a lower confidence bound on improvement is positive is standard high-confidence safe policy improvement / selective deployment. Retained as a correct tool, not a contribution |
| Per-instance safety mechanisms (clipping, epistemic-uncertainty selection) protect against a drifting baseline | **falsified**: clip90 -0.0015 (6.7 degraded), unc_selective 0.0045 (6.7 degraded), worst cases -0.43 to -0.44 - consistent with Proposition 7(iii): drift is series- and time-level, invisible to per-instance uncertainty |
| Conformal-quantile (worst-block) deployment | **safe but nearly useless**: 0 degraded everywhere, coverage 0.3-7.7%, withholds 16.7-18.9 series |
| Pooled over baselines | block-LCB vs always -0.0083 (p=0.037) with degraded 0.78 vs 3.71; vs clip90 -0.0091 (p=0.06) with 0.78 vs 3.04 |

### Pre-registered hypothesis H-M7b (registered 01:35, before any result)
Series-level block-LCB and instance-level clipping address different failure modes (Proposition 7 vs
Proposition 1), so the hybrid **clip90_lcb** (deploy per series iff block-LCB > 0; when deployed, clip at
the validation 90th percentile) should combine them. Criteria:
 (i)  FRESH stationary data (ETTh1, ETTh2, NYISO hourly): clip90_lcb mean skill >= block_lcb, paired
      Wilcoxon not significantly worse (p>=0.05 for block_lcb > clip90_lcb).
 (ii) FRESH stationary data: clip90_lcb degraded series <= clip90 degraded series.
 (iii) India operator schedule (IN-SAMPLE, labelled as such): clip90_lcb degraded <= block_lcb degraded + 0.5.
Falsified if (i) fails. alpha = 0.10 and the 90th percentile are fixed from M7, not re-tuned.

### Pre-registered hypotheses H-M9 and H-M10 (registered 13 Sep 2026 ~01:38, before post_chain3 was launched at 01:39; from reviewer audit round 1)
Disclosure: a one-run smoke test of each script (India persistence h=1 seed 0 for M9; India roll7 h=1 seed 0
at the three origins for M10) was executed to check that the code runs before these criteria were written.
Criteria were fixed from the audit objections, not from those numbers, and nothing was tuned afterwards.

**H-M9 (reviewer 1.3 - neural gate family).** Coarse dominance transfers to the field's default gate,
g = 1.5*sigmoid(MLP(features, series one-hot)) trained on V1 with early stopping on V2. Runs: India
{persistence, roll7, snaive7} x h in {1,3}; ETTh1, ETTh2 persistence x h in {1,24}; 5 seeds.
 (i)  neural gate NOT significantly better than per-series over all runs (one-sided Wilcoxon p >= 0.05);
 (ii) neural gate beats per-series in at most half of the configs (mean over seeds).
Falsified if either fails -> the dominance claim is restricted to boosted/ratio gates.
per_series_V1 (per-series on V1 only) is reported as the matched-data control but is not part of the verdict.

**H-M10 (reviewer 2.1 - single test period).** The India ladder ordering is a property of the method, not of
the 2024-25 test period. Origins test 2022 / 2023 / 2024-25, 4 baselines x h in {1,3} x 5 seeds, identical
protocol (D8 WLS bagged instance gate, D9 reparameterisation, clip [0,1.5]). For EVERY origin:
 (i)  the per-instance rung is the best rung in at most 1 of 8 configs;
 (ii) the per-instance rung is below the best coarse rung (global/series/cell) in more than half of the configs.
Falsified if any origin fails either criterion.

**Amendment to H-M9 (~01:50, before the full run started at 02:02; disclosed).** A second smoke test (India persistence/roll7/snaive7,
h=1, seeds 0-2 - 9 of the 50 planned runs) showed a data-budget asymmetry in the competitor: the early-stopped
neural gate trains on V1 only, while per-series uses all of V (it beat the matched per_series_V1 in 9/9 runs).
To give the competitor its strongest standard form, **neural_gate_refit** was added: the epoch count chosen by
early stopping on V1/V2 is re-used to retrain on all of V. The H-M9 verdict is taken against whichever neural
variant has the higher mean skill over all runs - a change that can only make H-M9 harder to pass. The smoke
numbers (refit vs per-series -0.0023, 5/9 runs, p=0.82) come from runs that are also part of the full run; the
full-run verdict is therefore not fully independent of them, and is labelled as such.

## M5c verdict (appended 13 Sep 2026, ~01:58) - H-M5c FALSIFIED

Real data (India 4 baselines x h in {1,2,3,7}, ETTh1/ETTh2 persistence h in {1,24}; 95 runs). All policies are
scored with the SAME gates refit on the full validation set, so the comparison has no data-budget confound.

| Criterion (registered 01:20) | Result | Verdict |
|---|---|---|
| mean regret: parsimony <= nested argmin | 0.0223 vs 0.0158 (parsimony lower in 13/95, p=0.007) | **FAIL** |
| p95 regret: parsimony < nested argmin | 0.0797 vs 0.0736 | **FAIL** |
| not significantly worse than fixed per-series | +0.0203, lower in 6/95, p=3.7e-10 | **FAIL** |

Synthetic replication (185 runs) points the same way: parsimony 0.0117 vs argmin 0.0031 (p=2.4e-4).

Diagnosis (audit of selections, not a re-test):
* The parsimony test lacks power at the available block counts (13 ETT, 27 India blocks in V2): it picks a rung
  COARSER than the oracle in 54% of real runs (global 46/95 times, oracle global 4/95). Under-selection is costly -
  fixed global regret is 0.031 on stationary India.
* The nested argmin errs in both directions (27% too coarse, 32% too fine on real data).
* **Both selectors lose to fixed coarse defaults on real data**: fixed per-series 0.0020, fixed per-cell 0.0026,
  argmin 0.0158 (vs per-series p=1.1e-7), parsimony 0.0223. The gap is largest with the drifting operator schedule
  (argmin 0.043, parsimony 0.058, per-series 0.003) and on ETT (argmin 0.020, per-cell 0.001).

| ID | Discovered | Problem | Fix | Outcome |
|---|---|---|---|---|
| D16 | M5c audit | parsimony rule falls back to rung0 (no correction) when V2 has < 4 blocks; happens only at synthetic n_val=300 (3 blocks), regret 0.080 | not fixed: it does not touch the real-data verdict (13-27 blocks everywhere) and the rule is withdrawn | **recorded; rule withdrawn** |

| Claim | Status after M5c |
|---|---|
| A parsimony (one-SE-style) rule fixes nested-argmin selection noise | **falsified** (pre-registered, real data) |
| Validation-only rung selection can work (M5b, synthetic) | **downgraded**: holds in the synthetic generator; on real data both selectors are dominated by a fixed per-series or per-cell gate |
| Fixed per-series is a better policy than any rung selection | **suggestive, NOT yet established**: per-series was itself identified as the best rung on the M1 India test period and M8 used the same ETT test split, so this comparison is partly in-sample. Tested out of sample in H-M10b |

### Pre-registered hypothesis H-M10b (registered ~01:59, before any fresh-origin result; M10 had not started)
Extension of M10: at each origin, the M5c selection protocol (V1 = first half of the validation year, V2 = second
half, weekly blocks, alpha = 0.10, identical code `m5c.real_one`) is run for India 4 baselines x h in {1,3} x 5
seeds. origin_2024 overlaps M5c Part B and is labelled in-sample. On the FRESH origins (test 2022, test 2023):
 (i)  at each fresh origin, neither nested argmin nor nested parsimony has significantly lower regret than the fixed
      per-series gate (one-sided Wilcoxon p >= 0.05);
 (ii) pooled over fresh origins, fixed per-series mean regret <= both selectors.
Falsified if any part fails -> rung selection pays on fresh test years and the fixed default is not recommended.
Only origin_2024 is smoke-tested before the run.

## M7b verdict (appended 13 Sep 2026, 02:03) - H-M7b FALSIFIED (criterion i)

Fresh stationary data: ETTh1, ETTh2 (persistence, h in {1,24}, 5 seeds), NYISO hourly (persistence and day-ahead,
h in {1,24}, 3 seeds) = 32 runs, 320 series-evaluations. In-sample: India persistence/roll7/operator schedule, h in {1,3}, 3 seeds.

| Criterion (registered 01:35) | Result | Verdict |
|---|---|---|
| (i) fresh: clip90_lcb >= block_lcb | -0.0321, higher in 1/32 runs, p=9.3e-10 | **FAIL** |
| (ii) fresh: clip90_lcb degraded <= clip90 | 0.00 vs 0.16 per run | PASS |
| (iii) in-sample drift: clip90_lcb degraded <= block_lcb + 0.5 | 0.17 vs 0.44 | PASS |

Secondary analysis on the same fresh data (NOT pre-registered; every rule was fixed before the run, nothing tuned):

| Comparison (fresh, 32 paired runs) | Skill difference | Degraded series (of 320) |
|---|---|---|
| block_lcb - always | -0.0041 (p=3.9e-4) | 4 vs 10 |
| clip90 - always | **-0.0329** (p=2.3e-9) | 5 vs 10 |
| unc_selective - always | -0.0095 (p=0.036) | 8 vs 10 |
| block_lcb - clip90 | +0.0289 (p=0.028) | 4 vs 5 |

Per dataset, clip90 keeps 0.019 of 0.089 (NYISO day-ahead) and 0.013 of 0.099 (NYISO persistence); on ETT it costs
0.002-0.009. In-sample India: block_lcb vs always is +0.034 under the drifting schedule (degraded 0.5 vs 7.2,
worst series -0.08 vs -0.55) and -0.011 / -0.022 on the stationary baselines.

| Claim | Status after M7b |
|---|---|
| clip90 is the best deployment rule on stationary baselines (M7, India) | **falsified out of sample**: on fresh data it is the worst non-trivial rule (-0.033 vs always). Mechanism: on spiky targets (NYISO prices) the largest corrections carry most of the squared-loss gain, and a validation-quantile cap removes exactly those. The M7 result was specific to India's light-tailed residuals |
| Series-level and instance-level protection combine (hybrid clip90_lcb) | **falsified** (pre-registered) - it inherits clip90's loss |
| block_lcb as the default safety rule | **supported, secondary (not pre-registered)**: the only rule without a large loss anywhere - costs 0.004 skill on fresh stationary data while cutting degraded series from 10 to 4 of 320, and gains +0.034 under drift (in-sample). Still not novel (standard high-confidence deployment test) |
| Per-instance safety (clipping, epistemic uncertainty) is the wrong level of protection | **strengthened**: fails against drift (M7) AND costs skill on fresh stationary data (M7b) |

## M9 / M10 / M10b verdicts (appended 13 Sep 2026, ~08:00; runs finished 02:22, PC was idle, then put to sleep by the safety net at 05:25 until the session resumed)

### H-M9 SURVIVES (neural sigmoid gate; 10 configs x 5 seeds = 50 paired runs)
| Comparison | Mean skill diff | Neural better | Wilcoxon p |
|---|---|---|---|
| neural_gate_refit - per_series | **-0.0054** | 16/50 | 3.0e-5 |
| neural_gate (early-stopped, V1 only) - per_series | -0.0114 | 6/50 | 5.9e-10 |
| neural_gate_refit - per_series_V1 (matched data) | +0.0063 | 40/50 | 7.6e-4 |
| neural_gate_refit - global | +0.0164 | 42/50 | 1.2e-8 |

Neural refit beats per-series in 2/10 configs (ETTh2 h=1 +0.0017; India persistence h=1 +0.0004). Largest loss:
ETTh1 h=1 (0.173 vs 0.207). Degraded series per run: per-series 0.84, neural refit 1.36. Reading: the neural gate
learns more than a global scalar and, given matched data, more than per-series; with the full validation set it
still loses to the per-series least-squares gate. Pre-registered verdict computed against the stronger variant.
Caveat: fixed a-priori hyperparameters (64 hidden, lr 1e-3, wd 1e-4) -> robustness check **M9b** (tuned grid,
selected on V2 only) registered below.

### H-M10 SURVIVES (rolling origin, India, 3 test years x 8 configs x 5 seeds)
| Origin (test year) | Instance best | Instance below best coarse | Mean gap | Best rungs |
|---|---|---|---|---|
| 2022 | 0/8 | 6/8 | -0.0027 | shrunk 5, series 2, cell 1 |
| 2023 | 1/8 | 7/8 | -0.0075 | series 3, shrunk 3, none 1, instance 1 |
| 2024-25 (canonical) | 0/8 | 6/8 | -0.0279 | series 5, shrunk 2, cell 1 |
Pooled: instance best in 1/24; below best coarse in 19/24 (binomial p=0.0033).

**New finding - which coarse rung wins is period-dependent (claim downgraded).** In 2022 the shrunk rung wins 5/8
configs and has the lowest fixed-policy regret (0.0035 vs per-series 0.0117); in 2024-25 per-series wins and shrunk
is worst among coarse policies (0.0259 vs 0.0020). The M1 statement "per-series beats shrunk, reversing RE-FUSED-5" is
therefore a property of the 2024-25 test period, not a stable ordering. The operator schedule also changes character:
correctable in 2022 (shrunk +0.30), uncorrectable in 2023 (every rung <= 0 at h=3), marginal in 2024-25 - the drift
that Proposition 7 describes, observed directly across origins.

### H-M10b SURVIVES (rung selection vs fixed per-series on fresh test years)
| Origin | argmin - fixed_series | parsimony - fixed_series |
|---|---|---|
| 2022 (fresh) | +0.0031 (selector lower 10/40, two-sided p=0.033) | +0.0023 (4/40, p=0.0015) |
| 2023 (fresh) | +0.0011 (11/40, p=0.61) | -0.0007 (12/40, p=0.98) |
| 2024-25 (in-sample) | +0.0122 (1/40, p=0.004) | +0.0217 (0/40, p<1e-6) |
Pooled fresh mean regret: fixed_series 0.0095, parsimony 0.0103, argmin 0.0116.
Honest reading: on fresh years selection does not beat the fixed default, but the margin (0.001-0.003) is much
smaller than the in-sample 2024-25 margin (0.012-0.022) - part of M5c's apparent gap was in-sample flattery of the
per-series default.

Fixed-policy regret across the three origins (not a verdict, reported for the recommendation):
| Fixed policy | 2022 | 2023 | 2024-25 | mean | worst |
|---|---|---|---|---|---|
| series | 0.0117 | 0.0072 | 0.0020 | 0.0070 | 0.0117 |
| cell | 0.0127 | 0.0078 | 0.0033 | 0.0079 | 0.0127 |
| shrunk | 0.0035 | 0.0107 | 0.0259 | 0.0134 | 0.0259 |
| global | 0.0244 | 0.0103 | 0.0421 | 0.0256 | 0.0421 |
| instance | 0.0152 | 0.0209 | 0.0509 | 0.0290 | 0.0509 |
Per-series is the minimax-regret and mean-regret fixed policy over the three test years; it is not the best
policy in every year.

### Pre-registered hypothesis H-M9b (registered ~08:00, before the run started at 08:01:38)
neural_gate_tuned: 8-point grid (hidden {32,128} x weight decay {1e-5,1e-3} x lr {1e-3,3e-3}), each trained on V1
with early stopping on V2; lowest V2 loss selected; retrained on all of V for its epoch count. Same 50 runs as M9.
 (i)  tuned neural gate NOT significantly better than per-series (one-sided Wilcoxon p >= 0.05);
 (ii) tuned neural gate beats per-series in at most half of the 10 configs.
Falsified if either fails -> H-M9 is fragile to tuning and the neural-gate claim is withdrawn.

| ID | Discovered | Problem | Fix | Outcome |
|---|---|---|---|---|
| D17 | final-package consistency check against ALL_STATS.md | M4 baseline MAEs in this file (2.45/1.58/1.43/1.40) were the whole cross-validation-window values printed in results/M4.log, while the scored ladder uses the test portion (2.77/1.65/1.49/1.46 in M4_modern_baselines.csv); the source was not stated | corrected to the CSV values with both quoted; every number in the final report is taken from ALL_STATS.md or a CSV | **fixed**; ordering and conclusion unchanged |

## M9b verdict (appended 13 Sep 2026, 08:12; run finished 08:11:10) - H-M9b SURVIVES

Tuned neural gate (8-point grid selected on V2, refit on V), 50 paired runs:
neural_gate_tuned − per_series = **−0.0046** (tuned better in 16/50, two-sided Wilcoxon p=1.2e-3; one-sided
p[neural better]=0.999). Configs won: 3/10 (ETTh2 h=1 +0.0068, India roll7 h=3 +0.0003, India persistence h=1
+0.0002); largest loss ETTh1 h=1 (0.178 vs 0.207). Tuning moved the neural gate from −0.0054 (fixed
hyperparameters, refit) to −0.0046: the loss is not an under-tuning artifact. Selected widths were 128 in 33/50
runs and 32 in 17/50, with no single dominant configuration.

| Criterion | Result | Verdict |
|---|---|---|
| (i) tuned neural gate not significantly better than per-series | p[neural better]=0.999 | PASS |
| (ii) beats per-series in <= half of configs | 3/10 | PASS |

| Claim | Status after M9 + M9b |
|---|---|
| The price of adaptivity extends to the neural sigmoid gate family | **established** for a bounded MLP gate with series one-hot input, fixed or tuned on validation, on India and ETTh1/ETTh2; untested for larger architectures or end-to-end training with the corrector |
