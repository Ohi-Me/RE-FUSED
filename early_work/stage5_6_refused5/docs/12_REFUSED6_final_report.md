# RE-FUSED-6 — final report

Status: research program complete (M1 → M3/M5 → M4 → M6 → M8 → M7 → theory → M5b/M5c/M7b/M9/M9b/M10).
Every number below is taken from `results/refused6/ALL_STATS.md` (generated from the CSVs) or from the verdict
tables in `docs/08_defect_lineage.md`. Lineage: `docs/08`. Theory: `docs/09`. Reviewer audits: `docs/10`.
Reproducibility: `docs/11`.

---

## 1. The finding in one paragraph

Gated residual correction — forecast = baseline + g · r̂, with a learned corrector r̂ and a gate g — is usually
built with the most adaptive gate available (a per-instance or neural gate). Across India daily generation,
four ETT datasets and NYISO prices, three test years, four gate estimators and a tuned neural gate, **the most
adaptive gate is almost never the best one**: a coarse least-squares gate (one per series, or per series × volatility
regime) is as good or better, and
the finer gates lose because they re-solve the residual regression on the small validation set (Proposition 1).
The same price of adaptivity appears one level up: **choosing the gate granularity from validation data does not
beat fixing it in advance** (pre-registered, fresh test years), and **per-instance safety mechanisms lose to a
series-level deployment test**. What remains is a simple, defensible procedure and a mechanism-level account of
why the fashionable alternatives fail. It is a strong applied/empirical contribution; it is not a new
general-purpose ML method.

---

## 2. Final procedure (what RE-FUSED-6 recommends)

The SCALE design in `docs/07` ("Selective Correction with Adaptivity-Level Estimation") assumed the adaptivity
level could be estimated from validation data. That assumption was tested and did not hold on real data (M5c,
M10b), so the name is retired and the selection step removed.

```
Input : baseline B; features x; train / validation / test in time order; series id s; blocks of time b;
        alpha = 0.10; clip interval [0, 1.5]
Stage 1  corrector (train only)
         r_hat <- gradient-boosted regressor  x -> R = Y - B
Stage 2  reparameterise (validation)                                   [D9]
         g1 <- <R, r_hat> / <r_hat, r_hat>  on all validation data;   r_hat <- g1 * r_hat
Stage 3  per-series gate (validation)                                  [D5, D6]
         for each series s with >= 30 validation points:
             g_s <- clip( <R_s, r_hat_s> / <r_hat_s, r_hat_s>, 0, 1.5 )
         otherwise g_s <- clip(g1 recomputed on rescaled r_hat, 0, 1.5)
Stage 4  deployment test per series (validation blocks)                [Prop 7]
         skill_{s,b} <- 1 - MSE(R - g_s r_hat on block b) / MSE(R on block b)
         LCB_s <- mean_b skill - t_{1-alpha, K-1} * sd_b skill / sqrt(K)
         deploy_s <- 1[LCB_s > 0]
Output   Y_hat = B + deploy_s * g_s * r_hat(x)
Not included, because each lost when tested:
         per-instance / neural gates (M1, M1b, M8, M9, M9b, M10); shrinkage toward them (M10: period-dependent);
         validation-based rung selection (M5c, M10b); instance-level clipping or uncertainty filtering (M7, M7b)
```

Cost: one corrector fit, one scalar per series, one t-bound per series. Nothing is tuned on test data.

**Per-series or per-cell?** Both sit on the plateau. Per-series is the minimax-regret fixed policy on India across
three test years (worst regret 0.0117, per-cell 0.0127) and on M5c real data overall (0.0020 vs 0.0026). On the hourly
ETT series, with ~580 validation points per cell, per-cell was better (M5c ETT regret 0.0013 vs 0.0055; best rung in
6/12 M8 configs). The project does **not** establish a rule for choosing between them. Picking one from validation is
exactly the selection step that did not pay (M5c, M10b). Use per-series as the default. Per-cell is a defensible
alternative when each series × regime cell has hundreds of validation points; that suggestion is untested as a rule.

---

## 3. Evidence by stage

| Stage | Question | Key result | Verdict |
|---|---|---|---|
| **M1** India, 15 configs × 10 seeds | where on the adaptivity ladder is the optimum? | skill: none 0, global 0.069, **per-series 0.099**, per-cell 0.094, shrunk 0.086, per-instance 0.041; best rung series 10, none 4, cell 1; oracle capture 0.223; integrity 14/14 | interior optimum; per-instance worst correcting rung |
| **M1b** 4 fine estimators | is the loss an estimator artifact (D8)? | vs per-series: WLS −0.0585 (0/150), WLS-bagged −0.0531 (0/150), ridge −0.0375 (1/150), raw ratio −0.0578 (0/150) | loss survives every estimator |
| **M1c/M1d** | under-scaled corrector (D9) or binding cap (D11)? | rescaling helps the fine rung +0.0207 (49/60) but it still wins 0/60; per-series best at caps 1.0, 1.5, 3.0, ∞ | loss survives both |
| **M3/M5, M5b** synthetic, 370 runs | can the rung be selected? | in-sample argmin regret 0.0384 (always picks finest); nested split 0.0031 (better in 365/370), ties fixed per-cell 0.0027 | selection works **in the synthetic generator** |
| **M4** ETTh1 deep baselines | does correction survive strong baselines? | per-series skill: persistence +0.244 (test MAE 2.77), NHITS +0.009 (1.49), DLinear −0.032 (1.65), PatchTST −0.068 (1.46); per-series is the best correcting rung for all four | correction value collapses as the baseline strengthens |
| **M6** NYISO matched zones/window | does finer settlement resolution create correction value? | 1-h horizon, day-ahead baseline: 15-min 0.097, 5-min 0.081, 1-h 0.077; 24-h day-ahead uncorrectable (residual R² ≈ 0) | RE-FUSED-5 monotone-resolution claim **falsified** |
| **M8** 6 datasets × 2 horizons × 5 seeds | does the ladder transfer? | fine rung never best (0/12), loses in 11/12 configs (p=0.0032), 55/60 runs; best rungs cell 6, shrunk 3, series 2, global 1; residual-R² "predictor" is a scale artifact (D15) | transfers; best coarse rung is dataset-dependent |
| **M7** India deployment rules | how to protect against harmful correction? | drifting schedule: block-LCB 0.0245, 0.4 degraded vs always 0.0032, 6.8; stationary: clip90 looked best (0.114, 0.126) | block-LCB under drift; clip90 appeared best when stationary |
| **M5c** pre-registered | does a parsimony rule fix selection noise? | real data regret: fixed series 0.0020, fixed cell 0.0026, argmin 0.0158, parsimony 0.0223 | **H-M5c falsified** (3/3 criteria) |
| **M7b** pre-registered, fresh data | do series- and instance-level protection combine? | clip90_lcb − block_lcb −0.032 (p=9e-10); clip90 − always −0.033 on fresh data; block_lcb − always −0.004 with degraded series 4 vs 10 of 320 | **H-M7b falsified**; clip90 falsified out of sample |
| **M9** pre-registered, 50 runs | does the neural sigmoid gate escape the price? | refit neural − per-series −0.0054 (better 16/50, p=3e-5); beats per-series in 2/10 configs; beats matched-data per-series +0.0063 | **H-M9 survives** |
| **M9b** pre-registered, tuned grid | is M9 an under-tuning artifact? | tuned neural − per-series −0.0046 (better 16/50, p=1.2e-3); wins 3/10 configs | **H-M9b survives** |
| **M10** pre-registered, 3 test years | is the ordering a single-period artifact? | instance best 0/8, 1/8, 0/8; below best coarse 6/8, 7/8, 6/8; pooled 19/24 (p=0.0033) | **H-M10 survives** |
| **M10b** pre-registered, fresh years | does rung selection beat a fixed default out of sample? | fresh 2022/2023 pooled regret: fixed series 0.0095, parsimony 0.0103, argmin 0.0116; margins 0.001–0.003 | **H-M10b survives**; margin small |

---

## 4. Claim-by-claim: RE-FUSED-5 → RE-FUSED-6

| RE-FUSED-5 claim | RE-FUSED-6 status | Decisive evidence |
|---|---|---|
| Shrunk hierarchical gate (SGRC) is the method | **retired**. Per-series beat shrunk on the 2024-25 test (M1, +0.0129) but shrunk won 5/8 configs on the 2022 test (M10): neither ordering is stable | M1, M8, M10 |
| Free per-instance gating loses to coarse gating | **established and generalised**: 4 estimators, reparameterised, 4 caps, 6 datasets, 3 test years, neural gate | M1b, M1c, M1d, M8, M9, M10 |
| g* = SNR/(1+SNR) shrinkage law | **narrowed**: valid for noise independent of x (Prop 2); for a fixed corrector the optimal gate is m/r̂ (Prop 1) | docs/09 §0 |
| ρ = W/V identifiability criterion ranks granularities | **falsified** as ranking and level rule; missing bias term (D10) | M1, M1b |
| Block-LCB risk control | **kept as the deployment rule**, not claimed as novel: only rule without a large loss anywhere | M7, M7b |
| CRC-style clipping is competitive on stationary baselines | **falsified out of sample** (−0.033 on fresh data; destroys NYISO gains) | M7b |
| Correction value tracks the stationarity of the baseline's bias | **established**, and observed directly across origins (operator schedule +0.30 in 2022, ≤ 0 in 2023) | M1, M7, M10 |
| Finer settlement resolution raises correction value | **falsified** under matched conditions | M6 |
| Context blocks add value | null result carried forward (not re-tested) | RE-FUSED-5 R2 |
| Cross-domain transfer of the shrunk gate | failure **dissolved** as a protocol artifact; what transfers is "fine gates lose" | M8 |
| (new) validation-only rung selection | works in synthetic data; **does not beat a fixed coarse default on real data** | M5b, M5c, M10b |
| (new) a validation descriptor predicts when fine gating loses | **not established** (scale artifact; heterogeneity suggestive only, Holm p=0.046, n=12) | M8, D15 |
| (new) strong deep baselines leave little to correct | **established** on ETTh1 | M4, D12 |

---

## 5. Pre-registered hypotheses (all criteria fixed before the run; disclosures in docs/08)

| Hypothesis | Tested on | Outcome |
|---|---|---|
| H-M5c parsimony selection fixes selection noise | India + ETTh1/ETTh2, 95 runs | **falsified** (3/3) |
| H-M7b hybrid series + instance protection | ETTh1/2, NYISO fresh; India in-sample | **falsified** (criterion i) |
| H-M9 neural gate does not beat per-series | India + ETTh1/2, 50 runs | **survives** (amendment disclosed: refit variant added after a 9-run smoke test) |
| H-M9b tuned neural gate does not beat per-series | same 50 runs | **survives** |
| H-M10 instance rung not best in any test year | India, test 2022 / 2023 / 2024-25 | **survives** |
| H-M10b rung selection does not beat fixed per-series on fresh years | India, test 2022 / 2023 | **survives** |

Two of the program's own hypotheses were falsified and are reported as such. The narrative was changed to match.

---

## 6. M9b result (tuned neural gate)

An 8-point grid (hidden width {32, 128} × weight decay {1e-5, 1e-3} × learning rate {1e-3, 3e-3}) was trained
on V1 with early stopping on V2; the lowest-V2-loss configuration was refit on all validation data.

| Dataset | Baseline | h | tuned neural gate | per-series |
|---|---|---|---|---|
| ETTh1 | persistence | 1 | 0.1779 | **0.2067** |
| ETTh1 | persistence | 24 | 0.0436 | **0.0498** |
| ETTh2 | persistence | 1 | **0.1524** | 0.1456 |
| ETTh2 | persistence | 24 | 0.0170 | **0.0181** |
| India | persistence | 1 | **0.1441** | 0.1439 |
| India | persistence | 3 | 0.1159 | **0.1193** |
| India | roll7 | 1 | 0.2019 | **0.2078** |
| India | roll7 | 3 | **0.0863** | 0.0860 |
| India | snaive7 | 1 | 0.3311 | **0.3333** |
| India | snaive7 | 3 | 0.1273 | **0.1334** |

Mean difference −0.0046 (tuned better in 16/50 runs, Wilcoxon p = 1.2e-3); per-series wins 7/10 configs.
Tuning narrowed the gap from −0.0054 to −0.0046, so the result is not an under-tuning artifact. **H-M9b survives.**

---

## 7. Defects found and fixed (D1–D17)

D1–D7 RE-FUSED-5 protocol (unit ambiguity, pooled unit types, seasonal-naive ≡ persistence, resolution confound,
loss mismatch, asymmetric clipping, mixed aggregations) · D8 exploding ratio gate → WLS · D9 under-scaled
corrector → reparameterisation · D10 ρ criterion missing bias · D11 binding cap → sweep · D12 corrector starvation
ruled out · D13 in-sample selection optimism → nested split · D14 nested-argmin noise → parsimony (falsified) ·
D15 scale confound in descriptors → relative gap + Holm · D16 parsimony fallback to "no correction" with < 4 blocks
(rule withdrawn) · D17 M4 MAEs quoted from the wrong window (corrected). None was fixed by looking at test results
of the hypothesis it affected; each fix was re-run and re-audited.

---

## 8. Theory status

Propositions 1–8 (`docs/09_theory.md`) are correct and each is tied to the experiment that tests it: exact excess
risk of a gate (1), shrinkage identity (2), estimation–approximation over partitions (3), unimodality conditions
(4), hold-out selection bound (5), in-sample optimism (6), the limit of validation under drift (7), and when
hold-out selection cannot beat a fixed default under period shift and data halving (8). All are elementary and
essentially known (forecast-combination puzzle, weighted projections, Mallows/Efron, hold-out bounds). They
explain the results; they are not a theoretical contribution by themselves.

---

## 9. Limitations

* Point forecasts only; no probabilistic evaluation (CRPS, interval coverage).
* Deep baselines on one dataset (ETTh1) and only at h = 24.
* India inference uses three test years and 20 series; seeds capture model randomness, not independent time.
* The neural gate is a small MLP (grid of 8); larger architectures were not tried.
* No finite-sample theory for boosted or neural per-instance gates with data-dependent capacity.
* The deployment rule's gain under drift is measured in-sample on India (M7/M7b); fresh-data evidence for it is
  on stationary series only, where it costs 0.004 skill.
* Multiplicity: Holm applied within M1 and M8; the headline claims in the paper should list their test family.

---


**What is new.** (a) A systematic, falsification-driven demonstration, with pre-registered tests and a public
defect lineage, that the adaptive gates used in recent residual-correction work are dominated by a per-series
least-squares gate across domains, estimators, test years and a tuned neural gate. (b) The same price of
adaptivity at the selection level and at the safety level. (c) A data audit of the Indian panel (price series
unusable) and a de-confounded refutation of the settlement-resolution claim. **What is not new.** The mechanism
(forecast-combination puzzle; Claeskens et al. 2016), the theory (Props 1–8), block-LCB deployment
(high-confidence policy improvement), hold-out selection.

|---|---|---|---|
| International Journal of Forecasting | **A** | the result is a forecasting-methodology finding in the tradition of the combination puzzle, with multi-domain evidence and pre-registration | probabilistic evaluation; more deep baselines |
| Energy and AI | **A−** | as above, lighter ML expectations | positioning versus residual-correction papers |
| IEEE TNNLS | **B** | rigorous empirical ML with mechanism | theory depth; neural gate family beyond MLP |
| AAAI / IJCAI | **B−** | clean negative result with mechanism | "known phenomenon"; limited novelty |
| NeurIPS time-series / forecasting workshop | **A** | exactly the scope: honest negative results and protocol lessons | none serious |
| NeurIPS / ICML / ICLR main track | **not supported** | the central mechanism and all theory are known; the method is deliberately simple; the program's own stopping rule applies (the gap is novelty, which experiments cannot close) | — |
| JMLR | **not supported** | would need new theory (finite-sample selection over data-dependent-capacity gates under drift) | — |

*"The price of adaptivity in gated forecast correction: coarse gates, fixed granularity, and series-level
deployment tests"*.
