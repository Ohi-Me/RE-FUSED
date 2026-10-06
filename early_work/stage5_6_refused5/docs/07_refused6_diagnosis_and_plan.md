# RE-FUSED-6: scientific diagnosis and research plan
Compiled 12 Sep 2026. No paper rewriting here — diagnosis, hypothesis, theory, experiment matrix,
algorithm changes, publication strategy. Every defect below was verified against the scripts and
result files (see `src/` audit run in the session log).

---

# Part I — Diagnosis

## I.1 Verified defects in RE-FUSED-5 (all reproducible)

| # | Defect | Evidence | Severity |
|---|---|---|---|
| D1 | **Experimental-unit ambiguity.** S1 is 28 (SNR × n) cells × 5 seeds = 140 runs. The log reports "25/28 cells", the stats dump "137/140 runs". Both true, inconsistently labelled. | `S1_synth_law.json`: 140 rows, 28 groups | medium (presentation) |
| D2 | **Invalid pooling in the core criterion test.** R7 correlates 72 synthetic *per-seed runs* with 16 India *per-config means* in one Spearman. Units differ, so the pooled statistic (and the "88 configs") is meaningless. | `R7_criterion.csv`: source counts 72/16, 12 synthetic configs | **high** |
| D3 | **Duplicate baseline.** At h=7 the "seasonal-naive" baseline is `shift(-h+7) = shift(0)` = the series itself, i.e. literally persistence. The h=7 rows of R5 are identical across the two baselines, inflating apparent replication. | `R5_multiseed.csv` h=7 rows identical | **high** |
| D4 | **Resolution comparison confounded.** 5-min was trimmed to 5 zones and ≥2022; 15-min to ≥2021; 1-h and 1-d used all 15 zones and the full span. n_test ranges 5,475 → 350,212. Resolution is not the only varying factor. | `resolution_law_nyiso.py` trim lines; `R3` n_test per res | **high** |
| D5 | **Loss definition inconsistent with the theory.** The shrinkage law is derived under squared loss; synthetic experiments use MSE-ratio skill, but *all* real-data experiments use MAE-ratio skill. | grep: synth = MSE, real = MAE | **high** |
| D6 | **Gate clipping asymmetry.** Per-instance and shrunk gates are clipped to [0,1]; the global/static gates are not (observed g up to 1.049 in India, 1.357 in NYISO). The comparison is therefore not like-for-like, and g>1 also reveals a *scale-biased corrector*, which violates the unbiasedness assumption behind g* = SNR/(1+SNR). | `R1_india_diag.csv`, `R3_resolution_law.csv` | **high** |
| D7 | **Two aggregations reported side by side.** For the identical configuration, R1 gives skill 0.0674 (mean of per-state skills) and R5 gives 0.0902 (pooled MAE ratio). | direct recomputation | medium |

## I.2 Further weaknesses (not defects, but rejection risks)
* **W1 Single-seed conclusions.** R1, R2, R4 and R3 use one corrector seed; only R5/S-series are multi-seed.
* **W2 Dependence-blind inference.** CIs are across states or seeds; none across time. Daily state panels and 5-min prices are strongly autocorrelated, so paired t-tests over states overstate significance.
* **W3 Validation used twice** in R4: to fit the gate *and* to set the deployment threshold. Mild selection bias in the reported safety numbers.
* **W4 No hyperparameter search anywhere.** Conservative for our method, but it makes any future comparison against tuned modern baselines unfair in the *other* direction.
* **W5 Two phenomena conflated.** The operator schedule is both the *weakest* baseline and the only *non-stationary-bias* baseline. Current results cannot separate "weak baseline" from "drifting baseline".
* **W6 Theory is an identity, not a theorem.** g* = SNR/(1+SNR) is two lines of algebra under strong assumptions (ε ⊥ m, E[ε|s]=0). No excess-risk statement, no finite-sample bound.
* **W7 No modern baseline anywhere.** Everything is corrected relative to persistence / rolling mean / seasonal-naive / an operator program.
* **W8 Probabilistic track absent on real data** (no CRPS, PIT, coverage).

## I.3 What is actually established (survives the audit)
1. **The gate identity and degradation condition** under squared loss, validated on ground truth: predicted vs empirical optimal gate corr 0.9915 over 28 cells × 5 seeds; degradation sign correct in 25/28 cells.
2. **Free per-instance gating loses to coarse gating** — four independent settings: synthetic S2c (0/10 seeds, twice), S3 (free>static in 14/72 runs), India (2/16 configs), ETT (4/12 configs). This is the most robust finding in the project.
3. **Context blocks carry no incremental value** on the India panel under permutation controls (only renewables at h=7, p=0.020, uncorrected for multiplicity → not established).
4. **Ungated correction of a drifting baseline is catastrophic** (−0.29 skill) and a validation-block confidence rule prevents it (0/20 states degraded at h=3).
5. **Day-ahead → real-time correction has large value at sub-hourly resolution** (residual R² 0.488, skill +0.445 at 5-min), though the *cross-resolution* comparison is confounded (D4).

## I.4 What was falsified (keep, do not bury)
* F-a Per-instance gate superiority (twice, in two designs).
* F-b The identifiability ratio ρ as a *ranking* rule (Spearman −0.371, wrong sign, even with bootstrap variance).
* F-c Cross-domain transfer of the shrunk gate (ETT: −0.001, p=0.80).

## I.5 Where RE-FUSED-5 gets rejected today

---

# Part II — The elevated research object

## II.1 Reframing
RE-FUSED-5 asked "what should the gate be?". That is a *parameterisation* question. The general question is:

> **Given a strong baseline and a residual corrector, how much adaptivity should the correction be
> allowed, and can that be decided before deployment from validation data alone?**

This makes the research object **correction allocation over an adaptivity ladder**, not gating:

```
rung 0  none            g = 0
rung 1  global          g = one scalar
rung 2  per-series      g = g_i                    (K = number of series)
rung 3  per-cell        g = g_{i,h,regime}         (K = cells)
rung 4  shrunk          g = λ ĝ(x) + (1-λ) g_cell  (empirical-Bayes interpolation)
rung 5  per-instance    g = ĝ(x)                   (unrestricted)
```

**Recommended name for the object:** *correction allocation*; for the method: **SCALE — Selective
Correction with Adaptivity-Level Estimation**. Recommended working title:

> **"When Should a Forecaster Correct Its Baseline? The Statistical Price of Adaptivity"**

This is honest about lineage: it is **structural risk minimisation specialised to correction
granularity**. The novelty is not "we invented model selection"; it is (i) closed-form risk per rung
when the baseline is *given and strong*, (ii) an estimable crossing point, (iii) the empirical fact
that the field's default (rung 5, learned sigmoid gates) is provably and measurably wrong in
identifiable conditions, and (iv) a deployment rule that is safe under baseline drift.

## II.2 Central hypothesis (H*)
> For a fixed baseline B and corrector r̂, the risk of rung k decomposes into a monotonically
> *decreasing* approximation term (between-cell gate heterogeneity that the rung can express) and a
> monotonically *increasing* estimation term (parameters per unit of validation data). The oracle risk
> curve over the ladder is therefore **unimodal**, its minimiser is characterised by an explicit
> crossing condition, and that minimiser can be estimated from validation data with bounded selection
> regret.

H* is falsifiable in three distinct ways: non-unimodal risk curves; crossing condition failing to
predict the realised argmin; selection regret not shrinking with validation size.

## II.3 Theory to derive (with honest status)

**Setup.** Y = B + R, corrector r̂(x) fixed after training (treated as given, so all randomness is in
the gate estimation). Partition P_k with cells c, weights p_c, validation counts n_c. Squared loss.

**Proposition 1 (risk of a rung; provable, elementary).** For per-cell least-squares gates,
```
Risk(k) = E[R²] − Σ_c p_c g*_c² E[r̂²|c]  +  Σ_c p_c Var(ĝ_c) E[r̂²|c],
Var(ĝ_c) ≈ Var(R r̂ | c) / (n_c E[r̂²|c]²)
⇒ Risk(k) ≈ Risk_oracle(k) + Σ_c p_c Var(R r̂|c) / (n_c E[r̂²|c]).
```
With K_k cells of roughly equal size, the estimation term grows ≈ K_k·σ̄²/(n_val·Ē[r̂²]), giving the
**crossing rule**: refine the partition only while the *between-cell variance of g\** gained exceeds
σ̄²/(n_val Ē[r̂²]) per added cell. This is the precise form of "adaptivity has a price", and it
predicts RE-FUSED-5's observations (ρ<1 ⇒ stay coarse; scarce validation ⇒ shrink).

**Proposition 2 (shrinkage optimality; classical, must be cited not claimed).** The rung-4 estimator
with λ = W/(W+V), W between-cell (or within-cell) gate variance and V estimation variance, is the
empirical-Bayes/James–Stein estimator and dominates both rung 3 and rung 5 under normality. This
explains why **bagging alone failed (p=0.42) while shrinkage worked (p=0.0029)**: bagging reduces V
but does not borrow strength across cells.

**Proposition 3 (selection regret; standard, provable).** Choosing k̂ by penalised validation risk over
a ladder of size M satisfies an oracle inequality R(k̂) ≤ min_k R(k) + C·sqrt(log M / n_val) under
bounded loss and independent validation blocks. This is what licenses the *pre-deployment* claim.

**Proposition 4 (deployment safety under blocks).** With B time blocks and per-block skill bounded in
[−b,b], a one-sided empirical-Bernstein bound gives P(deploy ∧ degrade) ≤ α. Honest caveat: requires
block independence or β-mixing; the alternative is an e-value/time-uniform construction. Must be
stated as an assumption, with sensitivity to block length reported.

**Proposition 5 (drifting baseline; the novel piece).** Decompose R = R_stat + δ_t with δ_t a slowly
varying baseline bias. Then correction excess risk acquires a term that validation *cannot* estimate
unless the validation window spans comparable drift. This yields a testable prediction: correction of
a drifting baseline degrades out-of-sample in proportion to between-period bias variance — exactly the
operator-schedule catastrophe (sched/actual ratio 0.987→1.058; ungated skill −0.29). This is the most
promising genuinely new theoretical contribution because it connects correction to distribution shift.

**What must NOT be claimed:** that g* = SNR/(1+SNR) is new (it is shrinkage/ridge/James–Stein), or
that ladder selection is new (it is SRM). The contributions are the specialisation, the crossing rule,
Proposition 5, and the empirical programme.

---

# Part III — Experiment matrix

Priority: **M = must-have for submission**, S = should-have, O = optional/low value.
Every row names the reviewer question it answers.

| ID | Experiment | Answers | Success criterion |
|---|---|---|---|
| **M1** | Canonical re-run: one experimental unit (per-seed run), squared loss primary + MAE secondary, symmetric clipping, 10 seeds everywhere, drop the h=7 duplicate baseline | D1–D7, W1 | all tables regenerate; no metric defined two ways |
| **M2** | **Ladder risk curves** on all datasets × horizons: rungs 0–5 + oracle rung; measure realised risk per rung | H*: is the curve unimodal? | unimodality rate reported with CIs |
| **M3** | **Causal synthetic grid**: independently vary between-cell gate variance, n_val, r̂ energy, AR(1) dependence, misspecification, baseline drift | Prop 1 crossing rule | predicted argmin = realised argmin above chance, with regret curve |
| **M4** | **Modern baselines as B**: DLinear, N-HiTS, PatchTST, iTransformer (tuned on validation) on ETT/NYISO/India; then apply the ladder to *their* residuals | W7, the decisive "not weak baselines" objection | does correction still pay when B is strong? theory predicts value shrinks and optimal rung coarsens |
| **M5** | **Pre-deployment selection**: choose the rung from validation only; report selection regret vs oracle rung, per dataset | the headline algorithmic claim | regret → 0 with n_val; beats "always rung 5" and "always rung 3" |
| **M6** | **De-confounded resolution study**: identical zones, identical window, identical clock horizons across 5min/15min/1h/1d; decompose value into baseline error vs residual predictability | D4 | statement about DA→RT value that survives matched conditions |
| **M7** | **Risk control comparison**: block-LCB vs conformal risk control vs selective-prediction thresholding vs CRC-style clipping; accuracy–safety frontier, worst-case degradation, withheld-but-useful rate | is the safety mechanism novel or a re-derivation? | frontier plot; explicit verdict either way |
| **M8** | **ETT failure diagnosis**: measure W, V, n_val, r̂ energy, AR dependence per dataset; test whether the crossing rule explains the reversal | F-c | the same rule that works on India explains ETT, or H* is narrowed |
| **S1** | Probabilistic track: quantile corrector, CRPS/PIT/coverage, λ derived from epistemic uncertainty (ensemble or conformal residual variance) rather than tuned | W8; "where does λ come from?" | λ from uncertainty ≈ λ from bootstrap |
| **S2** | Validation-size sweep on **real** data (n_val = 1, 3, 6, 12 months) | price-of-adaptivity on real data | rung choice shifts coarse→fine as n_val grows |
| **S3** | Multiplicity control across the config grid (Holm / BH), block bootstrap CIs everywhere | W2 | every headline survives correction |
| O1 | More energy datasets (GEFCom, Solar/Wind) | breadth | — |
| O2 | Foundation models (Chronos/TimesFM) as B | topicality | — |
| O3 | RL/dispatch downstream | domain appeal | — |

**Decision gate:** if **M5 fails** (validation-only selection cannot beat a fixed rung), the general ML
claim collapses and the work should be submitted as an energy paper built on the data audit, the
negative result, and the drift theory (Prop 5).

---

# Part IV — Algorithm changes (SCALE)

```
Input: baseline B, corrector r̂ (trained on train), validation V partitioned into time blocks,
       ladder of rungs k = 0..5, risk level α.
1  for each rung k:
2      fit gate parameters on V under SQUARED loss (matches the theory)
3      V_k  <- estimation variance by bootstrap over V (split-half is biased low: shown 2-8x)
4      W_k  <- between-cell variance of the fitted gates, bias-corrected by V_k
5      R̂_k  <- validation risk + penalty  C·sqrt(log M / n_val)        # Prop 3
6  k̂ <- argmin_k R̂_k                                                   # SRM over the ladder
7  if k̂ == 4:  λ <- W/(W+V)                                            # Prop 2, empirical Bayes
8  per cell c: deploy iff one-sided block bound LCB_α(skill_c) > 0      # Prop 4
9  output Ŷ = B + 1[deploy_c] · g_k̂(x) · r̂(x)
```

Changes versus RE-FUSED-5: (a) the gate is no longer *the* contribution — the **rung choice** is;
(b) squared loss for fitting (removes D5); (c) symmetric clipping (removes D6); (d) bootstrap variance
mandatory; (e) λ has a derivation (Prop 2) and optionally an uncertainty-based form (S1);
(f) deployment decision separated from gate estimation with its own block bound.

---

# Part V — Narrative choice

| Candidate claim | Novelty | Theory | Evidence now | Verdict |
|---|---|---|---|---|
| "Shrunk hierarchical gating works" | low | weak | mixed; fails on ETT | **reject** — this is RE-FUSED-5 and it died |
| "Value of information in forecasting" | medium | medium | context null result only | reject — the evidence is negative |
| **"Adaptivity has a statistical price: choose the correction rung, don't learn the finest gate"** | medium-high | Prop 1/2/3 + Prop 5 | strongest multi-domain evidence already in hand (negative + mechanism) | **choose this** |
| "Safe correction under baseline drift" (Prop 5 first) | medium-high | Prop 5 | operator-schedule result is striking | strong *fallback*, or a second paper |

Recommended single claim: **the risk curve over correction adaptivity is unimodal, its minimiser is
predictable from validation data, and the common default of maximal adaptivity is measurably wrong.**
The falsifications become the *evidence* for the claim rather than embarrassments.

---


Ratings: **A** = strong fit, **B** = plausible with the listed work, **C** = weak fit, **D** = not appropriate.

|---|---|---|---|---|
| IEEE TSG | C+ | **A−** | operational relevance of gating | M7 + dispatch-facing framing |
| Energy & AI | B− | **A−** | novelty vs residual-correction literature | M4 modern baselines |
| IEEE TNNLS | C | **B+** | theory depth | Prop 1 + Prop 3 with proofs |
| AAAI / IJCAI | C | **B** | "SRM applied to gates — incremental" | M5 selection regret + M3 causal grid |
| NeurIPS / ICML main | D | **B−** | is the principle general beyond forecasting? is Prop 5 genuinely new? | M5 succeeding **and** Prop 5 proved + tested |
| ICLR | D | **B−** | same, plus representation-learning fit is poor | as above |
| JMLR | D | **C+** | needs full theory + broad empirics | complete proofs and a deterministic-regret treatment |

**Honest bottom line.** RE-FUSED-5 today is an energy paper with a good audit and a robust negative
result. RE-FUSED-6 as specified is a legitimate *borderline* ML submission **only if M5 succeeds** — i.e.
validation-only rung selection demonstrably beats any fixed rung across domains, with the crossing rule
and publish the negative result plus Prop 5 as the scientific core. I will not inflate the ML claim
beyond what M5 returns.
