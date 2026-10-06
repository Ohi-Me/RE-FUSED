# RE-FUSED-6 theory: correction allocation over an adaptivity ladder

Written 13 Sep 2026 during the overnight loop. Every proposition states its assumptions, gives a proof or
proof sketch, links to the experiment that tests it, and states its novelty status honestly. Nothing here
is claimed as new mathematics unless it is.

---

## 0. Setting and a correction to earlier wording

A given baseline B produces residual R = Y − B. A corrector r̂ : 𝒳 → ℝ is trained on a training split and
then **fixed**. A gate g : 𝒳 → ℝ is estimated on a validation split, independent of training. The deployed
forecast is B + g(X) r̂(X), with correction loss

  L(g) = E[(R − g(X) r̂(X))²],  m(x) := E[R | X = x].

**Theory correction T1 (recorded in the lineage).** Earlier documents wrote the oracle gate as
g\* = m²/(m² + σ_ε²) and called it "a second-order functional of the same data". That form is correct only
when the corrector's error ε is noise **independent of X** (the construction used in S1 and in the M1b
estimator validation). When r̂ is a fixed function of x — the actual pipeline — the oracle gate is
g\*(x) = m(x)/r̂(x) (Proposition 1). The substantive claim survives and becomes sharper: a per-instance
gate must re-estimate the residual function itself, on the smaller validation set.

---

## Proposition 1 — exact excess risk of any gate

**Assumptions.** E[R²] < ∞, E[r̂(X)²] < ∞; r̂ fixed (measurable in x); g square-integrable against r̂².

**Statement.** On {r̂(x) ≠ 0} the risk minimiser is g\*(x) = m(x)/r̂(x), and for every gate g

  L(g) − L(g\*) = E[(g(X) r̂(X) − m(X))²].

**Proof.** Conditioning on X = x, E[(R − g r̂)² | x] = E[R² | x] − 2 g r̂ m + g² r̂², a quadratic in g minimised
at g = m/r̂ when r̂ ≠ 0. Substituting, E[(R − g r̂)² | x] − E[(R − g\* r̂)² | x] = r̂²(g − g\*)² = (g r̂ − m)².
Take expectations. ∎

**Consequence (why per-instance gates lose).** The excess risk of a gated corrector is exactly the squared
error of the product g·r̂ as an estimator of m. A gate fitted on n_val points can improve on r̂ (fitted on
n_train ≫ n_val points) only by correcting systematic, low-dimensional miscalibration of r̂ — a scale per
series, per regime. A per-instance gate is instead a second regression for m on less data, through the
heteroscedastic target R/r̂, which is unbounded as r̂ → 0: this is the mechanism behind defect D8.

**Evidence.** M1b: fine gates lose to per-series in 0–1 of 150 paired runs under four estimators;
M8: never the best rung in 12 dataset × horizon configs.
**Novelty.** Elementary algebra; the reframing "gating re-solves the regression" is the useful part.

---

## Proposition 2 — the shrinkage identity (averaging over corrector noise)

**Assumptions.** r̂ = m + ε with E[ε | X] = 0, Var(ε | X) = σ_ε²(X), ε independent of the noise in R.
Here the gate conditions on X but not on ε.

**Statement.** g\*(x) = E[R r̂ | x]/E[r̂² | x] = m(x)²/(m(x)² + σ_ε²(x)), and always-correcting degrades
(L(1) > L(0)) iff E[σ_ε²] > E[m²].

**Proof.** E[R r̂ | x] = m², E[r̂² | x] = m² + σ_ε². Then L(1) − L(0) = E[r̂²] − 2E[R r̂] = E[σ_ε²] − E[m²]. ∎

**Evidence.** S1: predicted vs empirical optimal gate corr 0.9915 over 28 cells × 5 seeds.
**Novelty.** None — James–Stein / ridge shrinkage. Kept because it states the degradation condition.

---

## Proposition 3 — exact decomposition over a gate partition (the ladder)

**Assumptions.** Proposition 1's setting. A rung is a partition 𝒫 of 𝒳 into cells c with p_c = P(X ∈ c).
Its population gate is the r̂²-weighted projection g_c^𝒫 = E[R r̂ | c]/E[r̂² | c]; its estimate ĝ_c is the
least-squares gate on the n_c validation points in c, independent of the test point.

**Statement.**

  L(ĝ^𝒫) − L(g\*) = E[ Σ_c 1{X∈c} r̂²(ĝ_c − g_c^𝒫)² ]  +  E[(g^𝒫(X) − g\*(X))² r̂(X)²]
            = estimation(𝒫)                   +  approximation(𝒫),

with no cross term. If 𝒫′ refines 𝒫 then approximation(𝒫′) ≤ approximation(𝒫). Asymptotically,

  estimation(𝒫) = Σ_c p_c Var(r̂(R − g_c r̂) | c) / (n_c E[r̂² | c]) + o(1/n),

which grows roughly linearly in the number of cells when n_c ≈ p_c n.

**Proof.** By Proposition 1, L(g) − L(g\*) = E[(g − g\*)² r̂²]. Write g − g\* = (ĝ_c − g_c) + (g_c − g\*) on c.
The cross term is Σ_c (ĝ_c − g_c) E[(g_c − g\*) r̂² 1{X∈c}], because ĝ_c − g_c is constant on c and
independent of the test X. Since g\* r̂² = m r̂ and E[m r̂ | c] = E[R r̂ | c] = g_c E[r̂² | c], each
E[(g_c − g\*) r̂² | c] = 0. The approximation term is the r̂²-weighted L² distance from g\* to 𝒫-constant
functions; refining enlarges that subspace, so the distance cannot increase. For the variance, ĝ_c − g_c =
Σ_{i∈c} r̂_i(R_i − g_c r̂_i) / Σ_{i∈c} r̂_i²; the numerator has mean zero and variance n_c Var(r̂(R − g_c r̂) | c),
and the delta method on the ratio gives the stated rate. ∎

**Evidence.** M1: interior optimum (per-series best 10/15); M3: oracle rung moves from global to per-cell as
between-cell heterogeneity rises; M5b: oracle rung moves finer as n_val grows (per-cell at 3,000 → shrunk at
8,000–20,000).
**Novelty.** A standard weighted-projection bias–variance decomposition; stated here for gate hierarchies.
Closest precedent: estimated combination weights add bias and variance (Claeskens, Magnus, Vasnev & Wang,
IJF 2016), which covers global weights, not partitions of a residual-correction gate.

---

## Proposition 4 — when the ladder is unimodal (the crossing rule)

**Assumptions.** A nested chain 𝒫_1 ⊂ … ⊂ 𝒫_J with total excess E_j = A_j + S_j/n from Proposition 3.
**(Diminishing returns)** A_j − A_{j+1} is non-increasing in j; **(non-decreasing cost)** S_{j+1} − S_j is
non-decreasing in j.

**Statement.** E_j is discretely convex, hence unimodal. Refining from j to j+1 helps iff
A_j − A_{j+1} > (S_{j+1} − S_j)/n.

**Proof.** E_{j+1} − E_j = −(A_j − A_{j+1}) + (S_{j+1} − S_j)/n is a sum of two non-decreasing sequences, hence
non-decreasing; a sequence with non-decreasing increments is unimodal. ∎

**Evidence and limits.** M3/M5: unimodal in 82.4% of runs — so the assumptions **fail in ~18% of runs**,
which is why unimodality is reported as a measured property, not a theorem about the data.
**Novelty.** None mathematically; the value is naming the two measurable conditions.

---

## Proposition 5 — nested (hold-out) selection is consistent

**Assumptions.** Gates for every rung fitted on V₁; per-point losses on V₂ bounded in [0, b] and i.i.d.
(for dependent data: β-mixing, replace n₂ by the number of approximately independent blocks).

**Statement.** With probability ≥ 1 − δ, the rung k̂ minimising V₂ risk over M rungs satisfies
L(k̂) ≤ min_k L(k) + 2b √(log(2M/δ)/(2n₂)).

**Proof.** Hoeffding on each of the M independent V₂ averages, union bound, then compare the selected and the
best rung. ∎

**Evidence.** M5b: regret 0.0097 (n_val = 300) → 0.0006 (20,000); nested beats in-sample selection in
365/370 runs; exact oracle recovery 50.3%.
**Novelty.** Standard hold-out model-selection bound.

---

## Proposition 6 — why in-sample selection always refines

**Assumptions.** Proposition 3; the gate for rung 𝒫 fitted and scored on the same validation data.

**Statement.** To first order, E[in-sample risk(𝒫)] = L(ĝ^𝒫) − 2·estimation(𝒫). In-sample selection therefore
subtracts twice the estimation cost that out-of-sample risk adds, and is biased toward the finest rung.
A Mallows-type correction must add 2×estimation(𝒫).

**Proof sketch.** Least squares through the origin in each cell: the classical optimism of a fitted linear
estimator equals twice its effective degrees of freedom times the noise variance (Mallows 1973; Efron 1986,
2004); per cell this is 2 Var(r̂(R − g_c r̂) | c)/(n_c E[r̂² | c]) weighted by p_c E[r̂² | c]. ∎

**Evidence.** M3/M5: in-sample argmin picked the finest rung (its regret equals "always per-instance" exactly,
0.03839); validation–test rank correlation 0.126. M5b: the ×2 Mallows penalty beat ×1 (0.029 vs 0.037), as
predicted — but both failed, because the per-instance rung is a boosted model, not a partition, and its
effective degrees of freedom were badly underestimated by the n/25-cell approximation.
**Novelty.** Classical (Mallows Cp, Efron optimism). The contribution is the diagnosis of a concrete failure.

---

## Proposition 7 — what validation cannot see: a drifting baseline

**Assumptions.** Constant baseline biases δ_tr (training), δ_val (validation), δ_te (test);
R = m(X) + δ + η; the corrector learned r̂ = m + δ_tr; E[m] = 0; m, η independent; global gate g.

**Statement.**
(i) L_te(1) − L_te(0) = δ_tr² − 2δ_tr δ_te − E[m²]: always-correcting degrades iff δ_tr(δ_tr − 2δ_te) > E[m²].
(ii) The validation-optimal global gate is g_val = (E[m²] + δ_val δ_tr)/(E[m²] + δ_tr²). If δ_val ≈ δ_tr but δ_te
differs, validation recommends correcting (g_val ≈ 1) while test risk degrades.
(iii) Therefore no validation-only rule can detect drift that begins after the validation window; a
block-level rule can only refuse correction when bias already varies **within** validation.

**Proof.** E[R_te r̂] = E[m²] + δ_te δ_tr and E[r̂²] = E[m²] + δ_tr², so L(1) − L(0) = E[r̂²] − 2E[R_te r̂]; the
validation gate is the ratio of the same moments with δ_val. ∎

**Evidence.** Operator schedule (sched/actual ratio 0.987→1.058 across years): ungated correction −0.29
skill (RE-FUSED-5), every rung negative in M1 (4/4 configs); block-LCB withholds correction and avoids
degradation (R4/M7). M6: at a 24-hour horizon the day-ahead price leaves nothing to correct
(residual R² −0.006).
**Novelty.** Elementary bias-shift algebra. Part (iii) — the precise limit of validation-based safety — is
the useful statement for deployment.

---

## Proposition 8 — when hold-out rung selection cannot beat a fixed default (theory tightening, added after M5c/M10)

Proposition 5 bounds regret **with respect to the V₂ distribution**. M5c and M10b showed that on real data a
fixed coarse rung matches or beats hold-out selection. Proposition 8 states the two terms Proposition 5 omits.

**Setting.** Rungs k = 1..M. Gates for selection are fitted on V₁ (n/2 points) and scored on V₂; the deployed
gates are refitted on V = V₁ ∪ V₂ (n points) and scored on the test period. Write
L_te^n(k) for the test risk of rung k fitted on n points, L_V₂^{n/2}(k) for the V₂ risk of rung k fitted on n/2
points, k* = argmin_k L_te^n(k), and decompose, as in Proposition 3, L^n(k) ≈ A(k) + E(k)/n with A the
approximation term and E(k)/n the estimation term (E increasing in refinement).

**Statement.**
(i) *Shift term.* With Δ(k) = L_te^{n/2}(k) − L_V₂^{n/2}(k) (population risks under the two periods) and ε = the
Proposition-5 sampling term, with probability ≥ 1 − δ
  L_te^{n/2}(k̂) − min_k L_te^{n/2}(k) ≤ 2ε + max_{k,k'} [Δ(k) − Δ(k')].
(ii) *Data-halving bias.* Under the Proposition-3 approximation, L_V₂^{n/2}(k) − L_V₂^{n/2}(k') =
[A(k) − A(k')] + 2[E(k) − E(k')]/n, whereas the deployed comparison is [A(k) − A(k')] + [E(k) − E(k')]/n. The
selection criterion double-counts the estimation cost, so hold-out selection is biased toward **coarser** rungs
than are optimal for the deployed n-point fit.
(iii) *Consequence.* A fixed rung k₀ chosen a priori has regret ρ₀ = L_te^n(k₀) − L_te^n(k*). Hold-out selection
can improve on k₀ only when ρ₀ exceeds the sum of the sampling term 2ε, the shift spread max[Δ(k) − Δ(k')], and the
halving bias max_k |E(k)|/n. When the rung-risk curve is flat near its optimum (a plateau), ρ₀ is small for any
coarse k₀ and selection cannot pay.

**Proof.** (i) L_te(k̂) − L_te(k°) = [L_V₂(k̂) − L_V₂(k°)] + [Δ(k̂) − Δ(k°)] for k° = argmin L_te; the first bracket is
≤ 2ε by Proposition 5 (L_V₂(k̂) is within ε of the empirical minimum, which is ≤ the empirical risk of k°), the
second is ≤ the maximum spread. (ii) substitute n/2 into A(k) + E(k)/n. (iii) combine (i) and (ii) with the
definition of ρ₀. ∎

**Evidence.**
* Plateau: fixed per-series and per-cell regrets are 0.0020 and 0.0026 on M5c real data; across M10 origins
  the fixed-policy regrets of series/cell stay within 0.002–0.013.
* Shift spread of the same order as rung differences: the shrunk rung's fixed regret moves from 0.0035 (test 2022)
  to 0.0259 (test 2024-25); the operator schedule is correctable in 2022 (+0.30) and not in 2023 (every rung ≤ 0).
* Halving bias toward coarse rungs: nested argmin is too coarse in 27% of M5c real runs; the parsimony rule, which
  adds a further penalty, is too coarse in 54%.
* Consequence: selection does not beat fixed per-series on fresh origins (H-M10b; margins 0.001–0.003) and loses
  clearly in-sample (M5c; 0.012–0.022).

**Novelty.** None mathematically: hold-out bounds plus a distribution-shift term and learning-curve extrapolation.
The useful statement is (iii): it turns "selection failed" into a checkable condition — selection only pays when
the regret of a sensible fixed default exceeds sampling noise, period-to-period shift, and the halving bias.

---

## Summary: what the theory does and does not provide

| Proposition | Mathematically | Novel | Tested by |
|---|---|---|---|
| 1 exact excess risk E[(g r̂ − m)²] | valid, exact | no (framing yes) | M1b, M8 |
| 2 shrinkage identity | valid | no | S1 |
| 3 estimation + approximation over partitions | valid, exact; variance asymptotic | no (specialisation) | M1, M3, M5b |
| 4 unimodality under diminishing returns | valid under stated assumptions | no | M3 (82.4% unimodal) |
| 5 nested selection consistency | valid (i.i.d. / mixing) | no | M5b |
| 6 in-sample optimism ⇒ over-refinement | valid to first order | no | M3/M5, M5b Mallows |
| 7 drift and the limit of validation | valid, exact | no (the limit statement is the point) | R4, M1, M7, M6, M7b, M10 |
| 8 hold-out selection vs a fixed default under shift | valid (i)(iii); (ii) under the Prop-3 approximation | no (the condition in (iii) is the point) | M5c, M10, M10b |

**Honest verdict.** The theory is correct and coherent, and every proposition is linked to an experiment
that tests it. It is also elementary and almost entirely known — the forecast combination puzzle,
weighted projections, Mallows/Efron optimism, hold-out selection bounds. It explains the empirical findings;
it does not by itself constitute a theoretical contribution at NeurIPS/ICML/JMLR level. A theory-first ML
submission would need, at minimum, a finite-sample bound for selection over a boosted per-instance gate with
data-dependent degrees of freedom, and a drift-aware selection guarantee under mixing — neither is proved here.
