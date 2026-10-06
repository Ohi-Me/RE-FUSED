# The price of adaptivity — theory for RE-FUSED

Status labels, used for every result: **[K]** known result restated; **[S]** specialisation of known results to
adaptive correction gates (derivation given, no novelty claimed for the mathematics); **[C]** a combination that
yields a testable operational criterion used in this project (novelty, if any, is in the criterion and its
empirical validation, not in the algebra). Every statement names the experiment that tests it.

---

## 0. Setting and notation

Target Y, baseline forecast B (fixed), residual R = Y − B. A corrector r̂(x) is trained on data independent of the
data used to fit gates (training period or cross-fitted folds). A **gate** is a function g(z) of an *adaptivity
index* z; the corrected forecast is B + g(z)·r̂(x). Loss is squared error unless stated.

The adaptivity index lives on one or more **axes**:
* unit axis — z = series id s (global gate = constant);
* state axis — z = (s, regime) or z = x (per-instance);
* time axis — z = (s, t) with gates re-estimated on recent windows.

A **partition gate** is constant on the cells c of a partition 𝒫 of the index space. A refinement 𝒫′ ⪰ 𝒫 splits
cells. Weights: w(c) = E[r̂² ; c] (signal energy in cell c), and ⟨f, g⟩ = E[r̂² f g] with ‖f‖² = ⟨f, f⟩.

Oracle cell gate: g*_𝒫(c) = E[R r̂ | c] / E[r̂² | c] — the weighted least-squares coefficient of R on r̂ in c.

---

## T1. Exact risk difference between any two fixed gates [K-algebra]

For fixed gates g_a, g_b, d = g_b − g_a and e_a = R − g_a r̂:

    L(g_a) − L(g_b) = 2 E[d r̂ e_a] − E[d² r̂²].

*Proof.* Expand (R − g_b r̂)² = (e_a − d r̂)² and take expectations. ∎

**Reading.** A finer gate helps iff its deviation from the coarser gate correlates with the coarse-corrected
residual by more than half of the deviation's own energy. Used by: every experiment (the realised gain is always
computed with this identity on test data, which gives an exact per-observation decomposition for block bootstrap).

---

## T2. Approximation gain of a refinement [S]

For 𝒫′ ⪰ 𝒫 at the population level, with h = g*_𝒫′ − g*_𝒫:

    L(g*_𝒫) − L(g*_𝒫′) = ‖h‖² = Σ_{c′} w(c′) (g*_𝒫′(c′) − g*_𝒫(parent(c′)))² =: τ²(𝒫 → 𝒫′).

*Proof.* Apply T1 with g_a = g*_𝒫, g_b = g*_𝒫′. On each cell c′, E[r̂ e_a | c′] = E[r̂²|c′](g*_𝒫′(c′) − g*_𝒫(c)), so
2E[d r̂ e_a] = 2‖h‖² and E[d² r̂²] = ‖h‖². ∎

τ² is the **signal-energy-weighted heterogeneity** of the oracle gate along the refinement. It is zero when the
coarse gate is already optimal in every sub-cell, whatever the heterogeneity of Y itself.

---

## T3. Estimation cost [S of K]

Let ĝ_𝒫 be the least-squares cell gates fitted on n observations, n(c) = n·p(c). Under standard regularity
(independent observations or a mixing process with long-run variance), ĝ_𝒫(c) − g*_𝒫(c) is asymptotically normal
with variance v(c)/n(c), where v(c) = E[r̂² e_c² | c] / E[r̂² | c]² and e_c = R − g*_𝒫(c) r̂ (sandwich form;
for dependent data E[r̂² e_c² | c] is replaced by the long-run variance of r̂ e_c in the cell). Then

    E[L(ĝ_𝒫)] = L(g*_𝒫) + (1/n) Σ_c p(c) E[r̂² | c]·v(c)/p(c) + o(1/n) = L(g*_𝒫) + V(𝒫)/n + o(1/n),
    V(𝒫) = Σ_c E[r̂² e_c² | c] / E[r̂² | c].

With homogeneous σ̄² := E[r̂² e² | c]/E[r̂² | c] across cells, **V(𝒫) = |𝒫|·σ̄²**: estimation cost grows
linearly in the number of cells and inversely in n. [Mallows 1973; Efron 2004 for the optimism form]

For estimators that are not partitions (boosted trees, neural nets), the same first-order cost is
σ̄²·df/n with df the **generalised degrees of freedom** GDF = Σ_i ∂ĝ_i/∂y_i [Ye 1998], estimated by
perturbation. This puts partition, boosted and neural gates on one axis. Tested by X3 (risk vs GDF/n).

---

## T4. The persistence-adjusted crossing condition [C]

Gates are fitted on validation (distribution P_V) and deployed on test (P_T). Let h_V, h_T be the refinement
heterogeneity (T2) under P_V and P_T, and assume the weights w are stable (r̂ has the same energy profile).
Let ĥ = h_V + ε be the fitted heterogeneity with E[ε] ≈ 0, E‖ε‖² = ΔV/n, ΔV = V(𝒫′) − V(𝒫), and ε independent of
the test data. Then the expected **test** gain of refining is

    G_T = 2 ⟨h_V, h_T⟩ − ‖h_V‖² − ΔV/n + (terms that vanish when ĥ ⟂ 𝒫-constant functions under P_T).

*Proof.* T1 with g_a = ĝ_𝒫, g_b = ĝ_𝒫 + ĥ evaluated under P_T. On each fine cell, E_T[r̂ R | c′] =
E_T[r̂² | c′](g*_{𝒫,T} + h_T)(c′), so 2E_T[ĥ r̂ e_a] = 2⟨ĥ, h_T⟩ + 2⟨ĥ, g*_{𝒫,T} − ĝ_𝒫⟩. The second inner product is
zero when ĥ is orthogonal to functions constant on 𝒫-cells, which holds exactly under P_V by construction of
least squares and approximately under P_T when weights are stable. Taking expectation over ε gives
2⟨h_V, h_T⟩ − (‖h_V‖² + ΔV/n). ∎

Define **persistence** π = ⟨h_V, h_T⟩ / ‖h_V‖². Refinement pays iff

    ‖h_V‖² (2π − 1) > ΔV / n.

**Consequences (each tested).**
1. π ≤ ½ ⇒ refinement never pays, for any n (non-persistent heterogeneity is pure cost). [X1, X6]
2. π = 1 (stationary) recovers the classical condition τ² > ΔV/n ≈ Δ|𝒫| σ̄²/n. [X1, X2]
3. Larger n helps only through the cost term; it cannot rescue non-persistent heterogeneity. [X2]
4. A strong baseline shrinks R and hence τ² (heterogeneity in *what is left to correct*), pushing the optimum
   coarse. [X4]
5. Heterogeneity that is structural (hour-of-day, unit identity) tends to be persistent; heterogeneity learned
   from transient states (per instance) tends not to be — an empirical claim measured by π̂. [X1, X7]

Relation to prior work: this is the gate analogue of the estimation-error and instability explanations of the
forecast-combination puzzle [Smith & Wallis 2009; Claeskens et al. 2016; Elliott & Timmermann 2005 on unstable
weights]; the algebra is elementary. What is used here is the *measurable* form (T8).

---

## T5. The time axis: tracking versus estimation [K]

Let the per-series oracle gate follow a random walk g_t = g_{t−1} + η_t, Var(η_t) = q per period, and let a
single period's gate estimate have variance s² (signal-weighted). A rolling window of w periods has

    MSE(w) ≈ s²/w + q (w + 1)(2w + 1)/(6w) ≈ s²/w + q w / 3,     w* ≈ √(3 s² / q),

and exponential forgetting with factor λ has MSE ≈ s² (1 − λ)/(1 + λ) + q/(1 − λ²) (steady state).
Static estimation (w = whole history) is optimal as q → 0. [Standard adaptive-filtering trade-off; e.g. Ljung &
Gunnarsson 1990.] Within T4, time adaptivity is a refinement along time whose persistence π is the
autocorrelation of gate deviations at the deployment lag. Tested by X6 (estimated q̂ and s² predict whether
rolling beats static).

---

## T6. Quantile (reserve) gates [S]

For level q, let ξ_c be the q-quantile of the residual-after-correction in cell c, with density f_c at ξ_c. The
excess pinball loss of using ξ′ is ≈ ½ f_c (ξ′ − ξ_c)² (local quadratic expansion, [Koenker 2005]). Hence

* approximation gain of refining: Σ_{c′} p(c′) ½ f_{c′} (ξ_{c′} − ξ_{parent})²;
* estimation cost of an empirical cell quantile: ½ f_c · q(1 − q)/(n(c) f_c²) = q(1 − q)/(2 n(c) f_c)
  [Bahadur representation], i.e. Σ_c q(1−q)/(2 n f_c) in total;
* finite-sample existence: a split-conformal upper quantile at level q is finite only if n(c) ≥ ⌈1/(1 − q)⌉ − 1
  [Lei et al. 2018], e.g. ≥ 99 points per cell for q = 0.99.

Whether the best level is coarser in the tail therefore depends on the tail density and the kind of
heterogeneity (location vs scale): under Gaussian scale heterogeneity it need not be; under heavy tails
(small f at ξ) and small n(c) it is. **No general tail-is-coarser claim is made**; the quantile-specific
crossing condition is tested on real reserve quantiles (X5). Relevance: operating-reserve dimensioning uses high
quantiles of forecast errors.

---

## T7. Selection from finite, shifted validation data [S] (RE-FUSED-6 Prop. 8, restated)

Hold-out selection among M levels has test regret ≤ 2ε_n + max_{k,k′}[Δ(k) − Δ(k′)] + halving bias, where ε_n is
the sampling term, Δ(k) the validation-to-test shift of level k's risk, and the halving bias arises because
selection fits gates on part of the validation data while deployment refits on all of it (double-counts the
estimation cost, biasing selection coarse). A fixed default beats hold-out selection when its own regret is below
the sum of these terms — the RE-FUSED-6 real-data finding.

---

## T8. PART — a persistence-adjusted refinement test (operational rule) [C]

Goal: estimate the three terms of T4 from validation data only, *without* the halving bias of hold-out selection.

Split the validation period into K contiguous time blocks. For a refinement 𝒫 → 𝒫′:

1. **Block heterogeneity.** In each block b fit coarse and fine gates and form ĥ_b(c′) = ĝ′_b(c′) − ĝ_b(parent(c′)).
   For model-based fine gates, fit per block and evaluate every ĥ_b on a common evaluation set.
2. **Cross-block products** P_{bb′} = Σ_{c′} ŵ(c′) ĥ_b(c′) ĥ_{b′}(c′) for b ≠ b′. Estimation errors in disjoint
   blocks are independent, so E[P_{bb′}] = ⟨h_b, h_{b′}⟩ **without noise bias**.
3. **Terms.** Ŝ = mean_{b≠b′} P_{bb′} estimates the heterogeneity energy; Ĉ = mean_{|b−b′| ≥ ⌈K/2⌉} P_{bb′}
   estimates heterogeneity that persists across the largest available time separation (proxy for ⟨h_V, h_T⟩).
4. **Cost at full n.** Ê = Σ_{c′} ŵ(c′) Var̂_JK(ĥ_V(c′)), the leave-one-block-out jackknife variance of the
   full-validation fit (robust to within-block dependence).
5. **Predicted gain** Ĝ = 2Ĉ − Ŝ − Ê; refine iff Ĝ > 0. Along a ladder, refine sequentially and stop at the first
   Ĝ ≤ 0.

Properties: validation-only; unbiased cross-product terms; the cost term refers to the full-validation fit that is
actually deployed (no halving bias); persistence is estimated at the longest lag available inside validation (a
stated assumption: persistence at the deployment lag is not better than at that lag). Limitation: for boosted or
neural fine gates the block fits use n/K points and are more regularised, so Ĉ and Ŝ under-state what the
full-n model learns — a bias toward coarse, stated and measured in X1. Tested: X1 (accuracy vs ground truth),
X7 (development regret), XC (confirmatory).

---

## T5b. Two refinements along time: updating is free, discounting is priced [S] (DF8)

Let a gate be deployed over a period during which new residuals keep arriving. (a) *Updating*: replacing a gate frozen
at the end of the fitting period by the expanding estimate that also uses residuals observed since then. The
expanding estimate has at least as many observations as the frozen one, so its estimation variance is not larger;
under drift it is also closer to the current oracle gate. Updating therefore has no estimation price (up to
second-order effects when new data come from a transient regime). (b) *Discounting*: replacing the expanding estimate
by a rolling window or exponential forgetting. This trades estimation variance (fewer effective observations,
s²/w in T5) against tracking error (q·w/3 in T5): it pays only when the drift is persistent at the deployment lag.
Development evidence: updating gained or was neutral in every development configuration (within noise); discounting
gained under drifting references and lost on stationary baselines (DF5, DF8).

## T8d. Prequential validation for sequential policies [K-method, used as selector] (DV6, DF8)

For policies that are themselves sequential (updating, discounting), the natural validation-only estimate of their
value is to replay them through the validation period using only information available at each decision time and
to compare with the matched comparator (the expanding gate) — prequential evaluation in the sense of Dawid.
It has neither in-sample optimism nor the halving bias of T7, because each decision uses only the past. Development
accuracy (India, 32 policy × configuration pairs): sign accuracy 0.84, 0.96 on decisive cases, r = 0.96.
The block-based PART-time estimator (T8b below) was superseded by this procedure.

## T8b. PART-time — the time-axis plug-in [C] (added in development, DF6; superseded by T8d)

A gate re-estimated on the previous block of B days (index b) replaces the static gate ḡ fitted on the whole
gate-fitting period. Let ĥ_b = ĝ_b − ḡ = h_b + ν_b with block estimation noise ν_b independent across blocks. By T1 the
gain of using ĝ_b during block b+1 is G_{b+1} = 2⟨ĥ_b, h_{b+1}⟩ − ‖ĥ_b‖², whose expectation over ν_b is
2⟨h_b, h_{b+1}⟩ − ‖h_b‖² − E‖ν_b‖². Because E⟨ĥ_b, ĥ_{b+1}⟩ = ⟨h_b, h_{b+1}⟩ and E‖ĥ_b‖² = ‖h_b‖² + E‖ν_b‖²,

    Ĝ_time = mean_b [ 2⟨ĥ_b, ĥ_{b+1}⟩ − ‖ĥ_b‖² ]

is **unbiased** for the expected gain: the estimation noise cancels without a variance estimate. (A small positive
bias of order Var(ḡ) arises because ḡ is shared by all blocks.) Tested: Dev-2 (sign accuracy for rolling gates),
XC (confirmatory).

## T8 v2. PART as frozen for confirmation [C] (supersedes steps 3–5 above; DV5)

Development showed that the v1 rule above under-estimated gains: clipping gates before forming block products biased
Ĉ and Ŝ towards zero, and the jackknife cost exceeded the total fitted energy (DV5). The frozen rule
(`04_code/refused_gate/selection.py::part2_partition`, `part2_instance`) is

* unclipped block heterogeneities ĥ_b on K blocks (confirmatory real data: interleaved weekly blocks, K = 6 for
  partitions, K = 4 for the per-instance gate; semi-synthetic grid: K = 6 contiguous blocks);
* Ĉ = mean of ⟨ĥ_b, ĥ_b′⟩ over block pairs that are at least `far` blocks apart (all b ≠ b′ for interleaved blocks,
  |b − b′| ≥ ⌈K/2⌉ for contiguous blocks);
* Ĝ = 2Ĉ − ‖ĥ_full‖², because E‖ĥ_full‖² = ‖h_V‖² + E‖ε‖² already contains the estimation cost of the full fit;
  no jackknife or variance estimate is needed.

Properties: E[Ĝ] = 2⟨h_V, h_persist⟩ − ‖h_V‖² − E‖ε‖² under independent block errors (proof: manuscript Appendix A).
The model-gate bias toward coarse (T4b) remains.

## T4b. Model-based fine gates: learnable heterogeneity [S]

For a regularised estimator (boosted trees, neural nets) fitted on n points, replace h_V in T4 by the estimator's
expected deviation ḣ_n = E[ĥ_n] − g_𝒫, whose energy ‖ḣ_n‖² ≤ ‖h_V‖² grows with n (less shrinkage), and ΔV/n by the
estimator's own variance E‖ĥ_n − ḣ_n‖². Both terms move in favour of the fine gate as n grows — the data-budget
curve observed in development (DF4: instance − series gap from −0.039 at 36 fitting days to +0.015 with five years).
Consequence for PART: block fits use n/K points, so they under-state ‖ḣ_n‖² (bias toward coarse), as observed.

## T8c. Matching the persistence target to deployment [C] (DV1)

Persistence must be measured between objects of the same kind as the deployed gate and its deployment period.
If gates are fitted on a full seasonal cycle and deployed for a full cycle, contiguous sub-seasonal blocks measure
*season-specific* heterogeneity and understate persistence; interleaved blocks remove that confound but cannot see
drift; calendar-cycle blocks on multi-cycle data measure both. The block design is therefore part of the rule and
is fixed before confirmatory use.

---

## T9. Where the principle does *not* apply (scope)

* Gates fitted on the same data used for evaluation (in-sample optimism, RE-FUSED-6 Prop. 6) — a protocol error, not a
  trade-off.
* Losses other than squared/pinball: the local-quadratic argument needs a smooth, strictly convex expected loss
  near the optimum.
* Joint end-to-end training of corrector and gate: estimation cost is shared and not separable; only partially
  explored (A9 variant).

---

## References to verify before citation (verification log in `09_theory/refs_verification.md`)

Bates & Granger (1969); Smith & Wallis (2009); Claeskens, Magnus, Vasnev & Wang (2016); Elliott & Timmermann
(2005); Mallows (1973); Efron (2004); Ye (1998); Stein (1981); Koenker & Bassett (1978); Koenker (2005);
Bahadur (1966); Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018); Ljung & Gunnarsson (1990);
Herbster & Warmuth (1998); Diebold & Mariano (1995); Künsch (1989); DerSimonian & Laird (1986);
Chernozhukov et al. (2018).
