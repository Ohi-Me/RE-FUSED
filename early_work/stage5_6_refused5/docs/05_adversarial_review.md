# RE-FUSED-5: simulated adversarial NeurIPS reviews (round 1, on evidence as of 12 Sep 2026)

## Reviewer A — Area Chair, time-series ML (rating: borderline reject → borderline accept if fixed)
**A1. Originality is thin.** "Correction = shrinkage by residual R²" is James–Stein/ridge in new
clothes, and CRC (2512.22428) already ships a gated corrector with a non-degradation bound.
*Status:* partly conceded. The defensible increment is (i) the *granularity* result — shrinkage applied
to the gate itself, with an identifiability criterion Var_within(g*) vs Var(ĝ); (ii) a block-risk rule
that beats CRC-style clipping on non-stationary baselines (R4). Neither is in CRC. The paper must
state the classical shrinkage lineage explicitly in §2, not bury it.
**A2. The headline is a negative result.** Free per-instance gating loses (S2c 10/10 seeds).
*Status:* accepted and made the contribution. Risk: reviewers may read the paper as "our fancy idea
failed, so we recommend the simple thing". Mitigation: lead with the *law* and the identifiability
criterion, not with the failed gate.
**A3. Effect sizes are small.** +0.056 vs +0.053 skill (shrunk vs static) on India.
*Status:* open. Needs the 10-seed CIs (R5, running) and a DM test; if the gap is inside noise, the
claim must fall back to "shrunk is never worse and is materially better when validation is scarce",
which S3 does support (10/10 seeds at n_val=500).

## Reviewer B — statistical experimentalist (rating: reject unless protocol tightened)
**B1. One dataset family.** India generation + NYISO prices; both energy.
*Status:* open. Public benchmarks (ETT/electricity/traffic) are permitted by the user and must be run.
**B2. Gate estimated on validation, evaluated on test — but the corrector's hyperparameters?**
*Status:* fixed by construction (no test access anywhere), but the paper must publish the exact search
space; currently hyperparameters are fixed a priori, which is *conservative but not tuned*. Must say so.
**B3. var_est via one split-half is crude.** It conflates estimator noise with split noise.
*Status:* valid. Fix: repeated (e.g. 10×) half-splits or out-of-fold estimation; cheap.
**B4. Skill is a ratio of MAEs; CIs across states are not independent** (common shocks).
*Status:* valid. Needs block-bootstrap over time, not just across states.

## Reviewer C — theory (rating: weak reject on theory, neutral on empirics)
**C1. The "law" is an identity, not a theorem.** g* = SNR/(1+SNR) follows from a quadratic objective
in two lines and assumes ε ⊥ m, E[ε|s]=0.
*Status:* conceded — it is a proposition, and the paper must call it that. The non-trivial part is the
*estimated* version: bounding the excess risk of ĝ, which requires a concentration argument the paper
does not yet have. Either prove: E[L(ĝ)] − L(g*) ≤ C·Var(ĝ)·E[r̂²] (a two-line bias-variance bound,
provable) plus a finite-sample bound on Var(ĝ), or drop theory claims to "proposition + empirics".
**C2. The block-LCB rule assumes independent blocks.** Monthly blocks under seasonality are not
independent.
*Status:* valid. Either move to a time-uniform/e-value construction or state the assumption and show
robustness to block length. Must do the latter at minimum.

## Reviewer D — energy domain (rating: accept for the domain claims, with caveats)
**D1. The data audit is the strongest part** (price identical across 18 states; 98% LOCF; balanced
station panel). Publishable as a dataset/benchmark note in its own right.
**D2. The operator-schedule result is the most interesting domain finding** — a real dispatch program
is beaten 3× by lag-1 persistence, and correcting it fails out-of-sample because its bias drifts.
Wants: per-state breakdown and a policy discussion.
**D3. 5-min vs 15-min framing must not overclaim.** India data is 15-min (and mostly missing);
the 5-min evidence is NYISO. Keep the claim as a resolution-scaling measurement across two markets.

## Reviewer E — hostile generalist (rating: reject)
**E1. "Where is the new ML knowledge?"** If the answer is "a shrinkage coefficient and a
confidence-bound switch", that is an applied-statistics contribution, not a NeurIPS one.
*Status:* the sharpest attack. The paper's survival depends on the identifiability criterion being
(a) predictive out-of-sample (does measured Var_within/Var_est predict which gate granularity wins?
— testable, not yet tested) and (b) transferable across domains beyond energy.
**E2. Baselines.** No TSFM, no N-HiTS/PatchTST/TiDE comparison yet.
*Status:* open; R6 and a deep-baseline run are required.
**E3. Everything reported is ≤0.2 skill on one grid.** Needs falsification experiments *presented as
such*, which the paper does have (F1–F5), and cross-domain replication, which it does not yet.

## Consolidated required work before submission
1. **Predictive test of the criterion** (E1a): across all cells/datasets, does the measured ratio
   Var_within/Var_est predict whether static or conditional gating wins? This is now the paper's core
   experiment and is not yet run.
2. Public-benchmark replication: ETT / electricity / traffic (B1, E2).
3. Strong learned baselines incl. one TSFM and one modern deep model (E2).
4. Repeated-split variance estimate; block bootstrap CIs; robustness to block length (B3, B4, C2).
5. Either the excess-risk proposition with proof, or explicit removal of theory claims (C1).
6. Per-state/per-zone breakdowns and the policy discussion for the domain story (D2, D3).
