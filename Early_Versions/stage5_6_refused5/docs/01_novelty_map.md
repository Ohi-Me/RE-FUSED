# RE-FUSED-5 novelty map (living document)
Compiled 12 Sep 2026. Sources are primary papers fetched/searched this session; each row states
what the work already solves and what it leaves open. **Rule: a RE-FUSED-5 component is rejected if it
is a renamed version of anything below.**

## Cluster 1 — Residual / post-hoc correction of a base forecaster
| Work | What it solves | What it does NOT solve |
|---|---|---|
| CRC, *Causality-Inspired Safe Residual Correction* (arXiv 2512.22428) | Y = Y_base + Δ with a 4-part "safety firewall" (direction gating, quantile clipping, per-(var,horizon) validation selection, shrink-to-base with ε threshold). Guarantees: pointwise non-degradation under MAE; validation-level non-degradation; PAND bound on test non-degradation rate via Hoeffding. NDR ≈ 95% on ETT/Weather/Electricity/Traffic. | Decision is **static per (variable,horizon)**, not per instance/context. Authors state: adjacency from correlation only; mechanism is **conservative**, may forgo large correctable errors; **no dynamic uncertainty-based bounds, no conformal integration**; safety-vs-accuracy trade-off unresolved. No notion of *which context* to consume, no horizon-resolved information value, no cost. |
| *Post-Training Corrections* (arXiv 2505.15354v2) | Bandit selection (SH/SR/LinUCB) over a library of affine/piecewise/trend corrections + HITL-LLM corrections; accepted only if validation MSE improves. Up to ~17% MSE gain on Autoformer. | **Global** selection on validation, **no per-instance gating**; no uncertainty; corrections are fixed simple transforms; MSE-aligned gains can worsen MAE/MAPE; no guarantees. |
| GARS (Electronics 15/18/4137), FRWKV+ "adaptive trust" gating (2605.15690), Minusformer (2402.02332), Multi-granularity residual w/ confidence (2022) | Plug-in residual selection / progressive residual learning / periodic-aware gate controlling correction strength. | Gates are heuristic sigmoids trained end-to-end on accuracy; no calibrated estimate of expected benefit, no risk control, no information-acquisition decision. |
| *Learning the Context of Errors* (arXiv 2606.14222) | Black-box online adaptation of TSFMs: adapter maps recent error context → correction; selectively applies corrections. | No formal guarantees ("no explicit generalization guarantees"); open problems they name: when error-context helps, adapter capacity, drift, theory of *when residual patterns enable correction*. |

**Verdict:** "residual + learned gate" is occupied. Any RE-FUSED-5 claim must be about *what the gate estimates*, its *calibration*, its *guarantee type*, and *horizon/context allocation* — not about having a gate.

## Cluster 2 — Selective prediction, abstention, risk control
| Work | Solves | Leaves open |
|---|---|---|
| Selective Conformal Risk Control (2512.12844); SCoRE (e-values); CRC under non-monotone losses (2604.01502); Confidence Gate Theorem (2603.09947) | Distribution-free, finite-sample control of general risks *when abstaining from prediction*; selective sets; e-value based selective risk. | Applied to **abstaining from predicting**, not to **abstaining from correcting** an existing strong baseline; assumes exchangeability (temporal shift unhandled); no notion of marginal value of a *context block*. |
| Learning-to-defer / cascades (2410.15729, 2502.01459, 2307.02764, NeurIPS 2025 cascades) | Route instances between cheap/expensive models with confidence-based deferral; regression deferral exists; theory on when confidence-based deferral suffices. | Deferral targets *another model/human*, cost = compute/expert; not *deferral to a no-change baseline* under temporal shift; no horizon dimension; no calibrated benefit estimate. |

## Cluster 3 — Expert aggregation (the strongest attack on a naive claim)
Sleeping experts / specialists (Freund et al.; Kleinberg et al. MLJ 2010), Fixed Share / tracking regret (1008.4532, 2106.13021), second-order sleeping-expert bounds applied to temperature forecasting (2506.15216).
**Solves:** adversarial regret vs the best (context-activated) expert, i.e. *provably competitive switching between a naive baseline and a model*, including context-dependent activation.
**Leaves open:** regret is relative and adversarial; it gives **no estimate of how much benefit a correction yields**, no calibration of that estimate, no per-horizon information value, no acquisition cost, and no finite-sample non-degradation statement in the batch forecasting setting.
**Consequence:** RE-FUSED-5 must compare against Hedge/Fixed-Share/sleeping-expert baselines or a reviewer will call it re-invented expert aggregation.

## Cluster 4 — Value of information / active feature acquisition
AFA survey (2502.11067, POMDP taxonomy), ACO (ICML 2024), NOCTA (2507.12412), generative surrogate AFA (2010.02433).
**Solves:** per-instance sequential feature acquisition under budget; POMDP formulation; the survey states most methods are **heuristic and lack formal guarantees**.
**Leaves open:** almost entirely i.i.d. tabular/medical; **not horizon-indexed**; no interaction with a strong temporal baseline; no distribution-shift guarantees; value of a *covariate block over forecast horizon* unstudied.

## Cluster 5 — Classical shrinkage & combination (the theory analogue)
Bates–Granger; Diebold–Pauly shrinkage weights; Stock–Watson (2004, 2006); forecast-combination puzzle (UCR 2025-14); bias–variance shrinkage of weights.
**Solves:** *unconditional* shrinkage of combination weights toward equal/naive weights; estimation-variance rationale for why simple averages win.
**Leaves open:** weights are **global**, not conditional on state/horizon/regime; no per-instance decision; no distribution-free control; no context-acquisition dimension.

## Cluster 6 — Supporting empirical facts (useful, not competing)
"Mind the naive forecast!" (low-predictability series); *Spurious Predictability in Financial ML* (2604.15531); TSFM zero-shot benchmarks where seasonal-naive remains competitive (2602.10848, 2502.12944); macro-forecasting variance-ratio thresholds (2510.11008).

## Candidate gap for RE-FUSED-5 (to be attacked, not assumed)
A **conditional, horizon-resolved estimate of the marginal value of (a) applying a learned correction and (b) consuming a context block**, which is (i) *calibrated* — predicted benefit matches realized loss reduction, (ii) *decision-usable* with finite-sample non-degradation control that is **less conservative than clipping** and valid under temporal (non-exchangeable) shift, and (iii) *explained by a closed-form threshold law* linking the optimal correction weight to residual signal-to-noise and estimation error. The law would predict, rather than merely exhibit, CRC's "corrector's dilemma" and the combination puzzle.

### Open novelty risks (must resolve before committing)
1. Conditional shrinkage by predicted residual R² may already exist in combination literature → search "conditional/state-dependent combination weights".
2. Sleeping experts may dominate the guarantee story → need a *different guarantee type* (finite-sample non-degradation under blocks, not regret).
3. If the calibrated-value estimator gives no accuracy gain over CRC-style static gating, the contribution collapses to a diagnostic → then reframe as a measurement/diagnostic paper with a law + falsification, which is still publishable but must be stated honestly.
