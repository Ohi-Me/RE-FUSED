# RE-FUSED-5: correction as conditional shrinkage — formulation (v0.1, 12 Sep 2026)

## 1. Setting
For series i (state/zone), time t, horizon h: target Y_{i,t+h}, a **given baseline** B_{i,t+h}
(lag-1 persistence, seasonal naive, operator schedule, or day-ahead price), residual
R_{i,t+h} = Y_{i,t+h} − B_{i,t+h}. A corrector r̂_θ(s_{i,t}, h) predicts the residual from state s
(history, covariate blocks C_1..C_K). The deployed forecast is

    Ŷ_{i,t+h} = B_{i,t+h} + g(s_{i,t}, h) · r̂_θ(s_{i,t}, h),   g ∈ [0,1].

RE-FUSED-4's contribution was showing target self-history matters; RE-FUSED-5 asks **when the correction
is worth applying at all, and which covariate block is worth consuming**, per state, horizon and regime.

## 2. The law (squared loss)
Write the conditional mean residual m(s,h) = E[R | s, h] and let the corrector be a noisy estimate
r̂ = m + ε with E[ε|s]=0, Var(ε|s)=σ_e²(s,h) (estimation error), ε ⊥ m.

Loss of a gated correction:
    L(g) = E[(R − g r̂)²] = E[R²] − 2g·E[m²] + g²·(E[m²] + σ_e²)

so, minimising over g,
    **g*(s,h) = E[m²|s,h] / (E[m²|s,h] + σ_e²(s,h)) = SNR / (1 + SNR)**            (L1)
    **Δ*(s,h) = L(0) − L(g*) = g*(s,h) · E[m²|s,h]**                                 (L2)
    **L(1) − L(0) = σ_e²(s,h) − E[m²|s,h]**                                          (L3)

(L3) is the *corrector's dilemma* in closed form: always-correct degrades exactly when estimation
noise exceeds explainable residual variance. (L1) says the optimal gate is a **conditional
shrinkage factor equal to the estimator's conditional R² of the residual**, not a free sigmoid.

### Value of a covariate block C (VOC)
With and without block C in the corrector's input:
    **VOC(C | s,h) = [E[m²_{+C}|s,h] − E[m²_{−C}|s,h]] − [σ²_{e,+C}(s,h) − σ²_{e,−C}(s,h)]**   (L4)
Information pays only when the explainable variance it adds exceeds the estimation variance it
costs. This predicts (i) blocks that help at short h can hurt at long h, (ii) blocks that help only
inside a regime, (iii) a sample-size dependence: VOC rises with n as σ²_e falls.

## 3. What is new (must survive §01_novelty_map)
Not new: shrinkage itself (James–Stein, ridge, Diebold–Pauly combination weights); residual
correction; gating; abstention.
Claimed new, to be tested:
* **N1 (estimator).** A learned, *conditional* estimate of (g*, Δ*) per (series, horizon, regime)
  trained to match realised benefit, rather than a free gate trained only on accuracy.
* **N2 (calibration as a first-class result).** Predicted benefit vs realised benefit reliability
  curves; a gate is only accepted if its value estimate is calibrated, not merely accuracy-improving.
* **N3 (risk-controlled deployment).** Deploy correction only where a time-blocked, distribution-free
  procedure bounds P(degradation) — targeting *less conservatism* than CRC's clipping at equal safety.
* **N4 (resolution law).** How Δ* and VOC scale across temporal resolutions (daily India panel →
  15-min IEX → 5-min NYISO) and horizons; the "15-min → 5-min" axis becomes a measurement of the law,
  not a data claim.

## 4. Falsification conditions (pre-registered)
F1 If learned ĝ does not approach g* on synthetic data with known SNR, N1 fails.
F2 If a single global gate (or CRC-style static per-(series,horizon) selection) matches the
   conditional gate within noise on all real datasets, N1/N2 collapse → report as negative result.
F3 If predicted benefit is uncorrelated with realised benefit (slope ≉ 1), N2 fails.
F4 If the risk-controlled rule is not less conservative than clipping at equal degradation rate, N3 fails.
F5 If VOC ordering does not transfer across datasets/resolutions, N4 fails.

## 5. Baseline strength as an experimental axis (enabled by the rebuilt panel)
The Indian raw data supplies baselines of very different quality for the same target:
operator schedule (MAE 14.49), seasonal naive lag-7 (9.95), persistence lag-1 (4.67).
The law predicts correction value falls as baseline strength rises; this is directly testable.
