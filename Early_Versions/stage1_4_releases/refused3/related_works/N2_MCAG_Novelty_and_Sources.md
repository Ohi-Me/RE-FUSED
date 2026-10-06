# N2 — MCAG (Multi-Carbon-Aware Gating)
## Novelty Summary & Literature Grounding

---

## What MCAG Is

MCAG is a learnable gating module that conditions patch-level Transformer representations
on carbon signals before projection to forecasting outputs.

**Architecture:**
```
Carbon signals c_t = {p_carbon_gcal, lcmp_carbon, carbon_budget_pressure,
                      freq_excursion_risk, monsoon_re_risk}   [NC=5 features]

Gate:    g_t = sigma(LayerNorm(W_g * mean_t(c_t) + b_g))    [shape: B x D]
Output:  h_gated = g_t (elementwise) h_t                     [soft modulation]
```

MCAG sits between the PatchTST Transformer encoder and the output projection layer.
The gate is trained end-to-end with the forecasting objective.

---

## What Was Already Known (Prior Work)

| Paper | Key Contribution | How MCAG Differs |
|---|---|---|
| **Nie et al. (2023)** *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers*, ICLR 2023 | PatchTST: subseries-level patches + channel-independent Transformer. SOTA on ETTh1/2, ETTm1/2, Weather, Exchange. | PatchTST has no external conditioning mechanism. MCAG adds a learnable gate conditioned on domain-specific carbon signals. |
| **Lim et al. (2021)** *Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting*, Int. J. Forecasting | TFT: Variable Selection Networks + Gated Residual Networks + multi-head attention. Uses static/known/observed covariates. | TFT's gating (GRN) is internal to the architecture, not conditioned on a specific external signal class (carbon). |
| **Zhou et al. (2021)** *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021 | ProbSparse self-attention for O(L log L) complexity. Encoder-decoder structure for multi-step prediction. | No domain conditioning. MCAG's contribution is signal-specific gating, not attention efficiency. |
| **Vaswani et al. (2017)** *Attention Is All You Need*, NeurIPS 2017 | Original Transformer. Multi-head self-attention + positional encoding. | General architecture; no mechanism for conditioning representations on a specific signal type. |
| **Wang et al. (2022)** *Towards Accurate and Reliable Forecasting for Carbon Emission Trading*, Energies | LSTM + attention for China ETS carbon price forecasting. Uses economic and policy features. | Predicts carbon prices, not grid dispatch targets. Does not gate representations on carbon signals. |
| **Hochreiter & Schmidhuber (1997)** *Long Short-Term Memory*, Neural Computation | LSTM gating (input, forget, output gates) for sequence modelling. | LSTM gates modulate internal cell state; not conditioned on external domain-specific signals. |
| **Dauphin et al. (2017)** *Language Modeling with Gated Convolutional Networks*, ICML 2017 | Gated Linear Units (GLU) for sequence modelling. h = (Xw + b) x sigma(Xv + c). | General multiplicative gating; no domain-specific conditioning on carbon/energy signals. |
| **FiLM (Perez et al., 2018)** *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018 | Feature-wise Linear Modulation: gamma*h + beta, where gamma/beta from conditioning signal. | Close conceptual relative. MCAG uses multiplicative-only gating (no bias shift) + LayerNorm + sigmoid, tuned for time-series carbon conditioning. |
| **Shih et al. (2019)** *Temporal Pattern Attention for Multivariate Time Series Forecasting*, ECML-PKDD | Attention over variable importance for multivariate forecasting. | Attends to variable importance, not to a specific signal class (carbon). |

### Mixture-of-experts / gating for energy time series (adjacent field, checked directly against MCAG's claim)

MCAG's "first carbon-class-specific gate" claim needs to be checked against the broader
gated/MoE-for-energy-forecasting literature, which is active and growing — these papers
gate on *scale* or *region*, not on a carbon-signal class, which is the precise distinction
MCAG's claim rests on:

| Paper | Key Contribution | How MCAG Differs |
|---|---|---|
| **Kim et al. (2025)** *MoEKAN: Multi-Scale Transformer-Based Gating KAN Experts Network for Time Series Forecasting*, Sensors 25(23) | Transformer-gated mixture of Kolmogorov-Arnold-Network experts for OLTC (transformer equipment) vibration forecasting; gate selects across multi-scale experts. | Gates on signal *scale*, not signal *class*. No carbon or domain-specific conditioning. |
| **Babakhani et al. (2026)** *An Explainable Transformer-Based Mixture of Experts for Heat Load Forecasting*, IEEE Access (TU Berlin) | Clustering-based expert specialization for building heat-load forecasting; 10-24% NRMSE reduction over single models across 3 datasets. | Experts specialize by building/cluster type, not by an external signal class. Confirms MoE/gating is a proven technique for energy forecasting generally — MCAG's contribution is *which* signal it conditions on. |
| **TriForecaster (2025)** *A Mixture of Experts Framework for Multi-Region Electric Load Forecasting with Tri-dimensional Specialization*, arXiv:2508.09753 | MoE with region/temporal/pattern-dimension expert specialization for multi-region load forecasting. | Specializes on region/pattern, not carbon. Closest in spirit (multi-region grid data, like RE-FUSED's 18-state panel) but no carbon signal. |
| **AdaMixT (IJCAI 2025)** | Adaptive gating network dynamically weighting multi-scale expert Transformers. | General multi-scale gating; not domain-signal-conditioned. |

**Revised claim (narrower, more defensible):** gating/MoE architectures for energy
time-series forecasting are an established and active technique (scale-conditioned,
region-conditioned, cluster-conditioned). What appears absent from this literature,
based on the search conducted for this work, is a gate conditioned specifically on a
*carbon-signal class* as opposed to scale/region/cluster structure. That is the precise
scope of MCAG's novelty claim — not "the first gating mechanism for energy forecasting,"
which would be false given the papers above.

---

## What Is New in MCAG (N2)

1. **Carbon-Class-Specific Gating**: gated/MoE architectures for energy forecasting are
   established (scale-, region-, and cluster-conditioned variants exist — see the MoE/gating
   table above). MCAG's claim is narrower and more precise: it appears to be the first such
   gate conditioned specifically on a carbon-signal class, rather than general covariate
   importance, temporal scale, or geographic region.

2. **India-Specific Carbon Signal Set**: The 5 signals {p_carbon_gcal, lcmp_carbon, carbon_budget_pressure, freq_excursion_risk, monsoon_re_risk} are India-specific. Prior energy forecasting work uses generic economic features; MCAG uses signals derived from N1 (PA-LMP) and N3 (GCAL) specifically.

3. **Strict Three-Way Ablation Protocol**: The MCAG-Real vs MCAG-Null vs MCAG-Price ablation is more rigorous than typical attention weight analysis. MCAG-Price specifically tests whether the gate is learning carbon-specific structure vs. price autocorrelation — a distinction not made in prior gating literature.

4. **Patch-Level Integration**: MCAG operates on patch-level representations (from PatchTST), not on raw time steps. The gate modulates the patch attention before final projection, allowing carbon conditioning at the temporal-pattern level rather than the raw feature level.

---

## Ablation Design Sources

| Ablation Variant | Inspired By | Purpose |
|---|---|---|
| MCAG-Real | Standard evaluation | Baseline (full model) |
| MCAG-Null | Goodfellow et al. (2016), *Deep Learning*, Ch. 11 (ablation studies) | Controls for gate trivially averaging |
| MCAG-Price | Bengio et al. (2013) *Representation Learning Review* — disentanglement analysis | Tests information orthogonality: carbon != price |

---

## Key References for MCAG Section

1. Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers*. ICLR 2023. [PatchTST backbone]
2. Lim, B., Arik, S., Loeff, N., Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting*. Int. J. Forecasting, 37(4), 1748-1764. [TFT baseline]
3. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018. [Closest prior gating work]
4. Dauphin, Y., Fan, A., Auli, M., Grangier, D. (2017). *Language Modeling with Gated Convolutional Networks*. ICML 2017. [GLU gating family]
5. Ba, J.L., Kiros, J.R., Hinton, G.E. (2016). *Layer Normalization*. arXiv:1607.06450. [LayerNorm in MCAG gate]
6. Wang, J., Li, Y., Chen, C. (2022). *Carbon emission price prediction in China's ETS markets*. Energy Reports. [Carbon price DL forecasting prior work]

