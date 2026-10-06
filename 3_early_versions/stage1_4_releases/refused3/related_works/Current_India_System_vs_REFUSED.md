# Current India Electricity System vs RE-FUSED-V2 Proposal
## Side-by-Side Comparison

---

## 1. Price Discovery

| Dimension | Current India System | RE-FUSED-V2 Proposal |
|---|---|---|
| **Mechanism** | IEX Day-Ahead Market (DAM): uniform-price double-sided auction. Clears at a single Market Clearing Price (MCP) for all buyers/sellers. | PA-LMP: node-differentiated deviation-clearing price layered on top of IEX MCP. Prices the deviation from PPA schedule, not full generation. |
| **Carbon signal** | None. MCP is carbon-blind. High-CI coal plants and zero-CI solar clear at the same price. | Carbon premium ν×CI_norm×p_max in PA-LMP. Carbon-intensive deviations cost more; clean deviations are incentivised. |
| **Reliability signal** | None in MCP. Ancillary Service Market (ASM) handles frequency regulation separately. | Reliability premium φ×1[price>CVaR90]×v_norm activates during tail price events (coal shocks, monsoon drops). |
| **PPA compatibility** | PPAs exist outside IEX; ~85% of generation never participates in DAM. | PA-LMP prices PPA *deviations* only. PPA contracts unchanged. Works within existing legal framework. |
| **Who pays** | All IEX buyers pay MCP. No state-wise differentiation. | States with high-CI or high-GSI deviations pay higher PA-LMP. Creates geographic efficiency signal. |

---

## 2. Carbon Pricing

| Dimension | Current India System | RE-FUSED-V2 Proposal |
|---|---|---|
| **Scheme** | PAT (Perform, Achieve & Trade): energy efficiency certificates for ~700 Designated Consumers. Covers industrial energy intensity, NOT power generation directly. | GCAL: 4-scheme blended price combining PAT (adapted), EU ETS (PPP-adjusted), Shadow SCC, and CBP. Covers power generation explicitly. |
| **Price level** | PAT ESCert price: ~INR 50–500/tCO2 equivalent (implicit). No published generation-sector carbon price. | GCAL mean: INR ~9,000–12,000/tCO2 (test set 2024-25). Progressive with carbon budget pressure. |
| **DISCOM impact** | Carbon cost is a pass-through: any additional cost raises DISCOM procurement cost and ultimately consumer tariff. | CFC mechanism: states generating cleaner than EF_coal baseline *earn* credits. Net effect on DISCOM can be positive (revenue) if RE-heavy. |
| **Regulatory basis** | CCTS notified in 2023 but not operationalised. No carbon price for power sector currently. | GCAL is designed to be CCTS-compatible when implemented. Provides a proxy price now. |

---

## 3. Grid Dispatch

| Dimension | Current India System | RE-FUSED-V2 Proposal |
|---|---|---|
| **Dispatch order** | FCFS merit order + PPAs (must-run). RLNG-based peakers dispatched last. Coal thermal is baseload. No optimization for carbon or reliability jointly. | DRO-CVaR-PPO: learned policy maximising E[reward] - lambda×DRO-CVaR(-R). Explicitly trades expected reward for tail-risk reduction. |
| **Uncertainty handling** | Deterministic forecast used for next-day scheduling. Deviation cost recovered via Unscheduled Interchange (UI) charges. | DRO uncertainty set (chi-squared ball, rho*=0.0806) explicitly hedges against coal-critical distributional shifts. |
| **Coal shock response** | When coal stock <7 days: capacity curtailed ad hoc, frequency drops, UI spikes. No pre-planned hedging. | DRO-CVaR-PPO's rho* is calibrated FROM coal-critical frequency (4.03%). Policy pre-hedges against these scenarios during training. |
| **RE intermittency** | Managed by DSM (Deviation Settlement Mechanism). RE forecasting error settled at UI rate. | DRO rho_regime captures Monsoon RE gap (5.4pp seasonal swing). Policy accounts for monsoon intermittency. |
| **RL vs. rule-based** | Rule-based merit order + operator judgment. No learning from historical patterns. | PPO learns dispatch policy from 7 years of real 18-state data (2017-2023). Test on 2024-25 unseen data. |

---

## 4. Forecasting

| Dimension | Current India System | RE-FUSED-V2 Proposal |
|---|---|---|
| **NLDC method** | ARIMA/SARIMA + expert adjustment for load forecasting. CEA uses regression models for generation planning. | PatchTST+MCAG: patch-level Transformer + carbon-aware gating. Multi-target: generation, price, grid stress, PA-LMP, log-carbon. |
| **Carbon awareness** | No forecasting model includes a carbon signal. Carbon is not a dispatch input today. | MCAG gate conditions patch representations on 5 carbon signals. Ablation shows carbon carries orthogonal information (MAPE improvement vs price-only gate). |
| **Horizon** | Day-ahead (24-hr horizon) for scheduling. | PRED=3 days (72-hour horizon). Provides earlier warning for coal-critical scenarios. |
| **Multi-state** | State-wise separate models (18 independent forecasts). | Single model trained on all 18 states jointly. Learns cross-state patterns (e.g., Maharashtra RE surplus offsets Jharkhand coal deficit). |

---

### Published India-specific electricity forecasting literature (checked directly, not assumed)

The "NLDC method" row above describes operational practice. Separately, there is a real
academic literature specifically on Indian electricity forecasting — checked directly so
RE-FUSED's positioning against India-specific prior work rests on published papers, not just
operational description:

| Paper | Task | Method | Gap vs. RE-FUSED |
|---|---|---|---|
| Gupta et al. (2023), *Predicting Indian electricity exchange-traded market prices*, OPEC Energy Review | IEX day-ahead price forecasting | SARIMA and MLP, compared via Diebold-Mariano test | Price only. No carbon, grid-stress, or dispatch-priority target. Single-target, not multi-target. |
| (2023), *Machine Learning Models for Prediction of Energy Prices in Indian Power-Energy Trading Market* | IEX/PXIL price prediction, survey | Review of ML techniques applied to Indian day-ahead price | Surveys price forecasting only; identifies renewable integration and forecasting accuracy as open challenges, doesn't address carbon-aware dispatch. |
| MPRA (2020), *Forecasting Hourly Prices in Indian Spot Electricity Market* | Hourly IEX price forecasting | Time-series regression model | Notes prior India-specific studies were either dated or data-limited — motivates RE-FUSED's use of the full 2017-2025 18-state panel. Price only, no dispatch/carbon integration. |
| Various (2025), *Modelling and Forecasting day ahead electricity price in Indian Energy Exchange* | IEX price volatility | MSARIMA-EGARCH | Models price volatility specifically; no carbon or grid-stress target, no RL dispatch layer. |

**Honest summary:** every India-specific academic forecasting paper found for this work
addresses IEX day-ahead price in isolation, using classical statistical or shallow ML
methods, evaluated with standard point-forecast metrics. None combine multi-target
forecasting (generation, price, grid stress, dispatch priority, carbon) with a
carbon-aware gate or a risk-aware dispatch policy. This is the specific, checkable basis
for RE-FUSED's "not yet done for India" claim — not an unverified assertion.

---

## 5. Why RE-FUSED-V2 Is Feasible for India (Not Just Theoretically Better)

| Barrier | Status | RE-FUSED-V2 Response |
|---|---|---|
| **PPA contracts** | 25-year contracts, legally binding. Cannot be renegotiated. | PA-LMP is a deviation signal — contracts unchanged. |
| **DISCOM solvency** | INR 6.5L crore losses. Cannot absorb new carbon costs. | CFC generates revenue from clean-dispatch deviation, offsets costs. |
| **Unmetered agri load** | ~22% of consumption is unmetered. Cannot respond to price signals. | Treated as irreducible demand floor in DRO uncertainty radius. |
| **Political economy** | Coal-dependent states (Jharkhand, Chhattisgarh, MP) oppose carbon pricing. | PA-LMP prices only *deviation from PPA* — coal states' base PPA revenue is untouched. |
| **Data availability** | NLDC SCADA and IEX historical data are publicly available (CEA, POSOCO). | Full dataset (2017-2025) built from public CEA/IEX/POSOCO sources. |
| **Regulatory readiness** | CERC has deferred MBED; CCTS framework exists but unimplemented. | GCAL is CCTS-compatible. PA-LMP can be piloted as a supplementary signal without full MBED. |

