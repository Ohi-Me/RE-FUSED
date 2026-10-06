# N3 — GCAL (Global Carbon Accounting Layer)
## Novelty Summary & Literature Grounding

---

## What GCAL Is

GCAL is a blended 4-scheme carbon pricing mechanism that:
1. Generates a unified carbon price signal for PA-LMP and MCAG
2. Produces Carbon Flexibility Credits (CFC) that offset DISCOM tariff burden

**Blended Carbon Price Formula:**
```
p_carbon_GCAL = 0.40 × PAT_progressive(CBP)
              + 0.30 × ETS_pilot_PPP_adjusted
              + 0.20 × ShadowSCC_growing

where:
  PAT_progressive  = 200 + 4800 × CBP^1.5      [INR/tCO2; progressive with budget pressure]
  ETS_pilot_PPP    = 5850 × 0.25 × (1 + CBP)   [EU ETS price, PPP-adjusted to India]
  ShadowSCC        = 6900 × 1.02^(year-2017)    [US EPA SCC, growing 2%/yr]

Carbon Flexibility Credit:
  CFC_{n,t} = (EF_coal - EF_actual_{n,t})+ × gen_{n,t} × p_carbon / 1000   [INR/state-day]
```

---

## What Was Already Known (Prior Work)

### Carbon Pricing Mechanisms

| Mechanism | Description | GCAL Difference |
|---|---|---|
| **EU ETS (2005-present)** | Cap-and-trade covering ~40% of EU emissions. Current price ~EUR 60-80/tCO2 (2024). | Designed for liberalised European market. PPP adjustment to India (×0.25) needed; direct import inapplicable. |
| **India PAT Scheme (Perform, Achieve & Trade), 2008** | Energy Efficiency Certificate (ESCert) trading for large industries (Designated Consumers). NOT electricity generation. | PAT covers ~700 DCs in industry, not power generators. GCAL borrows PAT's progressive structure and applies it to generation. |
| **India Carbon Credit Trading Scheme (CCTS), 2023** | CCTS framework notified by MoEFCC. Implementation pending. No published credit price yet. | GCAL is designed to be CCTS-compatible when implemented. The blended price provides a proxy until CCTS credits are liquid. |
| **US EPA Social Cost of Carbon (SCC)** | USD 51/tCO2 (2021), USD 190/tCO2 (2023 revised). Used for regulatory cost-benefit analysis. | SCC is a damage estimate, not a traded price. GCAL uses SCC as one component (20% weight) scaled to India PPP. |
| **China National ETS (2021-present)** | Power sector only (3.5 GtCO2/yr cap). Intensity-based benchmarks, not absolute caps. | China NETS uses intensity benchmarks similar to GCAL's emission factor approach. GCAL adapts this for India's mixed generation portfolio. |
| **Kumar & Managi (2009)** *CO2 Emission Trading in India*, Energy Policy | Early analysis of hypothetical carbon trading in India power sector. Found distributional concerns for coal-dependent states. | Motivates GCAL's CFC mechanism to offset tariff burden for coal-heavy DISCOM states. |
| **Shrimali et al. (2013)** *Renewable Energy in India*, Energy for Sustainable Development | Analysed RPO (Renewable Purchase Obligation) as India's de facto carbon signal. Found weak compliance. | GCAL replaces RPO-as-carbon-signal with an explicit, quantified carbon price. |

### DISCOM Financial Mechanism

| Reference | Finding | Relevance to GCAL CFC |
|---|---|---|
| **ICRA (2024)** *Indian Power Sector: Distribution Company Credit Analysis* | DISCOM accumulated losses: INR 6.5 lakh crore (USD 78B) as of 2024. | Primary motivation for CFC offsetting mechanism. |
| **Prayas (2022)** *DISCOM Financials Report* | Average AT&C losses: 17.4% (2021-22). DISCOMs cannot absorb cost increases. | CFC must be structured as a revenue credit, not a pass-through cost. |
| **MoP (2022)** *Revamped Distribution Sector Scheme (RDSS)* | INR 3.03 lakh crore scheme for DISCOM loss reduction 2021-26. | CFC complements RDSS by adding a carbon revenue stream to loss-making DISCOMs. |
| **Bhattacharya & Cropper (2010)** *Options for Energy Efficiency in India*, Resources for the Future | Found that carbon pricing above INR 500/tCO2 increases residential tariffs by >15%, risking affordability. | GCAL's CFC mechanism is designed so that carbon revenue (CFC) offsets tariff increases. |

---

### Note: GCAL (pricing) vs. carbon-intensity *forecasting* (a different task, checked separately)

GCAL prices carbon — it does not forecast carbon intensity. The `log_carbon` target
forecast elsewhere in this notebook (Stage 2/3/4 architecture comparison, Section 3) is
the forecasting task, and belongs to a separate, active international literature that
GCAL itself should not be conflated with:

| Paper | Task | Relevance |
|---|---|---|
| **Maji, Shenoy, Sitaraman** *DACF: Day-ahead Carbon Intensity Forecasting of Power Grids*, UMass Amherst (NSF) | Forecasts grid carbon intensity for EV-charging and demand-shifting applications. | Forecasts intensity, not price. Confirms carbon-intensity forecasting (not pricing) is the established international task most comparable to `log_carbon`. |
| **(2026)** *A node-aware Graph Neural Network-based carbon intensity forecasting model for cross-border power grids*, ScienceDirect | GNN forecasting carbon intensity across interconnected national grids (e.g. Switzerland importing from Germany/Austria/Italy/France). | Cross-border framing doesn't apply to India's single-national-grid, state-level structure, but the underlying forecasting task is the same one `log_carbon` addresses. |

This confirms GCAL's specific claim (a blended *pricing* mechanism) sits in a different
part of the literature than carbon-intensity forecasting, and should be cited/discussed
separately from the Stage 2/3/4 `log_carbon` forecasting results rather than bundled
together, since a reviewer familiar with the DACF/GNN carbon-forecasting literature could
otherwise mistake GCAL for a forecasting contribution rather than a pricing one.

---

## What Is New in GCAL (N3)

1. **First India-Specific Blended Carbon Price**: No prior work combines PAT (India scheme), EU ETS (PPP-adjusted), and SCC (growing trajectory) into a single blended price for power sector dispatch. Prior papers use one proxy only.

2. **CFC as DISCOM-Compatible Transfer**: Carbon Flexibility Credits are structured as *state-level credits* (not firm-level), making them compatible with India's state-wise DISCOM structure. Prior carbon trading designs (EU ETS, China NETS) are firm-level.

3. **Progressive CBP-Linked Pricing**: The PAT component scales as CBP^1.5 (progressive), meaning states that consistently over-use their carbon budget face exponentially higher marginal prices. This non-linear structure is absent from flat-rate carbon pricing in prior work.

4. **PPP-Adjustment Methodology**: GCAL's PPP adjustment (×0.25) for EU ETS price is derived from World Bank PPP conversion factors for India 2022-23, not assumed. Prior work using EU ETS as a proxy for India simply notes the price without adjustment.

---

## Key References for GCAL Section

**Carbon Markets:**
1. European Commission (2024). *EU ETS: Facts and Figures 2024*. Brussels. [EU ETS price benchmark]
2. BEE (2022). *PAT Scheme Cycle-III: Guideline Document*. Bureau of Energy Efficiency, New Delhi. [PAT structure]
3. MoEFCC (2023). *Carbon Credit Trading Scheme (CCTS) Notification*. Ministry of Environment, Forest & Climate Change, India. [India CCTS framework]
4. US EPA (2023). *Technical Support Document: Social Cost of Carbon, Methane, and Nitrous Oxide*. EPA, Washington DC. [SCC values]
5. World Bank (2024). *State and Trends of Carbon Pricing 2024*. Washington DC. [Global carbon pricing review]

**DISCOM Finance:**
6. ICRA (2024). *Indian Power Sector: Distribution Company Credit Analysis Q3 FY2024*. ICRA Ratings, Mumbai. [INR 6.5L cr losses]
7. Prayas Energy Group (2022). *Electricity Distribution Sector: Performance, Issues and Reforms*. Pune. [AT&C losses]
8. MoP (2022). *Revamped Distribution Sector Scheme: Implementation Guidelines*. Ministry of Power, New Delhi. [RDSS scheme]

**India Carbon Policy:**
9. Kumar, S., Managi, S. (2009). *CO2 emission reductions, co-benefits, and the Clean Development Mechanism in India*. Energy Policy, 37(9), 3341-3351. [Carbon trading distributional analysis]
10. CEA (2024). *CO2 Baseline Database for the Indian Power Sector: Version 17*. Central Electricity Authority. [Emission factor data]

