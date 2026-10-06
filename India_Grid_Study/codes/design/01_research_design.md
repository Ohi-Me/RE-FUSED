# RE-FUSED research design — forecasting, uncertainty and decision support for schedule deviations in India

Version 1.0 (14 Sep 2026). Design fixed before any RE-FUSED model result. Changes are logged in §11 with reasons.
Aligned to CPRI RSoP proposal *"An Integrated Forecasting, Uncertainty and Decision-Support Framework for
Schedule-Deviation Assessment in Renewable-Rich Indian Power Systems"* (objectives O1–O5, comparisons A1–A8).

---

## 1. Central questions

1. Can a leakage-audited, provenance-tracked State-day dataset be built entirely from official Indian government
   sources, and what does it reveal about schedule deviations? (O1)
2. Which forecasting family best predicts State generation, renewable generation, demand, drawal and the
   deviation price three days ahead, and are its uncertainty statements calibrated? (O2)
3. Does adaptive contextual fusion (MCAG) or the carbon–economic layer (GCAL) add information beyond fixed-weight
   context — and can the price-of-adaptivity condition predict *where* it does? (O3)
4. Does conditioning the assessment of a deviation on forecast confidence and operating context separate
   deviations that differ in their observed consequences but look identical to the energy-only measure? (O4)
5. Does coupling risk posture to forecast confidence and operating regime reduce realised deviation exposure and
   its tail risk, compared with fixed-risk policies and with the schedules actually recorded? Does the integrated
   chain beat its best single component on a reserved period without re-tuning? (O5)

**Unifying hypothesis (from RE-FUSED).** Each adaptive component — context gates (O3), regime-cell calibration (O2),
context weights (O4), regime/uncertainty-coupled risk (O5) — pays only when the heterogeneity it exploits is large,
persists into deployment and is estimable from the data available:
`2⟨h_V, h_T⟩ − ‖h_V‖² > ΔV/n` (persistence-adjusted crossing condition). Components that fail are removed and reported.

## 2. Data (official government sources only)

| Block | Publisher / product | Variables (canonical unit) | Resolution | Span used |
|---|---|---|---|---|
| G1 | Grid-India Daily PSP report | max demand met (MW), peak shortage (MW), energy met (GWh), drawal schedule (GWh), OD/UD (GWh), max OD (MW; max UD printed from 2023), energy shortage (GWh); all-India frequency profile (FVI; % time in bands); regional hydro, wind, solar (GWh); inter-regional schedule/actual/OD; source-wise generation; generation outage (MW) | State-day / region-day / all-India-day | 8 May 2017 – 13 Sep 2026 (3,284 dates; 129 missing in the study window) |
| G2 | Grid-India DSM rate files | normal rate of charges for deviation, DAM and RTM ACP, reference rate, by bid area (₹/MWh); regimes R1 (charge at 50 Hz + ACP, ≤ 4 Dec 2022), R2, R3 (DSM Regulations 2024, ≥ 16 Sep 2024) | block → daily mean, max, std, share of blocks at cap | 19 Dec 2018 – 6 Sep 2026 |
| C1 | CEA / National Power Portal, DGR sub-report 2 | station monitored capacity (MW), programmed and actual gross generation (GWh), capacity under outage (MW), coal-stock days, by fuel | station-day → State-day | 22 Mar 2018 – 11 Sep 2026 (no reports 13 Mar – 31 May 2020) |
| C2 | CEA / NPP, DGR sub-reports 10 and 6 | maintenance (MW); hydro reservoir levels | unit-day → State-day | 23 Mar 2018 – 11 Sep 2026 (downloaded; parsed if needed) |
| C3 | CEA / NPP, Daily Coal Report | stock, requirement, receipt | plant-day | 25 Sep 2018 – 11 Sep 2026 (downloaded; coal-stock days already in C1) |
| C4 | CEA CO₂ Baseline Database v22.0 | station absolute emissions and net generation → State × fuel × FY emission factor (tCO₂/MWh, net and NPP-gross basis) | State-fuel-year | FY2017-18 – FY2025-26 |
| C5 | CEA Daily Renewable Generation Report | State wind, solar, other RE (GWh), two scopes: `all` (State + ISGS) and `ctrl` (State control area, from 1 Jun 2020) | State-day | 1 Jul 2019 – 18 Nov 2025 (publication stopped; checked on the live CEA listing, 14 Sep 2026) |
| I1 | IMD Pune gridded data | rainfall (mm, 0.25°), Tmax/Tmin (°C, 1°) at State load centres; CDD24, HDD18 | State-day | 1 Jan 2017 – 31 Dec 2025 |
| L | Legacy India Data Portal copies of CEA/IMD reports (pilot) | same as C1–C3, C5, rainfall | — | **cross-validation of parsers only** |

**Entity registry.** Grid-India State control areas with continuous reporting (≈ 30 States/UTs; name changes mapped,
e.g. `J&K` → `J&K(UT) & Ladakh(UT)`, `DD`+`DNH` → `DNHDDPDCL`, `Pondy` → `Puducherry`); industrial and railway
entities excluded. Each State is mapped to its region and DSM bid area. The pilot's 18 States form a continuity subset.

**Build rules.** Raw files are stored unchanged with URL, retrieval time, size and SHA-256. Parsers are deterministic;
every field has a registry entry (source column, source unit, canonical unit, conversion). Cross-source checks:
PSP energy met vs CEA generation + drawal; NPP parse vs legacy IDP overlap. Missingness and imputation run lengths
are reported per field and State; no imputation is used for targets or for consequence outcomes.

## 3. Splits and blinding

| Split | Period | Use |
|---|---|---|
| Train | 1 Apr 2018 – 31 Mar 2023 | model fitting |
| Validation | 1 Apr 2023 – 31 Mar 2024 | selection, calibration, weight estimation |
| Development test | 1 Apr 2024 – 31 Mar 2025 | development only (includes the DSM Regulations 2024 change) |
| **Reserved (confirmatory)** | **1 Apr 2025 – 31 Aug 2026** | walk-forward replay once, after the pre-registration is frozen |

The data loader refuses reserved rows unless the pre-registration hash, the git tag `prereg-v1` and
`REFUSED_CONFIRMATORY=1` all match. Data-engineering audits of the reserved period are limited to schema,
missingness and unit checks (no target statistics).

## 4. O2 — forecasting and uncertainty

* **Targets** (State-day, observable): conventional generation (CEA NPP actual, gross), renewable generation (CEA C5
  `all` scope, State level where published, i.e. to 18 Nov 2025; regional wind and solar from G1 §A for the full
  period), energy met (demand), actual drawal; system-level: daily mean DAM ACP and DSM normal rate per bid area.
* **Information set (measured publication timing).** A forecast is issued at 12:00 IST on day t−1 for days t, t+1,
  t+2 (the day-ahead scheduling time). A column may be used up to date t − 1 − `avail_lag_days` (field registry): PSP
  and IMD 1 day (history ends t−2), NPP and CEA RE 2 days (ends t−3), Grid-India price files 14 days (ends t−15),
  CO₂ factors FY−2. Horizons are therefore h = 2, 3, 4 days after the last PSP observation, with a 14-day history window
  per source. A sensitivity run with history ending at t−1 for all sources ("instant publication") measures the cost
  of publication latency.
* **Models:** seasonal naive (7-day), persistence, moving average (anchors); LightGBM; BiLSTM, Temporal Fusion
  Transformer (static State covariates), PatchTST (as in the proposal); Chronos-Bolt zero-shot. Tuning budget recorded
  per family; selection on validation MASE; ≥ 5 seeds.
* **Uncertainty:** quantile heads (τ = 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95) and split-conformal calibration at
  three adaptivity levels (global, per State, per State × regime).
* **Metrics:** MASE and RMSSE vs seasonal naive; pinball loss, CRPS; empirical coverage at 80 % and 90 %.
* **Tests:** DM with HLN correction per State and on the cross-State mean loss differential; week-block bootstrap
  over days (all States resampled together).

## 5. O3 — contextual fusion (MCAG) and carbon–economic layer (GCAL)

* **MCAG:** sigmoid gate from the context vector (market, energy mix, environment, reliability, weather) scaling the
  temporal representation element-wise. **Fixed fusion:** the same context concatenated with learned but
  state-independent weights, matched parameter count.
* **GCAL:** State-day CO₂ emissions and carbon intensity = Σ_fuel NPP gross generation × State × fuel emission factor
  (factor of FY−2 for features available at forecast time; FY−1 for ex-post accounting), with RE in the denominator;
  consumption-based intensity adds net drawal at the day's regional average intensity.
* **Ablations:** A2 MCAG vs fixed fusion (matched capacity); A3 real vs null vs permuted context; A4 price-only vs
  multi-context; A6 with vs without GCAL; each by regime and season; keep-or-remove recorded.
* **Price-of-adaptivity test:** the gate is an adaptivity ladder (fixed → per-regime cell → per-instance). From
  validation data only, PART-style estimates of persistent heterogeneity of the optimal context weight predict for
  which target × region MCAG beats fixed fusion; the prediction is scored on the reserved period.

## 6. O4 — context-aware deviation assessment

For State n and day t:

* **Deviation:** Δ(n,t) = OD/UD (actual − scheduled drawal, GWh) [primary]; Δ_gen = actual − programmed generation
  (CEA) [secondary, the proposal's wording].
* **Current measure:** M₀ = |Δ| · λ̄, λ̄ = daily mean DSM normal rate of the State's bid area.
* **Assessment:** A = |Δ| · [λ̄ + w₁R + w₂S + w₃C + w₄U], with R = renewable ramp, S = ex-ante system stress (generation
  outage share, coal-critical share, previous-day shortage), C = carbon intensity, U = forecast uncertainty issued
  before day t (normalised interval width); all normalised on training data; w ≥ 0.
* **Ground truth (not inputs to A):** consequence outcomes Y₁ intra-day severity (max |OD/UD| MW relative to average
  demand), Y₂ reliability (energy shortage > 0), Y₃ system frequency stress (% time < 49.9 Hz, pooled national).
  Weights are fitted to Y on training data and frozen.
* **Tests:** A7 — within M₀ deciles, does A rank Y better than M₀ (bootstrap CI)? U vs random placebo of equal variance
  and vs permuted U; regime-stratified; variance decomposition of A across States.

## 7. O5 — risk-aware scheduling decision support

* **Decision:** day-ahead scheduled drawal S(n,t) chosen at t−1 from the forecast distribution of actual drawal.
* **Cost (realised):** procurement at the bid-area rate plus deviation charges on (actual − S) with the asymmetric
  structure of the applicable DSM Regulations (stylised; parameters documented), and a rolling 7-day deviation-volume
  budget that carries over between days (the sequential state).
* **Arms:** H historical recorded schedule; P point-forecast schedule (risk-neutral); A1 standard sequential (PPO,
  no risk term); A2 fixed-risk robust (DRO-CVaR, constant β, ρ); A3 volatility-scaled robust; A4 regime- and
  uncertainty-coupled β(n,t) (the proposal's adaptive arm). ρ calibrated from observed conditions on training data.
* **Metrics:** mean cost, CVaR₉₀ of cost, deviation energy, budget violations, hard-limit violations (must be 0),
  ≥ 5 seeds for learned arms.
* **Integrated validation (WP-6):** full chain vs best single-component configuration on the reserved period by
  walk-forward replay, no re-tuning.

## 8. Statistics and multiplicity

Evidence unit: State-day panel with day-block (7-day) bootstrap resampling all States jointly; seeds averaged first.
DM-HLN for pairwise forecast comparisons. Families: F1 forecasting (O2), F2 fusion (O3), F3 assessment (O4), F4
decision (O5), F5 price of adaptivity (O2–O5). Holm within family, Benjamini–Yekutieli across. Acceptance thresholds
follow the proposal (p < 0.05, ≥ 5 seeds, coverage ±5 pp, U beats placebo, zero hard-limit violations).

## 9. Development hypotheses (confirmatory forms fixed in the pre-registration)

| ID | Hypothesis (development form) |
|---|---|
| D1 | The selected model beats seasonal naive on MASE for each retained target (DM p < 0.05, ≥ 5 seeds). |
| D2 | Conformal-calibrated 80 % and 90 % intervals cover within ±5 pp of nominal; per-State calibration beats global on pinball loss; per-State × regime calibration pays only where regime heterogeneity persists. |
| D3 | MCAG beats matched fixed fusion and null/permuted context for a target × region only where validation PART > 0. |
| D4 | GCAL adds skill beyond price-only and multi-context without carbon. |
| D5 | A ranks the consequence outcomes better than M₀ within M₀ deciles; U beats its placebo. |
| D6 | The uncertainty- and regime-coupled risk arm lowers CVaR₉₀ of realised cost versus fixed-risk and historical schedules with no hard-limit violations; the integrated chain beats the best single component. |

## 10. Outputs

* Replication package, CLAIMS_MAP, frozen pre-registration, audits and logs, as in RE-FUSED.

## 11. Change log

| Date | Change | Reason |
|---|---|---|
| 14 Sep 2026 | v1.0 | from the alignment audit (`01_audit/01_alignment_audit.md`) |
| 14 Sep 2026 | v1.1: §2 spans replaced by verified coverage; C5 split into `all`/`ctrl` scopes; C4 at State × fuel × FY; RE target evaluated at State level to 18 Nov 2025 and at regional level for the full reserved period; GCAL factor lags FY−2 / FY−1 | data acquisition and parser validation (`data/docs/SOURCE_REGISTRY.md`). CEA stopped the daily State RE report after 18 Nov 2025; CEA and NPP names do not match reliably at station level; CO₂ factors are published after each FY. Decided before any target statistics of the reserved period were computed |
| 14 Sep 2026 | v1.2: information set defined from measured publication lags (issue 12:00 IST on t−1; per-source lags); latency-cost sensitivity added | publication timing measured from Grid-India API metadata and HTTP headers (PSP median 10.6 h; DSM/ACP files 8–11 days late; NPP d+1 evening). Without it, a t−1 history would use data that were not yet published. Decided before any modelling or reserved-period target statistic |
