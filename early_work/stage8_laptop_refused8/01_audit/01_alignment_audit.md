# RE-FUSED-8 / Stage 0 — alignment audit: CPRI RSoP proposal × pilot (RE-FUSED-Alpha) × RE-FUSED-7

Date: 14 Sep 2026. Inputs read in full: `CPRI_RSoP_Submission_18-Sep-2026/01_CPRI_RSoP_Proposal_Form_F1_Rev03.pdf`
(21 pp.), `Refused0/REFUSED_Ready/{README.md, LEAKAGE_AUDIT.md, PUBLICATION_AUDIT_ALPHA.md}`,
`Refused0/Data_Preprocessing/{dataset, cleaned_v2 diagnostics, logs}`, `REFUSED7/` (complete).

Purpose: RE-FUSED-8 must (i) deliver the five proposal objectives, (ii) repair every defect the pilot's own audits found,
(iii) reuse what RE-FUSED-7 established (protocol, statistics, the price-of-adaptivity principle), (iv) use only official
public Indian government data, and (v) be defensible at top journals.

---

## 1. Proposal objectives → evidence required → status before RE-FUSED-8

| Obj. | Proposal commitment (verbatim targets) | Evidence a top journal needs | Pilot status | RE-FUSED-8 action |
|---|---|---|---|---|
| **O1** Dataset & protocol | unit registry 100 % of fields; imputation rate and run length per field; zero features fitted outside training; checksum-identical rebuild; protocol frozen before comparisons | official sources with provenance; deterministic build; leakage scan; documented splits | panel of 47,286 state-days built from India Data Portal (a third-party aggregator); `consumption` 100 % missing, market price 85 %, coal stock 72 %, RE 32 %; 87 % of rows imputed; one leaky composite (GSI, refit Δ 0.44); GWh/MWh and t/kt label errors | **rebuild from primary government portals** (Grid-India, CEA/NPP, IMD, CEA CO₂ database); legacy files kept only for cross-validation of our parsers |
| **O2** Forecasting & uncertainty | state total generation, renewable generation, market price; 3-day horizon from 14-day history; beat seasonal naive, DM p < 0.05, ≥ 5 seeds; coverage within ±5 pp at 80 % and 90 % | observable targets, strong baselines, multi-seed, calibrated intervals, proper scores | single seed, point outputs only, selection by an invalid weighted score, 4 of 6 targets synthetic composites | observable targets only (+ demand and drawal, needed by O4/O5); BiLSTM, TFT, PatchTST as proposed **plus** seasonal naive, LightGBM, Chronos-Bolt zero-shot; quantile heads and conformal calibration; ≥ 5 seeds; DM-HLN and block bootstrap |
| **O3** MCAG & GCAL | each component kept only if it beats matched-capacity baseline (p < 0.05, ≥ 5 seeds), null and permuted context, and holds on fully observed data | pre-specified ablations; a principled prediction of *when* adaptive fusion should help | 5-way ablation, single seed, boolean "confirmed" labels | ablations A2–A4, A6 as specified **and** a pre-registered price-of-adaptivity prediction (RE-FUSED-7) of where adaptive context gating can pay |
| **O4** Context-aware deviation assessment (PA-LMP) | A = \|Δ\|·[λ + w₁R + w₂S + w₃C + w₄U]; weights estimated from data; U must beat a random placebo of equal variance; regime-stratified comparison with bootstrap CIs; variance decomposition across states | a ground truth: weights must be estimated against, and validated on, **observed consequences** not used to build A | PA-LMP was a closed-form function of its own inputs, forecast as a target (circular); 5-minute labels on daily data | Δ from official schedule data (Grid-India drawal schedule vs actual; CEA programmed vs actual generation); λ from Grid-India DSM normal rates by bid area; weights fitted to observed consequences (intra-day deviation severity, energy shortage, frequency stress) on training data only; placebo and permuted-U controls |
| **O5** Robust sequential decision support | four arms (standard sequential, fixed-risk robust, volatility-scaled robust, regime-coupled adaptive risk) ≥ 5 seeds, CVaR₉₀, zero hard-limit violations; integrated chain vs best single component on reserved period without re-tuning | costs from realised data, recorded historical actions as a baseline, honest environment | environment did not respond to actions (fixed later by a heuristic imprint); adaptive λ was first a constant; single run; reward built from synthetic composites | decision = risk-aware day-ahead drawal scheduling per state; cost = realised deviation × official DSM rate (+ CVaR); recorded Grid-India drawal schedule is the historical action; PPO kept as one arm, analytic DRO-CVaR policies as the others; walk-forward replay on a reserved period |

## 2. Pilot defects (from its own audits) and their RE-FUSED-8 resolution

| Pilot defect | Severity (pilot audit) | Resolution in RE-FUSED-8 |
|---|---|---|
| "5-minute / intraday" framing on daily data | reviewer-killer | daily resolution throughout; the only sub-daily quantities used are official 15-min DSM rates and Max OD/UD (MW), each labelled as such |
| Synthetic/circular targets (GSI, FCFS, PA-LMP, log-carbon) | high | forecast only observable quantities; derived indices are never targets; O4 weights are validated on consequences not used to build them |
| Leakage: global normalisation of composites; test-median imputation | high / medium | all transforms fitted on the training split; leakage scanner in the test suite; blinded reserved period |
| 87.3 % imputed rows; consumption 100 % missing; price 85 % missing | high | primary sources remove most gaps (Grid-India energy met and DSM rates are published daily); remaining gaps reported per field with run lengths; every headline result also on fully observed rows |
| Unit mislabelling (GWh as MWh, kt as t) | medium | unit registry with source unit, canonical unit and conversion per field; unit tests |
| Single seed, one split, boolean "confirmed" | high | ≥ 5 seeds, day-block bootstrap across states, DM-HLN, Holm/BY multiplicity, pre-registered thresholds |
| Weighted score selection | high | selection by MASE/CRPS on validation only |
| RL environment without action consequences; constant "adaptive" λ | high | decision evaluated on realised DSM exposure; risk weight β(n,t) genuinely time-varying from forecast uncertainty and inferred regime; environment dynamics for the PPO arm documented and tested |
| No probabilistic outputs | high | quantile heads + split-conformal calibration; coverage, pinball, CRPS |
| Data from a third-party aggregator | new requirement (user: government data only) | India Data Portal files are retained in `03_data/raw/legacy_idp/` only to cross-check our parsing of the original CEA/NPP reports |

## 3. What RE-FUSED-7 contributes (reused, not re-claimed)

* **Protocol:** development/confirmatory separation, SHA-256 + git tag pre-registration, a data loader that refuses
  reserved rows until unlocked, evidence-unit inference, block bootstrap, Holm/BY, claims map, `check_claims.py`.
* **Science:** the persistence-adjusted crossing condition (price of adaptivity). In RE-FUSED-8 it is the organising
  hypothesis for O3 (adaptive context gating vs fixed fusion), O2 (per-state vs regime-cell calibration) and O5
  (regime/uncertainty-adaptive risk posture vs fixed risk): adaptivity should pay only where the heterogeneity it
  exploits is large, persists into the reserved period and is estimable. RE-FUSED-7 found this quantitatively fragile
  near its boundary; RE-FUSED-8 tests it on a new domain and states the boundary sensitivity in advance.
* **Code:** statistics, probabilistic scoring, conformal quantiles, gates, selection estimators and the blinding guard
  are ported into the `refused8` package with attribution.

## 4. Data reality check (government portals, verified online 14 Sep 2026)

| Source (publisher) | Content | Coverage verified | Role |
|---|---|---|---|
| Grid-India Daily PSP report (Grid Controller of India Ltd, Govt. of India enterprise) | state-wise max demand met, shortage, energy met, **drawal schedule, OD/UD (MU), max OD (MW)**, energy shortage; all-India frequency profile (FVI, % time bands); regional hydro/wind/solar; source-wise generation; generation outage; 15-min all-India time series (recent) | FY2013-14 → 14 Sep 2026 via the site's public file API (PDF to ~2024, XLS from ~2024) | deviation Δ (drawal side), demand, reliability consequences, system stress |
| Grid-India DSM rates | 15-min normal rate of charges for deviation by bid area (A1…W3), paise/kWh | CY2018 → Sep 2026 (format changes with DSM Regulations 2014/2022/2024) | λ(t) by region; system-level price signal |
| CEA National Power Portal, Daily Generation Report (DGR sub-report 2) | unit-wise monitored capacity, **programmed and actual generation** | direct archive from ~2019 → 11 Sep 2026 (earlier through the legacy IDP copy of the same reports) | generation targets; generation-side Δ |
| CEA NPP DGR sub-report 10 / 11 | unit-wise planned and forced maintenance (MW) | ~2019 → Sep 2026 | outages, availability stress |
| CEA NPP Daily Coal Report | plant-wise coal stock (days), requirement, receipt | to 13 Sep 2026 (archive depth to verify) | fuel adequacy, coal-critical regime |
| CEA CO₂ Baseline Database (v19–v21 held; newer to check) | unit-wise emission factors | FY2022-23 → FY2024-25 versions | carbon intensity (GCAL) |
| IMD Pune gridded data | daily rainfall 0.25° (1901–2024/25), max/min temperature 1° | to 2024 verified; 2025–26 to verify | weather drivers |

**Known limitation (to be stated, not hidden):** India's schedules and deviations are settled per 15-min block; the
public daily PSP gives net daily OD/UD and the daily maximum OD (and, recently, maximum UD). Intra-day offsetting
deviations are invisible in the daily net. Market clearing prices are published by the power exchanges (not
government); RE-FUSED-8 therefore uses the DSM normal rates published by Grid-India, which since 2022 are computed from
exchange clearing prices, as the government-published price signal.

## 5. Consequences for the research design

1. The unit of analysis becomes the **state control area × day** from Grid-India (≈ 30 states/UTs), 2018–2026, with
   the pilot's 18 States as a continuity subset.
2. The proposal's deviation Δ is available from two official sides: drawal (Grid-India, the settled quantity for
   State control areas) and generation (CEA programme vs actual, the proposal's wording). Both are built; drawal-side
   Δ is primary for O4/O5 because it is what the Deviation Settlement Mechanism prices.
3. A reserved, never-inspected period (1 Apr 2025 – 31 Aug 2026) is enforced in code for the confirmatory walk-forward
   replay required by O5/WP-6.
4. Every O4 and O5 claim is validated against observed outcomes that are not inputs to the construct.
