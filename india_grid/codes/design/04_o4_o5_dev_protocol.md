# O4 and O5 development protocol (fixed before any O4/O5 computation)

Version 1.0 — 14 Sep 2026. Development data only (FY2018-19 … FY2024-25). It builds on the O2 protocol
(`03_o2_dev_protocol.md`) and research design v1.2 §6–§7.

## O4 — context-aware deviation assessment

### Quantities (State n, day t)

| Symbol | Definition | Timing |
|---|---|---|
| Δ | `dev_od_ud_gwh` (actual − scheduled drawal); secondary Δ_gen = `gen_conv_dev_gwh` | ex post |
| λ | `prc_dsm_normal` of the State's bid area, day t (₹/MWh); R1 charge at 50.00 Hz | ex post (assessment is settlement-type) |
| M₀ | abs(Δ) · λ | ex post |
| R | renewable ramp risk: abs(7-day change) of regional wind + solar (`reg_wind_gwh + reg_solar_gwh`), divided by its 28-day mean, at issue day t−1 (PSP lag 1) | ex ante |
| S | system stress: mean percentile of State outage share and coal capacity share below 7 days (lag 2), plus previous-day State energy shortage > 0 | ex ante |
| C | carbon intensity of State generation `co2_ci_conv_op` (lag 2) | ex ante |
| U | width of the calibrated 80 % interval of the O2 actual-drawal forecast (T2, H = 1, model selected on validation, rolling-180 conformal), divided by the series' MASE scale | ex ante (issued t−1) |

Context terms are measured ex ante so that the assessment reflects the risk known when the schedule was made and
cannot correlate mechanically with same-day consequences. Each of R, S, C and U is mapped to [0, 1] by its empirical
CDF on FY2018-19 … FY2022-23 (U: on FY2023-24, the first year with out-of-sample forecasts). λ is divided by its
FY2018-23 median.

**Assessment:** A = abs(Δ) · [λ/median(λ) + κ (w₁R + w₂S + w₃C + w₄U)], with w on the simplex and κ ∈ {0.25, 0.5, 1, 2}.

### Consequence outcomes (ground truth, never inputs)

* **Y₁ intra-day severity:** max(abs(max OD), abs(max UD)) in MW, divided by the State's average demand (energy met × 1000/24).
  Max UD is missing before 2023, so max OD alone is used where UD is missing (a sensitivity uses 2023 onward only).
* **Y₂ reliability:** 1 if State energy shortage > 0.
* **Y₃ system frequency stress:** All-India % time below 49.9 Hz.
* **Composite Y:** mean of the within-State percentile ranks of Y₁ and Y₂ (ties averaged) and the national percentile
  rank of Y₃.

### Estimation and tests

* **Weights and κ:** grid search (simplex step 0.1; 286 × 4 combinations) on **validation FY2023-24**, maximising the
  mean within-M₀-decile Kendall τ between A and composite Y. The frozen weights are evaluated on the development test
  (FY2024-25). *Deviation from design v1.2 ("training data"): U needs out-of-sample forecasts, which exist only from
  FY2023-24 onward.*
* **A7 (D5):** development-test difference in mean within-decile Kendall τ between A and M₀. M₀ ranks within a decile
  by abs(Δ)·λ. The 95 % CI comes from a 7-day block bootstrap over dates (all States resampled together); one-sided p.
  Outcome-specific versions for Y₁, Y₂, Y₃.
* **U vs placebo:** U is replaced by (i) the same values permuted across days within each State and (ii) Gaussian noise
  with U's mean and variance. Weights are refit on validation for each (50 placebo draws). The observed gain in τ from
  including U must exceed the 95th percentile of placebo gains.
* **Regime stratification:** results by DSM regime (R2: Apr–15 Sep 2024; R3: from 16 Sep 2024) and season.
* **Variance decomposition:** share of variance of log(1 + A) explained by State fixed effects, date fixed effects and
  the residual (two-way ANOVA sums of squares), development test.

## O5 — risk-aware scheduling decision support

### Stylised decision problem

State n chooses its day-ahead scheduled drawal S(n,t) on day t−1. The realised actual drawal X(n,t) is taken as given
(price-taker, no behavioural response: a stated limitation). The daily cost is:

cost = p·S + λ_OD·max(X − S, 0) − λ_UD·max(S − X, 0)

* p is the bid-area DAM ACP of day t (₹/MWh), and λ is the DSM normal rate of day t.
* Within a tolerance band of 10 % of abs(S): λ_OD = λ_UD = λ.
* Beyond the band: λ_OD = 1.2λ and λ_UD = 0.8λ.
* This is a stylised asymmetric settlement informed by the structure of the CERC DSM Regulations. It is **not** a
  replication of actual settlement, and it is labelled as such in all outputs.

Energy is in MWh (GWh × 1000); costs are in ₹ crore.

**Sequential state:** a rolling 7-day deviation-energy budget B(n,t) = b·Σ_{t−7..t−1} abs(S) − Σ_{t−7..t−1} abs(X − S), with
b = 5 %. A budget violation occurs when B < 0 after day t. **Hard limits** on the decision:
S ∈ [S_min, S_max] (the State's FY2018-23 1st and 99th percentile of recorded schedules), and a day-to-day change
of at most the FY2018-23 99th percentile. All arms are projected onto this set, so hard-limit violations must be 0
and are counted.

### Arms

| Arm | Rule |
|---|---|
| H | recorded schedule (`dev_drawal_schedule_gwh`) |
| P | median of the O2 drawal forecast (risk-neutral point) |
| A1 | PPO policy choosing the forecast quantile level of S (action ∈ [0.05, 0.95]); state = forecast quantiles, recent deviations, budget, calendar; reward = −cost; no risk term |
| A2 | DRO-CVaR: S minimises (1−β)·E[cost] + β·CVaR₉₀[cost] over scenarios from the calibrated forecast quantiles widened by factor (1+ρ) around the median. β and ρ are constant, chosen on validation |
| A3 | as A2 with ρ(n,t) proportional to the State's trailing 28-day deviation volatility (published data) |
| A4 | as A2 with β(n,t) = β₀ + β₁·U(n,t) + β₂·1[stress regime]. This is the proposal's regime- and uncertainty-coupled arm, with coefficients chosen on validation |

Scenarios: 99 levels interpolated monotonically from the 7 calibrated quantiles. Tuning grids (validation FY2023-24):
β ∈ {0, 0.25, 0.5, 0.75, 1}, ρ ∈ {0, 0.1, 0.25, 0.5}, β₁ and β₂ ∈ {0, 0.25, 0.5}; PPO 5 seeds, 200k environment steps
on validation, episodes of 28 days per State.

### Metrics and tests (development test FY2024-25)

* **Metrics:** mean daily cost, CVaR₉₀ of daily cost (per State, then averaged), deviation energy, budget violations
  and hard-limit violations.
* **D6:** A4 vs A2 and vs H on CVaR₉₀ and mean cost. Paired daily differences are pooled across States, with a 7-day
  block bootstrap CI and one-sided p. Holm correction within the O5 family.
* **Price of adaptivity (F5):** constant β (A2) → per-regime β → per-instance β(n,t) (A4). The gain of each step is
  compared with its validation PART estimate (RE-FUSED crossing condition).

## Addendum v1.1 (15 Sep 2026, before any O5 computation)

1. **Evaluation quantity.** Regret = cost(S) − p·X, the cost relative to a perfect schedule S = X. Regret depends only on
   the deviation X − S, removing the procurement bill that no schedule can avoid. The system-wide daily regret
   (sum over States, ₹ crore) is the evidence series. The O5 metrics are its mean and its CVaR₉₀ over days; per-State
   means and CVaR₉₀ are reported as secondary.
2. **Tuning criterion (validation FY2023-24).** Minimise J = ½·mean + ½·CVaR₉₀ of the system-wide daily regret.
   This applies to β, ρ (A2), ρ₀ (A3) and β₀, β₁, β₂ (A4). PPO (A1) is selected by the same J across its 5 seeds'
   validation runs, and all seeds are reported.
3. **Prices at decision time.** On day t−1 the rule uses the latest published prices (Grid-India files, 14-day lag):
   the 7-day mean DAM ACP and DSM normal rate ending t−14. Realised regret uses the actual p and λ of day t.
4. **Scenarios.** 99 levels are built by linear interpolation of the 7 calibrated quantiles in probability. Tails
   below 0.05 and above 0.95 are extrapolated with the slope of the adjacent segment. The decision grid has 81 values
   of S spanning the widened 1st–99th scenario range.
5. **F5 for O5.** A PART estimate needs a scalar gate on a corrector, which the β ladder is not. The adaptivity ladder
   for β (constant → per-regime → per-instance) is therefore assessed by prequential replay on validation (each rung
   tuned on FY2023-24 H1, scored on H2) and compared with realised development-test gains.
6. **Historical arm H** is the recorded schedule and is not projected onto the hard-limit set (that would alter
   history). Its exceedances of the envelope are reported; the hard-limit criterion applies to the decision arms.
7. **O4 missing context terms** (v1.1): entities without monitored conventional generation (carbon intensity undefined
   in more than half of their days) get C = 0. S is the mean of its available components. Found in a code check,
   where these NaNs removed about half of the State-days; decided before the O4 run with the selected O2 model.

## Addendum v1.2 (15 Sep 2026, after the O5 code check, before any O5 run with the selected forecasts)

8. **Recorded schedules are final (intra-day revised) schedules.** In the code check, the recorded schedule deviated
   1.18 GWh per State-day from actual drawal, against 5.9 GWh for the median day-ahead forecast. The recorded PSP
   schedule includes revisions made during day t, so it is not a day-ahead decision. `H_final` (recorded) is reported
   as an ex-post reference only. The historical day-ahead comparator is `H_pub`: the latest recorded schedule
   published by decision time (day t−2), projected like the other arms. D6 comparisons "vs H" use `H_pub`.
9. **No-arbitrage settlement.** The v1.0 stylised charges let regret become negative (`H_final` −0.38 crore/day), and
   tuned arms deviated deliberately to exploit gaps between the DSM rate and DAM price, which the DSM Regulations are
   designed to prevent. Charges are now λ⁺ = max(λ, p) for over-drawal (1.2λ⁺ beyond the band) and λ⁻ = min(λ, p)
   for under-drawal credit (0.8λ⁻ beyond the band). Regret is ≥ 0 and zero only when S = X.
   Both issues were found in a quick code check with PatchTST forecasts (development data only); no O5 result has been
   reported.
10. **Settlement parameters checked against the official regulations (v1.3).** Sources, downloaded to
    `03_data/raw/cerc_regulations/` with SHA-256: CERC (DSM and Related Matters) Regulations 2022 (notified 14 Mar 2022)
    and 2024 (notified 5 Aug 2024). Reg. 7 (2024): the normal rate is the highest of the IDAM ACP, the RTM ACP and an
    ancillary-based sum, so NR ≥ DAM ACP. Reg. 8(7) (2024): a buyer's under-drawal is receivable at 90 % of NR and
    over-drawal payable at NR at 50 Hz, with frequency linkage and volume limits. The stylised daily settlement now
    credits in-band under-drawal at 0.9·min(λ, p). Frequency linkage, 15-minute blocks and volume limits are not modelled
    (stated limitation). Changed before any O5 run with the selected forecasts.
