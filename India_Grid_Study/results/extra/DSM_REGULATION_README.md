# Exploratory: settlement closer to the DSM Regulations 2024 and rolling schedule limits (not pre-registered)

Script `codes/scripts/extra/x06_dsm_regulation.py`; H100 job 34926 (10 Oct 2026, 409 s), log
`logs/pbs/refused_dsm_regulation.out`; output `dsm_regulation.json`. An earlier run (job 34923,
`dsm_regulation_run1_unprojected_pub.json`) scored the persistence rule without projecting it onto the schedule
limits; the confirmatory run projects it (`d2_09_o5.py`), so job 34926 reports both. A first attempt (34922) was
stopped by the blinding guard before reading any data. Nothing was refitted: frozen solver settings, stored forecasts
of the confirmatory run, evaluation period 1 Apr 2025 - 29 Aug 2026 (516 days, already opened).

Settlements: the paper's stylised model; a daily approximation of the buyer rules of Reg. 8(7) at about 50.00 Hz
(tier 1 = min(10 % of |S|, 2.4 GWh), tier 2 = min(15 % of |S|, 4.8 GWh); over-drawal at lambda, lambda, 2 lambda;
under-drawal credited at 0.9 lambda, 0.8 lambda, 0); the same with over-drawal in tier 2 at 1.5 lambda (low frequency).
Assumptions: one frequency regime per day; small-buyer and renewable-rich limits not applied; 4 % of evaluation
State-days have a negative schedule and are treated with the same rates. Limits: the paper's fixed fitting-period
percentiles, or the same percentiles over the 365 days ending at t-2.

Mean system regret (Rs crore per day; A2 = fixed rule, P = forecast median, R0 = risk-neutral rule):

| Settlement, limits | Persistence (projected / not) | P | A2 | R0 | Fixed level (cal.) | Adaptive level (cal.) | A2 - persistence (projected) | A2 - persistence (not projected) | Adaptive - fixed level | Largest per-group hindsight gain |
|---|---|---|---|---|---|---|---|---|---|---|
| stylised, fixed | 18.54 / 16.71 | 15.92 | 15.80 | 15.63 | 15.49 | 15.49 | -2.74 [-3.32, -2.23] | -0.91 [-1.93, -0.04] | -0.003 [-0.038, +0.028] | 0.33 % |
| DSM nominal, fixed | 77.70 / 70.43 | 67.12 | 66.88 | 66.70 | 66.55 | 66.56 | -10.82 [-12.41, -9.29] | -3.54 [-7.64, -0.03] | +0.011 [-0.078, +0.098] | 0.33 % |
| DSM low freq., fixed | 82.59 / 74.47 | 71.78 | 70.96 | 70.78 | 70.65 | 70.64 | -11.63 [-13.35, -9.99] | -3.51 [-7.72, +0.13] | -0.019 [-0.070, +0.040] | 0.37 % |
| stylised, rolling | 16.50 / 16.71 | 13.61 | 13.49 | 13.30 | 13.14 | 13.15 | -3.01 [-3.65, -2.44] | -3.22 [-3.86, -2.64] | +0.014 [-0.016, +0.043] | 0.37 % |
| DSM nominal, rolling | 69.66 / 70.43 | 57.76 | 57.37 | 57.21 | 57.07 | 57.12 | -12.28 [-14.02, -10.56] | -13.06 [-14.81, -11.25] | +0.050 [-0.018, +0.120] | 0.42 % |
| DSM low freq., rolling | 73.71 / 74.47 | 61.66 | 60.58 | 60.39 | 60.31 | 60.33 | -13.13 [-15.06, -11.28] | -13.88 [-15.80, -11.91] | +0.019 [-0.032, +0.080] | 0.46 % |

Intervals: 7-day block bootstrap, 95 %. The DSM costs are larger because the normal rate, not min(lambda, p), prices
both directions and tier 3 is charged at 2 lambda; the absolute levels are not comparable across settlements.

## Reading

* The negative result on adaptivity holds in all six settings: a level rule adapted to the price ratio and to the
  band-relative risk, chosen on the calibration year, does not beat one fixed level, and per-group hindsight gains
  stay below 0.5 %.
* The fixed-period limits cost the rules about 2.3 Rs crore per day under the stylised settlement (A2 15.80 against
  13.49 with rolling limits); with rolling limits the projection binds on 2.9-3.9 % of A2's State-days instead of 18 %.
* The rules beat persistence in every setting. With the paper's fixed limits the margin over an unprojected
  persistence schedule is much smaller (stylised: -0.91 against -2.74), because projecting the recorded schedule onto
  the stale limits itself adds cost. With rolling limits the margin is large either way.
* R0 has lower mean regret than A2 in every setting (about -0.17), as in the paper; A2 keeps the better tail-weighted
  objective under the stylised settlement.
* The optimal level moves with the settlement (fixed level 54-57 %, best in hindsight 55-60 %), as the paper's
  design analysis says.
