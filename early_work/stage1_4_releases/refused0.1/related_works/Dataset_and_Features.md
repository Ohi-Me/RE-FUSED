# Dataset Description & Feature Engineering
## RE-FUSED-V2: India 18-State Grid (2017–2025)

---

## Dataset Overview

| Property | Value |
|---|---|
| Source file | `full_dataset.csv` (47,286 rows × 140 features) |
| Train split | 2017-10-03 to 2023-12-31 → 39,672 rows |
| Test split | 2024-01-01 to 2025-04-22 → 7,614 rows |
| States (18) | Bihar, Chhattisgarh, Delhi, Gujarat, Haryana, Himachal Pradesh, Jammu & Kashmir, Jharkhand, Karnataka, Madhya Pradesh, Maharashtra, Odisha, Puducherry, Punjab, Rajasthan, Tamil Nadu, Uttar Pradesh, Uttarakhand |
| Temporal resolution | Daily (state-day observations) |

---

## Feature Groups

### Raw Features (from IEX/CEA/POSOCO)
| Feature | Source | Unit | Notes |
|---|---|---|---|
| total_generation_mwh | CEA | MWh | Daily generation per state |
| renewable_generation_mwh | CEA | MWh | Solar + wind + hydro |
| avg_coal_stock_days | POSOCO | days | Average coal stock at thermal plants |
| total_outage_mw | NLDC | MW | Forced + planned outages |
| consumption_mwh | CEA | MWh | State-wise consumption |
| avg_market_price | IEX DAM | INR/MWh | Day-ahead market clearing price |
| supply_demand_gap | Computed | MWh | consumption - total_generation |

### Derived Time Features
| Feature | Formula | Purpose |
|---|---|---|
| month_sin/cos | sin/cos(2π×month/12) | Seasonal cyclicity (avoids month boundary discontinuity) |
| dow_sin/cos | sin/cos(2π×dow/7) | Day-of-week cyclicity |
| season | {Winter, Spring, Summer, Monsoon, Post-Monsoon} | India-specific season categories |
| regime | {Pre-Transition, Transition, Post-Shift} | Grid regime based on policy milestones |

### Lag & Rolling Features
| Feature | Formula | Purpose |
|---|---|---|
| gen_lag_1d/3d/7d/14d/30d | gen[t-k] | Auto-regressive generation signals |
| price_lag_1d/7d | price[t-k] | Price momentum |
| gen_rmean_7d/14d/30d | rolling mean | Trend smoothing |
| gen_rstd_7d/30d | rolling std | Volatility estimation |
| gen_mom_7d/30d | gen[t] - gen[t-k] | Momentum signals |

### Coal & Carbon Features
| Feature | Formula | Purpose |
|---|---|---|
| coal_critical | coal_stock < 7 days | Binary: critical coal level |
| coal_low | 7 <= coal_stock < 14 | Binary: low coal level |
| coal_outage_stress | coal_critical × outage_ratio | Combined stress signal |
| carbon_intensity | carbon_proxy / generation | tCO2/MWh |
| carbon_proxy_tons | Estimated from fuel mix | tCO2 per day |
| carbon_budget_pressure | YTD utilisation / annual budget | [0,1] normalised |

### Grid Stress Features
| Feature | Formula | Purpose |
|---|---|---|
| grid_stress_index | Composite of outage, SD gap, coal | [0,1] grid stress score |
| price_cvar90 | Rolling 90th percentile price | Tail price threshold |
| fcfs_priority_score | Must-run + FCFS weighting | Dispatch priority signal |
| ren_intermittency | abs(daily RE change) / mean RE | RE variability measure |

### RE-FUSED-V2 Engineered Features (Novel)
| Feature | Formula | Contribution |
|---|---|---|
| p_carbon_gcal | 0.40×PAT + 0.30×ETS_PPP + 0.20×SCC | N3/GCAL blended carbon price |
| lcmp_carbon | (CI - CI_5th) / (CI_95th - CI_5th) × price × 0.30 | Locational carbon marginal price |
| ef_actual | 0.95×(1-RR) + 0.02×RR | Actual emission factor |
| cfc_value_inr | (0.95 - ef_actual)+ × gen × p_carbon / 1000 | Carbon Flexibility Credits |
| discom_stress | 0.5×price_norm + 0.5×GSI | DISCOM financial stress proxy |
| monsoon_re_risk | (season==Monsoon) × ren_intermittency | India DRO signal |
| freq_excursion_risk | coal_critical OR GSI > 0.7 | Discrete jump indicator |
| ppa_deviation | (price - 3500) / 2556 | Price vs PPA contract normalised |
| palmp | PA-LMP formula (N1) | Full deviation-clearing price |

---

## Train/Test Split Rationale

The 2024-2025 test period was chosen because:
1. It falls entirely *after* India's Post-Shift regime (2022+), testing the model's ability to generalise across regime transitions
2. 2024 includes the first full year post-CCTS notification, testing whether GCAL-derived features remain informative
3. The 2025 data (partial, Jan-Apr) includes a winter-to-spring transition, testing seasonal generalisation

The split is chronological (not random) to prevent data leakage — a critical requirement for time-series forecasting validity.

