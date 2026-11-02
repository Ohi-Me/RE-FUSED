# Delay experiments (additional analysis, not pre-registered)

## E1. Publication delay as forecast horizon

| Target | Own lag (registry) | States | Effective delay d (median, days) | IQR | pub(h) vs inst(h+round d): cells within ±5% |
|---|---|---|---|---|---|
| T1 | 1 | 34 | 0.98 | 0.95–1.00 | 0.998 |
| T2 | 1 | 34 | 0.97 | 0.91–1.03 | 0.998 |
| T3 | 2 | 27 | 1.77 | 1.60–1.91 | 0.910 |
| T4 | 2 | 16 | 1.57 | 1.37–1.78 | 0.906 |
| T5 | 14 | 13 | 13.00 | 13.00–13.00 | 0.000 |

### Cost of one more day (instant LightGBM, pooled MAE by horizon)

- T1: h1 4.34, h2 6.16, h3 7.34, h4 8.42, h5 8.85, h6 9.43, h7 9.82, h8 10.06; first extra day +42%, then +19%, +15%
- T2: h1 4.34, h2 5.82, h3 6.78, h4 7.45, h5 7.78, h6 8.14, h7 8.35, h8 8.57; first extra day +34%, then +17%, +10%
- T3: h1 5.69, h2 8.30, h3 9.84, h4 10.73, h5 11.27, h6 11.66, h7 11.99, h8 12.29; first extra day +46%, then +18%, +9%
- T4: h1 5.70, h2 7.91, h3 9.11, h4 9.84, h5 10.12, h6 10.53, h7 10.93, h8 10.82; first extra day +39%, then +15%, +8%
- T5: h1 343.89, h2 473.49, h3 623.79, h4 776.82, h5 925.40, h6 953.63, h7 988.77, h8 1007.27; first extra day +38%, then +32%, +25%

## E2. Value of each source's timeliness (pooled MASE, h = 1-3, evaluation period)

| Target | published | instant | only psp | only dsm_prices | only cea_npp | only cea_re | only co2 | only imd_weather |
|---|---|---|---|---|---|---|---|---|
| T1 | 0.931 | 0.801 | 0.791 | 0.926 | 0.919 | 0.920 | 0.929 | 1.023 |
| T2 | 0.955 | 0.830 | 0.820 | 0.956 | 0.953 | 0.952 | 0.953 | 1.036 |
| T3 | 0.865 | 0.665 | 0.805 | 0.857 | 0.666 | 0.859 | 0.861 | 0.891 |
| T4 | 1.296 | 1.120 | 1.228 | 1.295 | 1.283 | 1.120 | 1.293 | 1.282 |
| T5 | 1.143 | 0.481 | 1.212 | 0.494 | 1.143 | 1.143 | 1.143 | 1.150 |

Share of the full timeliness gain (published − instant) captured by making one source timely:

- T1: psp 108%, dsm_prices 3%, cea_npp 9%, cea_re 8%, co2 1%, imd_weather -71%
- T2: psp 107%, dsm_prices -1%, cea_npp 2%, cea_re 3%, co2 2%, imd_weather -64%
- T3: psp 30%, dsm_prices 4%, cea_npp 99%, cea_re 3%, co2 2%, imd_weather -13%
- T4: psp 39%, dsm_prices 1%, cea_npp 7%, cea_re 100%, co2 2%, imd_weather 8%
- T5: psp -10%, dsm_prices 98%, cea_npp 0%, cea_re 0%, co2 0%, imd_weather -1%

## E3. Settlement value of timely publication (T2 drawal, rules P and A2, Rs crore a day)

| Source | Rule | MAE (GWh) | Mean regret | CVaR 90 | J | vs lgbm published (crore/day) | per year (crore) |
|---|---|---|---|---|---|---|---|
| lgbm | A2 | 5.817 | 16.105 | 39.179 | 27.642 | +0.000 | +0 |
| lgbm_instant | A2 | 4.222 | 11.818 | 32.167 | 21.992 | -4.288 | -1565 |
| e2_all_but_cea_npp | A2 | 4.185 | 11.797 | 31.970 | 21.884 | -4.309 | -1573 |
| e2_all_but_cea_re | A2 | 4.215 | 11.847 | 32.117 | 21.982 | -4.258 | -1554 |
| e2_all_but_co2 | A2 | 4.203 | 11.822 | 32.168 | 21.995 | -4.283 | -1563 |
| e2_all_but_dsm_prices | A2 | 4.216 | 11.779 | 31.986 | 21.882 | -4.327 | -1579 |
| e2_all_but_imd_weather | A2 | 4.115 | 11.768 | 31.813 | 21.791 | -4.338 | -1583 |
| e2_all_but_psp | A2 | 6.177 | 16.118 | 40.817 | 28.468 | +0.013 | +5 |
| e2_instant | A2 | 4.222 | 11.818 | 32.167 | 21.992 | -4.288 | -1565 |
| e2_only_cea_npp | A2 | 5.745 | 15.947 | 38.920 | 27.434 | -0.158 | -58 |
| e2_only_cea_re | A2 | 5.701 | 15.847 | 38.877 | 27.362 | -0.259 | -94 |
| e2_only_co2 | A2 | 5.843 | 16.121 | 39.073 | 27.597 | +0.015 | +6 |
| e2_only_dsm_prices | A2 | 5.829 | 16.190 | 39.304 | 27.747 | +0.084 | +31 |
| e2_only_imd_weather | A2 | 6.401 | 16.611 | 41.608 | 29.109 | +0.506 | +185 |
| e2_only_psp | A2 | 4.142 | 11.861 | 31.968 | 21.915 | -4.244 | -1549 |
| e2_published | A2 | 5.816 | 16.122 | 39.110 | 27.616 | +0.016 | +6 |
| ens_stack | A2 | 5.678 | 15.799 | 38.508 | 27.154 | -0.306 | -112 |
| lgbm | P | 5.817 | 16.273 | 42.274 | 29.274 | +0.000 | +0 |
| lgbm_instant | P | 4.222 | 12.113 | 33.582 | 22.847 | -4.160 | -1518 |
| e2_all_but_cea_npp | P | 4.185 | 12.046 | 33.440 | 22.743 | -4.227 | -1543 |
| e2_all_but_cea_re | P | 4.215 | 12.108 | 33.700 | 22.904 | -4.165 | -1520 |
| e2_all_but_co2 | P | 4.203 | 12.074 | 33.864 | 22.969 | -4.199 | -1533 |
| e2_all_but_dsm_prices | P | 4.216 | 12.050 | 33.528 | 22.789 | -4.223 | -1541 |
| e2_all_but_imd_weather | P | 4.115 | 11.734 | 32.609 | 22.172 | -4.539 | -1657 |
| e2_all_but_psp | P | 6.177 | 18.158 | 48.855 | 33.506 | +1.885 | +688 |
| e2_instant | P | 4.222 | 12.113 | 33.582 | 22.847 | -4.160 | -1518 |
| e2_only_cea_npp | P | 5.745 | 16.023 | 41.948 | 28.985 | -0.250 | -91 |
| e2_only_cea_re | P | 5.701 | 16.051 | 42.117 | 29.084 | -0.222 | -81 |
| e2_only_co2 | P | 5.843 | 16.308 | 42.216 | 29.262 | +0.035 | +13 |
| e2_only_dsm_prices | P | 5.829 | 16.375 | 42.279 | 29.327 | +0.102 | +37 |
| e2_only_imd_weather | P | 6.401 | 18.995 | 49.757 | 34.376 | +2.722 | +994 |
| e2_only_psp | P | 4.142 | 11.770 | 32.749 | 22.260 | -4.503 | -1644 |
| e2_published | P | 5.816 | 16.175 | 41.729 | 28.952 | -0.098 | -36 |
| ens_stack | P | 5.678 | 15.918 | 41.713 | 28.816 | -0.355 | -129 |
