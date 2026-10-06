# O2 round 2 — base models, combinations and the final forecaster

Development test FY2024-25 is reported only; every choice was made on the validation year (protocol `codes/design/06_dev2_protocol.md`). MASE < 1 beats the seasonal naive.

## T5 — DAM price (final: **chronos**, calibration window 90 days)

Cross-fitted validation MASE: best_single 0.4371, ens_top 0.5153, ens_stack 1.1911, ens_online 1.6515

| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |
|---|---|---|---|---|---|---|
| bilstm | 1.031 | nan | nan | nan | nan | nan / nan |
| bitcn | 2.649 | nan | nan | nan | nan | nan / nan |
| chronos **(final)** | 0.437 | nan | nan | nan | nan | nan / nan |
| chronos2 | 1.937 | nan | nan | nan | nan | nan / nan |
| chronos_base | 0.607 | nan | nan | nan | nan | nan / nan |
| ens_online | 1.652 | nan | nan | nan | nan | nan / nan |
| ens_stack | 0.731 | nan | nan | nan | nan | nan / nan |
| ens_top | 0.515 | nan | nan | nan | nan | nan / nan |
| lgbm | 1.463 | nan | nan | nan | nan | nan / nan |
| mcag_instance | 2.270 | nan | nan | nan | nan | nan / nan |
| nbeatsx | 3.258 | nan | nan | nan | nan | nan / nan |
| nhits | 1.433 | nan | nan | nan | nan | nan / nan |
| patchtst | 1.275 | nan | nan | nan | nan | nan / nan |
| seasonal_naive | 1.732 | nan | nan | nan | nan | – |
| tft | 1.643 | nan | nan | nan | nan | nan / nan |
| tide | 2.520 | nan | nan | nan | nan | nan / nan |
| xgb | 1.394 | nan | nan | nan | nan | nan / nan |

| test (dev) | H | mean diff | 95% block CI | p (final better) |
|---|---|---|---|---|
| vs seasonal_naive (seasonal_naive) | 1 | +nan | [+nan, +nan] | n/a |
| vs seasonal_naive (seasonal_naive) | 2 | +nan | [+nan, +nan] | n/a |
| vs seasonal_naive (seasonal_naive) | 3 | +nan | [+nan, +nan] | n/a |
| vs seasonal_naive (seasonal_naive) | all | +nan | [+nan, +nan] | n/a |
| vs round1_selected (bilstm) | 1 | +nan | [+nan, +nan] | n/a |
| vs round1_selected (bilstm) | 2 | +nan | [+nan, +nan] | n/a |
| vs round1_selected (bilstm) | 3 | +nan | [+nan, +nan] | n/a |
| vs round1_selected (bilstm) | all | +nan | [+nan, +nan] | n/a |

Seeds of the final forecaster: MASE nan–nan; seeds with p < 0.05 vs seasonal naive: 0/1
