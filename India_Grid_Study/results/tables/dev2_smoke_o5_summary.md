# O5 round 2 — scheduling with the round-2 forecaster (stylised settlement)

Forecast source: `lgbm` (window 180 days). Tuned on validation: {"A2": {"beta": 0, "rho": 0.5}, "A5": {"beta": 0}, "A4s": {"b0": -4, "bU": 0, "bS": 1, "rho": 0.5}, "A5c": {"b0": -4, "bU": 0, "bS": 1}}.

| split | arm | system_mean_crore | system_cvar90_crore | J | deviation_energy_gwh_per_state_day | hard_limit_envelope_exceedance_share | n_state_days | source |
|---|---|---|---|---|---|---|---|---|
| validation | H_final | 2.2488 | 4.029 | 3.1389 | 1.1951 | 0.1206 | 2040 | lgbm |
| validation | H_pub | 16.7864 | 40.0132 | 28.3998 | 6.0892 | 0.0 | 2040 | lgbm |
| validation | P | 14.7807 | 34.9562 | 24.8684 | 5.608 | 0.0 | 2040 | lgbm |
| validation | A2 | 15.0468 | 30.7976 | 22.9222 | 5.8802 | 0.0 | 2040 | lgbm |
| validation | A5 | 15.7201 | 35.9525 | 25.8363 | 5.9011 | 0.0 | 2040 | lgbm |
| validation | A4s | 15.1228 | 31.1459 | 23.1344 | 5.8955 | 0.0 | 2040 | lgbm |
| validation | A5c | 15.7494 | 36.2051 | 25.9773 | 5.9035 | 0.0 | 2040 | lgbm |
| dev_test | H_final | 2.4421 | 4.0389 | 3.2405 | 1.3355 | 0.2451 | 2040 | lgbm |
| dev_test | H_pub | 19.4158 | 41.7407 | 30.5782 | 7.1734 | 0.0 | 2040 | lgbm |
| dev_test | P | 18.1487 | 38.9878 | 28.5683 | 6.8617 | 0.0 | 2040 | lgbm |
| dev_test | A2 | 18.0413 | 36.7248 | 27.383 | 7.0043 | 0.0 | 2040 | lgbm |
| dev_test | A5 | 17.9259 | 36.6541 | 27.29 | 6.9536 | 0.0 | 2040 | lgbm |
| dev_test | A4s | 18.083 | 36.7411 | 27.4121 | 7.0204 | 0.0 | 2040 | lgbm |
| dev_test | A5c | 17.9549 | 36.7231 | 27.339 | 6.9592 | 0.0 | 2040 | lgbm |

Development-test paired tests (system-wide daily regret, crore; negative = a better):

| a | b | metric | mean_diff | lo | hi | p_a_better | n_days | p_holm |
|---|---|---|---|---|---|---|---|---|
| A5 | A2 | mean system regret | -0.1153 | -0.4913 | 0.4162 | 0.2879 | 60 | 1.0 |
| A5 | A2 | tail days (either arm above its 90th pct) | -0.4336 |  |  |  | 8 |  |
| A5 | P | mean system regret | -0.2228 | -0.431 | 0.2138 | 0.1877 | 60 | 1.0 |
| A5 | P | tail days (either arm above its 90th pct) | -2.3605 |  |  |  | 7 |  |
| A5 | H_pub | mean system regret | -1.4898 | -2.1929 | -0.6129 | 0.0007091 | 60 | 0.005 |
| A5 | H_pub | tail days (either arm above its 90th pct) | -4.4475 |  |  |  | 7 |  |
| A4s | A2 | mean system regret | 0.0417 | 0.0024 | 0.0826 | 0.9858 | 60 | 1.0 |
| A4s | A2 | tail days (either arm above its 90th pct) | 0.0163 |  |  |  | 6 |  |
| A5c | A5 | mean system regret | 0.0289 | 0.0132 | 0.0475 | 0.9749 | 60 | 1.0 |
| A5c | A5 | tail days (either arm above its 90th pct) | 0.069 |  |  |  | 6 |  |
| A4s | P | mean system regret | -0.0657 | -0.5456 | 0.414 | 0.4131 | 60 | 1.0 |
| A4s | P | tail days (either arm above its 90th pct) | -1.9704 |  |  |  | 7 |  |
| A2 | P | mean system regret | -0.1074 | -0.5813 | 0.3654 | 0.3578 | 60 | 1.0 |
| A2 | P | tail days (either arm above its 90th pct) | -1.9834 |  |  |  | 7 |  |
| P | H_pub | mean system regret | -1.267 | -1.8838 | -0.6183 | 6.008e-05 | 60 | 0.0004806 |
| P | H_pub | tail days (either arm above its 90th pct) | -2.5285 |  |  |  | 7 |  |

Chain check — same arms with the round-1 selected forecaster:

| split | arm | system_mean_crore | system_cvar90_crore | J | deviation_energy_gwh_per_state_day | hard_limit_envelope_exceedance_share | n_state_days | source |
|---|---|---|---|---|---|---|---|---|
| validation | P | 14.7807 | 34.9562 | 24.8684 | 5.608 | 0.0 | 2040 | lgbm |
| validation | A2 | 15.0468 | 30.7976 | 22.9222 | 5.8802 | 0.0 | 2040 | lgbm |
| validation | A5 | 15.7201 | 35.9525 | 25.8363 | 5.9011 | 0.0 | 2040 | lgbm |
| dev_test | P | 18.1487 | 38.9878 | 28.5683 | 6.8617 | 0.0 | 2040 | lgbm |
| dev_test | A2 | 18.0413 | 36.7248 | 27.383 | 7.0043 | 0.0 | 2040 | lgbm |
| dev_test | A5 | 17.9259 | 36.6541 | 27.29 | 6.9536 | 0.0 | 2040 | lgbm |
