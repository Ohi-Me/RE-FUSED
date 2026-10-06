# O5 round 2 — scheduling with the round-2 forecaster (stylised settlement)

Forecast source: `ens_stack` (window 365 days). Tuned on validation: {"A2": {"beta": 0, "rho": 0.5}, "A5": {"beta": 0}, "A4s": {"b0": -4, "bU": 0, "bS": 4, "rho": 0.5}, "A5c": {"b0": -4, "bU": 1, "bS": 2}}.

| split | arm | system_mean_crore | system_cvar90_crore | J | deviation_energy_gwh_per_state_day | hard_limit_envelope_exceedance_share | n_state_days | source |
|---|---|---|---|---|---|---|---|---|
| validation | H_final | 2.0257 | 4.0394 | 3.0326 | 1.1764 | 0.1236 | 12070 | ens_stack |
| validation | H_pub | 15.2385 | 35.9283 | 25.5834 | 6.1426 | 0.0 | 12070 | ens_stack |
| validation | P | 12.3625 | 28.4621 | 20.4123 | 5.3355 | 0.0 | 12070 | ens_stack |
| validation | A2 | 12.4299 | 26.5207 | 19.4753 | 5.7144 | 0.0 | 12070 | ens_stack |
| validation | A5 | 12.3582 | 28.251 | 20.3046 | 5.4764 | 0.0 | 12070 | ens_stack |
| validation | A4s | 12.4385 | 26.5394 | 19.4889 | 5.7164 | 0.0 | 12070 | ens_stack |
| validation | A5c | 12.3568 | 28.2731 | 20.3149 | 5.4759 | 0.0 | 12070 | ens_stack |
| dev_test | H_final | 2.2662 | 4.6478 | 3.457 | 1.3219 | 0.2018 | 17543 | ens_stack |
| dev_test | H_pub | 18.537 | 47.467 | 33.002 | 6.9367 | 0.0 | 17543 | ens_stack |
| dev_test | P | 15.9184 | 41.7127 | 28.8156 | 6.1282 | 0.0 | 17543 | ens_stack |
| dev_test | A2 | 15.7991 | 38.5083 | 27.1537 | 6.578 | 0.0 | 17543 | ens_stack |
| dev_test | A5 | 15.7956 | 39.9312 | 27.8634 | 6.2983 | 0.0 | 17543 | ens_stack |
| dev_test | A4s | 15.8161 | 38.539 | 27.1775 | 6.5876 | 0.0 | 17543 | ens_stack |
| dev_test | A5c | 15.8002 | 39.9704 | 27.8853 | 6.2992 | 0.0 | 17543 | ens_stack |

Development-test paired tests (system-wide daily regret, crore; negative = a better):

| a | b | metric | mean_diff | lo | hi | p_a_better | n_days | p_holm |
|---|---|---|---|---|---|---|---|---|
| A5 | A2 | mean system regret | -0.0035 | -0.2025 | 0.2343 | 0.4848 | 516 | 1.0 |
| A5 | A2 | tail days (either arm above its 90th pct) | 1.3183 | 0.531 | 2.278 | 0.9975 | 58 |  |
| A5 | P | mean system regret | -0.1228 | -0.335 | 0.0477 | 0.0746 | 516 | 0.4477 |
| A5 | P | tail days (either arm above its 90th pct) | -1.6654 | -2.6855 | -0.9652 | 0.0015 | 55 |  |
| A5 | H_pub | mean system regret | -2.7413 | -3.2026 | -2.2969 | 1.285e-30 | 516 | 1.028e-29 |
| A5 | H_pub | tail days (either arm above its 90th pct) | -7.0572 | -9.5523 | -4.9961 | 0.0004998 | 62 |  |
| A4s | A2 | mean system regret | 0.0169 | 0.0044 | 0.0265 | 0.9993 | 516 | 1.0 |
| A4s | A2 | tail days (either arm above its 90th pct) | 0.0306 | -0.0233 | 0.0732 | 0.8961 | 52 |  |
| A4s | H_pub | mean system regret | -2.7209 | -3.3047 | -2.2171 | 4.694e-22 | 516 | 3.286e-21 |
| A4s | H_pub | tail days (either arm above its 90th pct) | -7.8808 | -10.5877 | -5.6943 | 0.0004998 | 66 |  |
| A5c | A5 | mean system regret | 0.0045 | -0.0015 | 0.0105 | 0.9148 | 516 | 1.0 |
| A5c | A5 | tail days (either arm above its 90th pct) | 0.0392 | 0.0063 | 0.0656 | 0.993 | 52 |  |
| A4s | P | mean system regret | -0.1023 | -0.4899 | 0.2009 | 0.2558 | 516 | 1.0 |
| A4s | P | tail days (either arm above its 90th pct) | -2.7094 | -4.3857 | -1.4029 | 0.0004998 | 60 |  |
| A2 | P | mean system regret | -0.1193 | -0.5029 | 0.1831 | 0.22 | 516 | 1.0 |
| A2 | P | tail days (either arm above its 90th pct) | -2.7264 | -4.3783 | -1.4467 | 0.0004998 | 60 |  |
| P | H_pub | mean system regret | -2.6186 | -2.9963 | -2.2467 | 7.088e-39 | 516 | 6.379e-38 |
| P | H_pub | tail days (either arm above its 90th pct) | -5.5043 | -7.5919 | -3.4463 | 0.0004998 | 61 |  |

Chain check — same arms with the round-1 selected forecaster:

| split | arm | system_mean_crore | system_cvar90_crore | J | deviation_energy_gwh_per_state_day | hard_limit_envelope_exceedance_share | n_state_days | source |
|---|---|---|---|---|---|---|---|---|
| validation | P | 12.3625 | 28.4621 | 20.4123 | 5.3355 | 0.0 | 12070 | ens_stack |
| validation | A2 | 12.4299 | 26.5207 | 19.4753 | 5.7144 | 0.0 | 12070 | ens_stack |
| validation | A5 | 12.3582 | 28.251 | 20.3046 | 5.4764 | 0.0 | 12070 | ens_stack |
| dev_test | P | 15.9184 | 41.7127 | 28.8156 | 6.1282 | 0.0 | 17543 | ens_stack |
| dev_test | A2 | 15.7991 | 38.5083 | 27.1537 | 6.578 | 0.0 | 17543 | ens_stack |
| dev_test | A5 | 15.7956 | 39.9312 | 27.8634 | 6.2983 | 0.0 | 17543 | ens_stack |
| validation | P | 12.6924 | 29.1792 | 20.9358 | 5.4346 | 0.0 | 12070 | lgbm |
| validation | A2 | 12.7932 | 27.2602 | 20.0267 | 5.8544 | 0.0 | 12070 | lgbm |
| validation | A5 | 12.7107 | 29.0103 | 20.8605 | 5.5905 | 0.0 | 12070 | lgbm |
| dev_test | P | 16.2729 | 42.2742 | 29.2736 | 6.2507 | 0.0 | 17611 | lgbm |
| dev_test | A2 | 16.1054 | 39.179 | 27.6422 | 6.6748 | 0.0 | 17611 | lgbm |
| dev_test | A5 | 16.0515 | 40.0986 | 28.075 | 6.4091 | 0.0 | 17611 | lgbm |
