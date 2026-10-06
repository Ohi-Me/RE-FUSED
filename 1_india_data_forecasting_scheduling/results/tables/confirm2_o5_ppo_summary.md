# O5 round 2 — PPO arm (A1) with the round-2 forecaster

Forecast source `ens_stack` (window 365 days), 200,000 steps per seed, 5 seeds, policy on the GPU. Seed chosen on validation J: 1.

| seed | split | system_mean_crore | system_cvar90_crore | J | seconds | selected_on_validation |
|---|---|---|---|---|---|---|
| 0 | validation | 12.2345 | 26.4119 | 19.3232 | 328.7 | False |
| 0 | dev_test | 15.5947 | 38.7576 | 27.1761 | 328.7 | False |
| 1 | validation | 12.1704 | 26.4692 | 19.3198 | 328.3 | True |
| 1 | dev_test | 15.637 | 39.0722 | 27.3546 | 328.3 | True |
| 2 | validation | 13.0106 | 27.0504 | 20.0305 | 329.3 | False |
| 2 | dev_test | 15.9743 | 37.9812 | 26.9778 | 329.3 | False |
| 3 | validation | 12.5707 | 27.5057 | 20.0382 | 333.4 | False |
| 3 | dev_test | 15.7355 | 39.1961 | 27.4658 | 333.4 | False |
| 4 | validation | 12.4542 | 26.2735 | 19.3639 | 326.8 | False |
| 4 | dev_test | 15.6621 | 37.894 | 26.778 | 326.8 | False |

Development test, PPO against the other arms (negative = PPO better):

| a | b | metric | mean_diff | lo | hi | p_a_better | n_days |
|---|---|---|---|---|---|---|---|
| A1 (PPO) | H_final | mean system regret | 13.3708 | 11.7083 | 15.0743 | 1.0 | 516 |
| A1 (PPO) | H_pub | mean system regret | -2.9 | -3.4025 | -2.4239 | 6.22e-30 | 516 |
| A1 (PPO) | P | mean system regret | -0.2814 | -0.5343 | -0.0621 | 0.0052 | 516 |
| A1 (PPO) | A2 | mean system regret | -0.1621 | -0.3447 | 0.0674 | 0.0374 | 516 |
| A1 (PPO) | A5 | mean system regret | -0.1586 | -0.311 | 0.0092 | 0.0192 | 516 |
| A1 (PPO) | A4s | mean system regret | -0.1791 | -0.3637 | 0.0562 | 0.027 | 516 |
| A1 (PPO) | A5c | mean system regret | -0.1632 | -0.3156 | 0.0049 | 0.0172 | 516 |
