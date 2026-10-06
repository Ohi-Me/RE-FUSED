# O5 round 2 — PPO arm (A1) with the round-2 forecaster

Forecast source `lgbm` (window 180 days), 2,000 steps per seed, 1 seeds, policy on the GPU. Seed chosen on validation J: 0.

| seed | split | system_mean_crore | system_cvar90_crore | J | seconds | selected_on_validation |
|---|---|---|---|---|---|---|
| 0 | validation | 14.429 | 32.8652 | 23.6471 | 9.6 | True |
| 0 | dev_test | 18.277 | 38.351 | 28.314 | 9.6 | True |

Development test, PPO against the other arms (negative = PPO better):

| a | b | metric | mean_diff | lo | hi | p_a_better | n_days |
|---|---|---|---|---|---|---|---|
| A1 (PPO) | H_final | mean system regret | 15.8349 | 12.6837 | 18.4677 | 1.0 | 60 |
| A1 (PPO) | H_pub | mean system regret | -1.1387 | -2.1377 | -0.4364 | 0.0067 | 60 |
| A1 (PPO) | P | mean system regret | 0.1283 | -0.3114 | 0.2715 | 0.6846 | 60 |
| A1 (PPO) | A2 | mean system regret | 0.2357 | -0.3847 | 0.3636 | 0.7778 | 60 |
| A1 (PPO) | A5 | mean system regret | 0.3511 | -0.3345 | 0.5317 | 0.8409 | 60 |
| A1 (PPO) | A4s | mean system regret | 0.1964 | -0.4094 | 0.3391 | 0.7378 | 60 |
| A1 (PPO) | A5c | mean system regret | 0.3197 | -0.3763 | 0.5042 | 0.8189 | 60 |
