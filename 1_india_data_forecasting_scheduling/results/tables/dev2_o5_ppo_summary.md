# O5 round 2 — PPO arm (A1) with the round-2 forecaster

Forecast source `ens_stack` (window 365 days), 200,000 steps per seed, 5 seeds, policy on the GPU. Seed chosen on validation J: 2.

| seed | split | system_mean_crore | system_cvar90_crore | J | seconds | selected_on_validation |
|---|---|---|---|---|---|---|
| 0 | validation | 13.7051 | 32.0653 | 22.8852 | 318.5 | False |
| 0 | dev_test | 15.0016 | 33.4136 | 24.2076 | 318.5 | False |
| 1 | validation | 13.6493 | 31.1584 | 22.4039 | 315.6 | False |
| 1 | dev_test | 14.9755 | 33.1544 | 24.065 | 315.6 | False |
| 2 | validation | 13.8377 | 30.9547 | 22.3962 | 317.8 | True |
| 2 | dev_test | 15.0798 | 33.4022 | 24.241 | 317.8 | True |
| 3 | validation | 13.6148 | 31.9979 | 22.8063 | 322.4 | False |
| 3 | dev_test | 15.0882 | 33.6427 | 24.3654 | 322.4 | False |
| 4 | validation | 13.5975 | 32.9213 | 23.2594 | 322.8 | False |
| 4 | dev_test | 15.1562 | 34.4739 | 24.815 | 322.8 | False |

Development test, PPO against the other arms (negative = PPO better):

| a | b | metric | mean_diff | lo | hi | p_a_better | n_days |
|---|---|---|---|---|---|---|---|
| A1 (PPO) | H_final | mean system regret | 13.0523 | 11.5159 | 14.609 | 1.0 | 355 |
| A1 (PPO) | H_pub | mean system regret | -2.8962 | -3.5407 | -2.3469 | 8.119e-24 | 355 |
| A1 (PPO) | P | mean system regret | -0.2759 | -0.5688 | -0.0109 | 0.013 | 355 |
| A1 (PPO) | A2 | mean system regret | -0.2861 | -0.5266 | -0.0663 | 0.0019 | 355 |
| A1 (PPO) | A5 | mean system regret | -0.1908 | -0.3984 | 0.0023 | 0.0246 | 355 |
| A1 (PPO) | A4s | mean system regret | -0.3118 | -0.5616 | -0.0837 | 0.0009942 | 355 |
| A1 (PPO) | A5c | mean system regret | -0.1917 | -0.4016 | 0.0014 | 0.0244 | 355 |
