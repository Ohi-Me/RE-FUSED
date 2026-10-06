# Pre-registered hypotheses — development rehearsal (FY2024-25, already seen; not a test)

Holm within family, Benjamini–Yekutieli across all primary tests. A hypothesis is supported when its Holm-adjusted p < 0.05 and its extra rule holds (CI side, all seeds, coverage band, zero violations).

Supported: 32 of 50.

| id | family | what | estimate | lo | hi | p | p_holm | p_by | decision | note |
|---|---|---|---|---|---|---|---|---|---|---|
| H1a | F1 | T1: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.4461 | -0.4952 | -0.3887 | 4.555e-67 | 5.01e-66 | 1.655e-65 | supported | seeds p<0.05: 5/5 |
| H1f-T1 | F1 | T1: final forecaster beats the round-1 selected model (lgbm) | -0.0406 | -0.0494 | -0.0279 | 7.237e-19 | 2.895e-18 | 1.012e-17 | supported |  |
| H2-T1 | F1 | T1: 80 % / 90 % coverage within ±5 points | 0.8017 | 0.8017 | 0.8981 |  |  |  | supported | cov80 0.802, cov90 0.898 |
| H3-T1 | F1 | T1: publication-latency cost > 0 (LightGBM, % MASE) | 15.6032 | -0.1393 | -0.1091 | 1.171e-71 | 1.405e-70 | 5.32e-70 | supported |  |
| H1b | F1 | T2: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.4152 | -0.4581 | -0.3713 | 1.361e-83 | 1.906e-82 | 1.237e-81 | supported | seeds p<0.05: 5/5 |
| H1f-T2 | F1 | T2: final forecaster beats the round-1 selected model (lgbm) | -0.04 | -0.0476 | -0.031 | 1.958e-28 | 9.79e-28 | 3.235e-27 | supported |  |
| H2-T2 | F1 | T2: 80 % / 90 % coverage within ±5 points | 0.8027 | 0.8027 | 0.9011 |  |  |  | supported | cov80 0.803, cov90 0.901 |
| H3-T2 | F1 | T2: publication-latency cost > 0 (LightGBM, % MASE) | 16.343 | -0.1458 | -0.1184 | 2.728e-88 | 4.092e-87 | 4.958e-86 | supported |  |
| H1c | F1 | T3: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.2432 | -0.2766 | -0.2107 | 1.936e-48 | 1.743e-47 | 5.026e-47 | supported | seeds p<0.05: 5/5 |
| H1f-T3 | F1 | T3: final forecaster beats the round-1 selected model (lgbm) | -0.0217 | -0.0293 | -0.0133 | 8.352e-12 | 1.67e-11 | 9.486e-11 | supported |  |
| H2-T3 | F1 | T3: 80 % / 90 % coverage within ±5 points | 0.8084 | 0.8084 | 0.9005 |  |  |  | supported | cov80 0.808, cov90 0.900 |
| H3-T3 | F1 | T3: publication-latency cost > 0 (LightGBM, % MASE) | 26.8933 | -0.1774 | -0.1479 | 6.933e-81 | 9.013e-80 | 4.2e-79 | supported |  |
| H1d | F1 | T4: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.2701 | -0.3119 | -0.2301 | 1.002e-43 | 8.015e-43 | 2.276e-42 | supported | seeds p<0.05: 5/5 |
| H1f-T4 | F1 | T4: final forecaster beats the round-1 selected model (chronos) | -0.0513 | -0.062 | -0.0354 | 6.365e-14 | 1.909e-13 | 7.711e-13 | supported |  |
| H2-T4 | F1 | T4: 80 % / 90 % coverage within ±5 points | 0.7974 | 0.7974 | 0.8988 |  |  |  | supported | cov80 0.797, cov90 0.899 |
| H3-T4 | F1 | T4: publication-latency cost > 0 (LightGBM, % MASE) | 18.0887 | -0.1551 | -0.1149 | 8.074e-51 | 8.074e-50 | 2.445e-49 | supported |  |
| H1e | F1 | T5: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.1603 | -0.2892 | -0.0271 | 0.000339 | 0.000339 | 0.0029 | supported | seeds p<0.05: 5/5 |
| H1f-T5 | F1 | T5: final forecaster beats the round-1 selected model (bilstm) | -0.4471 | -0.5605 | -0.3304 | 1.026e-28 | 6.157e-28 | 1.865e-27 | supported |  |
| H2-T5 | F1 | T5: 80 % / 90 % coverage within ±5 points | 0.8429 | 0.8429 | 0.9358 |  |  |  | supported | cov80 0.843, cov90 0.936 |
| H3-T5 | F1 | T5: publication-latency cost > 0 (LightGBM, % MASE) | 148.7752 | -1.0047 | -0.6233 | 2.891e-33 | 2.024e-32 | 5.838e-32 | supported |  |
| H4-T1 | F2 | T1: gated (sd) beats matched fusion (sd), pinball | -0.0042 | -0.0064 | -0.0018 | 1.23e-06 | 1.722e-05 | 1.315e-05 | supported |  |
| H4-T2 | F2 | T2: gated (sd) beats matched fusion (sd), pinball | -0.0063 | -0.0088 | -0.0042 | 2.988e-14 | 4.482e-13 | 3.879e-13 | supported |  |
| H4-T3 | F2 | T3: gated (sd) beats matched fusion (sd), pinball | -0.0026 | -0.0042 | -0.0011 | 2.407e-05 | 0.000313 | 0.0002431 | supported |  |
| H4-T4 | F2 | T4: gated (sd) beats matched fusion (sd), pinball | 0.0017 | -0.0007893 | 0.0044 | 0.9633 | 1.0 | 1.0 | not supported |  |
| H4-T5 | F2 | T5: gated (sd) beats matched fusion (sd), pinball | 0.0152 | 0.0083 | 0.0243 | 1.0 | 1.0 | 1.0 | not supported |  |
| H5-T1 | F2 | T1: real context beats null and shuffled context, pinball | 0.0024 |  | 0.0086 | 0.8275 | 1.0 | 1.0 | not supported | larger of the two p values |
| H5-T2 | F2 | T2: real context beats null and shuffled context, pinball | -0.0026 |  | 0.0014 | 0.0542 | 0.4877 | 0.3517 | not supported | larger of the two p values |
| H5-T5 | F2 | T5: real context beats null and shuffled context, pinball | -0.0147 |  | 0.0166 | 0.0657 | 0.5258 | 0.4118 | not supported | larger of the two p values |
| H6-T1 | F2 | T1: online time-adaptive gate lowers scaled |e| | -0.0138 | -0.022 | -0.0032 | 7.694e-05 | 0.0009233 | 0.0007359 | supported |  |
| H6-T2 | F2 | T2: online time-adaptive gate lowers scaled |e| | -0.0118 | -0.0198 | -0.0029 | 0.0002095 | 0.0023 | 0.0019 | supported |  |
| H6-T3 | F2 | T3: online time-adaptive gate lowers scaled |e| | -0.0004677 | -0.0063 | 0.006 | 0.419 | 1.0 | 1.0 | not supported |  |
| H6-T4 | F2 | T4: online time-adaptive gate lowers scaled |e| | 0.0065 | -0.0007163 | 0.0142 | 0.9881 | 1.0 | 1.0 | not supported |  |
| H6-T5 | F2 | T5: online time-adaptive gate lowers scaled |e| | -0.5098 | -0.6594 | -0.3692 | 4.635e-24 | 7.416e-23 | 7.019e-23 | supported |  |
| H7-T1 | F2 | T1: gated model loses less than fusion when RE and weather go missing | 0.0122 | 0.0055 | 0.019 | 1.0 | 1.0 | 1.0 | not supported |  |
| H7-T2 | F2 | T2: gated model loses less than fusion when RE and weather go missing | 0.018 | 0.0112 | 0.0252 | 1.0 | 1.0 | 1.0 | not supported |  |
| H8-T3 | F2 | T3: carbon block improves pinball loss | -0.0011 | -0.0019 | -0.0001782 | 0.0035 | 0.0353 | 0.0268 | supported |  |
| H8-T1 | F2 | T1: carbon block does not worsen pinball (margin 1 %) | -0.0004634 | -0.0017 | 0.0011 |  |  |  | supported | margin 0.0029 |
| H8-T2 | F2 | T2: carbon block does not worsen pinball (margin 1 %) | -0.0061 | -0.0088 | -0.0039 |  |  |  | supported | margin 0.0029 |
| H9 | F3 | assessment A ranks consequences better than M0 within M0 deciles | 0.02 | -0.0053 | 0.0447 | 0.0498 | 0.0997 | 0.3354 | not supported |  |
| H10 | F3 | forecast information beats the shuffled placebo (95th percentile) | -0.0058 |  | 0.0199 | 0.9608 | 0.9608 | 1.0 | not supported |  |
| H11a | F4 | A4s has lower tail-day system regret than A2 | 0.0555 | 0.0205 | 0.0941 | 0.998 | 1.0 | 1.0 | not supported |  |
| H11c | F4 | A4s has lower tail-day system regret than P | -1.8494 | -3.0458 | -0.4736 | 0.0035 | 0.0175 | 0.0268 | supported |  |
| H11d | F4 | A5 has lower tail-day system regret than A2 | 1.0573 | 0.2873 | 1.727 | 0.9975 | 1.0 | 1.0 | not supported |  |
| H11e | F4 | A5 has lower tail-day system regret than P | -0.875 | -1.661 | -0.0228 | 0.02 | 0.06 | 0.1397 | not supported |  |
| H11f | F4 | PPO (A1) has lower mean system regret than A2 | -0.2861 | -0.5266 | -0.0663 | 0.0019 | 0.0113 | 0.0155 | supported |  |
| H11g | F4 | PPO (A1) has lower mean system regret than P | -0.2759 | -0.5688 | -0.0109 | 0.013 | 0.0522 | 0.0948 | not supported |  |
| H12 | F4 | no decision arm breaks the hard limits | 0.0 |  |  |  |  |  | supported | largest share of State-days outside the limits over all arms |
| H13-OTG vs fixed | F5 | PART sign agreement > 1/2 (OTG vs fixed) | 0.5333 |  |  | 0.5 | 1.0 | 1.0 | not supported | 8/15 cells |
| H13-instance vs fixed | F5 | PART sign agreement > 1/2 (instance vs fixed) | 0.4667 |  |  | 0.6964 | 1.0 | 1.0 | not supported | 7/15 cells |
| H13-regime vs fixed | F5 | PART sign agreement > 1/2 (regime vs fixed) | 0.2667 |  |  | 0.9824 | 1.0 | 1.0 | not supported | 4/15 cells |
