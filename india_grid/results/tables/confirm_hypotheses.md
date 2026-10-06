# Pre-registered hypotheses — confirmatory evaluation 1 Apr 2025 – 31 Aug 2026

Holm within family, Benjamini–Yekutieli across all primary tests. A hypothesis is supported when its Holm-adjusted p < 0.05 and its extra rule holds (CI side, all seeds, coverage band, zero violations).

Supported: 35 of 52.

| id | family | what | estimate | lo | hi | p | p_holm | p_by | decision | note |
|---|---|---|---|---|---|---|---|---|---|---|
| H1a | F1 | T1: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.4448 | -0.5 | -0.3955 | 1.769e-87 | 1.769e-86 | 5.672e-86 | supported | seeds p<0.05: 5/5 |
| H1f-T1 | F1 | T1: final forecaster beats the round-1 selected model (lgbm) | -0.0346 | -0.0429 | -0.0258 | 3.802e-21 | 2.662e-20 | 7.316e-20 | supported |  |
| H2-T1 | F1 | T1: 80 % / 90 % coverage within ±5 points | 0.7891 | 0.7891 | 0.8931 |  |  |  | supported | cov80 0.789, cov90 0.893 |
| H3-T1 | F1 | T1: publication-latency cost > 0 (LightGBM, % MASE) | 16.2044 | -0.1426 | -0.1174 | 3.27e-107 | 4.251e-106 | 2.097e-105 | supported | interval: daily scaled-error difference, instant minus published |
| H1b | F1 | T2: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.4168 | -0.468 | -0.3721 | 6.562e-93 | 7.218e-92 | 2.525e-91 | supported | seeds p<0.05: 5/5 |
| H1f-T2 | F1 | T2: final forecaster beats the round-1 selected model (lgbm) | -0.0318 | -0.0399 | -0.023 | 9.838e-20 | 5.903e-19 | 1.721e-18 | supported |  |
| H2-T2 | F1 | T2: 80 % / 90 % coverage within ±5 points | 0.792 | 0.792 | 0.8927 |  |  |  | supported | cov80 0.792, cov90 0.893 |
| H3-T2 | F1 | T2: publication-latency cost > 0 (LightGBM, % MASE) | 15.1392 | -0.1369 | -0.1142 | 7.476e-115 | 1.047e-113 | 7.191e-113 | supported | interval: daily scaled-error difference, instant minus published |
| H1c | F1 | T3: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.2701 | -0.3076 | -0.2348 | 1.419e-62 | 1.277e-61 | 3.899e-61 | supported | seeds p<0.05: 5/5 |
| H1f-T3 | F1 | T3: final forecaster beats the round-1 selected model (lgbm) | -0.0029 | -0.0052 | -0.0009436 | 0.0009116 | 0.0009116 | 0.008 | supported |  |
| H2-T3 | F1 | T3: 80 % / 90 % coverage within ±5 points | 0.8016 | 0.8016 | 0.8993 |  |  |  | supported | cov80 0.802, cov90 0.899 |
| H3-T3 | F1 | T3: publication-latency cost > 0 (LightGBM, % MASE) | 30.0267 | -0.2165 | -0.183 | 3.216e-115 | 4.823e-114 | 6.187e-113 | supported | interval: daily scaled-error difference, instant minus published |
| H1d | F1 | T4: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.508 | -0.7026 | -0.3573 | 2.95e-07 | 5.9e-07 | 3.547e-06 | supported | seeds p<0.05: 5/5 |
| H1f-T4 | F1 | T4: final forecaster beats the round-1 selected model (chronos) | -0.0574 | -0.0838 | -0.0358 | 4.902e-10 | 1.471e-09 | 6.736e-09 | supported |  |
| H2-T4 | F1 | T4: 80 % / 90 % coverage within ±5 points | 0.7782 | 0.7782 | 0.8857 |  |  |  | supported | cov80 0.778, cov90 0.886 |
| H3-T4 | F1 | T4: publication-latency cost > 0 (LightGBM, % MASE) | 15.7772 | -0.2249 | -0.1331 | 2.357e-18 | 1.178e-17 | 3.779e-17 | supported | interval: daily scaled-error difference, instant minus published |
| H1e | F1 | T5: final forecaster beats seasonal naive (scaled |e|, H pooled) | -0.286 | -0.3925 | -0.1811 | 1.283e-12 | 5.132e-12 | 1.899e-11 | supported | seeds p<0.05: 5/5 |
| H1f-T5 | F1 | T5: final forecaster beats the round-1 selected model (bilstm) | -1.8576 | -2.0822 | -1.6347 | 9.114e-97 | 1.094e-95 | 4.384e-95 | supported |  |
| H2-T5 | F1 | T5: 80 % / 90 % coverage within ±5 points | 0.7864 | 0.7864 | 0.892 |  |  |  | supported | cov80 0.786, cov90 0.892 |
| H3-T5 | F1 | T5: publication-latency cost > 0 (LightGBM, % MASE) | 137.6633 | -0.8024 | -0.5388 | 1.689e-44 | 1.351e-43 | 4.061e-43 | supported | interval: daily scaled-error difference, instant minus published |
| H4-T1 | F2 | T1: gated (sd) beats matched fusion (sd), pinball | -0.0009314 | -0.0023 | 0.0002906 | 0.0384 | 0.3843 | 0.2465 | not supported |  |
| H4-T2 | F2 | T2: gated (sd) beats matched fusion (sd), pinball | -0.0014 | -0.0026 | -0.0003035 | 0.0014 | 0.0181 | 0.0117 | supported |  |
| H4-T3 | F2 | T3: gated (sd) beats matched fusion (sd), pinball | -0.0025 | -0.0046 | -9.187e-05 | 0.0027 | 0.0329 | 0.0211 | supported |  |
| H4-T4 | F2 | T4: gated (sd) beats matched fusion (sd), pinball | -0.0008901 | -0.007 | 0.0067 | 0.3535 | 1.0 | 1.0 | not supported |  |
| H4-T5 | F2 | T5: gated (sd) beats matched fusion (sd), pinball | -0.0478 | -0.0786 | -0.02 | 1.413e-07 | 2.119e-06 | 1.812e-06 | supported |  |
| H5-T1 | F2 | T1: real context beats null and shuffled context, pinball | -0.0021 |  | 0.0021 | 0.0953 | 0.7625 | 0.5731 | not supported | larger of the two p values |
| H5-T2 | F2 | T2: real context beats null and shuffled context, pinball | -0.0051 |  | -0.0017 | 3.878e-05 | 0.000543 | 0.0004146 | supported | larger of the two p values |
| H5-T5 | F2 | T5: real context beats null and shuffled context, pinball | -0.0043 |  | 0.0285 | 0.3581 | 1.0 | 1.0 | not supported | larger of the two p values |
| H6-T1 | F2 | T1: online time-adaptive gate lowers scaled |e| | -0.0027 | -0.0068 | 0.002 | 0.0569 | 0.5121 | 0.3531 | not supported |  |
| H6-T2 | F2 | T2: online time-adaptive gate lowers scaled |e| | 0.0045 | -0.0002911 | 0.01 | 0.9905 | 1.0 | 1.0 | not supported |  |
| H6-T3 | F2 | T3: online time-adaptive gate lowers scaled |e| | -0.0021 | -0.0062 | 0.0023 | 0.1145 | 0.8015 | 0.6676 | not supported |  |
| H6-T4 | F2 | T4: online time-adaptive gate lowers scaled |e| | 0.0414 | 0.0007025 | 0.0949 | 0.9922 | 1.0 | 1.0 | not supported |  |
| H6-T5 | F2 | T5: online time-adaptive gate lowers scaled |e| | -0.6438 | -0.8178 | -0.441 | 3.198e-25 | 5.117e-24 | 6.836e-24 | supported |  |
| H7-T1 | F2 | T1: gated model loses less than fusion when RE and weather go missing | -0.0027 | -0.0055 | -0.0001537 | 0.005 | 0.0553 | 0.037 | not supported |  |
| H7-T2 | F2 | T2: gated model loses less than fusion when RE and weather go missing | 0.0019 | -0.0005351 | 0.0045 | 0.978 | 1.0 | 1.0 | not supported |  |
| H8-T3 | F2 | T3: carbon block improves pinball loss | 0.0008553 | -7.182e-05 | 0.0019 | 0.9773 | 1.0 | 1.0 | not supported |  |
| H8-T1 | F2 | T1: carbon block does not worsen pinball (margin 1 %) | 0.0008771 | -0.0004225 | 0.0022 |  |  |  | supported | margin 0.0028 |
| H8-T2 | F2 | T2: carbon block does not worsen pinball (margin 1 %) | 0.0006304 | -0.0002203 | 0.0014 |  |  |  | supported | margin 0.0029 |
| H9 | F3 | assessment A ranks consequences better than M0 within M0 deciles | 0.0108 | -0.0154 | 0.0351 | 0.2292 | 0.2292 | 1.0 | not supported |  |
| H10 | F3 | forecast information beats the shuffled placebo (95th percentile) | 0.0287 |  | 0.0198 | 0.0196 | 0.0392 | 0.1347 | supported |  |
| H11a | F4 | A4s has lower tail-day system regret than A2 | 0.0306 | -0.0233 | 0.0732 | 0.8961 | 1.0 | 1.0 | not supported |  |
| H11b | F4 | A4s has lower tail-day system regret than H_pub | -7.8808 | -10.5877 | -5.6943 | 0.0004998 | 0.0035 | 0.0046 | supported |  |
| H11c | F4 | A4s has lower tail-day system regret than P | -2.7094 | -4.3857 | -1.4029 | 0.0004998 | 0.0035 | 0.0046 | supported |  |
| H11d | F4 | A5 has lower tail-day system regret than A2 | 1.3183 | 0.531 | 2.278 | 0.9975 | 1.0 | 1.0 | not supported |  |
| H11e | F4 | A5 has lower tail-day system regret than P | -1.6654 | -2.6855 | -0.9652 | 0.0015 | 0.0075 | 0.012 | supported |  |
| H11f | F4 | PPO (A1) has lower mean system regret than A2 | -0.1621 | -0.3447 | 0.0674 | 0.0374 | 0.1123 | 0.2465 | not supported |  |
| H11g | F4 | PPO (A1) has lower mean system regret than P | -0.2814 | -0.5343 | -0.0621 | 0.0052 | 0.0208 | 0.037 | supported |  |
| H12 | F4 | no decision arm breaks the hard limits | 0.0 |  |  |  |  |  | supported | largest share of State-days outside the limits over all arms |
| H14 | F4 | the chain with the final forecaster lowers mean system regret (arm A2) vs the round-1 forecaster | -0.2914 | -0.4522 | -0.1457 | 6.554e-06 | 5.243e-05 | 7.417e-05 | supported |  |
| H13-OTG vs fixed | F5 | PART sign agreement > 1/2 (OTG vs fixed) | 0.6667 |  |  | 0.1509 | 0.3018 | 0.8538 | not supported | 10/15 cells |
| H13-instance vs fixed | F5 | PART sign agreement > 1/2 (instance vs fixed) | 0.9333 |  |  | 0.0004883 | 0.0015 | 0.0046 | supported | 14/15 cells |
| H13-regime vs fixed | F5 | PART sign agreement > 1/2 (regime vs fixed) | 0.2667 |  |  | 0.9824 | 0.9824 | 1.0 | not supported | 4/15 cells |
