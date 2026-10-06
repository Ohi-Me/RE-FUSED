# X0 - RE-FUSED-6 claims at the correct inferential unit

| claim | unit | units | supporting | mean diff | p (sign test, unit level) | p (RE-FUSED-6 run level) | note |
|---|---|---|---|---|---|---|---|
| M1 per-series > per-instance (India) | config (baseline x h) | 15 | 15 | +0.0578 | 3.1e-05 | 2.3e-77 | configs share one dataset and test period |
| M1 per-series > per-instance (India) | baseline | 4 | 4 | +0.0579 | 0.062 |  | 4 baselines on the same series |
| M1 per-series > shrunk (India 2024-25) | config | 15 | 14 | +0.0129 | 0.00049 | 4.7e-34 | M10 later showed this ordering is period-specific |
| M8 fine < best coarse | config (dataset x h) | 12 | 11 | -0.0301 | 0.0032 |  |  |
| M8 fine < best coarse | dataset (mean over h) | 6 | 5 | -0.0301 | 0.11 |  |  |
| M10 instance < best coarse | config (origin x baseline x h) | 24 | 19 | -0.0127 | 0.0033 |  |  |
| M10 instance < best coarse | origin (test year) | 3 | 3 | -0.0127 | 0.12 |  | 3 test years of one dataset |
| M9_ per-series > neural_gate_refit | config | 10 | 8 | +0.0054 | 0.055 | 3.0e-05 |  |
| M9_ per-series > neural_gate_refit | dataset | 3 | 2 | +0.0075 | 0.5 |  |  |
| M9b per-series > neural_gate_tuned | config | 10 | 7 | +0.0046 | 0.17 | 1.2e-03 |  |
| M9b per-series > neural_gate_tuned | dataset | 3 | 2 | +0.0058 | 0.5 |  |  |
