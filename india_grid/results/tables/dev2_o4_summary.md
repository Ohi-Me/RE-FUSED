# O4 round 2 — deviation assessment

Forecast source: `ens_stack` (T2, H = 1, rolling conformal C1, 365-day window). Fitted on validation (n = 7,493); development test (n = 7,431) reported only.

Two-fold week cross-validated within-decile tau on validation: O4a 0.0024, O4b 0.0167, O4c 0.0425. Chosen: **O4c** with {'depth': 3, 'k': 2.0}.

| test | outcome | tau(A) | tau(comparator) | difference | 95% CI / placebo 95th pct | p (one-sided) |
|---|---|---|---|---|---|---|
| A vs M0 | Y | 0.0118 | -0.0082 | +0.0200 | [-0.0053, +0.0447] | 0.05 |
| A vs M0 | Y1 | 0.0154 | -0.0114 | +0.0269 | [+0.0042, +0.0472] | 0.01 |
| A vs M0 | Y2 | -0.0108 | -0.0021 | -0.0088 | [-0.0305, +0.0194] | 0.75 |
| A vs M0 | Y3 | -0.0263 | -0.0052 | -0.0212 | [-0.0466, +0.0052] | 0.94 |
| forecast information gain (A vs A without U, F) | Y | 0.0118 | 0.0176 | -0.0058 | [-0.0180, +0.0057] | 0.82 |
| forecast information vs shuffled placebo | Y | nan | nan | -0.0058 | +0.0199 | 0.96 |