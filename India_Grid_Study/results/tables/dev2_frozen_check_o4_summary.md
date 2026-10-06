# O4 round 2 — deviation assessment

Forecast source: `lgbm` (T2, H = 1, rolling conformal C1, 180-day window). Fitted on validation (n = 3,000); development test (n = 3,000) reported only.

Two-fold week cross-validated within-decile tau on validation: . Chosen: **O4c** with {'depth': 3, 'k': 2.0}.

| test | outcome | tau(A) | tau(comparator) | difference | 95% CI / placebo 95th pct | p (one-sided) |
|---|---|---|---|---|---|---|
| A vs M0 | Y | -0.0213 | -0.0013 | -0.0200 | [-0.0480, +0.0234] | 0.86 |
| A vs M0 | Y1 | 0.0005 | 0.0106 | -0.0101 | [-0.0325, +0.0327] | 0.71 |
| A vs M0 | Y2 | -0.0363 | -0.0084 | -0.0278 | [-0.0658, -0.0016] | 0.94 |
| A vs M0 | Y3 | -0.0350 | -0.0111 | -0.0239 | [-0.0632, +0.0157] | 0.89 |
| forecast information gain (A vs A without U, F) | Y | -0.0213 | -0.0132 | -0.0081 | [-0.0238, +0.0102] | 0.82 |
| forecast information vs shuffled placebo | Y | nan | nan | -0.0081 | +0.0175 | 1 |