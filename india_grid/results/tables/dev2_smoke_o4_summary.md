# O4 round 2 — deviation assessment

Forecast source: `lgbm` (T2, H = 1, rolling conformal C1, 180-day window). Fitted on validation (n = 3,000); development test (n = 3,000) reported only.

Two-fold week cross-validated within-decile tau on validation: O4a 0.0006, O4b 0.0184, O4c 0.0497. Chosen: **O4c** with {'depth': 2, 'k': 0.25}.

| test | outcome | tau(A) | tau(comparator) | difference | 95% CI / placebo 95th pct | p (one-sided) |
|---|---|---|---|---|---|---|
| A vs M0 | Y | -0.0134 | -0.0013 | -0.0121 | [-0.0366, +0.0181] | 0.8 |
| A vs M0 | Y1 | -0.0038 | 0.0106 | -0.0144 | [-0.0344, +0.0188] | 0.87 |
| A vs M0 | Y2 | -0.0604 | -0.0084 | -0.0520 | [-0.0797, -0.0305] | 1 |
| A vs M0 | Y3 | -0.0180 | -0.0111 | -0.0069 | [-0.0317, +0.0184] | 0.71 |
| forecast information gain (A vs A without U, F) | Y | -0.0134 | -0.0192 | +0.0058 | [-0.0105, +0.0199] | 0.22 |
| forecast information vs shuffled placebo | Y | nan | nan | +0.0058 | +0.0202 | 1 |