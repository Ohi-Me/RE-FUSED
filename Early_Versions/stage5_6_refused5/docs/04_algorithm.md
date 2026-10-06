# RE-FUSED-5 algorithm: Shrunk Gated Residual Correction (SGRC) with block-risk control

## Inputs
Series i, horizon h, baseline B (given, not learned), training window T_tr, validation window T_va
(strictly later), test window T_te (strictly later). Cells c(i,h) = (series, horizon).

## Algorithm
```
# Stage 1: corrector  (train split only)
r_hat  <- fit_regressor( features(s_t) -> R_{t+h} = Y_{t+h} - B_{t+h} ) on T_tr

# Stage 2: gate estimation (validation split only)
g_glob <- <R, r_hat> / <r_hat, r_hat>                      # one scalar (Diebold-Pauly analogue)
for each cell c:                                           # CRC-style stratified gate
    g_cell[c] <- <R_c, r_hat_c> / <r_hat_c, r_hat_c>   if n_c > n_min  else g_glob

# Stage 3: conditional gate + its own estimation variance   (validation split only)
num(x) <- fit_regressor( x -> R * r_hat )                  # E[R r_hat | x]
den(x) <- fit_regressor( x -> r_hat^2 )                    # E[r_hat^2 | x]
g_cond(x) <- clip( num(x) / den(x), 0, 1 )
split validation in halves A,B; refit num/den on each -> g_A, g_B
var_est   <- mean( (g_A - g_B)^2 ) / 4                     # variance of the conditional estimator
var_within<- max( var(g_cond) - var_est, 0 )               # genuine within-cell signal
lambda    <- var_within / (var_within + var_est)           # SHRINKAGE ON THE GATE ITSELF

# Stage 4: shrunk gate
g_shrunk(x) <- clip( lambda * g_cond(x) + (1 - lambda) * g_cell[c(x)], 0, 1 )

# Stage 5: block-risk control (deployment decision per cell)
for each cell c:
    partition validation into K time blocks (e.g. months)
    s_k <- 1 - MAE(B + g_shrunk r_hat) / MAE(B)   on block k
    LCB <- mean(s_k) - t_{1-alpha, K-1} * sd(s_k) / sqrt(K)
    deploy correction in cell c  iff  LCB > 0     # else fall back to the baseline (g := 0)

# Output
Y_hat = B + 1[LCB_c > 0] * g_shrunk(x) * r_hat(x)
```

## Why each stage exists (and the evidence that it must stay)
| Stage | Purpose | Evidence it earns its place |
|---|---|---|
| 1 corrector | recover predictable residual structure | India: +0.19 skill vs seasonal-naive at h=1 |
| 2 cell gate | shrink by cell-level residual R² | beats always-correct on every baseline (R1) |
| 3 conditional gate **alone** | track within-cell variation | **fails alone** (S2c: worse than static in 10/10 seeds) |
| 3+4 shrinkage on the gate | pay for conditioning only when it is identifiable | best on real data for all 3 stationary baselines (R1, R5) |
| 5 block-risk control | refuse correction where validation evidence is weak under temporal shift | turns −0.0087 into +0.0211 with 0/20 degraded states on the drifting baseline (R4) |

## Complexity
Corrector: one fit. Gate: 2 fits + 4 half-sample fits (variance estimate). No test-time search.
Inference is one extra regressor pass over the baseline; no iterative computation.

## What SGRC deliberately does NOT include
Attention, mixture-of-experts, RL, foundation models, carbon conditioning. Each was either shown to
add nothing here (context blocks: R2) or lies outside the tested claim. They are excluded by the
brief's own rule: a component must earn reproducible incremental value or be removed.
