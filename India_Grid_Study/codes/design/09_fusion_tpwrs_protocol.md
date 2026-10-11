# Protocol: settlement-aware fusion scheduler (TPWRS follow-up), version 1

Status: FROZEN on 10 Oct 2026 after the development run (H100 job 34924, results/extra/fusion_tpwrs_dev.json);
the design, variants, features, seeds and tests below were not changed after that run. The only code change before
freezing added the projected persistence baseline, as in the confirmatory run. This file is hashed into `FUSION_TPWRS_SHA256.txt` when the design is frozen; the
script refuses the evaluation period (1 Apr 2025 - 31 Aug 2026) until the hash matches. Agreed evaluation plan
(authors, 10 Oct 2026): design and tune on development data only; freeze; one exploratory run on the already-opened
evaluation period, labelled as such; a prospective check on data published after the freeze is the confirmatory step.

## 1. Question and gap

that are constant within groups of one signal. A rule whose level varies continuously with several signals at once,
learned on the settlement cost itself, was not tested. The closest prior work learns a newsvendor order quantity from
features (Ban and Rudin 2019, Oper. Res. 67:90-108) or trains the forecaster for the decision (Stratigakos et al.
2022; Han et al. 2021). Here the calibrated forecaster is kept fixed and only the level at which to schedule is
learned, under the banded, asymmetric Indian settlement.

## 2. Model

For State n and day t, with calibrated forecast quantile function Q_{n,t} (99 levels, linear interpolation) and
features z_{n,t} known at decision time:

  tau_{n,t} = 0.01 + 0.98 * sigmoid(f_theta(z_{n,t})),     S_{n,t} = Q_{n,t}(tau_{n,t}) + c_{n,t} * b_theta(z_{n,t})

where c_{n,t} = (q90 - q10)/2 is the forecast scale and b_theta is an optional bias term (variant F). theta minimises
the mean settlement regret over State-days of the fitting period,

  min_theta  (1/N) sum_{n,t} l_k(S_{n,t}, X_{n,t}; p_t, lambda_t) + alpha * ||theta||^2

with l_k the settlement of design k (the paper's stylised model, or the closer daily DSM approximation of
`codes/scripts/extra/x06_dsm_regulation.py`), realised prices in the loss and prices as published before the decision
in the features. After training, S is projected onto the schedule limits like every other rule.

Features z (all available at t-1, publication delays respected): r_hat = lambda_hat / p_hat; band-relative risk
(forecast probability that |X - q50| > 0.1 |q50|); relative spread (q90 - q10)/|q50|; skew (q90 + q10 - 2 q50)/(q90 -
q10); uncertainty and stress percentiles U, Z; mean forecast error X - q50 over the 7 published days ending t-2,
divided by the scale; day-of-week and month (sine, cosine); area (one-hot).

## 3. Variants (ablations) and baselines

| Id | Level model | Features |
|---|---|---|
| F0 | constant (one learned level) | none; equals a fixed quantile rule fitted on the same data |
| F1 | logistic | r_hat |
| F2 | logistic | r_hat, band-relative risk (the two quantities the first-order condition names) |
| F3 | logistic | all features |
| F4 | MLP 2 x 32, ReLU | all features |
| F5 | F4 plus bias term b_theta | all features |

Baselines from the frozen pipeline: persistence (H_pub), forecast median P, A2, R0. Each variant is trained with 3
seeds; the last 20 % of the fitting days (in time order) are used only for early stopping.

## 4. Data and evaluation

development). Frozen run: fit on FY2024-25, evaluate on 1 Apr 2025 - 31 Aug 2026 (exploratory, period already opened).
Prospective: data published after the freeze, when available.

Primary comparison: F4 and F5 against F0 (does learned adaptivity beat a fixed level fitted the same way?) in mean
system regret per day, paired 7-day block bootstrap interval (2,000 resamples) and one-sided HLN p value. Secondary:
CVaR90, J, and each Fk against F0, P and A2. Holm over the five Fk-vs-F0 comparisons. Settlements: stylised and
dsm_nominal, reported separately. A variant "beats" F0 only if the Holm-adjusted p < 0.05 and the interval excludes 0.

## 5. What would count against the idea

the settlement cost, and is reported as such.
