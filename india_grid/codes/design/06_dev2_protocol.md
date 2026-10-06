# RE-FUSED development round 2 (dev2) — protocol

Written on 17 Sep 2026, before any dev2 result. It adds to the earlier protocols (03, 04, 05) and does not replace
them. Everything here runs only in PBS GPU jobs on the NIT Jalandhar H100 cluster.

## 0. Why a second round

The first development run (17 Sep 2026) left clear gaps: the T5 price model picked on the validation year did badly on
the development test, the uncertainty term in O4 got zero weight, the O5 risk coupling switched itself off, the gated
model lost more than fusion when sources were missing, and the price-of-adaptivity test agreed in sign in only 19 of
45 cells. Round 2 tries to close these gaps and to get the most accurate forecaster we can build from the same data.

## 1. Ground rules

1. **Choices use train and validation data only.** Train = up to 31 Mar 2023, validation = FY2023-24. Every model
   setting, ensemble weight, calibration window, assessment weight and scheduling parameter is picked on these years.
2. **The development test (FY2024-25) is only for reporting.** We have already seen it once, so it is not a clean test
   any more. It is used to report dev2 results next to round 1, never to pick anything.
3. **The reserved period (1 Apr 2025 – 31 Aug 2026) stays closed** until the pre-registration is frozen. That is where
   the claims will be tested.
4. **Everything listed here is run and reported**, including what does not work. Anything added later gets a dated
   note in section 9 with the reason, and is marked as added after seeing results.
5. **Compute only on the H100** (PBS GPU jobs on workq). A short smoke job runs first, then the full job.
6. Main accuracy measure: MASE (scaled by the train-period seasonal naive). Also RMSSE, pinball loss and 80 % / 90 %
   coverage. Tests: Diebold–Mariano with HLN correction and 7-day block bootstrap, as in round 1.

## 2. N2/N3 — more base models (O2)

Same targets (T1–T5), same issue time, same publication lags and the same train / early-stop / validation / test
split as round 1. Five seeds for every trained model.

| Name | What it is | Settings picked on validation (seed 0) |
|---|---|---|
| `xgb` | XGBoost multi-quantile trees on the GPU, same features as LightGBM | max depth 6 or 8 |
| `nhits` | N-HiTS with past covariates | input window 28 or 56 days |
| `tide` | TiDE with past and calendar covariates | input window 28 or 56 days |
| `bitcn` | BiTCN (bidirectional temporal convolution) with past and calendar covariates | input window 28 or 56 days |
| `nbeatsx` | N-BEATSx with past and calendar covariates | input window 28 or 56 days |
| `chronos2` | Chronos-2, zero-shot, with the same past covariates and the calendar as known future covariates | context 512 days; cross-learning on or off |
| `chronos_base` | Chronos-Bolt-Base, zero-shot, target only | none |

A pretrained model whose weights cannot be loaded on the cluster is recorded as not available and left out.

## 3. N3 — combining models (the "fused" forecaster)

Members are all learned models from round 1 and section 2, plus the LA-MCAG model (`mcag_instance`). Quantiles are
combined first and calibrated afterwards. For members without q05/q95 (Chronos-Bolt) the ends are extended linearly
from q10/q25 and q75/q90.

| Name | Rule |
|---|---|
| `ens_top` | Equal-weight average of the k best members by validation MASE. k (2–6) is picked by two-fold ISO-week cross-validation inside the validation year. |
| `ens_stack` | One weight per member for each target × horizon (weights ≥ 0, summing to 1), fitted to minimise pinball loss on the validation year, pulled towards equal weights. The pull strength (0, 0.1, 1) is picked by the same two-fold week split. |
| `ens_online` | **Lag-aware online weights.** Each day the weights are updated from losses that are already published (target date ≤ issue day − publication lag). Exponentiated-gradient update with learning rate η ∈ {0.5, 1, 2, 4} and forgetting factor ∈ {1, 0.99, 0.97}, pooled over series or by region. The run starts at the beginning of the validation year with equal weights; η, forgetting and pooling are picked on the validation year, and the run carries on through the development test with those settings frozen. |

Calibration of the combined quantiles: rolling conformal at level C1 with a window of 90, 180 or 365 days, picked on
validation pinball loss.

**Final forecaster per target** = the candidate (best single member, `ens_top`, `ens_stack`, `ens_online`) with the
lowest validation MASE; ties within 0.5 % go to the lower validation pinball loss.

Tests on the development test (reported, not used to choose): final forecaster vs seasonal naive (as H1), final
forecaster vs the round-1 selected model, and coverage within ±5 points (H2).

## 4. N7, N9 — LA-MCAG changes (O3)

* **N7, staleness-matched source dropout.** In round 1, a dropped block always got a staleness of 30 days during
  training, but in the source-loss test the staleness grows up to a year, which the model had never seen. Round 2
  draws the staleness of a dropped block from a log-uniform range of 1–365 days. Variants `mcag_instance_sd` and
  `fusion_fixed_sd` (matched capacity), T1–T5, 5 seeds. H7 is re-run with these two (T1, T2). Round-1 and sd variants
  are compared on validation MASE; the better one per target is the O3 member used in section 3 from round 2 on.
* **N9, decision form of PART.** Refine only when the lower end of an 80 % interval for G is above zero (interval from
  the spread of the block estimates). Scored by the realised development-test loss of the configuration PART picks,
  compared with "always refine" and "never refine". Sign agreement is also reported, both over all cells and over cells
  whose realised gain is clearly different from zero (block-bootstrap interval excludes zero).

## 5. N10 — deviation assessment (O4)

U comes from the final T2 forecaster (section 3). Three forms, all fitted on the validation year:

| Name | Form |
|---|---|
| `O4a` | Round-1 form A = \|Δ\|·[λ + κ(w·(R, S, C, U))] with the new U. |
| `O4b` | **Split deviation.** With F the forecast median issued the day before, the deviation X − S splits into a part known at scheduling time (F − S) and a surprise part (X − F). A = λ·[a·\|F − S\| + (1 − a)·\|X − F\|]·[1 + κ(w·(R, S, C, U))]; a on a 0.1 grid. |
| `O4c` | **Learned multiplier.** A = M₀·exp(κ·s), where s is a GPU gradient-boosted ranking score with non-decreasing effect of R, S, C and U, trained to order the consequence outcome within M₀ deciles; also uses the two split shares from O4b. |

The form with the best two-fold week cross-validated within-decile Kendall τ on the validation year is chosen.
H9 and H10 are re-run for it. The placebos permute U and F within State (50 draws each) and refit.

## 6. N11, N12 — scheduling (O5)

Forecast source: the final T2 forecaster (section 3). Round-1 arms are re-run with it. New arms:

* `A4s` — **smooth coupling.** β = sigmoid(b₀ + b_U(U − 0.5) + b_S(S − 0.5)) per State-day, b₀ ∈ {−4, −2, −1, 0, 1},
  b_U, b_S ∈ {0, 1, 2, 4}, tuned on validation J.
* `A5` — **system-tail scheduling with joint scenarios.** The 34 State schedules for a day are chosen together to
  minimise (1 − β)·mean + β·CVaR₉₀ of the *system* regret. Joint scenarios keep the dependence between States: each
  scenario takes the forecast-quantile ranks that all States had together on one of the 180 most recent published
  days (empirical copula) and maps them to today's forecast quantiles. β ∈ {0, 0.25, 0.5, 0.75, 1} tuned on validation J.
* `A5c` — A5 with β for the day coupled to the mean U and mean S across States (same smooth form as A4s).
* Chain check (N12, development form): A2 and A5 with the final forecaster vs the same arms with the best single model
  (round-1 T2 selection). The reserved-period chain run waits for the frozen pre-registration.

Tests as in round 1 (H11, H12), adding A5 vs A2, A5 vs P and A5 vs H_pub on the mean and on tail days.

## 7. Output layout

```
results/dev2/o2/preds       new base models and ensembles (same keys as round 1)
results/dev2/o2/eval        point, prob, tests, weights, selection_dev2.json
results/dev2/o3/preds, eval staleness-matched variants, L1 and PART decision results
results/dev2/o4             weights, tests, placebo, rows
results/dev2/o5             decisions, tuning, metrics, tests
results/tables/dev2_*.md    readable summaries
NOVELTY_MAP.md              N1–N12: aim, hypothesis, code, result files, round-1 and round-2 status
```

## 8. What is claimed from round 2

Round-2 numbers on the development test are development evidence only. Claims are made on the reserved period after
freezing the pre-registration (updated with the round-2 choices).

## 9. Changes after this protocol

| Date | Change | Reason | Seen results first? |
|---|---|---|---|
| 17 Sep 2026 | `tsmixerx` replaced by `bitcn` | TSMixer-x in neuralforecast 3.2.2 is a multivariate model (one joint forecast for all series); the other models here forecast each series, so it did not fit the same setup. Changed before any dev2 run. | No |
| 17 Sep 2026 | Final-forecaster choice uses cross-fitted validation MASE | Picking the best single member, the top-k set or the stack weights on the whole validation year and then scoring them on the same year flatters them against the online ensemble, which is scored as it runs. So each is fitted on one half of the ISO weeks and scored on the other half (and the other way round); the online ensemble keeps its day-by-day score. Written before the ensemble code ran. | No |
