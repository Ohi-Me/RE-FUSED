# CUEFO-8 pre-registration, version 2

Written on 18 Sep 2026, after development rounds 1 and 2 and before any row of the reserved period was loaded. The
reserved period has stayed closed behind the blinding guard (`codes/cuefo8/guard.py`): no script can read it until
this file is hashed into `PREREG_SHA256.txt` and the code is committed with the git tag `prereg-v1`. Version 1 (the
earlier draft) is replaced by this version because round 2 changed the forecaster, the assessment form and the
scheduling arms; both rounds are development work and are reported as such.

## 1. Data and periods

* Panel: `data/processed/cuefo8_state_day.parquet`, content hash
  `c6ef9c8ebaba692cc0952025f87ea311e09b54ffae9e77c1c04a1b0ffaec4bbd` (rebuilt from the raw files on the H100 with the
  same hash). Only official Government of India sources.
* The confirmatory phase is the development pipeline with every date boundary moved forward by one year:

  | Role | Target dates |
  |---|---|
  | fitting (last 180 days only for early stopping) | 1 Apr 2018 – 31 Mar 2024 |
  | calibration year (choices that are re-fitted, conformal history, online-weight warm-up, PPO training) | 1 Apr 2024 – 31 Mar 2025 |
  | **confirmatory evaluation** | **1 Apr 2025 – 31 Aug 2026** |

* The State RE target (T4) ends on 18 Nov 2025, where the CEA source ends; IMD weather ends on 31 Dec 2025 and is
  treated as a missing source after that date, exactly as the models already handle missing sources.
* Targets, horizons (1–3 days), issue time, measured publication lags, features, model code, seeds (0–4) and training
  budgets are those of the development protocols (03 O2 v1.3, 04 O4/O5, 05 O3 v1.2, 06 round 2).

## 2. Frozen choices

Nothing below is chosen again. Where a procedure re-fits numbers on the calibration year (for example ensemble
weights), the procedure and all its settings are frozen here and only the fitted numbers change.

### 2.1 Round 1 (used as members, comparators and secondary tests)

| Item | Frozen value | Source |
|---|---|---|
| O2 selected model per target | T1 lgbm, T2 lgbm, T3 lgbm, T4 chronos, T5 bilstm | `results/dev/o2/eval/selection.json` |
| O2 calibration level of the selected model | T1 C0, T2 C0, T3 C1, T4 C2, T5 C2 | same |
| O4 round-1 weights (R, S, C, U) and κ | (0, 1, 0, 0), κ = 0.25; without U the same | `results/dev/o4/weights.json` |
| O5 round-1 parameters | A2 β = 0, ρ = 0.5; A3 ρ₀ = 0.25; A4 β₀ = β_U = β_stress = 0; A2r β = (0, 0) | `results/dev/o5/frozen_params.json` |

### 2.2 Round 2 (the main confirmatory configuration)

| Item | Frozen value | Source |
|---|---|---|
| Final forecaster per target | T1 ens_stack, T2 ens_stack, T3 ens_top, T4 chronos2, T5 ens_top | `results/dev2/o2/eval/selection_dev2.json` |
| Conformal calibration of the final forecaster | rolling, level C1, 365-day window, all targets | same |
| Ensemble members | the 13 members listed in each target's ensemble file (round-1 models, round-2 models, one LA-MCAG variant) | `results/dev2/o2/logs/ensemble_T*.json` |
| LA-MCAG member in the ensemble | T1 mcag_instance_sd, T2 mcag_instance, T3 mcag_instance, T4 mcag_instance, T5 mcag_instance_sd | same |
| ens_top | members ranked by calibration-year MASE; k = 4, 2, 2, 2, 6 for T1–T5 | same |
| ens_stack | weights ≥ 0 summing to 1 per horizon, fitted on the calibration year; pull to equal weights λ = 0 (T1–T4), 0.1 (T5) | same |
| ens_online (reported, not final for any target) | η, forgetting, pooling = T1 (2, 0.99, all), T2 (4, 0.99, all), T3 (2, 0.99, all), T4 (2, 0.99, region), T5 (0.5, 1, all) | same |
| XGBoost depth | 6 for every target and horizon | `results/dev2/o2/logs/xgb_T*.json` |
| Neural input windows (N-HiTS, TiDE, BiTCN, N-BEATSx) | T1 56/56/56/56; T2 28/28/56/28; T3 28/28/28/28; T4 28/28/28/28; T5 28/56/56/28 days | `results/dev2/o2/logs/nf_*_T*.json` |
| Chronos-2 cross-learning | off for T1–T4, on for T5 | `results/dev2/o2/logs/chronos2_T*.json` |
| O4 assessment form | O4c learned monotone multiplier, tree depth 3, κ = 2; re-fitted on the calibration year with these settings; the no-forecast comparator and the placebos use the same settings | `results/dev2/o4/choice.json` |
| O5 forecast source | T2 final forecaster (ens_stack), H = 1, rolling C1, 365 days | `results/dev2/o5/params.json` |
| O5 round-2 parameters (not re-tuned) | A2 β = 0, ρ = 0.5; A4s b₀ = −4, b_U = 0, b_S = 4, ρ = 0.5; A5 β = 0; A5c b₀ = −4, b_U = 1, b_S = 2 | same |
| O5 PPO (A1) | 200,000 steps, seeds 0–4 trained on the calibration year; the seed with the lowest calibration-year J is the one tested | protocol |

SHA-256 of the frozen choice files and of the scoring script:

```
71d2ec7992376bcebb2ca6cb322845bc8dc65c8c93ce769a856dd65a71ca30ca  results/dev/o2/eval/selection.json
f93c9f25744f7f492a6d169d94a19522c810c0414434cf44a6e40615448da372  results/dev/o4/weights.json
5251d233e4550391fde92c9f2fb110e8032b8d6acf1335ad4ef2863521ed1e34  results/dev/o5/frozen_params.json
e075eb5a9edf793ee65d80ca1cdc2eb70929495d0bdf129068ed700bc8b310e5  results/dev2/o2/eval/selection_dev2.json
8c2f35bbe2b210a21d0b4584fe601b14d2e8432f0850390ea88f6b5a33dcd9eb  results/dev2/o2/logs/ensemble_T1.json
fed3362138a9a5866762dd44b9ecbb97e6fe9d65c2a91743cde43277be757764  results/dev2/o2/logs/ensemble_T2.json
424144916fc4aa1e071e2ec3d243767a7d9b1346ef167fd99267feafcb01733d  results/dev2/o2/logs/ensemble_T3.json
1bfac3682aa224f1004de18368f4e338b482aa75387ebb0279d913d96b20cb25  results/dev2/o2/logs/ensemble_T4.json
d8f956602326027b7d71e4610ce4b9ffe79de88fc6a1db3084fdfafa8637f4e9  results/dev2/o2/logs/ensemble_T5.json
51ce681b5bc831871fa53191fad94e7e90ed86d558d1d356352ace7bd5324718  results/dev2/o2/logs/xgb_T1.json
6520117556a457cf5e0ce7fdc98684c1cc95a5a6cbc22a841d5d71a38e5ab60c  results/dev2/o2/logs/xgb_T2.json
3fbe8daceb88669341af71d7ba2d94f9b99802756083db2b9e336a173edc6c7e  results/dev2/o2/logs/xgb_T3.json
ef1a6070cff2142bd9b67a4df1ca716892c67617c3b945e089020b9ac27ae67f  results/dev2/o2/logs/xgb_T4.json
8a0635aee1946d995c00a1ca25195b6e4c0221370fd2ec40e0b30bd4b12a022c  results/dev2/o2/logs/xgb_T5.json
621e9da2f40e4e4beb1e5f32e884d1a88f5318836c1060a7a5422e5ef4a1825b  results/dev2/o2/logs/nf_bitcn_T1.json
8a19146d91277de4ddb6de259c135c04f00d66be1b47b12dce3e285eedf16ac9  results/dev2/o2/logs/nf_bitcn_T2.json
140021d358106c93fbec5911219753fc240ea3c434278db1e54c2a92521408ee  results/dev2/o2/logs/nf_bitcn_T3.json
eb9dad9d604278f277f2fc6a46c5375e87fd20fc19a1e6d10c456d84b6947afc  results/dev2/o2/logs/nf_bitcn_T4.json
fb5da26d32c416f1d574c21c337eee774e4be511a836446d7a86ce56eba3e580  results/dev2/o2/logs/nf_bitcn_T5.json
c9ef92c05c4b4c3a94f6f0e1fc45cdfd5a00eb4b4d803bab3beb08032a866f69  results/dev2/o2/logs/nf_nbeatsx_T1.json
d81280a6c1d3088acfc841583beaea8de17c5b6c5454eebc1a401448dcd257c7  results/dev2/o2/logs/nf_nbeatsx_T2.json
604a52ea457b08346ad28f3195af053b2a71714903a11f1590e1620e9fad7e5d  results/dev2/o2/logs/nf_nbeatsx_T3.json
3ffaefdbef606565b0ca87e1177360676d1266da7349121ecfeb1934f3277c03  results/dev2/o2/logs/nf_nbeatsx_T4.json
ef243f91de6a0cccc59c6f6cc5882ae3ed8212765a0e618963aa3654c7da361b  results/dev2/o2/logs/nf_nbeatsx_T5.json
259b3e2369091ee85e53d54c0e43d43fef660fd7f5bcac73a7458d1b24eb1eee  results/dev2/o2/logs/nf_nhits_T1.json
2a032af2ef395a1da3d7195869d4c085543ebc42cfb83f5dbd478da116050958  results/dev2/o2/logs/nf_nhits_T2.json
03649b1ec616c92c3edfce9ece931b3e93a04fa76dfc0617ff36e67da261b953  results/dev2/o2/logs/nf_nhits_T3.json
3062a5ca5dcda95815a1f6ab47c8d463a86406ab687a76c96aee09a599f667d7  results/dev2/o2/logs/nf_nhits_T4.json
9fd6ae64564cc1d02695614dda635423bd0253b08a2f7c32854e48ee6fee6868  results/dev2/o2/logs/nf_nhits_T5.json
88f11a833805d97277c0560bc5ae845f9870baaeb7f9a75a0d9629ed23881ee3  results/dev2/o2/logs/nf_tide_T1.json
f483b6dbee6fdcedd2f1d5c9b4e7d4522bb57699a4e68cc8f7b8ba3890b642d0  results/dev2/o2/logs/nf_tide_T2.json
9f8ee8edd1fd4d680c1a346d04ab7cd59e1f9a322236a7855006388de511214b  results/dev2/o2/logs/nf_tide_T3.json
f83ce2c0d9289dbe54e36a5e74255cbb251b05364001491f86f2cc5523cd77b3  results/dev2/o2/logs/nf_tide_T4.json
6dc699b3692a994d313536157ddbc79588d3e94bd32dfb3805ed1574fc9b8b8a  results/dev2/o2/logs/nf_tide_T5.json
3a92f0067a62eed9502ce10e712d3ff98feffb9fd273a1b40cb6c4fedb696117  results/dev2/o2/logs/chronos2_T1.json
4dd8be15b6b93beaa069553948e216f5e52a015ee6dc63673b9c7f7da91ddb1b  results/dev2/o2/logs/chronos2_T2.json
090046721616b9e4b383a6e769f888a0bed37b06fa038b659cf2cb656a2762b9  results/dev2/o2/logs/chronos2_T3.json
7439120260f4e693809420d1094e1ebd0276772a44541c1b487e22ac9d66a11e  results/dev2/o2/logs/chronos2_T4.json
0c944cd05c05766be48615a147afbb7b7fd205c5c032d2e23252dadcab562b10  results/dev2/o2/logs/chronos2_T5.json
e6229131723ee56b5d9f1d8f20fff7a7302cf1725081f3e17c9dd8adec0dde07  results/dev2/o4/choice.json
ce53716026a0bf951fb682017d61bb64b392328f6e03102a1106379d60ba86ef  results/dev2/o5/params.json
f07727d831f03717f351d75c1c9554dd6e1c3d4e8d9dfe8f725beeb0df5d713a  codes/scripts/confirm/c01_hypotheses.py
```

## 3. Hypotheses and decision rules

Evidence unit: the day. Errors are pooled across States by the daily mean of scaled errors. Paired comparisons use
Diebold–Mariano tests with the HLN correction (one-sided in the stated direction) and 7-day block-bootstrap 95 %
intervals. Holm correction inside each family; Benjamini–Yekutieli across all primary tests is also reported.
A hypothesis is **supported** when its Holm-adjusted p < 0.05 and its extra rule (if any) holds. The scoring is done by
`codes/scripts/confirm/c01_hypotheses.py`, whose hash is listed above. A rehearsal of this scoring on the development
results (already seen, so not a test) gave 32 of 50 checks supported (`results/tables/dev_hypotheses.md`).

### Family F1 — forecasting (O2)

* **H1a–e.** For each target T1–T5, the final forecaster has lower scaled absolute error than the seasonal naive,
  horizons pooled. Extra rule: every one of the 5 seeds (members rebuilt from that seed with the frozen settings) is
  also better at p < 0.05.
* **H1f.** For each target, the final forecaster has lower scaled absolute error than the round-1 selected model
  (T4 compares chronos2 with chronos).
* **H2.** For each target, the calibrated 80 % and 90 % intervals of the final forecaster cover within ±5 points of
  nominal (descriptive rule, no p value).
* **H3.** For each target, LightGBM with measured publication lags has higher scaled error than with instant
  publication (the publication-latency cost is positive).

### Family F2 — context gating and the carbon layer (O3)

* **H4.** For each target, the staleness-aware gated model (`mcag_instance_sd`) has lower scaled pinball loss than the
  matched-capacity fusion model (`fusion_fixed_sd`). Development evidence: supported for T1–T3, not for T4 and T5.
* **H5.** For T1, T2 and T5, real context beats both null context and shuffled context on pinball loss (the larger of
  the two p values decides).
* **H6.** For each target, the online time-adaptive gate lowers scaled absolute error of `mcag_instance`.
* **H7.** For T1 and T2, when the RE and weather blocks go missing on every evaluation day, the extra error of
  `mcag_instance_sd` is smaller than that of `fusion_fixed_sd`. Extra rule: the upper end of the interval is below 0.
  Development evidence: not supported.
* **H8.** The carbon block improves pinball loss for T3 (one-sided test), and does not worsen it for T1 and T2
  (upper end of the interval below 1 % of the no-carbon loss).

### Family F3 — deviation assessment (O4)

* **H9.** The O4c assessment ranks the composite consequence better than the energy-only measure M₀ within M₀
  deciles. Extra rule: the week-block bootstrap interval of the difference in Kendall τ is above 0.
* **H10.** The gain from forecast information (U and the split into known and surprise parts) beats the 95th
  percentile of 50 placebo gains in which U and the forecast are shuffled within State and the form is re-fitted.
  Development evidence: not supported.

### Family F4 — scheduling (O5) and the chain

* **H11a–c.** The smoothly coupled arm A4s has lower tail-day system regret than A2, than the latest published
  schedule (H_pub) and than the plain forecast arm P. Extra rule: interval upper end below 0.
* **H11d–e.** The system-tail arm A5 has lower tail-day system regret than A2 and than P. Extra rule as above.
* **H11f–g.** PPO (A1) has lower mean system regret than A2 and than P.
* **H12.** No decision arm puts any State-day outside the hard limits (descriptive rule: largest share = 0).
* **H14.** The chain with the round-2 forecaster has lower mean system regret than the same arm (A2, frozen
  parameters) with the round-1 forecaster.

### Family F5 — price of adaptivity (PART)

* **H13.** For each refinement (fixed → regime, fixed → instance, fixed → online gate), PART computed on the
  calibration year agrees in sign with the realised evaluation-period gain in more than half of the target × horizon
  cells (one-sided binomial test). Development evidence: not supported. The decision-rule version (refine only when
  the lower end of the 80 % interval is above 0) is reported alongside.

## 4. Reporting commitments

* Every hypothesis is reported with its estimate, interval and both adjusted p values, supported or not.
* Development results (rounds 1 and 2) are labelled as development evidence and kept apart from these results.
* Any deviation (a code bug, missing data) goes into the table in section 6 with its date and reason, before the
  affected result is read where possible. Bug fixes do not change any frozen choice.
* The O5 settlement is stylised (no frequency linkage, no 15-minute blocks, no volume limits) and is labelled as such.
* The reserved period is evaluated once.

## 5. Computation plan (H100 only)

One PBS GPU job on the NIT Jalandhar H100 cluster (`run_all.py`, job `cuefo_confirm`): freeze (hash, commit, tag) →
environment and panel check → leakage test in the confirmatory phase → round-1 samples and models → round-2 models →
LA-MCAG variants → round-1 and round-2 evaluations → O4 (round 1 and O4c) → O5 (round-1 rules, round-2 arms, PPO) →
evidence table → `c01_hypotheses.py`. Results go to `results/confirm` and `results/confirm2`; the hypothesis table to
`results/tables/confirm_hypotheses.md`.

## 6. Deviations after the freeze

| Date | What | Why | Result read before the change? |
|---|---|---|---|
