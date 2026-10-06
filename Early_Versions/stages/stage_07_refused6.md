# RE-FUSED-6 — the price of adaptivity, found (12 September 2026)

**Folder then:** inside `REFUSED5` (`src/refused6`, `docs/07`–`12`, `results/`)
**Kept now:** `Early_Versions/stage5_6_refused5/`; a frozen snapshot also sits in
`Forecast_Correction_India/90_prior_refused6/`

## Objective

Fix RE-FUSED-5's defects and answer one question properly: **where on the adaptivity ladder should a correction
gate sit?**

The ladder runs from no correction, through one global gate, one per series, one per series × regime cell, a
shrunk version, up to one gate per instance.

## Method

A battery of experiments, M1 to M10, across India daily generation, four ETT datasets and NYISO prices, three
test years, four gate estimators and a tuned neural gate, with multiple seeds and integrity checks.

## The finding

**The most adaptive gate is almost never the best.** On India, mean skill by rung was: none 0, global 0.069,
**per-series 0.099**, per-cell 0.094, shrunk 0.086, per-instance 0.041. Per-series was the best rung in 10 of
15 configurations; per-instance in none.

The loss survived every attempt to explain it away: four different estimators (M1b), rescaling an under-scaled
corrector (M1c), and removing the clipping cap (M1d). The mechanism is estimation, not implementation — the
finer gate re-solves the residual regression on a small validation set.

The same pattern appears one level up: **choosing the granularity from validation data does not beat fixing it
in advance** (M5c, M10b), and per-instance safety mechanisms lose to a simple per-series deployment test.

## What it recommends

A four-stage procedure: fit the corrector on training data, reparameterise on validation, fit one gate per
series with clipping, and deploy it only if a block-wise lower confidence bound on its skill is positive.
Nothing is tuned on test data. What was explicitly **not** included: per-instance and neural gates, shrinkage
toward them, validation-based rung selection, and instance-level clipping — each lost when tested.

An earlier design called SCALE assumed the adaptivity level could be estimated from validation data. That
assumption was tested, did not hold on real data, and the name was retired.

## Honest limits, as stated at the time

The project does **not** establish a rule for choosing between per-series and per-cell gates. Both sit on the
plateau, and picking one from validation is exactly the step that did not pay.

## What changed after it

This became the research question of RE-FUSED-7, where it was pre-registered and tested on data collected after
the hypotheses were frozen. It also became family F5 of the India programme, so the same question is asked in
two different settings.

## Status

Superseded by RE-FUSED-7 as evidence; retained as the origin of the idea and of the defect list that forced
pre-registration.
