# RE-FUSED-1 — the feature fix (17 August 2026)

**Folder then:** `Refused1` (68 files, 141 MB)
**Kept now:** `Early_Versions/stage1_4_releases/refused1/`

## Objective

Fix what the RE-FUSED-0 audits had found: leakage in four features, and models that could not see each target's own
recent history.

## What changed

1. Every target got **its own history as an input** (25 → 30 features).
2. Four leaky features were refit using training-period statistics only.
3. The forecasting harness gained four exogenous columns.

The models themselves were not changed. The notebook was re-executed end to end on the laptop GPU
(1,615.5 seconds, 28 of 28 cells).

## Result — the largest measured gain in the whole early line

| Measure | Change |
|---|---|
| Mean MAPE of each architecture family's best model | −49.9 % to −56.7 % |
| Headline MASE on the observable targets | 3.475 → 0.976 (−71.9 %) |
| Average CRPS | −50.3 % |

The gain came from giving the model information it should always have had, not from a new architecture. That
lesson — inputs and baselines before architecture — is the one the whole later research is built on.

## What did not improve

Even after the fix, the best deep model in the notebook still did not beat a seasonal naive on either observable
target (TFT-Lite generation MASE 1.267 against 0.994; price 0.686 against 0.527). The separate tree harness
remained the strongest forecaster anywhere in the stage.

## Problems discovered

The carbon-gate claim weakened: the random-noise control gate beat the real carbon gate (14.08 % against
15.29 % MAPE), and the notebook's own Diebold–Mariano test favoured the price-only gate. The checklist in the
notebook still printed "N2 CONFIRMED", which is exactly the kind of drift between prose and outputs that the
later stages were designed to prevent.

## Status

Superseded. RE-FUSED-2, 3 and 4 reuse its trained outputs unchanged — the result tables and figures carry the same
MD5 in all four folders, from four separate GPU runs.
