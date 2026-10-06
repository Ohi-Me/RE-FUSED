# RE-FUSED-3 — the negative results (20 August 2026)

**Folder then:** `Refused3` (80 files)
**Kept now:** `3_early_versions/stage1_4_releases/refused3/`

## Objective

Test two things that the framework had been assuming: that a learned price model beats persistence in the hard
regimes, and that feeding forecasts into the reinforcement-learning state improves dispatch.

## Experiments

| Experiment | Design | Result |
|---|---|---|
| Regime-stratified price evaluation | The learned price model against persistence, split by regime | **Ties persistence in every regime** |
| Forecast → RL ablation, arms A–E | Rule-based, PPO, PPO with a point forecast, with a probabilistic forecast, and with DRO; five seeds, paired t-tests | **No step is significant** (p = 0.093, 0.486, 0.378) |
| Recency weighting in the tree model, opt-in | On and off, five seeds | Helps only one target at one horizon (−0.41 % at h = 3, +0.96 % at h = 2). Correctly shipped as off by default |

## Problems discovered

The A–E ablation tables are timestamped 16 August, before RE-FUSED-1 was packaged, and **the script that produced
them ships in no folder**. The result stands as recorded, but it cannot be re-run from the archive. That is one
of the reasons every later stage keeps the generating script next to the result.

## Why it matters

These are the first properly multi-seed negative results in the project, and they hold up: the same shape of
answer — extra machinery on top of a good forecast does not pay — appears again in both finished programmes.

## Status

Superseded as a result, kept as part of the argument. The core model outputs are identical to RE-FUSED-1's.
