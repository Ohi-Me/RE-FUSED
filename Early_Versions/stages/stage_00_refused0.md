# RE-FUSED-0 — the origin (February to August 2026)

**Folder then:** `Refused0` (834 files, 3.33 GB)
**Kept now:** `Early_Versions/stage0_refused0/`, panel in `Early_Versions/datasets_early/`, the raw
aggregator files in `India_Grid_Study/data/raw/legacy_refused0/`

## Objective

Build a working end-to-end system for the Indian grid: a dataset, forecasts of several targets, a carbon-aware
element, a settlement-price idea, and a reinforcement-learning dispatch agent. At this stage the aim was
breadth — find out what could be built at all.

## Research question

Can carbon intensity, market signals and forecast uncertainty be combined into one framework that both forecasts
and dispatches, on Indian State-level daily data?

## Data

An 18-State daily panel, `full_dataset.csv`, 47,286 rows × 140 columns, covering 2017 to 2025, built from an
aggregator export (Indian Data Portal) together with CEA, POSOCO, IEX and weather files. The build ran in three
stages: `dataset` → `cleaned` → `cleaned_v2` → panel. Split: train 39,672 State-days (Oct 2017 – Dec 2023), test
7,614 State-days (Jan 2024 – Apr 2025).

## What was built

| Piece | What it was |
|---|---|
| Data pipeline | Notebooks in `data_pipeline/refine/` that clean and join the raw exports |
| RL_V1 → V2 → V3 | Three dispatch prototypes: scalar-action PPO; then an LSTM carbon twin, quantile forecasters, a 5-dimensional action and Lagrangian PPO; then statistical tests only |
| Paper_Model | A market-design model and several notebook generations, including A\* variants |
| RE-FUSED-Alpha | The stage's final pipeline, with six numbered contributions: PA-LMP (a settlement price signal), MCAG (a carbon-aware gate), GCAL (a carbon layer), a settlement counterfactual with regime-coupled DRO dispatch, a staged architecture comparison (BiLSTM, TFT, PatchTST families), and an action-conditioned dispatch simulator |
| Forecasting-upgrade harness | A separate harness fitting gradient-boosted trees to the residual from yesterday's value |
| Audits | Leakage audit, units table, imputation report, baseline anchors, a five-seed run |

## Results worth remembering

- The harness model — trees on the residual from persistence — reached **MASE 0.377 against persistence at
  0.408**, better than every deep model in the stage.
- The five-seed run showed a seed-to-seed spread of about 0.15 MASE on the headline metric, larger than most of
  the gains being claimed at the time.
- The leakage audit found four features built with statistics from the whole period.

## Problems discovered

1. Four leaky features.
2. Inconsistent units between sources.
3. Heavy imputation in the market-price column, which later turned out to be fatal for the price work.
4. Single-seed results were being reported as findings.

## What changed after it

The leakage and feature findings led directly to RE-FUSED-1, the one retraining event of the early line. The
seed-spread finding is why every later stage runs multiple seeds.

## What was kept

Notebooks (the main one and its backups, plus four executed multi-seed copies), the audit reports, the result
tables and figures, the related-work notes, the RL prototypes with their logs, and the early panel.

Dropped: the intermediate cleaning stages (`cleaned/`, `cleaned_v2/`, about 1.1 GB, rebuildable from the raw
copy with the notebooks that are kept) and the notebook and harness caches (`_ns_cache.pkl`, `_registry*.pkl`,
about 870 MB of generated pickles).

## Where the descendants are

Nothing from this stage is used in a current result. The India programme was rebuilt from official sources in
RE-FUSED-8; the only surviving artefact is the raw aggregator copy, kept for cross-checks.
