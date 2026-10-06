# RE-FUSED-0.1 to RE-FUSED-4 — preserved material

Five release folders that together took 738 MB, of which only about 18 MB differed. The shared parts — the
dataset (141 MB in each folder) and the identical notebook outputs — are kept once, elsewhere. What is here is
what actually distinguishes each release.

| Folder | What differs | Stage document |
|---|---|---|
| `refused0.1/` | The portable repackage. Nothing changed in code, data or notebook | [0.1](../stages/stage_01_refused0_1.md) |
| `refused1/` | The feature fix: notebook, the session report, `refused_upgrade/features.py` and `trees.py`, the execution log, and the result tables and figures | [1](../stages/stage_02_refused1.md) |
| `refused2/` | The rewritten related-work documents and the narrowed novelty claims | [2](../stages/stage_03_refused2.md) |
| `refused3/` | The regime-stratified price test, opt-in recency weighting, and the A–E forecast-to-RL ablation tables | [3](../stages/stage_04_refused3.md) |
| `refused4/` | The LOCF imputation audit and the per-cell execution report | [4](../stages/stage_05_refused4.md) |

## The thing to know

RE-FUSED-1, 2, 3 and 4 were executed four separate times and produced **byte-identical** result tables and
figures. Under seed 42 on the same hardware the pipeline was deterministic, and only RE-FUSED-1 ever retrained a
model. So the result files in `refused2/`, `refused3/` and `refused4/` are not new evidence — they are the same
evidence, re-run.

If a number from this period is needed, take it from `refused1/results/` or from the evolution dossier.

## Where the data went

`full_dataset.csv`, `RL_TRAIN_DATA.csv` and `RL_TEST_DATA.csv` were identical in all five folders and in
RE-FUSED-0. One copy is in [`../datasets_early/`] (not published: non-government data).
