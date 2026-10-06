# Rename record — CUEFO-8 to RE-FUSED, 24 September 2026

The study was renamed from CUEFO to **RE-FUSED** (Renewable Energy Forecasting, Uncertainty,
Schedule-Deviation, Evaluation & Decision-Support) after the confirmatory run was complete and scored. This
file records exactly what the rename touched, so that the frozen record stays checkable.

## What was not touched

* `codes/design/02_preregistration.md` and `PREREG_SHA256.txt` are frozen documents and were **not edited**.
  They still carry the hashes recorded on 18 September 2026, before the evaluation data were opened.
* Nothing under `results/` was edited. Every stored result file is bit-for-bit as the confirmatory run left it.
* Nothing under `logs/` was edited, so the PBS job names of the runs that actually happened (`cuefo_o2_gpu`,
  `cuefo_confirm`, …) are preserved as run evidence.
* `data/raw/` is unchanged, including the pilot copy `legacy_cuefo0/` and every source manifest.
* The published Zenodo record (DOI 10.5281/zenodo.22870921) is unchanged, and the historical notes that name
  its uploaded file `CUEFO_Zenodo.zip` still name it that.

## What the rename changed

| Kind | From | To |
|---|---|---|
| Python package | `cuefo8` | `refused` |
| Environment variables | `CUEFO8_PHASE`, `CUEFO8_CONFIRMATORY`, `CUEFO8_ROOT`, … | `REFUSED_PHASE`, `REFUSED_CONFIRMATORY`, `REFUSED_ROOT`, … |
| Panel file | `data/processed/cuefo8_state_day.parquet` | `data/processed/refused_state_day.parquet` |
| Entity table | `data/processed/cuefo8_entities.csv` | `data/processed/refused_entities.csv` |
| LaTeX helpers | `\cuefofit`, `\cuefobox` | `\refusedfit`, `\refusedbox` |
| PBS job names (future runs) | `cuefo_*` | `refused_*` |
| Conda environment (fresh installs) | `cuefo_h100` | `refused_h100` |
| Cluster folder | `/Data3/<user>/CuefoH100` | `/Data3/<user>/RE-FUSED` |
| Prose in code, protocols and manuscripts | CUEFO-8, CUEFO-7 | RE-FUSED |

Historical stage names — CUEFO-0 to CUEFO-6, `legacy_cuefo0`, `90_prior_cuefo6` — were deliberately left
alone. They name things that existed under those names, and renaming them would misrepresent the record.

## Effect on the frozen hashes

Of the **42 files hashed inside the pre-registration, 41 are unchanged**. One changed:

| File | Hash in the pre-registration | Hash after the rename |
|---|---|---|
| `codes/scripts/confirm/c01_hypotheses.py` | `f07727d831f03717f351d75c1c9554dd6e1c3d4e8d9dfe8f725beeb0df5d713a` | `a1e2296c9bc3e6a77785f8209013bba56549de72a65a0c31b9f12a1f8312f9ee` |

The whole difference is five import lines:

```
-from cuefo8 import dev2 as D          +from refused import dev2 as D
-from cuefo8 import o2data as O        +from refused import o2data as O
-from cuefo8 import stats as S         +from refused import stats as S
-from cuefo8.paths import TAB          +from refused.paths import TAB
-from cuefo8.report import md_table    +from refused.report import md_table
```

No statistic, threshold, correction, decision rule or data path in the scorer changed. The pre-rename file is
recoverable from git at tag `pre-refused-rename`.

## How the rename is verified

A hash comparison cannot prove that a rename is harmless, so the check used is stronger: **the renamed scorer
is run again over the stored results and its output is compared, cell by cell, with the archived hypothesis
table.** That is what `codes/scripts/audit/a02_integrity.py` does, and the run after the rename reproduced the
archived table identically — 52 checks, 35 supported, no cell different.

The integrity report therefore states the frozen-file result as "41 of 42 unchanged, 1 renamed and recorded
here", and the reproducibility claim rests on the re-scoring, which is exact.
