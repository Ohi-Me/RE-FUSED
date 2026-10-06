# RE-FUSED-5 and RE-FUSED-6 — preserved material

The two stages that turned the work from "build more" into "test what is actually true". Full accounts:
[RE-FUSED-5](../stages/stage_06_refused5.md), [RE-FUSED-6](../stages/stage_07_refused6.md).

| Folder | What is in it |
|---|---|
| `docs/` | The twelve documents of the two stages — see the table below — plus the RE-FUSED-5 and RE-FUSED-6 reports as HTML and PDF |
| `src/` | The code: panel and model-data builders, the NYISO fetcher, the multi-seed core, the criterion test, the public benchmarks, the audit scripts, and the `refused6` package |
| `results/` | Result JSON and CSV files, and the run logs M1 to M10 |
| `audit/` | The price-provenance audit: per-file raw prices, archive comparison, day coverage |
| `data_small/` | The small derived files only: the rebuilt India panel, the model-daily table, and the D1–D3 reports |

## The documents, in reading order

| File | What it is |
|---|---|
| `01_novelty_map.md` | Six clusters of prior work, what each solves and what it leaves open. The rule: a component is rejected if it is a renamed version of anything in it |
| `02_formulation.md` | The formal statement of the correction-and-gate problem |
| `03_results_log.md` | A running record of every measured number, with the script and result file behind each |
| `04_algorithm.md` | The algorithm as it stood in RE-FUSED-5 |
| `05_adversarial_review.md` | A deliberate attempt to break the stage's own claims |
| `06_reproducibility.md`, `11_reproducibility.md` | How to re-run the two stages |
| `07_refused6_diagnosis_and_plan.md` | **The turning point**: nine verified defects in RE-FUSED-5, with evidence and severity, and the plan that followed |
| `08_defect_lineage.md` | Which defect produced which wrong number, and what the corrected verdict is |
| `09_theory.md` | The theory, with what is known, what is specialised and what is new |
| `10_reviewer_audit.md` | A reviewer-style audit of the result claims |
| `12_REFUSED6_final_report.md` | The final report: the finding, the recommended procedure, and the evidence stage by stage |

## What is not here

The bulk NYISO data (about 365 MB of parquet). The gated-correction programme fetches and documents its own
copy, and `src/fetch_nyiso.py` is kept here, so nothing is lost.

## Status

Superseded by RE-FUSED-7 as evidence. Still the best statement of *why* the research is organised the way it is —
in particular `07_refused6_diagnosis_and_plan.md`, which is the direct reason everything after it was
pre-registered.
