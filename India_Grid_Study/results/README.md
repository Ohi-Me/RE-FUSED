# Results

| Folder | What it is | Final? |
|---|---|---|
| `confirm2/` | All final models and the final combined forecaster, scored once on the held-out period (1 Apr 2025 – 31 Aug 2026) | **yes: read this one** |
| `confirm/` | First-round models scored on the same held-out period (the final combinations build on them) | yes, first round |
| `tables/` | Readable summaries. Start with `confirm_hypotheses.md` (all 52 tests) and the `confirm2_*_summary.md` files | **yes** |
| `extra/` | Analyses added after scoring (value of accuracy, risk levels, correction in energy units); not pre-registered | yes, labelled extra |
| `audit/` | Integrity check: frozen hashes, independent rescoring (`INTEGRITY.md`) | yes |
| `rebuild/` | The panel rebuilt from the raw reports, and the checksum comparison (`VERIFY.md`) | yes |
| `dev/`, `dev2/` | Development rounds 1 and 2, on data before 1 April 2025 only | no, development |
| `dev2_smoke/`, `dev2_frozen_check/` | Quick test runs of the round-2 code before it was frozen | no, checks |

Inside each run folder: `o2` forecasting, `o3` context-gated fusion, `o4` deviation assessment, `o5` scheduling.
Each has `preds/` (forecasts; large files are on Zenodo), `eval/` (scores) and `logs/`.
