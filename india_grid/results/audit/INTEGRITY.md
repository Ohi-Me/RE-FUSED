# Integrity and reproducibility check

Run 2026-10-06T12:45:08+0530 on `master` (PBS job 34188.master), phase `confirm`.

| Check | Result |
|---|---|
| Pre-registration hash matches `PREREG_SHA256.txt` | yes |
| Frozen choice files and scorer (42 files) unchanged | yes (1 renamed on 24 Sep 2026, see RENAME_RECORD.md: codes/scripts/confirm/c01_hypotheses.py) |
| Panel content hash matches `CHECKSUM.txt` | yes (104550 rows x 107 columns) |
| Git tag `prereg-v1` present | yes (4d3807ed275342c1f83f2e7009f094e838f20bad) |
| Re-scoring the stored results reproduces `hypotheses.csv` | yes, identical |

Supported checks in the archived table: 35 of 52; in the re-scored table: 35.

The scoring script is `codes/scripts/confirm/c01_hypotheses.py`, sha256 `a1e2296c9bc3e6a77785f8209013bba56549de72a65a0c31b9f12a1f8312f9ee`. The pre-registration, written before the evaluation data were opened, records `f07727d831f0…` for this file; it differs only in the five import lines changed by the rename of 24 Sep 2026 (`codes/design/RENAME_RECORD.md`), and the re-scoring above is what shows that the change has no effect.
