# State-day panel, version 1.1

Version 1.1 of the State-day panel, built on 25 October 2026. It differs from version 1.0 (`data/processed/`) in one
source only.

In November 2025 the Central Electricity Authority moved its Daily Renewable Generation Report to its RE generation
portal (https://gen-re.cea.gov.in/reports) and has published it there every day since. The report listing on
cea.nic.in that version 1.0 read has no report after 18 November 2025. Version 1.1 adds the portal reports from
1 November 2025 to 31 August 2026 (304 files; one day only as PDF). On the 18 days published on both listings the
two copies give identical values in all 5,040 parsed cells, and on every shared date the old listing is kept.

What changes, from 19 November 2025 onwards only: the `re_*` columns, `avail_re`, and the consumption- and
generation-based carbon accounting columns `co2_ci_gen_acc` and `co2_ci_cons_acc`. Rows, columns and every other
value are those of version 1.0.

| Folder | Content |
|---|---|
| `data/processed/` | `refused_state_day.parquet` (104,550 rows x 107 columns), `refused_entities.csv`, `CHECKSUM.txt` |
| `data/docs/` | `FIELD_REGISTRY.csv`, `QC_REPORT.md`, `validation.json` (including the cross-listing check) |
| `data/interim/` | the renewable State-day table and its parse log (the other interim tables are those of version 1.0) |

Rebuild (from the project folder, with the raw files downloaded):

    py -3.10 codes/scripts/acquire/cea.py re_portal
    set REFUSED_ROOT=<project>/versions/v1_1, REFUSED_RAW=<project>/data/raw, REFUSED_RE_PORTAL=1
    py -3.10 codes/scripts/build/b04_cea_re.py      (the other interim tables are copied from data/interim)
    py -3.10 codes/scripts/build/b07_panel.py
    py -3.10 codes/scripts/build/b08_validation.py

Version 1.0 stays the frozen input of the pre-registered studies; use version 1.1 for new work.
