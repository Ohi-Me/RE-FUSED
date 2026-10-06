# RE-FUSED: State-day panel of India's power system, with code and pre-registered results

Version {version}

**Authors**

* Rohit Kumar, Centre for Artificial Intelligence, Dr. B. R. Ambedkar National Institute of Technology Jalandhar,
  India. ORCID 0009-0004-7936-9683. Contact: rohitaryan0613@gmail.com
* Urvashi Bansal, Department of Computer Science & Engineering, Dr. B. R. Ambedkar National Institute of Technology
  Jalandhar, India. ORCID 0000-0002-3395-9942
* Sukumar Nandi, Department of Computer Science and Engineering, Indian Institute of Technology Guwahati, India.
  ORCID 0000-0002-5869-1057

{doi_line}

## What this record is

RE-FUSED is our study of day-ahead forecasting and decision support for the State control areas of India's power
system. We built a daily panel from official government reports, trained and compared forecasting models on it,
tested whether context information and calibrated uncertainty improve forecasts, and followed the forecasts
through to deviation assessment and day-ahead scheduling. Before looking at the final evaluation period we froze
all our choices in a pre-registration and hashed it, and then scored the results once.

This record contains everything needed to check that work: the panel, its data dictionary, the code, the frozen
choices, the confirmatory results, and the figures and tables used in our papers.

## The panel in short

* {rows:,} rows and {cols} columns, one row per State control area and day
* {entities} State control areas, from {first} to {last}
* built only from reports published by Grid Controller of India (Grid-India), the Central Electricity Authority
  (CEA) and the India Meteorological Department (IMD)
* every column has a measured publication delay, so a forecast can use only what was public at the time
* content hash (SHA-256 of the panel as read from the file): `{content}`

## Folder structure

```
REFUSED_Zenodo/
├── README.md                    this file
├── LICENSE-DATA.txt             CC BY 4.0 for data, tables, figures and documentation
├── LICENSE-CODE.txt             MIT licence for the code
├── CITATION.cff                 how to cite this record
├── MANIFEST.sha256              SHA-256 of every file in the record
├── data/
│   ├── processed_panel/         the panel (Parquet and gzipped CSV), entity list, content hash
│   ├── data_dictionary/         field registry, datasheet, source registry, QC report, validation numbers
│   ├── interim_tables/          source tables at their native resolution, before the panel join
│   └── source_manifests/        URL, size, retrieval time and SHA-256 of every downloaded report
├── code/
│   ├── refused_library/          shared Python package (paths, loaders, statistics, parsers)
│   ├── preprocessing/           download scripts and the panel build (b01 to b08)
│   ├── forecasting/             forecasting models, context gating, round-2 combination
│   ├── analysis/                deviation assessment, scheduling, scoring, paper tables, audit checks
│   └── reproducibility/         driver script, environment set-up, tests, pre-registration, protocols
│                                and the scripts that check this record
├── results/
│   ├── hypotheses.csv           the {n_checks} pre-registered checks, as scored
│   ├── INTEGRITY.md             hash and re-scoring check made after the run
│   ├── frozen_results/          frozen choice files and the confirmatory evaluation outputs
│   ├── additional_analyses/     analyses made after the confirmatory scoring (not pre-registered)
│   └── tables/                  every result table and number used in the papers
├── figures/                     figures used in the papers
└── documentation/
    ├── data_sources.md          where the data come from and how they were collected
    ├── methodology.md           what we did, step by step
    └── reproducibility.md       how to check or rerun the work
```

## Quick start

```python
import pandas as pd

panel = pd.read_parquet("data/processed_panel/refused_state_day.parquet")
fields = pd.read_csv("data/data_dictionary/FIELD_REGISTRY.csv")   # unit, source and delay of every column
print(panel.shape)
```

The same panel is also given as `refused_state_day.csv.gz` for tools that do not read Parquet.

## Main results

Of the {n_checks} pre-registered checks, {n_supported} are supported. The checks that failed are kept in
`results/hypotheses.csv` and are discussed in our papers as findings, not removed. The full list, with estimates,
intervals and adjusted p-values, is in that file.

## Changes from version 1.0

* File, package and folder names use the study's name, RE-FUSED, throughout (the panel is
  `refused_state_day.parquet`, the library `code/refused_library/`). The data and results are unchanged; the
  panel's content hash is the same as in version 1.0.
* Added the scripts and outputs of the additional analyses reported in our papers as not pre-registered
  (`code/analysis/extra/`, `results/additional_analyses/`): the value of forecast accuracy in the settlement, where
  the settlement places the best schedule, hindsight bounds on adaptivity, and the forecast-error energy of each
  correction-gate design.
* The per-row forecasts and decisions behind the evaluation are in a separate file of this record,
  `REFUSED_predictions_v{version}.zip`, so that this archive stays small.
* The same code, with run logs and one folder per paper, is at https://github.com/Ohi-Me/RE-FUSED.

## Licence

Data, tables, figures and documentation: CC BY 4.0. Code: MIT. The original government reports are not
redistributed here; `data/source_manifests/` records where each one came from.

## Acknowledgement

The computations were run on the High Performance Computing facility of Dr. B. R. Ambedkar National Institute of
Technology Jalandhar, which we gratefully acknowledge.
