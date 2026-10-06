# RE-FUSED

**Renewable Energy Forecasting, Uncertainty, Schedule-Deviation, Evaluation & Decision-Support**

Research on India's power system, built only from official Government of India data: daily reports of Grid
Controller of India (Grid-India), the Central Electricity Authority (CEA) and the India Meteorological Department
(IMD). It asks four practical questions:

1. How well can each State's demand, drawal, generation and market price be forecast a day ahead, using data only
   as it is actually published?
2. How should a State set its day-ahead schedule when deviations from it are charged, and does a better forecast
   lower that charge?
3. When does adaptive correction of a day-ahead forecast actually help, and when does it only add noise?
4. Can all of this be released as an open dataset that anyone can rebuild from the original reports?

Author: **Rohit Kumar**, Centre for Artificial Intelligence, Dr. B. R. Ambedkar National Institute of Technology
Jalandhar. All computation ran on the institute's NVIDIA H100 cluster.

![Overview of the research](Docs/figures/architecture.svg)

## Repository layout

```
RE-FUSED/
├── README.md                 this page
├── FINAL_VERSION.md          which code produced each final result, where the result is, and its run log
├── India_Grid_Study/         main study: official data, Indian datasets, forecasting, deviation assessment, scheduling
├── Forecast_Correction_India/   forecast correction on real Indian data
│   └── Other_Countries/      the same method on data from other countries (comparison only)
├── Early_Versions/           earlier versions of the work and why it changed
└── Docs/                     guides (data, results, how to rerun) and all figures
```

Every folder opens with a `README.md` that says what is inside it. `India_Grid_Study` and
`Forecast_Correction_India` are the **final version** of the work.

| Folder | What it holds |
|---|---|
| [`India_Grid_Study/`](India_Grid_Study/) | Download and checking of every official report; the State-day panel and the all-India 15-minute and hourly series; day-ahead forecasting; deviation assessment; scheduling under deviation charges; all results and run logs |
| [`Forecast_Correction_India/`](Forecast_Correction_India/) | Can a forecast be improved by learning its past errors, and how finely should that correction be tuned? Run on the all-India hourly series and the State daily series, with its frozen plan and run logs |
| [`Other_Countries/`](Forecast_Correction_India/Other_Countries/) | The same correction method on New York, European and Portuguese data, kept only to compare with the Indian results |
| [`Early_Versions/`](Early_Versions/) | Code, results and notes of the earlier versions |
| [`Docs/`](Docs/) | [Data](Docs/DATA.md), [results](Docs/RESULTS.md), [how to rerun](Docs/REPRODUCE.md), and all figures |

**Gated correction, in one line:** a forecast `B` is improved by adding a learned correction `r` of its usual
errors, scaled by a gate `g` between 0 and 1.5: `final forecast = B + g × r`. The gate can be one number for
everything, or one per series, per hour or per day; the study tests when a finer gate actually helps.

## The main findings, in short

All numbers come from result files in this repository. Each claim was written down and frozen before the
evaluation data were opened ([how](Docs/figures/protocol.svg)).

- **Late data is expensive.** Using reports only after they are published, instead of pretending they arrive at
  once, raises day-ahead forecast error by 15 % (State drawal) to 138 % (market price).
- **The final forecasts beat the seasonal naive forecast on all five targets** (scaled error 0.74 to 1.11 against
  1.03 to 1.62), and the 80 % and 90 % intervals hold their promised coverage on every target.
- **A better forecast lowers the deviation charge.** With the scheduling rule held fixed, the final forecast
  lowers system regret by 0.29 Rs crore per day. Each GWh of mean drawal-forecast error per State-day adds about
  2.6 Rs crore per day.
- **No adaptive scheduling rule beats one well-chosen fixed risk level**, including a learned (PPO) policy.
- **One simple correction gate is usually enough.** On the State series, one fixed gate removes about 13,400 GWh
  a year of drawal forecast error across the 34 control areas; finer gates add little or make it worse. Market
  prices are the exception: there only a gate that keeps re-learning over time helps.
- **The dataset is open**: 34 State control areas, 1 April 2018 to 31 August 2026, 104,550 State-days × 107
  columns, Zenodo DOI [to be added on publication](https://zenodo.org).

Details, with intervals and the tests that were *not* supported: [Docs/RESULTS.md](Docs/RESULTS.md).

| | |
|---|---|
| ![Forecast accuracy](Docs/figures/forecast_accuracy.png) | ![Cost of late data and interval coverage](Docs/figures/late_data_and_intervals.png) |
| ![Daily scheduling regret](Docs/figures/scheduling_regret.png) | ![Value of forecast accuracy](Docs/figures/value_of_accuracy.png) |

## How each part works

| | |
|---|---|
| Forecasting with data as published | ![forecasting](Docs/figures/forecasting.svg) |
| Schedule and deviation charge | ![scheduling](Docs/figures/scheduling.svg) |
| Gated forecast correction | ![correction](Docs/figures/correction.svg) |

## Data and licences

Code: MIT licence ([`LICENSE`](LICENSE)). The processed Indian data, documentation and results: CC BY 4.0
([`LICENSE-DATA.md`](LICENSE-DATA.md)). The original report files are not copied here; they stay with the agencies
that publish them, and the download scripts fetch them again and check them against the recorded SHA-256 hashes.
Files too large for GitHub are in the [Zenodo record](https://zenodo.org) ([list](Docs/LARGE_FILES_ON_ZENODO.md)).

## Citation

See [`CITATION.cff`](CITATION.cff). Rohit Kumar (2026). *RE-FUSED: research on India's power system from official
government data — code, results, logs and datasets* (version 1.0). Zenodo. DOI to be added on publication.
