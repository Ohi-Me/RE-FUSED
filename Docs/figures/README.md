# Figures

| File | What it shows |
|---|---|
| `architecture.svg` | The whole research, from official reports to tested results |
| `forecasting.svg` | How day-ahead forecasts are made with data only as published |
| `scheduling.svg` | From forecast to schedule to deviation charge |
| `correction.svg` | Gated forecast correction and the gate ladder |
| `protocol.svg` | How every claim is tested: develop, freeze, unlock, run once, report |
| `data_coverage.png` | Which official source covers which period |
| `publication_lags.png` | How late each report is published |
| `forecast_accuracy.png` | Final forecast accuracy against the seasonal naive forecast |
| `late_data_and_intervals.png` | Cost of late data; coverage of the forecast intervals |
| `scheduling_regret.png` | Daily regret of each scheduling rule |
| `value_of_accuracy.png` | Money value of forecast accuracy |
| `correction_skill_hourly.png` | Error removed by forecast correction on the all-India hourly series |

The diagrams are drawn by `tools/make_github_diagrams.py` in the research archive; the charts are copied from the
study folders, where the code that made them is kept.
