# Results of the forecast-correction study on Indian data

| Folder | What it is | Final? |
|---|---|---|
| `india/confirm/` | The pre-registered all-India hourly run on the held-out months (1 Mar – 13 Sep 2026), H100 job 34223 | **yes: read this one** |
| `india/confirm/analysis/verdicts.json` | The verdict, estimate and corrected p-value of each of the 21 tests | **yes** |
| `india/explore/` | Correction ladders on the five daily State targets of the 34 control areas | exploratory |
| `india/dev/` | Development runs on data before 1 March 2026 | no, development |

Inside a run: `IH1` correction ladders for each forecast, `IH2` data-budget ladder, `prob_energy` uncertainty,
reserve and energy scoring, `x1` the simulation on the 34 State control areas, `analysis` the tests.

Results of the same method on data from other countries are in `../comparison_other_countries/06_results/`.
