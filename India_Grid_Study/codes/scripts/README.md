# scripts

| Folder | Step |
|---|---|
| `acquire/` | Download the official reports (Grid-India, CEA, IMD) |
| `build/` | `b01`–`b08` build and validate the State-day panel; `b09` all-India 15-minute series; `b10` hourly prices |
| `o2/` | Forecasting models, round 1, and their evaluation |
| `o3/` | Context-gated fusion (LA-MCAG) and its evaluation |
| `o4/` | Deviation assessment |
| `o5/` | Scheduling rules and the learned PPO policy |
| `dev2/` | Round 2: more models, the final combined forecaster, evaluation of all parts |
| `prereg/` | Filling the pre-registration tables from development results |
| `confirm/` | Rehearsal and the scoring of all pre-registered tests |
| `extra/` | Analyses added after scoring (labelled as not pre-registered) |
| `audit/` | Integrity checks: hashes, rescoring, strata |
| `release/` | Building and checking the Zenodo package |
