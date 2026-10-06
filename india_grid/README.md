# india_grid

Everything built on the official Indian reports: the data pipeline, the State-day panel, the all-India series,
day-ahead forecasting, deviation assessment and scheduling, with all results and run logs.

## Folders

| Folder | What it holds |
|---|---|
| `data/processed/` | The State-day panel, the all-India 15-minute and hourly series, the hourly prices, and `CHECKSUM.txt` |
| `data/interim/` | Parsed tables from each source before they are joined, and the parse logs |
| `data/docs/` | Field registry (source, unit, availability lag of every column), datasheet, quality reports, source registry, validation |
| `codes/refused/` | The library: parsers, panel builder, targets, models, gating, assessment, scheduling, statistics |
| `codes/scripts/acquire/` | Download scripts for Grid-India, CEA and IMD; each writes a manifest with URL and SHA-256 |
| `codes/scripts/build/` | `b01`–`b08` build and validate the State-day panel; `b09` the all-India 15-minute series; `b10` hourly prices |
| `codes/scripts/o2`–`o5` | Forecasting models and their evaluation (`o2`), context-gated fusion (`o3`), deviation assessment (`o4`), scheduling rules and the learned PPO policy (`o5`) |
| `codes/scripts/dev`, `dev2` | Development rounds 1 and 2 |
| `codes/scripts/confirm/` | The confirmatory scoring of all pre-registered hypotheses |
| `codes/scripts/extra/` | Additional analyses run after scoring (labelled as not pre-registered) |
| `codes/scripts/audit/` | Integrity checks: frozen hashes, rescoring, panel rebuild |
| `codes/design/` | Research design and the frozen pre-registration with its hashes |
| `codes/tests/` | The leakage test: no feature may use data before it was published |
| `results/` | `dev/`, `dev2/` development results; `confirm/`, `confirm2/` confirmatory results; `tables/` summaries; `extra/` additional analyses; `audit/` integrity reports; `rebuild/` the panel rebuilt from the raw files |
| `logs/` | Every stage's log, the PBS job scripts and their output, with job numbers and the GPU instance used |
| `hpc/` | Job files for the H100 cluster and the environment setup |

## Running it

One driver, `run_all.py`, runs every stage in order and skips what is already done:

```bash
python run_all.py status          # which stages are done
python run_all.py stage o2_tft    # run one stage
python run_all.py submit          # on the cluster: submit all PBS jobs with their dependencies
python run_all.py local           # on any machine: run all stages one after another
```

Settings (cluster user, host, conda environment, cores) go in a private `.env`; `.env.example` lists the keys.
All results here were produced on the H100 cluster inside PBS jobs, one MIG instance and at most 8 CPU cores per
job.

## Rules the code follows

- **Data as published.** A column enters a forecast only after its availability lag has passed; `codes/tests`
  fails if any feature would have been unknown at forecast time.
- **Blinding.** During development the code does not load rows from 1 April 2025 onwards. The confirmatory run
  needs the frozen pre-registration: its hash, its git tag and an explicit flag must all match.
- **Nothing typed by hand.** Every reported number is read from a result file.

The results are summarised in plain words in [`../docs/RESULTS.md`](../docs/RESULTS.md).
