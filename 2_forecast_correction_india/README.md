# 2 · Forecast correction on real Indian data (final version)

Which script produced each final result, and where that result is: [`../FINAL_VERSION.md`](../FINAL_VERSION.md).

When does adaptive correction of a day-ahead forecast pay? A base forecast `B` is corrected by a learned term
`r`, scaled by a gate `g`: `ŷ = B + g·r`. The gate can be one number for everything, one per series, per hour, per
instance, or re-learned over time. Finer gates can follow real differences, but they also fit noise. This study
measures which effect wins, on Indian data, and tests rules that pick the gate from validation data alone.

![Gated correction](../docs/figures/correction.svg)

## The Indian study

| Part | Data | Status |
|---|---|---|
| All-India hourly study | demand, net demand, wind and solar, hourly, from the Grid-India SCADA series; held-out period 1 Mar – 13 Sep 2026 | pre-registered (`02_design/05_preregistration_india.md`, git tag `prereg-india-v1`), run once on the H100 |
| State daily ladders | the five targets of the 34 State control areas, simple and learned base forecasts | exploratory |
| State panel replication | the same question inside the India programme's own pre-registration (family F5) | confirmatory, see `../1_india_data_forecasting_scheduling` |

| Folder / file | What it holds |
|---|---|
| `02_design/` | The frozen pre-registration of the Indian hourly study, its SHA-256 and freeze time, and the list of deviations (`06_deviations_india.md`) |
| `04_code/refused_gate/` | The method: gates, ladders, selection rules (PART, prequential), probabilistic and energy scoring, statistics, data loaders, and `guard.py`, which refuses held-out data until the pre-registration is frozen |
| `04_code/scripts/india/` | The Indian runs: ladders, probabilistic and energy scoring, simulations, analysis |
| `04_code/scripts/dev/` | Learned base forecasts (LightGBM, NHITS, PatchTST, TiDE, DLinear, Chronos) and the development ladders |
| `03_data/processed/` | Learned base forecasts for the Indian hourly series |
| `06_results/india/` | `dev/` development, `explore/` State daily ladders, `confirm/` the pre-registered run and its `analysis/verdicts.json` |
| `12_logs/` | Run logs, including every confirmatory attempt |
| `13_env/` | Pinned requirements and CUDA constraints |
| `hpc/` | PBS job files (`india_confirm.pbs` is the pre-registered run) |
| `comparison_other_countries/` | The first test of the method, on data from other countries; kept only for comparison |

## Rerun

```bash
pip install -r 13_env/requirements-lock.txt -c 13_env/constraints.txt
qsub -o 12_logs/india_confirm.out hpc/india_confirm.pbs     # on the cluster, at tag prereg-india-v1 or later
```

The Indian data come from `../1_india_data_forecasting_scheduling/data/processed/`. Gradient boosting is deterministic for a fixed thread
count; GPU-trained networks are not bit-for-bit reproducible, so a rerun can differ in the last digits (all
reported results are means over seeds).
