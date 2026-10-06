# RE-FUSED-7 — the gated-correction study (13 September 2026)

**Now:** live, as [`2_forecast_correction_india/`](../../2_forecast_correction_india/)

## Objective

Take RE-FUSED-6's finding and test it properly: hypotheses frozen first, then scored once on data that were not
used during development.

## Research question

When does refining a forecast-correction gate — per series, per state, per instance, over time — actually pay?

## Method

1. A critical audit of RE-FUSED-6.
2. A research design and a pre-registration, frozen with a recorded SHA-256 and the git tag `prereg-v1`.
3. A confirmatory battery. Indian data were not yet available in the needed form, so this first test used
   public data from other systems: NYISO zonal load (January to August 2026), 46 European TSO forecast areas, 60
   UCI electricity clients, and a fresh semi-synthetic grid. These results are now kept as a comparison in
   [`2_forecast_correction_india/comparison_other_countries/`](../../2_forecast_correction_india/comparison_other_countries/).
4. Two logged deviations after unblinding (DC1 and DC1b), reported rather than hidden.
5. A full rebuild from frozen results that reproduced 520 of 520 file hashes.

## Results

- **19 of 23 pre-registered tests supported** (tag `confirm-v1`).
- Refining pays only when the variation it exploits is large, persists from fitting to use, and can be estimated
  from the available data. Two validation-only ways of checking this were built: PART for partitions and
  prequential validation for time policies.
- On NYISO, a coarse per-zone correction cut deviation energy by about 854 GWh per year and 99 % upward reserve
  by 234 MW at almost unchanged reliability (99.2 % against 99.3 %), but did **not** lower two-settlement
  imbalance cost.
- Known weak point: under DC1b the two partition criteria of H1 land just below their thresholds (0.848 and
  0.746 against 0.85 and 0.75). This is reported, not hidden.

## What it gave the rest of the work

- The working method reused everywhere since: pre-registration with hashes, a confirmatory battery, a map from
  every claim to a result file, and an automated check that fails if a number drifts.
- The same question was then asked on Indian data: first on the State-day panel (family F5 of the India study),
  then on the all-India hourly series with its own pre-registration (`prereg-india-v1`).
