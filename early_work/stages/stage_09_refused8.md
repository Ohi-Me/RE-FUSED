# RE-FUSED-8 — the India study (14 to 21 September 2026)

**Now:** live, as [`india_grid/`](../../india_grid/)

## Objective

Answer five questions on data that can be trusted: build the dataset, forecast under publication delay, combine
models with calibrated uncertainty, assess deviations by their consequence, and schedule under the deviation
settlement.

## Why it exists

Every earlier attempt at the India problem used a panel with an imputed price column and no record of when each
value became public. This stage rebuilt everything from official Government of India reports, and recorded the
publication delay of every column.

## How it ran

| Phase | Date | What happened |
|---|---|---|
| Alignment audit and design | 14 Sep | Earlier stages compared; sources verified |
| Data acquisition and panel | 14 Sep | Nine official products downloaded with manifests; the panel built, audited and checksummed |
| Development round 1 | 15–17 Sep | Forecasting models, the LA-MCAG gating model, the assessment and scheduling protocols |
| Move to the cluster | 17 Sep | A self-contained copy on the NIT Jalandhar H100. From here everything runs inside PBS jobs |
| Development round 2 | 17–18 Sep | Added models and the stacked combination, to close the gaps round 1 left |
| Pre-registration frozen | 18 Sep | Job 31823 recorded the hash of the document, of 42 choice files and of the scoring script |
| Confirmatory run | 20 Sep | All 29 stages on the reserved period, scored once: 35 of 52 supported |
| Release and extra analyses | 21 Sep | The Zenodo record and the additional (not pre-registered) analyses |

## Results

See [`docs/RESULTS.md`](../../docs/RESULTS.md). In short: the dataset works and is published; forecasting under
real publication delay works and the delay is expensive; context gating helps unevenly; the consequence-aware
assessment does not beat the energy-only measure; risk-aware scheduling beats what is published today, but no
adaptive rule beats a well-tuned fixed one; and changing only the forecast does improve the schedule.

## The laptop stage

`REFUSED8` was the laptop-era copy. Its code and data were carried to the cluster copy, which then became the only
working folder on 18 September. What is kept from it in `early_work/stage8_laptop_refused8/` is only what does not
exist in the cluster copy: the early status and audit documents and the acquisition logs. Its round-1 development
results were superseded by the cluster runs, which are the ones the pre-registration hashes.
