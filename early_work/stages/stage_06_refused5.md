# RE-FUSED-5 — rebuild and reposition (12 September 2026)

**Folder then:** `REFUSED5` (433 files, 399 MB)
**Kept now:** `history/preserved/stage5_6_refused5/` (documents, code, results and the small parquet files)

## Objective

Two things at once: find out what the residual-correction literature already does, and rebuild the data so that
a claim could be measured at all.

## Research question

Is there anything left to claim about correcting a forecast behind a gate, once selective prediction, expert
aggregation, active feature acquisition and classical shrinkage are accounted for?

## What was done

### 1. A novelty map

Six clusters of prior work were read and written up with what each one solves and what it leaves open:
residual and post-hoc correction; selective prediction and risk control; expert aggregation (sleeping experts,
fixed share); value of information and active feature acquisition; classical shrinkage and combination; and
supporting empirical facts. The rule was stated explicitly: a component is rejected if it is a renamed version
of anything in the map.

The gap that survived: a **conditional, horizon-resolved estimate of the marginal value of applying a
correction**, calibrated, decision-usable under temporal shift, and explained by a closed-form threshold.

### 2. A data audit of the RE-FUSED-4 panel

| Finding | Value |
|---|---|
| Panel price identical across all 18 States | 100 % of 2,627 dates |
| Effective price sample | 2,627 day-level values, not 47,286 rows |
| Price imputed in test | 98.21 %, and 78.01 % of test rows sit at the Rs 10,000 cap |
| Panel price against the raw IEX daily mean | correlation 0.649, MAE Rs 1,207, exact match 9.49 % |
| Raw observed blocks per day | 94 in 2015 falling to 6 in 2023 |

This confirmed RE-FUSED-4 independently, from the raw archive rather than from the panel.

### 3. Rebuilt datasets

- **India daily panel**: 799 balanced units, 20 States, no imputation, strict chronological split.
  Baseline ladder on the test period (mean MASE over States): persistence **0.502** < rolling-7 0.675 <
  seasonal naive 0.955 < operator schedule 1.305.
- **NYISO**: 84 months of 5-minute real-time and day-ahead zonal prices, 15 zones, resampled to 5 min, 15 min,
  1 hour and 1 day from a single source — a testbed where a price target is real.

## Problems discovered later

RE-FUSED-6 audited this stage and found seven reproducible defects, including invalid pooling across different
experimental units, a duplicate baseline at one horizon, a confounded resolution comparison, and gate clipping
applied to some arms but not others. They are listed in
`history/preserved/stage5_6_refused5/docs/07_refused6_diagnosis_and_plan.md`.

## What was retained

The novelty map, the data audit and the baseline ladder. The NYISO idea survived into RE-FUSED-7 with its own
fetch script and its own documentation.

Dropped: the bulk NYISO parquet files (about 365 MB), because the gated-correction programme fetches and stores
its own copy and the fetch script is kept.

## Status

Superseded by RE-FUSED-6 and RE-FUSED-7. The documents remain useful: the novelty map is still the clearest statement
of what the gated-correction programme may and may not claim.
