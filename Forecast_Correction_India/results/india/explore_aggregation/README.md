# Exploratory: online combination of the gate designs (not pre-registered)

Script `code/scripts/india/explore_aggregation.py`, H100 job 34927 (10 Oct 2026), log
`logs/explore_aggregation.out`. Stored per-row outputs of the hourly ladders only (IH1); nothing refitted.

Question: the validation-only rules (PART, prequential validation) could not tell in advance which gate design would
pay on the hourly series. Can an online combiner, which reweights the 14 corrected forecasts (no correction and the
13 designs) day by day from errors published two days earlier, follow whichever design is paying?

Protocol: the learning rate and share parameter were chosen on the development split (results/india/dev/IH1, one
seed) and written to `choice.json` before any test file was read (same job, phase dev first). Frozen: EWA (eta 1),
Fixed Share (eta 20, alpha 0.05), and LEAD30 (the design with the lowest error over the last 30 published days).
Weights start equal at the start of each evaluation file (the per-row outputs hold only the test period).

## Results (test months 1 Mar - 13 Sep 2026; skill = 1 - SSE / SSE of the uncorrected forecast; mean of 3 seeds)

Mean over the 32 forecast x series pairs (`skill_confirm.csv`):

| Method | Mean skill |
|---|---|
| Best single design, chosen with hindsight per pair | 0.074 |
| Fixed Share combination (frozen) | 0.069 |
| EWA combination (frozen) | 0.061 |
| LEAD30 (frozen) | 0.057 |
| Fixed Share over baseline and correction (best single design on average) | 0.053 |
| Per-series gate updated daily | 0.043 |
| Static per-series gate | 0.017 |

The Fixed Share combination beat the daily-updated per-series gate in 24 of 32 pairs and the static per-series gate
in 25 of 32. For weekly persistence it matched the hindsight-best design (demand 0.330 against 0.319, wind 0.332
against 0.334). For N-HiTS and Chronos-Bolt it changed the error by -0.8 % to +1.3 %, less harm than the static gate
(down to -3.1 %), but it did not help them.

## Reading

Online combination solved the selection problem that the validation-only rules could not, for the forecasts that had
something to gain; for strong forecasts, where no design helped, it also gained nothing. The pairs share series and
test months, so the 24/32 count is descriptive, not a test. This is exploratory evidence on an already-opened period;
a prospective check on data published after 13 Sep 2026 is needed before it can be claimed as a finding.
