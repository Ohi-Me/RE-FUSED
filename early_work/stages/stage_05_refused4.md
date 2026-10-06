# RE-FUSED-4 — the audit that explained everything (21 August 2026)

**Folder then:** `Refused4` (77 files)
**Kept now:** `history/preserved/stage1_4_releases/refused4/`

## Objective

Explain why persistence kept winning on the price target, instead of assuming the models were at fault.

## What was done

An imputation-mechanism audit was written into the notebook, and a per-cell execution report was produced.

## The finding

**98.2 % of test-period market prices are last-observation-carried-forward** — the same value repeated from an
earlier day. The median run is 31 days and the longest is 423. In the fan chart, the "actual" line sits flat at
Rs 10,000 for 100 days.

Consequence: price and PA-LMP skill **cannot be measured on that panel**. Persistence wins by construction, and
every price result in RE-FUSED-0 to RE-FUSED-3 is unmeasurable rather than wrong.

## Why it matters

It ended a line of work that had run for months, and it is the reason the India programme was later rebuilt
from official Grid-India, CEA and IMD reports, with the publication delay of every value recorded. RE-FUSED-5
repeated the audit independently and reached the same conclusion from the raw IEX archive.

It is also the best version of the early line to cite: it has RE-FUSED-1's model results plus the most complete
evidence about their limits.

## Status

Superseded as a result; decisive as a diagnosis.
