# Confirmatory audit (13 Sep 2026, after CONFIRM_CHAIN_DONE 19:08:59)

## 1. Execution integrity

| Check | Result |
|---|---|
| Code version | chain started at commit `b5ef80d` = tag `prereg-v1`; guard unlocked only with hash + tag + env flag |
| Pre-registration hash | `a64981ba…0ebbae3`, unchanged |
| Runs | learned baselines ✓; ladders 67/67 (C1 27, C2 12, C3 6, C4 6, C5 6, C6 4, C7 6), no skips, no reruns; X5/X8 3 seeds ✓; X1 partition 576/576 ✓; X1 instance 72/72 ✓; analysis ✓ |
| Crashes / deviations | none; §9 of the pre-registration remains "none" |
| Results tag | `confirm-v1` |

## 2. Independent recomputation (outside `analyze_confirmatory.py`)

Recomputed from the X5/X8 per-row files (NYISO, 2026-01-01 … 2026-08-31, 243 days, 11 zones, 64,108 rows):

| Quantity | Analysis script | Independent |
|---|---|---|
| Deviation energy, ISO → corrected (GWh/yr) | −853.7 | 6,828 → 5,974 (−854) |
| MAE (MW) | — | 70.9 → 62.0 |
| 99 % reserve, zone × hour (MW, sum of zones) | −234.3 | 2,815 → 2,580 (−235) |
| Coverage of that reserve | 0.9934 / 0.9921 | 0.9934 / 0.9921 |
| Two-settlement imbalance cost (M$/yr) | −32.1 saved | 40.4 → 72.5 (+32.1) |

All agree. Note: the CI of H8b is narrow by construction — both reserve schedules are static quantile tables, so the
daily difference in MW is almost constant; its reliability check is the coverage condition, which holds.

## 3. Exploratory analysis (not pre-registered) — why imbalance cost rose

* The corrector removes a systematic under-forecast (ISO mean error +27.0 MW; mean correction +26.6 MW, negative in only
  14 % of rows), so the corrected load-serving entity buys more energy day-ahead.
* In 2026 the real-time price was on average below the day-ahead price (mean RT − DA spread −4.76 $/MWh, sd 67.4), a
  day-ahead premium. Moving purchases from real time to day-ahead therefore costs money on average.
* The effect is concentrated: 94 % of the annualised increase (+30.1 of +32.1 M$/yr) comes from January 2026, and
  60 % (+19.2 M$/yr) from the two days of the cold-weather event of 27–28 January (day-ahead prices exceeding
  real-time prices by up to 760–840 $/MWh in ten of eleven zones, 419 $/MWh in LONGIL); hours with |spread| above the
  90th percentile (35.5 $/MWh) contribute 81 %.
  *Erratum (13 Sep 2026):* the first version of this bullet, computed ad hoc on one seed, labelled the event share
  (+20.0 M$/yr, 60–62 %) as the January share and stated a tail share of 88 %. The scripted, seed-averaged
  recomputation `04_code/scripts/confirm/exploratory_imbalance.py` (output `imbalance_decomposition.json`) gives the
  figures above, which are the ones used in the manuscript. Per-seed ranges: event share 0.57–0.63, tail share 0.80–0.82.
* Interpretation: under a two-settlement market with a day-ahead premium and spike risk, the accuracy-optimal (unbiased)
  forecast is not the cost-optimal purchase quantity; a cost-aware gate or bid would target a spread-dependent
  quantile. This is reported as an exploratory finding and a limitation of accuracy-based gating, with no claim that
  a cost-aware rule would have done better on these data.
* Output: `06_results/confirm/exploratory/imbalance_decomposition.parquet` and `.json` (script
  `04_code/scripts/confirm/exploratory_imbalance.py`).

## 4. Verdict summary (from `06_results/confirm/analysis/verdicts.json`)

Supported 19 of 23 tests with a verdict (plus one estimation-only quantity):
H1 partition (0.928, 0.787) · H2 NYISO, OPSD, pooled · H3a, H3b, H3c · H4a ×2, H4b · H5 · H6a sign, H6a vs always, H6b ·
H7a, H7b · H8a, H8b.
Not supported: H1 instance oracle (0.727 < 0.85), H1 instance PART (0.432 < 0.75; expected), H2 UCI (p = 0.14),
H6a regret vs never-discount (p = 0.37).
Tests stated in advance as at risk: H1-instance PART (failed, as expected); H3b, H3c, H7a, H7b (all supported).

## 5. Items the manuscript must state

1. The instance-axis crossing condition over-predicts gains for regularised model gates and its sign accuracy (0.73)
   misses the threshold; the principle is confirmed for partition axes, and only qualitatively (T4b) for model gates.
2. Prequential validation beats always-discounting but not never-discounting: in the confirmatory units discounting
   rarely paid, so "never discount" was already near-optimal; the rule avoids the losses of always-discounting.
3. UCI alone does not confirm the mechanism (wide CI); the pooled effect is positive with I² = 0.69.
4. Accuracy gains did not reduce two-settlement imbalance cost (exploratory explanation above).

## 6. Sensitivity analyses after unblinding (DC1, DC1b)

* **DC1** (OPSD real-data blocks, plausibility filter from 2015–2017 data): no verdict changed; OPSD estimates
  strengthened; two descriptive statements about the European areas reversed (per-instance vs per-series now a tie;
  OOF decline of the OPSD-2019 budget ladder disappears).
* **DC1b** (semi-synthetic grid rebuilt on the filtered 12-area substrate; triggered because the DC1 entry had wrongly
  stated that the grid was unaffected — ME and SE_1 carry 101 flagged values in 2018–2019): an unfiltered re-run of 24
  configurations reproduced the frozen results exactly, so the filter causes the differences. H1 partition sign accuracy
  fell from 0.928 to 0.848 (threshold 0.85) and PART from 0.787 to 0.746 (threshold 0.75): **both partition criteria of
  H1 are not supported in the sensitivity analysis.** Errors concentrate near the decision boundary (|predicted gain| <
  ΔV). The primary pre-registered verdicts stand as the confirmatory result, but the manuscripts must present the
  quantitative H1 support as fragile.
