# RE-FUSED-6 four-perspective reviewer audit (round 1, 13 Sep 2026, 01:33)

Evidence base: M1 (frozen), M1b–M1d, M3/M5, M4, M5b, M6, M8, M7, theory (docs/09_theory.md).
Pending at time of writing: M5c (pre-registered parsimony selection), M7b (pre-registered hybrid safety).

Addressable: **yes** (experiment/analysis can fix) · **partly** · **no** (cannot be fixed without

---

## Reviewer 1 — skeptical NeurIPS / ICML / ICLR

| # | Criticism | Sev | Addressable | Response / action | Status |
|---|---|---|---|---|---|
| 1.2 | Theory is elementary; no finite-sample result for a boosted per-instance gate with data-dependent degrees of freedom | Ma | partly | Stated as a limitation; proving it is out of scope tonight | open (limitation) |
| 1.3 | The "fine" gate is a gradient-boosted ratio/WLS estimator, but the field's default is a **neural sigmoid gate trained end-to-end**. The dominance claim may not transfer to that gate family | **Ma** | **yes** | **M9**: bounded neural gate g = 1.5·sigmoid(MLP(x, series one-hot)), trained on V1 with early stopping on V2, compared with per-series gates on India and ETT | **queued** |
| 1.4 | Effect sizes are small (coarse vs fine 0.01–0.06 skill; nested vs fixed per-cell a tie) | Mo | partly | All effects reported with paired tests; the tie is reported as a tie | addressed |
| 1.5 | Synthetic generator is linear-in-features with multiplicative cell amplitudes | Mo | partly | Real-data replication across 6 datasets (M8) is the stronger evidence; synthetic used only for mechanism | addressed |

## Reviewer 2 — time-series / forecasting

| # | Criticism | Sev | Addressable | Response / action | Status |
|---|---|---|---|---|---|
| 2.1 | Single chronological split for India (test 2024–25). Are the ladder conclusions a property of one test period? | **Ma** | **yes** | **M10**: rolling-origin re-evaluation with two earlier origins (test 2022; test 2023), same protocol and seeds | **queued** |
| 2.2 | Deep baselines only on ETTh1 (M4); correction of deep baselines is mostly negative, so the method's practical scope is weak or mid-strength baselines | Ma | partly | Scope stated explicitly: "correction pays when the baseline leaves systematic residual; with a strong deep baseline it does not". M4 internal control rules out corrector starvation (D12) | addressed as scope |
| 2.3 | No probabilistic evaluation (CRPS, calibration, interval coverage) | Mo | yes (large) | Not attempted tonight; listed as future work — the paper's claim is about point correction | open (scope) |
| 2.4 | Skill scores only; no MASE/MAE tables for comparability with the forecasting literature | Mi | yes | MAE skill is logged per run (M1); add MASE in the paper tables | open (write-up) |

## Reviewer 3 — energy / power systems

| # | Criticism | Sev | Addressable | Response / action | Status |
|---|---|---|---|---|---|
| 3.1 | The Indian target (total generation of a balanced unit panel) has limited operational meaning; price was dropped | Ma | partly | Price dropped for a documented data-integrity reason (identical across 18 states on 100% of dates; 98.2% carry-forward). The data audit itself is a contribution for this community | addressed by framing |
| 3.2 | The operator schedule is a dispatch program, not a forecast; calling it a baseline is loose | Mo | yes | Reframed as "a biased, drifting operational reference" — the case Proposition 7 describes | addressed in wording |
| 3.3 | RE-FUSED-5's "5-minute settlement creates forecasting value" claim | Ma | yes | Tested and **falsified** in M6 under matched conditions; replaced by the accurate statement (weaker baselines leave more to recover; the day-ahead price is uncorrectable at its own 24 h horizon) | addressed |
| 3.4 | Only 5 matched NYISO zones in M6 | Mi | partly | Matching is the point of M6; the unmatched 15-zone run is retained as a confounded comparison | addressed |

## Reviewer 4 — statistical learning theory

| # | Criticism | Sev | Addressable | Response / action | Status |
|---|---|---|---|---|---|
| 4.1 | Paired tests across (config, seed) treat seeds as replicates, but seeds share the same time series — they capture model randomness, not sampling variability over time | **Ma** | partly | Block bootstrap over time used in M1; time-block LCB used for deployment and parsimony selection; M10 adds temporal replication across test years. Remaining inference is stated with its unit | partly addressed (+ M10) |
| 4.2 | Many tests across stages; multiplicity not controlled uniformly | Ma | yes | Holm applied in M1 audit and M8; the paper's headline claims will be listed with their family and correction | open (write-up) |
| 4.3 | Proposition 3's variance is asymptotic and covers partitions; the per-instance gate is not a partition | Mo | no (tonight) | Stated in theory doc | limitation |
| 4.4 | Unimodality holds in only 82.4% of synthetic runs | Mo | no | Reported as a measured property; Proposition 4 states the assumptions that fail in the rest | addressed |
| 4.5 | "Pre-registration" is timestamped in an internal file, not externally registered | Mi | no | Honest wording: "specified before results in the version-controlled lineage" | addressed in wording |

---

## Decision from round 1

* **Fatal for NeurIPS/ICML/ICLR main track and not addressable by experiments:** 1.1 (novelty). The top-tier
  general-ML claim is therefore **stopped**, per the stopping rule in the brief: the remaining gap cannot be
  closed without manufacturing novelty.
* **Major and addressable tonight:** 1.3 (neural-gate family) → **M9**; 2.1 (single test period) → **M10**.
  Both answer a concrete objection and neither tunes anything on test data.
* Everything else is either addressed, a scope statement, or a write-up task.

Round 2 of this audit will be appended after M5c, M7b, M9 and M10.

---

# Round 2 (13 Sep 2026, 08:10) — after M5c, M7b, M9, M9b, M10, M10b

## Status of round-1 items

| # | Round-1 criticism | Action | Result | Status |
|---|---|---|---|---|
| 1.3 | fine gate is boosted, field default is a neural sigmoid gate | M9 (pre-registered) + M9b (tuned grid) | refit neural − per-series −0.0054 (16/50, p=3e-5); wins 2/10 configs; M9b tuned grid −0.0046 (16/50, p=1.2e-3), wins 3/10 | **closed** |
| 2.1 | single India test period | M10 rolling origin (pre-registered) | instance rung below best coarse in 6/8, 7/8, 6/8 configs across test years 2022, 2023, 2024-25 | **closed**, with a new caveat (2.5) |
| 4.1 | seeds are not temporal replicates | M10 adds three test years | "fine loses" holds in every year; the coarse-rung ordering does not | **partly closed** |

## New criticisms raised by round-2 evidence

| # | Criticism | Sev | Addressable | Response / action | Status |
|---|---|---|---|---|---|
| 2.5 | M1's reversal of RE-FUSED-5 ("per-series beats shrunk") is period-specific: shrunk wins 5/8 configs in the 2022 test year | **Ma** | yes (by claim change) | claim downgraded in docs/08 and docs/12: the stable statement is "per-instance loses; the best coarse rung is a plateau whose member varies by dataset and period" | **addressed** |
| 1.6 | SCALE's defining step (estimate the adaptivity level from validation) failed on real data; the paper cannot present rung selection as the method | **Ma** | yes (by claim change) | M5c falsified, M10b shows selection ≈ fixed default on fresh years; selection removed from the final procedure; Proposition 8 states when selection cannot pay | **addressed** |
| 3.5 | clip90 looked best on stationary India (M7) but destroys NYISO gains on fresh data (M7b) — was M7 over-read? | Ma | yes | M7b secondary analysis reported; clip90 falsified out of sample; recommendation is block-LCB | **addressed** |
| 3.6 | block-LCB's advantage under drift is only measured in-sample on India | Mo | partly | fresh drifting baselines are not available in the collected data; stated as a limitation | **limitation** |
| 1.7 | M9 amendment (refit variant) was added after a 9-run smoke test that overlaps the full run | Mi | no | disclosed in docs/08; the amendment strengthens the competitor, so it can only make H-M9 harder to pass; M9b re-tests with a tuned grid | **disclosed** |
| 4.6 | M10b fresh-year margins are small (0.001–0.003); "selection does not pay" is not "selection hurts" | Mo | no | wording fixed: fixed default recommended because it is simpler and never worse on fresh years, not because selection is harmful | **addressed in wording** |
| 4.7 | numbers quoted in docs drifted from the CSVs (D17) | Mi | yes | every number in the final report is taken from ALL_STATS.md; D17 fixed | **fixed** |

## Stopping decision (round 2)

* All **major addressable** items from rounds 1 and 2 are closed by experiment or by an explicit claim change.
* Remaining items are either limitations that need data the project does not have (fresh drifting baselines,
* Any further experiment would test a new hypothesis, not repair a weakness in the current claims. Per the
  brief's stopping rule, the loop stops here and the final package is produced (docs/12).
