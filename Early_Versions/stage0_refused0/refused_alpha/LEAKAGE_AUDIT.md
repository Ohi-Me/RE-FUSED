# RE-FUSED-Alpha — Leakage & Data-Integrity Audit

Scope: `notebook/REFUSED_Alpha_Final.ipynb` + `data/{full_dataset,RL_TRAIN_DATA,RL_TEST_DATA}.csv`.
Method: static read-only audit (the notebook could not be executed in this session).
Evidence lines refer to positions inside the minified `.ipynb`.

---

## 0. Executive verdict

| Area | Status | Severity |
|---|---|---|
| Model-input standardization (`fm/fs`, `tm/ts`) | Train-only — **correct** | — |
| Alpha features recomputed in `prep()` per split | Causal — **correct** | — |
| `price_norm`, `lcmp_carbon`, GSI-family inputs baked upstream | **Leaks test info** | **HIGH** |
| Median imputation `fillna(df[nc].median())` per split | Test uses test median | MEDIUM |
| 4 of 6 forecast targets are synthetic composites | Circular / no ground truth | **HIGH** |
| Daily data vs. 5-minute / intraday / 4-hour claims | Resolution mismatch | **HIGH (reviewer-killer)** |
| MAPE on near-zero targets + arbitrary `weighted_score` | Unstable / unjustified metric | **HIGH** |
| Unit labels (`*_mwh`, `*_tons`) | Off by ~1000× | MEDIUM |
| Single seed, no significance test, no strong baselines | Not A*-grade rigor | **HIGH** |

---

## 1. What is CORRECT (keep as-is)

- **Chronological split** into `RL_TRAIN_DATA.csv` (2017–2023) and `RL_TEST_DATA.csv` (2024–2025) — no random shuffling. Good for time series.
- **Train-only scaling** (notebook line ~102):
  `fm = df_train[FEATS].mean(); fs = df_train[FEATS].std().replace(0,1)` and
  `tm = df_train[TARGETS].mean(); ts = df_train[TARGETS].std()`. Applied to both splits. This is the textbook-correct pattern.
- **`prep()` applied separately** to `df_train` and `df_test` — the N3 GCAL carbon price and the Alpha features are recomputed inside each split, so those do not leak across the boundary.
- **`rolling_cvar4hr`** = per-state, date-sorted, `rolling(6, min_periods=1)` — causal/backward-looking. (Mislabeled; see §4.)

## 2. What LEAKS (fix — Task #1)

These columns are **model inputs** (they appear in `FEATS`, notebook line ~102) yet were produced by an **upstream** script over the full 47,286-row panel. Train-only re-standardization cannot remove a leak already encoded in the value.

| Column | Formula (per `Dataset_and_Features.md`) | Leak mechanism | Fix |
|---|---|---|---|
| `price_norm` | min-max of `avg_market_price` | global min/max include 2024–25 test extremes | fit min/max on **train only**, apply to test |
| `lcmp_carbon` | `(CI-CI_5th)/(CI_95th-CI_5th)·price·0.30` | `CI_5th`,`CI_95th` are **global percentiles** | fit 5th/95th pct on **train only** |
| `grid_stress_index` | composite of outage/SD-gap/coal, `[0,1]` | component normalizers likely global | refit component min-max on train only |
| `discom_stress` | `0.5·price_norm + 0.5·GSI` | inherits leak from the two above | recompute after fixing inputs |
| (screen) `gen_zscore`,`price_zscore`,`*_cvar90`,`state_gen_rank`,`state_gen_tier`,`state_*_norm` | z-score / percentile / cross-state rank | global stats | **not currently in `FEATS`** → verify they never enter any model/RL input; if used, refit train-only |

**Also (MEDIUM):** `prep()` uses `df[nc].fillna(df[nc].median())`. Called on `df_test`, the median is the **test** median. Fix: compute medians on `df_train`, pass them into `prep(df_test, medians=train_medians)`.

Drop-in fix: `refused_fixes/causal_features.py` (fit on train, transform any split).
Diagnostic you can run today: `causal_features.leakage_scan(df_train, df_test)`.

## 3. Circular / synthetic targets (Task #3 + narrative)

`TARGETS = ['total_generation_mwh','avg_market_price','grid_stress_index','fcfs_priority_score','palmp','log_carbon']`.

- **Observable (defensible):** `total_generation_mwh`, `avg_market_price`.
- **Synthetic composites of the model's own inputs (circular):** `grid_stress_index`, `fcfs_priority_score`, `palmp`, `log_carbon`.
  `palmp` is a deterministic weighted sum of `price, CI, GSI, FCFS`; forecasting it while feeding (lags of) those same signals is near-tautological and has **no real-world ground truth** to validate against.

Recommendation: report forecast skill on the **observable** targets as the headline; present `palmp`/GSI as **derived policy overlays** with sensitivity analysis, not as forecasting targets with a MAPE.

## 4. Resolution mismatch (Task #4 — the reviewer-killer)

Data is **daily state-day**. Yet the notebook/README claim: 5-minute settlement `pa_lmp_t`, intraday CFC from a "5-min CI stream", 4-hour rolling CVaR, and Directional Accuracy as "5-min settlement quality". The `rolling_cvar4hr` is literally a **6-day** window. On daily data these are unfalsifiable.
Fix: either obtain sub-hourly data (IEX RTM 15-min, Grid-India SCADA) or relabel every sub-daily artifact as a **daily-resolution counterfactual simulation** and delete the 5-min/intraday claims.

## 5. Metrics (Task #3)

- `weighted_score = 0.35·palmp_MAPE + 0.20·GSI_MAPE + 0.20·price_MAPE + 0.15·logcarbon_MAPE + 0.10·gen_MAPE` — arbitrary weights across heterogeneous targets, dominated by the synthetic `palmp`.
- **MAPE explodes** on near-zero targets (`renewable_generation` = 0.58; momentum/gap signals cross zero).
- Fix: `refused_fixes/metrics.py` → **MASE / RMSSE** (scale-free, zero-safe), **pinball + CRPS** for the probabilistic upgrade, **Diebold–Mariano** for significance, empirical **coverage** for calibration.

## 6. Units (Task #2)

`total_generation_mwh ≈ 70` for a whole state-day is impossible in MWh (Bihar ≈ 150 GWh/day). Values are **GWh** mislabeled `_mwh`; `carbon_proxy_tons` is consistent only as **kilotons**. Numerically harmless for ratios, but a credibility hit in a paper. Fix: rename columns / add an explicit units table; report `*_imputed_flag` rates and add a robustness run excluding imputed rows.

---

## Fix order (as prioritized)
1. **Leakage/causality** → `refused_fixes/causal_features.py` + `leakage_scan`.
2. **Units/integrity** → rename + units table + imputation report.
3. **Probabilistic + rigor** → `refused_fixes/metrics.py` + baselines + multi-seed + DM tests.

---

## 7. Measured results (diagnostics executed 2026-07-19, follow-up session)

The checks promised above were actually run (`py -3.10 run_audit_diagnostics.py`) on
`RL_TRAIN_DATA.csv` (39,672 rows, 2017-10-03 → 2023-12-31) vs `RL_TEST_DATA.csv`
(7,614 rows, 2024-01-01 → 2025-04-22). Evidence files:
`results/tables/{leakage_scan,units_table,imputation_report}.csv` and
`results/outputs/audit_diagnostics_summary.txt`.

### Leakage scan (→ §2), measured
- **`grid_stress_index` leak CONFIRMED**: refitting its component min-max on train
  only changes test-split values by up to **0.44** on a [0,1] feature — the shipped
  values were built with non-train statistics. Apply `CausalFeatureFitter` before
  building `FEATS`.
- **`price_norm`: leaky mechanism, nil effect** — the train-only refit reproduces the
  shipped values exactly (max |Δ| = 0.0000) because 2024–25 prices never exceeded the
  per-state train extremes. Keep the fix anyway; it protects any future data refresh.
- `gen_cvar90`: 2.7% of test rows lie outside the train range (distribution shift —
  verify it never enters any model/RL input).
- `lcmp_carbon` / `discom_stress` are not shipped in the CSVs (recomputed in `prep()`),
  so the scan skips them; the fitter covers them going forward.
- All other suspects (z-scores, `state_*_norm`, ranks/tiers): ≤0.4% out-of-train-range,
  zero refit delta.

### Units (→ §6), measured
- Median state-day `total_generation_mwh` = **135.0** → physically plausible only as
  **GWh**; the `_mwh` mislabel is confirmed.
- 43/133 numeric columns trip the unit-jump heuristic, but most flags reflect
  **cross-state size heterogeneity** (~600× between small and large states), not a
  within-column unit switch — the decisive evidence is the magnitude check above.
- Fix shipped: `refused_fixes.units.RENAME_MAP` + `rename_to_true_units()`
  (label-only rename to `_gwh` / `_kilotons`; values untouched).

### Imputation (→ §6), measured — worse than the static audit assumed
- **87.3% of all state-day rows carry at least one imputed value**
  (`market_price`, `coal`, `consumption` each 87.3%; `renewable` 0%).
- The observed-only robustness slice is only **5,996 / 47,286 rows (12.7%)**.
  This must be disclosed; re-run headline metrics on `units.observed_only(df)`.

### Session close-out (updated 2026-07-20)
- Done: `refused_fixes/` is a proper package (`__init__.py`); `units.py` registry
  rewritten to the real schema; diagnostics executed and archived under `results/`.
- **Done — notebook patched (35 → 36 cells, all code cells verified to compile):**
  - New **Cell 1.1b** wires `CausalFeatureFitter` (fit on train, transform both
    splits) in *before* `prep()`/`FEATS` — fixes the §2 HIGH leak
    (`grid_stress_index` Δ up to 0.44) plus `price_norm`/`discom_stress`.
  - `prep()` now freezes **all** statistics on the train split via a `ref=` frame:
    median imputation, quantiles, and min-max — fixes the §2 MEDIUM item.
  - `fit_eval` reports **MASE / RMSSE / sMAPE** per target plus a
    `headline_mase_observable` aggregate over the observable targets only (§3, §5);
    `weighted_score` is retained but demoted to **LEGACY (V2 comparability only)**.
  - Stage-3 comparison cell (3d) adds **Diebold–Mariano (HLN)** winner-vs-challenger
    significance tests (§5).
  - Cell 7.1 exports MASE alongside the legacy metrics in
    `results/tables/metrics_summary.json`.
- **Done — reference baselines (§5 rigor):** `refused_fixes/baselines.py` +
  `run_baselines.py` → `results/tables/baseline_metrics.csv`
  (naive / seasonal-naive m=7 / moving-average on the true test split).
  Key anchors: `total_generation_mwh` naive MASE ≈ 0.494 (beat ~0.49 to claim skill);
  `avg_market_price` naive MASE ≈ 0.075 — near-trivially persistent, consistent with
  the 87.3% imputation smoothing, so price-skill claims must beat *that*, not MAPE.
- **Done — narrative relabels (§4):** README scrubbed of all 5-min / intraday /
  sub-daily claims; N4 renamed "Settlement Counterfactual (daily resolution)";
  legacy column names flagged as label-only; "Rigor & Integrity" section added.
- Still open (requires a full GPU training re-run, not doable statically):
  execute the patched notebook end-to-end; **multi-seed (≥5) mean±std** + DM on the
  final head-to-head; observed-only robustness pass (`units.observed_only`, 12.7%
  slice); regenerate figures/tables from the fixed pipeline.
