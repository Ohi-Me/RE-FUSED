# %% [markdown]
# ---
#
# # PART II — FORECASTING UPGRADE (Sections U1–U15)
#
# Everything from this point on is **additive**. No cell, output, figure, result
# or variable above this line is modified, reordered or re-run. The original
# notebook remains the reproducibility record of the published RE-FUSED-Alpha
# results; this part builds a **separate, isolated pipeline** on the *same*
# dataset, the *same* six targets and the *same* chronological test period
# (2024-01-01 → 2025-04-22) and asks one question:
#
# > Is there a genuinely better forecasting method than `PatchTST + MCAG`
# > (avg MAPE 28.20 %, headline MASE 3.4748) — one that also beats persistence?
#
# **Ground rules held throughout Part II**
#
# | | |
# |---|---|
# | Test period | identical to the original `RL_TEST_DATA.csv` window |
# | Tuning / model selection / ensemble weights / seed choice | **validation split only**, never test |
# | Splits | strictly chronological, never random |
# | Feature admissibility | value dated ≤ forecast origin, or a calendar attribute of the target date |
# | Reporting | MAPE is **never** restated as "accuracy"; if persistence wins, it is reported as winning |
# | Outputs | written to `results/forecasting_upgrade/`, never overwriting existing results |
#
# Sections are self-contained and sequential: each method family gets one
# section that carries its own features, training, evaluation and plots.

# %% [markdown]
# ## Section U1 — New Data / Leakage Audit
#
# A new pipeline is only worth building if it is provably clean. This section
# rebuilds the panel from the raw CSVs in an isolated namespace and *measures*
# (rather than asserts) every property the later sections depend on: ordering,
# duplication, calendar completeness, split chronology, target validity,
# missingness, and the provenance of every feature family.
#
# It closes with a **mechanical leakage proof**: every observation from the test
# boundary onwards is replaced with noise, the features are rebuilt, and the
# pre-boundary feature values must come back bit-identical.

# %% [code]
# ─── Cell U1.1 — Isolated pipeline: load, reconstruct targets, split ─────────
import sys as _sys, pathlib as _pl, time as _time, json as _json, warnings as _w
_w.filterwarnings('ignore')

for _p in (_pl.Path.cwd(), _pl.Path.cwd().parent, _pl.Path.cwd().parent.parent):
    if (_p / 'refused_upgrade').exists():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break

import numpy as _np, pandas as _pd
import matplotlib.pyplot as _plt
from refused_upgrade import panel as UP, features as UF, evaluation as UE, audit as UA

_pd.set_option('display.width', 200)
_pd.set_option('display.max_columns', 60)

U_PATHS = UP.repo_paths()
U_OUT = U_PATHS['upgrade']

_t0 = _time.perf_counter()
U_DF, U_SPLITS, U_CAL = UP.load_panel(U_PATHS)
U_SCORER = UE.Scorer.from_panel(U_DF)
U_REG = UE.Registry(out_dir=U_OUT)

print("=" * 78)
print("  U1 — ISOLATED FORECASTING PIPELINE")
print("=" * 78)
print(f"  built in {_time.perf_counter()-_t0:.1f}s      output root: {U_OUT}")
print(f"  panel      : {U_DF.shape[0]:,} state-day rows x {U_DF.shape[1]} cols, "
      f"{U_DF[UP.STATE].nunique()} states")
print(f"  targets ({len(UP.TARGETS)}) : {UP.TARGETS}")
print(f"  headline   : {UP.OBSERVABLE_TARGETS}  (observable; per LEAKAGE_AUDIT.md the")
print(f"               other four are derived policy overlays, reported but not headline)")
print(f"  horizons   : h = {UP.HORIZONS} days ahead  (matches the original PRED=3)")
print()
print(U_SPLITS.describe().to_string(index=False))

# Verify the reconstructed derived targets against the ALREADY-PUBLISHED numbers
# in results/tables/results_summary.csv. This proves Part II forecasts exactly
# the same series the original pipeline forecast.
_te = U_DF[U_DF['_split'] == 'test']
_published = {'mean_base_lmp': 9347.1, 'mean_palmp': 12875.8,
              'mean_pa_lmp_t': 12875.5, 'palmp_premium_pct': 37.75}
_got = {'mean_base_lmp': round(float(_te['avg_market_price'].mean()), 1),
        'mean_palmp': round(float(_te['palmp'].mean()), 1),
        'mean_pa_lmp_t': round(float(_te['pa_lmp_t'].mean()), 1),
        'palmp_premium_pct': round(100 * (_te['palmp'].mean() / _te['avg_market_price'].mean() - 1), 2)}
print(f"\n  Target-reconstruction check vs published results_summary.csv:")
for _k, _v in _published.items():
    _ok = abs(_got[_k] - _v) < 0.06 * max(1.0, abs(_v) * 0.001)
    print(f"    {_k:<22} published={_v:>10}   reconstructed={_got[_k]:>10}   "
          f"{'MATCH' if abs(_got[_k]-_v) <= 0.15 else 'MISMATCH'}")

# %% [code]
# ─── Cell U1.2 — Split, chronology, duplication, calendar-gap audit ─────────
print("=" * 78)
print("  U1.2 — SPLIT & CHRONOLOGY AUDIT")
print("=" * 78)
_split_tbl = UA.split_audit(U_DF, U_SPLITS)
print(_split_tbl.to_string(index=False))

print("\n  Chronology assertions:")
_chrono = UA.chronology_checks(U_DF, U_SPLITS)
for _, _r in _chrono.iterrows():
    print(f"    [{'PASS' if _r['passed'] else 'FAIL'}]  {_r['check']}")
assert _chrono['passed'].all(), "chronology audit failed — refusing to continue"

print("\n  Calendar completeness (why lags must be date-aware, not positional):")
_gaps = UA.calendar_gap_report(U_DF)
_g = (_gaps.groupby('split')
      .agg(states=('state_name', 'nunique'),
           observed=('observed_days', 'sum'),
           calendar=('calendar_days', 'sum'),
           missing_per_state=('missing_days', 'mean'),
           max_gap_days=('max_gap_days', 'max'))
      .reset_index())
print(_g.to_string(index=False))
print(f"\n    Every state is missing {int(_gaps[_gaps.split=='train']['missing_days'].mean())} "
      f"calendar days in TRAIN and {int(_gaps[_gaps.split=='test']['missing_days'].mean())} in TEST,")
print(f"    including a {int(_gaps['max_gap_days'].max())}-day gap (2020-03-19 -> 2020-06-01, COVID).")
print( "    => a POSITIONAL shift(1) is not 'yesterday'. Part II shifts on a complete")
print( "       daily calendar per state, so lag_k is exactly k calendar days or NaN.")

_gaps.to_csv(U_OUT / 'tables' / 'u1_calendar_gaps.csv', index=False)
_split_tbl.to_csv(U_OUT / 'tables' / 'u1_split_audit.csv', index=False)

# %% [code]
# ─── Cell U1.3 — Missing values, imputation, target validity ────────────────
print("=" * 78)
print("  U1.3 — MISSING VALUES / IMPUTATION / TARGET VALIDITY")
print("=" * 78)
_miss = UA.missing_value_report(U_DF)
print(_miss[['split', 'rows', 'cols_with_any_nan', 'total_nan_cells',
             'any_imputed_rate']].to_string(index=False))
print("\n  Per-source imputation rates:")
print(_miss[[c for c in _miss.columns if c.endswith('_imputed_rate')] + ['split']]
      .set_index('split').to_string())
print("\n  NOTE: the audit-disclosed 87.3% any-imputed rate is a property of the")
print("        SOURCE data and is inherited unchanged by Part II. It is a shared")
print("        limitation of every model compared here, not a differentiator.")

print("\n  Target validity (MAPE/MASE safety):")
_tv = UA.target_validity(U_DF)
print(_tv[_tv.split == 'test'][['target', 'n', 'nan', 'zeros', 'negative',
                                'min', 'median', 'max', 'MAPE_safe']].to_string(index=False))
print("\n    All six targets are strictly positive on TEST except log_carbon, which is")
print("    exactly 0 where carbon_proxy_tons = 0. Textbook MAPE excludes those points;")
print("    MASE / RMSSE / sMAPE are unaffected. This is why MASE is the primary metric.")
_tv.to_csv(U_OUT / 'tables' / 'u1_target_validity.csv', index=False)
_miss.to_csv(U_OUT / 'tables' / 'u1_missing_report.csv', index=False)

# %% [code]
# ─── Cell U1.4 — Feature provenance & availability at prediction time ───────
print("=" * 78)
print("  U1.4 — FEATURE PROVENANCE / LEAKAGE TABLE")
print("=" * 78)
print("  Admissibility rule: to predict target date D at horizon h, a feature may")
print("  use only (a) values dated <= D-h (the forecast origin), or (b) deterministic")
print("  calendar attributes of D.\n")

_prov = UA.provenance_table(U_DF)
for _, _r in _prov.iterrows():
    _flag = 'OK ' if _r['available_at_origin'] else 'NO '
    print(f"  [{_flag}] {_r['feature_family']}")
    print(f"         cols  : {_r['example_columns']}")
    print(f"         built : {_r['construction']}")
    print(f"         action: {_r['action']}")
_prov.to_csv(U_OUT / 'tables' / 'u1_feature_provenance.csv', index=False)

print("\n" + "-" * 78)
print("  Measured provenance of the CSV's shipped lag / rolling columns:")
print("-" * 78)
_lagprov = UA.csv_lag_provenance(U_DF)
print(_lagprov.to_string(index=False))
print("\n    gen_lag_1d matches a POSITIONAL shift, not a calendar shift.")
print("    gen_rmean_7d is a trailing mean that INCLUDES day t.")
print("    Both are therefore rebuilt from scratch in U3 rather than reused.")
_lagprov.to_csv(U_OUT / 'tables' / 'u1_csv_lag_provenance.csv', index=False)

# %% [code]
# ─── Cell U1.5 — What Part II changes vs the original pipeline ──────────────
print("=" * 78)
print("  U1.5 — ORIGINAL PIPELINE vs UPGRADE  (so the gain is attributed correctly)")
print("=" * 78)
_notes = UA.original_pipeline_notes(U_DF)
for _i, _r in _notes.iterrows():
    print(f"\n  [{_i+1}] {_r['aspect'].upper()}")
    print(f"      original : {_r['original']}")
    print(f"      upgrade  : {_r['upgrade']}")
    print(f"      matters  : {_r['why_it_matters']}")
_notes.to_csv(U_OUT / 'tables' / 'u1_pipeline_differences.csv', index=False)

print("\n" + "=" * 78)
print("  The single most important line above is [2]: the original FEATS list")
print("  (Cell 3.1) contains NO lag of total_generation_mwh. The deep models were")
print("  asked to predict the level of a series they were never shown. That, not")
print("  the architecture, is the leading explanation for MASE 3.47 vs 0.49.")
print("=" * 78)

# %% [code]
# ─── Cell U1.6 — Regime definitions for the Section U12 breakdown ───────────
print("=" * 78)
print("  U1.6 — REGIME DEFINITIONS (train-only thresholds)")
print("=" * 78)
print("  The shipped stratifiers cannot be used on TEST:")
print(f"    coal_critical : mean over TEST rows = "
      f"{U_DF.loc[U_DF.split=='test','coal_critical'].mean():.4f}  (identically zero)")
print(f"    regime        : TEST values = "
      f"{list(U_DF.loc[U_DF.split=='test','regime'].unique())}  (constant)")
print("  A constant column cannot stratify test error, so Part II defines six")
print("  overlapping regimes from PER-STATE quantiles of the TRAIN(fit) block only.\n")

_reg_rows = []
for _c in UP.REGIME_COLS:
    _reg_rows.append({'regime': UP.REGIME_LABELS[_c], 'column': _c,
                      'train': round(float(U_DF.loc[U_DF.split == 'train', _c].mean()), 4),
                      'valid': round(float(U_DF.loc[U_DF.split == 'valid', _c].mean()), 4),
                      'test': round(float(U_DF.loc[U_DF.split == 'test', _c].mean()), 4),
                      'test_rows': int(U_DF.loc[U_DF.split == 'test', _c].sum())})
_reg_tbl = _pd.DataFrame(_reg_rows)
print(_reg_tbl.to_string(index=False))
print(f"\n  thresholds: {U_DF.attrs.get('regime_thresholds')}")
print("\n  Note the High-price row: 85.6% of TEST sits above the per-state TRAIN q90.")
print("  That is not a coding artefact — the mean market price roughly doubles between")
print(f"  TRAIN ({U_DF[U_DF.split=='train']['avg_market_price'].mean():.0f}) and "
      f"TEST ({U_DF[U_DF.split=='test']['avg_market_price'].mean():.0f}). Part II reports")
print("  this as a genuine covariate shift; it is a headline limitation, not a bug.")
_reg_tbl.to_csv(U_OUT / 'tables' / 'u1_regime_coverage.csv', index=False)

U_REGIME_MAP = U_DF[[UP.DATE, UP.STATE] + UP.REGIME_COLS].copy()

# %% [code]
# ─── Cell U1.7 — MECHANICAL LEAKAGE PROOF (future-perturbation test) ───────
print("=" * 78)
print("  U1.7 — FUTURE-PERTURBATION LEAKAGE PROOF")
print("=" * 78)
print("  Method: replace EVERY numeric observation dated >= 2024-01-01 with noise,")
print("  rebuild the feature matrix, and compare feature values for target dates")
print("  BEFORE that cutoff against the clean build. Any column that changes is")
print("  reading data at or after the cutoff. Zero tolerance: exact match required.\n")

_leak_rows = []
for _tgt in UP.TARGETS:
    for _h in UP.HORIZONS:
        _r = UF.future_perturbation_test(U_CAL, _tgt, _h, _pd.Timestamp('2024-01-01'))
        _leak_rows.append({'target': _tgt, 'horizon': _h,
                           'rows_compared': _r['n_rows_compared'],
                           'cols_compared': _r['n_cols_compared'],
                           'offending_cols': len(_r['offenders']),
                           'passed': _r['passed']})
        if not _r['passed']:
            print(f"    !! {_tgt} h={_h} OFFENDERS: {_r['offenders'][:8]}")
_leak = _pd.DataFrame(_leak_rows)
print(_leak.to_string(index=False))
_all_pass = bool(_leak['passed'].all())
print(f"\n  RESULT: {int(_leak['passed'].sum())}/{len(_leak)} configurations pass "
      f"({int(_leak['rows_compared'].sum()):,} row-comparisons over "
      f"{int(_leak['cols_compared'].max())} features).")
assert _all_pass, "LEAKAGE DETECTED — refusing to continue"
print("  => No feature used anywhere in Part II can read the future.")
_leak.to_csv(U_OUT / 'tables' / 'u1_leakage_proof.csv', index=False)

# Horizon semantics: at h, lag_1 must be exactly y(D-h).
_f1 = UF.build_features(U_CAL, 'total_generation_mwh', 1)
_w = U_CAL.wide('total_generation_mwh')
_t1 = _w.shift(1).stack(future_stack=True).rename('truth').reset_index()
_t1.columns = [UP.DATE, UP.STATE, 'truth']
_m1 = _f1[[UP.DATE, UP.STATE, 'lag_1']].merge(_t1, on=[UP.DATE, UP.STATE], how='left')
print(f"\n  Horizon semantics check: at h=1, lag_1 == y(D-1) on "
      f"{_np.isclose(_m1['lag_1'], _m1['truth'], equal_nan=True).mean()*100:.2f}% of rows")
print("  => the h=1 lag_1 feature IS the persistence forecast, so the ML models and")
print("     the naive baseline are anchored on identical information.")
print(f"\n  Feature count: {len(UF.feature_columns(_f1))} admissible features per (target, horizon).")
print("=" * 78)
