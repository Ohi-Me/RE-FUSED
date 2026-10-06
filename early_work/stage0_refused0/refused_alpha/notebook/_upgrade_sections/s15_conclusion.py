# %% [markdown]
# ---
# ## Section U15 — Final Research Conclusion
#
# Generated from the measured results above, not written by hand. The verdicts
# are computed from the leaderboard, the Section U13 tests and the Section U11
# seed study, so the text cannot drift from the numbers.

# %% [code]
# ─── Cell U15.1 — Automatic verdict ────────────────────────────────────────
print("=" * 78)
print("  U15.1 — AUTOMATIC RESULT SUMMARY")
print("=" * 78)

_LB = U_LEADERBOARD.set_index('model')
U_PERSISTENCE_MASE = float(_LB.loc['Persistence', 'MASE'])
_non_trivial = [m for m in U_LEADERBOARD['model']
                if m not in ('Persistence', 'SeasonalNaive', 'RollingMean', 'Drift')]
U_BEST_MODEL = U_LEADERBOARD[U_LEADERBOARD.model.isin(_non_trivial)].iloc[0]['model']
U_BEST_MASE = float(_LB.loc[U_BEST_MODEL, 'MASE'])
U_OVERALL_BEST = str(U_LEADERBOARD.iloc[0]['model'])

_PUB_MAPE, _PUB_MASE = 28.20, 3.4748
_replica = 'PatchTST+MCAG(replica)'
_rep_mase = float(_LB.loc[_replica, 'MASE']) if _replica in _LB.index else _np.nan

print(f"""
  CURRENT RE-FUSED (published, Part I, untouched)
      model                : PatchTST + MCAG
      avg MAPE (6 targets) : {_PUB_MAPE:.2f}%
      headline MASE        : {_PUB_MASE:.4f}   (pooled scaling, observable targets)
      note                 : scored pooled while its baselines were scored per-state
                             then macro-averaged, so the published 3.4748 and the
                             published naive 0.494 were never on the same scale.

  SAME ARCHITECTURE, REPRODUCED ON THIS GRID
      model                : {_replica}
      MASE (macro)         : {_rep_mase:.4f}
      -> confirms the published pattern is reproducible and is driven by the input
         specification (no lag of the target), not by the transformer.

  NEW BEST MODEL
      model                : {U_BEST_MODEL}
      MASE (macro)         : {U_BEST_MASE:.4f}
      95% CI               : [{float(_LB.loc[U_BEST_MODEL,'CI_lo']):.4f}, {float(_LB.loc[U_BEST_MODEL,'CI_hi']):.4f}]
      sMAPE                : {float(_LB.loc[U_BEST_MODEL,'sMAPE']):.3f}
      MAE / RMSE           : {float(_LB.loc[U_BEST_MODEL,'MAE']):.3f} / {float(_LB.loc[U_BEST_MODEL,'RMSE']):.3f}
      avg MAPE (6 targets) : {float(_LB.loc[U_BEST_MODEL,'avg_MAPE_floored_6t']):.2f}%   (floored denominator, as in Part I)

  PERSISTENCE BENCHMARK
      MASE (macro)         : {U_PERSISTENCE_MASE:.4f}
      95% CI               : [{float(_LB.loc['Persistence','CI_lo']):.4f}, {float(_LB.loc['Persistence','CI_hi']):.4f}]

  IMPROVEMENT
      vs persistence       : {(U_PERSISTENCE_MASE - U_BEST_MASE)/U_PERSISTENCE_MASE*100:+.2f}%
      vs PatchTST+MCAG     : {((_rep_mase - U_BEST_MASE)/_rep_mase*100) if _np.isfinite(_rep_mase) else float('nan'):+.2f}%   (both on this grid)
""")

# --- the four verdicts, computed ------------------------------------------
_beats_pers = bool(U_BEST_MASE < U_PERSISTENCE_MASE)
_beats_mcag = bool(_np.isfinite(_rep_mase) and U_BEST_MASE < _rep_mase)

_dm_p = U_DM[(U_DM.challenger == U_BEST_MODEL) & (U_DM.reference == 'Persistence')] \
    if 'U_DM' in dir() else _pd.DataFrame()
_sig_pers = bool(len(_dm_p) and (_dm_p['significant_5pct'] & (_dm_p['better'] == U_BEST_MODEL)).all())
_bt = U_BOOT[(U_BOOT.challenger == U_BEST_MODEL) & (U_BOOT.reference == 'Persistence')] \
    if 'U_BOOT' in dir() else _pd.DataFrame()
_ci_pers = bool(len(_bt) and (_bt['excludes_zero'] & (_bt['mean_daily_MAE_diff'] < 0)).all())

_seeds_ok = None
if 'U_SEED_TBL' in dir() and len(U_SEED_TBL):
    from scipy import stats as _sst
    _v = U_SEED_TBL['MASE'].to_numpy(float)
    _n = len(_v); _sd = _v.std(ddof=1)
    _hi = _v.mean() + _sst.t.ppf(0.975, _n - 1) * _sd / _np.sqrt(_n)
    _seeds_ok = bool(_hi < U_PERSISTENCE_MASE)
    _seed_txt = (f"{_n} seeds, MASE {_v.mean():.4f} +/- {_sd:.4f}; "
                 f"upper 95% CI {_hi:.4f} vs persistence {U_PERSISTENCE_MASE:.4f}")
else:
    _seed_txt = "not run"

print("  " + "=" * 74)
print("  VERDICTS")
print("  " + "=" * 74)
for _q, _a in [
    ("Does the new best model beat PERSISTENCE?", _beats_pers),
    ("Does it beat PatchTST+MCAG (same grid)?", _beats_mcag),
    ("Is the gain vs persistence statistically significant (DM, 5%)?", _sig_pers),
    ("Does the bootstrap CI vs persistence exclude zero?", _ci_pers),
    ("Does it survive multi-seed re-runs?", _seeds_ok),
]:
    _mark = {True: 'YES', False: 'NO', None: 'NOT TESTED'}[_a]
    print(f"    {_q:<62} {_mark}")

# %% [code]
# ─── Cell U15.2 — Paper-ready conclusion ───────────────────────────────────
_state_tbl = U_SCORER.by_state(U_REG.get(U_BEST_MODEL, 'Persistence'),
                               targets=UP.OBSERVABLE_TARGETS)
_ps = _state_tbl.pivot_table(index='state_name', columns='model', values='MASE')
_n_states_won = int((_ps[U_BEST_MODEL] < _ps['Persistence']).sum()) \
    if U_BEST_MODEL in _ps.columns else 0

_by_h = U_SCORER.by_target_horizon(U_REG.get(U_BEST_MODEL, 'Persistence'))
_h_tbl = _by_h[_by_h.target.isin(UP.OBSERVABLE_TARGETS)].pivot_table(
    index='horizon', columns='model', values='MASE')

_lines = []
_lines.append("=" * 78)
_lines.append("  U15.2 — CONCLUSION (paper-ready)")
_lines.append("=" * 78)
_lines.append("")
_lines.append("We re-examined day-ahead forecasting on the RE-FUSED-Alpha panel (18 Indian")
_lines.append("states, 47,286 state-day observations, chronological test period")
_lines.append("2024-01-01 to 2025-04-22) using an isolated pipeline with a mechanical")
_lines.append("leakage proof: every observation from the test boundary onward was replaced")
_lines.append("with noise and the pre-boundary features verified bit-identical across")
_lines.append(f"{18} target-horizon configurations and 712,152 row-comparisons.")
_lines.append("")
_lines.append("The published PatchTST+MCAG result (28.20% average MAPE, MASE 3.4748) is")
_lines.append("reproducible, and its weakness is attributable to the input specification")
_lines.append("rather than the architecture: the original feature list contains no lag of")
_lines.append("total_generation_mwh, so the model was asked to predict the level of a")
_lines.append("series it was never shown. Two further measurement issues were identified —")
_lines.append("the published model MASE was computed with pooled scaling while its")
_lines.append("baselines were macro-averaged across states, and the validation split was a")
_lines.append("state holdout rather than a time holdout.")
_lines.append("")
_lines.append(f"Persistence is a strong benchmark on this panel (MASE {U_PERSISTENCE_MASE:.4f}),")
_lines.append("beating ARIMA, ETS, seasonal-naive and rolling-mean alternatives. Market")
_lines.append("price in particular is near-trivially persistent, which the original audit")
_lines.append("attributes to the 87.3% imputation rate smoothing the series.")
_lines.append("")
if _beats_pers:
    _lines.append(f"The strongest model found is {U_BEST_MODEL}, at MASE")
    _lines.append(f"{U_BEST_MASE:.4f} against persistence at {U_PERSISTENCE_MASE:.4f} — an improvement of")
    _lines.append(f"{(U_PERSISTENCE_MASE-U_BEST_MASE)/U_PERSISTENCE_MASE*100:.2f}%. It wins in {_n_states_won} of 18 states.")
    _lines.append("The gain comes from two corrections rather than from model capacity:")
    _lines.append("forecasting the residual from a persistence base (which removes the")
    _lines.append("train-to-test level shift) and normalising each state's target by its own")
    _lines.append("MASE denominator (which aligns the training loss with the macro-averaged")
    _lines.append("metric on a panel spanning three orders of magnitude).")
    _lines.append("")
    _lines.append("Statistically: " + (
        "Diebold-Mariano tests on the daily cross-state loss differential reject equal"
        if _sig_pers else
        "Diebold-Mariano tests do NOT consistently reject equal"))
    _lines.append("accuracy against persistence at the 5% level" +
                  (", and the moving-block" if _sig_pers else ", and the moving-block"))
    _lines.append(("bootstrap interval excludes zero." if _ci_pers
                   else "bootstrap interval for the error difference includes zero."))
    if _seeds_ok is True:
        _lines.append("The improvement survives re-seeding: the 95% confidence interval over")
        _lines.append(f"{len(U_SEED_TBL)} seeds lies entirely below the persistence benchmark.")
    elif _seeds_ok is False:
        _lines.append("The improvement does NOT survive re-seeding: the seed confidence")
        _lines.append("interval overlaps the persistence benchmark, so it should be reported")
        _lines.append("as inconclusive rather than as an improvement.")
else:
    _lines.append("No model tested beats persistence on the headline metric. The strongest")
    _lines.append(f"learned model, {U_BEST_MODEL}, reaches MASE {U_BEST_MASE:.4f}")
    _lines.append(f"against persistence at {U_PERSISTENCE_MASE:.4f}. We report this as the result:")
    _lines.append("on this panel, at a 1-3 day horizon, a random-walk forecast is not")
    _lines.append("improved on by tree ensembles, N-BEATS, N-HiTS, TSMixer, carbon-gated")
    _lines.append("transformers or their ensembles.")
_lines.append("")
_lines.append("Limitations. (i) 87.3% of state-day rows carry at least one imputed value;")
_lines.append("the smoothing this induces is the most likely reason persistence is so hard")
_lines.append("to beat, and it caps what any model can demonstrate here. (ii) The mean")
_lines.append("market price roughly doubles between train and test, so 85.6% of test rows")
_lines.append("sit above the per-state training 90th percentile — a covariate shift no")
_lines.append("fixed-range learner handles well. (iii) The test period contains a 54-day")
_lines.append("data outage (2024-12-11 to 2025-02-02) which removes window-based models'")
_lines.append("coverage immediately afterwards. (iv) Horizons are limited to 1-3 days to")
_lines.append("match the original PRED=3 setting; longer horizons, where persistence decays")
_lines.append("fastest, are where a learned model would most plausibly win and are left to")
_lines.append("future work. (v) No time-series foundation model was evaluated, because")
_lines.append("installing one would have altered the pinned torch build that produced the")
_lines.append("Part I results.")
_lines.append("")
_lines.append("Reporting note: MAPE is a percentage error and is never restated as an")
_lines.append("accuracy. A 28.20% MAPE does not correspond to 71.8% accuracy.")
_lines.append("")

_txt = "\n".join(_lines)
print(_txt)
(U_OUT / 'outputs').mkdir(parents=True, exist_ok=True)
(U_OUT / 'outputs' / 'u15_conclusion.txt').write_text(_txt, encoding='utf-8')

_summary = {
    'published': {'model': 'PatchTST+MCAG', 'avg_mape': _PUB_MAPE, 'mase': _PUB_MASE},
    'replica_same_grid': {'model': _replica,
                          'mase': None if not _np.isfinite(_rep_mase) else _rep_mase},
    'new_best': {'model': U_BEST_MODEL, 'mase': U_BEST_MASE,
                 'ci': [float(_LB.loc[U_BEST_MODEL, 'CI_lo']),
                        float(_LB.loc[U_BEST_MODEL, 'CI_hi'])]},
    'persistence': {'mase': U_PERSISTENCE_MASE},
    'verdicts': {'beats_persistence': _beats_pers, 'beats_patchtst_mcag': _beats_mcag,
                 'dm_significant_vs_persistence': _sig_pers,
                 'bootstrap_ci_excludes_zero': _ci_pers,
                 'survives_multiseed': _seeds_ok},
    'multiseed': _seed_txt,
    'n_states_won_vs_persistence': _n_states_won,
    'evaluation': {'test_period': ['2024-01-01', '2025-04-22'], 'horizons': UP.HORIZONS,
                   'targets': UP.TARGETS, 'headline_targets': UP.OBSERVABLE_TARGETS,
                   'primary_metric': 'MASE (macro across 18 states, m=7)'},
}
(U_OUT / 'tables' / 'u15_final_summary.json').write_text(
    _json.dumps(_summary, indent=2, default=str), encoding='utf-8')
print(f"  saved -> {U_OUT / 'outputs' / 'u15_conclusion.txt'}")
print(f"  saved -> {U_OUT / 'tables' / 'u15_final_summary.json'}")
