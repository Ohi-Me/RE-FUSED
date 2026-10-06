# %% [markdown]
# ---
# ## Section U4 — Hybrid Residual Forecasting
#
# Section U2 showed persistence is the strongest baseline; Section U3 showed a
# tree predicting the *level* cannot survive the train→test level shift. The
# hybrid resolves both at once:
#
# ```
#   base   = a statistical forecast that handles the level  (persistence, snaive, ETS)
#   r      = y − base                                        (the part the base misses)
#   r̂      = gradient-boosted model on the safe feature set
#   ŷ      = base + r̂
# ```
#
# Two properties make this the right structure for this dataset:
#
# 1. **The level problem disappears.** `r` is a *change*, which is close to
#    stationary, so the tree is never asked to extrapolate beyond its training
#    range even when the level doubles.
# 2. **The baseline is a floor, not a rival.** If the model learns nothing,
#    `r̂ ≈ 0` and the hybrid degenerates to its base. It can only lose by
#    predicting badly, and validation-based early stopping guards that.
#
# **Variants** — A: Persistence + LightGBM · B: Seasonal-naive + LightGBM ·
# C: ETS + LightGBM · D: best U2 baseline + XGBoost.
#
# **Base-model honesty.** ETS/ARIMA bases used here are fitted on the TRAIN(fit)
# block *only*, so the residuals the model learns from on TRAIN come from the
# same parameterisation that generates the TEST base. Their recursive state still
# updates on observed data through the test period; only the smoothing
# coefficients are frozen. Persistence and seasonal-naive have no fitted
# parameters at all, so variants A and B are entirely free of this concern.

# %% [code]
# ─── Cell U4.1 — Build the base forecasts ───────────────────────────────────
from refused_upgrade import trees as UT

print("=" * 78)
print("  U4.1 — BASE FORECASTS FOR THE HYBRIDS")
print("=" * 78)

U_BASE_WIDE = {}

# Parameter-free bases: defined on every date, no fitting, no leakage surface.
for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        U_BASE_WIDE[('Persistence', _t, _h)] = UC.persistence(U_CAL, _t, _h)
        U_BASE_WIDE[('SeasonalNaive', _t, _h)] = UC.seasonal_naive(U_CAL, _t, _h)
print("  Persistence / SeasonalNaive bases ready (parameter-free).")

# ETS base refitted on TRAIN(fit) only, using the configuration U2.2 selected.
_ETS_CFG_LOOKUP = {
    'ETS(trend=add,seasonal=add)':    UC.ETSConfig(True, False, True),
    'ETS(trend=damped,seasonal=add)': UC.ETSConfig(True, True, True),
    'ETS(trend=none,seasonal=add)':   UC.ETSConfig(False, False, True),
    'ETS(trend=add,seasonal=none)':   UC.ETSConfig(True, False, False),
}
_t0 = _time.perf_counter()
for _t in UP.TARGETS:
    _cfg = _ETS_CFG_LOOKUP[U_REG.meta['ETS']['config_by_target'][_t]]
    _w, _ = UC.ets_forecast_wide(U_CAL, _t, UP.HORIZONS, _cfg, U_SPLITS.train_end)
    for _h in UP.HORIZONS:
        U_BASE_WIDE[('ETS', _t, _h)] = _w[_h]
print(f"  ETS base refitted on TRAIN(fit) only  ({_time.perf_counter()-_t0:.0f}s)")

print(f"\n  Base coverage on the test grid (share of grid rows with a usable base):")
for _b in ['Persistence', 'SeasonalNaive', 'ETS']:
    _cov = []
    for _t in UP.TARGETS:
        for _h in UP.HORIZONS:
            _g = U_GRID[(_t, _h)]
            _bw = U_BASE_WIDE[(_b, _t, _h)]
            _s = _bw.stack(future_stack=True).rename('b').reset_index()
            _s.columns = [UP.DATE, UP.STATE, 'b']
            _cov.append(_g.merge(_s, on=[UP.DATE, UP.STATE], how='left')['b'].notna().mean())
    print(f"    {_b:<16} {_np.mean(_cov)*100:6.2f}%")

# %% [code]
# ─── Cell U4.2 — Fit the four hybrid variants ──────────────────────────────
print("=" * 78)
print("  U4.2 — HYBRID RESIDUAL MODELS")
print("=" * 78)

U_HYBRIDS = [
    ('Hybrid_Persistence_LGBM',  'Persistence',   'lgb'),
    ('Hybrid_SeasonalNaive_LGBM', 'SeasonalNaive', 'lgb'),
    ('Hybrid_ETS_LGBM',          'ETS',           'lgb'),
    ('Hybrid_Persistence_XGB',   'Persistence',   'xgb'),   # variant D: best U2 base
]


def u_attach_base(feat, base_wide):
    """Add the base forecast as a column, aligned on (date, state)."""
    _s = base_wide.stack(future_stack=True).rename('base').reset_index()
    _s.columns = [UP.DATE, UP.STATE, 'base']
    out = feat.merge(_s, on=[UP.DATE, UP.STATE], how='left')
    return out[out['base'].notna()].reset_index(drop=True)


_hyb_pred = {n: [] for n, _, _ in U_HYBRIDS}
_hyb_val = {n: [] for n, _, _ in U_HYBRIDS}
_hyb_diag = []
_t_start = _time.perf_counter()

for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        _feat0 = UF.build_features(U_CAL, _t, _h)
        for _name, _basename, _kind in U_HYBRIDS:
            _feat = u_attach_base(_feat0, U_BASE_WIDE[(_basename, _t, _h)])
            # `base` and its gap to the last observation are both known at the
            # forecast origin, so both are admissible inputs to the residual model.
            _feat['base_minus_lag1'] = _feat['base'] - _feat['lag_1']
            _fcols = UF.feature_columns(_feat)
            _scale = U_SCORER.scale_for(_feat[UP.STATE].to_numpy(), _t)

            # The model learns  z = (y - base) / MASE_denominator(state).
            # Removing the base kills the level shift; dividing by the MASE
            # denominator kills the cross-state scale spread that U3.2 showed
            # is fatal to the macro metric. The residual needs BOTH: it inherits
            # the scale pathology just as the raw level does.
            _fit, _yhat, _te, _yval, _va = UT.fit_predict_normalized(
                _kind, _feat, _fcols, _scale,
                base=_feat['base'].to_numpy(float), seed=42)

            # --- validation-calibrated shrinkage -----------------------------
            # A residual model can only help where the base leaves exploitable
            # structure. On avg_market_price the base is already near-perfect
            # (persistence MASE ~0.08), so an uncalibrated correction injects
            # more noise than it removes. lambda* is chosen on VALIDATION only:
            #   yhat = base + lambda * residual_hat,  lambda in [0, 1]
            # lambda*=0 returns the pure base, so the hybrid is protected from
            # ever being much worse than the baseline it is built on, and the
            # test split plays no part in the choice.
            _va_base = _va['base'].to_numpy(float)
            _va_resid_hat = _yval - _va_base
            _best_lam, _best_sc = 0.0, _np.inf
            for _lam in _np.linspace(0, 1, 11):
                _cand = _pd.DataFrame({
                    UP.DATE: _va[UP.DATE].to_numpy(), UP.STATE: _va[UP.STATE].to_numpy(),
                    'target': _t, 'horizon': _h, 'y': _va['y'].to_numpy(float),
                    'yhat': _va_base + _lam * _va_resid_hat})
                _s = U_SCORER.score_group(_cand, _t)['MASE']
                if _s < _best_sc:
                    _best_lam, _best_sc = float(_lam), float(_s)

            _yhat = _te['base'].to_numpy(float) + _best_lam * (
                _yhat - _te['base'].to_numpy(float))
            _yval_shrunk = _va_base + _best_lam * _va_resid_hat
            _hyb_pred[_name].append(_pd.DataFrame({
                UP.DATE: _te[UP.DATE].to_numpy(), UP.STATE: _te[UP.STATE].to_numpy(),
                'target': _t, 'horizon': _h,
                'y': _te['y'].to_numpy(float), 'yhat': _yhat}))
            _hyb_val[_name].append(_pd.DataFrame({
                UP.DATE: _va[UP.DATE].to_numpy(), UP.STATE: _va[UP.STATE].to_numpy(),
                'target': _t, 'horizon': _h,
                'y': _va['y'].to_numpy(float), 'yhat': _yval_shrunk}))

            _r_te = (_te['y'] - _te['base']).to_numpy(float)
            _pred_resid = _yhat - _te['base'].to_numpy(float)
            _hyb_diag.append({
                'model': _name, 'target': _t, 'horizon': _h,
                'best_rounds': _fit.best_rounds,
                'lambda_valid': _best_lam,
                'valid_MASE_at_lambda': round(_best_sc, 4),
                'resid_std_test': float(_np.std(_r_te)),
                'corr_pred_resid': float(_np.corrcoef(_pred_resid, _r_te)[0, 1])
                if (len(_r_te) > 2 and _np.std(_pred_resid) > 0) else _np.nan,
            })
        del _feat0
    print(f"    {_t:<24} done  ({_time.perf_counter()-_t_start:.0f}s elapsed)")

for _name, _, _ in U_HYBRIDS:
    U_REG.add(_name, _pd.concat(_hyb_pred[_name], ignore_index=True),
              section='U4', notes='base + gradient-boosted residual, '
                                  'validation-calibrated shrinkage')
    U_REG_VAL.add(_name, _pd.concat(_hyb_val[_name], ignore_index=True), section='U4')
U_HYBRID_NAMES = [n for n, _, _ in U_HYBRIDS]
U_REG.checkpoint()
U_REG_VAL.checkpoint('_registry_valid.pkl')
print(f"\n  fitted {len(_hyb_diag)} hybrid models in {_time.perf_counter()-_t_start:.0f}s")

# %% [code]
# ─── Cell U4.3 — Common evaluation of the hybrid family ────────────────────
print("=" * 78)
print("  U4.3 — HYBRID EVALUATION")
print("=" * 78)

_cmp = U_HYBRID_NAMES + ['Persistence', 'SeasonalNaive', 'ETS', 'ARIMA', 'LightGBM']
_hy_stack = U_REG.stacked(_cmp)
_hy_head = U_SCORER.headline(_hy_stack)

print("\n  [A] Headline — observable targets, macro across states, h=1..3 pooled:\n")
print(_hy_head[['model', 'MASE', 'RMSSE', 'MAE', 'RMSE', 'MAPE', 'sMAPE',
                'avg_MAPE_floored_6t', 'n_obs']].to_string(index=False))

_pers_mase = float(_hy_head.set_index('model').loc['Persistence', 'MASE'])
print(f"\n  Persistence reference MASE = {_pers_mase:.4f}")
print("  Improvement over persistence (negative = better):\n")
for _, _r in _hy_head.iterrows():
    if _r['model'] in U_HYBRID_NAMES:
        _d = (_r['MASE'] - _pers_mase) / _pers_mase * 100
        _v = 'BEATS persistence' if _r['MASE'] < _pers_mase else 'does NOT beat persistence'
        print(f"    {_r['model']:<28} MASE={_r['MASE']:.4f}  ({_d:+.2f}%)  {_v}")

print("\n  [B] MASE by target x horizon:\n")
_hth = U_SCORER.by_target_horizon(_hy_stack)
print(_hth.pivot_table(index=['target', 'horizon'], columns='model',
                       values='MASE')[_cmp].round(4).to_string())

print("\n  [C] Validation-calibrated shrinkage lambda* per (target, horizon).")
print("      lambda*=0 means the residual model was rejected on validation and the")
print("      hybrid fell back to its pure base; lambda*=1 means it was accepted whole.\n")
_hd = _pd.DataFrame(_hyb_diag)
print(_hd.pivot_table(index=['target', 'horizon'], columns='model',
                      values='lambda_valid').round(2).to_string())

print("\n  [D] Does the residual model carry real signal?")
print("      Correlation between the predicted and realised test residual")
print("      (~0 means nothing was learned beyond the base):\n")
print(_hd.pivot_table(index='horizon', columns='model',
                      values='corr_pred_resid').round(3).to_string())

_hy_head.to_csv(U_OUT / 'tables' / 'u4_hybrid_headline.csv', index=False)
_hth.to_csv(U_OUT / 'tables' / 'u4_hybrid_by_target_horizon.csv', index=False)
_hd.to_csv(U_OUT / 'tables' / 'u4_hybrid_diagnostics.csv', index=False)

U_BEST_HYBRID = str(_hy_head[_hy_head.model.isin(U_HYBRID_NAMES)].iloc[0]['model'])
print(f"\n  BEST HYBRID: {U_BEST_HYBRID}")

# %% [code]
# ─── Cell U4.4 — Hybrid visualisation ───────────────────────────────────────
_fig, _axes = _plt.subplots(1, 3, figsize=(17, 4.6))

_ax = _axes[0]
_d = _hy_head.set_index('model').reindex(_cmp)['MASE']
_cols = ['#1a7f37' if m in U_HYBRID_NAMES else '#8a8a8a' for m in _d.index]
_ax.bar(range(len(_d)), _d.values, color=_cols)
_ax.axhline(_pers_mase, color='crimson', ls='--', lw=1.4)
_ax.text(len(_d) - 0.4, _pers_mase, ' persistence', color='crimson', fontsize=8, va='bottom', ha='right')
_ax.set_xticks(range(len(_d)))
_ax.set_xticklabels([m.replace('Hybrid_', 'H:') for m in _d.index], rotation=35, ha='right', fontsize=7.5)
_ax.set_ylabel('MASE'); _ax.set_title('(a) Hybrids vs their bases', fontweight='bold', fontsize=10)
_ax.grid(axis='y', alpha=0.3)

_ax = _axes[1]
for _m in [U_BEST_HYBRID, 'Persistence', 'ETS', 'LightGBM']:
    _s = (_hth[(_hth.model == _m) & (_hth.target.isin(UP.OBSERVABLE_TARGETS))]
          .groupby('horizon')['MASE'].mean())
    _ax.plot(_s.index, _s.values, marker='o', lw=2, label=_m.replace('Hybrid_', 'H:'))
_ax.set_xlabel('horizon (days)'); _ax.set_ylabel('MASE'); _ax.set_xticks(UP.HORIZONS)
_ax.set_title('(b) Gain concentrates at longer horizons', fontweight='bold', fontsize=10)
_ax.legend(fontsize=8); _ax.grid(alpha=0.3)

_ax = _axes[2]
_bs = U_SCORER.by_state(U_REG.get(U_BEST_HYBRID, 'Persistence'))
_bs = _bs[_bs.target == 'total_generation_mwh'].pivot(index='state_name',
                                                      columns='model', values='MASE')
_bs['delta_%'] = (_bs[U_BEST_HYBRID] - _bs['Persistence']) / _bs['Persistence'] * 100
_bs = _bs.sort_values('delta_%')
_ax.barh(_bs.index, _bs['delta_%'],
         color=['#1a7f37' if v < 0 else '#c0392b' for v in _bs['delta_%']])
_ax.axvline(0, color='black', lw=1)
_ax.set_xlabel('MASE change vs persistence (%)  — negative is better')
_ax.set_title('(c) Per-state effect, generation', fontweight='bold', fontsize=10)
_ax.tick_params(axis='y', labelsize=7); _ax.grid(axis='x', alpha=0.3)

_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U4_hybrid_residual.png', dpi=150, bbox_inches='tight')
print(f"  saved -> {U_OUT / 'figures' / 'U4_hybrid_residual.png'}")
_plt.show()
