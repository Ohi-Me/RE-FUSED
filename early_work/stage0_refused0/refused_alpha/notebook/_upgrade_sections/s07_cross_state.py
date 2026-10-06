# %% [markdown]
# ---
# ## Section U7 — Cross-State / State-Aware Forecasting
#
# An 18-state panel offers a genuine choice: train one model on everything, or
# one model per state. Pooling buys sample size and lets a state borrow strength
# from its neighbours; separating buys specialisation. This section measures the
# trade-off directly, holding everything else fixed.
#
# | | Configuration |
# |---|---|
# | A | **pooled global** — one model for all states, no state identifier at all |
# | B | **state-specific** — 18 independent models, each trained on its own state |
# | C | **global + state** — pooled, with a state identifier |
# | D | **global + state + regime** — pooled, with state identifier and the six train-only regime flags |
#
# Every configuration uses the same architecture (LightGBM), the same features,
# the same protocol and the same MASE-normalised persistence-residual target as
# the winning Section U4 hybrid, so the only thing varying is state-awareness.
#
# Results are reported three ways, because they can disagree: **overall pooled**,
# **macro-averaged across states** (the headline), and **state-by-state**.

# %% [code]
# ─── Cell U7.1 — Four state-awareness configurations ───────────────────────
print("=" * 78)
print("  U7.1 — POOLED vs STATE-SPECIFIC vs STATE-AWARE")
print("=" * 78)

U_XS_CONFIGS = {
    'A-pooled-global':      dict(state=False, regime=False, per_state=False),
    'B-state-specific':     dict(state=False, regime=False, per_state=True),
    'C-global+state':       dict(state=True,  regime=False, per_state=False),
    'D-global+state+regime': dict(state=True, regime=True,  per_state=False),
}

_xs_pred = {k: [] for k in U_XS_CONFIGS}
_xs_val = {k: [] for k in U_XS_CONFIGS}
_xs_diag = []
_t_start = _time.perf_counter()

for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        _feat0 = UF.build_features(U_CAL, _t, _h)
        _feat0 = u_attach_base(_feat0, U_BASE_WIDE[('Persistence', _t, _h)])
        _feat0['base_minus_lag1'] = _feat0['base'] - _feat0['lag_1']
        _feat0 = _feat0.merge(U_REGIME_MAP, on=[UP.DATE, UP.STATE], how='left')
        _all_cols = UF.feature_columns(_feat0)

        for _name, _cfg in U_XS_CONFIGS.items():
            _cols = [c for c in _all_cols if c not in UP.REGIME_COLS]
            if not _cfg['state']:
                _cols = [c for c in _cols if c != 'state_id']
            if _cfg['regime']:
                _cols = _cols + list(UP.REGIME_COLS)

            if not _cfg['per_state']:
                _scale = U_SCORER.scale_for(_feat0[UP.STATE].to_numpy(), _t)
                _fit, _yh, _te, _yv, _va = UT.fit_predict_normalized(
                    'lgb', _feat0, _cols, _scale,
                    base=_feat0['base'].to_numpy(float), seed=42)
                _xs_pred[_name].append(u_frame(_te, _yh, _t, _h))
                _xs_val[_name].append(u_frame(_va, _yv, _t, _h))
                _xs_diag.append({'config': _name, 'target': _t, 'horizon': _h,
                                 'n_models': 1, 'rounds': _fit.best_rounds})
            else:
                # 18 independent models, each seeing only its own state's history
                _tp, _vp, _rounds = [], [], []
                for _st, _g in _feat0.groupby(UP.STATE, sort=False):
                    _g = _g.reset_index(drop=True)
                    if (_g['split'] == 'train').sum() < 200:
                        continue
                    _sc1 = U_SCORER.scale_for(_g[UP.STATE].to_numpy(), _t)
                    _f2, _yh2, _te2, _yv2, _va2 = UT.fit_predict_normalized(
                        'lgb', _g, [c for c in _cols if c != 'state_id'], _sc1,
                        base=_g['base'].to_numpy(float), seed=42)
                    _tp.append(u_frame(_te2, _yh2, _t, _h))
                    _vp.append(u_frame(_va2, _yv2, _t, _h))
                    _rounds.append(_f2.best_rounds)
                _xs_pred[_name].append(_pd.concat(_tp, ignore_index=True))
                _xs_val[_name].append(_pd.concat(_vp, ignore_index=True))
                _xs_diag.append({'config': _name, 'target': _t, 'horizon': _h,
                                 'n_models': len(_tp),
                                 'rounds': float(_np.mean(_rounds)) if _rounds else _np.nan})
        del _feat0
    print(f"    {_t:<24} done  ({_time.perf_counter()-_t_start:.0f}s elapsed)")

for _name in U_XS_CONFIGS:
    U_REG.add(f'XS_{_name}', _pd.concat(_xs_pred[_name], ignore_index=True),
              section='U7', notes='persistence-residual LightGBM, state-awareness ablation')
    U_REG_VAL.add(f'XS_{_name}', _pd.concat(_xs_val[_name], ignore_index=True), section='U7')
U_XS_NAMES = [f'XS_{k}' for k in U_XS_CONFIGS]
U_REG.checkpoint(); U_REG_VAL.checkpoint('_registry_valid.pkl')
print(f"\n  done in {_time.perf_counter()-_t_start:.0f}s")

# %% [code]
# ─── Cell U7.2 — Overall, macro and state-wise evaluation ──────────────────
print("=" * 78)
print("  U7.2 — CROSS-STATE EVALUATION")
print("=" * 78)

_cmp = U_XS_NAMES + ['Persistence']
_xs_stack = U_REG.stacked(_cmp)
_xs_head = U_SCORER.headline(_xs_stack)

print("\n  [A] Headline — observable targets (MASE / RMSSE are macro-averaged")
print("      across the 18 states; *_pooled treat the panel as one series):\n")
print(_xs_head[['model', 'MASE', 'MASE_pooled', 'RMSSE', 'MAE', 'RMSE',
                'MAPE', 'sMAPE', 'n_obs']].to_string(index=False))

print("\n  Note how MASE (macro) and MASE_pooled rank the configurations differently —")
print("  pooling hides what happens in the smallest states, which is exactly the")
print("  effect Section U3.2 isolated.")

print("\n  [B] MASE by target x horizon:\n")
_xth = U_SCORER.by_target_horizon(_xs_stack)
print(_xth.pivot_table(index=['target', 'horizon'], columns='model',
                       values='MASE')[_cmp].round(4).to_string())

print("\n  [C] State-wise MASE, total_generation_mwh (h pooled):\n")
_xs_state = U_SCORER.by_state(_xs_stack, targets=['total_generation_mwh'])
_piv = _xs_state.pivot(index='state_name', columns='model', values='MASE')[_cmp]
_scale_med = U_DF[U_DF['_split'] == 'train'].groupby(UP.STATE)['total_generation_mwh'].median()
_piv.insert(0, 'train_median', _scale_med)
print(_piv.sort_values('train_median').round(4).to_string())

_bestcfg = str(_xs_head.iloc[0]['model'])
print(f"\n  BEST STATE-AWARENESS CONFIGURATION: {_bestcfg}  "
      f"(MASE {float(_xs_head.iloc[0]['MASE']):.4f})")

_xs_head.to_csv(U_OUT / 'tables' / 'u7_crossstate_headline.csv', index=False)
_xs_state.to_csv(U_OUT / 'tables' / 'u7_crossstate_by_state.csv', index=False)
_pd.DataFrame(_xs_diag).to_csv(U_OUT / 'tables' / 'u7_crossstate_diagnostics.csv', index=False)

_fig, _axes = _plt.subplots(1, 2, figsize=(15, 4.8))
_ax = _axes[0]
_w = 0.35
_x = _np.arange(len(_cmp))
_ax.bar(_x - _w/2, _xs_head.set_index('model').reindex(_cmp)['MASE'], _w,
        label='MASE (macro across states)', color='#2c7fb8')
_ax.bar(_x + _w/2, _xs_head.set_index('model').reindex(_cmp)['MASE_pooled'], _w,
        label='MASE (pooled panel)', color='#f0a202')
_ax.set_xticks(_x); _ax.set_xticklabels(_cmp, rotation=30, ha='right', fontsize=7.5)
_ax.set_ylabel('MASE'); _ax.legend(fontsize=8); _ax.grid(axis='y', alpha=0.3)
_ax.set_title('(a) Macro vs pooled scoring', fontweight='bold', fontsize=10)

_ax = _axes[1]
_pl = _piv.drop(columns='train_median').sort_index()
for _c in _pl.columns:
    _ax.plot(range(len(_pl)), _pl[_c].values, marker='o', ms=4, lw=1.4, label=_c)
_ax.set_xticks(range(len(_pl)))
_ax.set_xticklabels(_pl.index, rotation=75, fontsize=6.5)
_ax.set_ylabel('MASE'); _ax.set_yscale('log')
_ax.set_title('(b) State-wise, generation', fontweight='bold', fontsize=10)
_ax.legend(fontsize=7); _ax.grid(alpha=0.3)
_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U7_cross_state.png', dpi=150, bbox_inches='tight')
print(f"\n  saved -> {U_OUT / 'figures' / 'U7_cross_state.png'}")
_plt.show()
