# %% [markdown]
# ---
# ## Section U14 — Final Leaderboard
#
# One table, every model from Sections U2–U11, all scored on the **intersection
# of every model's evaluation points** so no row enjoys an easier subset.
#
# * **Primary ranking: MASE** (macro-averaged across the 18 states, pooled over
#   h = 1..3, observable targets). MASE < 1 beats a seasonal-naive forecast.
# * Secondary: RMSSE, RMSE, MAE, MAPE, sMAPE.
# * `DM vs Persistence` and `DM vs PatchTST+MCAG` carry the test statistic and
#   p-value from Section U13.
# * `95% CI` is the moving-block bootstrap interval for MASE.
#
# **MAPE is a percentage error, not an accuracy.** A 28.20% MAPE is not "71.8%
# accurate", and no column in this table should ever be relabelled that way.

# %% [code]
# ─── Cell U14.1 — Assemble the leaderboard ─────────────────────────────────
print("=" * 78)
print("  U14.1 — FINAL LEADERBOARD")
print("=" * 78)

U_LB_MODELS = [m for m in U_REG.preds if m != 'LightGBM-rawlevel']
_lb_stack = U_REG.stacked(U_LB_MODELS)
_n_common = len(_lb_stack) // max(len(U_LB_MODELS), 1)
print(f"  {len(U_LB_MODELS)} models on {_n_common:,} common evaluation points each")
print(f"  (intersection across every model; the widest single-model grid was "
      f"{max(len(p) for p in U_REG.preds.values()):,} rows)\n")

_lb = U_SCORER.headline(_lb_stack)

# bootstrap CI for each model's MASE, and DM against the two anchors
_ci, _dmp, _dmm = {}, {}, {}
_ref_p = U_REG.preds.get('Persistence')
_ref_m = U_REG.preds.get('PatchTST+MCAG(replica)')
for _m in U_LB_MODELS:
    _p = U_REG.preds[_m]
    _d = []
    for _t in UP.OBSERVABLE_TARGETS:
        _s = _p[_p.target == _t]
        _sc = _np.array([U_SCORER.scales.get((st, _t), 1.0) for st in _s[UP.STATE]])
        _d.append(_pd.DataFrame({UP.DATE: _s[UP.DATE],
                                 'e': _np.abs(_s['y'] - _s['yhat']) / _sc})
                  .groupby(UP.DATE)['e'].mean())
    _daily = _pd.concat(_d, axis=1).mean(axis=1).sort_index().to_numpy()
    _pt, _lo, _hi = UE.block_bootstrap_ci(_daily, block=14, n_boot=1000, seed=7)
    _ci[_m] = (_lo, _hi)
    for _ref, _store in [(_ref_p, _dmp), (_ref_m, _dmm)]:
        if _ref is None or _m in ('Persistence', 'PatchTST+MCAG(replica)'):
            _store[_m] = (_np.nan, _np.nan)
            continue
        _rs = [UE.dm_panel(_p, _ref, _t, loss='mae') for _t in UP.OBSERVABLE_TARGETS]
        _rs = [r for r in _rs if _np.isfinite(r['DM'])]
        _store[_m] = ((float(_np.mean([r['DM'] for r in _rs])),
                       float(_np.max([r['p'] for r in _rs]))) if _rs else (_np.nan, _np.nan))

_lb['CI_lo'] = _lb['model'].map(lambda m: _ci[m][0])
_lb['CI_hi'] = _lb['model'].map(lambda m: _ci[m][1])
_lb['DM_vs_Persistence'] = _lb['model'].map(lambda m: _dmp.get(m, (_np.nan,))[0])
_lb['p_vs_Persistence'] = _lb['model'].map(lambda m: _dmp.get(m, (_np.nan, _np.nan))[1])
_lb['DM_vs_PatchTST_MCAG'] = _lb['model'].map(lambda m: _dmm.get(m, (_np.nan,))[0])
_lb['p_vs_PatchTST_MCAG'] = _lb['model'].map(lambda m: _dmm.get(m, (_np.nan, _np.nan))[1])
_lb = _lb.sort_values('MASE').reset_index(drop=True)
_lb.insert(0, 'Rank', _np.arange(1, len(_lb) + 1))
U_LEADERBOARD = _lb

print(f"  {'#':>3} {'Model':<28} {'MASE':>7} {'sMAPE':>7} {'MAE':>9} {'RMSE':>9} "
      f"{'RMSSE':>7} {'MAPE':>7} {'95% CI (MASE units)':>22}")
print("  " + "-" * 112)
for _, _r in _lb.iterrows():
    print(f"  {int(_r['Rank']):>3} {_r['model']:<28} {_r['MASE']:>7.4f} {_r['sMAPE']:>7.3f} "
          f"{_r['MAE']:>9.3f} {_r['RMSE']:>9.3f} {_r['RMSSE']:>7.4f} {_r['MAPE']:>7.3f} "
          f"[{_r['CI_lo']:>8.4f},{_r['CI_hi']:>8.4f}]")

print("\n  Significance columns (mean DM over the observable targets, worst-case p;")
print("  DM < 0 means the row beats the reference):\n")
print(f"  {'#':>3} {'Model':<28} {'DM vs Persist':>14} {'p':>8} {'DM vs PatchTST':>15} {'p':>8}")
print("  " + "-" * 80)
for _, _r in _lb.iterrows():
    print(f"  {int(_r['Rank']):>3} {_r['model']:<28} {_r['DM_vs_Persistence']:>14.3f} "
          f"{_r['p_vs_Persistence']:>8.4f} {_r['DM_vs_PatchTST_MCAG']:>15.3f} "
          f"{_r['p_vs_PatchTST_MCAG']:>8.4f}")

U_LEADERBOARD.to_csv(U_OUT / 'tables' / 'u14_final_leaderboard.csv', index=False)
U_REG.dump_meta('u14_model_registry.json')
print(f"\n  saved -> {U_OUT / 'tables' / 'u14_final_leaderboard.csv'}")

# %% [code]
# ─── Cell U14.2 — Leaderboard figure ───────────────────────────────────────
_fig, _axes = _plt.subplots(1, 2, figsize=(16, max(5, 0.32 * len(U_LEADERBOARD))))

_ax = _axes[0]
_d = U_LEADERBOARD.iloc[::-1]
_pers = float(U_LEADERBOARD.set_index('model').loc['Persistence', 'MASE'])
_cols = ['#c0392b' if m == 'Persistence' else
         ('#1a7f37' if v < _pers else '#7f8c8d')
         for m, v in zip(_d['model'], _d['MASE'])]
_ax.barh(range(len(_d)), _d['MASE'], color=_cols)
_ax.errorbar(_d['MASE'], range(len(_d)),
             xerr=[_d['MASE'] - _d['CI_lo'], _d['CI_hi'] - _d['MASE']],
             fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
_ax.axvline(_pers, color='crimson', ls='--', lw=1.4)
_ax.set_yticks(range(len(_d)))
_ax.set_yticklabels(_d['model'], fontsize=7)
_ax.set_xlabel('MASE (observable targets, macro across states)')
_ax.set_title('(a) Final leaderboard — lower is better', fontweight='bold', fontsize=11)
_ax.grid(axis='x', alpha=0.3)

_ax = _axes[1]
_top6 = U_LEADERBOARD.head(6)['model'].tolist()
_tth = U_SCORER.by_target_horizon(U_REG.stacked(list(dict.fromkeys(_top6 + ['Persistence']))))
for _m in dict.fromkeys(_top6 + ['Persistence']):
    _s = (_tth[(_tth.model == _m) & (_tth.target.isin(UP.OBSERVABLE_TARGETS))]
          .groupby('horizon')['MASE'].mean())
    _ax.plot(_s.index, _s.values, marker='o', lw=1.8,
             ls='--' if _m == 'Persistence' else '-', label=_m)
_ax.set_xlabel('forecast horizon (days ahead)'); _ax.set_ylabel('MASE')
_ax.set_xticks(UP.HORIZONS)
_ax.set_title('(b) Top models by horizon', fontweight='bold', fontsize=11)
_ax.legend(fontsize=7.5); _ax.grid(alpha=0.3)

_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U14_leaderboard.png', dpi=150, bbox_inches='tight')
print(f"  saved -> {U_OUT / 'figures' / 'U14_leaderboard.png'}")
_plt.show()
