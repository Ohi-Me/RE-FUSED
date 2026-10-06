# %% [markdown]
# ---
# ## Section U12 — Regime and State Error Analysis
#
# A single headline number hides where a model actually works. This section
# breaks the final candidates down two ways: by operating regime, and across all
# 18 states.
#
# **On the regimes.** The shipped stratifiers are unusable on TEST — `coal_critical`
# is identically zero across all 7,614 test rows, and `regime` is constant
# ("Post-Shift(2022-25)"). Section U1.6 therefore redefined six overlapping
# regimes from **per-state TRAIN quantiles**: Normal, Coal-critical, Monsoon,
# High-renewable, High-price, High-stress. Their test coverage is reported
# alongside every result, because a regime with few test rows cannot support a
# confident claim.

# %% [code]
# ─── Cell U12.1 — Final candidates ─────────────────────────────────────────
print("=" * 78)
print("  U12.1 — CANDIDATES FOR THE BREAKDOWN")
print("=" * 78)

_all_head = U_SCORER.headline(U_REG.stacked(list(U_REG.preds)))
_top = [m for m in _all_head['model'].tolist()
        if m not in ('Persistence', 'SeasonalNaive')][:3]
U_FINAL_CANDIDATES = list(dict.fromkeys(_top + ['Persistence', 'SeasonalNaive']))
if 'PatchTST+MCAG(replica)' in U_REG.preds:
    U_FINAL_CANDIDATES.append('PatchTST+MCAG(replica)')
print(f"  {U_FINAL_CANDIDATES}")

_fin_stack = U_REG.stacked(U_FINAL_CANDIDATES)
print(f"  aligned evaluation points: {len(_fin_stack)//len(U_FINAL_CANDIDATES):,} per model")

# %% [code]
# ─── Cell U12.2 — Error by regime ──────────────────────────────────────────
print("=" * 78)
print("  U12.2 — ERROR BY REGIME  (observable targets)")
print("=" * 78)

_reg_tbl = U_SCORER.by_regime(_fin_stack, U_REGIME_MAP, UP.REGIME_COLS,
                              targets=UP.OBSERVABLE_TARGETS)
_reg_tbl['regime_label'] = _reg_tbl['regime'].map(UP.REGIME_LABELS)

for _metric in ['MASE', 'MAE', 'RMSE', 'MAPE']:
    print(f"\n  --- {_metric} ---")
    _p = _reg_tbl.pivot_table(index='regime_label', columns='model', values=_metric)
    _p = _p.reindex([UP.REGIME_LABELS[c] for c in UP.REGIME_COLS])
    print(_p[[c for c in U_FINAL_CANDIDATES if c in _p.columns]].round(4).to_string())

print("\n  Test-row support per regime (both observable targets, h=1..3):\n")
_supp = (_reg_tbl[_reg_tbl.model == U_FINAL_CANDIDATES[0]]
         .groupby('regime_label')['n_obs'].sum().reindex(
             [UP.REGIME_LABELS[c] for c in UP.REGIME_COLS]))
for _r, _n in _supp.items():
    _flag = '' if _n > 500 else '   <- thin support, treat with caution'
    print(f"    {_r:<18} {int(_n):>7,} rows{_flag}")
_reg_tbl.to_csv(U_OUT / 'tables' / 'u12_error_by_regime.csv', index=False)

# %% [code]
# ─── Cell U12.3 — Error by state (all 18) ──────────────────────────────────
print("=" * 78)
print("  U12.3 — ERROR BY STATE  (all 18, observable targets)")
print("=" * 78)

_st_tbl = U_SCORER.by_state(_fin_stack, targets=UP.OBSERVABLE_TARGETS)
_scale_med = U_DF[U_DF['_split'] == 'train'].groupby(UP.STATE)['total_generation_mwh'].median()

for _t in UP.OBSERVABLE_TARGETS:
    print(f"\n  --- {_t}: MASE by state ---\n")
    _p = _st_tbl[_st_tbl.target == _t].pivot(index='state_name', columns='model',
                                             values='MASE')
    _p = _p[[c for c in U_FINAL_CANDIDATES if c in _p.columns]]
    _p.insert(0, 'train_median_gen', _scale_med)
    _best_model = U_FINAL_CANDIDATES[0]
    if _best_model in _p.columns and 'Persistence' in _p.columns:
        _p['delta_vs_pers_%'] = (_p[_best_model] - _p['Persistence']) / _p['Persistence'] * 100
    print(_p.sort_values('train_median_gen').round(4).to_string())
    if 'delta_vs_pers_%' in _p.columns:
        _won = int((_p['delta_vs_pers_%'] < 0).sum())
        print(f"\n    {_best_model} beats persistence in {_won}/{len(_p)} states "
              f"(median change {_p['delta_vs_pers_%'].median():+.2f}%)")

_st_tbl.to_csv(U_OUT / 'tables' / 'u12_error_by_state.csv', index=False)

# %% [code]
# ─── Cell U12.4 — Regime/state visualisation ───────────────────────────────
_fig, _axes = _plt.subplots(1, 2, figsize=(16, 5))

_ax = _axes[0]
_p = _reg_tbl.pivot_table(index='regime_label', columns='model', values='MASE')
_p = _p.reindex([UP.REGIME_LABELS[c] for c in UP.REGIME_COLS])
_p = _p[[c for c in U_FINAL_CANDIDATES if c in _p.columns]]
_p.plot(kind='bar', ax=_ax, width=0.8)
_ax.set_ylabel('MASE'); _ax.set_xlabel('')
_ax.set_title('(a) Error by operating regime', fontweight='bold', fontsize=11)
_ax.legend(fontsize=7); _ax.grid(axis='y', alpha=0.3)
_plt.setp(_ax.get_xticklabels(), rotation=25, ha='right', fontsize=8)

_ax = _axes[1]
_p2 = _st_tbl[_st_tbl.target == 'total_generation_mwh'].pivot(
    index='state_name', columns='model', values='MASE')
_bm = U_FINAL_CANDIDATES[0]
if _bm in _p2.columns and 'Persistence' in _p2.columns:
    _d = ((_p2[_bm] - _p2['Persistence']) / _p2['Persistence'] * 100).sort_values()
    _ax.barh(_d.index, _d.values,
             color=['#1a7f37' if v < 0 else '#c0392b' for v in _d.values])
    _ax.axvline(0, color='black', lw=1)
    _ax.set_xlabel(f'MASE change of {_bm} vs persistence (%)')
    _ax.set_title('(b) Per-state effect, generation', fontweight='bold', fontsize=11)
    _ax.tick_params(axis='y', labelsize=7.5); _ax.grid(axis='x', alpha=0.3)
_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U12_regime_state.png', dpi=150, bbox_inches='tight')
print(f"  saved -> {U_OUT / 'figures' / 'U12_regime_state.png'}")
_plt.show()
