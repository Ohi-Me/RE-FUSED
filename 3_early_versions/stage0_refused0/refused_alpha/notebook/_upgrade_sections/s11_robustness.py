# %% [markdown]
# ---
# ## Section U11 — Hyperparameters & Multi-Seed Robustness
#
# Two questions about the leading candidates, both answered without touching the
# test split for selection:
#
# 1. **Does the result depend on hyperparameters I happened to pick?**
#    A small grid is searched on VALIDATION only.
# 2. **Does it survive re-seeding?** The strongest candidates are retrained under
#    several seeds and reported as mean ± std with a 95% confidence interval.
#
# The seed is *never* chosen by test performance. A single-seed improvement that
# vanishes across seeds is not a result, and this section is what decides which
# of those two the upgrade has.

# %% [code]
# ─── Cell U11.1 — Validation-only hyperparameter search ────────────────────
print("=" * 78)
print("  U11.1 — HYPERPARAMETER SEARCH (validation only)")
print("=" * 78)

U_HP_GRID = [
    dict(num_leaves=31,  learning_rate=0.06, min_data_in_leaf=40),
    dict(num_leaves=63,  learning_rate=0.06, min_data_in_leaf=40),
    dict(num_leaves=63,  learning_rate=0.03, min_data_in_leaf=20),
    dict(num_leaves=127, learning_rate=0.06, min_data_in_leaf=60),
    dict(num_leaves=31,  learning_rate=0.10, min_data_in_leaf=20),
]

_hp_rows = []
_t_start = _time.perf_counter()
for _gi, _grid in enumerate(U_HP_GRID):
    _vals = []
    for _t in UP.OBSERVABLE_TARGETS:          # search on the headline targets
        for _h in UP.HORIZONS:
            _feat = UF.build_features(U_CAL, _t, _h)
            _feat = u_attach_base(_feat, U_BASE_WIDE[('Persistence', _t, _h)])
            _feat['base_minus_lag1'] = _feat['base'] - _feat['lag_1']
            _fc = UF.feature_columns(_feat)
            _sc = U_SCORER.scale_for(_feat[UP.STATE].to_numpy(), _t)
            _fit, _yh, _te, _yv, _va = UT.fit_predict_normalized(
                'lgb', _feat, _fc, _sc, base=_feat['base'].to_numpy(float),
                params=_grid, seed=42)
            _vals.append(U_SCORER.score_group(u_frame(_va, _yv, _t, _h), _t)['MASE'])
            del _feat
    _hp_rows.append({**_grid, 'valid_MASE': float(_np.mean(_vals))})
    print(f"    grid {_gi+1}/{len(U_HP_GRID)} {_grid} -> valid MASE "
          f"{_np.mean(_vals):.4f}   ({_time.perf_counter()-_t_start:.0f}s)")

U_HP = _pd.DataFrame(_hp_rows).sort_values('valid_MASE').reset_index(drop=True)
U_BEST_HP = {k: v for k, v in U_HP.iloc[0].items() if k != 'valid_MASE'}
U_BEST_HP = {k: (int(v) if k in ('num_leaves', 'min_data_in_leaf') else float(v))
             for k, v in U_BEST_HP.items()}
print(f"\n  best on validation: {U_BEST_HP}  (valid MASE {U_HP.iloc[0]['valid_MASE']:.4f})")
print(f"  spread across the grid: {U_HP['valid_MASE'].min():.4f} .. "
      f"{U_HP['valid_MASE'].max():.4f}  "
      f"({(U_HP['valid_MASE'].max()/U_HP['valid_MASE'].min()-1)*100:.1f}% range)")
U_HP.to_csv(U_OUT / 'tables' / 'u11_hyperparameter_search.csv', index=False)

# %% [code]
# ─── Cell U11.2 — Multi-seed robustness ────────────────────────────────────
print("=" * 78)
print("  U11.2 — MULTI-SEED ROBUSTNESS")
print("=" * 78)

U_SEEDS = [42, 123, 456, 789, 2024, 31337, 777, 8080]
print(f"  seeds: {U_SEEDS}  ({len(U_SEEDS)} runs)")
print("  The seed is never chosen on test. Every seed is reported.\n")

_seed_pred, _seed_rows = {}, []
_t_start = _time.perf_counter()
for _seed in U_SEEDS:
    _frames = []
    for _t in UP.TARGETS:
        for _h in UP.HORIZONS:
            _feat = UF.build_features(U_CAL, _t, _h)
            _feat = u_attach_base(_feat, U_BASE_WIDE[('Persistence', _t, _h)])
            _feat['base_minus_lag1'] = _feat['base'] - _feat['lag_1']
            _fc = UF.feature_columns(_feat)
            _sc = U_SCORER.scale_for(_feat[UP.STATE].to_numpy(), _t)
            _fit, _yh, _te, _yv, _va = UT.fit_predict_normalized(
                'lgb', _feat, _fc, _sc, base=_feat['base'].to_numpy(float),
                params=U_BEST_HP, seed=_seed)
            # same validation-calibrated shrinkage as Section U4
            _vb = _va['base'].to_numpy(float)
            _vr = _yv - _vb
            _bl, _bs = 0.0, _np.inf
            for _lam in _np.linspace(0, 1, 11):
                _s = U_SCORER.score_group(
                    u_frame(_va, _vb + _lam * _vr, _t, _h), _t)['MASE']
                if _s < _bs:
                    _bl, _bs = float(_lam), float(_s)
            _tb = _te['base'].to_numpy(float)
            _frames.append(u_frame(_te, _tb + _bl * (_yh - _tb), _t, _h))
            del _feat
    _p = _pd.concat(_frames, ignore_index=True)
    _seed_pred[_seed] = _p
    _hd = U_SCORER.headline(_p.assign(model=f'seed{_seed}'))
    _seed_rows.append({'seed': _seed,
                       'MASE': float(_hd.iloc[0]['MASE']),
                       'RMSSE': float(_hd.iloc[0]['RMSSE']),
                       'MAE': float(_hd.iloc[0]['MAE']),
                       'sMAPE': float(_hd.iloc[0]['sMAPE']),
                       'avg_MAPE_floored_6t': float(_hd.iloc[0]['avg_MAPE_floored_6t'])})
    print(f"    seed {_seed:<6} MASE={_seed_rows[-1]['MASE']:.4f}   "
          f"({_time.perf_counter()-_t_start:.0f}s)")

U_SEED_TBL = _pd.DataFrame(_seed_rows)

# %% [code]
# ─── Cell U11.3 — Seed summary, CI, and the persistence comparison ─────────
print("=" * 78)
print("  U11.3 — SEED SUMMARY")
print("=" * 78)

_pers_head = U_SCORER.headline(U_REG.preds['Persistence'].assign(model='Persistence'))
U_PERS_MASE = float(_pers_head.iloc[0]['MASE'])

print()
print(U_SEED_TBL.round(4).to_string(index=False))

_n = len(U_SEED_TBL)
_summary = {}
for _m in ['MASE', 'RMSSE', 'MAE', 'sMAPE', 'avg_MAPE_floored_6t']:
    _v = U_SEED_TBL[_m].to_numpy(float)
    _mean, _sd = float(_v.mean()), float(_v.std(ddof=1))
    _se = _sd / _np.sqrt(_n)
    from scipy import stats as _sst
    _tcrit = float(_sst.t.ppf(0.975, df=_n - 1))
    _summary[_m] = {'mean': _mean, 'std': _sd,
                    'ci_lo': _mean - _tcrit * _se, 'ci_hi': _mean + _tcrit * _se,
                    'min': float(_v.min()), 'max': float(_v.max())}

print(f"\n  Across {_n} seeds:\n")
print(f"    {'metric':<24} {'mean':>9} {'std':>9} {'95% CI':>22} {'min':>9} {'max':>9}")
for _m, _s in _summary.items():
    print(f"    {_m:<24} {_s['mean']:>9.4f} {_s['std']:>9.4f} "
          f"[{_s['ci_lo']:>9.4f}, {_s['ci_hi']:>8.4f}] {_s['min']:>9.4f} {_s['max']:>9.4f}")

_mm = _summary['MASE']
print(f"\n  Persistence MASE = {U_PERS_MASE:.4f}")
_all_beat = bool(U_SEED_TBL['MASE'].max() < U_PERS_MASE)
_ci_beat = bool(_mm['ci_hi'] < U_PERS_MASE)
print(f"    every seed beats persistence            : {_all_beat}")
print(f"    upper 95% CI bound below persistence    : {_ci_beat}")
print(f"    mean improvement                        : "
      f"{(U_PERS_MASE - _mm['mean'])/U_PERS_MASE*100:+.2f}%")
if _ci_beat:
    print("\n  => the improvement survives re-seeding: the entire confidence interval")
    print("     sits below the persistence benchmark.")
else:
    print("\n  => the improvement does NOT clearly survive re-seeding; the confidence")
    print("     interval overlaps the persistence benchmark and the result must be")
    print("     reported as inconclusive.")

# The seed-averaged forecast is itself a legitimate (and more stable) model.
_avg = None
for _s, _p in _seed_pred.items():
    _q = _p.set_index([UP.DATE, UP.STATE, 'target', 'horizon'])
    _avg = _q[['y', 'yhat']] if _avg is None else _avg.join(
        _q[['yhat']], rsuffix=f'_{_s}', how='inner')
_yh_cols = [c for c in _avg.columns if c.startswith('yhat')]
_avg['yhat_mean'] = _avg[_yh_cols].mean(axis=1)
_seed_avg = _avg.reset_index()[[UP.DATE, UP.STATE, 'target', 'horizon', 'y', 'yhat_mean']] \
    .rename(columns={'yhat_mean': 'yhat'})
U_REG.add('Hybrid_Pers_LGBM_seedavg', _seed_avg, section='U11',
          notes=f'mean forecast over {len(U_SEEDS)} seeds, tuned hyperparameters')
U_REG.checkpoint()

_sa = U_SCORER.headline(_seed_avg.assign(model='seedavg'))
print(f"\n  Seed-averaged forecast MASE = {float(_sa.iloc[0]['MASE']):.4f} "
      f"(vs single-seed mean {_mm['mean']:.4f})")

U_SEED_TBL.to_csv(U_OUT / 'tables' / 'u11_multiseed.csv', index=False)
_pd.DataFrame(_summary).T.to_csv(U_OUT / 'tables' / 'u11_seed_summary.csv')

_fig, _ax = _plt.subplots(figsize=(7.5, 4.4))
_ax.axhline(U_PERS_MASE, color='crimson', ls='--', lw=1.5, label='Persistence')
_ax.axhspan(_mm['ci_lo'], _mm['ci_hi'], color='#1a7f37', alpha=0.18, label='95% CI of mean')
_ax.axhline(_mm['mean'], color='#1a7f37', lw=1.6, label='seed mean')
_ax.plot(range(len(U_SEED_TBL)), U_SEED_TBL['MASE'], 'o', color='#0b4f6c', label='individual seeds')
_ax.set_xticks(range(len(U_SEED_TBL)))
_ax.set_xticklabels(U_SEED_TBL['seed'], fontsize=8)
_ax.set_xlabel('seed'); _ax.set_ylabel('MASE (observable targets)')
_ax.set_title('Multi-seed robustness of the leading model', fontweight='bold', fontsize=11)
_ax.legend(fontsize=8); _ax.grid(alpha=0.3)
_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U11_multiseed.png', dpi=150, bbox_inches='tight')
print(f"\n  saved -> {U_OUT / 'figures' / 'U11_multiseed.png'}")
_plt.show()
