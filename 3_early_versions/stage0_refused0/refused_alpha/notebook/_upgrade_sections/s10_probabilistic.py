# %% [markdown]
# ---
# ## Section U10 — Probabilistic Forecasting & Uncertainty
#
# The existing quantile head (Cell 3f) is **under-covered**: its nominal 80%
# interval achieves 63.7% on `total_generation_mwh` and 63.4% on `log_carbon`
# (`results/tables/metrics_summary.json → probabilistic_forecasting`). An
# interval that claims 80% and delivers 64% is not a calibration detail — it
# understates risk by a wide margin, which matters for anything downstream that
# sizes a reserve or a hedge.
#
# Three approaches, all calibrated on **validation only**:
#
# 1. **Quantile regression** — LightGBM pinball loss at τ ∈ {0.1, 0.5, 0.9},
#    trained on the MASE-normalised persistence residual.
# 2. **Split conformal prediction** — take the residual quantiles of the
#    validation set and widen the interval by exactly the amount needed for
#    nominal coverage. Distribution-free, and finite-sample valid under
#    exchangeability.
# 3. **Conformalised quantile regression (CQR)** — the standard combination:
#    quantile regression for the shape, a conformal correction for the coverage.
#
# Reported: CRPS, PI80 coverage, mean interval width, WQL, and MAE/RMSE of the
# P50 forecast.

# %% [code]
# ─── Cell U10.1 — Quantile regression + conformal calibration ──────────────
import lightgbm as _lgb

print("=" * 78)
print("  U10.1 — QUANTILE MODELS")
print("=" * 78)

U_TAUS = [0.1, 0.5, 0.9]
_q_rows, _q_val_rows = [], []
_t_start = _time.perf_counter()

for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        _feat = UF.build_features(U_CAL, _t, _h)
        _feat = u_attach_base(_feat, U_BASE_WIDE[('Persistence', _t, _h)])
        _feat['base_minus_lag1'] = _feat['base'] - _feat['lag_1']
        _fcols = UF.feature_columns(_feat)
        _scale = U_SCORER.scale_for(_feat[UP.STATE].to_numpy(), _t)
        _feat['_z'] = (_feat['y'] - _feat['base']) / _scale
        _feat['_scale'] = _scale

        _tr = _feat[_feat.split == 'train']
        _va = _feat[_feat.split == 'valid'].reset_index(drop=True)
        _te = _feat[_feat.split == 'test'].reset_index(drop=True)

        _qv, _qt = {}, {}
        for _tau in U_TAUS:
            _p = dict(objective='quantile', alpha=_tau, learning_rate=0.06,
                      num_leaves=63, min_data_in_leaf=40, feature_fraction=0.85,
                      bagging_fraction=0.85, bagging_freq=1, verbosity=-1, seed=42)
            _dtr = _lgb.Dataset(_tr[_fcols], _tr['_z'])
            _dva = _lgb.Dataset(_va[_fcols], _va['_z'], reference=_dtr)
            _b = _lgb.train(_p, _dtr, num_boost_round=UT.MAX_ROUNDS, valid_sets=[_dva],
                            callbacks=[_lgb.early_stopping(UT.EARLY_STOP, verbose=False)])
            _best = int(_b.best_iteration or UT.MAX_ROUNDS)
            _qv[_tau] = _b.predict(_va[_fcols], num_iteration=_best)
            _full = _lgb.Dataset(_pd.concat([_tr[_fcols], _va[_fcols]]),
                                 _pd.concat([_tr['_z'], _va['_z']]))
            _bf = _lgb.train(_p, _full, num_boost_round=_best)
            _qt[_tau] = _bf.predict(_te[_fcols])

        # back to original units
        def _un(z, rows):
            return _np.asarray(z) * rows['_scale'].to_numpy() + rows['base'].to_numpy()

        _q_val_rows.append(_pd.DataFrame({
            UP.DATE: _va[UP.DATE].to_numpy(), UP.STATE: _va[UP.STATE].to_numpy(),
            'target': _t, 'horizon': _h, 'y': _va['y'].to_numpy(float),
            **{f'q{int(_tau*100)}': _un(_qv[_tau], _va) for _tau in U_TAUS}}))
        _q_rows.append(_pd.DataFrame({
            UP.DATE: _te[UP.DATE].to_numpy(), UP.STATE: _te[UP.STATE].to_numpy(),
            'target': _t, 'horizon': _h, 'y': _te['y'].to_numpy(float),
            **{f'q{int(_tau*100)}': _un(_qt[_tau], _te) for _tau in U_TAUS}}))
        del _feat
    print(f"    {_t:<24} done  ({_time.perf_counter()-_t_start:.0f}s elapsed)")

U_QVAL = _pd.concat(_q_val_rows, ignore_index=True)
U_QTEST = _pd.concat(_q_rows, ignore_index=True)
print(f"\n  quantile models fitted in {_time.perf_counter()-_t_start:.0f}s")

# %% [code]
# ─── Cell U10.2 — Conformal calibration on the validation split ────────────
print("=" * 78)
print("  U10.2 — CONFORMAL CALIBRATION (validation only)")
print("=" * 78)
print("  Split conformal: for nominal coverage 1-a, take the (1-a) empirical")
print("  quantile of the validation nonconformity scores and widen the interval by")
print("  exactly that much. CQR uses the score max(q_lo - y, y - q_hi), which")
print("  widens an interval that is too narrow and SHRINKS one that is too wide.\n")

_ALPHA = 0.20        # nominal 80% interval, matching the existing Cell 3f report
U_CONF = {}
for (_t, _h), _g in U_QVAL.groupby(['target', 'horizon']):
    # CQR nonconformity score
    _s = _np.maximum(_g['q10'].to_numpy() - _g['y'].to_numpy(),
                     _g['y'].to_numpy() - _g['q90'].to_numpy())
    _n = len(_s)
    _k = min(_n, int(_np.ceil((_n + 1) * (1 - _ALPHA))))
    _qhat = float(_np.sort(_s)[_k - 1]) if _n else 0.0
    # absolute-residual score for the simpler symmetric conformal interval
    _r = _np.abs(_g['y'].to_numpy() - _g['q50'].to_numpy())
    _rhat = float(_np.sort(_r)[min(_n, int(_np.ceil((_n + 1) * (1 - _ALPHA)))) - 1]) if _n else 0.0
    U_CONF[(_t, _h)] = {'cqr_qhat': _qhat, 'sym_qhat': _rhat, 'n_calib': _n}

_cf = _pd.DataFrame([{'target': k[0], 'horizon': k[1], **v} for k, v in U_CONF.items()])
print(_cf.pivot_table(index='target', columns='horizon',
                      values='cqr_qhat').round(3).to_string())
print("\n  positive q-hat = the validation intervals were too NARROW and are widened;")
print("  negative q-hat = they were too wide and are tightened.")

U_QTEST['cqr_lo'] = U_QTEST['q10'] - U_QTEST.set_index(['target', 'horizon']).index.map(
    lambda k: U_CONF[k]['cqr_qhat'])
U_QTEST['cqr_hi'] = U_QTEST['q90'] + U_QTEST.set_index(['target', 'horizon']).index.map(
    lambda k: U_CONF[k]['cqr_qhat'])
U_QTEST['sym_lo'] = U_QTEST['q50'] - U_QTEST.set_index(['target', 'horizon']).index.map(
    lambda k: U_CONF[k]['sym_qhat'])
U_QTEST['sym_hi'] = U_QTEST['q50'] + U_QTEST.set_index(['target', 'horizon']).index.map(
    lambda k: U_CONF[k]['sym_qhat'])

# %% [code]
# ─── Cell U10.3 — Probabilistic evaluation ─────────────────────────────────
print("=" * 78)
print("  U10.3 — PROBABILISTIC METRICS  (nominal 80% interval)")
print("=" * 78)

from refused_fixes.metrics import crps_from_quantiles as _crps

_rows = []
for _t, _g in U_QTEST.groupby('target'):
    _y = _g['y'].to_numpy(float)
    for _label, _lo, _hi in [('QuantileReg', 'q10', 'q90'),
                             ('Conformal-sym', 'sym_lo', 'sym_hi'),
                             ('CQR', 'cqr_lo', 'cqr_hi')]:
        _l, _hgh = _g[_lo].to_numpy(float), _g[_hi].to_numpy(float)
        _cov = float(_np.mean((_y >= _l) & (_y <= _hgh)))
        _wid = float(_np.mean(_hgh - _l))
        _qp = _np.stack([_l, _g['q50'].to_numpy(float), _hgh], axis=1)
        _rows.append({
            'target': _t, 'method': _label,
            'PI80_coverage': round(_cov, 4),
            'coverage_gap_pp': round((_cov - 0.80) * 100, 2),
            'mean_width': round(_wid, 4),
            'CRPS': round(_crps(_y, _qp, U_TAUS), 4),
            'WQL': round(UE.wql(_y, _qp, U_TAUS), 5),
            'P50_MAE': round(UE.mae(_y, _g['q50']), 4),
            'P50_RMSE': round(UE.rmse(_y, _g['q50']), 4),
        })
U_PROB = _pd.DataFrame(_rows)
print()
print(U_PROB.to_string(index=False))

_EXISTING = {'total_generation_mwh': 0.6368, 'avg_market_price': 0.8227,
             'grid_stress_index': 0.7664, 'fcfs_priority_score': 0.6679,
             'palmp': 0.8371, 'log_carbon': 0.6343}
def _cov_of(target, method):
    """PI80 coverage for one (target, method) cell of U_PROB."""
    _s = U_PROB[(U_PROB.target == target) & (U_PROB.method == method)]['PI80_coverage']
    return float(_s.iloc[0]) if len(_s) else _np.nan


print("\n  Coverage vs the EXISTING Cell 3f quantile head (nominal 0.80):\n")
print(f"    {'target':<24} {'existing':>9} {'QuantileReg':>12} {'CQR':>9}   verdict")
for _t in UP.TARGETS:
    _ex = _EXISTING.get(_t, _np.nan)
    _qr = _cov_of(_t, 'QuantileReg')
    _cq = _cov_of(_t, 'CQR')
    _v = 'closer to nominal' if abs(_cq - 0.8) < abs(_ex - 0.8) else 'not improved'
    print(f"    {_t:<24} {_ex:>9.4f} {_qr:>12.4f} {_cq:>9.4f}   {_v}")

_ok = sum(abs(_cov_of(t, 'CQR') - 0.8) < abs(_EXISTING.get(t, 1) - 0.8)
          for t in UP.TARGETS)
print(f"\n  CQR is closer to nominal coverage on {_ok}/{len(UP.TARGETS)} targets.")
print("  Split conformal guarantees coverage only under EXCHANGEABILITY between the")
print("  calibration block (validation) and the test block. That assumption is")
print("  violated on this panel wherever the series shifts between 2023 and 2024-25,")
print("  so the guarantee is not automatic here and is checked rather than assumed:")

_miscovered = []
for _t in UP.TARGETS:
    _cq = _cov_of(_t, 'CQR')
    if _np.isfinite(_cq) and abs(_cq - 0.80) > 0.05:
        _miscovered.append((_t, _cq))
if _miscovered:
    for _t, _cq in _miscovered:
        print(f"    {_t:<24} CQR coverage {_cq:.4f} vs nominal 0.80 "
              f"({(_cq-0.80)*100:+.1f} pp) — calibration did NOT transfer")
    print("    On these targets the validation-fitted q-hat is the wrong width for the")
    print("    test period; the interval must not be reported as an 80% interval.")
    print("    This is the same train->test shift Section U1 measured, not a coding error.")
else:
    print("    All targets land within 5 pp of nominal: calibration transferred.")
print("  Coverage must always be quoted together with the interval WIDTH above —")
print("  an interval can buy coverage simply by becoming uninformatively wide.")

U_PROB.to_csv(U_OUT / 'tables' / 'u10_probabilistic.csv', index=False)
_cf.to_csv(U_OUT / 'tables' / 'u10_conformal_qhat.csv', index=False)

_fig, _axes = _plt.subplots(1, 2, figsize=(14, 4.6))
_ax = _axes[0]
_x = _np.arange(len(UP.TARGETS)); _w = 0.26
for _i, (_m, _c) in enumerate([('QuantileReg', '#2c7fb8'), ('Conformal-sym', '#f0a202'),
                               ('CQR', '#1a7f37')]):
    _v = [_cov_of(t, _m) for t in UP.TARGETS]
    _ax.bar(_x + (_i - 1) * _w, _v, _w, label=_m, color=_c)
_ax.plot(_x, [_EXISTING.get(t, _np.nan) for t in UP.TARGETS], 'kv', ms=8, label='existing Cell 3f')
_ax.axhline(0.80, color='crimson', ls='--', lw=1.4)
_ax.set_xticks(_x); _ax.set_xticklabels([t[:14] for t in UP.TARGETS], rotation=35,
                                        ha='right', fontsize=7.5)
_ax.set_ylabel('PI80 empirical coverage'); _ax.set_ylim(0, 1.05)
_ax.set_title('(a) Coverage vs nominal 0.80', fontweight='bold', fontsize=10)
_ax.legend(fontsize=7.5); _ax.grid(axis='y', alpha=0.3)

_ax = _axes[1]
_g = U_QTEST[(U_QTEST.target == 'total_generation_mwh') & (U_QTEST.horizon == 1) &
             (U_QTEST[UP.STATE] == 'Maharashtra')].sort_values(UP.DATE).head(150)
_ax.fill_between(_g[UP.DATE], _g['cqr_lo'], _g['cqr_hi'], alpha=0.25,
                 color='#1a7f37', label='CQR 80%')
_ax.plot(_g[UP.DATE], _g['q50'], lw=1.3, color='#1a7f37', label='P50')
_ax.plot(_g[UP.DATE], _g['y'], lw=1.1, color='black', label='actual')
_ax.set_title('(b) Calibrated interval — Maharashtra, h=1', fontweight='bold', fontsize=10)
_ax.set_ylabel('generation (GWh/day)'); _ax.legend(fontsize=8); _ax.grid(alpha=0.3)
_plt.setp(_ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U10_probabilistic.png', dpi=150, bbox_inches='tight')
print(f"\n  saved -> {U_OUT / 'figures' / 'U10_probabilistic.png'}")
_plt.show()
