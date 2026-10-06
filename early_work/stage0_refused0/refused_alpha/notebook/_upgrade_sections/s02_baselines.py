# %% [markdown]
# ---
# ## Section U2 — Statistical Baselines
#
# This section decides what "good" actually means on this dataset. Everything in
# Sections U3–U9 is judged against the numbers produced here.
#
# Five families, all rolling-origin, all scored on the identical
# (date, state, target, horizon) grid:
#
# | Baseline | Forecast at origin T for T+h |
# |---|---|
# | Persistence | `y(T)` — the random walk; at h=1 this is the classic naive anchor |
# | Seasonal naive | `y(T + h − 7·⌈h/7⌉)` — last week's value |
# | Rolling mean | mean of the trailing *w* days ending at T, *w* chosen on validation |
# | Drift | random walk plus a trailing 28-day slope |
# | ETS | Holt-Winters, coefficients fitted on TRAIN, recursion rolled forward |
# | ARIMA | SARIMAX, order chosen on validation, exact h-step from the Kalman filter |
#
# **Protocol.** Every tunable choice (rolling window, ETS configuration, ARIMA
# order) is made by fitting on the TRAIN(fit) block and scoring on VALIDATION.
# The selected specification is then refitted on the full original train period
# (fit + validation) and applied once to TEST. The test split is never consulted
# during selection.

# %% [code]
# ─── Cell U2.1 — Evaluation grid + trivial baselines ────────────────────────
from refused_upgrade import classical as UC

print("=" * 78)
print("  U2.1 — EVALUATION GRID & TRIVIAL BASELINES")
print("=" * 78)

# One canonical grid per (target, horizon), reused by EVERY later section.
# A row qualifies only where the realised target AND the persistence forecast
# both exist — the benchmark defines the grid, so no learned model can be
# scored on points where the benchmark is undefined.
U_GRID = {}
for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        U_GRID[(_t, _h)] = UC.eval_grid(U_CAL, _t, _h, split='test')
U_GRID_VAL = {}
for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        U_GRID_VAL[(_t, _h)] = UC.eval_grid(U_CAL, _t, _h, split='valid')

_gsz = _pd.DataFrame([{'target': t, 'h': h, 'test_rows': len(U_GRID[(t, h)]),
                       'valid_rows': len(U_GRID_VAL[(t, h)])}
                      for t in UP.TARGETS for h in UP.HORIZONS])
print(_gsz.pivot(index='target', columns='h', values='test_rows').to_string())
print(f"\n  total test evaluation points: {sum(len(v) for v in U_GRID.values()):,}")


# A SECOND registry holds every model's VALIDATION predictions. Sections U9
# (ensemble weights) and U10 (conformal calibration) fit on these, which is how
# those procedures stay strictly free of test information.
U_REG_VAL = UE.Registry(out_dir=U_OUT)


def u_register_wide(name, wide_by_h, section, notes="", grid=None, **extra):
    """Score a {(target, horizon): wide-forecast} dict on the canonical grid.

    Registers TEST predictions in `U_REG` and VALIDATION predictions in
    `U_REG_VAL` from the same forecaster.
    """
    grid = grid or U_GRID
    frames, vframes = [], []
    for _t in UP.TARGETS:
        for _h in UP.HORIZONS:
            w = wide_by_h.get((_t, _h))
            if w is None:
                continue
            frames.append(UC.to_pred_frame(w, U_CAL.wide(_t), _t, _h, grid[(_t, _h)]))
            vframes.append(UC.to_pred_frame(w, U_CAL.wide(_t), _t, _h,
                                            U_GRID_VAL[(_t, _h)]))
    U_REG_VAL.add(name, _pd.concat(vframes, ignore_index=True), section=section)
    pred = _pd.concat(frames, ignore_index=True)
    return U_REG.add(name, pred, section=section, notes=notes, **extra)


# --- persistence / seasonal naive / drift ---------------------------------
_pers, _snaive, _drift = {}, {}, {}
for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        _pers[(_t, _h)] = UC.persistence(U_CAL, _t, _h)
        _snaive[(_t, _h)] = UC.seasonal_naive(U_CAL, _t, _h)
        _drift[(_t, _h)] = UC.drift(U_CAL, _t, _h, window=28)

u_register_wide('Persistence', _pers, 'U2', 'yhat(D)=y(D-h); random walk')
u_register_wide('SeasonalNaive', _snaive, 'U2', 'yhat(D)=y(D-7); m=7')
u_register_wide('Drift', _drift, 'U2', 'random walk + trailing 28d slope')

# --- rolling mean: window selected on VALIDATION --------------------------
print("\n  Rolling-mean window selection (validation MASE, mean over h=1..3):")
_rm_sel = {}
for _t in UP.TARGETS:
    _scores = {}
    for _wnd in [3, 7, 14, 28]:
        _vals = []
        for _h in UP.HORIZONS:
            _w = UC.rolling_mean(U_CAL, _t, _h, _wnd)
            _pr = UC.to_pred_frame(_w, U_CAL.wide(_t), _t, _h,
                                   U_GRID_VAL[(_t, _h)]).dropna(subset=['yhat'])
            _vals.append(U_SCORER.score_group(_pr, _t)['MASE'])
        _scores[_wnd] = float(_np.mean(_vals))
    _best = min(_scores, key=_scores.get)
    _rm_sel[_t] = _best
    print(f"    {_t:<24} " + "  ".join(f"w={k}:{v:.4f}" for k, v in _scores.items())
          + f"   -> w={_best}")

_rmean = {}
for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        _rmean[(_t, _h)] = UC.rolling_mean(U_CAL, _t, _h, _rm_sel[_t])
u_register_wide('RollingMean', _rmean, 'U2',
                f'trailing mean, per-target window chosen on validation: {_rm_sel}',
                window_by_target=_rm_sel)

print("\n  Registered:", [k for k in U_REG.preds if U_REG.meta[k]['section'] == 'U2'])

# %% [code]
# ─── Cell U2.2 — ETS / exponential smoothing ────────────────────────────────
print("=" * 78)
print("  U2.2 — EXPONENTIAL SMOOTHING (Holt-Winters)")
print("=" * 78)
print("  Coefficients are fitted once by statsmodels on the training history;")
print("  the additive recursion is then rolled forward through validation/test,")
print("  updating on observed values only. Equivalent to a fixed-parameter")
print("  rolling forecast, but one pass instead of ~420 refits per series.\n")

_ETS_CANDS = [
    UC.ETSConfig(trend=True,  damped=False, seasonal=True),
    UC.ETSConfig(trend=True,  damped=True,  seasonal=True),
    UC.ETSConfig(trend=False, damped=False, seasonal=True),
    UC.ETSConfig(trend=True,  damped=False, seasonal=False),
]

_ets_sel, _ets_wide = {}, {}
for _t in UP.TARGETS:
    _scores = {}
    _cache = {}
    for _cfg in _ETS_CANDS:
        # SELECTION: fit on the TRAIN(fit) block only, score on VALIDATION.
        _w, _ = UC.ets_forecast_wide(U_CAL, _t, UP.HORIZONS, _cfg, U_SPLITS.train_end)
        _vals = []
        for _h in UP.HORIZONS:
            _pr = UC.to_pred_frame(_w[_h], U_CAL.wide(_t), _t, _h,
                                   U_GRID_VAL[(_t, _h)]).dropna(subset=['yhat'])
            _vals.append(U_SCORER.score_group(_pr, _t)['MASE'] if len(_pr) else _np.nan)
        _scores[_cfg.name] = float(_np.nanmean(_vals))
        _cache[_cfg.name] = _cfg
    _best_name = min(_scores, key=lambda k: _scores[k])
    _ets_sel[_t] = _best_name
    print(f"  {_t:<24}")
    for _k, _v in _scores.items():
        print(f"      {'->' if _k == _best_name else '  '} {_k:<42} valMASE={_v:.4f}")
    # FINAL: refit on the full original train period (fit + validation).
    _wf, _fits = UC.ets_forecast_wide(U_CAL, _t, UP.HORIZONS,
                                      _cache[_best_name], U_SPLITS.valid_end)
    for _h in UP.HORIZONS:
        _ets_wide[(_t, _h)] = _wf[_h]

u_register_wide('ETS', _ets_wide, 'U2',
                'Holt-Winters; per-target config selected on validation',
                config_by_target=_ets_sel)
print(f"\n  Selected ETS configs: {_ets_sel}")

# %% [code]
# ─── Cell U2.3 — ARIMA / SARIMAX ────────────────────────────────────────────
print("=" * 78)
print("  U2.3 — ARIMA (SARIMAX)")
print("=" * 78)
print("  Order is chosen per target on VALIDATION. The h-step forecast at every")
print("  origin is read exactly off the Kalman filter as  yhat(t+h|t) = Z T^h a(t|t),")
print("  verified to 1e-14 against statsmodels get_forecast(). Stationarity and")
print("  invertibility are enforced: without them the MLE can pick an explosive AR")
print("  root that looks fine at h=1 and diverges once raised to the power h.\n")

_ARIMA_CANDS = [((1, 1, 1), (0, 0, 0, 0)),
                ((2, 1, 2), (0, 0, 0, 0)),
                ((0, 1, 1), (0, 0, 0, 0)),
                ((1, 0, 0), (1, 0, 0, 7)),
                ((1, 1, 1), (1, 0, 0, 7))]

_ar_sel, _ar_wide = {}, {}
for _t in UP.TARGETS:
    _scores = {}
    for _o, _so in _ARIMA_CANDS:
        _w, _info = UC.arima_forecast_wide(U_CAL, _t, UP.HORIZONS, _o, _so,
                                           U_SPLITS.train_end)
        _errs = sum('error' in v for v in _info.values())
        _vals = []
        for _h in UP.HORIZONS:
            _pr = UC.to_pred_frame(_w[_h], U_CAL.wide(_t), _t, _h,
                                   U_GRID_VAL[(_t, _h)]).dropna(subset=['yhat'])
            _vals.append(U_SCORER.score_group(_pr, _t)['MASE'] if len(_pr) else _np.nan)
        _scores[(_o, _so)] = (float(_np.nanmean(_vals)), _errs)
    _best = min(_scores, key=lambda k: _scores[k][0] if _np.isfinite(_scores[k][0]) else 1e9)
    _ar_sel[_t] = _best
    print(f"  {_t:<24}")
    for _k, (_v, _e) in _scores.items():
        _lbl = f"ARIMA{_k[0]}x{_k[1]}"
        print(f"      {'->' if _k == _best else '  '} {_lbl:<32} "
              f"valMASE={_v:.4f}  failed_states={_e}")
    _wf, _info = UC.arima_forecast_wide(U_CAL, _t, UP.HORIZONS, _best[0], _best[1],
                                        U_SPLITS.valid_end)
    _nfail = sum('error' in v for v in _info.values())
    if _nfail:
        print(f"      NOTE: {_nfail} state(s) rejected at final fit "
              f"-> those rows are absent for ARIMA and excluded from any aligned comparison")
    for _h in UP.HORIZONS:
        _ar_wide[(_t, _h)] = _wf[_h]

u_register_wide('ARIMA', _ar_wide, 'U2',
                'SARIMAX; per-target order selected on validation',
                order_by_target={k: str(v) for k, v in _ar_sel.items()})
print(f"\n  Selected orders: " + ", ".join(f"{k}: {v[0]}x{v[1]}" for k, v in _ar_sel.items()))

# %% [code]
# ─── Cell U2.4 — Common evaluation of all statistical baselines ─────────────
print("=" * 78)
print("  U2.4 — STATISTICAL BASELINE LEADERBOARD  (test 2024-01-01 .. 2025-04-22)")
print("=" * 78)

U_BASELINE_NAMES = ['Persistence', 'SeasonalNaive', 'RollingMean', 'Drift', 'ETS', 'ARIMA']
_bl_stack = U_REG.stacked(U_BASELINE_NAMES)

print("\n  [A] Headline — observable targets only (macro-averaged across 18 states,")
print("      pooled over h=1..3). Primary ranking metric is MASE.\n")
_bl_head = U_SCORER.headline(_bl_stack)
print(_bl_head[['model', 'MASE', 'RMSSE', 'MAE', 'RMSE', 'MAPE', 'sMAPE',
                'avg_MAPE_floored_6t', 'n_obs']].to_string(index=False))

print("\n  [B] Per target x horizon MASE:\n")
_bth = U_SCORER.by_target_horizon(_bl_stack)
_piv = _bth.pivot_table(index=['target', 'horizon'], columns='model', values='MASE')
print(_piv[U_BASELINE_NAMES].round(4).to_string())

print("\n  [C] Full metric suite on the two observable targets (h pooled):\n")
_bt = U_SCORER.by_target(_bl_stack)
for _t in UP.OBSERVABLE_TARGETS:
    print(f"    --- {_t} ---")
    _s = (_bt[_bt.target == _t]
          .set_index('model')[['MAE', 'RMSE', 'MAPE', 'sMAPE', 'MASE', 'RMSSE']]
          .reindex(U_BASELINE_NAMES).round(4))
    print(_s.to_string())

_bl_head.to_csv(U_OUT / 'tables' / 'u2_baseline_headline.csv', index=False)
_bth.to_csv(U_OUT / 'tables' / 'u2_baseline_by_target_horizon.csv', index=False)
U_REG.checkpoint()          # so a later failure never costs these fits
U_REG_VAL.checkpoint('_registry_valid.pkl')

U_BEST_BASELINE = str(_bl_head.iloc[0]['model'])
U_BEST_BASELINE_MASE = float(_bl_head.iloc[0]['MASE'])
print("\n" + "=" * 78)
print(f"  STRONGEST STATISTICAL BASELINE: {U_BEST_BASELINE}  (MASE = {U_BEST_BASELINE_MASE:.4f})")
print(f"  Persistence: MASE = {float(_bl_head.set_index('model').loc['Persistence','MASE']):.4f}")
print("  This is the bar every learned model in U3-U9 must clear.")
print("=" * 78)

# %% [code]
# ─── Cell U2.5 — Baseline visualisation ─────────────────────────────────────
_fig, _axes = _plt.subplots(1, 3, figsize=(17, 4.6))

# (a) MASE by model, observable targets
_ax = _axes[0]
_d = _bt[_bt.target.isin(UP.OBSERVABLE_TARGETS)].pivot(index='model', columns='target',
                                                       values='MASE').reindex(U_BASELINE_NAMES)
_d.plot(kind='bar', ax=_ax, width=0.78, color=['#2c7fb8', '#f0a202'])
_ax.axhline(1.0, color='crimson', ls='--', lw=1.2)
_ax.text(0.02, 1.02, 'MASE = 1  (seasonal-naive skill)', transform=_ax.get_yaxis_transform(),
         color='crimson', fontsize=8, va='bottom')
_ax.set_ylabel('MASE (lower is better)'); _ax.set_xlabel('')
_ax.set_title('(a) Statistical baselines — observable targets', fontweight='bold', fontsize=10)
_ax.legend(fontsize=8); _ax.grid(axis='y', alpha=0.3)
_plt.setp(_ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

# (b) error growth with horizon
_ax = _axes[1]
for _m in U_BASELINE_NAMES:
    _s = (_bth[(_bth.model == _m) & (_bth.target.isin(UP.OBSERVABLE_TARGETS))]
          .groupby('horizon')['MASE'].mean())
    _ax.plot(_s.index, _s.values, marker='o', label=_m, lw=1.8)
_ax.set_xlabel('forecast horizon (days ahead)'); _ax.set_ylabel('MASE')
_ax.set_xticks(UP.HORIZONS)
_ax.set_title('(b) Error growth with horizon', fontweight='bold', fontsize=10)
_ax.legend(fontsize=8); _ax.grid(alpha=0.3)

# (c) per-state persistence difficulty
_ax = _axes[2]
_ps = U_SCORER.by_state(U_REG.get('Persistence'))
_ps = _ps[_ps.target == 'total_generation_mwh'].sort_values('MASE')
_ax.barh(_ps['state_name'], _ps['MASE'], color='#4a7ba7')
_ax.set_xlabel('MASE — Persistence, total_generation_mwh')
_ax.set_title('(c) Per-state difficulty (persistence)', fontweight='bold', fontsize=10)
_ax.tick_params(axis='y', labelsize=7); _ax.grid(axis='x', alpha=0.3)

_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U2_statistical_baselines.png', dpi=150, bbox_inches='tight')
print(f"  saved -> {U_OUT / 'figures' / 'U2_statistical_baselines.png'}")
_plt.show()
