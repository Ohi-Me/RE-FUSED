# %% [markdown]
# ---
# ## Section U3 — Tree / Tabular Models
#
# Three gradient-boosting libraries on one shared, prediction-time-safe feature
# set. Feature construction, training, diagnosis and evaluation all live here.
#
# **Features** (82 per target × horizon, every one cleared by the U1.7 perturbation test)
#
# | Block | Content |
# |---|---|
# | Target lags | 1, 2, 3, 7, 14, 21, 28, 30, 60, 90 — measured from the forecast origin |
# | Rolling | windows 3, 7, 14, 28 × {mean, std, min, max}, ending at the origin |
# | Change / volatility | Δ1, Δ2, Δ7, Δ14, acceleration, %-changes, deviation from the 7/28-day mean, 7-day z-score, 7-vs-28 trend, coefficient of variation |
# | Temporal | day, weekday, week, month, quarter, year, month start/end, weekend, + cyclic sin/cos of weekday, month, day-of-year |
# | Exogenous (lagged) | renewable generation, outage, consumption, coal stock, carbon intensity, renewable ratio, supply-demand gap, outage ratio — lag 1, lag 7, trailing 7-day mean |
# | Cross-state | national mean at the origin, state-to-national ratio, cross-sectional rank |
#
# **Model form.** One *global* model per (target, horizon) pooling all 18 states,
# with `state_id` as a feature. Per-state models are a separate question and are
# answered in Section U7.
#
# **Protocol.** Fit on TRAIN(fit) → early-stop on VALIDATION → refit on
# TRAIN + VALIDATION for the selected round count → predict TEST once.
#
# ### The scale problem, and why it decides the whole section
#
# This panel spans three orders of magnitude — Puducherry generates ~0.7 GWh/day,
# Uttar Pradesh ~368. A global model trained on raw levels minimises *pooled*
# absolute error, but the headline metric is MASE **macro-averaged across states**.
# Those two objectives disagree sharply here, and U3.2 measures the consequence
# directly before U3.3 fixes it by learning a MASE-normalised target.

# %% [code]
# ─── Cell U3.1 — Feature build + reference fit ──────────────────────────────
from refused_upgrade import trees as UT

print("=" * 78)
print("  U3.1 — FEATURE SET")
print("=" * 78)
_f_demo = UF.build_features(U_CAL, 'total_generation_mwh', 1)
_fcols_demo = UF.feature_columns(_f_demo)
print(f"  rows (all splits) : {len(_f_demo):,}")
print(f"  admissible features: {len(_fcols_demo)}")
_blocks = {'target lags': [c for c in _fcols_demo if c.startswith('lag_')],
           'rolling': [c for c in _fcols_demo if c.startswith('roll')],
           'change/volatility': [c for c in _fcols_demo if c.startswith(('d_', 'pct_', 'dev_', 'zscore', 'trend_', 'cv', 'accel'))],
           'temporal': [c for c in _fcols_demo if c.startswith('cal_')],
           'exogenous (lagged)': [c for c in _fcols_demo if c.startswith('x_')],
           'cross-state': [c for c in _fcols_demo if c.startswith('xs_')],
           'state id': [c for c in _fcols_demo if c == 'state_id']}
for _k, _v in _blocks.items():
    print(f"    {_k:<22} {len(_v):>3}   e.g. {_v[:4]}")
del _f_demo

# %% [code]
# ─── Cell U3.2 — DIAGNOSIS: raw-level global tree vs the macro metric ───────
print("=" * 78)
print("  U3.2 — WHY A RAW-LEVEL GLOBAL TREE FAILS THE MACRO METRIC")
print("=" * 78)

_t, _h = 'total_generation_mwh', 1
_feat = UF.build_features(U_CAL, _t, _h)
_fcols = UF.feature_columns(_feat)

_fit_raw = UT.fit_predict('lgb', _feat, _fcols, seed=42)
_pr_raw = UT.pred_frame_from_fit(_feat, _fit_raw, _t, _h, 'test')

_pers_pr = U_REG.preds['Persistence']
_pers_pr = _pers_pr[(_pers_pr.target == _t) & (_pers_pr.horizon == _h)]

_a = U_SCORER.by_state(_pr_raw.assign(model='LightGBM-raw'), targets=[_t])
_b = U_SCORER.by_state(_pers_pr.assign(model='Persistence'), targets=[_t])
_cmp_state = _a.merge(_b, on=['state_name', 'target'], suffixes=('_lgb', '_pers'))
_scale_map = U_DF[U_DF['_split'] == 'train'].groupby(UP.STATE)[_t].median()
_cmp_state['train_median'] = _cmp_state['state_name'].map(_scale_map)
_cmp_state = _cmp_state.sort_values('train_median')

print(f"\n  Per-state comparison, {_t}, h=1:\n")
print(_cmp_state[['state_name', 'train_median', 'MAE_lgb', 'MAE_pers',
                  'MASE_lgb', 'MASE_pers']].round(4).to_string(index=False))

_pooled_lgb = float(_np.abs(_pr_raw.y - _pr_raw.yhat).mean())
_pooled_pers = float(_np.abs(_pers_pr.y - _pers_pr.yhat).mean())
print(f"\n  POOLED  MAE : LightGBM-raw {_pooled_lgb:.4f}   Persistence {_pooled_pers:.4f}"
      f"   -> tree WINS by {(1-_pooled_lgb/_pooled_pers)*100:.1f}%")
print(f"  MACRO   MASE: LightGBM-raw {_cmp_state.MASE_lgb.mean():.4f}   "
      f"Persistence {_cmp_state.MASE_pers.mean():.4f}   -> tree LOSES badly")
_worst = _cmp_state.loc[_cmp_state.MASE_lgb.idxmax()]
print(f"\n  The entire gap is one state: {_worst['state_name']} "
      f"(train median {_worst['train_median']:.2f} GWh/day)")
print(f"    MASE {_worst['MASE_lgb']:.2f} vs {_worst['MASE_pers']:.2f}; its absolute error is "
      f"{_worst['MAE_lgb']/_worst['MAE_pers']:.0f}x persistence's,")
print(f"    and a macro average over 18 states gives that one series 1/18 of the weight")
print(f"    regardless of its magnitude.")
print(f"\n  Extrapolation is NOT the cause here: share of test targets outside the")
print(f"  train range = {_fit_raw.extras['oob_extrapolation_rate']:.3f}.")
print("\n  FIX: learn z = y / MASE_denominator(state) instead of y. The denominator is")
print("  estimated on TRAIN only, and it makes the training loss equal the reported")
print("  metric up to a constant, so the optimiser and the leaderboard finally agree.")
del _feat

# %% [code]
# ─── Cell U3.3 — LightGBM / XGBoost / CatBoost on the MASE-normalised target ─
print("=" * 78)
print("  U3.3 — TREE MODELS (MASE-normalised global target)")
print("=" * 78)

U_TREE_KINDS = [('LightGBM', 'lgb'), ('XGBoost', 'xgb'), ('CatBoost', 'cat')]
_tree_pred = {n: [] for n, _ in U_TREE_KINDS}
_tree_val = {n: [] for n, _ in U_TREE_KINDS}
_tree_raw_pred = []
U_TREE_DIAG, U_TREE_IMP = [], {}
_t_start = _time.perf_counter()


def u_frame(rows, yhat, target, horizon):
    """Canonical prediction frame from a returned row slice."""
    return _pd.DataFrame({
        UP.DATE: rows[UP.DATE].to_numpy(), UP.STATE: rows[UP.STATE].to_numpy(),
        'target': target, 'horizon': horizon,
        'y': rows['y'].to_numpy(float), 'yhat': _np.asarray(yhat, float)})


for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        _feat = UF.build_features(U_CAL, _t, _h)
        _fcols = UF.feature_columns(_feat)
        _scale = U_SCORER.scale_for(_feat[UP.STATE].to_numpy(), _t)

        for _name, _kind in U_TREE_KINDS:
            _fit, _yh, _te, _yv, _va = UT.fit_predict_normalized(
                _kind, _feat, _fcols, _scale, seed=42)
            _tree_pred[_name].append(u_frame(_te, _yh, _t, _h))
            _tree_val[_name].append(u_frame(_va, _yv, _t, _h))
            U_TREE_DIAG.append({'model': _name, 'target': _t, 'horizon': _h,
                                'best_rounds': _fit.best_rounds})
            if _h == 1:
                U_TREE_IMP[(_name, _t)] = _fit.importance

        # keep the raw-level LightGBM as the documented counterexample
        _fit_r = UT.fit_predict('lgb', _feat, _fcols, seed=42)
        _tree_raw_pred.append(UT.pred_frame_from_fit(_feat, _fit_r, _t, _h, 'test'))
        del _feat
    print(f"    {_t:<24} done  ({_time.perf_counter()-_t_start:.0f}s elapsed)")

for _name, _ in U_TREE_KINDS:
    U_REG.add(_name, _pd.concat(_tree_pred[_name], ignore_index=True), section='U3',
              notes='global model, target normalised by the per-state MASE denominator')
    U_REG_VAL.add(_name, _pd.concat(_tree_val[_name], ignore_index=True), section='U3')
U_REG.add('LightGBM-rawlevel', _pd.concat(_tree_raw_pred, ignore_index=True), section='U3',
          notes='global model on raw levels — retained as the U3.2 counterexample')
U_TREE_NAMES = [n for n, _ in U_TREE_KINDS]
U_REG.checkpoint()
U_REG_VAL.checkpoint('_registry_valid.pkl')
print(f"\n  fitted {len(U_TREE_DIAG)+18} models in {_time.perf_counter()-_t_start:.0f}s")

# %% [code]
# ─── Cell U3.4 — Common evaluation of the tree family ──────────────────────
print("=" * 78)
print("  U3.4 — TREE MODEL EVALUATION")
print("=" * 78)

_cmp_names = U_TREE_NAMES + ['LightGBM-rawlevel', 'Persistence', 'ETS', 'ARIMA']
_tree_stack = U_REG.stacked(_cmp_names)
_tree_head = U_SCORER.headline(_tree_stack)

print("\n  [A] Headline — observable targets, macro across states, h=1..3 pooled:\n")
print(_tree_head[['model', 'MASE', 'RMSSE', 'MAE', 'RMSE', 'MAPE', 'sMAPE',
                  'avg_MAPE_floored_6t']].to_string(index=False))

_pers_mase = float(_tree_head.set_index('model').loc['Persistence', 'MASE'])
print(f"\n  vs Persistence (MASE {_pers_mase:.4f}):")
for _, _r in _tree_head.iterrows():
    if _r['model'] in U_TREE_NAMES + ['LightGBM-rawlevel']:
        _d = (_r['MASE'] - _pers_mase) / _pers_mase * 100
        print(f"    {_r['model']:<20} {_r['MASE']:.4f}  ({_d:+6.2f}%)  "
              f"{'BEATS' if _r['MASE'] < _pers_mase else 'does NOT beat'} persistence")

print("\n  [B] MASE by target x horizon:\n")
_tth = U_SCORER.by_target_horizon(_tree_stack)
print(_tth.pivot_table(index=['target', 'horizon'], columns='model',
                       values='MASE')[_cmp_names].round(4).to_string())

print("\n  [C] Boosting rounds selected on validation (mean over horizons):\n")
_diag = _pd.DataFrame(U_TREE_DIAG)
print(_diag.pivot_table(index='target', columns='model',
                        values='best_rounds', aggfunc='mean').round(0).to_string())

_tree_head.to_csv(U_OUT / 'tables' / 'u3_tree_headline.csv', index=False)
_tth.to_csv(U_OUT / 'tables' / 'u3_tree_by_target_horizon.csv', index=False)
_diag.to_csv(U_OUT / 'tables' / 'u3_tree_diagnostics.csv', index=False)
_cmp_state.to_csv(U_OUT / 'tables' / 'u3_scale_pathology_by_state.csv', index=False)

# %% [code]
# ─── Cell U3.5 — Feature importance and figure ─────────────────────────────
print("=" * 78)
print("  U3.5 — WHAT THE TREES USE  (LightGBM gain, h=1)")
print("=" * 78)
for _t in UP.OBSERVABLE_TARGETS:
    _imp = U_TREE_IMP[('LightGBM', _t)]
    _top = _imp.head(10) / _imp.sum() * 100
    print(f"\n  --- {_t} ---")
    for _f, _v in _top.items():
        print(f"      {_f:<26} {_v:5.1f}%  {'#' * max(1, int(_v))}")

_fig, _axes = _plt.subplots(1, 3, figsize=(17, 4.6))

_ax = _axes[0]
_d = _tree_head.set_index('model').reindex(_cmp_names)['MASE']
_cols = ['#2c7fb8' if m in U_TREE_NAMES else ('#c0392b' if m == 'LightGBM-rawlevel' else '#8a8a8a')
         for m in _d.index]
_ax.bar(range(len(_d)), _d.values, color=_cols)
_ax.axhline(_pers_mase, color='crimson', ls='--', lw=1.3)
_ax.set_xticks(range(len(_d)))
_ax.set_xticklabels(_d.index, rotation=35, ha='right', fontsize=7.5)
_ax.set_ylabel('MASE (observable targets)')
_ax.set_title('(a) Trees vs baselines', fontweight='bold', fontsize=10)
_ax.grid(axis='y', alpha=0.3)

_ax = _axes[1]
_ax.scatter(_cmp_state['train_median'], _cmp_state['MASE_lgb'],
            s=55, color='#c0392b', label='LightGBM (raw level)')
_ax.scatter(_cmp_state['train_median'], _cmp_state['MASE_pers'],
            s=55, color='#2c7fb8', marker='s', label='Persistence')
_ax.set_xscale('log'); _ax.set_yscale('log')
_ax.set_xlabel('state train median generation (GWh/day, log)')
_ax.set_ylabel('MASE (log)')
_ax.set_title('(b) The scale pathology', fontweight='bold', fontsize=10)
_ax.legend(fontsize=8); _ax.grid(alpha=0.3, which='both')

_ax = _axes[2]
for _m in U_TREE_NAMES + ['Persistence']:
    _s = (_tth[(_tth.model == _m) & (_tth.target.isin(UP.OBSERVABLE_TARGETS))]
          .groupby('horizon')['MASE'].mean())
    _ax.plot(_s.index, _s.values, marker='o', lw=1.8, label=_m)
_ax.set_xlabel('horizon (days)'); _ax.set_ylabel('MASE'); _ax.set_xticks(UP.HORIZONS)
_ax.set_title('(c) Error growth with horizon', fontweight='bold', fontsize=10)
_ax.legend(fontsize=8); _ax.grid(alpha=0.3)

_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U3_tree_models.png', dpi=150, bbox_inches='tight')
print(f"\n  saved -> {U_OUT / 'figures' / 'U3_tree_models.png'}")
_plt.show()
