# %% [markdown]
# ---
# ## Section U9 — Ensembles
#
# Candidates are the strongest model from each family in Sections U2–U7. Three
# combination rules, in increasing order of how much they can overfit:
#
# 1. **Simple average** — equal weights, nothing estimated.
# 2. **Validation-weighted** — non-negative weights on the simplex, chosen to
#    minimise the *validation* macro-MASE objective.
# 3. **Stacking** — a ridge meta-learner on the validation predictions.
#
# **Every weight is estimated on the VALIDATION split only.** That is the whole
# reason `U_REG_VAL` exists: each model's validation predictions come from the
# early-stopping fit, which never saw the validation block, and the test split is
# untouched until the final scoring line.

# %% [code]
# ─── Cell U9.1 — Candidate selection (on validation) ───────────────────────
from scipy.optimize import minimize as _minimize

print("=" * 78)
print("  U9.1 — ENSEMBLE CANDIDATES")
print("=" * 78)

# Only models with BOTH test and validation predictions can enter an ensemble.
_eligible = [m for m in U_REG.preds if m in U_REG_VAL.preds]
_val_head = U_SCORER.headline(U_REG_VAL.stacked(_eligible))
print("\n  Ranked by VALIDATION MASE (test not consulted):\n")
print(_val_head[['model', 'MASE', 'RMSSE', 'MAE', 'sMAPE']].head(14).to_string(index=False))

# One representative per family, plus persistence as the anchor.
_families = {
    'statistical': [m for m in _eligible if m in ('Persistence', 'ETS', 'ARIMA',
                                                  'RollingMean', 'SeasonalNaive', 'Drift')],
    'tree': [m for m in _eligible if m in ('LightGBM', 'XGBoost', 'CatBoost')],
    'hybrid': [m for m in _eligible if m.startswith('Hybrid_')],
    'deep': [m for m in _eligible if m in ('N-BEATS', 'N-HiTS', 'TSMixer')],
    'mcag': [m for m in _eligible if m.startswith('MCAGv2')],
    'crossstate': [m for m in _eligible if m.startswith('XS_')],
}
_rank = _val_head.set_index('model')['MASE']
U_ENS_CANDIDATES = []
for _fam, _members in _families.items():
    _members = [m for m in _members if m in _rank.index]
    if not _members:
        continue
    _best = min(_members, key=lambda m: _rank[m])
    U_ENS_CANDIDATES.append(_best)
    print(f"    {_fam:<12} -> {_best:<26} (valid MASE {_rank[_best]:.4f})")
U_ENS_CANDIDATES = list(dict.fromkeys(U_ENS_CANDIDATES))
if 'Persistence' not in U_ENS_CANDIDATES:
    U_ENS_CANDIDATES.append('Persistence')
print(f"\n  candidates: {U_ENS_CANDIDATES}")

# %% [code]
# ─── Cell U9.2 — Build the three ensembles ─────────────────────────────────
print("=" * 78)
print("  U9.2 — COMBINATION RULES")
print("=" * 78)

_KEYS = [UP.DATE, UP.STATE, 'target', 'horizon']


def _wide_preds(reg, names):
    """(keys + y) x one column per model, on the models' common rows."""
    base = None
    for n in names:
        p = reg.preds[n][_KEYS + ['y', 'yhat']].rename(columns={'yhat': n})
        base = p if base is None else base.merge(p.drop(columns='y'), on=_KEYS, how='inner')
    return base


_val_w = _wide_preds(U_REG_VAL, U_ENS_CANDIDATES)
_test_w = _wide_preds(U_REG, U_ENS_CANDIDATES)
print(f"  aligned rows — validation {len(_val_w):,}   test {len(_test_w):,}")

# Errors are divided by the per-(state,target) MASE denominator so the objective
# being minimised IS the macro-MASE the leaderboard reports.
def _scale_col(df):
    return _np.array([U_SCORER.scales.get((s, t), 1.0)
                      for s, t in zip(df[UP.STATE], df['target'])], dtype=float)


_sv = _scale_col(_val_w)
_Pv = _val_w[U_ENS_CANDIDATES].to_numpy(float)
_yv = _val_w['y'].to_numpy(float)
_st_ = _scale_col(_test_w)
_Pt = _test_w[U_ENS_CANDIDATES].to_numpy(float)

# --- 1. simple average ----------------------------------------------------
_w_avg = _np.ones(len(U_ENS_CANDIDATES)) / len(U_ENS_CANDIDATES)

# --- 2. validation-weighted (non-negative, sums to 1) ---------------------
def _obj(w):
    return float(_np.mean(_np.abs(_Pv @ w - _yv) / _sv))


_res = _minimize(_obj, _w_avg, method='SLSQP',
                 bounds=[(0.0, 1.0)] * len(U_ENS_CANDIDATES),
                 constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
                 options={'maxiter': 300, 'ftol': 1e-10})
_w_opt = _res.x / _res.x.sum()

# --- 3. stacking: ridge meta-learner on validation ------------------------
from sklearn.linear_model import Ridge as _Ridge
_Zv = _Pv / _sv[:, None]
_Zt = _Pt / _st_[:, None]
_stack = _Ridge(alpha=1.0, fit_intercept=True).fit(_Zv, _yv / _sv)

print("\n  Learned weights (validation only):\n")
print(f"    {'model':<28} {'simple':>8} {'val-weighted':>14} {'stack coef':>12}")
for _i, _m in enumerate(U_ENS_CANDIDATES):
    print(f"    {_m:<28} {_w_avg[_i]:>8.3f} {_w_opt[_i]:>14.3f} {_stack.coef_[_i]:>12.3f}")
print(f"    {'(intercept)':<28} {'':>8} {'':>14} {_stack.intercept_:>12.3f}")

for _nm, _yhat in [
    ('Ensemble-SimpleAvg', _Pt @ _w_avg),
    ('Ensemble-ValWeighted', _Pt @ _w_opt),
    ('Ensemble-Stacked', _stack.predict(_Zt) * _st_),
]:
    _pr = _test_w[_KEYS + ['y']].copy()
    _pr['yhat'] = _yhat
    U_REG.add(_nm, _pr, section='U9', notes=f'candidates={U_ENS_CANDIDATES}')
U_ENS_NAMES = ['Ensemble-SimpleAvg', 'Ensemble-ValWeighted', 'Ensemble-Stacked']
U_REG.checkpoint()

# %% [code]
# ─── Cell U9.3 — Ensemble evaluation ───────────────────────────────────────
print("=" * 78)
print("  U9.3 — ENSEMBLE EVALUATION")
print("=" * 78)

_cmp = U_ENS_NAMES + U_ENS_CANDIDATES
_ens_stack = U_REG.stacked(_cmp)
_ens_head = U_SCORER.headline(_ens_stack)
print()
print(_ens_head[['model', 'MASE', 'RMSSE', 'MAE', 'RMSE', 'MAPE', 'sMAPE',
                 'avg_MAPE_floored_6t', 'n_obs']].to_string(index=False))

_best_single = min([m for m in U_ENS_CANDIDATES],
                   key=lambda m: float(_ens_head.set_index('model').loc[m, 'MASE']))
_bs = float(_ens_head.set_index('model').loc[_best_single, 'MASE'])
print(f"\n  best single candidate: {_best_single} (MASE {_bs:.4f})")
for _e in U_ENS_NAMES:
    _v = float(_ens_head.set_index('model').loc[_e, 'MASE'])
    print(f"    {_e:<24} {_v:.4f}   ({(_v-_bs)/_bs*100:+6.2f}% vs best single)")

_ens_head.to_csv(U_OUT / 'tables' / 'u9_ensemble_headline.csv', index=False)
_pd.DataFrame({'model': U_ENS_CANDIDATES, 'w_simple': _w_avg,
               'w_validation': _w_opt, 'stack_coef': _stack.coef_}).to_csv(
    U_OUT / 'tables' / 'u9_ensemble_weights.csv', index=False)
