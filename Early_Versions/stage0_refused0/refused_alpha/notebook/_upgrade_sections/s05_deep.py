# %% [markdown]
# ---
# ## Section U5 — New Deep Time-Series Models
#
# Three modern architectures, none of which exist in the original notebook:
#
# | Model | Idea |
# |---|---|
# | **N-BEATS** | residual stack of fully-connected blocks; each block subtracts what it can explain and adds its forecast |
# | **N-HiTS** | N-BEATS with multi-rate input pooling, so different stacks specialise on different frequencies |
# | **TSMixer** | all-MLP: alternating time-mixing and channel-mixing layers with residual connections |
#
# The existing `PatchTST` / `MCAG` models are **not** touched, re-run or replaced;
# they are reproduced separately in Section U6 for a like-for-like comparison.
#
# ### Two design decisions, both forced by Section U3.2
#
# **Instance normalisation.** Every window is normalised as
# `x ← (x − last_observed_value) / window_std` and the forecast is mapped back
# the same way. This (a) removes the train→test level shift, and (b) puts
# Puducherry (~0.7 GWh/day) and Uttar Pradesh (~368) on one scale, without which
# a global model is fitted almost entirely to the largest states. Because the
# offset is the *last observed value*, **a model that outputs zero is exactly the
# persistence forecast** — so these models start from the same floor as the
# Section U4 hybrids and any gain is genuinely learned.
#
# **Fully-observed windows only.** A 28-day lookback is required to be complete;
# no value is ever fabricated to fill a window. The test period contains one
# 54-day outage (2024-12-11 → 2025-02-02), so this costs coverage immediately
# after that block. Coverage is reported below, and the final leaderboard scores
# every model on the intersection of all models' evaluation points.

# %% [code]
# ─── Cell U5.1 — Windowed panel datasets ────────────────────────────────────
from refused_upgrade import deep as UD

print("=" * 78)
print("  U5.1 — WINDOWED PANEL DATASETS")
print("=" * 78)
print(f"  device: {UD.DEVICE}")

U_SPEC = UD.WindowSpec(lookback=28, horizons=tuple(UP.HORIZONS), calendar=True)
_ds_tr, _ds_va, _ds_te = UD.build_datasets(U_CAL, 'total_generation_mwh', U_SPEC)
print(f"  lookback={U_SPEC.lookback}  horizons={U_SPEC.horizons}")
print(f"  windows: train={len(_ds_tr):,}  valid={len(_ds_va):,}  test={len(_ds_te):,}")
_X0, _y0, _s0, _n0, _t0i = _ds_tr[0]
U_NFEAT = _X0.shape[1]
print(f"  input tensor per window: {tuple(_X0.shape)}  "
      f"(1 target channel + 6 calendar channels)")
print(f"  test grid rows at h=1: {len(U_GRID[('total_generation_mwh', 1)]):,}  "
      f"-> window coverage ~{len(_ds_te)/max(len(U_GRID[('total_generation_mwh',1)]),1)*100:.0f}%")

# %% [code]
# ─── Cell U5.2 — Train N-BEATS / N-HiTS / TSMixer ──────────────────────────
print("=" * 78)
print("  U5.2 — TRAINING DEEP MODELS")
print("=" * 78)

U_DEEP_BUILDERS = {
    'N-BEATS': lambda: UD.NBeats(U_SPEC.lookback, len(U_SPEC.horizons),
                                 n_blocks=4, width=256, n_feat=U_NFEAT),
    'N-HiTS':  lambda: UD.NHiTS(U_SPEC.lookback, len(U_SPEC.horizons),
                                pool_sizes=(8, 4, 1), width=256, n_feat=U_NFEAT),
    'TSMixer': lambda: UD.TSMixer(U_SPEC.lookback, len(U_SPEC.horizons),
                                  n_feat=U_NFEAT, n_blocks=4, hidden=128),
}

U_DEEP_CFG = UD.TrainConfig(epochs=40, batch_size=512, lr=1e-3,
                            patience=8, loss='l1', seed=42, verbose=False)

_deep_pred = {k: [] for k in U_DEEP_BUILDERS}
_deep_val = {k: [] for k in U_DEEP_BUILDERS}
U_DEEP_HIST = {}
_t_start = _time.perf_counter()

for _t in UP.TARGETS:
    _dtr, _dva, _dte = UD.build_datasets(U_CAL, _t, U_SPEC)
    for _name, _build in U_DEEP_BUILDERS.items():
        UD.set_seed(42)
        _model = _build()
        _model, _hist = UD.train_model(_model, _dtr, _dva, U_DEEP_CFG)
        _pr = UD.predict(_model, _dte, U_SPEC, U_CAL.dates, U_CAL.states, _t)
        # restrict to the canonical evaluation grid
        _keep = []
        for _h in UP.HORIZONS:
            _g = U_GRID[(_t, _h)][[UP.DATE, UP.STATE]]
            _keep.append(_pr[_pr.horizon == _h].merge(_g, on=[UP.DATE, UP.STATE], how='inner'))
        _deep_pred[_name].append(_pd.concat(_keep, ignore_index=True))
        # validation predictions feed the U9 ensemble weights and U10 conformal
        # calibration; the model never saw this block during training
        _pv = UD.predict(_model, _dva, U_SPEC, U_CAL.dates, U_CAL.states, _t)
        _keepv = [_pv[_pv.horizon == _h].merge(U_GRID_VAL[(_t, _h)][[UP.DATE, UP.STATE]],
                                               on=[UP.DATE, UP.STATE], how='inner')
                  for _h in UP.HORIZONS]
        _deep_val[_name].append(_pd.concat(_keepv, ignore_index=True))
        U_DEEP_HIST[(_name, _t)] = _hist
        del _model
    print(f"    {_t:<24} done  ({_time.perf_counter()-_t_start:.0f}s elapsed)")

for _name in U_DEEP_BUILDERS:
    U_REG.add(_name, _pd.concat(_deep_pred[_name], ignore_index=True), section='U5',
              notes=f'global model, instance-normalised windows, lookback={U_SPEC.lookback}')
    U_REG_VAL.add(_name, _pd.concat(_deep_val[_name], ignore_index=True), section='U5')
U_DEEP_NAMES = list(U_DEEP_BUILDERS)
U_REG.checkpoint()
U_REG_VAL.checkpoint('_registry_valid.pkl')
print(f"\n  trained {len(UP.TARGETS)*len(U_DEEP_BUILDERS)} deep models "
      f"in {_time.perf_counter()-_t_start:.0f}s")

# %% [code]
# ─── Cell U5.3 — Common evaluation of the deep family ──────────────────────
print("=" * 78)
print("  U5.3 — DEEP MODEL EVALUATION")
print("=" * 78)

_cmp = U_DEEP_NAMES + ['Persistence', 'ETS', 'LightGBM'] + \
       [n for n in ['Hybrid_Persistence_LGBM'] if n in U_REG.preds]
_deep_stack = U_REG.stacked(_cmp)
_deep_head = U_SCORER.headline(_deep_stack)

print("\n  NOTE: models are aligned onto their common evaluation points before")
print("  scoring, so the baseline numbers here are computed on the deep models'")
print(f"  window-restricted grid ({int(_deep_head['n_obs'].iloc[0]):,} rows) rather than the")
print("  full grid used in U2-U4. Compare rows within this table, not across tables.\n")

print(_deep_head[['model', 'MASE', 'RMSSE', 'MAE', 'RMSE', 'MAPE', 'sMAPE',
                  'avg_MAPE_floored_6t', 'n_obs']].to_string(index=False))

_pers_m = float(_deep_head.set_index('model').loc['Persistence', 'MASE'])
print(f"\n  vs Persistence (MASE {_pers_m:.4f}) on this aligned grid:")
for _, _r in _deep_head.iterrows():
    if _r['model'] in U_DEEP_NAMES:
        _d = (_r['MASE'] - _pers_m) / _pers_m * 100
        print(f"    {_r['model']:<12} {_r['MASE']:.4f}  ({_d:+6.2f}%)  "
              f"{'BEATS' if _r['MASE'] < _pers_m else 'does NOT beat'} persistence")

print("\n  MASE by target x horizon:\n")
_dth = U_SCORER.by_target_horizon(_deep_stack)
print(_dth.pivot_table(index=['target', 'horizon'], columns='model',
                       values='MASE')[_cmp].round(4).to_string())

_deep_head.to_csv(U_OUT / 'tables' / 'u5_deep_headline.csv', index=False)
_dth.to_csv(U_OUT / 'tables' / 'u5_deep_by_target_horizon.csv', index=False)

# %% [code]
# ─── Cell U5.4 — Deep model visualisation ───────────────────────────────────
_fig, _axes = _plt.subplots(1, 3, figsize=(17, 4.6))

_ax = _axes[0]
_d = _deep_head.set_index('model').reindex(_cmp)['MASE']
_cols = ['#6a3d9a' if m in U_DEEP_NAMES else '#8a8a8a' for m in _d.index]
_ax.bar(range(len(_d)), _d.values, color=_cols)
_ax.axhline(_pers_m, color='crimson', ls='--', lw=1.3)
_ax.set_xticks(range(len(_d)))
_ax.set_xticklabels([m.replace('Hybrid_', 'H:') for m in _d.index],
                    rotation=35, ha='right', fontsize=7.5)
_ax.set_ylabel('MASE'); _ax.set_title('(a) Deep models vs references', fontweight='bold', fontsize=10)
_ax.grid(axis='y', alpha=0.3)

_ax = _axes[1]
for _name in U_DEEP_NAMES:
    _h = U_DEEP_HIST[(_name, 'total_generation_mwh')]
    _ax.plot(_h['valid'], lw=1.8, label=f"{_name} (valid)")
_ax.set_xlabel('epoch'); _ax.set_ylabel('validation L1 (normalised units)')
_ax.set_title('(b) Convergence — generation', fontweight='bold', fontsize=10)
_ax.legend(fontsize=8); _ax.grid(alpha=0.3)

_ax = _axes[2]
for _m in U_DEEP_NAMES + ['Persistence']:
    _s = (_dth[(_dth.model == _m) & (_dth.target.isin(UP.OBSERVABLE_TARGETS))]
          .groupby('horizon')['MASE'].mean())
    _ax.plot(_s.index, _s.values, marker='o', lw=1.8, label=_m)
_ax.set_xlabel('horizon (days)'); _ax.set_ylabel('MASE'); _ax.set_xticks(UP.HORIZONS)
_ax.set_title('(c) Error growth with horizon', fontweight='bold', fontsize=10)
_ax.legend(fontsize=8); _ax.grid(alpha=0.3)

_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U5_deep_models.png', dpi=150, bbox_inches='tight')
print(f"  saved -> {U_OUT / 'figures' / 'U5_deep_models.png'}")
_plt.show()
