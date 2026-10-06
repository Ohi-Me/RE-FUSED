# %% [markdown]
# ---
# ## Section U13 — Statistical Significance
#
# A lower MAPE is not evidence. This section tests whether the differences
# survive an explicit hypothesis test and a resampling interval.
#
# **Diebold–Mariano.** Applied to the *daily cross-state mean loss differential*,
# not to the pooled state-day vector. Pooling would treat 18 contemporaneous,
# strongly cross-correlated observations as independent draws and inflate the
# test statistic — with ~7,500 "observations" almost any difference becomes
# "significant". Averaging across states first leaves ~420 genuinely sequential
# daily observations, which is the honest sample size. Both absolute-error and
# squared-error loss are reported.
#
# **Moving-block bootstrap.** 95% confidence intervals for the MASE difference,
# resampling 14-day blocks so short-range serial dependence is preserved.
#
# Comparisons are run against all three references the brief requires:
# persistence, seasonal naive, and PatchTST+MCAG.

# %% [code]
# ─── Cell U13.1 — Diebold-Mariano tests ────────────────────────────────────
print("=" * 78)
print("  U13.1 — DIEBOLD-MARIANO TESTS")
print("=" * 78)

U_REFERENCES = [r for r in ['Persistence', 'SeasonalNaive', 'PatchTST+MCAG(replica)']
                if r in U_REG.preds]
_challengers = [m for m in U_FINAL_CANDIDATES if m not in U_REFERENCES][:3]
print(f"  challengers: {_challengers}")
print(f"  references : {U_REFERENCES}\n")

_dm_rows = []
for _ch in _challengers:
    for _ref in U_REFERENCES:
        _pair = U_REG.get(_ch, _ref)
        _a = _pair[_pair.model == _ch]
        _b = _pair[_pair.model == _ref]
        for _t in UP.OBSERVABLE_TARGETS:
            for _loss in ['mae', 'mse']:
                _r = UE.dm_panel(_a, _b, _t, loss=_loss)
                _dm_rows.append({'challenger': _ch, 'reference': _ref, 'target': _t,
                                 'loss': _loss, 'n_days': _r['n'],
                                 'DM': _r['DM'], 'p_value': _r['p'],
                                 'better': (_ch if _r.get('better') == 'a' else _ref)
                                 if _r.get('better') else None,
                                 'significant_5pct': (_r['p'] < 0.05)
                                 if _np.isfinite(_r['p']) else False})
U_DM = _pd.DataFrame(_dm_rows)

for _ch in _challengers:
    print(f"\n  --- {_ch} ---")
    _s = U_DM[U_DM.challenger == _ch]
    print(f"    {'reference':<26} {'target':<22} {'loss':<5} {'DM':>8} {'p':>8}  verdict")
    for _, _r in _s.iterrows():
        _v = ('no significant difference' if not _r['significant_5pct']
              else f"{_r['better']} better (p<0.05)")
        print(f"    {_r['reference']:<26} {_r['target']:<22} {_r['loss']:<5} "
              f"{_r['DM']:>8.3f} {_r['p_value']:>8.4f}  {_v}")

U_DM.to_csv(U_OUT / 'tables' / 'u13_diebold_mariano.csv', index=False)

# %% [code]
# ─── Cell U13.2 — Block bootstrap confidence intervals ─────────────────────
print("=" * 78)
print("  U13.2 — MOVING-BLOCK BOOTSTRAP (14-day blocks, 2000 resamples)")
print("=" * 78)
print("  CI is for the DIFFERENCE in daily mean absolute error (challenger minus")
print("  reference). An interval strictly below zero means the challenger is better")
print("  at the 5% level under resampling.\n")

_bs_rows = []
for _ch in _challengers:
    for _ref in U_REFERENCES:
        _pair = U_REG.get(_ch, _ref)
        for _t in UP.OBSERVABLE_TARGETS:
            _a = _pair[(_pair.model == _ch) & (_pair.target == _t)]
            _b = _pair[(_pair.model == _ref) & (_pair.target == _t)]
            _m = _a.merge(_b, on=[UP.DATE, UP.STATE, 'target', 'horizon'],
                          suffixes=('_a', '_b'))
            _d = (_pd.DataFrame({
                UP.DATE: _m[UP.DATE],
                'd': _np.abs(_m['y_a'] - _m['yhat_a']) - _np.abs(_m['y_a'] - _m['yhat_b'])})
                .groupby(UP.DATE)['d'].mean().sort_index().to_numpy())
            _pt, _lo, _hi = UE.block_bootstrap_ci(_d, block=14, n_boot=2000, seed=42)
            _bs_rows.append({'challenger': _ch, 'reference': _ref, 'target': _t,
                             'mean_daily_MAE_diff': _pt, 'ci_lo': _lo, 'ci_hi': _hi,
                             'excludes_zero': bool(_hi < 0 or _lo > 0),
                             'direction': 'challenger better' if _pt < 0 else 'reference better'})
U_BOOT = _pd.DataFrame(_bs_rows)
print(U_BOOT.round(4).to_string(index=False))
U_BOOT.to_csv(U_OUT / 'tables' / 'u13_block_bootstrap.csv', index=False)

print("\n  Summary of the significance evidence:")
for _ch in _challengers:
    for _ref in U_REFERENCES:
        _dm_sig = U_DM[(U_DM.challenger == _ch) & (U_DM.reference == _ref)]
        _n_sig = int((_dm_sig['significant_5pct'] & (_dm_sig['better'] == _ch)).sum())
        _bt = U_BOOT[(U_BOOT.challenger == _ch) & (U_BOOT.reference == _ref)]
        _n_ci = int((_bt['excludes_zero'] & (_bt['mean_daily_MAE_diff'] < 0)).sum())
        print(f"    {_ch:<26} vs {_ref:<24} DM favours it in {_n_sig}/{len(_dm_sig)} tests; "
              f"bootstrap CI excludes 0 in {_n_ci}/{len(_bt)} targets")
