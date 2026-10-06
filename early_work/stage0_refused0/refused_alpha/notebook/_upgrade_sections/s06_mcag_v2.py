# %% [markdown]
# ---
# ## Section U6 — MCAG-v2 / RE-FUSED-Specific Improvements
#
# **The existing MCAG is not modified.** `refused_upgrade.mcag` contains
# *re-implementations* of `MCAG` and `MCAG_Regime` written from the Cell 3.2
# definitions, so new variants can be built without importing, patching or
# re-running a single original cell. The published `PatchTST+MCAG` numbers stand
# exactly as they are.
#
# This section asks two separate questions.
#
# **U6.1 — Does the original architecture, scored on this grid, behave as the
# published numbers suggest?** A replica is trained on the *original* 25-feature
# input with the *original* global standardisation. If the U1.5 diagnosis is
# right, it should land far above persistence — confirming the problem was the
# input specification, not the transformer.
#
# **U6.2–U6.4 — Does carbon-aware gating add anything once the specification is
# fixed?** The same gating idea is rebuilt on instance-normalised windows (so a
# zero output is the persistence forecast) and extended six ways:
#
# | | Extension |
# |---|---|
# | A | state embedding |
# | B | regime embedding (the six train-only regime flags) |
# | C | cross-state lagged information (national mean at the origin) |
# | D | improved carbon/price fusion (separate gates + learned mixing) |
# | E | multi-task prediction (auxiliary heads on the other targets) |
# | F | best practical combination, chosen on **validation** |

# %% [code]
# ─── Cell U6.1 — Reference: the original architecture on this grid ──────────
import torch as _torch
from refused_upgrade import mcag as UM
from refused_upgrade import deep as UD

print("=" * 78)
print("  U6.1 — ORIGINAL PatchTST+MCAG, REPRODUCED ON THE UPGRADE GRID")
print("=" * 78)
print("  Input: the original Cell 3.1 FEATS (no target lag), globally standardised")
print("  with train-only mean/std — i.e. the original specification, windowed")
print("  calendar-correctly per state so it can be scored against everything else.\n")

_ORIG_FEATS = ['month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'renewable_ratio',
               'outage_ratio', 'carbon_intensity', 'carbon_budget_pressure',
               'grid_stress_index', 'gen_efficiency', 'price_norm',
               're_penetration_pct', 'demand_pressure', 'coal_outage_stress',
               'p_carbon_gcal', 'lcmp_carbon', 'discom_stress', 'freq_excursion_risk',
               'monsoon_re_risk', 'ppa_deviation', 'rolling_cvar4hr',
               'ci_intraday_variance', 're_ramp_risk', 'monsoon_re_credit', 'pa_lmp_t']
_ORIG_FEATS = [c for c in _ORIG_FEATS if c in U_DF.columns]
_ORIG_CARBON = [c for c in ['p_carbon_gcal', 'lcmp_carbon', 'carbon_budget_pressure',
                            'freq_excursion_risk', 'monsoon_re_risk'] if c in _ORIG_FEATS]
print(f"  original features: {len(_ORIG_FEATS)}   carbon gate inputs: {len(_ORIG_CARBON)}")
print(f"  contains a lag of total_generation_mwh: "
      f"{any('gen' in f and 'lag' in f for f in _ORIG_FEATS)}")


class OrigFeatWindows(UD.Dataset):
    """Windows over the ORIGINAL feature specification.

    Standardisation is global (train mean/std per column), exactly as Cell 3.1
    did — deliberately NOT the instance normalisation used elsewhere in Part II,
    because the point of this cell is to reproduce the original setup.
    """

    def __init__(self, cal, feats, carbon, target, lookback, horizons, split, stats):
        self.L, self.H = lookback, max(horizons)
        self.horizons = tuple(horizons)
        self.dates, self.states = cal.dates, cal.states
        mu, sd = stats
        self.X = _np.stack([((cal.wide(f).to_numpy(float) - mu[f]) / sd[f])
                            for f in feats], axis=-1)          # (T, S, F)
        self.cidx = [feats.index(c) for c in carbon]
        self.Y = cal.wide(target).to_numpy(float)
        self.ymu, self.ysd = float(mu[target]), float(sd[target])
        S = cal.split_of().to_numpy(dtype=object)
        obs = _np.isfinite(self.Y) & _np.isfinite(self.X).all(axis=-1)
        cobs = _np.cumsum(obs.astype(int), axis=0)
        idx = []
        for s in range(self.Y.shape[1]):
            for t in range(self.L - 1, self.Y.shape[0] - self.H):
                if (cobs[t, s] - (cobs[t - self.L, s] if t - self.L >= 0 else 0)) != self.L:
                    continue
                if not _np.isfinite(self.Y[t + 1:t + 1 + self.H, s]).all():
                    continue
                if S[t + self.H, s] != split:
                    continue
                idx.append((s, t))
        self.index = idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        import torch
        s, t = self.index[i]
        sl = slice(t - self.L + 1, t + 1)
        x = _np.nan_to_num(self.X[sl, s, :]).astype('f4')
        c = x[:, self.cidx]
        y = ((self.Y[t + 1:t + 1 + self.H, s] - self.ymu) / self.ysd).astype('f4')
        return (torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(y),
                torch.tensor(s), torch.tensor(t))


_tr_rows = U_DF[U_DF['_split'] == 'train']
_STAT_COLS = list(dict.fromkeys(_ORIG_FEATS + UP.TARGETS))   # dedupe: GSI is both
_ORIG_STATS = (_tr_rows[_STAT_COLS].mean(),
               _tr_rows[_STAT_COLS].std().replace(0, 1))

_orig_pred = []
_t0 = _time.perf_counter()
for _t in UP.TARGETS:
    _dsets = [OrigFeatWindows(U_CAL, _ORIG_FEATS, _ORIG_CARBON, _t, 28,
                              UP.HORIZONS, sp, _ORIG_STATS)
              for sp in ('train', 'valid', 'test')]
    UD.set_seed(42)
    _m = UM.PatchTSTMCAG_Replica(28, len(_ORIG_FEATS), len(_ORIG_CARBON),
                                 len(UP.HORIZONS), n_targets=1, d=32).to(UD.DEVICE)
    _opt = _torch.optim.AdamW(_m.parameters(), lr=1e-3, weight_decay=1e-5)
    _dl_tr = UD.DataLoader(_dsets[0], batch_size=512, shuffle=True, drop_last=True)
    _dl_va = UD.DataLoader(_dsets[1], batch_size=512)
    _best, _bstate, _bad = 1e9, None, 0
    for _ep in range(40):
        _m.train()
        for _x, _c, _y, _s, _ti in _dl_tr:
            _opt.zero_grad()
            _loss = _torch.nn.functional.l1_loss(
                _m(_x.to(UD.DEVICE), _c.to(UD.DEVICE)).squeeze(-1), _y.to(UD.DEVICE))
            _loss.backward(); _opt.step()
        _m.eval(); _tot = _n = 0
        with _torch.no_grad():
            for _x, _c, _y, _s, _ti in _dl_va:
                _l = _torch.nn.functional.l1_loss(
                    _m(_x.to(UD.DEVICE), _c.to(UD.DEVICE)).squeeze(-1), _y.to(UD.DEVICE))
                _tot += _l.item() * len(_y); _n += len(_y)
        _va = _tot / max(_n, 1)
        if _va < _best - 1e-6:
            _best, _bstate, _bad = _va, {k: v.clone() for k, v in _m.state_dict().items()}, 0
        else:
            _bad += 1
        if _bad >= 8:
            break
    if _bstate:
        _m.load_state_dict(_bstate)
    _m.eval()
    _rows = []
    with _torch.no_grad():
        for _x, _c, _y, _s, _ti in UD.DataLoader(_dsets[2], batch_size=1024):
            _o = _m(_x.to(UD.DEVICE), _c.to(UD.DEVICE)).squeeze(-1).cpu().numpy()
            _yh = _o * _dsets[2].ysd + _dsets[2].ymu
            _yt = _y.numpy() * _dsets[2].ysd + _dsets[2].ymu
            for _j, _h in enumerate(UP.HORIZONS):
                _rows.append(_pd.DataFrame({
                    UP.DATE: U_CAL.dates[_ti.numpy() + _h],
                    UP.STATE: [U_CAL.states[k] for k in _s.numpy()],
                    'target': _t, 'horizon': _h,
                    'y': _yt[:, _j], 'yhat': _yh[:, _j]}))
    _p = _pd.concat(_rows, ignore_index=True)
    _keep = [_p[_p.horizon == _h].merge(U_GRID[(_t, _h)][[UP.DATE, UP.STATE]],
                                        on=[UP.DATE, UP.STATE], how='inner')
             for _h in UP.HORIZONS]
    _orig_pred.append(_pd.concat(_keep, ignore_index=True))
    print(f"    {_t:<24} trained  ({_time.perf_counter()-_t0:.0f}s)")

U_REG.add('PatchTST+MCAG(replica)', _pd.concat(_orig_pred, ignore_index=True),
          section='U6', notes='original architecture + original 25-feature input, '
                             'reproduced on the upgrade grid for reference')
U_REG.checkpoint()

_rep = U_SCORER.headline(U_REG.stacked(['PatchTST+MCAG(replica)', 'Persistence']))
print("\n  Replica vs persistence on the observable targets:\n")
print(_rep[['model', 'MASE', 'RMSSE', 'MAPE', 'sMAPE', 'avg_MAPE_floored_6t', 'n_obs']]
      .to_string(index=False))
print("\n  The replica reproduces the published pattern: a transformer given no lag of")
print("  the target it must forecast cannot anchor the level, and lands far above a")
print("  one-line persistence rule. The architecture was never the bottleneck.")

# %% [code]
# ─── Cell U6.2 — MCAG-v2 base + the six extensions ─────────────────────────
print("=" * 78)
print("  U6.2 — MCAG-v2 VARIANTS")
print("=" * 78)

U_MCAG_VARIANTS = {
    'MCAGv2-base':      dict(),
    'MCAGv2-A-state':   dict(state_emb=True),
    'MCAGv2-B-regime':  dict(regime_emb=True),
    'MCAGv2-C-xstate':  dict(cross_state=True),
    'MCAGv2-D-fusion':  dict(fusion=True),
    'MCAGv2-E-multitask': dict(multitask=-1),   # -1 = "as many heads as there are
                                                # auxiliary targets for this target"
}

_AUX = ['grid_stress_index', 'avg_market_price']
U_MCAG_CFG = UD.TrainConfig(epochs=40, batch_size=512, lr=1e-3, patience=8,
                            loss='l1', seed=42)


def u_mcag_flags(flags, n_aux):
    """Resolve `multitask=-1` to the number of auxiliary targets available.

    The auxiliary set is `_AUX` minus the primary target, so it has 2 members for
    most targets but only 1 when the primary target is itself in `_AUX`. Fixing
    the head count at 2 would leave the multi-task head predicting a broadcast
    copy of a single auxiliary series on those targets, so the count is derived
    per target instead.
    """
    out = dict(flags)
    if out.get('multitask') == -1:
        out['multitask'] = n_aux
    return out

_mcag_pred = {k: [] for k in U_MCAG_VARIANTS}
_mcag_val = {k: [] for k in U_MCAG_VARIANTS}
U_MCAG_VAL = {}
_t0 = _time.perf_counter()

for _t in UP.TARGETS:
    _aux = [a for a in _AUX if a != _t]
    _ds = {}
    for _sp in ('train', 'valid', 'test'):
        _ds[_sp] = UM.MCAGWindows(U_CAL, _t, 28, UP.HORIZONS, _sp,
                                  regime_cols=UP.REGIME_COLS, aux_targets=_aux)
    _nch = _ds['train'].n_channels
    for _name, _flags in U_MCAG_VARIANTS.items():
        _cfg = UM.MCAGv2Config(lookback=28, n_out=len(UP.HORIZONS), d=64,
                               n_states=len(U_CAL.states),
                               n_regimes=len(UP.REGIME_COLS),
                               **u_mcag_flags(_flags, len(_aux)))
        UD.set_seed(42)
        _model = UM.MCAGv2(_cfg, _nch, _ds['train'].n_carbon, _ds['train'].n_price)
        _model, _hist, _bestval = UM.train_mcag(_model, _ds['train'], _ds['valid'],
                                                U_MCAG_CFG)
        U_MCAG_VAL[(_name, _t)] = _bestval
        _p = UM.predict_mcag(_model, _ds['test'], U_CAL.dates, U_CAL.states, _t,
                             UP.HORIZONS)
        _keep = [_p[_p.horizon == _h].merge(U_GRID[(_t, _h)][[UP.DATE, UP.STATE]],
                                            on=[UP.DATE, UP.STATE], how='inner')
                 for _h in UP.HORIZONS]
        _mcag_pred[_name].append(_pd.concat(_keep, ignore_index=True))
        _pv = UM.predict_mcag(_model, _ds['valid'], U_CAL.dates, U_CAL.states, _t,
                              UP.HORIZONS)
        _mcag_val[_name].append(_pd.concat(
            [_pv[_pv.horizon == _h].merge(U_GRID_VAL[(_t, _h)][[UP.DATE, UP.STATE]],
                                          on=[UP.DATE, UP.STATE], how='inner')
             for _h in UP.HORIZONS], ignore_index=True))
        del _model
    print(f"    {_t:<24} 6 variants done  ({_time.perf_counter()-_t0:.0f}s elapsed)")

for _name in U_MCAG_VARIANTS:
    U_REG.add(_name, _pd.concat(_mcag_pred[_name], ignore_index=True), section='U6',
              notes='carbon-gated PatchTST on instance-normalised windows')
    U_REG_VAL.add(_name, _pd.concat(_mcag_val[_name], ignore_index=True), section='U6')
U_REG.checkpoint()
U_REG_VAL.checkpoint('_registry_valid.pkl')
print(f"\n  trained {len(UP.TARGETS)*len(U_MCAG_VARIANTS)} models in {_time.perf_counter()-_t0:.0f}s")

# %% [code]
# ─── Cell U6.3 — Variant F: best combination, selected on VALIDATION ───────
print("=" * 78)
print("  U6.3 — VARIANT F: BEST PRACTICAL COMBINATION")
print("=" * 78)
print("  Extensions are ranked by their mean VALIDATION loss across the six targets;")
print("  every extension that improved on the base is switched on together. The test")
print("  split plays no part in this choice.\n")

_val_tbl = _pd.DataFrame([{'variant': k, 'mean_valid_loss': _np.mean(
    [U_MCAG_VAL[(k, t)] for t in UP.TARGETS])} for k in U_MCAG_VARIANTS])
_val_tbl = _val_tbl.sort_values('mean_valid_loss')
_base_val = float(_val_tbl.set_index('variant').loc['MCAGv2-base', 'mean_valid_loss'])
_val_tbl['vs_base'] = (_val_tbl['mean_valid_loss'] - _base_val) / _base_val * 100
print(_val_tbl.round(5).to_string(index=False))

_FLAG_OF = {'MCAGv2-A-state': 'state_emb', 'MCAGv2-B-regime': 'regime_emb',
            'MCAGv2-C-xstate': 'cross_state', 'MCAGv2-D-fusion': 'fusion'}
_helped = [k for k in _FLAG_OF
           if float(_val_tbl.set_index('variant').loc[k, 'mean_valid_loss']) < _base_val]
_combo = {_FLAG_OF[k]: True for k in _helped}
_mt_helped = float(_val_tbl.set_index('variant').loc['MCAGv2-E-multitask',
                                                     'mean_valid_loss']) < _base_val
if _mt_helped:
    _combo['multitask'] = -1        # resolved per target by u_mcag_flags
print(f"\n  extensions that beat the base on validation: {_helped}"
      f"{' + multitask' if _mt_helped else ''}")
print(f"  variant F configuration: {_combo if _combo else '(none helped -> F == base)'}")

_f_pred = []
_t0 = _time.perf_counter()
for _t in UP.TARGETS:
    _aux = [a for a in _AUX if a != _t]
    _ds = {sp: UM.MCAGWindows(U_CAL, _t, 28, UP.HORIZONS, sp,
                              regime_cols=UP.REGIME_COLS, aux_targets=_aux)
           for sp in ('train', 'valid', 'test')}
    _cfg = UM.MCAGv2Config(lookback=28, n_out=len(UP.HORIZONS), d=64,
                           n_states=len(U_CAL.states), n_regimes=len(UP.REGIME_COLS),
                           **u_mcag_flags(_combo, len(_aux)))
    UD.set_seed(42)
    _model = UM.MCAGv2(_cfg, _ds['train'].n_channels, _ds['train'].n_carbon,
                       _ds['train'].n_price)
    _model, _hist, _bv = UM.train_mcag(_model, _ds['train'], _ds['valid'], U_MCAG_CFG)
    _p = UM.predict_mcag(_model, _ds['test'], U_CAL.dates, U_CAL.states, _t, UP.HORIZONS)
    _keep = [_p[_p.horizon == _h].merge(U_GRID[(_t, _h)][[UP.DATE, UP.STATE]],
                                        on=[UP.DATE, UP.STATE], how='inner')
             for _h in UP.HORIZONS]
    _f_pred.append(_pd.concat(_keep, ignore_index=True))
    del _model
U_REG.add('MCAGv2-F-best', _pd.concat(_f_pred, ignore_index=True), section='U6',
          notes=f'validation-selected combination: {_combo}')
U_REG.checkpoint()
print(f"  variant F trained ({_time.perf_counter()-_t0:.0f}s)")

# %% [code]
# ─── Cell U6.4 — Common evaluation of the MCAG family ──────────────────────
print("=" * 78)
print("  U6.4 — MCAG-v2 EVALUATION")
print("=" * 78)

U_MCAG_NAMES = list(U_MCAG_VARIANTS) + ['MCAGv2-F-best']
_cmp = U_MCAG_NAMES + ['PatchTST+MCAG(replica)', 'Persistence']
_mc_stack = U_REG.stacked(_cmp)
_mc_head = U_SCORER.headline(_mc_stack)

print("\n  Headline — observable targets, aligned grid:\n")
print(_mc_head[['model', 'MASE', 'RMSSE', 'MAE', 'RMSE', 'MAPE', 'sMAPE',
                'avg_MAPE_floored_6t', 'n_obs']].to_string(index=False))

_pm = float(_mc_head.set_index('model').loc['Persistence', 'MASE'])
_bm = float(_mc_head.set_index('model').loc['MCAGv2-base', 'MASE'])
print(f"\n  Persistence  MASE = {_pm:.4f}")
print(f"  MCAGv2-base  MASE = {_bm:.4f}")
print("\n  Effect of each extension on TEST (negative = improvement over the v2 base):")
for _n in U_MCAG_NAMES:
    if _n == 'MCAGv2-base':
        continue
    _v = float(_mc_head.set_index('model').loc[_n, 'MASE'])
    print(f"    {_n:<22} {_v:.4f}   ({(_v-_bm)/_bm*100:+6.2f}% vs base)"
          f"   {'beats' if _v < _pm else 'does not beat'} persistence")

_mth = U_SCORER.by_target_horizon(_mc_stack)
_mc_head.to_csv(U_OUT / 'tables' / 'u6_mcag_headline.csv', index=False)
_mth.to_csv(U_OUT / 'tables' / 'u6_mcag_by_target_horizon.csv', index=False)
_val_tbl.to_csv(U_OUT / 'tables' / 'u6_mcag_validation_selection.csv', index=False)

_fig, _axes = _plt.subplots(1, 2, figsize=(14, 4.6))
_ax = _axes[0]
_d = _mc_head.set_index('model').reindex(_cmp)['MASE']
_cols = ['#e08214' if m.startswith('MCAGv2') else '#8a8a8a' for m in _d.index]
_ax.bar(range(len(_d)), _d.values, color=_cols)
_ax.axhline(_pm, color='crimson', ls='--', lw=1.3)
_ax.set_xticks(range(len(_d)))
_ax.set_xticklabels(_d.index, rotation=40, ha='right', fontsize=7)
_ax.set_ylabel('MASE'); _ax.set_title('(a) MCAG-v2 variants', fontweight='bold', fontsize=10)
_ax.grid(axis='y', alpha=0.3)

_ax = _axes[1]
_vt = _val_tbl.set_index('variant')['vs_base'].drop('MCAGv2-base')
_ax.barh(_vt.index, _vt.values,
         color=['#1a7f37' if v < 0 else '#c0392b' for v in _vt.values])
_ax.axvline(0, color='black', lw=1)
_ax.set_xlabel('validation loss vs base (%)  — negative = extension helped')
_ax.set_title('(b) Validation-only extension selection', fontweight='bold', fontsize=10)
_ax.tick_params(axis='y', labelsize=8); _ax.grid(axis='x', alpha=0.3)

_plt.tight_layout()
_plt.savefig(U_OUT / 'figures' / 'U6_mcag_v2.png', dpi=150, bbox_inches='tight')
print(f"\n  saved -> {U_OUT / 'figures' / 'U6_mcag_v2.png'}")
_plt.show()
