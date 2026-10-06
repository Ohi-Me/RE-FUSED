# %% [markdown]
# ---
# ## Section U8 — Time-Series Foundation Models
#
# Zero-shot foundation models (TimesFM, Chronos / Chronos-Bolt) are the obvious
# recent comparison point. They are only worth including if they can be run
# **without disturbing the environment that produced the original results** —
# a torch downgrade or a numpy bump would silently invalidate every number in
# Part I of this notebook.
#
# So this section checks availability first and reports honestly:
#
# * if a compatible model is importable, it is run zero-shot on the same grid;
# * if not, the section prints **"Not available — skipped."** and moves on.
#
# No result is ever fabricated, and no package is installed that would alter the
# pinned `torch` / `numpy` / `pandas` versions.

# %% [code]
# ─── Cell U8.1 — Availability check ─────────────────────────────────────────
import importlib as _il

print("=" * 78)
print("  U8.1 — FOUNDATION MODEL AVAILABILITY")
print("=" * 78)

_PINNED = {'torch': _torch.__version__ if '_torch' in dir() else 'n/a',
           'numpy': _np.__version__, 'pandas': _pd.__version__}
print(f"  environment currently pinned at: {_PINNED}")
print("  a foundation model is only used if it imports WITHOUT changing these.\n")

U_FM_CANDIDATES = ['chronos', 'timesfm', 'momentfm', 'uni2ts']
U_FM_STATUS = {}
for _mod in U_FM_CANDIDATES:
    try:
        _il.import_module(_mod)
        U_FM_STATUS[_mod] = 'available'
    except Exception as _e:
        U_FM_STATUS[_mod] = f'not installed ({type(_e).__name__})'
    print(f"    {_mod:<12} {U_FM_STATUS[_mod]}")

U_FM_AVAILABLE = [k for k, v in U_FM_STATUS.items() if v == 'available']
print()
if not U_FM_AVAILABLE:
    print("  Not available — skipped.")
    print()
    print("  Rationale for not installing one here: `chronos-forecasting` and")
    print("  `timesfm` both pull a transformers/accelerate stack and pin their own")
    print("  torch build. This environment holds torch 2.5.1+cu121, which produced")
    print("  every result in Part I of this notebook. Silently resolving a different")
    print("  torch to add one zero-shot baseline would put the original")
    print("  reproducibility record at risk for a comparison that is not required")
    print("  by any claim made in Part II.")
    print()
    print("  To add it deliberately in an isolated environment:")
    print("      python -m venv .venv-fm && .venv-fm/Scripts/pip install chronos-forecasting")
    print("      # then score it on results/forecasting_upgrade/tables/ grids")
else:
    print(f"  Available: {U_FM_AVAILABLE} — running zero-shot below.")

_pd.DataFrame([{'model': k, 'status': v} for k, v in U_FM_STATUS.items()]).to_csv(
    U_OUT / 'tables' / 'u8_foundation_availability.csv', index=False)

# %% [code]
# ─── Cell U8.2 — Zero-shot run (only if a model is importable) ─────────────
U_FM_NAMES = []
if not U_FM_AVAILABLE:
    print("=" * 78)
    print("  U8.2 — Not available — skipped.")
    print("=" * 78)
    print("  No foundation-model row appears in the Section U14 leaderboard.")
    print("  This is recorded as a gap in the comparison, not as a negative result:")
    print("  nothing here licenses any claim about how TimesFM or Chronos would")
    print("  perform on this panel.")
else:
    print("=" * 78)
    print("  U8.2 — ZERO-SHOT FOUNDATION MODEL")
    print("=" * 78)
    _fm_pred = []
    if 'chronos' in U_FM_AVAILABLE:
        from chronos import ChronosPipeline
        _pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small",
                                                device_map=str(UD.DEVICE),
                                                torch_dtype=_torch.float32)
        _ctx = 128
        for _t in UP.TARGETS:
            _yw = U_CAL.wide(_t)
            _rows = []
            for _st in U_CAL.states:
                _s = _yw[_st]
                for _h in UP.HORIZONS:
                    _g = U_GRID[(_t, _h)]
                    _g = _g[_g[UP.STATE] == _st]
                    for _, _r in _g.iterrows():
                        _origin = _r[UP.DATE] - _pd.Timedelta(days=_h)
                        _hist = _s.loc[:_origin].dropna().to_numpy()[-_ctx:]
                        if len(_hist) < 16:
                            continue
                        _fc = _pipe.predict(_torch.tensor(_hist, dtype=_torch.float32),
                                            prediction_length=_h, num_samples=20)
                        _rows.append({UP.DATE: _r[UP.DATE], UP.STATE: _st,
                                      'target': _t, 'horizon': _h, 'y': _r['y'],
                                      'yhat': float(_fc[0, :, -1].median())})
            _fm_pred.append(_pd.DataFrame(_rows))
        U_REG.add('Chronos-zeroshot', _pd.concat(_fm_pred, ignore_index=True),
                  section='U8', notes='zero-shot, context 128, median of 20 samples')
        U_FM_NAMES.append('Chronos-zeroshot')
        U_REG.checkpoint()
        print(U_SCORER.headline(U_REG.stacked(U_FM_NAMES + ['Persistence']))
              [['model', 'MASE', 'RMSSE', 'MAE', 'sMAPE', 'n_obs']].to_string(index=False))
