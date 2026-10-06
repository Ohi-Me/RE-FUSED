"""
_bootstrap.py — DEVELOPMENT ONLY. Not part of the notebook.

Recreates the cheap shared state (imports, panel, scorer, registry, evaluation
grids) and reloads any checkpointed model predictions, so a single section can
be developed in isolation without re-running the expensive upstream fits.

The notebook itself never uses this file: there, the same state is created by
running cells U1.1 and U2.1 in order.
"""
from __future__ import annotations

import sys as _sys
import pathlib as _pl
import time as _time
import json as _json
import warnings as _w

_w.filterwarnings("ignore")

for _p in (_pl.Path.cwd(), _pl.Path.cwd().parent, _pl.Path.cwd().parent.parent):
    if (_p / "refused_upgrade").exists():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break

import numpy as _np
import pandas as _pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt

from refused_upgrade import panel as UP, features as UF, evaluation as UE, audit as UA
from refused_upgrade import classical as UC

_pd.set_option("display.width", 200)
_pd.set_option("display.max_columns", 60)

U_PATHS = UP.repo_paths()
U_OUT = U_PATHS["upgrade"]

U_DF, U_SPLITS, U_CAL = UP.load_panel(U_PATHS)
U_SCORER = UE.Scorer.from_panel(U_DF)
U_REG = UE.Registry(out_dir=U_OUT)
U_REG_VAL = UE.Registry(out_dir=U_OUT)
_n = U_REG.restore()
_nv = U_REG_VAL.restore("_registry_valid.pkl")

U_REGIME_MAP = U_DF[[UP.DATE, UP.STATE] + UP.REGIME_COLS].copy()

U_GRID, U_GRID_VAL = {}, {}
for _t in UP.TARGETS:
    for _h in UP.HORIZONS:
        U_GRID[(_t, _h)] = UC.eval_grid(U_CAL, _t, _h, split="test")
        U_GRID_VAL[(_t, _h)] = UC.eval_grid(U_CAL, _t, _h, split="valid")


def u_register_wide(name, wide_by_h, section, notes="", grid=None, **extra):
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


def u_frame(rows, yhat, target, horizon):
    """Canonical prediction frame from a returned row slice."""
    return _pd.DataFrame({
        UP.DATE: rows[UP.DATE].to_numpy(), UP.STATE: rows[UP.STATE].to_numpy(),
        "target": target, "horizon": horizon,
        "y": rows["y"].to_numpy(float), "yhat": _np.asarray(yhat, float)})


U_BASELINE_NAMES = ["Persistence", "SeasonalNaive", "RollingMean", "Drift", "ETS", "ARIMA"]

print(f"[bootstrap] panel {U_DF.shape}, restored {_n} checkpointed model(s): "
      f"{sorted(U_REG.preds)}")
