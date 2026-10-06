"""
_smoke.py — DEVELOPMENT ONLY. Not part of the notebook.

Runs the entire Part II sequence at drastically reduced scale so that runtime
errors surface in minutes instead of hours. It changes only cost knobs — number
of targets, boosting rounds, training epochs — never any modelling logic, so a
clean smoke run means the code paths are sound, NOT that the numbers are final.

Smoke output is redirected to `results/forecasting_upgrade_smoke/` so it can
never overwrite a real run.
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(globals().get("__file__", "_smoke.py")).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from refused_upgrade import panel as _UP, trees as _UT, deep as _UD, mcag as _UM

# --- redirect all output ---------------------------------------------------
_orig_paths = _UP.repo_paths


def _smoke_paths(start=None):
    d = _orig_paths(start)
    d["upgrade"] = d["results"] / "forecasting_upgrade_smoke"
    for sub in ("tables", "figures", "outputs", "models"):
        (d["upgrade"] / sub).mkdir(parents=True, exist_ok=True)
    return d


_UP.repo_paths = _smoke_paths

# --- shrink the workload ---------------------------------------------------
_UP.TARGETS[:] = ["total_generation_mwh", "avg_market_price"]
_UT.MAX_ROUNDS = 60
_UT.EARLY_STOP = 20

_orig_train = _UD.train_model


def _fast_train(model, ds_tr, ds_va, cfg):
    cfg.epochs = min(cfg.epochs, 3)
    cfg.patience = min(cfg.patience, 2)
    return _orig_train(model, ds_tr, ds_va, cfg)


_UD.train_model = _fast_train

_orig_mcag = _UM.train_mcag


def _fast_mcag(model, ds_tr, ds_va, cfg, aux_weight=0.3):
    cfg.epochs = min(cfg.epochs, 3)
    cfg.patience = min(cfg.patience, 2)
    return _orig_mcag(model, ds_tr, ds_va, cfg, aux_weight)


_UM.train_mcag = _fast_mcag

print("[smoke] targets=%s  MAX_ROUNDS=%d  epochs<=3" % (_UP.TARGETS, _UT.MAX_ROUNDS))

# --- then the normal bootstrap ---------------------------------------------
_boot = (_HERE / "_bootstrap.py").read_text(encoding="utf-8")
exec(compile(_boot, str(_HERE / "_bootstrap.py"), "exec"), globals())
