"""
refused_upgrade — forecasting-upgrade harness for RE-FUSED-Alpha.

This package is ADDITIVE. It does not import from, patch, or mutate any part of
the original notebook pipeline; `refused_fixes` is imported read-only so that the
new work reuses the *already published* metric implementations (MASE / RMSSE /
sMAPE / Diebold-Mariano) rather than re-deriving them. Every number produced
here is therefore on the same metric code path as
`results/tables/baseline_metrics.csv`.

Modules
-------
panel       isolated, calendar-aware panel construction + chronological splits
features    strictly-lagged, prediction-time-safe feature builder
evaluation  unified h=1..3 evaluation protocol, metric suite, results registry
audit       leakage / feature-provenance diagnostics
classical   rolling-origin statistical baselines (persistence/snaive/ETS/ARIMA)
trees       LightGBM/XGBoost/CatBoost under one fixed 4-step protocol
deep        PyTorch harness (N-BEATS/N-HiTS/TSMixer) -- requires torch
mcag        MCAG-v2 carbon-aware gating + PatchTST replica -- requires torch

`deep` and `mcag` import torch at module scope and are deliberately NOT
eager-imported below, so `import refused_upgrade` keeps working in an
environment without torch installed (panel/features/evaluation/audit/classical
have no torch dependency). Import them explicitly:
    from refused_upgrade import deep, mcag
when torch is available.

Design rules enforced throughout
--------------------------------
1. A feature for target date D at horizon h may only use information dated
   <= D - h (the forecast origin), plus deterministic calendar attributes of D.
2. Every distributional statistic (median, quantile, min/max, mean/std) is fit
   on the TRAIN split only and frozen before being applied to validation/test.
3. Validation is a chronological block carved out of TRAIN. The TEST split is
   never used for feature selection, tuning, model selection, ensemble weights
   or seed selection.
4. Lags and rolling windows are computed on a complete daily CALENDAR grid per
   state, so a "lag 1" is always exactly one calendar day, never "the previous
   surviving row".
"""
from __future__ import annotations

__version__ = "1.0.0"

from . import panel, features, evaluation, audit  # noqa: F401

__all__ = ["panel", "features", "evaluation", "audit", "__version__"]
