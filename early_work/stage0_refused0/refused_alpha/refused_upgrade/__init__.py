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
