"""refused_fixes — drop-in fixes for the RE-FUSED-Alpha audit (see ../LEAKAGE_AUDIT.md).

Task #1  causal_features : train-only refit of leaky engineered columns + leakage_scan
Task #2  units           : unit registry / rename map, unit-jump detector, imputation report
Task #3  metrics         : MASE / RMSSE / sMAPE, pinball / CRPS, coverage, Diebold-Mariano
"""
from .causal_features import CausalFeatureFitter, SUSPECT_COLS, leakage_scan
from .metrics import (
    crps_from_quantiles,
    diebold_mariano,
    interval_coverage,
    mase,
    pinball_loss,
    point_report,
    prob_report,
    rmsse,
    smape,
)
from .units import (
    CANONICAL_UNITS,
    RENAME_MAP,
    convert,
    detect_unit_jumps,
    imputation_report,
    observed_only,
    rename_to_true_units,
    units_table,
)

__all__ = [
    "CausalFeatureFitter", "SUSPECT_COLS", "leakage_scan",
    "smape", "mase", "rmsse", "pinball_loss", "crps_from_quantiles",
    "interval_coverage", "diebold_mariano", "point_report", "prob_report",
    "CANONICAL_UNITS", "RENAME_MAP", "convert", "rename_to_true_units",
    "detect_unit_jumps", "units_table", "imputation_report", "observed_only",
]
