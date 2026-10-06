"""
units.py — data & unit hygiene for RE-FUSED-Alpha (Task #2).

Two problems this module addresses:
  1. UNIT AMBIGUITY: energy columns mix GWh/MWh and emissions mix tons/kilotons
     across upstream sources, so the same physical quantity appears at 1000x
     different magnitudes.  We (a) declare a canonical unit per column, (b) flag
     columns whose within-column magnitude jumps look like a unit switch, and
     (c) provide explicit, auditable conversions.
  2. HIDDEN IMPUTATION: `*_imputed_flag` columns silently mark filled rows.
     Models trained/scored on filled rows can look better than they are.  We
     report imputation rates and produce an "observed-only" robustness slice.

Pure pandas/numpy. Nothing here mutates in place unless a `_inplace` helper is
named as such; `normalize_units` returns a new frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- registry
# Canonical unit of the VALUES AS STORED in data/RL_*.csv, per LEAKAGE_AUDIT.md
# §6: state-day generation ~70 is only physically plausible in GWh (Bihar is
# ~150 GWh/day), so the `_mwh` suffixes are mislabels; `carbon_proxy_tons` is
# consistent only as kilotons. Unknown columns are ignored (reported, not
# touched).
CANONICAL_UNITS: Dict[str, str] = {
    "total_generation_mwh": "GWh",        # mislabeled `_mwh`
    "renewable_generation_mwh": "GWh",    # mislabeled `_mwh`
    "non_renewable_gen": "GWh",
    "consumption_mwh": "GWh",             # mislabeled `_mwh`
    "demand_deficit_mwh": "GWh",          # mislabeled `_mwh`
    "demand_surplus_mwh": "GWh",          # mislabeled `_mwh`
    "gen_ytd": "GWh",
    "national_gen_avg": "GWh",
    "national_ren_avg": "GWh",
    "total_outage_mw": "MW",
    "avg_coal_stock_days": "days",
    "avg_market_price": "INR_per_MWh",    # CONFIRM vs. IEX source (Rs/MWh vs Rs/kWh)
    "carbon_proxy_tons": "kilotons",      # mislabeled `_tons`
    "carbon_avoided_tons": "kilotons",    # mislabeled `_tons`
    "carbon_intensity": "tons_per_MWh",
    "supply_demand_gap": "GWh",
}

# Corrected column names for publication (audit §6 fix: "rename columns / add
# an explicit units table"). Values are NOT rescaled — only the label changes.
RENAME_MAP: Dict[str, str] = {
    "total_generation_mwh": "total_generation_gwh",
    "renewable_generation_mwh": "renewable_generation_gwh",
    "consumption_mwh": "consumption_gwh",
    "demand_deficit_mwh": "demand_deficit_gwh",
    "demand_surplus_mwh": "demand_surplus_gwh",
    "carbon_proxy_tons": "carbon_proxy_kilotons",
    "carbon_avoided_tons": "carbon_avoided_kilotons",
}


def rename_to_true_units(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the mislabeled columns renamed to their true units.
    No values are changed — this is a label-only fix."""
    return df.rename(columns=RENAME_MAP)

# Known scalar conversions TO the canonical unit. key = (from_unit, to_unit).
_CONVERSIONS: Dict[tuple, float] = {
    ("MWh", "GWh"): 1e-3,
    ("GWh", "MWh"): 1e3,
    ("kWh", "GWh"): 1e-6,
    ("tons", "kilotons"): 1e-3,
    ("kilotons", "tons"): 1e3,
    ("kg", "tons"): 1e-3,
}


def convert(series: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
    if from_unit == to_unit:
        return series
    key = (from_unit, to_unit)
    if key not in _CONVERSIONS:
        raise KeyError(f"no declared conversion {from_unit}->{to_unit}")
    return series * _CONVERSIONS[key]


# ------------------------------------------------------------- unit-jump detector
@dataclass
class UnitJumpFinding:
    column: str
    ratio_p95_p05: float          # spread of magnitudes within the column
    suspected_multiplier: Optional[int]  # e.g. 1000 if a GWh/MWh mix is likely
    note: str


def detect_unit_jumps(df: pd.DataFrame,
                      cols: Optional[List[str]] = None,
                      jump_ratio: float = 100.0) -> List[UnitJumpFinding]:
    """Heuristic: within a single physical column, if the ratio of large to
    small positive values spans ~1000x it usually means two units were merged.
    We look at the 95th/5th percentile ratio of |nonzero| values."""
    cols = cols or [c for c in df.columns if df[c].dtype.kind in "fi"]
    out: List[UnitJumpFinding] = []
    for c in cols:
        v = np.abs(pd.to_numeric(df[c], errors="coerce").dropna().values)
        v = v[v > 0]
        if len(v) < 20:
            continue
        p05, p95 = np.percentile(v, [5, 95])
        if p05 == 0:
            continue
        ratio = float(p95 / p05)
        if ratio >= jump_ratio:
            mult = None
            for m in (1000, 100, 1_000_000):
                if abs(ratio - m) / m < 0.5:
                    mult = m
                    break
            out.append(UnitJumpFinding(
                column=c, ratio_p95_p05=round(ratio, 1),
                suspected_multiplier=mult,
                note=("likely unit switch (e.g. GWh<->MWh / tons<->kilotons); "
                      "inspect source files before modeling"),
            ))
    return out


def units_table(df: pd.DataFrame) -> pd.DataFrame:
    """A publishable table: each numeric column, its declared canonical unit
    (or 'UNDECLARED'), observed min/median/max, and a unit-jump flag."""
    rows = []
    jumps = {f.column: f for f in detect_unit_jumps(df)}
    for c in df.columns:
        if df[c].dtype.kind not in "fi":
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append({
            "column": c,
            "canonical_unit": CANONICAL_UNITS.get(c, "UNDECLARED"),
            "min": float(v.min()) if len(v) else np.nan,
            "median": float(v.median()) if len(v) else np.nan,
            "max": float(v.max()) if len(v) else np.nan,
            "unit_jump_flag": c in jumps,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- imputation report
@dataclass
class ImputationReport:
    per_column: pd.DataFrame = field(default_factory=pd.DataFrame)
    any_imputed_rate: float = 0.0

    def __str__(self) -> str:
        return (f"imputed rows (any flag): {self.any_imputed_rate:.1%}\n"
                + self.per_column.to_string(index=False))


def imputation_report(df: pd.DataFrame, flag_suffix: str = "_imputed_flag") -> ImputationReport:
    """Summarise every `*_imputed_flag` column: what fraction of rows were
    filled, and (union) how many rows have ANY imputed value."""
    flags = [c for c in df.columns if c.endswith(flag_suffix)]
    if not flags:
        return ImputationReport(pd.DataFrame(columns=["column", "imputed_rate"]), 0.0)
    per = pd.DataFrame({
        "column": [c.replace(flag_suffix, "") for c in flags],
        "imputed_rate": [float(pd.to_numeric(df[c], errors="coerce").fillna(0).mean())
                         for c in flags],
    }).sort_values("imputed_rate", ascending=False)
    any_mask = np.zeros(len(df), dtype=bool)
    for c in flags:
        any_mask |= pd.to_numeric(df[c], errors="coerce").fillna(0).astype(bool).values
    return ImputationReport(per, float(any_mask.mean()))


def observed_only(df: pd.DataFrame, flag_suffix: str = "_imputed_flag") -> pd.DataFrame:
    """Robustness slice: rows with NO imputed values on any flagged column.
    Re-run headline metrics on this slice; if results collapse, imputation was
    doing the heavy lifting and must be disclosed."""
    flags = [c for c in df.columns if c.endswith(flag_suffix)]
    if not flags:
        return df.copy()
    keep = np.ones(len(df), dtype=bool)
    for c in flags:
        keep &= ~pd.to_numeric(df[c], errors="coerce").fillna(0).astype(bool).values
    return df.loc[keep].copy()


__all__ = [
    "CANONICAL_UNITS", "RENAME_MAP", "convert", "rename_to_true_units",
    "detect_unit_jumps", "units_table",
    "imputation_report", "observed_only", "UnitJumpFinding", "ImputationReport",
]
