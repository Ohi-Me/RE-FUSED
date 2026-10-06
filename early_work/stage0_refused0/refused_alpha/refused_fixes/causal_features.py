"""
causal_features.py — leakage-free (causal) recomputation of RE-FUSED input features.

Context
-------
The notebook's model-input standardization is already correct (train-only:
`fm/fs` from df_train). The residual problem is that a few engineered columns
shipped *inside* RL_TRAIN_DATA.csv / RL_TEST_DATA.csv were produced by an
upstream script over the FULL 2017-2025 panel. Any column that is a min-max
normalization, a global percentile/CVaR threshold, a cross-sectional rank/tier,
or a whole-panel z-score leaks test-period information into training features.
Re-standardizing train-only does NOT undo a leak already encoded in the value.

Confirmed leakage-prone model inputs (present in the notebook FEATS list):
    price_norm         min-max of avg_market_price        -> global min/max
    lcmp_carbon        uses carbon_intensity 5th/95th pct  -> global percentiles
    grid_stress_index  composite of globally-normed parts  -> propagated
    discom_stress      0.5*price_norm + 0.5*GSI            -> inherits the above

Usage in the notebook (replace the CSV-baked columns before building FEATS):

    from refused_fixes.causal_features import CausalFeatureFitter, leakage_scan

    # optional one-line diagnostic you can run today:
    print(leakage_scan(df_train, df_test).to_string())

    fitter = CausalFeatureFitter().fit(df_train)     # learns stats on TRAIN only
    df_train = fitter.transform(df_train)            # recompute leaky cols
    df_test  = fitter.transform(df_test)             # same frozen params
    # ... then build FEATS / standardize with fm,fs exactly as before.

Everything here is pure pandas/numpy and side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

DATE = "date"
STATE = "state_name"

# GSI reconstruction — CONFIRM these against your upstream grid_stress_index
# formula. Per related_works/Dataset_and_Features.md it is a [0,1] composite of
# "outage, SD gap, coal". Adjust the component list / weights to match exactly.
GSI_COMPONENTS = ("outage_ratio", "supply_demand_gap", "coal_outage_stress")
GSI_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)

LCMP_CI_COL = "carbon_intensity"   # CI in the lcmp_carbon formula
LCMP_SCALE = 0.30                  # the 0.30 multiplier in the documented formula


def _minmax_apply(s: pd.Series, lo: float, hi: float) -> pd.Series:
    rng = hi - lo
    if not np.isfinite(rng) or rng == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / rng).clip(0.0, 1.0)


@dataclass
class CausalFeatureFitter:
    """Learns every normalization statistic on the TRAIN split only, then
    applies the frozen parameters to any split (train or test)."""

    per_state: bool = True                    # fit min/max & percentiles per state
    price_col: str = "avg_market_price"

    # learned parameters (populated by .fit)
    price_lo: Dict = field(default_factory=dict)
    price_hi: Dict = field(default_factory=dict)
    ci_q05: Dict = field(default_factory=dict)
    ci_q95: Dict = field(default_factory=dict)
    gsi_lo: Dict = field(default_factory=dict)
    gsi_hi: Dict = field(default_factory=dict)
    train_medians: Optional[pd.Series] = None
    _fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df_train: pd.DataFrame) -> "CausalFeatureFitter":
        d = df_train
        keys = d[STATE].unique() if self.per_state else ["__ALL__"]

        def groups(col):
            if self.per_state:
                return {k: d.loc[d[STATE] == k, col] for k in keys}
            return {"__ALL__": d[col]}

        pg = groups(self.price_col)
        self.price_lo = {k: float(np.nanmin(v)) for k, v in pg.items()}
        self.price_hi = {k: float(np.nanmax(v)) for k, v in pg.items()}

        if LCMP_CI_COL in d.columns:
            cg = groups(LCMP_CI_COL)
            self.ci_q05 = {k: float(np.nanquantile(v, 0.05)) for k, v in cg.items()}
            self.ci_q95 = {k: float(np.nanquantile(v, 0.95)) for k, v in cg.items()}

        for c in GSI_COMPONENTS:
            if c in d.columns:
                self.gsi_lo[c] = float(np.nanmin(d[c]))
                self.gsi_hi[c] = float(np.nanmax(d[c]))

        num = d.select_dtypes(include=[np.number]).columns
        self.train_medians = d[num].median()
        self._fitted = True
        return self

    # ------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call .fit(df_train) before .transform().")
        out = df.copy()

        # leakage-free median imputation using TRAIN medians
        num = out.select_dtypes(include=[np.number]).columns
        shared = [c for c in num if c in self.train_medians.index]
        out[shared] = out[shared].fillna(self.train_medians[shared])

        def key_series(idx_frame):
            if self.per_state:
                return idx_frame[STATE]
            return pd.Series("__ALL__", index=idx_frame.index)

        ks = key_series(out)

        # --- price_norm : train-fit min-max ---------------------------------
        if self.price_col in out.columns:
            lo = ks.map(self.price_lo)
            hi = ks.map(self.price_hi)
            out["price_norm"] = _minmax_apply(out[self.price_col], lo, hi) \
                if not self.per_state else \
                ((out[self.price_col] - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1).fillna(0.0)

        # --- lcmp_carbon : train-fit CI percentiles -------------------------
        if LCMP_CI_COL in out.columns and self.ci_q05:
            q05 = ks.map(self.ci_q05)
            q95 = ks.map(self.ci_q95)
            denom = (q95 - q05).replace(0, np.nan)
            ci_scaled = ((out[LCMP_CI_COL] - q05) / denom).clip(0, 1).fillna(0.0)
            out["lcmp_carbon"] = ci_scaled * out[self.price_col] * LCMP_SCALE

        # --- grid_stress_index : train-fit component min-max ----------------
        present = [c for c in GSI_COMPONENTS if c in out.columns and c in self.gsi_lo]
        if present:
            acc = np.zeros(len(out))
            wsum = 0.0
            for c, w in zip(GSI_COMPONENTS, GSI_WEIGHTS):
                if c in present:
                    acc = acc + w * _minmax_apply(out[c], self.gsi_lo[c], self.gsi_hi[c]).to_numpy()
                    wsum += w
            if wsum > 0:
                out["grid_stress_index"] = np.clip(acc / wsum, 0.0, 1.0)

        # --- discom_stress : depends on the two fixed inputs ----------------
        if {"price_norm", "grid_stress_index"}.issubset(out.columns):
            out["discom_stress"] = 0.5 * out["price_norm"] + 0.5 * out["grid_stress_index"]

        return out


# ----------------------------------------------------------------------------
# Diagnostic — run this today to *measure* the leak before/after fixing.
# ----------------------------------------------------------------------------
SUSPECT_COLS = (
    "price_norm", "lcmp_carbon", "grid_stress_index", "discom_stress",
    "gen_zscore", "price_zscore", "outage_zscore", "ren_zscore",
    "gen_cvar90", "outage_cvar90", "price_cvar90",
    "state_gen_norm", "state_outage_norm", "state_coal_norm",
    "state_ren_norm", "state_price_norm", "state_gen_rank", "state_gen_tier",
)


def leakage_scan(df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 cols: Iterable[str] = SUSPECT_COLS) -> pd.DataFrame:
    """For each suspect column, flag signs that it encodes whole-panel stats.

    Heuristics (no upstream script needed):
      * test_out_of_train_range : fraction of test rows whose value lies outside
        the observed TRAIN range. For a correctly train-fit min-max column this
        is ~0 for [0,1] features; large values suggest the column was scaled with
        global (train+test) extremes, i.e. leakage OR at least distribution shift.
      * refit_max_abs_delta : if the column can be recomputed train-only
        (price_norm / lcmp_carbon / grid_stress_index), the largest absolute
        change on the TEST split after refitting. >~0.01 means the shipped values
        used non-train statistics.
    """
    rows = []
    fitter = CausalFeatureFitter().fit(df_train)
    fixed_test = fitter.transform(df_test)
    for c in cols:
        if c not in df_train.columns or c not in df_test.columns:
            continue
        tr = df_train[c].astype(float)
        te = df_test[c].astype(float)
        lo, hi = np.nanmin(tr), np.nanmax(tr)
        oob = float(((te < lo) | (te > hi)).mean())
        delta = np.nan
        if c in fixed_test.columns:
            delta = float(np.nanmax(np.abs(fixed_test[c].astype(float) - te)))
        rows.append({
            "column": c,
            "train_min": round(lo, 4), "train_max": round(hi, 4),
            "test_out_of_train_range": round(oob, 4),
            "refit_max_abs_delta": (round(delta, 4) if np.isfinite(delta) else None),
        })
    return pd.DataFrame(rows)


__all__ = ["CausalFeatureFitter", "leakage_scan", "SUSPECT_COLS"]
