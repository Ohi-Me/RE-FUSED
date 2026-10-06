"""
panel.py — isolated, calendar-aware panel construction for the forecasting upgrade.

Why a new loader instead of reusing the notebook's `df_train` / `df_test`?
-------------------------------------------------------------------------
The original pipeline builds model inputs by *row position* on a frame sorted by
(state_name, date). That is fine for the original windowing, but it silently
conflates two different things once you start building lag features:

  * the panel has 77 missing calendar days per state in TRAIN and 55 in TEST
    (including a 75-day 2020-03-19 -> 2020-06-01 COVID gap), so a positional
    shift(1) is NOT always "yesterday";
  * a positional rolling window straddles those gaps.

This module rebuilds every state's series on a COMPLETE daily calendar, so a
lag of k is exactly k calendar days and is NaN when that day was not observed.
Rows whose required inputs (or whose target) are unobserved are dropped from the
evaluation grid rather than silently imputed.

The target definitions are reproduced bit-for-bit from the original notebook
(Cell 1.1b `CausalFeatureFitter` -> Cell 1.2 `prep`) so that the upgrade is
scored on exactly the same six targets over exactly the same test period.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

DATE = "date"
STATE = "state_name"

# The six targets of the original pipeline (Cell 3.1), unchanged.
TARGETS: List[str] = [
    "total_generation_mwh", "avg_market_price", "grid_stress_index",
    "fcfs_priority_score", "palmp", "log_carbon",
]

# The two directly-observed targets the audit permits headline claims on
# (LEAKAGE_AUDIT.md: palmp / GSI / fcfs / log_carbon are derived policy overlays).
OBSERVABLE_TARGETS: List[str] = ["total_generation_mwh", "avg_market_price"]

# Chronological split boundaries. TEST is the original test CSV period and is
# never altered. VALID is carved out of the tail of TRAIN.
VALID_START = pd.Timestamp("2023-01-01")

HORIZONS: List[int] = [1, 2, 3]   # matches the original PRED=3


# --------------------------------------------------------------------------- IO
def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate the REFUSED_Ready root (the directory holding data/ and refused_fixes/)."""
    here = Path(start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "data" / "RL_TRAIN_DATA.csv").exists() and (cand / "refused_fixes").exists():
            return cand
    # notebook is usually one level down in notebook/
    for cand in [here / "REFUSED_Ready", here.parent / "REFUSED_Ready"]:
        if (cand / "data" / "RL_TRAIN_DATA.csv").exists():
            return cand
    raise FileNotFoundError(
        f"Could not locate REFUSED_Ready root from {here}. Expected data/RL_TRAIN_DATA.csv."
    )


def repo_paths(start: Optional[Path] = None) -> Dict[str, Path]:
    root = _find_repo_root(start)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    out = {
        "root": root,
        "data": root / "data",
        "results": root / "results",
        "upgrade": root / "results" / "forecasting_upgrade",
    }
    for sub in ("tables", "figures", "outputs", "models"):
        (out["upgrade"] / sub).mkdir(parents=True, exist_ok=True)
    return out


def load_raw(paths: Optional[Dict[str, Path]] = None) -> pd.DataFrame:
    """Load the two original CSVs into one long frame tagged with `_split`.

    No rows are added, removed or reordered relative to the source files; the
    only change is the `_split` marker and a stable (state, date) sort.
    """
    paths = paths or repo_paths()
    tr = pd.read_csv(paths["data"] / "RL_TRAIN_DATA.csv", parse_dates=[DATE])
    te = pd.read_csv(paths["data"] / "RL_TEST_DATA.csv", parse_dates=[DATE])
    tr["_split"] = "train"
    te["_split"] = "test"
    df = pd.concat([tr, te], ignore_index=True)
    return df.sort_values([STATE, DATE], kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------- target reconstruction
def _prep_replica(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Faithful replication of the notebook's Cell 1.2 `prep()`.

    Reproduced verbatim (same constants, same clipping, same train-`ref`
    statistics) so the upgrade forecasts the identical `palmp` / `log_carbon` /
    `grid_stress_index` / `fcfs_priority_score` series the original pipeline
    forecast. Verified against the published `results_summary.csv`
    (mean_palmp = 12875.8, mean_carbon_price = 2239.0).
    """
    df = df.copy()
    nc = df.select_dtypes(include=[np.number]).columns
    df[nc] = df[nc].fillna(ref[nc].median())
    cbp = df["carbon_budget_pressure"].clip(0, 1)
    yr = (df["year"] - 2017).clip(0)

    df["p_carbon_gcal"] = (
        0.40 * (200 + 4800 * (cbp ** 1.5))
        + 0.30 * 5850 * 0.25 * (1 + cbp)
        + 0.20 * 6900 * (1.02 ** yr)
    )
    df["lcmp_carbon"] = (
        (df["carbon_intensity"] - ref["carbon_intensity"].quantile(0.05)).clip(0)
        / (ref["carbon_intensity"].quantile(0.95) - ref["carbon_intensity"].quantile(0.05) + 1e-6)
        * df["avg_market_price"] * 0.30
    )
    df["ef_actual"] = 0.95 * (1 - df["renewable_ratio"].clip(0, 1)) + 0.02 * df["renewable_ratio"].clip(0, 1)
    df["cfc_value_inr"] = (0.95 - df["ef_actual"]).clip(0) * df["total_generation_mwh"] * df["p_carbon_gcal"] / 1000.0
    df["discom_stress"] = (df["avg_market_price"].clip(0, 10000) / 10000 * 0.5
                           + df["grid_stress_index"].fillna(0) * 0.5).clip(0, 1)

    df["monsoon_re_risk"] = (df["season"] == "Monsoon").astype(float) * df["ren_intermittency"].fillna(0)
    df["freq_excursion_risk"] = (
        (df["coal_critical"].fillna(0).astype(bool)) | (df["grid_stress_index"].fillna(0) > 0.7)
    ).astype(float)
    df["ppa_deviation"] = (df["avg_market_price"] - 3500.0) / 2556.0

    lm = ref["avg_market_price"].quantile(0.99)

    def mm(x, r):
        return (x - r.min()) / (r.max() - r.min() + 1e-8)

    cb = (df["avg_market_price"] > df["price_cvar90"].fillna(1e9)).astype(float)
    df["palmp"] = (df["avg_market_price"]
                   + 0.30 * mm(df["carbon_intensity"], ref["carbon_intensity"]) * lm
                   + 0.25 * cb * mm(df["price_cvar90"].fillna(df["avg_market_price"]),
                                    ref["price_cvar90"].fillna(ref["avg_market_price"])) * lm
                   + 0.15 * mm(df["grid_stress_index"].fillna(0), ref["grid_stress_index"].fillna(0)) * lm
                   + 0.10 * mm(df["fcfs_priority_score"].fillna(0), ref["fcfs_priority_score"].fillna(0)) * lm)
    df["palmp_carbon_comp"] = 0.30 * mm(df["carbon_intensity"], ref["carbon_intensity"]) * lm
    df["palmp_cvar_comp"] = 0.25 * cb * mm(df["price_cvar90"].fillna(df["avg_market_price"]),
                                           ref["price_cvar90"].fillna(ref["avg_market_price"])) * lm
    df["palmp_stress_comp"] = 0.15 * mm(df["grid_stress_index"].fillna(0), ref["grid_stress_index"].fillna(0)) * lm
    df["log_carbon"] = np.log1p(df["carbon_proxy_tons"].clip(0))

    # --- Alpha N4 block (needed only to reproduce the original PatchTST inputs).
    # NOTE: these are POSITIONAL rolling windows in the original code and are
    # reproduced as such purely for faithful replication. The upgrade's own
    # feature builder uses calendar-correct equivalents instead.
    df["rolling_cvar4hr"] = (
        df.sort_values(DATE).groupby(STATE)["avg_market_price"]
          .transform(lambda x: x.rolling(6, min_periods=1).quantile(0.90))
    )
    df["ci_intraday_variance"] = (
        df.sort_values(DATE).groupby(STATE)["carbon_intensity"]
          .transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)
    )
    df["re_ramp_risk"] = df["ren_day_change_pct"].abs().fillna(0) * (df["season"] == "Monsoon").astype(float)
    df["monsoon_re_credit"] = (
        (df["season"] == "Monsoon").astype(float)
        * df["ren_day_change"].clip(0).fillna(0)
        * df["p_carbon_gcal"] / 1e6
    ).clip(0, 1)
    df["pa_lmp_t"] = (df["palmp"] - 0.05 * df["monsoon_re_credit"] * lm).clip(lower=0)
    return df.copy()          # de-fragment after the many column inserts


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the original Cell 1.1b -> Cell 1.2 chain, train-fit only.

    Returns the long panel with the six TARGETS materialised.
    """
    from refused_fixes.causal_features import CausalFeatureFitter

    tr_mask = df["_split"] == "train"
    fitter = CausalFeatureFitter().fit(df.loc[tr_mask])
    out = fitter.transform(df)
    train_ref = out.loc[tr_mask].copy()          # frozen TRAIN reference
    out = _prep_replica(out, ref=train_ref)
    return out.sort_values([STATE, DATE], kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------- calendar panel
@dataclass
class CalendarPanel:
    """A (date x state) wide view of the panel on a COMPLETE daily calendar.

    Every lag / rolling operation goes through here, which guarantees that:
      * a shift of k is exactly k calendar days for every state;
      * missing calendar days are NaN and propagate as "unknown", never as
        "the previous surviving observation";
      * no value can bleed across a state boundary.
    """
    long: pd.DataFrame
    dates: pd.DatetimeIndex
    states: List[str]
    _cache: Dict[str, pd.DataFrame] = None

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> "CalendarPanel":
        dates = pd.date_range(df[DATE].min(), df[DATE].max(), freq="D")
        states = sorted(df[STATE].unique())
        return cls(long=df, dates=dates, states=states, _cache={})

    def wide(self, col: str) -> pd.DataFrame:
        """(date x state) matrix for `col`, reindexed onto the full calendar."""
        if col in self._cache:
            return self._cache[col]
        w = (self.long.pivot_table(index=DATE, columns=STATE, values=col, aggfunc="first")
             .reindex(index=self.dates, columns=self.states))
        self._cache[col] = w
        return w

    def observed_mask(self) -> pd.DataFrame:
        """True where a (date, state) row actually exists in the source CSVs."""
        m = (self.long.assign(_one=1)
             .pivot_table(index=DATE, columns=STATE, values="_one", aggfunc="first")
             .reindex(index=self.dates, columns=self.states))
        return m.notna()

    def split_of(self, column: Optional[str] = None) -> pd.DataFrame:
        """(date x state) matrix of the split label, for building masks.

        Prefers the three-way `split` column (train / valid / test) and falls
        back to the raw two-way `_split` (train / test) only if it is absent.
        """
        col = column or ("split" if "split" in self.long.columns else "_split")
        return (self.long.pivot_table(index=DATE, columns=STATE, values=col,
                                      aggfunc="first")
                .reindex(index=self.dates, columns=self.states))


# ------------------------------------------------------------------- splitting
@dataclass
class Splits:
    """Chronological split boundaries used by every upgrade experiment."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp          # end of the FIT block (validation excluded)
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def label(self, d: pd.Series) -> pd.Series:
        out = pd.Series(index=d.index, dtype=object)
        out[(d >= self.train_start) & (d <= self.train_end)] = "train"
        out[(d >= self.valid_start) & (d <= self.valid_end)] = "valid"
        out[(d >= self.test_start) & (d <= self.test_end)] = "test"
        return out

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"split": "train", "start": self.train_start.date(), "end": self.train_end.date()},
            {"split": "valid", "start": self.valid_start.date(), "end": self.valid_end.date()},
            {"split": "test",  "start": self.test_start.date(),  "end": self.test_end.date()},
        ])


def make_splits(df: pd.DataFrame, valid_start: pd.Timestamp = VALID_START) -> Splits:
    """TRAIN(fit) | VALID | TEST, strictly chronological and non-overlapping.

    TEST is exactly the original RL_TEST_DATA.csv period. VALID is the tail of
    the original train period, so the TEST split is untouched by tuning.
    """
    tr = df.loc[df["_split"] == "train", DATE]
    te = df.loc[df["_split"] == "test", DATE]
    return Splits(
        train_start=tr.min(),
        train_end=valid_start - pd.Timedelta(days=1),
        valid_start=valid_start,
        valid_end=tr.max(),
        test_start=te.min(),
        test_end=te.max(),
    )


# --------------------------------------------------------------------- regimes
def add_regimes(df: pd.DataFrame, splits: Splits) -> pd.DataFrame:
    """Data-driven regime flags for Section 12, thresholded on TRAIN only.

    The shipped `regime` column is a fixed calendar label
    (Pre-Transition / Transition / Post-Shift) whose TEST slice is 100%
    "Post-Shift", and the shipped `coal_critical` flag is 0 for every TEST row.
    Neither can stratify test error, so the upgrade defines regimes from
    train-only quantiles of observable drivers.
    """
    out = df.copy()
    fit = out[(out[DATE] >= splits.train_start) & (out[DATE] <= splits.train_end)]

    # Thresholds are PER STATE: the 18 states differ by three orders of
    # magnitude in generation (Puducherry ~0.7 vs Uttar Pradesh ~400), so a
    # single national cut-off would label whole states as permanently extreme.
    def _q(col: str, q: float) -> pd.Series:
        thr = fit.groupby(STATE)[col].quantile(q)
        return out[STATE].map(thr)

    out["rg_monsoon"] = (out["season"] == "Monsoon").astype(int)
    out["rg_high_price"] = (out["avg_market_price"] >= _q("avg_market_price", 0.90)).astype(int)
    out["rg_high_renewable"] = (out["renewable_ratio"] >= _q("renewable_ratio", 0.90)).astype(int)
    out["rg_high_stress"] = (out["grid_stress_index"] >= _q("grid_stress_index", 0.90)).astype(int)
    out["rg_coal_critical"] = (out["avg_coal_stock_days"] <= _q("avg_coal_stock_days", 0.10)).astype(int)
    out["rg_normal"] = (
        (out[["rg_high_price", "rg_high_renewable", "rg_high_stress",
              "rg_coal_critical"]].sum(axis=1) == 0)
        & (out["rg_monsoon"] == 0)
    ).astype(int)

    out.attrs["regime_thresholds"] = {
        "basis": "per-state quantiles of the TRAIN(fit) block only",
        "high_price": "q90", "high_renewable": "q90",
        "high_stress": "q90", "coal_critical": "q10",
    }
    return out


REGIME_COLS = ["rg_normal", "rg_coal_critical", "rg_monsoon",
               "rg_high_renewable", "rg_high_price", "rg_high_stress"]

REGIME_LABELS = {
    "rg_normal": "Normal", "rg_coal_critical": "Coal-critical",
    "rg_monsoon": "Monsoon", "rg_high_renewable": "High-renewable",
    "rg_high_price": "High-price", "rg_high_stress": "High-stress",
}


# ------------------------------------------------------------------ entrypoint
def load_panel(paths: Optional[Dict[str, Path]] = None,
               valid_start: pd.Timestamp = VALID_START):
    """One call: raw -> causal fix -> targets -> regimes -> calendar panel.

    Returns (df_long, splits, cal).
    """
    paths = paths or repo_paths()
    raw = load_raw(paths)
    df = build_targets(raw)
    splits = make_splits(df, valid_start=valid_start)
    df = add_regimes(df, splits)
    df["split"] = splits.label(df[DATE])
    cal = CalendarPanel.from_long(df)
    return df, splits, cal


__all__ = [
    "DATE", "STATE", "TARGETS", "OBSERVABLE_TARGETS", "HORIZONS", "VALID_START",
    "REGIME_COLS", "REGIME_LABELS",
    "repo_paths", "load_raw", "build_targets", "CalendarPanel", "Splits",
    "make_splits", "add_regimes", "load_panel",
]
