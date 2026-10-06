"""baselines.py — reference forecasters for RE-FUSED-Alpha (audit fix-order #3).

A sound evaluation requires any deep model to be anchored against trivial
baselines under scale-free metrics. Provides per-state, strictly causal
one-step-style baselines on the daily state-day panel:

  * naive       : y_hat(t) = y(t-1)
  * snaive      : y_hat(t) = y(t-m)              (m=7, weekly seasonality)
  * moving_avg  : y_hat(t) = mean(y(t-w) .. y(t-1))

`evaluate_baselines` scores them on the TEST split with MASE / RMSSE / sMAPE,
computed per state and then averaged unweighted across states (M4-style),
using each state's TRAIN history for the MASE/RMSSE scale. A deep model that
cannot beat `snaive` (MASE < 1) on the observable targets has no forecasting
claim — report this table next to the model table.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import mase, rmsse, smape

DATE = "date"
STATE = "state_name"

BASELINES = ("naive", "snaive", "moving_avg")


def baseline_predictions(df_train: pd.DataFrame,
                         df_test: pd.DataFrame,
                         target: str,
                         m: int = 7,
                         ma_window: int = 7) -> pd.DataFrame:
    """Test-split predictions for each baseline.

    History is the concatenated per-state train+test series so the first test
    days draw on late-train values (no boundary artifacts). Strictly causal:
    every prediction uses only values dated strictly before the predicted day.
    """
    cols = [DATE, STATE, target]
    tr = df_train[cols].copy()
    tr["_split"] = "train"
    te = df_test[cols].copy()
    te["_split"] = "test"
    full = pd.concat([tr, te], ignore_index=True)
    full = full.sort_values([STATE, DATE], kind="mergesort").reset_index(drop=True)

    g = full.groupby(STATE, sort=False)[target]
    full["naive"] = g.shift(1)
    full["snaive"] = g.shift(m)
    full["moving_avg"] = g.transform(
        lambda x: x.shift(1).rolling(ma_window, min_periods=1).mean())

    return (full[full["_split"] == "test"]
            .drop(columns="_split")
            .reset_index(drop=True))


def evaluate_baselines(df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       targets: Sequence[str],
                       m: int = 7,
                       ma_window: int = 7) -> pd.DataFrame:
    """MASE/RMSSE/sMAPE per (target, baseline), averaged across states."""
    rows = []
    for tgt in targets:
        preds = baseline_predictions(df_train, df_test, tgt, m=m, ma_window=ma_window)
        for bl in BASELINES:
            per_state = []
            for st, grp in preds.groupby(STATE, sort=False):
                mask = grp[bl].notna() & grp[tgt].notna()
                if int(mask.sum()) < m + 2:
                    continue
                y = grp.loc[mask, tgt].to_numpy(dtype=float)
                p = grp.loc[mask, bl].to_numpy(dtype=float)
                hist = (df_train.loc[df_train[STATE] == st, tgt]
                        .dropna().to_numpy(dtype=float))
                if len(hist) <= m + 1:
                    continue
                per_state.append({
                    "MASE": mase(y, p, hist, m=m),
                    "RMSSE": rmsse(y, p, hist, m=m),
                    "sMAPE": smape(y, p),
                })
            if not per_state:
                continue
            agg = pd.DataFrame(per_state)
            rows.append({
                "target": tgt,
                "baseline": bl,
                "n_states": len(agg),
                "MASE": round(float(agg["MASE"].mean()), 4),
                "RMSSE": round(float(agg["RMSSE"].mean()), 4),
                "sMAPE": round(float(agg["sMAPE"].mean()), 3),
                "MASE_worst_state": round(float(agg["MASE"].max()), 4),
            })
    return pd.DataFrame(rows)


__all__ = ["BASELINES", "baseline_predictions", "evaluate_baselines"]
