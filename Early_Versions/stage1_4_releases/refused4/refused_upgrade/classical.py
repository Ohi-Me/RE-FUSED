"""
classical.py — statistical baselines on the calendar-aware panel.

Every baseline here is a genuine ROLLING-ORIGIN forecaster: at each forecast
origin T it emits h = 1, 2, 3 day-ahead predictions using only observations
dated <= T, and it is scored on exactly the same (date, state, target, horizon)
grid as every learned model in Part II.

Parameter estimation is strictly train-only:

  * ETS   — Holt-Winters coefficients are fitted once on the training history by
            statsmodels, then the standard additive recursion is run forward
            through validation/test *updating on observed values only*. Running
            the recursion by hand (rather than refitting at every origin) is what
            makes an exact 423-origin x 18-state x 6-target evaluation tractable,
            and it is exactly equivalent to a fixed-parameter rolling forecast.
  * ARIMA — SARIMAX is fitted on the training history, the fitted parameters are
            re-applied to the full series with `refit=False`, and the h-step
            forecast at every origin is read straight off the Kalman filter as
            yhat(t+h|t) = Z @ T^h @ a(t|t). That is the exact multi-step
            predictor, not an approximation, and it costs one filter pass.

Calendar gaps are handled natively: the ETS recursion carries its state forward
across an unobserved day, and the Kalman filter treats NaN as a missing
observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .panel import DATE, STATE, CalendarPanel

SEASON_M = 7


# ------------------------------------------------------------------ plumbing
def to_pred_frame(yhat_wide: pd.DataFrame, y_wide: pd.DataFrame, target: str,
                  horizon: int, grid: pd.DataFrame) -> pd.DataFrame:
    """Wide forecasts -> the canonical long prediction frame, restricted to `grid`."""
    try:
        yh = yhat_wide.stack(future_stack=True)
    except TypeError:
        yh = yhat_wide.stack(dropna=False)
    yh.index.names = [DATE, STATE]
    out = yh.rename("yhat").reset_index()
    out = grid.merge(out, on=[DATE, STATE], how="left")
    out["target"] = target
    out["horizon"] = horizon
    return out[[DATE, STATE, "target", "horizon", "y", "yhat"]]


def eval_grid(cal: CalendarPanel, target: str, horizon: int,
              split: str = "test") -> pd.DataFrame:
    """The evaluation points every model in Part II must predict on.

    A row qualifies when the realised target is observed AND the persistence
    forecast y(D-h) exists — i.e. the benchmark itself is computable there.
    Scoring a learned model on rows where persistence is undefined would flatter
    it, so the grid is defined by the benchmark.
    """
    y = cal.wide(target)
    lag = y.shift(horizon)
    keep = y.notna() & lag.notna()
    sp = cal.split_of()
    keep = keep & (sp == split)
    idx = keep.stack()
    idx = idx[idx].index
    out = pd.DataFrame(index=idx).reset_index()
    out.columns = [DATE, STATE]
    ys = y.stack(future_stack=True).rename("y").reset_index()
    ys.columns = [DATE, STATE, "y"]
    return out.merge(ys, on=[DATE, STATE], how="left")


# ---------------------------------------------------------- trivial baselines
def persistence(cal: CalendarPanel, target: str, h: int) -> pd.DataFrame:
    """yhat(D) = y(D-h). At h=1 this is the classic naive/random-walk forecast."""
    return cal.wide(target).shift(h)


def seasonal_naive(cal: CalendarPanel, target: str, h: int, m: int = SEASON_M) -> pd.DataFrame:
    """yhat(D) = y(D - m*ceil(h/m)); for h <= 7 that is simply last week's value."""
    k = int(np.ceil(h / m))
    return cal.wide(target).shift(m * k)


def rolling_mean(cal: CalendarPanel, target: str, h: int, window: int = 7) -> pd.DataFrame:
    """Mean of the `window` observations ending at the forecast origin."""
    return cal.wide(target).shift(h).rolling(window, min_periods=max(2, window // 2)).mean()


def drift(cal: CalendarPanel, target: str, h: int, window: int = 28) -> pd.DataFrame:
    """Random walk with drift estimated over the trailing `window` days."""
    base = cal.wide(target).shift(h)
    slope = (base - base.shift(window)) / float(window)
    return base + slope * h


# ------------------------------------------------------------------- ETS
@dataclass
class ETSConfig:
    trend: bool = True
    damped: bool = False
    seasonal: bool = True
    m: int = SEASON_M

    @property
    def name(self) -> str:
        t = ("damped" if self.damped else "add") if self.trend else "none"
        return f"ETS(trend={t},seasonal={'add' if self.seasonal else 'none'})"


def fit_ets_params(y_train: np.ndarray, cfg: ETSConfig) -> Optional[Dict[str, float]]:
    """Fit Holt-Winters on the training history; returns coefficients + init state."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = np.asarray(y_train, float)
    y = y[np.isfinite(y)]
    need = cfg.m * 3 + 5
    if len(y) < need:
        return None
    try:
        mod = ExponentialSmoothing(
            y,
            trend="add" if cfg.trend else None,
            damped_trend=cfg.damped if cfg.trend else False,
            seasonal="add" if cfg.seasonal else None,
            seasonal_periods=cfg.m if cfg.seasonal else None,
            initialization_method="estimated",
        )
        res = mod.fit(optimized=True)
    except Exception:
        return None
    p = res.params
    return {
        "alpha": float(p.get("smoothing_level", 0.3) or 0.3),
        "beta": float(p.get("smoothing_trend", 0.0) or 0.0) if cfg.trend else 0.0,
        "gamma": float(p.get("smoothing_seasonal", 0.0) or 0.0) if cfg.seasonal else 0.0,
        "phi": float(p.get("damping_trend", 1.0) or 1.0) if cfg.damped else 1.0,
        "l0": float(res.level[-1]) if hasattr(res, "level") else float(np.nanmean(y)),
        "b0": float(res.trend[-1]) if (cfg.trend and hasattr(res, "trend")) else 0.0,
        "s0": (list(np.asarray(res.season[-cfg.m:], float))
               if (cfg.seasonal and hasattr(res, "season")) else [0.0] * cfg.m),
        "n_train": int(len(y)),
    }


def ets_rolling(y_full: np.ndarray, params: Dict[str, float], cfg: ETSConfig,
                horizons: Sequence[int]) -> Dict[int, np.ndarray]:
    """Additive Holt-Winters run forward over the whole series.

    At every index t the state (l_t, b_t, s_t) summarises observations up to and
    including t; the emitted forecast for t+h therefore uses no future data.
    Unobserved days advance the state without an update.
    """
    y = np.asarray(y_full, float)
    n = len(y)
    m = cfg.m
    a, b_, g, phi = params["alpha"], params["beta"], params["gamma"], params["phi"]
    level = params["l0"]
    trend = params["b0"] if cfg.trend else 0.0
    seas = list(params["s0"]) if cfg.seasonal else [0.0] * m

    out = {h: np.full(n, np.nan) for h in horizons}
    hist_s: List[float] = list(seas)

    for t in range(n):
        # --- emit forecasts from the state as of t-1 is NOT what we want; we
        # emit AFTER updating with y_t, so index t holds the origin-t forecast.
        yt = y[t]
        l_prev, b_prev = level, trend
        s_prev = hist_s[-m] if len(hist_s) >= m else 0.0
        if np.isfinite(yt):
            level = a * (yt - s_prev) + (1 - a) * (l_prev + phi * b_prev)
            if cfg.trend:
                trend = b_ * (level - l_prev) + (1 - b_) * phi * b_prev
            if cfg.seasonal:
                s_new = g * (yt - l_prev - phi * b_prev) + (1 - g) * s_prev
            else:
                s_new = 0.0
        else:                                   # unobserved day: carry forward
            level = l_prev + phi * b_prev
            if cfg.trend:
                trend = phi * b_prev
            s_new = s_prev
        hist_s.append(s_new)

        for h in horizons:
            if t + h >= n:
                continue
            damp = sum(phi ** i for i in range(1, h + 1)) if cfg.damped else h
            s_idx = len(hist_s) - m + ((h - 1) % m)
            s_h = hist_s[s_idx] if 0 <= s_idx < len(hist_s) else 0.0
            out[h][t + h] = level + (damp * trend if cfg.trend else 0.0) + (s_h if cfg.seasonal else 0.0)
    return out


def ets_forecast_wide(cal: CalendarPanel, target: str, horizons: Sequence[int],
                      cfg: ETSConfig, train_end: pd.Timestamp
                      ) -> Tuple[Dict[int, pd.DataFrame], Dict[str, dict]]:
    """Per-state ETS: fit on history <= train_end, roll forward over everything."""
    y_wide = cal.wide(target)
    fits: Dict[str, dict] = {}
    out = {h: pd.DataFrame(np.nan, index=y_wide.index, columns=y_wide.columns)
           for h in horizons}
    train_mask = y_wide.index <= train_end
    for st in y_wide.columns:
        series = y_wide[st].to_numpy(float)
        p = fit_ets_params(series[train_mask], cfg)
        if p is None:
            continue
        fits[st] = p
        preds = ets_rolling(series, p, cfg, horizons)
        for h in horizons:
            out[h][st] = preds[h]
    return out, fits


# ------------------------------------------------------------------- ARIMA
def arima_forecast_wide(cal: CalendarPanel, target: str, horizons: Sequence[int],
                        order: Tuple[int, int, int],
                        seasonal_order: Tuple[int, int, int, int],
                        train_end: pd.Timestamp,
                        ) -> Tuple[Dict[int, pd.DataFrame], Dict[str, dict]]:
    """Per-state SARIMAX with exact h-step forecasts at every origin.

    The fitted (train-only) parameters are re-applied to the full series with
    `refit=False`; the filter then yields a(t|t) for every t, and

        yhat(t+h | t) = Z @ T^h @ a(t|t)

    is the exact h-step predictor of the state-space form. One filter pass
    replaces ~420 sequential refits per series.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y_wide = cal.wide(target)
    out = {h: pd.DataFrame(np.nan, index=y_wide.index, columns=y_wide.columns)
           for h in horizons}
    info: Dict[str, dict] = {}
    train_mask = y_wide.index <= train_end

    for st in y_wide.columns:
        full = y_wide[st].to_numpy(float)
        tr = full[train_mask]
        if np.isfinite(tr).sum() < 200:
            continue
        try:
            # Stationarity/invertibility are ENFORCED. Without them the MLE can
            # land on an explosive AR polynomial; the one-step fit still looks
            # fine but T^h amplifies the unstable root and the h-step forecasts
            # diverge by many orders of magnitude.
            mod = SARIMAX(tr, order=order, seasonal_order=seasonal_order,
                          simple_differencing=False,
                          enforce_stationarity=True, enforce_invertibility=True)
            res = mod.fit(disp=False, maxiter=300)
            full_mod = SARIMAX(full, order=order, seasonal_order=seasonal_order,
                               simple_differencing=False,
                               enforce_stationarity=True, enforce_invertibility=True)
            fres = full_mod.filter(res.params)
            fr = fres.filter_results
            Z = np.asarray(fr.design[:, :, 0], float)        # (1, k)
            T = np.asarray(fr.transition[:, :, 0], float)    # (k, k)
            a_filt = np.asarray(fr.filtered_state, float)    # (k, n)

            # Guard: reject a state whose transition matrix is explosive beyond
            # the unit roots that differencing legitimately introduces.
            rho = float(np.max(np.abs(np.linalg.eigvals(T))))
            if not np.isfinite(rho) or rho > 1.05:
                info[st] = {"error": f"unstable transition (rho={rho:.3f})"}
                continue

            lo, hi = np.nanmin(tr), np.nanmax(tr)
            span = hi - lo if hi > lo else max(abs(hi), 1.0)
            preds_ok = True
            staged = {}
            for h in horizons:
                M = Z @ np.linalg.matrix_power(T, h)          # (1, k)
                yhat = (M @ a_filt).ravel()                   # forecast for t+h from t
                shifted = np.full(len(full), np.nan)
                shifted[h:] = yhat[:len(full) - h]
                # a sane forecast cannot sit 10 training-ranges outside the data
                finite = shifted[np.isfinite(shifted)]
                if finite.size and (finite.min() < lo - 10 * span or
                                    finite.max() > hi + 10 * span):
                    preds_ok = False
                    break
                staged[h] = shifted
            if not preds_ok:
                info[st] = {"error": "forecast out of plausible range"}
                continue
            info[st] = {"aic": float(res.aic), "n_params": len(res.params),
                        "spectral_radius": round(rho, 4)}
            for h, v in staged.items():
                out[h][st] = v
        except Exception as exc:                              # pragma: no cover
            info[st] = {"error": str(exc)[:120]}
            continue
    return out, info


__all__ = [
    "SEASON_M", "to_pred_frame", "eval_grid", "persistence", "seasonal_naive",
    "rolling_mean", "drift", "ETSConfig", "fit_ets_params", "ets_rolling",
    "ets_forecast_wide", "arima_forecast_wide",
]
