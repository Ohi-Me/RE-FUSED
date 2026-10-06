"""
metrics.py — evaluation harness for RE-FUSED-Alpha.

Replaces MAPE (unstable near zero) and the arbitrary `weighted_score` with:
  * scale-free, zero-safe POINT metrics : MASE, RMSSE, sMAPE
  * PROBABILISTIC metrics               : pinball loss, CRPS (quantile approx),
                                          interval coverage / width
  * SIGNIFICANCE                        : Diebold-Mariano test (HLN corrected)

All functions are pure numpy and operate on 1-D arrays unless noted.
For daily state-day data use seasonal period m=7 (weekly).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats  # only for the DM p-value; drop if scipy unavailable


# --------------------------------------------------------------- point metrics
def _asarray(x):
    return np.asarray(x, dtype=float).ravel()


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0,200]%. Zero-safe (denominator uses |y|+|yhat|)."""
    y, f = _asarray(y_true), _asarray(y_pred)
    denom = np.abs(y) + np.abs(f)
    mask = denom > 0
    return float(200.0 * np.mean(np.abs(f[mask] - y[mask]) / denom[mask]))


def mase(y_true, y_pred, y_train, m: int = 7) -> float:
    """Mean Absolute Scaled Error. Scale of the in-sample seasonal-naive error.
    <1 beats seasonal-naive, >1 is worse. Undefined-safe."""
    y, f = _asarray(y_true), _asarray(y_pred)
    yt = _asarray(y_train)
    if len(yt) <= m:
        m = 1
    scale = np.mean(np.abs(yt[m:] - yt[:-m]))
    if scale == 0:
        scale = np.mean(np.abs(np.diff(yt))) or 1.0
    return float(np.mean(np.abs(f - y)) / scale)


def rmsse(y_true, y_pred, y_train, m: int = 7) -> float:
    """Root Mean Squared Scaled Error (the M5 metric)."""
    y, f = _asarray(y_true), _asarray(y_pred)
    yt = _asarray(y_train)
    if len(yt) <= m:
        m = 1
    scale = np.mean((yt[m:] - yt[:-m]) ** 2)
    if scale == 0:
        scale = np.mean(np.diff(yt) ** 2) or 1.0
    return float(np.sqrt(np.mean((f - y) ** 2) / scale))


# ------------------------------------------------------- probabilistic metrics
def pinball_loss(y_true, q_pred, tau: float) -> float:
    """Quantile (pinball) loss at level tau in (0,1)."""
    y, q = _asarray(y_true), _asarray(q_pred)
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def crps_from_quantiles(y_true, q_preds, taus: Sequence[float]) -> float:
    """CRPS approximated from a set of predictive quantiles.

    For quantile levels {tau} the CRPS equals the integral of the pinball loss;
    the discrete approximation is 2 * mean_tau pinball(tau).  q_preds shape
    (n_obs, n_quantiles) aligned to `taus`.
    """
    q_preds = np.asarray(q_preds, dtype=float)
    taus = list(taus)
    losses = [pinball_loss(y_true, q_preds[:, i], t) for i, t in enumerate(taus)]
    return float(2.0 * np.mean(losses))


def interval_coverage(y_true, lo, hi) -> Dict[str, float]:
    """Empirical coverage and mean width of a predictive interval [lo, hi]."""
    y, l, h = _asarray(y_true), _asarray(lo), _asarray(hi)
    return {
        "coverage": float(np.mean((y >= l) & (y <= h))),
        "mean_width": float(np.mean(h - l)),
    }


# --------------------------------------------------------- Diebold-Mariano test
def diebold_mariano(y_true, pred1, pred2, h: int = 1, loss: str = "mse"):
    """Diebold-Mariano test of equal predictive accuracy (pred1 vs pred2),
    with the Harvey-Leybourne-Newbold small-sample correction.

    Returns (dm_stat, p_value). H0: equal accuracy.
    dm_stat < 0  => pred1 has the smaller loss (pred1 is better).
    """
    y = _asarray(y_true)
    e1 = y - _asarray(pred1)
    e2 = y - _asarray(pred2)
    if loss == "mse":
        d = e1 ** 2 - e2 ** 2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    n = len(d)
    dbar = d.mean()
    # long-run variance via Newey-West with (h-1) lags
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * (1.0 - k / h) * cov
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / n)
    # HLN correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    p = 2.0 * (1.0 - stats.t.cdf(np.abs(dm), df=n - 1))
    return float(dm), float(p)


# ------------------------------------------------------------------- convenience
def point_report(y_true, y_pred, y_train, m: int = 7) -> Dict[str, float]:
    """One call for the headline point metrics on an observable target."""
    return {
        "MASE": round(mase(y_true, y_pred, y_train, m), 4),
        "RMSSE": round(rmsse(y_true, y_pred, y_train, m), 4),
        "sMAPE": round(smape(y_true, y_pred), 3),
    }


def prob_report(y_true, q_preds, taus: Sequence[float]) -> Dict[str, float]:
    """Headline probabilistic metrics; expects a 0.5 quantile and a symmetric
    outer pair (e.g. 0.1/0.9) somewhere in `taus`."""
    taus = list(taus)
    out = {"CRPS": round(crps_from_quantiles(y_true, q_preds, taus), 4)}
    if 0.1 in taus and 0.9 in taus:
        lo = np.asarray(q_preds)[:, taus.index(0.1)]
        hi = np.asarray(q_preds)[:, taus.index(0.9)]
        cov = interval_coverage(y_true, lo, hi)
        out["PI80_coverage"] = round(cov["coverage"], 4)   # target ~0.80
        out["PI80_width"] = round(cov["mean_width"], 4)
    return out


__all__ = [
    "smape", "mase", "rmsse",
    "pinball_loss", "crps_from_quantiles", "interval_coverage",
    "diebold_mariano", "point_report", "prob_report",
]
