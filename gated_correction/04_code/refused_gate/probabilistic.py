"""Probabilistic track: adaptivity ladder for residual quantiles (09_theory T6)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

CRPS_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)          # 19 equally spaced levels
TAIL_LEVELS = np.array([0.01, 0.025, 0.975, 0.99])
ALL_LEVELS = np.unique(np.concatenate([CRPS_LEVELS, TAIL_LEVELS]))


def conformal_quantile(e, q):
    """Split-conformal quantile of calibration residuals e at level q (finite-sample adjusted).
    Upper levels use rank ceil((n+1)q); lower levels use floor((n+1)q); +/-inf if the rank is outside 1..n."""
    e = np.sort(np.asarray(e, float))
    n = len(e)
    if n == 0:
        return np.nan
    if q >= 0.5:
        k = int(np.ceil((n + 1) * q))
        return e[k - 1] if k <= n else np.inf
    k = int(np.floor((n + 1) * q))
    return e[k - 1] if k >= 1 else -np.inf


class PartitionQuantiles:
    """Per-cell conformal quantiles; cells with fewer than min_cell rows use the parent (global) quantiles."""

    def __init__(self, levels=ALL_LEVELS, min_cell=30):
        self.levels, self.min_cell = np.asarray(levels), min_cell

    def fit(self, e, keys):
        e, keys = np.asarray(e, float), np.asarray(keys)
        self.global_ = np.array([conformal_quantile(e, q) for q in self.levels])
        order = np.argsort(keys, kind="stable")
        ks, es = keys[order], e[order]
        uniq, starts = np.unique(ks, return_index=True)
        ends = np.append(starts[1:], len(ks))
        self.table = {}
        for u, a, b in zip(uniq, starts, ends):
            if b - a >= self.min_cell:
                self.table[u] = np.array([conformal_quantile(es[a:b], q) for q in self.levels])
        return self

    def predict(self, keys):
        return np.vstack([self.table.get(k, self.global_) for k in np.asarray(keys)])


class ConditionalQuantiles:
    """Per-instance conditional quantiles by gradient boosting with pinball loss (one model per level)."""

    def __init__(self, levels=ALL_LEVELS, seed=0, max_iter=200, learning_rate=0.08, max_depth=5, min_samples_leaf=40):
        self.levels, self.seed = np.asarray(levels), seed
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
                       min_samples_leaf=min_samples_leaf)

    def fit(self, F, e):
        self.models = [HGB(loss="quantile", quantile=float(q), random_state=self.seed, **self.kw).fit(F, e)
                       for q in self.levels]
        return self

    def predict(self, F):
        Q = np.column_stack([m.predict(F) for m in self.models])
        return np.sort(Q, axis=1)                                     # remove quantile crossing


def pinball(y, Q, levels):
    """Mean pinball loss per level; y shape (n,), Q shape (n, L)."""
    y = np.asarray(y, float)[:, None]
    Q = np.asarray(Q, float)
    Qf = np.where(np.isfinite(Q), Q, np.where(Q > 0, y.max() * 10 + 1e6, y.min() * 10 - 1e6))
    u = y - Qf
    lv = np.asarray(levels)[None, :]
    return np.mean(np.maximum(lv * u, (lv - 1) * u), axis=0)


def pinball_rows(y, Q, levels):
    y = np.asarray(y, float)[:, None]
    u = y - np.asarray(Q, float)
    lv = np.asarray(levels)[None, :]
    return np.maximum(lv * u, (lv - 1) * u)


def crps_from_quantiles(y, Q, levels):
    """CRPS approximation: 2 * mean over equally spaced levels of the pinball loss (per row)."""
    return 2.0 * pinball_rows(y, Q, levels).mean(axis=1)


def coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def conditional_coverage_deviation(y, upper, keys, nominal):
    """Mean absolute deviation of one-sided coverage P(y <= upper) from nominal across cells."""
    y, upper, keys = np.asarray(y, float), np.asarray(upper, float), np.asarray(keys)
    devs = []
    for k in np.unique(keys):
        m = keys == k
        devs.append(abs(np.mean(y[m] <= upper[m]) - nominal))
    return float(np.mean(devs))
