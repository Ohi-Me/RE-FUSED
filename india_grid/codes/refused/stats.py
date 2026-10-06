# Ported from RE-FUSED (04_code/refused_gate/stats.py) without change of behaviour.
"""Inference at the evidence-unit level (audit A1-A3)."""
import numpy as np
from scipy import stats as st


def moving_block_bootstrap(d, block, n_boot=2000, seed=0):
    """Bootstrap distribution of the mean of a time-ordered series d using moving blocks of length `block`."""
    d = np.asarray(d, float)
    n = len(d)
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    csum = np.concatenate([[0.0], np.cumsum(d)])
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    sums = (csum[starts + block] - csum[starts]).sum(axis=1)
    return sums / (n_blocks * block)


def block_ci(d, block, n_boot=2000, seed=0, alpha=0.05):
    """Mean, percentile CI and one-sided bootstrap p-values (H1: mean > 0, H1: mean < 0)."""
    d = np.asarray(d, float)
    b = moving_block_bootstrap(d, block, n_boot, seed)
    m = float(d.mean())
    c = b - b.mean()
    return dict(mean=m, lo=float(np.quantile(b, alpha / 2)), hi=float(np.quantile(b, 1 - alpha / 2)),
                se=float(b.std(ddof=1)),
                p_greater=float((np.sum(c >= m) + 1) / (n_boot + 1)),
                p_less=float((np.sum(c <= m) + 1) / (n_boot + 1)), n=len(d), block=int(block))


def dm_hac(d, h=1):
    """Diebold-Mariano statistic with Newey-West (Bartlett) long-run variance, lag max(1, h-1)."""
    d = np.asarray(d, float)
    n = len(d)
    dc = d - d.mean()
    L = max(1, h - 1)
    lrv = dc @ dc / n
    for k in range(1, L + 1):
        lrv += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / n
    t = d.mean() / np.sqrt(max(lrv, 1e-300) / n)
    return float(t), float(2 * (1 - st.norm.cdf(abs(t))))


def dersimonian_laird(effects, ses):
    """Random-effects meta-analysis: pooled effect, SE, 95% CI, tau^2, I^2, Q, two-sided p."""
    y = np.asarray(effects, float)
    v = np.asarray(ses, float) ** 2
    w = 1 / v
    fixed = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - fixed) ** 2))
    k = len(y)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c) if (k > 1 and c > 0) else 0.0
    ws = 1 / (v + tau2)
    mu = np.sum(ws * y) / np.sum(ws)
    se = np.sqrt(1 / np.sum(ws))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(effect=float(mu), se=float(se), lo=float(mu - 1.96 * se), hi=float(mu + 1.96 * se),
                tau2=float(tau2), I2=float(I2), Q=Q, k=k, p=float(2 * (1 - st.norm.cdf(abs(mu / se)))))


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def benjamini_yekutieli(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    cm = np.sum(1.0 / np.arange(1, m + 1))
    adj = np.empty(m)
    prev = 1.0
    for rank_rev, i in enumerate(np.argsort(p)[::-1]):
        rank = m - rank_rev
        prev = min(prev, p[i] * m * cm / rank)
        adj[i] = min(1.0, prev)
    return adj


def fmt_p(p):
    """Scientific notation for p-values (audit S4: never print 0.0)."""
    if p is None or not np.isfinite(p):
        return "n/a"
    return "%.2g" % p if p >= 1e-3 else "%.1e" % p
