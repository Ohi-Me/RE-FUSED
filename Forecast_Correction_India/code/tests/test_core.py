"""Unit tests for the RE-FUSED core. Run: py -3.10 -m pytest code/tests -q (from the project root)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from refused_gate import gates, guard, probabilistic, selection, stats  # noqa: E402


def test_guard_blocks_without_env(monkeypatch):
    monkeypatch.delenv("REFUSED_CONFIRMATORY", raising=False)
    with pytest.raises(guard.BlindingError):
        guard.require_confirmatory()


def test_guard_blocks_with_env_but_no_prereg(monkeypatch):
    monkeypatch.setenv("REFUSED_CONFIRMATORY", "1")
    if os.path.exists(guard.PREREG) and os.path.exists(guard.SHAFILE):
        pytest.skip("pre-registration already frozen")
    with pytest.raises(guard.BlindingError):
        guard.require_confirmatory()


def test_ls_gate_recovers_coefficient():
    rng = np.random.default_rng(0)
    r = rng.normal(size=20000)
    R = 0.7 * r + rng.normal(scale=0.5, size=r.size)
    assert abs(gates.ls_gate(R, r) - 0.7) < 0.02


def test_partition_gate_fallback():
    rng = np.random.default_rng(1)
    r = rng.normal(size=1000)
    keys = np.array(["a"] * 990 + ["b"] * 10)
    R = np.where(keys == "a", 1.0, 0.2) * r
    g = gates.PartitionGate(min_cell=30).fit(R, r, keys)
    pred = g.predict(np.array(["a", "b"]))
    assert abs(pred[0] - 1.0) < 1e-9 and abs(pred[1] - g.global_) < 1e-9


def _synthetic_partition(n, tau, persist, noise, K=6, seed=0):
    """Cells with gate 1 + h(c); h in the second half of time equals persist*h1 + sqrt(1-persist^2)*h2."""
    rng = np.random.default_rng(seed)
    cells = 20
    h1 = rng.normal(scale=tau, size=cells)
    h2 = rng.normal(scale=tau, size=cells)
    c = rng.integers(0, cells, n)
    t = np.sort(rng.random(n))
    r = rng.normal(size=n)
    return rng, c, t, r, h1, h2


def test_part_detects_persistent_heterogeneity_and_rejects_noise():
    n = 60000
    rng = np.random.default_rng(3)
    cells = 12
    c = rng.integers(0, cells, n)
    r = rng.normal(size=n)
    blocks = (np.arange(n) * 6) // n
    # strong persistent heterogeneity
    h = np.linspace(-0.5, 0.5, cells)
    R = (1.0 + h[c]) * r + rng.normal(scale=1.0, size=n)
    out = selection.part_partition(R, r, np.zeros(n, int), c, blocks)
    assert out["G"] > 0 and out["C"] > 0
    # no heterogeneity -> predicted gain should not be positive beyond noise
    R0 = 1.0 * r + rng.normal(scale=1.0, size=n)
    out0 = selection.part_partition(R0, r, np.zeros(n, int), c, blocks)
    assert out0["G"] < out["G"] and out0["G"] < 1e-3
    # heterogeneity that flips sign in the second half of time (non-persistent)
    sign = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    Rf = (1.0 + sign * h[c]) * r + rng.normal(scale=1.0, size=n)
    outf = selection.part_partition(Rf, r, np.zeros(n, int), c, blocks)
    assert outf["C"] < 0 and outf["G"] < 0


def test_realised_gain_identity():
    rng = np.random.default_rng(4)
    n = 5000
    r = rng.normal(size=n)
    R = 0.8 * r + rng.normal(size=n)
    gc = np.full(n, 0.5)
    gf = rng.uniform(0, 1.2, n)
    lhs = np.mean((R - gc * r) ** 2) - np.mean((R - gf * r) ** 2)
    rhs = np.mean(selection.realised_gain(R, r, gc, gf))
    assert abs(lhs - rhs) < 1e-10


def test_block_bootstrap_ci_covers_mean_ar1():
    rng = np.random.default_rng(5)
    hits = 0
    for rep in range(60):
        e = rng.normal(size=1500)
        x = np.empty_like(e)
        x[0] = e[0]
        for i in range(1, len(e)):
            x[i] = 0.6 * x[i - 1] + e[i]
        ci = stats.block_ci(x, block=30, n_boot=500, seed=rep)
        hits += ci["lo"] <= 0 <= ci["hi"]
    assert hits >= 50          # nominal 95% -> expect ~57 of 60


def test_meta_and_multiplicity():
    m = stats.dersimonian_laird([0.1, 0.12, 0.08], [0.02, 0.03, 0.025])
    assert m["lo"] < 0.1 < m["hi"] and m["k"] == 3
    adj = stats.holm([0.01, 0.04, 0.03])
    assert np.allclose(sorted(adj), [0.03, 0.06, 0.06])
    by = stats.benjamini_yekutieli([0.001, 0.02, 0.5])
    assert np.all(by >= np.array([0.001, 0.02, 0.5]))


def test_conformal_quantile_finite_sample():
    e = np.arange(1, 100, dtype=float)          # n = 99
    assert probabilistic.conformal_quantile(e, 0.99) == 99.0
    assert np.isinf(probabilistic.conformal_quantile(e[:50], 0.99))
    Q = np.array([[0.0, 1.0]])
    assert np.allclose(probabilistic.pinball(np.array([2.0]), Q, [0.1, 0.9]), [0.2, 0.9])


def test_fixed_share_moves_to_better_expert():
    lb = np.ones(500)
    lc = np.zeros(500)
    w = gates.fixed_share_weights(lb, lc, eta=2.0, alpha=0.01)
    assert w[0] == 0.5 and w[-1] > 0.9
