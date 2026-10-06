"""Leakage test for the O2 information set.

For several (target, series, issue day τ) triples, every panel value that is not yet published at issue time
(date > τ − L for a column with availability lag L) is replaced by noise. Features for issue day τ must not change,
and anchors must not change. Target values (y) are allowed to change (they are what is forecast).
Run: py -3.10 04_code/tests/test_leakage.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from refused import o2data as O  # noqa: E402


def perturb(P, tau, L, rng):
    Q = P.copy()
    for c in Q.columns:
        if c in ("entity", "date", "region", "bid_area", "pilot18", "split") or not np.issubdtype(Q[c].dtype, np.number):
            continue
        lag = L.get(c, 0)
        m = Q.date > tau - pd.Timedelta(days=lag)
        Q.loc[m, c] = Q.loc[m, c] * rng.uniform(0.5, 1.5, m.sum()) + rng.normal(0, 1, m.sum())
    return Q


def main():
    rng = np.random.default_rng(0)
    P = O.load_panel()
    L = O.lags()
    checks = 0
    # issue days must come after the training period, because train-period averages scale the features; in the
    # confirmatory phase the training period ends a year later, so the same days are moved forward by one year
    shift = pd.DateOffset(years=1) if O.PHASE == "confirm" else pd.DateOffset(years=0)
    for tid, tau in [("T1", "2023-07-15"), ("T2", "2024-11-02"), ("T3", "2024-01-20"), ("T4", "2023-12-05"),
                     ("T5", "2024-06-10")]:
        tau = pd.Timestamp(tau) + shift
        assert tau > O.TRAIN_END, f"{tid}: issue day {tau.date()} is inside the training period"
        S0 = O.tabular_samples(P, tid)
        S1 = O.tabular_samples(perturb(P, tau, L, rng), tid)
        cols = O.feature_columns(S0) + ["a_seasonal_naive", "a_persistence", "a_ma7", "base"]
        a = S0[S0.issue_date == tau].sort_values(["sid", "H"]).reset_index(drop=True)
        b = S1[S1.issue_date == tau].sort_values(["sid", "H"]).reset_index(drop=True)
        assert len(a) == len(b) and len(a) > 0, tid
        # the train-period normalisation constants (s, means) are computed from train rows only; tau is after train,
        # so they are unaffected by the perturbation
        diff = (a[cols] - b[cols]).abs().max()
        bad = diff[diff > 1e-9]
        assert bad.empty, f"{tid}: features changed after perturbing unpublished data: {bad.to_dict()}"
        yd = (a.y - b.y).abs().max()
        print(f"{tid} τ={tau.date()}: {len(a)} rows, {len(cols)} feature/anchor columns unchanged; "
              f"targets changed as expected (max |Δy| = {yd:.2f})")
        checks += 1
    print(f"LEAKAGE TEST PASSED ({checks} targets)")


if __name__ == "__main__":
    main()
