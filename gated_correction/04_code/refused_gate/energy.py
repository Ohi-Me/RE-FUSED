"""Energy-system valuation of forecast errors (design §5).

Conventions (NYISO two-settlement system, hourly):
  A  actual load (MW, hourly integrated = MWh per hour)
  F  day-ahead load forecast used to purchase energy in the day-ahead market
  P_DA, P_RT  zonal day-ahead and real-time LBMP ($/MWh); P_RT is the hourly mean of 5-minute prices
A load-serving entity that buys F day-ahead settles the deviation (A - F) at the real-time price, so relative to
buying the realised load day-ahead its imbalance cost is (A - F) * (P_RT - P_DA). This can be negative.
All functions return per-row values so they can be block-bootstrapped over time.
"""
import numpy as np


def deviation_energy(A, F):
    """Absolute deviation energy |A - F| (MWh per hourly row)."""
    return np.abs(np.asarray(A, float) - np.asarray(F, float))


def imbalance_cost(A, F, p_rt, p_da):
    """Two-settlement imbalance cost (A - F)(P_RT - P_DA), $ per row."""
    return (np.asarray(A, float) - np.asarray(F, float)) * (np.asarray(p_rt, float) - np.asarray(p_da, float))


def deviation_exposure(A, F, p_rt, p_da):
    """Upper-bound exposure |A - F| |P_RT - P_DA|, $ per row (sign-free sensitivity measure)."""
    return np.abs(np.asarray(A, float) - np.asarray(F, float)) * np.abs(np.asarray(p_rt, float) - np.asarray(p_da, float))


def reserve_outcomes(A, F, reserve_up, reserve_price):
    """Ex-ante upward reserve procured to cover under-forecast (A - F > 0).

    Returns per-row arrays: procured MW (clipped at 0), cost $ at the reserve price, shortfall MWh
    max(0, A - F - reserve), and a covered indicator.
    """
    err = np.asarray(A, float) - np.asarray(F, float)
    res = np.maximum(np.asarray(reserve_up, float), 0.0)
    return dict(reserve_mw=res, reserve_cost=res * np.asarray(reserve_price, float),
                shortfall_mwh=np.maximum(0.0, err - res), covered=(err <= res).astype(float))
