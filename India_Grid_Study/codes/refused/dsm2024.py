"""Deviation charges of a buyer under the CERC DSM Regulations, 2024, Regulation 8(7) (numbered 8(20) before the
errata of 18 Nov 2024), per 15-minute block, with the frequency linkage and the volume limits.

The amendments of 2024, 2025 and 2026 do not change these provisions (they concern infirm power, wind and solar
sellers, storage and payment terms). Rates are multiples of the Normal Rate (NR) of the block.

Conventions: drawal and deviation in MWh per block, buyer sign (positive deviation = over-drawal, payable);
nr in Rs/MWh; f in Hz. Returns the net amount payable by the buyer in Rs (negative = receivable).
"""
import numpy as np

BLOCK_H = 0.25


def limits(sched_mwh, category="standard"):
    """Upper edges of volume-limit tiers VLB(1) and VLB(2) in MWh per block."""
    s = np.abs(sched_mwh)
    if category == "small":            # buyer with a schedule up to 400 MW: VLB(1) up to min(20 %, 40 MW), VLB(2) beyond
        l1 = np.minimum(0.20 * s, 40 * BLOCK_H)
        return l1, np.full_like(l1, np.inf)
    if category == "re_rich":
        return np.full_like(s, 200 * BLOCK_H), np.full_like(s, 300 * BLOCK_H)
    if category == "super_re_rich":
        return np.full_like(s, 250 * BLOCK_H), np.full_like(s, 350 * BLOCK_H)
    return np.minimum(0.10 * s, 100 * BLOCK_H), np.maximum(np.minimum(0.15 * s, 200 * BLOCK_H),
                                                           np.minimum(0.10 * s, 100 * BLOCK_H))


def _steps(f):
    """Frequency in steps of 0.01 Hz, rounded as in the published accounts."""
    return np.round((np.asarray(f, float) - 50.0) / 0.01)


def over_mult(f, tier):
    """Payable multiple of NR for over-drawal in tier 1, 2 or 3."""
    k = _steps(f)
    if tier == 1:
        return np.select([k < -10, k < 0, k == 0, k <= 5, k < 10], [1.5, 1 - 0.05 * k, 1.0, 1 - 0.05 * k, 0.5], 0.0)
    if tier == 2:
        return np.select([k < 0, k <= 5, k < 10], [1.5, 1.0, 0.75], 0.0)
    return np.select([k < 0, k < 10], [2.0, 1.0], 0.5)


def under_mult(f, tier):
    """Receivable multiple of NR for under-drawal; negative values are payments (f >= 50.10 Hz)."""
    k = _steps(f)
    if tier == 1:
        return np.select([k < -10, k < 0, k == 0, k <= 5, k < 10], [1.0, 0.9 - 0.01 * k, 0.9, 0.9 - 0.08 * k, 0.0], -0.1)
    if tier == 2:
        return np.select([k <= 0, k <= 5, k < 10], [0.8, 0.5, 0.0], -0.1)
    return np.where(k < 10, 0.0, -0.1)


def charge(dev_mwh, sched_mwh, nr, f, category="standard"):
    """Net payable (Rs) of a buyer for one block or arrays of blocks."""
    dev = np.asarray(dev_mwh, float)
    l1, l2 = limits(sched_mwh, category)
    a = np.abs(dev)
    t1 = np.minimum(a, l1)
    t2 = np.clip(a - l1, 0, None) if category == "small" else np.clip(np.minimum(a, l2) - l1, 0, None)
    t3 = np.zeros_like(a) if category == "small" else np.clip(a - l2, 0, None)
    pay = nr * (t1 * over_mult(f, 1) + t2 * over_mult(f, 2) + t3 * over_mult(f, 3))
    rec = nr * (t1 * under_mult(f, 1) + t2 * under_mult(f, 2) + t3 * under_mult(f, 3))
    return np.where(dev > 0, pay, -rec)
