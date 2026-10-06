"""Window-level normalisation shared by the BiLSTM (O2) and LA-MCAG (O3) models.

Target: centred on the mean of the last 7 available target values inside the input window (the same base as the
LightGBM samples), scaled by the series' train standard deviation; outputs are mapped back with that base, so the models
follow level shifts (e.g. growing RE, calmer prices in FY2024-25) instead of extrapolating train means.
Context: each numeric column (already z-scored with train statistics) is split into a within-window centred channel and
a window-mean level channel; mask columns are passed unchanged.
"""
import numpy as np


def target_window(yn_raw, avail, sl, sd, fallback):
    """yn_raw: raw target values; returns (centred window, base)."""
    yw, aw = yn_raw[sl], avail[sl]
    last = np.where(aw > 0)[0]
    base = float(np.mean(yw[last[-7:]])) if len(last) else fallback
    return np.where(aw > 0, (yw - base) / sd, 0.0).astype(np.float32), base


def context_window(arr, mask_idx):
    """arr: (14, c) z-scored columns incl. masks; returns (14, 2*numeric + masks)."""
    num = [j for j in range(arr.shape[1]) if j not in mask_idx]
    x = arr[:, num]
    lvl = x.mean(axis=0, keepdims=True)
    out = [x - lvl, np.repeat(lvl, arr.shape[0], axis=0)]
    if mask_idx:
        out.append(arr[:, mask_idx])
    return np.concatenate(out, axis=1).astype(np.float32)
