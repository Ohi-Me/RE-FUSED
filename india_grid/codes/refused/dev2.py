"""Shared bits for development round 2 (protocol codes/design/06_dev2_protocol.md).

Round 1 outputs stay in results/dev. Round 2 writes to results/dev2 (or results/dev2_smoke for the smoke job, set with
REFUSED_DEV2_DIR). All helpers here only read and reshape forecasts; nothing picks settings on the development test.
"""
import json
import os

import numpy as np
import pandas as pd

from . import o2data as O
from .paths import RES

# round-1 outputs of the current phase (results/dev or results/confirm) and round-2 outputs of the same phase
# (results/dev2 or results/confirm2; smoke and rehearsal runs set REFUSED_DEV2_DIR to their own folder)
DEV1 = os.path.join(RES, O.PHASE_DIR)
DEV2 = os.path.join(RES, os.environ.get("REFUSED_DEV2_DIR", O.PHASE_DIR + "2"))
TAG = os.path.basename(DEV2)
SMOKE = TAG not in ("dev2", "confirm2")
# In the confirmatory phase every round-2 choice is read from the development record (results/dev2) instead of being
# picked again. REFUSED_USE_FROZEN=1 does the same in the dev phase, to rehearse that code before the freeze.
FROZEN_FROM = os.path.join(RES, "dev2")
USE_FROZEN = O.PHASE == "confirm" or os.environ.get("REFUSED_USE_FROZEN") == "1"


def frozen(*parts):
    """A development choice file from results/dev2, e.g. frozen("o2", "logs", "xgb_T1.json")."""
    return json.load(open(os.path.join(FROZEN_FROM, *parts)))
KEYS = ["tid", "sid", "issue_date", "H", "target_date", "split", "season", "y"]
QN = [f"q{int(round(q * 100)):02d}" for q in O.QUANTILES]
LEV = np.array(O.QUANTILES)
MIN_ROWS_VAL = 200  # a member needs at least this many validation rows to take part in a combination


def folder(*parts):
    path = os.path.join(DEV2, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def lag_of(tid):
    return O.lags().get(O.TARGETS[tid]["col"], 1)


def fill_tails(d):
    """Chronos-Bolt gives q10..q90 only. Stretch the ends in a straight line so every member has q05..q95."""
    d = d.copy()
    miss_lo = d["q05"].isna() & d["q10"].notna() & d["q25"].notna()
    miss_hi = d["q95"].isna() & d["q90"].notna() & d["q75"].notna()
    d.loc[miss_lo, "q05"] = d.loc[miss_lo, "q10"] - (d.loc[miss_lo, "q25"] - d.loc[miss_lo, "q10"]) * (0.05 / 0.15)
    d.loc[miss_hi, "q95"] = d.loc[miss_hi, "q90"] + (d.loc[miss_hi, "q90"] - d.loc[miss_hi, "q75"]) * (0.05 / 0.15)
    return d


def find_member(tid, name):
    """Where a member's averaged forecast file lives: round 2 first, then round 1 (O2, then O3)."""
    for base in (os.path.join(DEV2, "o2", "preds"), os.path.join(DEV2, "o3", "preds"),
                 os.path.join(DEV1, "o2", "preds"), os.path.join(DEV1, "o3", "preds")):
        f = os.path.join(base, f"{tid}__{name}.parquet")
        if os.path.exists(f):
            return f
    return None


def load_member(tid, name, seed=None):
    """Averaged forecast of a member, or one seed of it (members without seeds, like Chronos, are the same for every
    seed)."""
    f = find_member(tid, name)
    if f is None:
        return None
    fs = f[:-len(".parquet")] + "_seeds.parquet"
    if seed is not None and os.path.exists(fs):
        d = pd.read_parquet(fs)
        d = d[d.seed == seed]
        if d.empty:
            return None
    else:
        d = pd.read_parquet(f)
    d = d[d.split.isin(["validation", "dev_test"])].copy()
    for c in QN:
        if c not in d:
            d[c] = np.nan
    d = fill_tails(d)
    full = d[QN].notna().all(axis=1).to_numpy()
    if full.any():  # anchors carry only q50: sorting their row would move that value into q05
        d.loc[full, QN] = np.sort(d.loc[full, QN].to_numpy(), axis=1)
    return d[KEYS + QN]


def scales(tid):
    sc = pd.read_parquet(os.path.join(DEV1, "o2", "scales.parquet"))
    return sc[sc.tid == tid][["sid", "H", "scale_mae", "scale_mse"]]


def regions(tid):
    s = pd.read_parquet(os.path.join(DEV1, "o2", "samples", f"{tid}.parquet"), columns=["sid", "region"])
    return s.drop_duplicates("sid").set_index("sid").region


def member_stack(tid, names, seed=None):
    """Line up members on the rows they all cover. Returns the row frame and an array (rows, members, 7)."""
    frames = {}
    for n in names:
        d = load_member(tid, n, seed)
        if d is None or d[QN].isna().any(axis=1).mean() > 0.01:
            continue
        frames[n] = d.dropna(subset=QN)
    if not frames:
        return None, None, []
    key = ["sid", "issue_date", "H"]
    rows = None
    for d in frames.values():
        k = d[key]
        rows = k if rows is None else rows.merge(k, on=key)
    first = next(iter(frames.values()))
    base = first[KEYS].merge(rows, on=key).merge(scales(tid), on=["sid", "H"])
    base = base[np.isfinite(base.scale_mae) & (base.scale_mae > 0)]
    base = base.sort_values(["target_date", "sid", "H"]).reset_index(drop=True)
    arr = np.stack([base[key].merge(frames[n][key + QN], on=key, how="left")[QN].to_numpy() for n in frames], axis=1)
    return base, arr, list(frames)


def pinball_rows(y, Q, scale):
    """Scaled pinball loss per row, averaged over the 7 levels. Q is (rows, 7)."""
    u = (y[:, None] - Q) / scale[:, None]
    return np.maximum(LEV[None, :] * u, (LEV[None, :] - 1) * u).mean(axis=1)


def mase_rows(y, q50, scale):
    return np.abs(y - q50) / scale


def mase(frame, q50):
    """MASE as in round 1: mean scaled |error| per series and horizon, then the mean of those."""
    e = np.abs(frame.y.to_numpy() - q50) / frame.scale_mae.to_numpy()
    return float(pd.Series(e).groupby([frame.sid.to_numpy(), frame.H.to_numpy()]).mean().mean())


def iso_week_fold(dates):
    return pd.to_datetime(pd.Series(dates)).dt.isocalendar().week.to_numpy().astype(int) % 2
