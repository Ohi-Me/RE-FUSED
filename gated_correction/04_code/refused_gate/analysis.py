"""Analysis helpers: seed-averaged per-row losses, refinement contrasts with day-block bootstrap, skill tables."""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import stats as st
from .paths import RES


def load_runs(experiment, pattern):
    """Load all per-row outputs whose key matches pattern (glob on the key without _s<seed>)."""
    files = sorted(glob.glob(os.path.join(RES, "dev", experiment, pattern + "_s*.parquet")))
    runs = []
    for f in files:
        d = pd.read_parquet(f)
        diag = json.load(open(f.replace(".parquet", ".json")))
        runs.append((os.path.basename(f)[:-8], d, diag))
    return runs


def row_losses(runs):
    """Seed-averaged squared error per row and rung. Rows are aligned on (sid, ts)."""
    mats = []
    for _, d, _ in runs:
        R = (d.y_t - d.B).to_numpy()
        cols = [c for c in d.columns if c.startswith("g_")]
        e = pd.DataFrame({c[2:]: (R - d[c].to_numpy() * d.r.to_numpy()) ** 2 for c in cols})
        e["sid"], e["ts"], e["base"] = d.sid.to_numpy(), d.ts.to_numpy(), R ** 2
        mats.append(e.set_index(["sid", "ts"]))
    L = sum(mats) / len(mats)
    return L.reset_index()


def skill_table(L):
    rungs = [c for c in L.columns if c not in ("sid", "ts", "base")]
    base = L.groupby("sid").base.mean()
    out = []
    for k in rungs:
        s = 1 - L.groupby("sid")[k].mean() / base
        out.append(dict(rung=k, skill_macro=float(s.mean()), skill_pooled=float(1 - L[k].mean() / L.base.mean()),
                        degraded=int((s < 0).sum())))
    return pd.DataFrame(out).sort_values("skill_macro", ascending=False)


def contrast(L, coarse, fine, block_days=7, n_boot=2000, seed=0):
    """Gain of fine over coarse (positive = fine better), in skill units relative to the baseline MSE.
    Daily aggregation over all series, moving-block bootstrap over days."""
    d = L.assign(day=pd.to_datetime(L.ts).dt.normalize())
    daily = d.groupby("day")[[coarse, fine, "base"]].sum().sort_index()
    diff = (daily[coarse] - daily[fine]).to_numpy() / daily.base.mean()
    ci = st.block_ci(diff, block=block_days, n_boot=n_boot, seed=seed)
    return dict(coarse=coarse, fine=fine, gain_skill=ci["mean"], lo=ci["lo"], hi=ci["hi"],
                p_fine_better=ci["p_greater"], p_coarse_better=ci["p_less"], days=ci["n"])
