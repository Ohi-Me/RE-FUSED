"""Additional analysis, not pre-registered (delay study, Nov 2026): robustness of the delay-horizon law (E4a).

Reads the E1 predictions of x13 (results/extra/delay/E1/preds; published and instant publication, horizons 1-14) and
computes, per target, model, State, horizon and season, the pinball loss over the 7 quantiles, the MAE of the median
and the share of outcomes inside the 10-90% band. No model is fitted. Writes results/extra/delay/E4/robust.csv only.
Usage: python codes/scripts/extra/x16_delay_robustness.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused.paths import RES  # noqa: E402

QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QN = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
D = os.path.join(RES, "extra", "delay")

if __name__ == "__main__":
    os.makedirs(os.path.join(D, "E4"), exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "E1", "preds", "T*__*.parquet"))):
        p = pd.read_parquet(f)
        p = p[(p.split == "dev_test") & p.y.notna()].copy()
        u = p.y.to_numpy()[:, None] - p[QN].to_numpy()
        t = np.array(QS)[None, :]
        p["pin"] = np.maximum(t * u, (t - 1) * u).mean(axis=1)
        p["ae"] = (p.y - p.q50).abs()
        p["cov80"] = ((p.y >= p.q10) & (p.y <= p.q90)).astype(float)
        rows.append(p.groupby(["tid", "model", "sid", "H", "season"])
                    .agg(pin=("pin", "mean"), mae=("ae", "mean"), cov80=("cov80", "mean"), n=("ae", "size"))
                    .reset_index())
        print(os.path.basename(f), len(p), flush=True)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(D, "E4", "robust.csv"), index=False)
    print("written", os.path.join(D, "E4", "robust.csv"), flush=True)
