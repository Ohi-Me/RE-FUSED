"""Exploratory check (not pre-registered; added after the scoring): sensitivity of the intervals of the
State-panel energy valuation (x05_gate_energy.py) to the bootstrap.

Same stored predictions, designs and daily differences as x05 (nothing is refitted). Each cell's 95 % interval is
recomputed with moving blocks of 7, 14 and 28 days and with a stationary bootstrap (mean block 7 days). Days are
resampled with all 34 control areas together, so dependence between areas on the same day is kept.
Protocol: Forecast_Correction_India/codes/design/11_ecm_revision_exploratory.md (R7).
Output: results/extra/gate_energy_blocks.csv (never overwrites a file).
Usage:  REFUSED_PHASE=confirm python codes/scripts/extra/x12_gate_energy_blocks.py   (hpc/extra_gate_energy_blocks.pbs)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import x05_gate_energy as X  # noqa: E402

DST = os.path.join(X.OUT, "gate_energy_blocks.csv")


def stationary(d, mean_block=7, n_boot=2000, seed=0):
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        new = rng.random(n) < 1.0 / mean_block
        new[0] = True
        starts = rng.integers(0, n, size=n)
        idx = np.empty(n, int)
        cur = 0
        for t in range(n):
            cur = starts[t] if new[t] else (cur + 1) % n
            idx[t] = cur
        means[b] = d[idx].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    if os.path.exists(DST):
        print("exists, kept:", DST)
        return
    rows = []
    for tid in X.O.TARGETS:
        J = X.designs(tid)
        if J is None:
            continue
        energy = tid != "T5"
        for H in X.O.HORIZONS:
            j = J[J.H == H]
            preds = [c for c in j.columns if c.startswith("pred_")]
            err = {p[5:]: (j.y - j[p]).abs() for p in preds}
            if energy:
                daily = {k: v.groupby(j.target_date).sum().sort_index() for k, v in err.items()}
            else:
                daily = {k: v.groupby(j.target_date).mean().sort_index() for k, v in err.items()}
            scale = X.DAYS if energy else 1.0
            for k, s in daily.items():
                if k == "fixed":
                    continue
                diff = (s - daily["fixed"]).to_numpy()
                row = dict(tid=tid, target=X.O.TARGETS[tid]["name"], H=H, design=k,
                           diff_vs_fixed=float(diff.mean() * scale), n_days=len(diff))
                for b in (7, 14, 28):
                    ci = X.S.block_ci(diff, b, n_boot=2000, seed=0)
                    row[f"lo_b{b}"], row[f"hi_b{b}"] = ci["lo"] * scale, ci["hi"] * scale
                    row[f"sig_b{b}"] = bool(ci["lo"] > 0 or ci["hi"] < 0)
                lo, hi = stationary(diff)
                row["lo_stat7"], row["hi_stat7"] = lo * scale, hi * scale
                row["sig_stat7"] = bool(lo > 0 or hi < 0)
                rows.append(row)
        print(tid, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(DST, index=False)
    fin = R[R.design.isin(["regime", "instance", "otg"]) & (R.tid != "T5")]
    for b in ("b7", "b14", "b28", "stat7"):
        print(f"{b}: significantly worse {int((fin[f'lo_{b}'] > 0).sum())}, "
              f"better {int((fin[f'hi_{b}'] < 0).sum())} of {len(fin)} cells", flush=True)


if __name__ == "__main__":
    main()
