"""Analyse X5/X8 (probabilistic ladder and energy valuation) from the per-row files.

Inputs: results/<phase>/x5_x8/prob_energy_s*.parquet  (phase = dev or confirm)
Outputs: results/<phase>/analysis/x5_x8_*.csv
All uncertainty: moving-block bootstrap over days (7-day blocks), all zones summed per day.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import energy, stats as st  # noqa: E402
from refused_gate.paths import RES  # noqa: E402

FORECASTS = ["iso", "corr_series", "corr_instance", "corr_roll30"]
QLEVELS = ["q_global", "q_zone", "q_zone_hourblock", "q_zone_hour", "q_zone_month", "q_instance", "q_roll90"]


def daily_ci(df, col_a, col_b=None, block=7):
    """Mean daily total of col_a (or of col_a - col_b) with block-bootstrap CI."""
    x = df.groupby("day")[col_a].sum() if col_b is None else df.groupby("day").apply(lambda g: (g[col_a] - g[col_b]).sum())
    ci = st.block_ci(x.sort_index().to_numpy(), block=block, n_boot=2000, seed=0)
    return ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="dev")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RES, a.phase, "x5_x8", "prob_energy_s*.parquet")))
    out_dir = os.path.join(RES, a.phase, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    frames = [pd.read_parquet(f) for f in files]
    base = frames[0][["sid", "ts", "y_t", "da_lbmp", "rt_lbmp", "da_spin"]].copy()
    base["day"] = base.ts.dt.normalize()
    num = [c for c in frames[0].columns if c.startswith(("F_", "crps_", "up99_", "up975_", "pin99_"))]
    avg = sum(f[num].to_numpy(dtype=float) for f in frames) / len(frames)          # seed-averaged per row
    D = pd.concat([base.reset_index(drop=True), pd.DataFrame(avg, columns=num)], axis=1)
    n_days = D.day.nunique()

    # ---- point forecasts: energy metrics ----
    rows = []
    for fc in FORECASTS:
        e = D.y_t - D[f"F_{fc}"]
        D[f"dev_{fc}"] = np.abs(e)
        D[f"imb_{fc}"] = energy.imbalance_cost(D.y_t, D[f"F_{fc}"], D.rt_lbmp, D.da_lbmp)
        D[f"sq_{fc}"] = e ** 2
        rows.append(dict(forecast=fc, mae_mw=float(np.abs(e).mean()), rmse_mw=float(np.sqrt((e ** 2).mean())),
                         deviation_gwh_per_year=float(D[f"dev_{fc}"].sum() / 1000 * 365 / n_days),
                         imbalance_musd_per_year=float(np.nansum(D[f"imb_{fc}"]) / 1e6 * 365 / n_days)))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(out_dir, "x5_x8_point_energy.csv"), index=False)
    print(P.round(3).to_string())
    comp = []
    for a_, b_ in [("iso", "corr_series"), ("corr_series", "corr_instance"), ("corr_series", "corr_roll30")]:
        for metric, scale, unit in [("dev", 1 / 1000 * 365, "GWh/yr"), ("imb", 1 / 1e6 * 365, "M$/yr")]:
            ci = daily_ci(D, f"{metric}_{a_}", f"{metric}_{b_}")
            comp.append(dict(comparison=f"{a_} - {b_}", metric=metric, unit=unit, mean=ci["mean"] * scale,
                             lo=ci["lo"] * scale, hi=ci["hi"] * scale, p_b_better=ci["p_greater"]))
    C = pd.DataFrame(comp)
    C.to_csv(os.path.join(out_dir, "x5_x8_point_comparisons.csv"), index=False)
    print(C.round(3).to_string())

    # ---- probabilistic ladder and reserves ----
    prow = []
    for fc in FORECASTS:
        err = (D.y_t - D[f"F_{fc}"]).to_numpy()
        for q in QLEVELS:
            up99 = D[f"up99_{fc}_{q}"].to_numpy()
            res = energy.reserve_outcomes(D.y_t, D[f"F_{fc}"], up99, D.da_spin)
            prow.append(dict(forecast=fc, qlevel=q, crps=float(D[f"crps_{fc}_{q}"].mean()),
                             pinball99=float(D[f"pin99_{fc}_{q}"].mean()), coverage99=float(np.mean(err <= up99)),
                             reserve_mw_mean=float(res["reserve_mw"].mean() * D.sid.nunique()),
                             reserve_cost_musd_per_year=float(res["reserve_cost"].sum() / 1e6 * 365 / n_days),
                             shortfall_mwh_per_year=float(res["shortfall_mwh"].sum() * 365 / n_days)))
    Q = pd.DataFrame(prow)
    Q.to_csv(os.path.join(out_dir, "x5_x8_probabilistic_reserves.csv"), index=False)
    print(Q.round(4).to_string())
    for fc in FORECASTS:
        best = Q[Q.forecast == fc].sort_values("crps").qlevel.iloc[0]
        best99 = Q[Q.forecast == fc].sort_values("pinball99").qlevel.iloc[0]
        print(f"{fc:14s} best CRPS level: {best:18s} best pinball@0.99 level: {best99}")
    qc = []
    for fc in FORECASTS:
        for a_, b_ in [("q_zone_hour", "q_instance"), ("q_zone_hour", "q_global"), ("q_zone_hour", "q_roll90"),
                       ("q_zone", "q_zone_hour")]:
            for kind in ("crps", "pin99"):
                ci = daily_ci(D, f"{kind}_{fc}_{a_}", f"{kind}_{fc}_{b_}")
                qc.append(dict(forecast=fc, comparison=f"{a_} - {b_}", metric=kind, mean_daily_sum=ci["mean"],
                               lo=ci["lo"], hi=ci["hi"], p_b_better=ci["p_greater"]))
    QC = pd.DataFrame(qc)
    QC.to_csv(os.path.join(out_dir, "x5_x8_quantile_comparisons.csv"), index=False)
    print(QC.round(3).to_string())


if __name__ == "__main__":
    main()
