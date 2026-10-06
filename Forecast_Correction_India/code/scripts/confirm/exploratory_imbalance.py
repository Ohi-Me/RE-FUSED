"""Exploratory (NOT pre-registered) decomposition of the two-settlement imbalance-cost difference on NYISO 2026.

Imbalance cost of a load-serving entity that buys forecast F day-ahead: (A - F)(P_RT - P_DA). With the corrected forecast
F_c = F_iso + c, the cost difference per row is  -c (P_RT - P_DA)  (positive = the correction costs money).
Inputs : results/confirm/x5_x8/prob_energy_s*.parquet (seed-averaged forecasts, as in make_tables.py)
Outputs: results/confirm/exploratory/imbalance_decomposition.parquet  (per-row spread, correction, cost difference)
         results/confirm/exploratory/imbalance_decomposition.json     (all numbers quoted in the manuscript)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate.paths import RES  # noqa: E402

OUT = os.path.join(RES, "confirm", "exploratory")


def main():
    files = sorted(glob.glob(os.path.join(RES, "confirm", "x5_x8", "prob_energy_s*.parquet")))
    fr = [pd.read_parquet(f) for f in files]
    D = fr[0][["sid", "ts", "y_t", "da_lbmp", "rt_lbmp"]].copy()
    for c in ("F_iso", "F_corr_series"):
        D[c] = sum(f[c].to_numpy(dtype=float) for f in fr) / len(fr)
    days = D.ts.dt.normalize().nunique()
    ann = 365 / days / 1e6  # $ -> M$/yr
    D["spread"] = D.rt_lbmp - D.da_lbmp
    D["corr"] = D.F_corr_series - D.F_iso
    D["dcost"] = -D["corr"] * D["spread"]
    ok = D.spread.notna()
    total = D.loc[ok, "dcost"].sum() * ann
    jan = D.loc[ok & (D.ts.dt.month == 1), "dcost"].sum() * ann
    ev = ok & (D.ts.dt.month == 1) & D.ts.dt.day.isin([27, 28])
    event = D.loc[ev, "dcost"].sum() * ann
    thr = D.loc[ok, "spread"].abs().quantile(0.9)
    tail = D.loc[ok & (D.spread.abs() > thr), "dcost"].sum() * ann
    monthly = (D[ok].groupby(D.ts.dt.month)["dcost"].sum() * ann).round(2)
    by_zone_event = (D[ev].groupby("sid")["spread"].agg(["min", "max"])).round(0)
    res = {
        "rows": int(len(D)), "rows_with_prices": int(ok.sum()), "days": int(days), "seeds": len(files),
        "iso_mean_error_mw": float((D.y_t - D.F_iso).mean()),
        "mean_correction_mw": float(D["corr"].mean()),
        "share_rows_negative_correction": float((D["corr"] < 0).mean()),
        "mean_spread_rt_minus_da": float(D.loc[ok, "spread"].mean()),
        "sd_spread": float(D.loc[ok, "spread"].std()),
        "imbalance_iso_musd_yr": float(np.nansum((D.y_t - D.F_iso) * D.spread) * ann),
        "imbalance_corrected_musd_yr": float(np.nansum((D.y_t - D.F_corr_series) * D.spread) * ann),
        "cost_increase_musd_yr": float(total),
        "january_increase_musd_yr": float(jan), "january_share": float(jan / total),
        "event_27_28_jan_increase_musd_yr": float(event), "event_share": float(event / total),
        "abs_spread_p90": float(thr), "tail_increase_musd_yr": float(tail), "tail_share": float(tail / total),
        "event_min_spread_by_zone": by_zone_event["min"].to_dict(),
        "monthly_increase_musd_yr": {int(k): float(v) for k, v in monthly.items()},
    }
    os.makedirs(OUT, exist_ok=True)
    D[["sid", "ts", "spread", "corr", "dcost"]].to_parquet(os.path.join(OUT, "imbalance_decomposition.parquet"), index=False)
    json.dump(res, open(os.path.join(OUT, "imbalance_decomposition.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
