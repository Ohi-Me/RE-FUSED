"""Forecast-error quantiles, reserve and deviation energy for the all-India hourly series (Indian counterpart of
x5_x8_prob_energy.py, whose functions are reused unchanged).

Point forecasts for each series (demand, net demand, wind, solar), day ahead:
  lgbm            the LightGBM day-ahead forecast, uncorrected (reference; India has no published operator forecast)
  corr_series     LightGBM + static per-series gate x corrector
  corr_instance   LightGBM + per-instance gate x corrector
  corr_roll30     LightGBM + per-series gate re-estimated daily on the last 30 days
Calibration residuals on the validation period are out of sample (cross-fitted static gates, online rolling gate).
Quantile ladder of e = A - F: q_global, q_zone (= per series), q_zone_hourblock, q_zone_hour, q_zone_month,
q_instance (gradient-boosted quantiles), q_roll90; the names follow the earlier study so its analysis code applies.
Energy: absolute deviation energy (MWh); upward reserve at q = 0.99 and 0.975 in MW with realised coverage and
shortfall energy; a stylised two-settlement imbalance cost (energy bought day-ahead at the forecast, the deviation
settled in real time) at the national day-ahead and real-time market clearing prices published by Grid-India in its
DSM files (India_Grid_Study build step 10). Estimation only.
Modes: dev (never reads data from 1 March 2026 on) | confirm (requires the India pre-registration lock).
Output: results/india/<mode>/prob_energy/prob_energy_s<seed>.parquet|json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ["REFUSED_GATE_STUDY"] = "india"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev"))
import pandas as pd  # noqa: E402
from refused_gate import data, energy, gates as G, ladder, probabilistic as P  # noqa: E402
from refused_gate.paths import PROC, RES  # noqa: E402
import x5_x8_prob_energy as X  # noqa: E402

LEVELS = P.ALL_LEVELS


def frames(mode, seed):
    ds = data.india_hourly(mode)
    L = pd.read_parquet(os.path.join(PROC, f"india_hourly_learned_baselines_{mode}_s0.parquet"))
    df = ds["df"].merge(L[["sid", "ts", "B_lgbm"]], on=["sid", "ts"], how="left")
    from refused_gate.paths import INDIA_PROC
    pr = os.path.join(INDIA_PROC, "refused_allindia_hourly_prices.parquet")
    if os.path.exists(pr):
        df = df.merge(pd.read_parquet(pr)[["ts", "dam_mcp_rs_mwh", "rtm_mcp_rs_mwh"]], on="ts", how="left")
    else:
        df["dam_mcp_rs_mwh"] = df["rtm_mcp_rs_mwh"] = np.nan
    x = df.assign(y_t=df.y, B=df.B_lgbm).dropna(subset=ds["feats"] + ["y_t", "B"]).copy()
    x["R"] = x.y_t - x.B
    x["day"] = x.ts.dt.normalize()
    tr, va, te = (x[x.split == s].copy() for s in ("train", "val", "test"))
    corr = ladder._fit_corrector(tr[ds["feats"]].to_numpy(np.float32), tr.R.to_numpy(), ladder.LadderConfig(), seed)
    va["r"] = corr.predict(va[ds["feats"]].to_numpy(np.float32))
    te["r"] = corr.predict(te[ds["feats"]].to_numpy(np.float32))
    g1 = G.ls_gate(va.R.to_numpy(), va.r.to_numpy())
    va["r"] *= g1
    te["r"] *= g1
    return ds, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mode", default="dev", choices=["dev", "confirm"])
    a = ap.parse_args()
    out_dir = os.path.join(RES, "india", a.mode, "prob_energy")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for seed in range(a.seeds):
        dst = os.path.join(out_dir, f"prob_energy_s{seed}.parquet")
        if os.path.exists(dst):
            continue
        ds, va, te = frames(a.mode, seed)
        feats = ds["feats"]
        wk = ((va.day - va.day.min()).dt.days // 7) % X.FOLDS
        cf_series, cf_inst = np.zeros(len(va)), np.zeros(len(va))
        for k in range(X.FOLDS):
            m = (wk == k).to_numpy()
            gs, gi = X.static_gates(va[~m], va[m], feats, seed + k)
            cf_series[m], cf_inst[m] = gs, gi
        ts_series, ts_inst = X.static_gates(va, te, feats, seed)
        va_roll = X.rolling_gate_online(va.iloc[0:0], va, 30, 2)
        va_roll = np.where(np.isnan(va_roll), cf_series, va_roll)
        te_roll = X.rolling_gate_online(va, te, 30, 2)
        te_roll = np.where(np.isnan(te_roll), ts_series, te_roll)
        forecasts = {"lgbm": (np.zeros(len(va)), np.zeros(len(te))), "corr_series": (cf_series, ts_series),
                     "corr_instance": (cf_inst, ts_inst), "corr_roll30": (va_roll, te_roll)}
        rows = te[["sid", "ts", "y_t", "B", "hour", "dam_mcp_rs_mwh", "rtm_mcp_rs_mwh"]].copy()
        summary = {}
        i = {q: int(np.where(np.isclose(LEVELS, q))[0][0]) for q in (0.05, 0.5, 0.95, 0.975, 0.99)}
        ci = [int(np.where(np.isclose(LEVELS, q))[0][0]) for q in P.CRPS_LEVELS]
        for fname, (gv, gt) in forecasts.items():
            Fv = va.B.to_numpy() + gv * va.r.to_numpy()
            Ft = te.B.to_numpy() + gt * te.r.to_numpy()
            cal = va.assign(e=va.y_t.to_numpy() - Fv)
            tst = te.assign(e=te.y_t.to_numpy() - Ft)
            rows[f"F_{fname}"] = Ft.astype(np.float32)
            Q = X.quantile_ladder(cal, tst, feats, seed)
            # a cell with too few calibration residuals for the conformal rank gives +/-inf (e.g. solar at dawn);
            # such rows use the per-series quantile instead (rule fixed on development data, before the test months)
            for qname in [k for k in Q if k.startswith("q_zone_")]:
                Q[qname] = np.where(np.isfinite(Q[qname]), Q[qname], Q["q_zone"])
            y = tst.e.to_numpy()
            for qname, Qm in Q.items():
                pin = P.pinball(y, Qm, LEVELS)
                crps_rows = P.crps_from_quantiles(y, Qm[:, ci], P.CRPS_LEVELS)
                summary[f"{fname}|{qname}"] = dict(
                    crps=float(crps_rows.mean()), pinball_99=float(pin[i[0.99]]), pinball_975=float(pin[i[0.975]]),
                    pinball_50=float(pin[i[0.5]]), cov90=P.coverage(y, Qm[:, i[0.05]], Qm[:, i[0.95]]),
                    width90=float(np.mean(Qm[:, i[0.95]] - Qm[:, i[0.05]])),
                    cov99_upper=float(np.mean(y <= Qm[:, i[0.99]])), cov975_upper=float(np.mean(y <= Qm[:, i[0.975]])),
                    ccd99_zone=P.conditional_coverage_deviation(y, Qm[:, i[0.99]], tst.sid.to_numpy(), 0.99),
                    ccd99_zone_hour=P.conditional_coverage_deviation(y, Qm[:, i[0.99]], G.cell_keys(tst.sid, tst.hour), 0.99))
                rows[f"crps_{fname}_{qname}"] = crps_rows.astype(np.float32)
                rows[f"up99_{fname}_{qname}"] = Qm[:, i[0.99]].astype(np.float32)
                rows[f"up975_{fname}_{qname}"] = Qm[:, i[0.975]].astype(np.float32)
                rows[f"pin99_{fname}_{qname}"] = P.pinball_rows(y, Qm[:, [i[0.99]]], [0.99])[:, 0].astype(np.float32)
            dev = energy.deviation_energy(te.y_t, Ft)
            imb = energy.imbalance_cost(te.y_t, Ft, te.rtm_mcp_rs_mwh, te.dam_mcp_rs_mwh)
            summary[f"{fname}|energy"] = dict(dev_mwh=float(dev.sum()), mae=float(dev.mean()),
                                              imbalance_rs=float(np.nansum(imb)),
                                              rmse=float(np.sqrt(np.mean((te.y_t - Ft) ** 2))))
            print(f"  seed {seed} {fname:14s} done [{time.time()-t0:.0f}s]", flush=True)
        rows.to_parquet(dst, index=False)
        json.dump(summary, open(dst.replace(".parquet", ".json"), "w"), indent=1)
    print("DONE India prob/energy", a.mode, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
