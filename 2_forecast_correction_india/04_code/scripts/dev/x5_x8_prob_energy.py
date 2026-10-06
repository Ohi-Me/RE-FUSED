"""X5 + X8 (development): adaptivity ladder for forecast-error quantiles and energy-system valuation, NYISO load.

Point forecasts F (day-ahead, per zone-hour):
  iso              NYISO day-ahead load forecast (operational reference)
  corr_series      ISO + static per-zone gate x corrector (RE-FUSED-6 recommended coarse procedure)
  corr_instance    ISO + per-instance WLS gate x corrector
  corr_roll30      ISO + per-zone gate re-estimated daily on the last 30 days (time axis)
Calibration residuals on the validation year are OUT-OF-SAMPLE: static gates are cross-fitted over interleaved
week folds (fit on the other folds, predict this fold); the rolling gate uses only data up to D-2.
Test-time gates: static gates refitted on the whole validation year; rolling gate updated online (delay 2 days).

Quantile ladder for e = A - F (the adaptivity of the uncertainty model):
  q_global, q_zone, q_zone_hourblock, q_zone_hour, q_zone_month, q_instance (gradient-boosted quantiles),
  q_roll90 (per zone, last 90 days of residuals, updated online with delay 2 days)
Metrics (test year): pinball per level, CRPS (19 levels), coverage of the 90% central interval, one-sided
coverage of the 97.5% / 99% upper quantile, conditional coverage deviation by zone and by zone x hour.
Energy: absolute deviation energy, two-settlement imbalance cost, upward reserve at q = 0.99 and 0.975
(MW, cost at the day-ahead 10-min spinning reserve price, shortfall energy). All per-row, saved for bootstrap.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import data, energy, gates as G, ladder, probabilistic as P  # noqa: E402
from refused_gate.paths import RES  # noqa: E402

LEVELS = P.ALL_LEVELS
FOLDS = 4


def corrector_and_frames(ds, seed):
    x = ds["df"].assign(y_t=ds["df"].y, B=ds["df"].iso_fc).dropna(subset=ds["feats"] + ["y_t", "B"]).copy()
    x["R"] = x.y_t - x.B
    x["day"] = x.ts.dt.normalize()
    tr, va, te = (x[x.split == s].copy() for s in ("train", "val", "test"))
    cfg = ladder.LadderConfig()
    corr = ladder._fit_corrector(tr[ds["feats"]].to_numpy(np.float32), tr.R.to_numpy(), cfg, seed)
    va["r"] = corr.predict(va[ds["feats"]].to_numpy(np.float32))
    te["r"] = corr.predict(te[ds["feats"]].to_numpy(np.float32))
    g1 = G.ls_gate(va.R.to_numpy(), va.r.to_numpy())
    va["r"] *= g1
    te["r"] *= g1
    return va, te


def static_gates(fit, pred, feats, seed):
    gs = G.PartitionGate(30).fit(fit.R.to_numpy(), fit.r.to_numpy(), fit.sid.to_numpy())
    g_series = np.clip(gs.predict(pred.sid.to_numpy()), *G.CLIP)
    gi = G.InstanceGate(n_bag=3, seed=seed).fit(fit[feats].to_numpy(np.float32), fit.R.to_numpy(), fit.r.to_numpy())
    g_inst = np.clip(gi.predict(pred[feats].to_numpy(np.float32)), *G.CLIP)
    return g_series, g_inst


def rolling_gate_online(hist, pred, window=30, delay=2):
    """Per-zone LS gate on the last `window` days of residuals known by D - delay (hist rows must precede)."""
    both = pd.concat([hist.assign(_p=0), pred.assign(_p=1)], ignore_index=False)
    both["Rr"], both["rr"] = both.R * both.r, both.r * both.r
    out = pd.Series(np.nan, index=pred.index)
    for s, g in both.groupby("sid"):
        daily = g.groupby("day")[["Rr", "rr"]].sum()
        days = daily.index.to_numpy()
        cRr, crr = np.cumsum(daily.Rr.to_numpy()), np.cumsum(daily.rr.to_numpy())
        gmap = {}
        for D in np.unique(g[g._p == 1].day.to_numpy()):
            j = np.searchsorted(days, D - np.timedelta64(delay, "D"), side="right") - 1
            i0 = np.searchsorted(days, days[j] - np.timedelta64(window, "D"), side="right") - 1 if j >= 0 else -1
            if j < 0:
                continue
            num = cRr[j] - (cRr[i0] if i0 >= 0 else 0)
            den = crr[j] - (crr[i0] if i0 >= 0 else 0)
            gmap[D] = num / den if den > 1e-12 else np.nan
        m = pred.sid == s
        out[m] = pred.loc[m, "day"].map(gmap).to_numpy()
    return np.clip(out.to_numpy(), *G.CLIP)


def quantile_ladder(cal, tst, feats, seed):
    """Returns dict level_name -> (n_test x len(LEVELS)) quantiles of e = A - F."""
    e = cal.e.to_numpy()
    Q = {}
    Q["q_global"] = np.tile(np.array([P.conformal_quantile(e, q) for q in LEVELS]), (len(tst), 1))
    for name, keys_c, keys_t in [
        ("q_zone", cal.sid, tst.sid),
        ("q_zone_hourblock", G.cell_keys(cal.sid, cal.hour_block), G.cell_keys(tst.sid, tst.hour_block)),
        ("q_zone_hour", G.cell_keys(cal.sid, cal.hour), G.cell_keys(tst.sid, tst.hour)),
        ("q_zone_month", G.cell_keys(cal.sid, cal.ts.dt.month), G.cell_keys(tst.sid, tst.ts.dt.month)),
    ]:
        Q[name] = P.PartitionQuantiles(LEVELS, min_cell=30).fit(e, np.asarray(keys_c)).predict(np.asarray(keys_t))
    cq = P.ConditionalQuantiles(LEVELS, seed=seed).fit(cal[feats].to_numpy(np.float32), e)
    Q["q_instance"] = cq.predict(tst[feats].to_numpy(np.float32))
    # time axis: rolling 90-day per-zone empirical quantiles, updated online with delay 2 days
    both = pd.concat([cal.assign(_p=0), tst.assign(_p=1)], ignore_index=False)
    qr = np.full((len(tst), len(LEVELS)), np.nan)
    pos = {ix: i for i, ix in enumerate(tst.index)}
    for s, g in both.groupby("sid"):
        g = g.sort_values("ts")
        days = g.day.to_numpy()
        ev = g.e.to_numpy()
        for D in np.unique(g[g._p == 1].day.to_numpy()):
            m = (days <= D - np.timedelta64(2, "D")) & (days > D - np.timedelta64(92, "D"))
            vals = np.array([P.conformal_quantile(ev[m], q) for q in LEVELS]) if m.sum() >= 100 else None
            rows = g.index[(g._p == 1) & (g.day == D)]
            for ix in rows:
                qr[pos[ix]] = vals if vals is not None else np.nan
    fallback = Q["q_zone"]
    Q["q_roll90"] = np.where(np.isnan(qr), fallback, qr)
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mode", default="dev", choices=["dev", "confirm"])
    a = ap.parse_args()
    out_dir = os.path.join(RES, a.mode, "x5_x8")
    os.makedirs(out_dir, exist_ok=True)
    ds = data.nyiso_load(a.mode)
    feats = ds["feats"]
    t0 = time.time()
    for seed in range(a.seeds):
        dst = os.path.join(out_dir, f"prob_energy_s{seed}.parquet")
        if os.path.exists(dst):
            continue
        va, te = corrector_and_frames(ds, seed)
        # cross-fitted static gates on validation (interleaved week folds)
        wk = ((va.day - va.day.min()).dt.days // 7) % FOLDS
        cf_series, cf_inst = np.zeros(len(va)), np.zeros(len(va))
        for k in range(FOLDS):
            m = (wk == k).to_numpy()
            gs, gi = static_gates(va[~m], va[m], feats, seed + k)
            cf_series[m], cf_inst[m] = gs, gi
        ts_series, ts_inst = static_gates(va, te, feats, seed)
        # rolling gates: validation year online from its own start; test online with validation history
        va_roll = rolling_gate_online(va.iloc[0:0], va, 30, 2)
        va_roll = np.where(np.isnan(va_roll), cf_series, va_roll)
        te_roll = rolling_gate_online(va, te, 30, 2)
        te_roll = np.where(np.isnan(te_roll), ts_series, te_roll)
        forecasts = {"iso": (np.zeros(len(va)), np.zeros(len(te))), "corr_series": (cf_series, ts_series),
                     "corr_instance": (cf_inst, ts_inst), "corr_roll30": (va_roll, te_roll)}
        rows = te[["sid", "ts", "y_t", "B", "da_lbmp", "rt_lbmp", "da_spin", "hour"]].copy()
        summary = {}
        for fname, (gv, gt) in forecasts.items():
            Fv = va.B.to_numpy() + gv * va.r.to_numpy()
            Ft = te.B.to_numpy() + gt * te.r.to_numpy()
            cal = va.assign(e=va.y_t.to_numpy() - Fv)
            tst = te.assign(e=te.y_t.to_numpy() - Ft)
            rows[f"F_{fname}"] = Ft.astype(np.float32)
            Q = quantile_ladder(cal, tst, feats, seed)
            y = tst.e.to_numpy()
            ci = [int(np.where(np.isclose(LEVELS, q))[0][0]) for q in P.CRPS_LEVELS]
            for qname, Qm in Q.items():
                pin = P.pinball(y, Qm, LEVELS)
                crps_rows = P.crps_from_quantiles(y, Qm[:, ci], P.CRPS_LEVELS)
                i05, i95 = int(np.where(np.isclose(LEVELS, 0.05))[0][0]), int(np.where(np.isclose(LEVELS, 0.95))[0][0])
                i975, i99 = int(np.where(np.isclose(LEVELS, 0.975))[0][0]), int(np.where(np.isclose(LEVELS, 0.99))[0][0])
                key = f"{fname}|{qname}"
                summary[key] = dict(
                    crps=float(crps_rows.mean()), pinball_99=float(pin[i99]), pinball_975=float(pin[i975]),
                    pinball_50=float(pin[int(np.where(np.isclose(LEVELS, 0.5))[0][0])]),
                    cov90=P.coverage(y, Qm[:, i05], Qm[:, i95]), width90=float(np.mean(Qm[:, i95] - Qm[:, i05])),
                    cov99_upper=float(np.mean(y <= Qm[:, i99])), cov975_upper=float(np.mean(y <= Qm[:, i975])),
                    ccd99_zone=P.conditional_coverage_deviation(y, Qm[:, i99], tst.sid.to_numpy(), 0.99),
                    ccd99_zone_hour=P.conditional_coverage_deviation(y, Qm[:, i99], G.cell_keys(tst.sid, tst.hour), 0.99))
                rows[f"crps_{fname}_{qname}"] = crps_rows.astype(np.float32)
                rows[f"up99_{fname}_{qname}"] = Qm[:, i99].astype(np.float32)
                rows[f"up975_{fname}_{qname}"] = Qm[:, i975].astype(np.float32)
                rows[f"pin99_{fname}_{qname}"] = P.pinball_rows(y, Qm[:, [i99]], [0.99])[:, 0].astype(np.float32)
            dev = energy.deviation_energy(te.y_t, Ft)
            imb = energy.imbalance_cost(te.y_t, Ft, te.rt_lbmp, te.da_lbmp)
            summary[f"{fname}|energy"] = dict(dev_mwh=float(dev.sum()), imbalance_usd=float(np.nansum(imb)),
                                              mae=float(dev.mean()), rmse=float(np.sqrt(np.mean((te.y_t - Ft) ** 2))))
            print(f"  seed {seed} {fname:14s} MAE {dev.mean():7.2f} imbalance ${np.nansum(imb)/1e6:7.2f}M "
                  f"CRPS(zone_hour) {summary[f'{fname}|q_zone_hour']['crps']:.2f} [{time.time()-t0:.0f}s]", flush=True)
        rows.to_parquet(dst, index=False)
        json.dump(summary, open(dst.replace(".parquet", ".json"), "w"), indent=1)
    print("DONE x5_x8", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
