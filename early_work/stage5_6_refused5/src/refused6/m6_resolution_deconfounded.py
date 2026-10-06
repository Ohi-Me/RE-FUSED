"""RE-FUSED-6 / M6: de-confounded resolution study (resolves defect D4).

RE-FUSED-5 compared 5-min / 15-min / hourly / daily while simultaneously varying zones (5 vs 15),
window (2022+ vs 2019+) and, critically, the forecast horizon in CLOCK time. The apparent
"correction value rises as resolution refines" claim was therefore withdrawn.

Here everything is matched:
  * the same 5 zones and the same window for every resolution
  * the same CLOCK horizons (1 hour and 24 hours ahead), converted to steps per resolution
  * the same split dates, the same features, the same ladder, the same seeds
So the only thing varying is the sampling resolution of the series.

Reported per (resolution, clock horizon, baseline): baseline MAE, out-of-sample residual R^2,
and ladder skill. This separates two distinct questions:
  Q1 does the BASELINE get worse at finer resolution?      (a property of the market/series)
  Q2 does the learned CORRECTION recover more at finer resolution?  (a property of predictability)

Usage: py -3.10 -u src/refused6/m6_resolution_deconfounded.py
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

ROOT = r"D:\\REFUSED5"
OUT = os.path.join(ROOT, "results", "refused6")
CLIP, N_SEEDS, N_BOOT, MIN_CELL, N_REGIMES = 1.5, 5, 5, 30, 3
Z_CLIP = 10.0
WINDOW_START = "2022-01-01"
STEPS_PER_HOUR = {"5min": 12, "15min": 4, "1h": 1, "1d": None}
CLOCK_HORIZONS = [1, 24]          # hours ahead


def ls_gate(R, r):
    den = float(np.dot(r, r))
    return float(np.dot(R, r) / den) if den > 1e-12 else 0.0


def wls_gate(F, R, r, seed):
    w = r ** 2
    ok = w > 1e-12
    z = np.clip(R[ok] / r[ok], -Z_CLIP, Z_CLIP)
    m = HGB(max_iter=150, learning_rate=0.08, max_depth=5, random_state=seed)
    m.fit(F[ok], z, sample_weight=w[ok])
    return m


def load(res, zones):
    f = os.path.join(ROOT, "data", f"nyiso_{res}.parquet")
    d = pd.read_parquet(f, columns=["zone", "ts", "y", "da", "split"])
    d = d[d.zone.isin(zones) & (d.ts >= WINDOW_START)].dropna(subset=["y"]).copy()
    lim = {"5min": 11, "15min": 3, "1h": 0, "1d": 0}[res]
    if lim:
        d["da"] = d.groupby("zone")["da"].ffill(limit=lim)
    d = d.sort_values(["zone", "ts"]).reset_index(drop=True)
    g = d.groupby("zone")["y"]
    for L in [1, 2, 3, 6, 12]:
        d[f"lag{L}"] = g.shift(L)
    d["rm6"] = g.shift(1).rolling(6).mean().reset_index(0, drop=True)
    d["rs6"] = g.shift(1).rolling(6).std().reset_index(0, drop=True)
    d["rm24"] = g.shift(1).rolling(24).mean().reset_index(0, drop=True)
    d["hour"] = d.ts.dt.hour
    d["dow"] = d.ts.dt.dayofweek
    d["h_sin"] = np.sin(2 * np.pi * d.hour / 24)
    d["h_cos"] = np.cos(2 * np.pi * d.hour / 24)
    d["zid"] = d.zone.astype("category").cat.codes
    return d


FEATS = ["lag1", "lag2", "lag3", "lag6", "lag12", "rm6", "rs6", "rm24", "h_sin", "h_cos", "dow", "zid"]


def run(res, clock_h, zones, seed):
    spf = STEPS_PER_HOUR[res]
    if res == "1d":
        if clock_h != 24:
            return None
        steps = 1
    else:
        steps = spf * clock_h
    d = load(res, zones)
    d["y_t"] = d.groupby("zone")["y"].shift(-steps)
    d["b_pers"] = d["y"]
    d["b_da"] = d.groupby("zone")["da"].shift(-steps)
    rows = []
    for bname in ["b_pers", "b_da"]:
        cols = list(dict.fromkeys(FEATS + ["y_t", bname, "split", "zone", "zid", "rs6", "ts"]))
        x = d[cols].dropna().copy()
        assert x.columns.is_unique, "duplicate columns would break 1-D indexing"
        if len(x) < 3000:
            continue
        x["R"] = x.y_t - x[bname]
        tr, va, te = [x[x.split == s] for s in ("train", "val", "test")]
        if min(len(tr), len(va), len(te)) < 600:
            continue
        sub = min(len(tr), 200_000)
        trs = tr.sample(sub, random_state=seed) if len(tr) > sub else tr
        corr = HGB(max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed,
                   l2_regularization=1.0).fit(trs[FEATS], trs.R)
        rv, rt = corr.predict(va[FEATS]), corr.predict(te[FEATS])
        Rv, Rt = va.R.to_numpy(), te.R.to_numpy()
        g1 = ls_gate(Rv, rv)
        rv, rt = g1 * rv, g1 * rt
        g1e = ls_gate(Rv, rv)
        zv, zt = va.zid.to_numpy(), te.zid.to_numpy()
        lo, hi = np.nanquantile(tr.rs6, [1 / 3, 2 / 3])
        qv, qt = np.digitize(va.rs6.to_numpy(), [lo, hi]), np.digitize(te.rs6.to_numpy(), [lo, hi])
        g2map = {z: (ls_gate(Rv[zv == z], rv[zv == z]) if (zv == z).sum() >= MIN_CELL else g1e) for z in np.unique(zv)}
        g2 = np.array([g2map.get(z, g1e) for z in zt])
        g3map = {}
        for z in np.unique(zv):
            for q in range(N_REGIMES):
                k = (zv == z) & (qv == q)
                g3map[(z, q)] = ls_gate(Rv[k], rv[k]) if k.sum() >= MIN_CELL else g2map.get(z, g1e)
        g3 = np.array([g3map.get((z, q), g1e) for z, q in zip(zt, qt)])
        Fv, Ft = va[FEATS].to_numpy(), te[FEATS].to_numpy()
        bw = np.empty((N_BOOT, len(Ft)))
        for b in range(N_BOOT):
            rb = np.random.default_rng(900 + 31 * seed + b)
            idx = rb.integers(0, len(Fv), len(Fv))
            bw[b] = wls_gate(Fv[idx], Rv[idx], rv[idx], seed * 29 + b).predict(Ft)
        g5 = bw.mean(axis=0)
        V, W = float(np.mean(bw.var(axis=0, ddof=1))), float(np.var(g5))
        lam = W / (W + V) if (W + V) > 0 else 0.0
        gates = {"rung0_none": np.zeros(len(te)), "rung1_global": np.full(len(te), g1e),
                 "rung2_series": g2, "rung3_cell": g3, "rung5_instance": g5,
                 "rung4_shrunk": lam * g5 + (1 - lam) * g2}
        base = Rt ** 2
        for rung, g in gates.items():
            gc = np.clip(g, 0, CLIP)
            sq = (Rt - gc * rt) ** 2
            msq = [1 - sq[zt == z].mean() / base[zt == z].mean() for z in np.unique(zt)]
            rows.append(dict(res=res, clock_h=clock_h, steps=steps, baseline=bname, rung=rung, seed=seed,
                             skill_sq_macro=float(np.mean(msq)),
                             base_mae=float(np.abs(Rt).mean()), base_mse=float(base.mean()),
                             resid_R2=float(1 - np.mean((Rt - rt) ** 2) / np.var(Rt)),
                             y_sd=float(te.y_t.std()), n_test=len(te), n_val=len(va),
                             lam=lam, rho=(W / V if V > 0 else np.nan)))
    return rows


def main():
    # the five zones with the most 5-min coverage, used for EVERY resolution
    z5 = pd.read_parquet(os.path.join(ROOT, "data", "nyiso_5min.parquet"), columns=["zone"])
    zones = z5.zone.value_counts().head(5).index.tolist()
    print("matched zones:", zones)
    print("matched window start:", WINDOW_START)
    allr, t0 = [], time.time()
    for res in ["5min", "15min", "1h", "1d"]:
        for ch in CLOCK_HORIZONS:
            for seed in range(N_SEEDS):
                r = run(res, ch, zones, seed)
                if r:
                    allr += r
            d = pd.DataFrame(allr).query("res == @res and clock_h == @ch")
            if len(d):
                for bn in d.baseline.unique():
                    dd = d[d.baseline == bn]
                    m = dd.groupby("rung").skill_sq_macro.mean()
                    print(f"{res:5s} clock_h={ch:2d}h steps={int(dd.steps.iloc[0]):3d} {bn:7s} | "
                          f"baseMAE={dd.base_mae.mean():7.3f} residR2={dd.resid_R2.mean():+.3f} | " +
                          " ".join(f"{k.replace('rung','r')[:10]}={m.get(k, float('nan')):+.4f}" for k in
                                   ["rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]) +
                          f" [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(allr)
    R.to_csv(os.path.join(OUT, "M6_resolution_deconfounded.csv"), index=False)
    print(f"\nwrote {len(R)} rows")
    if len(R):
        print("\n=== Q1: baseline error by resolution at matched clock horizon ===")
        print(R[R.rung == "rung1_global"].pivot_table(index=["clock_h", "baseline"], columns="res",
                                                      values="base_mae").round(3).to_string())
        print("\n=== Q2: correction value (best coarse rung) by resolution ===")
        best = R[R.rung.isin(["rung2_series", "rung3_cell"])].groupby(
            ["clock_h", "baseline", "res"]).skill_sq_macro.max().unstack("res")
        print(best.round(4).to_string())
        print("\n=== residual predictability by resolution ===")
        print(R[R.rung == "rung1_global"].pivot_table(index=["clock_h", "baseline"], columns="res",
                                                      values="resid_R2").round(3).to_string())
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
