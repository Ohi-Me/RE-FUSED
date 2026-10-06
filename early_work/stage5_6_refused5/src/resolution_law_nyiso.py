"""RE-FUSED-5 R3: the RESOLUTION LAW (N4) on NYISO prices.
Same series at 5min / 15min / 1h / 1d; baselines = persistence and the day-ahead price.
Question: how do residual predictability E[m^2]/Var(R), the optimal gate g*, and the value of
correction scale with temporal resolution and horizon? (India 15-min -> US 5-min settlement axis.)"""
import pandas as pd, numpy as np, os, json, warnings, time, gc
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
D = r"D:\\REFUSED5\data"; OUT = r"D:\\REFUSED5\results"
STEPS = {"5min": [1, 3, 12], "15min": [1, 4], "1h": [1, 24], "1d": [1, 7]}   # ~5min,15min,1h / 1d,1w
rows = []; t0 = time.time()
for res, hs in STEPS.items():
    f = os.path.join(D, f"nyiso_{res}.parquet")
    if not os.path.exists(f):
        print(f"skip {res}: not built yet"); continue
    d = pd.read_parquet(f, columns=["zone","ts","y","da","split"]).sort_values(["zone","ts"])
    d["y"] = d["y"].astype("float32"); d["da"] = d["da"].astype("float32")
    d = d.dropna(subset=["y"])
    lim = {"5min": 11, "15min": 3, "1h": 0, "1d": 0}[res]
    if lim: d["da"] = d.groupby("zone")["da"].ffill(limit=lim)
    if res == "5min":
        keep_zones = d.zone.value_counts().head(5).index
        d = d[d.zone.isin(keep_zones) & (d.ts >= "2022-01-01")].copy()
    elif res == "15min":
        d = d[d.ts >= "2021-01-01"].copy()
    print(f"[{res}] rows after memory trim: {len(d):,} zones={d.zone.nunique()}", flush=True)
    g = d.groupby("zone")["y"]
    for L in [1, 2, 3, 6, 12, 24]: d[f"lag{L}"] = g.shift(L)
    d["rm6"] = g.shift(1).rolling(6).mean().reset_index(0, drop=True)
    d["rs6"] = g.shift(1).rolling(6).std().reset_index(0, drop=True)
    d["rm24"] = g.shift(1).rolling(24).mean().reset_index(0, drop=True)
    d["hour"] = d.ts.dt.hour; d["dow"] = d.ts.dt.dayofweek
    d["h_sin"] = np.sin(2*np.pi*d.hour/24); d["h_cos"] = np.cos(2*np.pi*d.hour/24)
    d["zid"] = d.zone.astype("category").cat.codes
    F = ["lag1","lag2","lag3","lag6","lag12","lag24","rm6","rs6","rm24","h_sin","h_cos","dow","zid","da"]
    for h in hs:
        gc.collect()
        dd = d.copy()
        dd["y_t"] = dd.groupby("zone")["y"].shift(-h)
        dd["b_pers"] = dd["y"]
        dd["b_da"] = dd.groupby("zone")["da"].shift(-h)     # DA price for the delivery interval
        for bname in ["b_pers", "b_da"]:
            keep = F + ["y_t", bname, "split", "zone", "y"]
            ddb = dd[keep].dropna().copy()
            for c in ddb.columns:
                if ddb[c].dtype == "float64": ddb[c] = ddb[c].astype("float32")
            if len(ddb) > 2_000_000: ddb = ddb.iloc[::max(1, len(ddb)//2_000_000)].copy()
            ddb["R"] = ddb.y_t - ddb[bname]
            tr, va, te = [ddb[ddb.split == s] for s in ["train", "val", "test"]]
            if min(len(tr), len(va), len(te)) < 2000: continue
            sub = min(len(tr), 250_000)
            tr_s = tr.sample(sub, random_state=0) if len(tr) > sub else tr
            m = HGB(max_iter=300, learning_rate=0.07, max_depth=6, random_state=0).fit(tr_s[F], tr_s.R)
            rv, rt = m.predict(va[F]), m.predict(te[F])
            Rv, Rt = va.R.values, te.R.values
            g_glob = float(np.dot(Rv, rv) / np.dot(rv, rv))
            cell = {}
            for z, idx in va.groupby("zone").indices.items():
                cell[z] = float(np.dot(Rv[idx], rv[idx]) / np.dot(rv[idx], rv[idx])) if len(idx) > 50 else g_glob
            g_static = te.zone.map(cell).fillna(g_glob).values
            preds = {"never": np.zeros(len(te)), "always": np.ones(len(te)),
                     "global": np.full(len(te), g_glob), "static_zone": g_static}
            out = {}
            for nm, gv in preds.items():
                pr = te[bname].values + gv * rt
                err = np.abs(te.y_t.values - pr); base = np.abs(Rt)
                out[nm] = float(np.mean([1 - err[i].mean()/base[i].mean()
                                         for i in te.groupby("zone").indices.values()]))
            # residual predictability on test
            r2 = float(1 - np.mean((Rt - rt) ** 2) / np.var(Rt))
            rows.append(dict(res=res, h=h, baseline=bname, g_global=g_glob, resid_R2=r2,
                             base_MAE=float(np.abs(Rt).mean()), y_sd=float(te.y.std()),
                             n_test=int(len(te)), **out))
            print(f"{res:5s} h={h:3d} {bname:7s} | baseMAE={np.abs(Rt).mean():7.2f} residR2={r2:+.3f} "
                  f"g={g_glob:.3f} | always={out['always']:+.3f} global={out['global']:+.3f} "
                  f"static={out['static_zone']:+.3f} [{time.time()-t0:.0f}s]", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "R3_resolution_law.csv"), index=False)

print("DONE", time.time() - t0)
