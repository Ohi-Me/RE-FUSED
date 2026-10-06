"""RE-FUSED-5 R8: cross-domain replication on public benchmarks (ETT family).
Same protocol as India/NYISO: persistence baseline, residual corrector, gate strategies
(never/always/global/static-per-series/shrunk), chronological 70/10/20 split.
Tests whether the granularity criterion transfers outside energy-panel data."""
import pandas as pd, numpy as np, os, io, json, urllib.request, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
D = r"D:\\REFUSED5\data\public"; OUT = r"D:\\REFUSED5\results"
os.makedirs(D, exist_ok=True)
BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
SETS = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]
def get(name):
    f = os.path.join(D, name)
    if not os.path.exists(f):
        urllib.request.urlretrieve(BASE + name, f)
    return pd.read_csv(f, parse_dates=["date"])
rows = []; t0 = time.time()
for name in SETS:
    try:
        d = get(name)
    except Exception as e:
        print(f"{name}: download failed {type(e).__name__} {e}"); continue
    # long format: each of the 7 channels is a "series"
    chans = [c for c in d.columns if c != "date"]
    L = d.melt(id_vars="date", value_vars=chans, var_name="series", value_name="y").dropna()
    L = L.sort_values(["series", "date"]).reset_index(drop=True)
    g = L.groupby("series")["y"]
    for lag in [1, 2, 3, 6, 12, 24]: L[f"lag{lag}"] = g.shift(lag)
    L["rm6"] = g.shift(1).rolling(6).mean().reset_index(0, drop=True)
    L["rm24"] = g.shift(1).rolling(24).mean().reset_index(0, drop=True)
    L["rs6"] = g.shift(1).rolling(6).std().reset_index(0, drop=True)
    L["hour"] = L.date.dt.hour; L["dow"] = L.date.dt.dayofweek
    L["h_sin"] = np.sin(2*np.pi*L.hour/24); L["h_cos"] = np.cos(2*np.pi*L.hour/24)
    L["cid"] = L.series.astype("category").cat.codes
    F = ["lag1","lag2","lag3","lag6","lag12","lag24","rm6","rm24","rs6","h_sin","h_cos","dow","cid"]
    ts = np.sort(L.date.unique()); q70, q80 = ts[int(.7*len(ts))], ts[int(.8*len(ts))]
    L["split"] = np.where(L.date <= q70, "train", np.where(L.date <= q80, "val", "test"))
    for h in [1, 6, 24]:
        dd = L.copy(); dd["y_t"] = dd.groupby("series")["y"].shift(-h); dd["B"] = dd["y"]
        dd = dd.dropna(subset=F + ["y_t", "B"]); dd["R"] = dd.y_t - dd.B
        tr, va, te = [dd[dd.split == s] for s in ["train","val","test"]]
        if min(len(tr), len(va), len(te)) < 500: continue
        m = HGB(max_iter=300, learning_rate=0.06, max_depth=6, random_state=0).fit(tr[F], tr.R)
        rv, rt = m.predict(va[F]), m.predict(te[F])
        gg = float(np.dot(va.R, rv)/np.dot(rv, rv))
        cell = {s_: (float(np.dot(va.R.values[i], rv[i])/np.dot(rv[i], rv[i])) if len(i) > 30 else gg)
                for s_, i in va.groupby("series").indices.items()}
        gs = te.series.map(cell).fillna(gg).values
        num = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=1).fit(va[F], va.R*rv)
        den = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=1).fit(va[F], rv**2)
        gf = np.clip(num.predict(te[F])/np.maximum(den.predict(te[F]), 1e-8), 0, 1)
        rng = np.random.default_rng(0); idx = rng.permutation(len(va)); a_, b_ = idx[::2], idx[1::2]
        def half(ii, sd):
            n2 = HGB(max_iter=150, learning_rate=0.07, max_depth=5, random_state=sd).fit(va[F].iloc[ii], (va.R.values*rv)[ii])
            d2 = HGB(max_iter=150, learning_rate=0.07, max_depth=5, random_state=sd).fit(va[F].iloc[ii], (rv**2)[ii])
            return np.clip(n2.predict(te[F])/np.maximum(d2.predict(te[F]), 1e-8), 0, 1)
        gA, gB = half(a_, 11), half(b_, 12)
        ve = float(np.mean((gA-gB)**2)/4.0); vw = max(float(np.var(gf))-ve, 0.0)
        lam = vw/(vw+ve+1e-12); gsh = np.clip(lam*gf + (1-lam)*gs, 0, 1)
        base = np.abs(te.R.values)
        out = {}
        for k, gv in [("never", 0.0), ("always", 1.0), ("global", gg), ("static", gs), ("free", gf), ("shrunk", gsh)]:
            err = np.abs(te.y_t.values - (te.B.values + gv*rt))
            per = [1 - err[i].mean()/base[i].mean() for i in te.groupby("series").indices.values()]
            out[k] = float(np.mean(per))
        rows.append(dict(dataset=name.replace(".csv",""), h=h, var_within=vw, var_est=ve, lam=lam,
                         rho=(vw/ve if ve > 0 else np.nan), **out))
        print(f"{name[:6]} h={h:3d} | " + " ".join(f"{k}={out[k]:+.4f}" for k in
              ["always","global","static","free","shrunk"]) + f" | rho={rows[-1]['rho']:.2f} [{time.time()-t0:.0f}s]", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "R8_public_benchmarks.csv"), index=False)
print("DONE")
