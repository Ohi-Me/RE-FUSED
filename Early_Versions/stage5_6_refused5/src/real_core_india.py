"""RE-FUSED-5 R1: correction-value law on REAL data (India balanced panel).
For each baseline (graded strength) and horizon h: residual corrector + gating strategies.
Strategies: never | always | global | static per (state,h) | free per-instance | shrunk hierarchical.
Strict protocol: fit corrector on TRAIN, fit gates on VAL(2023), evaluate on TEST(2024-2025)."""
import pandas as pd, numpy as np, json, os, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
D = r"D:\\REFUSED5\data"; OUT = r"D:\\REFUSED5\results"
p = pd.read_parquet(os.path.join(D, "india_model_daily.parquet")).sort_values(["state_name","date"])
HOR = [1, 2, 3, 7]
def add_feats(d):
    g = d.groupby("state_name")["y"]
    for L in [1, 2, 3, 7, 14, 28]: d[f"lag{L}"] = g.shift(L)
    d["rm7"] = g.shift(1).rolling(7).mean().reset_index(0, drop=True)
    d["rm28"] = g.shift(1).rolling(28).mean().reset_index(0, drop=True)
    d["rs7"] = g.shift(1).rolling(7).std().reset_index(0, drop=True)
    d["diff1"] = d.lag1 - d.lag2
    d["dow_sin"] = np.sin(2*np.pi*d.dow/7); d["dow_cos"] = np.cos(2*np.pi*d.dow/7)
    d["doy_sin"] = np.sin(2*np.pi*d.doy/365.25); d["doy_cos"] = np.cos(2*np.pi*d.doy/365.25)
    d["sid"] = d.state_name.astype("category").cat.codes
    return d
p = add_feats(p)
FEATS = ["lag1","lag2","lag3","lag7","lag14","lag28","rm7","rm28","rs7","diff1",
         "dow_sin","dow_cos","doy_sin","doy_cos","cap","n_units","sid"]
rows, diag = [], []
t0 = time.time()
for h in HOR:
    d = p.copy()
    d["y_t"] = d.groupby("state_name")["y"].shift(-h)                 # target at t+h
    d["b_persist"] = d["y"]                                           # last observed value
    d["b_snaive"] = d.groupby("state_name")["y"].shift(-h + 7) if h <= 7 else np.nan
    d["b_sched"] = d.groupby("state_name")["sched"].shift(-h)         # operator program for t+h
    d["b_rm7"] = d["rm7"]
    d = d.dropna(subset=["y_t","b_persist","b_snaive","b_sched","b_rm7"] + FEATS)
    for bname in ["b_persist", "b_rm7", "b_snaive", "b_sched"]:
        d["R"] = d.y_t - d[bname]
        tr, va, te = [d[d.split == s] for s in ["train","val","test"]]
        if min(len(tr), len(va), len(te)) < 500: continue
        corr = HGB(max_iter=400, learning_rate=0.06, max_depth=6, random_state=0).fit(tr[FEATS], tr.R)
        rv, rt = corr.predict(va[FEATS]), corr.predict(te[FEATS])
        Rv, Rt = va.R.values, te.R.values
        g_glob = float(np.dot(Rv, rv) / np.dot(rv, rv))
        # static per-state gate (cell = state at this horizon)
        cellg = {}
        for st_, idx in va.groupby("state_name").indices.items():
            cellg[st_] = float(np.dot(Rv[idx], rv[idx]) / np.dot(rv[idx], rv[idx])) if len(idx) > 30 else g_glob
        g_static = te.state_name.map(cellg).fillna(g_glob).values
        Fva, Fte = va[FEATS].values, te[FEATS].values
        num = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=1).fit(Fva, Rv * rv)
        den = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=1).fit(Fva, rv ** 2)
        g_free = np.clip(num.predict(Fte) / np.maximum(den.predict(Fte), 1e-8), 0, 1)
        rng = np.random.default_rng(0); idx = rng.permutation(len(Fva)); a_, b_ = idx[::2], idx[1::2]
        gA = np.clip(HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=2).fit(Fva[a_], (Rv*rv)[a_]).predict(Fte) /
                     np.maximum(HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=2).fit(Fva[a_], (rv**2)[a_]).predict(Fte), 1e-8), 0, 1)
        gB = np.clip(HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=3).fit(Fva[b_], (Rv*rv)[b_]).predict(Fte) /
                     np.maximum(HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=3).fit(Fva[b_], (rv**2)[b_]).predict(Fte), 1e-8), 0, 1)
        var_est = float(np.mean((gA - gB) ** 2) / 4.0); var_tot = float(np.var(g_free))
        var_within = max(var_tot - var_est, 0.0); lam = var_within / (var_within + var_est + 1e-12)
        g_shrunk = np.clip(lam * g_free + (1 - lam) * g_static, 0, 1)
        strategies = {"never": np.zeros(len(te)), "always": np.ones(len(te)), "global": np.full(len(te), g_glob),
                      "static_state": g_static, "free": g_free, "shrunk": g_shrunk}
        for sname, gv in strategies.items():
            pred = te[bname].values + gv * rt
            err = np.abs(te.y_t.values - pred)
            base_err = np.abs(Rt)
            sk, deg = [], 0
            for st_, idx in te.groupby("state_name").indices.items():
                s_ = 1 - err[idx].mean() / base_err[idx].mean(); sk.append(s_); deg += int(s_ < 0)
            rows.append(dict(h=h, baseline=bname, strategy=sname, MAE=float(err.mean()),
                             skill=float(np.mean(sk)), degraded_states=deg, n_states=len(sk)))
        diag.append(dict(h=h, baseline=bname, g_global=g_glob, var_within=var_within,
                         var_est=var_est, lam=float(lam), base_MAE=float(np.abs(Rt).mean())))
        print(f"h={h} {bname:10s} g_glob={g_glob:.3f} lam={lam:.2f} "
              f"| " + " ".join(f"{s}={[r for r in rows if r['h']==h and r['baseline']==bname and r['strategy']==s][0]['skill']:+.3f}"
                               for s in ["always","global","static_state","free","shrunk"]) + f"  [{time.time()-t0:.0f}s]", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "R1_india_core.csv"), index=False)
pd.DataFrame(diag).to_csv(os.path.join(OUT, "R1_india_diag.csv"), index=False)
print("\n=== mean skill vs baseline (over horizons), by baseline and strategy")
print(R.pivot_table(index="baseline", columns="strategy", values="skill").round(4).to_string())
print("\n=== degraded states (sum over horizons, max 20 per h)")
print(R.pivot_table(index="baseline", columns="strategy", values="degraded_states", aggfunc="sum").to_string())
