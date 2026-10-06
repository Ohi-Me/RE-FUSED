"""RE-FUSED-5 R2: value of a context block (VOC), by horizon, with permutation controls.
For each block C: train corrector with and without C (identical capacity/seeds), measure
skill change on TEST, paired across states; control = C permuted within state (same capacity,
no information). VOC is credited only if real-C beats permuted-C."""
import pandas as pd, numpy as np, os, json, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from scipy import stats
D = r"D:\\REFUSED5\data"; OUT = r"D:\\REFUSED5\results"
p = pd.read_parquet(os.path.join(D, "india_model_daily.parquet"))
ctx = pd.read_parquet(os.path.join(D, "india_panel_raw_rebuild.parquet"))[
        ["date","state_name","re_total","coal_cons","coal_req","outage_mw","wind","solar"]]
p = p.merge(ctx, on=["date","state_name"], how="left").sort_values(["state_name","date"])
p = p[p.date >= "2019-11-01"]                      # window where renewables+coal exist
g = p.groupby("state_name")["y"]
for L in [1,2,3,7,14,28]: p[f"lag{L}"] = g.shift(L)
p["rm7"] = g.shift(1).rolling(7).mean().reset_index(0,drop=True)
p["rm28"] = g.shift(1).rolling(28).mean().reset_index(0,drop=True)
p["rs7"] = g.shift(1).rolling(7).std().reset_index(0,drop=True)
p["dow_sin"] = np.sin(2*np.pi*p.dow/7); p["dow_cos"] = np.cos(2*np.pi*p.dow/7)
p["doy_sin"] = np.sin(2*np.pi*p.doy/365.25); p["doy_cos"] = np.cos(2*np.pi*p.doy/365.25)
p["sid"] = p.state_name.astype("category").cat.codes
# context blocks are lagged (only past values are available at forecast time)
for c in ["re_total","coal_cons","coal_req","outage_mw","wind","solar"]:
    p[c+"_l1"] = p.groupby("state_name")[c].shift(1)
BASE = ["lag1","lag2","lag3","lag7","lag14","lag28","rm7","rm28","rs7",
        "dow_sin","dow_cos","doy_sin","doy_cos","cap","n_units","sid"]
BLOCKS = {"renewables": ["re_total_l1","wind_l1","solar_l1"],
          "coal":       ["coal_cons_l1","coal_req_l1"],
          "outage":     ["outage_mw_l1"]}
rows = []; t0 = time.time()
for h in [1, 3, 7]:
    d = p.copy(); d["y_t"] = d.groupby("state_name")["y"].shift(-h); d["B"] = d["y"]
    need = BASE + sum(BLOCKS.values(), []) + ["y_t"]
    d = d.dropna(subset=need); d["R"] = d.y_t - d.B
    tr, te = d[d.split=="train"], d[d.split=="test"]
    if len(tr) < 1000 or len(te) < 500: continue
    def fit_eval(cols, seed, permute=None):
        Xtr, Xte = tr[cols].copy(), te[cols].copy()
        if permute:
            rng = np.random.default_rng(seed)
            for c in permute:                       # break the information, keep the capacity
                Xtr[c] = tr.groupby("state_name")[c].transform(lambda s: rng.permutation(s.values))
                Xte[c] = te.groupby("state_name")[c].transform(lambda s: rng.permutation(s.values))
        m = HGB(max_iter=400, learning_rate=0.06, max_depth=6, random_state=seed).fit(Xtr, tr.R)
        pred = te.B.values + m.predict(Xte)
        err = np.abs(te.y_t.values - pred); base = np.abs(te.R.values)
        per_state = [1 - err[i].mean()/base[i].mean() for i in te.groupby("state_name").indices.values()]
        return np.array(per_state)
    seeds = range(5)
    base_sk = np.mean([fit_eval(BASE, s) for s in seeds], axis=0)
    for bname, cols in BLOCKS.items():
        with_sk = np.mean([fit_eval(BASE+cols, s) for s in seeds], axis=0)
        perm_sk = np.mean([fit_eval(BASE+cols, s, permute=cols) for s in seeds], axis=0)
        t_real, p_real = stats.ttest_rel(with_sk, base_sk)
        t_ctrl, p_ctrl = stats.ttest_rel(with_sk, perm_sk)
        rows.append(dict(h=h, block=bname, skill_base=float(base_sk.mean()), skill_with=float(with_sk.mean()),
                         skill_perm=float(perm_sk.mean()), voc=float(with_sk.mean()-base_sk.mean()),
                         voc_vs_perm=float(with_sk.mean()-perm_sk.mean()), p_vs_base=float(p_real),
                         p_vs_perm=float(p_ctrl), states_improved=int((with_sk>base_sk).sum()), n_states=len(base_sk)))
        print(f"h={h} {bname:11s} base={base_sk.mean():+.4f} with={with_sk.mean():+.4f} perm={perm_sk.mean():+.4f} "
              f"| VOC={rows[-1]['voc']:+.4f} (p={p_real:.3f}) vs-perm={rows[-1]['voc_vs_perm']:+.4f} "
              f"(p={p_ctrl:.3f}) states+={rows[-1]['states_improved']}/{len(base_sk)} [{time.time()-t0:.0f}s]", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "R2_voc_india.csv"), index=False)
print("\nDONE", time.time()-t0)
