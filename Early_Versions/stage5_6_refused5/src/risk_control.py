"""RE-FUSED-5 R4: risk-controlled deployment (tests N3 / falsification F4).
Rule: correct a cell only if a one-sided lower confidence bound on its validation skill,
computed over TIME BLOCKS (no exchangeability across time assumed), exceeds 0.
Compare with: always | static gate (no risk control) | CRC-style direction-gating + clipping.
Metrics: test skill, degraded cells (safety), and conservatism (cells withheld that would have helped)."""
import pandas as pd, numpy as np, os, json, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
D = r"D:\\REFUSED5\data"; OUT = r"D:\\REFUSED5\results"
p = pd.read_parquet(os.path.join(D, "india_model_daily.parquet")).sort_values(["state_name","date"])
g = p.groupby("state_name")["y"]
for L in [1,2,3,7,14,28]: p[f"lag{L}"] = g.shift(L)
p["rm7"] = g.shift(1).rolling(7).mean().reset_index(0,drop=True)
p["rm28"] = g.shift(1).rolling(28).mean().reset_index(0,drop=True)
p["rs7"] = g.shift(1).rolling(7).std().reset_index(0,drop=True)
p["dow_sin"]=np.sin(2*np.pi*p.dow/7); p["dow_cos"]=np.cos(2*np.pi*p.dow/7)
p["doy_sin"]=np.sin(2*np.pi*p.doy/365.25); p["doy_cos"]=np.cos(2*np.pi*p.doy/365.25)
p["sid"]=p.state_name.astype("category").cat.codes
F = ["lag1","lag2","lag3","lag7","lag14","lag28","rm7","rm28","rs7","dow_sin","dow_cos",
     "doy_sin","doy_cos","cap","n_units","sid"]
def block_lcb(skills_by_block, alpha=0.1):
    """One-sided lower bound on the mean block skill (t-based, blocks assumed independent)."""
    s = np.asarray(skills_by_block, float); n = len(s)
    if n < 3: return -np.inf
    from scipy import stats as st
    return float(s.mean() - st.t.ppf(1 - alpha, n - 1) * s.std(ddof=1) / np.sqrt(n))
rows = []; t0 = time.time()
for h in [1, 2, 3, 7]:
    d = p.copy(); d["y_t"] = d.groupby("state_name")["y"].shift(-h)
    for bname, bcol in [("persistence","y"), ("schedule_shift","sched")]:
        d["B"] = d.groupby("state_name")[bcol].shift(-h) if bcol == "sched" else d[bcol]
        dd = d.dropna(subset=F+["y_t","B"]).copy(); dd["R"] = dd.y_t - dd.B
        tr, va, te = [dd[dd.split==s] for s in ["train","val","test"]]
        if min(len(tr), len(va), len(te)) < 500: continue
        m = HGB(max_iter=400, learning_rate=0.06, max_depth=6, random_state=0).fit(tr[F], tr.R)
        rv, rt = m.predict(va[F]), m.predict(te[F])
        va = va.assign(rhat=rv); te = te.assign(rhat=rt)
        va["blk"] = va.date.dt.to_period("M").astype(str)         # monthly validation blocks
        g_glob = float(np.dot(va.R, rv)/np.dot(rv, rv))
        res = {k: [] for k in ["always","static","risk_controlled","crc_like"]}
        deg = {k: 0 for k in res}; withheld = 0; withheld_would_help = 0; n_cells = 0
        for st_, v in va.groupby("state_name"):
            t_ = te[te.state_name == st_]
            if len(t_) < 30 or len(v) < 60: continue
            n_cells += 1
            gs = float(np.dot(v.R, v.rhat)/np.dot(v.rhat, v.rhat))
            # per-block validation skill of the gated corrector
            blk_sk = []
            for _, vb in v.groupby("blk"):
                e_c = np.abs(vb.R - gs*vb.rhat).mean(); e_b = np.abs(vb.R).mean()
                blk_sk.append(1 - e_c/e_b)
            lcb = block_lcb(blk_sk, alpha=0.1)
            base_e = np.abs(t_.R).mean()
            sk = lambda pred: float(1 - np.abs(t_.y_t - pred).mean()/base_e)
            s_always = sk(t_.B + t_.rhat); s_static = sk(t_.B + gs*t_.rhat)
            s_rc = s_static if lcb > 0 else 0.0
            if lcb <= 0:
                withheld += 1; withheld_would_help += int(s_static > 0)
            # CRC-like: direction gate + quantile clip at validation 90th pct of |rhat|
            tau = float(np.quantile(np.abs(v.rhat), 0.9))
            adj = np.clip(t_.rhat, -tau, tau)
            adj = np.where(np.sign(adj) == np.sign(t_.R.where(t_.R.notna(), 0)), adj, adj*0.0) if False else adj
            s_crc = sk(t_.B + np.clip(gs, 0, 1)*adj)
            for k, val in [("always",s_always),("static",s_static),("risk_controlled",s_rc),("crc_like",s_crc)]:
                res[k].append(val); deg[k] += int(val < 0)
        rows.append(dict(h=h, baseline=bname, n_cells=n_cells, withheld=withheld,
                         withheld_would_help=withheld_would_help,
                         **{f"skill_{k}": float(np.mean(v)) for k, v in res.items()},
                         **{f"degraded_{k}": deg[k] for k in res}))
        r = rows[-1]
        print(f"h={h} {bname:15s} cells={n_cells:3d} | always={r['skill_always']:+.4f}({r['degraded_always']}) "
              f"static={r['skill_static']:+.4f}({r['degraded_static']}) "
              f"RC={r['skill_risk_controlled']:+.4f}({r['degraded_risk_controlled']}) "
              f"CRC={r['skill_crc_like']:+.4f}({r['degraded_crc_like']}) "
              f"| withheld={withheld} of which would-have-helped={withheld_would_help} [{time.time()-t0:.0f}s]", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "R4_risk_control.csv"), index=False)
print("DONE")
