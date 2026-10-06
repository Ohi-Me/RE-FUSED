"""RE-FUSED-5 R5: 10-seed core comparison with statistical tests (India daily panel).
Strategies: never | always | global | static-state | shrunk, for 3 stationary baselines x h in {1,2,3,7}.
Reports mean/sd/CI over seeds, paired t-test and Diebold-Mariano (HLN) vs the un-corrected baseline."""
import pandas as pd, numpy as np, os, json, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from scipy import stats
D = r"D:\\REFUSED5\data"; OUT = r"D:\\REFUSED5\results"
p = pd.read_parquet(os.path.join(D, "india_model_daily.parquet")).sort_values(["state_name","date"])
g = p.groupby("state_name")["y"]
for L in [1,2,3,7,14,28]: p[f"lag{L}"] = g.shift(L)
p["rm7"]=g.shift(1).rolling(7).mean().reset_index(0,drop=True)
p["rm28"]=g.shift(1).rolling(28).mean().reset_index(0,drop=True)
p["rs7"]=g.shift(1).rolling(7).std().reset_index(0,drop=True)
p["dow_sin"]=np.sin(2*np.pi*p.dow/7); p["dow_cos"]=np.cos(2*np.pi*p.dow/7)
p["doy_sin"]=np.sin(2*np.pi*p.doy/365.25); p["doy_cos"]=np.cos(2*np.pi*p.doy/365.25)
p["sid"]=p.state_name.astype("category").cat.codes
F=["lag1","lag2","lag3","lag7","lag14","lag28","rm7","rm28","rs7","dow_sin","dow_cos",
   "doy_sin","doy_cos","cap","n_units","sid"]
def dm_hln(e1, e2, h):
    """Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction, absolute-error loss."""
    d = np.abs(e1) - np.abs(e2); n = len(d)
    gamma = [np.mean((d[:n-k]-d.mean())*(d[k:]-d.mean())) for k in range(h)]
    var = (gamma[0] + 2*sum(gamma[1:])) / n
    if var <= 0: return np.nan, np.nan
    stat = d.mean()/np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2*h + h*(h-1)/n)/n)
    return float(stat), float(2*(1 - stats.t.cdf(abs(stat), n-1)))
rows=[]; per_seed_rows=[]; t0=time.time()
for h in [1,2,3,7]:
    d = p.copy(); d["y_t"]=d.groupby("state_name")["y"].shift(-h)
    for bname, bser in [("persistence", d["y"]), ("roll7", d["rm7"]),
                        ("snaive7", d.groupby("state_name")["y"].shift(-h+7) if h<=7 else None)]:
        if bser is None: continue
        d["B"]=bser; dd=d.dropna(subset=F+["y_t","B"]).copy(); dd["R"]=dd.y_t-dd.B
        tr,va,te=[dd[dd.split==s] for s in ["train","val","test"]]
        if min(len(tr),len(va),len(te))<500: continue
        per_seed={k:[] for k in ["always","global","static","shrunk"]}
        base_err=np.abs(te.R.values)
        for seed in range(10):
            m=HGB(max_iter=400, learning_rate=0.06, max_depth=6, random_state=seed,
                  l2_regularization=1.0).fit(tr[F], tr.R)
            rv,rt=m.predict(va[F]), m.predict(te[F])
            gg=float(np.dot(va.R,rv)/np.dot(rv,rv))
            cell={st_: (float(np.dot(va.R.values[i],rv[i])/np.dot(rv[i],rv[i])) if len(i)>30 else gg)
                  for st_,i in va.groupby("state_name").indices.items()}
            gs=te.state_name.map(cell).fillna(gg).values
            num=HGB(max_iter=200,learning_rate=0.07,max_depth=5,random_state=seed).fit(va[F], va.R*rv)
            den=HGB(max_iter=200,learning_rate=0.07,max_depth=5,random_state=seed).fit(va[F], rv**2)
            gf=np.clip(num.predict(te[F])/np.maximum(den.predict(te[F]),1e-8),0,1)
            rng=np.random.default_rng(seed); idx=rng.permutation(len(va)); a_,b_=idx[::2],idx[1::2]
            gA=np.clip(HGB(max_iter=150,learning_rate=0.07,max_depth=5,random_state=seed+7).fit(va[F].iloc[a_],(va.R.values*rv)[a_]).predict(te[F])/
                       np.maximum(HGB(max_iter=150,learning_rate=0.07,max_depth=5,random_state=seed+7).fit(va[F].iloc[a_],(rv**2)[a_]).predict(te[F]),1e-8),0,1)
            gB=np.clip(HGB(max_iter=150,learning_rate=0.07,max_depth=5,random_state=seed+9).fit(va[F].iloc[b_],(va.R.values*rv)[b_]).predict(te[F])/
                       np.maximum(HGB(max_iter=150,learning_rate=0.07,max_depth=5,random_state=seed+9).fit(va[F].iloc[b_],(rv**2)[b_]).predict(te[F]),1e-8),0,1)
            ve=float(np.mean((gA-gB)**2)/4.0); vw=max(float(np.var(gf))-ve,0.0); lam=vw/(vw+ve+1e-12)
            gsh=np.clip(lam*gf+(1-lam)*gs,0,1)
            for k,gv in [("always",1.0),("global",gg),("static",gs),("shrunk",gsh)]:
                err=np.abs(te.y_t.values-(te.B.values+gv*rt))
                per_seed[k].append(float(1-err.mean()/base_err.mean()))
            if seed==0:
                e_shrunk=te.y_t.values-(te.B.values+gsh*rt); e_base=te.R.values
                dm,pv=dm_hln(e_base,e_shrunk,h)
        for k, vals in per_seed.items():
            for sd_, val in enumerate(vals):
                per_seed_rows.append(dict(h=h, baseline=bname, strategy=k, seed=sd_, skill=val))
        for k,v in per_seed.items():
            v=np.array(v); ci=stats.t.interval(0.95,len(v)-1,loc=v.mean(),scale=stats.sem(v)) if v.std()>0 else (v.mean(),v.mean())
            rows.append(dict(h=h,baseline=bname,strategy=k,skill_mean=float(v.mean()),skill_sd=float(v.std()),
                             ci_lo=float(ci[0]),ci_hi=float(ci[1]),best_seed=float(v.max()),worst_seed=float(v.min()),
                             dm_stat=(dm if k=="shrunk" else np.nan), dm_p=(pv if k=="shrunk" else np.nan)))
        print(f"h={h} {bname:11s} " + " ".join(f"{k}={np.mean(per_seed[k]):+.4f}+-{np.std(per_seed[k]):.4f}"
              for k in ["always","global","static","shrunk"]) + f" | DM(shrunk vs base) p={pv:.4f} [{time.time()-t0:.0f}s]", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "R5_multiseed.csv"), index=False)
ps = pd.DataFrame(per_seed_rows)
ps.to_csv(os.path.join(OUT, "R5_per_seed.csv"), index=False)
print(chr(10) + "=== paired tests across seeds (shrunk vs static) ===")
for (h_, b_), grp in ps.groupby(["h", "baseline"]):
    a = grp[grp.strategy == "shrunk"].sort_values("seed").skill.values
    c = grp[grp.strategy == "static"].sort_values("seed").skill.values
    if len(a) == len(c) and len(a) > 2:
        t_ = stats.ttest_rel(a, c)
        wp = stats.wilcoxon(a, c).pvalue if len(a) > 5 else float("nan")
        print("h=%d %-11s shrunk-static=%+.5f wins=%d/%d t=%+.2f p=%.4f wilcoxon_p=%.4f" %
              (h_, b_, float((a - c).mean()), int((a > c).sum()), len(a),
               float(t_.statistic), float(t_.pvalue), wp))
print("DONE")
