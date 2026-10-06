"""RE-FUSED-6 / M8: mechanistic cross-domain analysis of the ladder.

RE-FUSED-5 found that the shrunk gate did not transfer to ETT (p=0.80). M4 then showed correction value
collapses as the baseline strengthens. So the M8 question is no longer "why did ETT fail" but the
sharper, testable one:

    Do MEASURABLE, validation-computable properties of a dataset predict the shape of its
    adaptivity ladder - i.e. which rung wins and how much fine gating loses?

Descriptors (all computed WITHOUT test data):
    resid_R2_val      how much of the residual the corrector explains on validation
    gate_var_between  between-series variance of the per-series gate  (heterogeneity signal)
    gate_var_est      bootstrap variance of the per-instance gate     (estimation cost)
    n_val_per_cell    validation points per (series x regime) cell
    resid_acf1        lag-1 autocorrelation of the residual           (dependence)
    base_mae_over_sd  baseline error relative to series scale         (baseline strength)

Outcome variables (test):
    best_rung, fine_minus_coarse = skill(rung5) - skill(best coarse rung)

Datasets: India daily, ETTh1/h2/m1/m2, NYISO hourly - all with the persistence baseline so the
comparison isolates dataset properties rather than baseline choice.

Usage: py -3.10 -u src/refused6/m8_cross_domain.py
"""
import os, time, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1c", os.path.join(HERE, "m1_canonical.py"))
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)
OUT = r"D:\\REFUSED5\results\refused6"
CLIP, N_SEEDS, N_BOOT, MIN_CELL = 1.5, 5, 5, 30
Z_CLIP = 10.0


def wls_gate(F, R, r, seed):
    w = r ** 2
    ok = w > 1e-12
    z = np.clip(R[ok] / r[ok], -Z_CLIP, Z_CLIP)
    m = HGB(max_iter=150, learning_rate=0.08, max_depth=5, random_state=seed)
    m.fit(F[ok], z, sample_weight=w[ok])
    return m


def run(ds, h, seed):
    x = m1.build_target_and_baseline(ds["df"], h, "lag0_self", None, ds["season"])
    F = ds["feats"]
    x = x.dropna(subset=F + ["y_t", "B"])
    tr, va, te = [x[x.split == s] for s in ("train", "val", "test")]
    if min(len(tr), len(va), len(te)) < 500:
        return None
    for f in (tr, va, te):
        f["R"] = f.y_t - f.B
    sub = min(len(tr), 200_000)
    trs = tr.sample(sub, random_state=seed) if len(tr) > sub else tr
    corr = HGB(max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed,
               l2_regularization=1.0).fit(trs[F], trs.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    Rv, Rt = va.R.to_numpy(), te.R.to_numpy()
    g1 = m1.ls_gate(Rv, rv)
    rv, rt = g1 * rv, g1 * rt
    g1e = m1.ls_gate(Rv, rv)
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()
    vcol = "rs7" if "rs7" in F else "rs6"
    lo, hi = np.nanquantile(tr[vcol], [1 / 3, 2 / 3])
    qv = np.digitize(va[vcol].to_numpy(), [lo, hi])
    qt = np.digitize(te[vcol].to_numpy(), [lo, hi])
    g2map = {s: (m1.ls_gate(Rv[sid_v == s], rv[sid_v == s]) if (sid_v == s).sum() >= MIN_CELL else g1e)
             for s in np.unique(sid_v)}
    g2 = np.array([g2map.get(s, g1e) for s in sid_t])
    g3map = {}
    for s in np.unique(sid_v):
        for q in range(3):
            k = (sid_v == s) & (qv == q)
            g3map[(s, q)] = m1.ls_gate(Rv[k], rv[k]) if k.sum() >= MIN_CELL else g2map.get(s, g1e)
    g3 = np.array([g3map.get((s, q), g1e) for s, q in zip(sid_t, qt)])
    Fv, Ft = va[F].to_numpy(), te[F].to_numpy()
    bw = np.empty((N_BOOT, len(Ft)))
    for b in range(N_BOOT):
        rb = np.random.default_rng(1200 + 37 * seed + b)
        idx = rb.integers(0, len(Fv), len(Fv))
        bw[b] = wls_gate(Fv[idx], Rv[idx], rv[idx], seed * 43 + b).predict(Ft)
    g5 = bw.mean(axis=0)
    V, W = float(np.mean(bw.var(axis=0, ddof=1))), float(np.var(g5))
    lam = W / (W + V) if (W + V) > 0 else 0.0
    gates = {"rung0_none": np.zeros(len(te)), "rung1_global": np.full(len(te), g1e),
             "rung2_series": g2, "rung3_cell": g3, "rung5_instance": g5,
             "rung4_shrunk": lam * g5 + (1 - lam) * g2}
    base = Rt ** 2
    sk = {}
    for rung, g in gates.items():
        gc = np.clip(g, 0, CLIP)
        sq = (Rt - gc * rt) ** 2
        sk[rung] = float(np.mean([1 - sq[sid_t == s].mean() / base[sid_t == s].mean()
                                  for s in np.unique(sid_t)]))
    # ---- validation-only descriptors ----
    acf1 = float(pd.Series(Rv).autocorr(lag=1))
    ncell = float(np.mean([((sid_v == s) & (qv == q)).sum() for s in np.unique(sid_v) for q in range(3)]))
    row = dict(dataset=ds["name"], h=h, seed=seed,
               resid_R2_val=float(1 - np.mean((Rv - rv) ** 2) / np.var(Rv)),
               gate_var_between=float(np.var(list(g2map.values()))),
               gate_var_est=V, n_val_per_cell=ncell, resid_acf1=acf1,
               base_mae_over_sd=float(np.abs(Rv).mean() / (va.y_t.std() + 1e-9)),
               n_val=len(va), n_test=len(te), n_series=int(len(np.unique(sid_v))), lam=lam,
               **{f"skill_{k}": v for k, v in sk.items()})
    coarse = max(sk["rung1_global"], sk["rung2_series"], sk["rung3_cell"])
    row["best_rung"] = max(sk, key=sk.get)
    row["fine_minus_coarse"] = sk["rung5_instance"] - coarse
    row["best_coarse_skill"] = coarse
    return row


def main():
    datasets = [m1.load_india()] + m1.load_ett()
    nn = m1.load_nyiso_1h()
    if nn is not None:
        datasets.append(nn)
    rows, t0 = [], time.time()
    for ds in datasets:
        hs = [1, 3] if ds["name"] == "india_daily" else ([1, 24] if ds["name"] != "nyiso_1h" else [1, 24])
        for h in hs:
            got = False
            for seed in range(N_SEEDS):
                r = run(ds, h, seed)
                if r is None:
                    break
                rows.append(r); got = True
            if got:
                d = pd.DataFrame(rows).query("dataset == @ds['name'] and h == @h")
                print(f"{ds['name']:12s} h={h:2d} | residR2_val={d.resid_R2_val.mean():+.3f} "
                      f"acf1={d.resid_acf1.mean():+.3f} varB={d.gate_var_between.mean():.4f} "
                      f"varE={d.gate_var_est.mean():.4f} n/cell={d.n_val_per_cell.mean():.0f} | " +
                      " ".join(f"{k.replace('skill_rung','r')[:4]}={d[k].mean():+.4f}" for k in
                               ["skill_rung1_global", "skill_rung2_series", "skill_rung3_cell",
                                "skill_rung4_shrunk", "skill_rung5_instance"]) +
                      f" | best={d.best_rung.mode().iloc[0]} fine-coarse={d.fine_minus_coarse.mean():+.4f}"
                      f" [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "M8_cross_domain.csv"), index=False)
    print(f"\nwrote {len(R)} rows over {R.dataset.nunique()} datasets")
    print("\n=== best rung by dataset ===")
    print(R.groupby(["dataset", "h"]).best_rung.agg(lambda s: s.mode().iloc[0]).to_string())
    print("\n=== do validation-only descriptors predict the fine-rung gap? (unit = config mean) ===")
    cfg = R.groupby(["dataset", "h"]).mean(numeric_only=True).reset_index()
    for d in ["resid_R2_val", "gate_var_between", "gate_var_est", "n_val_per_cell",
              "resid_acf1", "base_mae_over_sd", "lam"]:
        if cfg[d].std() > 0:
            sp = stats.spearmanr(cfg[d], cfg.fine_minus_coarse)
            pe = stats.pearsonr(cfg[d], cfg.fine_minus_coarse)
            print(f"  {d:20s} spearman={sp[0]:+.3f} (p={sp[1]:.3f})   pearson={pe[0]:+.3f} (p={pe[1]:.3f})")
    print("\n=== config-level table ===")
    print(cfg[["dataset", "h", "resid_R2_val", "resid_acf1", "gate_var_between", "gate_var_est",
               "n_val_per_cell", "best_coarse_skill", "fine_minus_coarse"]].round(4).to_string(index=False))
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
