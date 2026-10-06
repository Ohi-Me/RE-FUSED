"""RE-FUSED-6 / M1b: numerically stable per-instance gate estimators (fixes newly found defect D8).

D8: the ratio estimator g(x) = num(x)/den(x) explodes when den -> 0. In M1 the measured gate
variance was 1e18-1e19 and 40% of per-instance gate values fell outside [0, 1.5] before clipping.
Every conclusion about FINE rungs in RE-FUSED-5 and M1 therefore confounds "the price of adaptivity"
with "a broken estimator".

Principled fix. Because g*(x) = E[R r_hat | x] / E[r_hat^2 | x] is exactly the weighted least-squares
projection of the raw gate z = R / r_hat onto x with weights w = r_hat^2, we can fit that regression
directly. The weights damp the same small-r_hat observations that make the ratio blow up, so no
clipping is needed to obtain a finite estimate.

Rungs added here (canonical protocol otherwise identical to M1):
  rung5_wls          weighted least squares gate (the principled estimator)
  rung5_ridge        ridge-damped ratio  num / (den + tau * mean(den))
  rung4_wls_series   lambda * rung5_wls + (1 - lambda) * per-series      <- RE-FUSED-5's shrink target
  rung4_wls_cell     lambda * rung5_wls + (1 - lambda) * per-series x regime
plus the M1 reference rungs so the comparison is within-run and paired.

Usage: py -3.10 -u src/refused6/m1b_stable_gates.py
"""
import os, sys, json, time, importlib.util, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1c", os.path.join(HERE, "m1_canonical.py"))
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)

OUT = r"D:\\REFUSED5\results\refused6"
CLIP, N_SEEDS, N_BOOT, MIN_CELL = m1.CLIP, m1.N_SEEDS, m1.N_BOOT, m1.MIN_CELL
Z_CLIP = 10.0      # raw-gate clip inside the WLS target (a modelling choice, logged)
TAU = 0.10         # ridge damping as a fraction of mean(den)


def fit_wls_gate(F, R, r, seed):
    """g(x) by weighted least squares: regress z = R/r on x with weights r^2."""
    w = r ** 2
    safe = w > 1e-12
    z = np.zeros_like(R)
    z[safe] = np.clip(R[safe] / r[safe], -Z_CLIP, Z_CLIP)
    m = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed)
    m.fit(F[safe], z[safe], sample_weight=w[safe])
    return m


def fit_ratio_models(F, R, r, seed):
    a = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(F, R * r)
    b = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(F, r ** 2)
    return a, b


def run_config(ds, bname, h, seed):
    kind, arg = ds["baselines"][bname]
    x = m1.build_target_and_baseline(ds["df"], h, kind, arg, ds["season"])
    if x is None:
        return None
    F = ds["feats"]
    x = x.dropna(subset=F + ["y_t", "B"])
    tr, va, te = [x[x.split == s] for s in ("train", "val", "test")]
    if min(len(tr), len(va), len(te)) < 400:
        return None
    for f in (tr, va, te):
        f["R"] = f.y_t - f.B
    sub = min(len(tr), 300_000)
    trs = tr.sample(sub, random_state=seed) if len(tr) > sub else tr
    corr = HGB(max_iter=400, learning_rate=0.06, max_depth=6, random_state=seed,
               l2_regularization=1.0).fit(trs[F], trs.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    Rv, Rt = va.R.to_numpy(), te.R.to_numpy()
    Fva, Fte = va[F].to_numpy(), te[F].to_numpy()

    vcol = "rs7" if "rs7" in F else "rs6"
    lo, hi = np.nanquantile(tr[vcol], [1 / 3, 2 / 3])
    reg_v, reg_t = m1.regime_labels(va, lo, hi), m1.regime_labels(te, lo, hi)
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()

    g1 = m1.ls_gate(Rv, rv)
    g2map = {s: (m1.ls_gate(Rv[sid_v == s], rv[sid_v == s]) if (sid_v == s).sum() >= MIN_CELL else g1)
             for s in np.unique(sid_v)}
    g2 = np.array([g2map.get(s, g1) for s in sid_t])
    g3map = {}
    for s in np.unique(sid_v):
        for r_ in range(m1.N_REGIMES):
            k = (sid_v == s) & (reg_v == r_)
            g3map[(s, r_)] = m1.ls_gate(Rv[k], rv[k]) if k.sum() >= MIN_CELL else g2map.get(s, g1)
    g3 = np.array([g3map.get((s, r_), g1) for s, r_ in zip(sid_t, reg_t)])

    # ---- fine estimators ----
    wls = fit_wls_gate(Fva, Rv, rv, seed)
    g5w = wls.predict(Fte)
    num, den = fit_ratio_models(Fva, Rv, rv, seed)
    dn = den.predict(Fte)
    g5r = num.predict(Fte) / (dn + TAU * max(float(np.mean(dn)), 1e-12))
    g5raw = num.predict(Fte) / np.maximum(dn, 1e-8)          # the M1 (unstable) form, for contrast

    # bootstrap variance of the STABLE estimator
    bw = np.empty((N_BOOT, len(Fte)))
    for b in range(N_BOOT):
        rb = np.random.default_rng(4400 + 97 * seed + b)
        idx = rb.integers(0, len(Fva), len(Fva))
        bw[b] = fit_wls_gate(Fva[idx], Rv[idx], rv[idx], seed * 17 + b).predict(Fte)
    g5w_bag = bw.mean(axis=0)
    V = float(np.mean(bw.var(axis=0, ddof=1)))
    W = float(np.var(g5w_bag))
    lam = W / (W + V) if (W + V) > 0 else 0.0

    gates = {
        "rung1_global": np.full(len(te), g1), "rung2_series": g2, "rung3_cell": g3,
        "rung5_raw_ratio": g5raw, "rung5_ridge": g5r, "rung5_wls": g5w, "rung5_wls_bagged": g5w_bag,
        "rung4_wls_series": lam * g5w_bag + (1 - lam) * g2,
        "rung4_wls_cell": lam * g5w_bag + (1 - lam) * g3,
        "rung0_none": np.zeros(len(te)),
    }
    base_sq, base_ab = Rt ** 2, np.abs(Rt)
    rows = []
    for rung, g in gates.items():
        frac = float(np.mean((g < 0) | (g > CLIP)))
        gc = np.clip(g, 0.0, CLIP)
        err = Rt - gc * rt
        sq, ab = err ** 2, np.abs(err)
        msq, mab = [], []
        for s in np.unique(sid_t):
            k = sid_t == s
            msq.append(1 - sq[k].mean() / base_sq[k].mean())
            mab.append(1 - ab[k].mean() / base_ab[k].mean())
        rows.append(dict(dataset=ds["name"], baseline=bname, h=h, seed=seed, rung=rung,
                         skill_sq_macro=float(np.mean(msq)), skill_sq_pooled=float(1 - sq.mean() / base_sq.mean()),
                         skill_mae_macro=float(np.mean(mab)), skill_mae_pooled=float(1 - ab.mean() / base_ab.mean()),
                         degraded_series=int(sum(1 for v in msq if v < 0)), n_series=len(msq),
                         frac_clipped=frac, gate_absmax=float(np.max(np.abs(g))),
                         n_test=int(len(te)), n_val=int(len(va)),
                         var_est_stable=V, var_within_stable=W, lam_stable=lam,
                         rho_stable=(W / V if V > 0 else np.nan)))
    return rows


def main():
    ds = m1.load_india()
    rows, t0 = [], time.time()
    for bname in ds["baselines"]:
        for h in ds["horizons"]:
            ok = False
            for seed in range(N_SEEDS):
                r = run_config(ds, bname, h, seed)
                if r is None:
                    break
                rows += r; ok = True
            if ok:
                d = pd.DataFrame(rows).query("baseline == @bname and h == @h")
                mm = d.groupby("rung").skill_sq_macro.mean()
                print(f"{bname:18s} h={h:2d} | " + " ".join(
                    f"{k.replace('rung','r')}={mm.get(k, float('nan')):+.4f}" for k in
                    ["rung1_global", "rung2_series", "rung3_cell", "rung4_wls_series",
                     "rung4_wls_cell", "rung5_wls", "rung5_ridge", "rung5_raw_ratio"]) +
                    f" | rho_stable={d.rho_stable.mean():.2f} lam={d.lam_stable.mean():.2f}"
                    f" clip_wls={d[d.rung=='rung5_wls'].frac_clipped.mean():.3f}"
                    f" clip_raw={d[d.rung=='rung5_raw_ratio'].frac_clipped.mean():.3f} [{time.time()-t0:.0f}s]",
                    flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "M1b_runs_india.csv"), index=False)
    json.dump(dict(z_clip=Z_CLIP, tau=TAU, n_seeds=N_SEEDS, clip=CLIP,
                   note="D8 fix: weighted-least-squares gate; ratio form retained for contrast"),
              open(os.path.join(OUT, "M1b_definitions.json"), "w"), indent=2)
    print(f"\nwrote {len(R)} rows")
    print("\n=== variance diagnostics: stable vs M1 unstable ===")
    print(R.groupby("rung")[["gate_absmax", "frac_clipped"]].mean().round(4).to_string())
    print("\nrho_stable range: %.3f .. %.3f" % (R.rho_stable.min(), R.rho_stable.max()))
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
