"""RE-FUSED-6 / M1b independent estimator validation.

Question the audit demands: do the gate estimators actually estimate their target quantity?

Controlled construction where the target is known in CLOSED FORM. Let
    R    = m(x) + eta,      eta ~ N(0, s_eta^2)   independent of x
    rhat = m(x) + eps,      eps ~ N(0, s_eps^2)   injected directly (no trained model),
so that
    E[R rhat | x] = m(x)^2,      E[rhat^2 | x] = m(x)^2 + s_eps^2,
    ==> g*(x) = m(x)^2 / (m(x)^2 + s_eps^2)                        (known, per-point)

We then measure, for each estimator: error against g*, bias, scale behaviour, admissible fraction,
variance across seeds, and the realised risk. This isolates estimator quality from every modelling
choice in the real pipeline.

Usage: py -3.10 -u src/refused6/m1b_estimator_validation.py
"""
import os, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

OUT = r"D:\\REFUSED5\results\refused6"
os.makedirs(OUT, exist_ok=True)
CLIP, Z_CLIP, TAU = 1.5, 10.0, 0.10


def make(n, seed, s_eps, s_eta, d_sig=4, d_noise=12):
    rng = np.random.default_rng(seed)
    d = d_sig + d_noise
    X = rng.normal(0, 1, (n, d))
    beta = np.linspace(1.0, 0.4, d_sig)
    core = X[:, :d_sig] @ beta
    m = 0.8 * np.abs(core) / (np.std(core) + 1e-9)      # non-negative, heterogeneous magnitude
    R = m + rng.normal(0, s_eta, n)
    rhat = m + rng.normal(0, s_eps, n)
    g_star = m ** 2 / (m ** 2 + s_eps ** 2)
    return X, R, rhat, m, g_star


def est_raw(Fv, Rv, rv, Ft, seed):
    a = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(Fv, Rv * rv)
    b = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(Fv, rv ** 2)
    return a.predict(Ft) / np.maximum(b.predict(Ft), 1e-8)


def est_ridge(Fv, Rv, rv, Ft, seed):
    a = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(Fv, Rv * rv)
    b = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(Fv, rv ** 2)
    dn = b.predict(Ft)
    return a.predict(Ft) / (dn + TAU * max(float(np.mean(dn)), 1e-12))


def est_wls(Fv, Rv, rv, Ft, seed):
    w = rv ** 2
    ok = w > 1e-12
    z = np.clip(Rv[ok] / rv[ok], -Z_CLIP, Z_CLIP)
    m = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(Fv[ok], z, sample_weight=w[ok])
    return m.predict(Ft)


def est_wls_bag(Fv, Rv, rv, Ft, seed, B=5):
    out = []
    for b in range(B):
        rb = np.random.default_rng(31 * seed + b)
        idx = rb.integers(0, len(Fv), len(Fv))
        out.append(est_wls(Fv[idx], Rv[idx], rv[idx], Ft, seed * 7 + b))
    return np.vstack(out).mean(axis=0)


ESTS = {"raw_ratio": est_raw, "ridge_ratio": est_ridge, "wls": est_wls, "wls_bagged": est_wls_bag}

rows = []
t0 = time.time()
for s_eps in (0.3, 0.7, 1.5):                 # corrector noise: high s_eps => g* small
    for n_val in (1000, 4000):
        for seed in range(8):
            Xv, Rv, rv, mv, gv = make(n_val, seed, s_eps, 1.0)
            Xt, Rt, rt, mt, gt = make(20000, seed + 500, s_eps, 1.0)
            base = float(np.mean(Rt ** 2))
            for name, fn in ESTS.items():
                g = fn(Xv, Rv, rv, Xt, seed)
                gc = np.clip(g, 0, CLIP)
                rows.append(dict(
                    s_eps=s_eps, n_val=n_val, seed=seed, estimator=name,
                    mse_vs_gstar=float(np.mean((gc - gt) ** 2)),
                    bias_vs_gstar=float(np.mean(gc - gt)),
                    corr_vs_gstar=float(np.corrcoef(gc, gt)[0, 1]),
                    frac_inadmissible=float(np.mean((g < 0) | (g > CLIP))),
                    absmax=float(np.max(np.abs(g))),
                    risk=float(np.mean((Rt - gc * rt) ** 2)) / base,
                    risk_oracle_gstar=float(np.mean((Rt - gt * rt) ** 2)) / base,
                    mean_g=float(np.mean(gc)), mean_gstar=float(np.mean(gt)),
                ))
        d = pd.DataFrame(rows).query("s_eps == @s_eps and n_val == @n_val")
        print(f"s_eps={s_eps} n_val={n_val:5d} | " + "  ".join(
            f"{e}: mse={d[d.estimator==e].mse_vs_gstar.mean():.4f} corr={d[d.estimator==e].corr_vs_gstar.mean():+.3f} "
            f"risk={d[d.estimator==e].risk.mean():.4f}" for e in ESTS) +
            f" | oracle_risk={d.risk_oracle_gstar.mean():.4f} mean_gstar={d.mean_gstar.mean():.3f} [{time.time()-t0:.0f}s]",
            flush=True)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "M1b_estimator_validation.csv"), index=False)
print("\n=== estimator quality, averaged over all settings (unit = one seed-setting run) ===")
print(R.groupby("estimator")[["mse_vs_gstar", "bias_vs_gstar", "corr_vs_gstar",
                              "frac_inadmissible", "absmax", "risk"]].mean().round(4).to_string())
print("\noracle (g* itself) mean relative risk: %.4f" % R.risk_oracle_gstar.mean())
best = R.groupby("estimator").risk.mean().idxmin()
print(f"lowest-risk estimator: {best}")
print("DONE", time.time() - t0)
