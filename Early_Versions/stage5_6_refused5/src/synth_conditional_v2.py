"""RE-FUSED-5 S2b: when (if ever) does PER-INSTANCE gating beat STATIC PER-CELL gating?
Case A: predictability varies only across (regime x horizon) cells  -> static cell gate should suffice.
Case B: predictability also varies WITHIN a cell with an observable state v -> per-instance should win.
Metric: per-cell skill = 1 - MSE/MSE_never, averaged over cells (scale-free), + degradation count."""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
OUT = r"D:\\REFUSED5\results"; os.makedirs(OUT, exist_ok=True)
HOR = [1, 3, 6, 12, 24]; REG = [0, 1, 2]
SIG = {0: 0.15, 1: 0.60, 2: 1.50}        # signal sd by regime (noise sd fixed = 1)

def simulate(n, seed, case, d_sig=5, d_noise=25):
    rng = np.random.default_rng(seed)
    reg = rng.integers(0, 3, n); h = np.array(rng.choice(HOR, n))
    X = rng.normal(0, 1, (n, d_sig + d_noise))
    v = rng.uniform(0, 1, n)                                  # observable state (e.g. local volatility)
    beta = rng.normal(0, 1, d_sig); core = X[:, :d_sig] @ beta; core /= core.std()
    decay = 1.0 / (1.0 + 0.25 * (h - 1))
    amp = np.array([SIG[r] for r in reg]) * decay
    if case == "B":                                           # signal also modulated within cell by v
        amp = amp * (0.2 + 1.6 * v)
    m = core * amp
    R = m + rng.normal(0, 1.0, n)                             # homoscedastic noise
    F = np.column_stack([X, reg, h, v])
    return F, R, m, reg, h, v

def run(seed, case):
    Ftr, Rtr, *_ = simulate(12000, seed, case)
    Fva, Rva, mva, rva, hva, vva = simulate(8000, seed + 500, case)
    Fte, Rte, mte, rte, hte, vte = simulate(30000, seed + 999, case)
    corr = Ridge(alpha=5.0).fit(Ftr, Rtr)
    rv, rt = corr.predict(Fva), corr.predict(Fte)
    g_glob = float(np.dot(Rva, rv) / np.dot(rv, rv))
    cell = {}
    for r_ in REG:
        for h_ in HOR:
            k = (rva == r_) & (hva == h_)
            cell[(r_, h_)] = float(np.dot(Rva[k], rv[k]) / np.dot(rv[k], rv[k])) if k.sum() > 20 else g_glob
    g_static = np.array([cell[(r_, h_)] for r_, h_ in zip(rte, hte)])
    num = HGB(max_iter=300, learning_rate=0.06, max_depth=6).fit(Fva, Rva * rv)
    den = HGB(max_iter=300, learning_rate=0.06, max_depth=6).fit(Fva, rv ** 2)
    g_cvc = np.clip(num.predict(Fte) / np.maximum(den.predict(Fte), 1e-8), 0.0, 1.0)
    g_orc = np.clip(np.array([np.dot(Rte[(rte==r_)&(hte==h_)], rt[(rte==r_)&(hte==h_)]) /
                              np.dot(rt[(rte==r_)&(hte==h_)], rt[(rte==r_)&(hte==h_)])
                              for r_, h_ in zip(rte, hte)]), 0, 1) if False else None
    preds = {"never": np.zeros_like(rt), "always": rt, "global": g_glob * rt,
             "static_cell": g_static * rt, "cvc": g_cvc * rt}
    out = {}
    for name, p in preds.items():
        sk, deg = [], 0
        for r_ in REG:
            for h_ in HOR:
                k = (rte == r_) & (hte == h_)
                mse_n = np.mean(Rte[k] ** 2); mse_m = np.mean((Rte[k] - p[k]) ** 2)
                sk.append(1 - mse_m / mse_n); deg += int(mse_m > mse_n)
        out[name] = dict(skill=float(np.mean(sk)), degraded_cells=deg,
                         mse=float(np.mean((Rte - p) ** 2)))
    # calibration of predicted benefit (CVC)
    pb = 2 * g_cvc * num.predict(Fte) - g_cvc ** 2 * den.predict(Fte)
    rb = Rte ** 2 - (Rte - g_cvc * rt) ** 2
    q = np.quantile(pb, np.linspace(0, 1, 11)); pts = []
    for i in range(10):
        k = (pb >= q[i]) & (pb <= q[i + 1])
        if k.sum() > 50: pts.append((pb[k].mean(), rb[k].mean()))
    A = np.array([[p_, 1] for p_, _ in pts]); y = np.array([r_ for _, r_ in pts])
    sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - ((y - A @ [sl, ic]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out["_cal"] = dict(slope=float(sl), r2=float(r2), gate_sd=float(np.std(g_cvc)))
    return out

for case in ["A", "B"]:
    runs = [run(s, case) for s in range(10)]
    print(f"\n===== CASE {case}: " + ("across-cell heterogeneity only" if case == "A"
          else "plus within-cell heterogeneity (observable state v)"))
    print(f"{'strategy':>12} | {'mean cell skill':>15} {'sd':>7} | {'degraded cells/15':>18} | pooled MSE")
    for k in ["never", "always", "global", "static_cell", "cvc"]:
        s = np.array([r[k]["skill"] for r in runs]); d = np.mean([r[k]["degraded_cells"] for r in runs])
        m = np.mean([r[k]["mse"] for r in runs])
        print(f"{k:>12} | {s.mean():15.4f} {s.std():7.4f} | {d:18.1f} | {m:.4f}")
    cv = np.array([r["cvc"]["skill"] for r in runs]); st = np.array([r["static_cell"]["skill"] for r in runs])
    diff = cv - st
    print(f"  CVC - static_cell skill: {diff.mean():+.5f} (sd {diff.std():.5f}); CVC better {int((diff>0).sum())}/10 seeds")
    print(f"  calibration slope={np.mean([r['_cal']['slope'] for r in runs]):.3f} "
          f"R2={np.mean([r['_cal']['r2'] for r in runs]):.3f} "
          f"gate sd={np.mean([r['_cal']['gate_sd'] for r in runs]):.3f}")
    json.dump(runs, open(os.path.join(OUT, f"S2b_case{case}.json"), "w"), indent=1, default=float)
