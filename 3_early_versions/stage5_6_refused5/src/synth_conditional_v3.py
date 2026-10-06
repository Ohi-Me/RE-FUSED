"""RE-FUSED-5 S2c (fixed): ONE data-generating process per seed, sampled into train/val/test.
Case A: predictability varies across (regime x horizon) cells only.
Case B: also varies within a cell with an observable state v.
Strategies: never | always | global gate | static per-cell gate (CRC-style) | CVC per-instance gate."""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
OUT = r"D:\\REFUSED5\results"; os.makedirs(OUT, exist_ok=True)
HOR = [1, 3, 6, 12, 24]; REG = [0, 1, 2]; SIG = {0: 0.15, 1: 0.60, 2: 1.50}

class Process:
    """Fixed DGP: shared beta across all splits (this was the bug in S2b)."""
    def __init__(self, seed, case, d_sig=5, d_noise=25):
        r = np.random.default_rng(seed)
        self.beta = r.normal(0, 1, d_sig); self.d_sig, self.d_noise = d_sig, d_noise
        self.case = case; self.scale = None
    def sample(self, n, rng):
        X = rng.normal(0, 1, (n, self.d_sig + self.d_noise))
        reg = rng.integers(0, 3, n); h = np.array(rng.choice(HOR, n)); v = rng.uniform(0, 1, n)
        core = X[:, :self.d_sig] @ self.beta
        if self.scale is None: self.scale = core.std()
        core = core / self.scale
        amp = np.array([SIG[r_] for r_ in reg]) / (1.0 + 0.25 * (h - 1))
        if self.case == "B": amp = amp * (0.2 + 1.6 * v)
        m = core * amp
        R = m + rng.normal(0, 1.0, n)
        return np.column_stack([X, reg, h, v]), R, m, reg, h, v

def run(seed, case):
    P = Process(seed, case); rng = np.random.default_rng(10_000 + seed)
    Ftr, Rtr, mtr, *_ = P.sample(12000, rng)
    Fva, Rva, mva, rva, hva, vva = P.sample(8000, rng)
    Fte, Rte, mte, rte, hte, vte = P.sample(30000, rng)
    corr = HGB(max_iter=300, learning_rate=0.06, max_depth=6).fit(Ftr, Rtr)
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
    preds = {"never": np.zeros_like(rt), "always": rt, "global": g_glob * rt,
             "static_cell": g_static * rt, "cvc": g_cvc * rt}
    out = {}
    for name, p in preds.items():
        sk, deg = [], 0
        for r_ in REG:
            for h_ in HOR:
                k = (rte == r_) & (hte == h_)
                n_, m_ = np.mean(Rte[k] ** 2), np.mean((Rte[k] - p[k]) ** 2)
                sk.append(1 - m_ / n_); deg += int(m_ > n_)
        out[name] = dict(skill=float(np.mean(sk)), degraded=deg, mse=float(np.mean((Rte - p) ** 2)))
    pb = 2 * g_cvc * num.predict(Fte) - g_cvc ** 2 * den.predict(Fte)
    rb = Rte ** 2 - (Rte - g_cvc * rt) ** 2
    q = np.quantile(pb, np.linspace(0, 1, 11)); pts = []
    for i in range(10):
        k = (pb >= q[i]) & (pb <= q[i + 1])
        if k.sum() > 50: pts.append((pb[k].mean(), rb[k].mean()))
    A = np.array([[a, 1] for a, _ in pts]); y = np.array([b for _, b in pts])
    sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - ((y - A @ [sl, ic]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # oracle per-instance gate (uses true m): g = E[R rhat|x]/E[rhat^2|x] with R->m
    g_orc = np.clip((mte * rt) / np.maximum(rt ** 2, 1e-8), 0, 1)
    out["_diag"] = dict(cal_slope=float(sl), cal_r2=float(r2), gate_sd=float(np.std(g_cvc)),
                        gate_mae_vs_oracle=float(np.mean(np.abs(g_cvc - g_orc))),
                        corr_r2_resid=float(1 - np.mean((Rte - rt) ** 2) / np.mean(Rte ** 2)))
    return out

for case in ["A", "B"]:
    runs = [run(s, case) for s in range(10)]
    print(f"\n===== CASE {case}: " + ("across-cell only" if case == "A" else "+ within-cell state v"))
    print(f"{'strategy':>12} | {'cell skill':>11} {'sd':>7} | {'degraded/15':>11} | pooled MSE")
    for k in ["never", "always", "global", "static_cell", "cvc"]:
        s = np.array([r[k]["skill"] for r in runs])
        print(f"{k:>12} | {s.mean():11.4f} {s.std():7.4f} | "
              f"{np.mean([r[k]['degraded'] for r in runs]):11.1f} | {np.mean([r[k]['mse'] for r in runs]):.4f}")
    cv = np.array([r["cvc"]["skill"] for r in runs]); st = np.array([r["static_cell"]["skill"] for r in runs])
    gl = np.array([r["global"]["skill"] for r in runs]); d1, d2 = cv - st, cv - gl
    print(f"  CVC-static: {d1.mean():+.5f} (sd {d1.std():.5f}) better {int((d1>0).sum())}/10 | "
          f"CVC-global: {d2.mean():+.5f} better {int((d2>0).sum())}/10")
    print(f"  calib slope={np.mean([r['_diag']['cal_slope'] for r in runs]):.3f} "
          f"R2={np.mean([r['_diag']['cal_r2'] for r in runs]):.3f} | "
          f"gate MAE vs oracle={np.mean([r['_diag']['gate_mae_vs_oracle'] for r in runs]):.3f} | "
          f"corrector residual-R2={np.mean([r['_diag']['corr_r2_resid'] for r in runs]):.3f}")
    json.dump(runs, open(os.path.join(OUT, f"S2c_case{case}.json"), "w"), indent=1, default=float)
