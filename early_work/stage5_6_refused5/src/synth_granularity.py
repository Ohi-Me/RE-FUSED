"""RE-FUSED-5 S3 (core): the GATE GRANULARITY LAW.
Hypothesis: conditioning the gate pays iff Var_within(g*) > Var(ghat_est).
Strategies: static per-cell | free per-instance (CVC) | SHRUNK hierarchical gate | oracle per-instance.
Sweep: within-cell gate variation amplitude x validation size. 10 seeds."""
import numpy as np, json, os, warnings, itertools, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
OUT = r"D:\\REFUSED5\results"; os.makedirs(OUT, exist_ok=True)
HOR = [1, 3, 6, 12, 24]; REG = [0, 1, 2]; SIG = {0: 0.15, 1: 0.60, 2: 1.50}

class Process:
    def __init__(self, seed, amp_within, d_sig=5, d_noise=25):
        r = np.random.default_rng(seed)
        self.beta = r.normal(0, 1, d_sig); self.d_sig, self.d_noise = d_sig, d_noise
        self.a = amp_within; self.scale = None
    def sample(self, n, rng):
        X = rng.normal(0, 1, (n, self.d_sig + self.d_noise))
        reg = rng.integers(0, 3, n); h = np.array(rng.choice(HOR, n)); v = rng.uniform(0, 1, n)
        core = X[:, :self.d_sig] @ self.beta
        if self.scale is None: self.scale = core.std()
        core = core / self.scale
        amp = np.array([SIG[r_] for r_ in reg]) / (1.0 + 0.25 * (h - 1))
        amp = amp * (1.0 + self.a * (v - 0.5) * 2.0).clip(0.05, None)   # within-cell modulation
        m = core * amp
        return np.column_stack([X, reg, h, v]), m + rng.normal(0, 1, n), m, reg, h, v

def gate_models(F, R, rhat, seed):
    num = HGB(max_iter=220, learning_rate=0.07, max_depth=6, random_state=seed).fit(F, R * rhat)
    den = HGB(max_iter=220, learning_rate=0.07, max_depth=6, random_state=seed).fit(F, rhat ** 2)
    return num, den

def run(seed, amp, n_val):
    P = Process(seed, amp); rng = np.random.default_rng(10_000 + seed)
    Ftr, Rtr, *_ = P.sample(12000, rng)
    Fva, Rva, mva, rva, hva, vva = P.sample(n_val, rng)
    Fte, Rte, mte, rte, hte, vte = P.sample(30000, rng)
    corr = HGB(max_iter=250, learning_rate=0.07, max_depth=6, random_state=seed).fit(Ftr, Rtr)
    rv, rt = corr.predict(Fva), corr.predict(Fte)
    g_glob = float(np.dot(Rva, rv) / np.dot(rv, rv))
    cell = {}
    for r_ in REG:
        for h_ in HOR:
            k = (rva == r_) & (hva == h_)
            cell[(r_, h_)] = float(np.dot(Rva[k], rv[k]) / np.dot(rv[k], rv[k])) if k.sum() > 20 else g_glob
    g_static = np.array([cell[(r_, h_)] for r_, h_ in zip(rte, hte)])
    num, den = gate_models(Fva, Rva, rv, seed)
    g_free = np.clip(num.predict(Fte) / np.maximum(den.predict(Fte), 1e-8), 0, 1)
    # split-half estimate of gate estimation variance (the cost of conditioning)
    idx = rng.permutation(len(Fva)); a_, b_ = idx[::2], idx[1::2]
    nA, dA = gate_models(Fva[a_], Rva[a_], rv[a_], seed + 1)
    nB, dB = gate_models(Fva[b_], Rva[b_], rv[b_], seed + 2)
    gA = np.clip(nA.predict(Fte) / np.maximum(dA.predict(Fte), 1e-8), 0, 1)
    gB = np.clip(nB.predict(Fte) / np.maximum(dB.predict(Fte), 1e-8), 0, 1)
    var_est = float(np.mean((gA - gB) ** 2) / 2.0) / 2.0      # half-sample -> full-sample variance
    var_tot = float(np.var(g_free))
    var_within = max(var_tot - var_est, 0.0)
    lam = var_within / (var_within + var_est + 1e-12)          # shrinkage toward the cell gate
    g_shrunk = np.clip(lam * g_free + (1 - lam) * g_static, 0, 1)
    g_orac = np.clip((mte * rt) / np.maximum(rt ** 2, 1e-8), 0, 1)
    out = {}
    for name, g in [("static", g_static), ("free", g_free), ("shrunk", g_shrunk), ("oracle", g_orac)]:
        p = g * rt; sk = []
        for r_ in REG:
            for h_ in HOR:
                k = (rte == r_) & (hte == h_)
                sk.append(1 - np.mean((Rte[k] - p[k]) ** 2) / np.mean(Rte[k] ** 2))
        out[name] = float(np.mean(sk))
    out.update(dict(var_est=var_est, var_within=var_within, lam=float(lam),
                    var_within_oracle=float(np.var(g_orac - g_static))))
    return out

grid = list(itertools.product([0.0, 0.5, 1.0, 2.0], [500, 2000, 8000]))
res = {}
t0 = time.time()
ckpt = open(os.path.join(OUT, "S3_cells.jsonl"), "a", encoding="utf-8")
for amp, nv in grid:
    runs = [run(s, amp, nv) for s in range(6)]
    key = f"amp{amp}_nval{nv}"
    res[key] = runs
    ckpt.write(json.dumps({"cell": key, "runs": runs}, default=float) + chr(10)); ckpt.flush()
    st = np.array([r["static"] for r in runs]); fr = np.array([r["free"] for r in runs])
    sh = np.array([r["shrunk"] for r in runs]); orc = np.array([r["oracle"] for r in runs])
    print(f"amp={amp:4.1f} n_val={nv:6d} | static={st.mean():.4f} free={fr.mean():.4f} "
          f"shrunk={sh.mean():.4f} oracle={orc.mean():.4f} | free-static={fr.mean()-st.mean():+.4f} "
          f"({int((fr>st).sum())}/6) shrunk-static={sh.mean()-st.mean():+.4f} ({int((sh>st).sum())}/10) "
          f"| var_within={np.mean([r['var_within'] for r in runs]):.4f} "
          f"var_est={np.mean([r['var_est'] for r in runs]):.4f} lam={np.mean([r['lam'] for r in runs]):.2f} "
          f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(os.path.join(OUT, "S3_granularity.json"), "w"), indent=1, default=float)
print("DONE", time.time() - t0)
