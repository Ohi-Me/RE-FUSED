"""RE-FUSED-5 S4: is the criterion's failure an ESTIMATOR artifact or a property of the world?

The split-half estimate of gate variance gave rho > 1 everywhere (R7), so the decision rule was
uninformative. Here the gate estimation variance is measured properly by bootstrap:

    ghat_b(x) for b = 1..B on bootstrap resamples of validation
    var_est    = mean_x  Var_b[ ghat_b(x) ]          (estimation variance)
    var_within = Var_x [ mean_b ghat_b(x) ]          (genuine within-cell signal)
    rho        = var_within / var_est

Also evaluates a BAGGED gate (mean_b ghat_b): if bagging alone closes the gap to the shrunk gate,
the whole phenomenon is gate-estimation variance and nothing else.
"""
import numpy as np, pandas as pd, json, os, itertools, time, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

OUT = r"D:\\REFUSED5\results"
HOR = [1, 3, 6, 12, 24]
REG = [0, 1, 2]
SIG = {0: 0.15, 1: 0.60, 2: 1.50}
B = 5


class Process:
    def __init__(self, seed, amp_within, d_sig=5, d_noise=25):
        r = np.random.default_rng(seed)
        self.beta = r.normal(0, 1, d_sig)
        self.d_sig, self.d_noise, self.a = d_sig, d_noise, amp_within
        self.scale = None

    def sample(self, n, rng):
        X = rng.normal(0, 1, (n, self.d_sig + self.d_noise))
        reg = rng.integers(0, 3, n)
        h = np.array(rng.choice(HOR, n))
        v = rng.uniform(0, 1, n)
        core = X[:, :self.d_sig] @ self.beta
        if self.scale is None:
            self.scale = core.std()
        core = core / self.scale
        amp = np.array([SIG[r_] for r_ in reg]) / (1.0 + 0.25 * (h - 1))
        amp = amp * (1.0 + self.a * (v - 0.5) * 2.0).clip(0.05, None)
        m = core * amp
        return np.column_stack([X, reg, h, v]), m + rng.normal(0, 1, n), m, reg, h, v


def gate_fit(F, y_num, y_den, seed):
    n = HGB(max_iter=180, learning_rate=0.07, max_depth=6, random_state=seed).fit(F, y_num)
    d = HGB(max_iter=180, learning_rate=0.07, max_depth=6, random_state=seed).fit(F, y_den)
    return n, d


def run(seed, amp, n_val):
    P = Process(seed, amp)
    rng = np.random.default_rng(10_000 + seed)
    Ftr, Rtr, *_ = P.sample(12000, rng)
    Fva, Rva, mva, rva, hva, vva = P.sample(n_val, rng)
    Fte, Rte, mte, rte_r, hte, vte = P.sample(30000, rng)
    corr = HGB(max_iter=250, learning_rate=0.07, max_depth=6, random_state=seed).fit(Ftr, Rtr)
    rv, rt = corr.predict(Fva), corr.predict(Fte)

    g_glob = float(np.dot(Rva, rv) / np.dot(rv, rv))
    cell = {}
    for r_ in REG:
        for h_ in HOR:
            k = (rva == r_) & (hva == h_)
            cell[(r_, h_)] = (float(np.dot(Rva[k], rv[k]) / np.dot(rv[k], rv[k]))
                              if k.sum() > 20 else g_glob)
    g_static = np.array([cell[(r_, h_)] for r_, h_ in zip(rte_r, hte)])

    num, den = gate_fit(Fva, Rva * rv, rv ** 2, seed)
    g_free = np.clip(num.predict(Fte) / np.maximum(den.predict(Fte), 1e-8), 0, 1)

    # bootstrap ensemble of gates -> proper variance decomposition
    boots = []
    for b in range(B):
        rb = np.random.default_rng(1000 * seed + b)
        idx = rb.integers(0, len(Fva), len(Fva))
        nb, db = gate_fit(Fva[idx], (Rva * rv)[idx], (rv ** 2)[idx], seed * 10 + b)
        boots.append(np.clip(nb.predict(Fte) / np.maximum(db.predict(Fte), 1e-8), 0, 1))
    Bm = np.vstack(boots)
    g_bag = Bm.mean(axis=0)
    var_est = float(np.mean(Bm.var(axis=0, ddof=1)))
    var_within = float(np.var(g_bag))
    rho = var_within / var_est if var_est > 0 else np.nan
    lam = var_within / (var_within + var_est) if var_est > 0 else 0.0
    g_shrunk = np.clip(lam * g_bag + (1 - lam) * g_static, 0, 1)
    g_orac = np.clip((mte * rt) / np.maximum(rt ** 2, 1e-8), 0, 1)

    out = {}
    for name, g in [("static", g_static), ("free", g_free), ("bagged", g_bag),
                    ("shrunk", g_shrunk), ("oracle", g_orac)]:
        p = g * rt
        sk = []
        for r_ in REG:
            for h_ in HOR:
                k = (rte_r == r_) & (hte == h_)
                sk.append(1 - np.mean((Rte[k] - p[k]) ** 2) / np.mean(Rte[k] ** 2))
        out[name] = float(np.mean(sk))
    # what the oracle gate's genuine within-cell variation actually is
    within_true = float(np.mean([np.var(g_orac[(rte_r == r_) & (hte == h_)])
                                 for r_ in REG for h_ in HOR]))
    out.update(dict(var_est=var_est, var_within=var_within, rho=rho, lam=lam,
                    within_true=within_true, amp=amp, n_val=n_val, seed=seed))
    return out


rows = []
t0 = time.time()
grid = list(itertools.product([0.0, 0.5, 1.0, 2.0], [500, 8000]))
for amp, nv in grid:
    for seed in range(4):
        rows.append(run(seed, amp, nv))
    sub = [r for r in rows if r["amp"] == amp and r["n_val"] == nv]
    f = lambda k: float(np.mean([r[k] for r in sub]))
    print("amp=%4.1f n_val=%6d | static=%.4f free=%.4f bagged=%.4f shrunk=%.4f oracle=%.4f "
          "| rho=%.2f var_est=%.4f var_within=%.4f within_true=%.4f | free-static=%+.4f bag-static=%+.4f [%.0fs]"
          % (amp, nv, f("static"), f("free"), f("bagged"), f("shrunk"), f("oracle"), f("rho"),
             f("var_est"), f("var_within"), f("within_true"),
             f("free") - f("static"), f("bagged") - f("static"), time.time() - t0), flush=True)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "S4_variance_fix.csv"), index=False)
from scipy import stats
d = d.dropna(subset=["rho"])
acc = float(((d.rho > 1).astype(int) == ((d.free - d.static) > 0).astype(int)).mean())
sp = stats.spearmanr(d.rho, d.free - d.static)
print("")
print("=== criterion with bootstrap variance ===")
print("configs=%d | rho range %.2f..%.2f | sign accuracy=%.3f | spearman=%+.3f (p=%.4f)"
      % (len(d), d.rho.min(), d.rho.max(), acc, sp[0], sp[1]))
print("conditional (free) wins %d/%d | bagged wins %d/%d | shrunk wins %d/%d"
      % (int((d.free > d.static).sum()), len(d), int((d.bagged > d.static).sum()), len(d),
         int((d.shrunk > d.static).sum()), len(d)))
t1 = stats.ttest_rel(d.shrunk, d.static)
t2 = stats.ttest_rel(d.bagged, d.static)
print("shrunk-static mean=%+.5f t=%+.2f p=%.4f | bagged-static mean=%+.5f t=%+.2f p=%.4f"
      % ((d.shrunk - d.static).mean(), t1.statistic, t1.pvalue,
         (d.bagged - d.static).mean(), t2.statistic, t2.pvalue))
print("DONE")
