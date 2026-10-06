"""RE-FUSED-5 S2: heterogeneous predictability across regimes x horizons.
Question (N1/N2): does a LEARNED CONDITIONAL gate beat (a) always-correct, (b) a single global
gate (Diebold-Pauly style), (c) a static per-(regime,horizon) gate (CRC-style)? Is it calibrated?
CVC estimator: g(x) = E[R*rhat | x] / E[rhat^2 | x], from two conditional-expectation regressions."""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
OUT = r"D:\\REFUSED5\results"; os.makedirs(OUT, exist_ok=True)

REGIME_SNR = {0: 0.02, 1: 0.20, 2: 1.00}          # calm / mixed / predictable
HORIZONS   = [1, 3, 6, 12, 24]
def snr_of(regime, h):                              # predictability decays with horizon
    return REGIME_SNR[regime] * (1.0 / (1.0 + 0.25 * (h - 1)))

def simulate(n, seed, d_sig=5, d_noise=25):
    rng = np.random.default_rng(seed)
    reg = rng.integers(0, 3, n)
    h   = rng.choice(HORIZONS, n)
    X   = rng.normal(0, 1, (n, d_sig + d_noise))
    beta = rng.normal(0, 1, d_sig)
    core = X[:, :d_sig] @ beta
    core = core / core.std()
    s = np.array([snr_of(r, hh) for r, hh in zip(reg, h)])
    m = core * 1.0                                   # Var(m)=1 by construction
    eta = rng.normal(0, 1, n) / np.sqrt(s)           # noise scaled so SNR = Var(m)/Var(eta) = s
    R = m + eta
    feats = np.column_stack([X, reg, h])             # regime & horizon observable
    return feats, R, m, s, reg, h

def fit_corrector(Ftr, Rtr):
    return Ridge(alpha=5.0).fit(Ftr, Rtr)

def evaluate(seed):
    Ftr, Rtr, *_ = simulate(8000, seed)
    Fva, Rva, mva, sva, rva, hva = simulate(4000, seed + 500)
    Fte, Rte, mte, ste, rte, hte = simulate(20000, seed + 999)
    corr = fit_corrector(Ftr, Rtr)
    rv, rt = corr.predict(Fva), corr.predict(Fte)

    res = {}
    mse = lambda pred: float(np.mean((Rte - pred) ** 2))
    res["never"]  = mse(0.0)
    res["always"] = mse(rt)
    g_glob = float(np.dot(Rva, rv) / np.dot(rv, rv))                       # single global gate
    res["global_gate"] = mse(g_glob * rt)
    # static per-(regime,horizon) gate, fit on validation  (CRC-style stratified selection)
    g_cell = {}
    for r_ in np.unique(rva):
        for h_ in np.unique(hva):
            k = (rva == r_) & (hva == h_)
            g_cell[(r_, h_)] = float(np.dot(Rva[k], rv[k]) / np.dot(rv[k], rv[k])) if k.sum() > 20 else g_glob
    g_static = np.array([g_cell[(r_, h_)] for r_, h_ in zip(rte, hte)])
    res["static_cell_gate"] = mse(g_static * rt)
    # CVC: learned conditional gate from two conditional expectations
    num = HGB(max_iter=250, learning_rate=0.06, max_depth=6).fit(Fva, Rva * rv)
    den = HGB(max_iter=250, learning_rate=0.06, max_depth=6).fit(Fva, rv ** 2)
    g_cvc = np.clip(num.predict(Fte) / np.maximum(den.predict(Fte), 1e-8), 0.0, 1.0)
    res["cvc_gate"] = mse(g_cvc * rt)
    # oracle conditional gate from the known design
    sig_e2_cell = {}
    for r_ in np.unique(rva):
        for h_ in np.unique(hva):
            k = (rva == r_) & (hva == h_)
            sig_e2_cell[(r_, h_)] = float(np.mean((rv[k] - mva[k]) ** 2)) if k.sum() > 20 else 1.0
    g_or = np.array([1.0 / (1.0 + sig_e2_cell[(r_, h_)]) for r_, h_ in zip(rte, hte)])  # E[m^2]=1
    res["oracle_cell_gate"] = mse(g_or * rt)

    # calibration of predicted benefit vs realised benefit (N2)
    pred_ben = g_cvc * num.predict(Fte) / np.maximum(g_cvc, 1e-6) * 0 + (2 * g_cvc * num.predict(Fte) - g_cvc**2 * den.predict(Fte))
    real_ben = Rte ** 2 - (Rte - g_cvc * rt) ** 2
    q = np.quantile(pred_ben, np.linspace(0, 1, 11))
    cal = []
    for i in range(10):
        k = (pred_ben >= q[i]) & (pred_ben <= q[i + 1])
        if k.sum() > 30: cal.append((float(pred_ben[k].mean()), float(real_ben[k].mean()), int(k.sum())))
    A = np.array([[c[0], 1] for c in cal]); y = np.array([c[1] for c in cal])
    slope, icept = np.linalg.lstsq(A, y, rcond=None)[0]
    ss = 1 - ((y - A @ [slope, icept]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # per-cell gate recovery
    gate_err = float(np.mean(np.abs(g_cvc - g_or)))
    return res, dict(cal_slope=float(slope), cal_intercept=float(icept), cal_r2=float(ss),
                     gate_mae_vs_oracle=gate_err, g_global=g_glob), cal

ALL, CALS = [], []
for seed in range(10):
    r, c, _ = evaluate(seed); ALL.append(r); CALS.append(c)
keys = ["never", "always", "global_gate", "static_cell_gate", "cvc_gate", "oracle_cell_gate"]
base = np.array([a["never"] for a in ALL])
print(f"{'strategy':>18} | {'test MSE mean':>13} {'sd':>7} | {'%% vs never':>11} | wins/10")
for k in keys:
    v = np.array([a[k] for a in ALL]); imp = 100 * (base - v) / base
    print(f"{k:>18} | {v.mean():13.4f} {v.std():7.4f} | {imp.mean():10.2f}% | {int((v < base).sum())}")
cv = np.array([a["cvc_gate"] for a in ALL]); st = np.array([a["static_cell_gate"] for a in ALL])
d = st - cv
print(f"\nCVC vs static-cell gate: mean MSE diff={d.mean():+.5f} (sd {d.std():.5f}), CVC better in {(d>0).sum()}/10 seeds")
print(f"calibration slope={np.mean([c['cal_slope'] for c in CALS]):.3f} "
      f"(1.0=perfect), R^2={np.mean([c['cal_r2'] for c in CALS]):.3f}, "
      f"gate MAE vs oracle={np.mean([c['gate_mae_vs_oracle'] for c in CALS]):.3f}")
json.dump({"runs": ALL, "cal": CALS}, open(os.path.join(OUT, "S2_conditional.json"), "w"), indent=1, default=float)
