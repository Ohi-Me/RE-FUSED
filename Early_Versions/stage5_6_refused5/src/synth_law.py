"""RE-FUSED-5 S1: does the shrinkage law hold, and can a learned gate find it?
Ground truth is controlled: explainable residual variance and estimator noise are set by design.
Tests L1 (g* = SNR/(1+SNR)), L3 (always-correct degrades iff sigma_e^2 > E[m^2])."""
import numpy as np, json, os
from sklearn.linear_model import Ridge
rng_global = np.random.default_rng(0)
OUT = r"D:\\REFUSED5\results"; os.makedirs(OUT, exist_ok=True)

def make_env(n, d_signal, d_noise, snr, seed, n_test=20000):
    """Residual R = m(x) + eta, with m linear in d_signal informative features.
    snr controls Var(m)/Var(eta): the *intrinsic* predictability of the residual."""
    rng = np.random.default_rng(seed)
    d = d_signal + d_noise
    beta = np.zeros(d); beta[:d_signal] = rng.normal(0, 1, d_signal)
    def draw(m_):
        X = rng.normal(0, 1, (m_, d))
        mu = X @ beta
        mu = mu / mu.std()                      # Var(m) = 1
        eta = rng.normal(0, np.sqrt(1.0 / snr), m_)   # Var(eta) = 1/snr
        return X, mu, mu + eta
    return draw(n), draw(n_test)

def run(n, d_signal, d_noise, snr, seed):
    (Xtr, mtr, Rtr), (Xte, mte, Rte) = make_env(n, d_signal, d_noise, snr, seed)
    model = Ridge(alpha=1.0).fit(Xtr, Rtr)
    rhat = model.predict(Xte)
    # empirical optimal gate on held-out data (ground truth for this draw)
    g_emp = float(np.dot(Rte, rhat) / np.dot(rhat, rhat))
    # theory: g* = E[m^2] / (E[m^2] + sigma_e^2), with sigma_e^2 measured as E[(rhat-m)^2]
    Em2 = float(np.mean(mte ** 2)); sig_e2 = float(np.mean((rhat - mte) ** 2))
    g_theory = Em2 / (Em2 + sig_e2)
    L0 = float(np.mean(Rte ** 2))                       # never correct
    L1_ = float(np.mean((Rte - rhat) ** 2))             # always correct
    Lg = float(np.mean((Rte - g_theory * rhat) ** 2))   # theory-gated
    Lge = float(np.mean((Rte - g_emp * rhat) ** 2))     # oracle-gated
    return dict(n=n, d_signal=d_signal, d_noise=d_noise, snr=snr, seed=seed,
                g_emp=g_emp, g_theory=g_theory, Em2=Em2, sigma_e2=sig_e2,
                L_never=L0, L_always=L1_, L_gated=Lg, L_oracle_gate=Lge,
                always_helps=L1_ < L0, law_predicts_always_helps=sig_e2 < Em2,
                gain_gated_pct=100 * (L0 - Lg) / L0, gain_always_pct=100 * (L0 - L1_) / L0)

rows = []
for snr in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
    for n in [200, 500, 2000, 10000]:
        for seed in range(5):
            rows.append(run(n, d_signal=5, d_noise=45, snr=snr, seed=seed))
import statistics as st
print(f"{'snr':>5} {'n':>6} | {'g_theory':>9} {'g_emp':>7} | {'E[m2]':>7} {'sig_e2':>7} | "
      f"{'gain_gate%':>10} {'gain_always%':>12} | law_ok")
agree = 0
for snr in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
    for n in [200, 500, 2000, 10000]:
        g = [r for r in rows if r["snr"] == snr and r["n"] == n]
        gt, ge = st.mean(r["g_theory"] for r in g), st.mean(r["g_emp"] for r in g)
        em2, se2 = st.mean(r["Em2"] for r in g), st.mean(r["sigma_e2"] for r in g)
        gg, ga = st.mean(r["gain_gated_pct"] for r in g), st.mean(r["gain_always_pct"] for r in g)
        ok = all(r["always_helps"] == r["law_predicts_always_helps"] for r in g); agree += ok
        print(f"{snr:5.2f} {n:6d} | {gt:9.3f} {ge:7.3f} | {em2:7.3f} {se2:7.3f} | {gg:10.2f} {ga:12.2f} | {ok}")
print(f"\nL3 sign prediction correct in {agree}/28 cells")
corr = np.corrcoef([r["g_theory"] for r in rows], [r["g_emp"] for r in rows])[0, 1]
mae = np.mean([abs(r["g_theory"] - r["g_emp"]) for r in rows])
print(f"g_theory vs g_empirical: corr={corr:.4f}  MAE={mae:.4f}")
json.dump(rows, open(os.path.join(OUT, "S1_synth_law.json"), "w"), indent=1, default=float)
