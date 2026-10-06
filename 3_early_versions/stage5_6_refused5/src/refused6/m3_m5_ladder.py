"""RE-FUSED-6 / M3 + M5: causal synthetic grid and validation-only rung selection.

M3 (mechanism): independently vary between-cell gate heterogeneity, within-cell heterogeneity,
validation size, residual signal strength, AR(1) dependence, corrector misspecification, baseline
drift, and the number of adaptive cells. Includes deliberately adversarial settings (zero signal,
pure noise gate structure, drift visible only in test).

M5 (selection): choose a rung on the ladder using VALIDATION ONLY, and measure oracle regret against
the best rung chosen with test knowledge. Selectors:
    srm_c      penalised validation risk:  R_hat_k + c * sqrt(log M / n_val)
    rho_rule   the RE-FUSED-5/6 crossing rule: refine only while W/V > 1
    fixed_k    always use rung k (the field's default is the finest rung)
Unimodality of the realised risk curve is recorded as a falsifiable property, not assumed.

Usage:  py -3.10 -u src/refused6/m3_m5_ladder.py [quick|full]
"""
import os, sys, json, time, itertools, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge

OUT = r"D:\\REFUSED5\results\refused6"
os.makedirs(OUT, exist_ok=True)
CLIP = 1.5
RUNG_ORDER = ["rung0_none", "rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]
N_BOOT = 5

DEFAULT = dict(n_series=12, n_regimes=3, between=0.30, within=0.0, snr=0.30, phi=0.0,
               mis=0, drift=0.0, n_train=8000, n_val=3000, n_test=20000, d_sig=5, d_noise=20)


def simulate(k, seed):
    """One process; train/val/test drawn from it. Returns dict of arrays."""
    p = dict(DEFAULT); p.update(k)
    rng = np.random.default_rng(seed)
    d = p["d_sig"] + p["d_noise"]
    beta = rng.normal(0, 1, p["d_sig"])
    # cell-level gate amplitudes: between-cell heterogeneity
    n_cells = p["n_series"] * p["n_regimes"]
    amp_cell = np.clip(1.0 + rng.normal(0, p["between"], n_cells), 0.05, None)

    def draw(n, is_test=False):
        X = rng.normal(0, 1, (n, d))
        sid = rng.integers(0, p["n_series"], n)
        reg = rng.integers(0, p["n_regimes"], n)
        cid = sid * p["n_regimes"] + reg
        u = rng.uniform(-1, 1, n)
        core = X[:, :p["d_sig"]] @ beta
        core = core / (core.std() + 1e-9)
        amp = amp_cell[cid] * (1.0 + p["within"] * u)
        m = core * np.clip(amp, 0.0, None) * np.sqrt(p["snr"])
        # AR(1) noise gives temporal dependence
        e = rng.normal(0, 1, n)
        if p["phi"] > 0:
            for i in range(1, n):
                e[i] = p["phi"] * e[i - 1] + np.sqrt(1 - p["phi"] ** 2) * e[i]
        R = m + e
        if is_test and p["drift"] > 0:            # baseline bias drift unseen in validation
            R = R + p["drift"] * np.linspace(0, 1, n)
        return dict(X=X, sid=sid, reg=reg, cid=cid, m=m, R=R, t=np.arange(n))

    tr, va, te = draw(p["n_train"]), draw(p["n_val"]), draw(p["n_test"], is_test=True)
    # corrector features: optionally misspecified (hide informative columns)
    keep = np.arange(d)
    if p["mis"] > 0:
        keep = np.arange(p["mis"], d)             # drop the first `mis` informative features
    for part in (tr, va, te):
        part["F"] = np.column_stack([part["X"][:, keep], part["sid"], part["reg"]])
    return p, tr, va, te


def ls_gate(R, r):
    den = float(np.dot(r, r))
    return float(np.dot(R, r) / den) if den > 1e-12 else 0.0


def wls_gate(F, R, r, seed, z_clip=10.0):
    """Validated estimator (M1b): regress z = R/r on x with weights r^2 (the projection defining g*)."""
    w = r ** 2
    ok = w > 1e-12
    z = np.clip(R[ok] / r[ok], -z_clip, z_clip)
    m = HGB(max_iter=150, learning_rate=0.08, max_depth=5, random_state=seed)
    m.fit(F[ok], z, sample_weight=w[ok])
    return m


def gate_models(F, ynum, yden, seed):
    a = HGB(max_iter=120, learning_rate=0.09, max_depth=5, random_state=seed).fit(F, ynum)
    b = HGB(max_iter=120, learning_rate=0.09, max_depth=5, random_state=seed).fit(F, yden)
    return a, b


def build_rungs(tr, va, te, seed, min_cell=30):
    corr = HGB(max_iter=200, learning_rate=0.08, max_depth=6, random_state=seed).fit(tr["F"], tr["R"])
    rv, rt = corr.predict(va["F"]), corr.predict(te["F"])
    Rv, Rt = va["R"], te["R"]
    g1 = ls_gate(Rv, rv)
    g2map = {s: (ls_gate(Rv[va["sid"] == s], rv[va["sid"] == s]) if (va["sid"] == s).sum() >= min_cell else g1)
             for s in np.unique(va["sid"])}
    g3map = {}
    for c in np.unique(va["cid"]):
        k = va["cid"] == c
        s = c // (va["reg"].max() + 1)
        g3map[c] = ls_gate(Rv[k], rv[k]) if k.sum() >= min_cell else g2map.get(s, g1)
    num, den = gate_models(va["F"], Rv * rv, rv ** 2, seed)
    g5raw_te = num.predict(te["F"]) / np.maximum(den.predict(te["F"]), 1e-8)
    w0 = wls_gate(va["F"], Rv, rv, seed)
    g5_te, g5_va = w0.predict(te["F"]), w0.predict(va["F"])
    boots_te, boots_va = [], []
    for b in range(N_BOOT):
        rb = np.random.default_rng(977 * seed + b)
        idx = rb.integers(0, len(va["F"]), len(va["F"]))
        wb = wls_gate(va["F"][idx], Rv[idx], rv[idx], seed * 31 + b)
        boots_te.append(wb.predict(te["F"]))
        boots_va.append(wb.predict(va["F"]))
    Bte, Bva = np.vstack(boots_te), np.vstack(boots_va)
    V = float(np.mean(Bte.var(axis=0, ddof=1)))
    W = float(np.var(Bte.mean(axis=0)))
    lam = W / (W + V) if (W + V) > 0 else 0.0
    gate_te = {
        "rung0_none": np.zeros(len(Rt)), "rung1_global": np.full(len(Rt), g1),
        "rung2_series": np.array([g2map.get(s, g1) for s in te["sid"]]),
        "rung3_cell": np.array([g3map.get(c, g1) for c in te["cid"]]),
        "rung5_instance": g5_te, "rung5_raw_ratio": g5raw_te,
    }
    gate_te["rung4_shrunk"] = lam * Bte.mean(axis=0) + (1 - lam) * gate_te["rung3_cell"]
    gate_va = {
        "rung0_none": np.zeros(len(Rv)), "rung1_global": np.full(len(Rv), g1),
        "rung2_series": np.array([g2map.get(s, g1) for s in va["sid"]]),
        "rung3_cell": np.array([g3map.get(c, g1) for c in va["cid"]]),
        "rung5_instance": g5_va, "rung5_raw_ratio": num.predict(va["F"]) / np.maximum(den.predict(va["F"]), 1e-8),
    }
    gate_va["rung4_shrunk"] = lam * Bva.mean(axis=0) + (1 - lam) * gate_va["rung3_cell"]
    # oracle gate on test (upper bound only, never used for selection)
    g_or = np.where(np.abs(rt) > 1e-9, Rt / np.where(np.abs(rt) > 1e-9, rt, 1.0), 0.0)
    return dict(rv=rv, rt=rt, Rv=Rv, Rt=Rt, gate_te=gate_te, gate_va=gate_va,
                V=V, W=W, lam=lam, g1=g1, g_or=g_or,
                n_cells_used=int(len(np.unique(va["cid"]))), n_series=int(len(np.unique(va["sid"]))))


def risk(R, g, r):
    return float(np.mean((R - np.clip(g, 0, CLIP) * r) ** 2))


def run_one(knobs, seed):
    p, tr, va, te = simulate(knobs, seed)
    z = build_rungs(tr, va, te, seed)
    base_te, base_va = float(np.mean(z["Rt"] ** 2)), float(np.mean(z["Rv"] ** 2))
    test_risk = {k: risk(z["Rt"], g, z["rt"]) for k, g in z["gate_te"].items()}
    val_risk = {k: risk(z["Rv"], g, z["rv"]) for k, g in z["gate_va"].items()}
    # ---- M5 selection, validation only ----
    M = len(RUNG_ORDER)
    n_params = {"rung0_none": 0, "rung1_global": 1, "rung2_series": z["n_series"],
                "rung3_cell": z["n_cells_used"], "rung4_shrunk": z["n_cells_used"] + 1,
                "rung5_instance": 50}
    sel = {}
    for c in (0.0, 0.5, 1.0):
        pen = {k: val_risk[k] + c * base_va * np.sqrt(np.log(max(M, 2)) / len(z["Rv"]))
                  * np.sqrt(max(n_params[k], 1)) for k in RUNG_ORDER}
        sel[f"srm_c{c}"] = min(pen, key=pen.get)
    sel["rho_rule"] = "rung5_instance" if (z["W"] / max(z["V"], 1e-12)) > 1 else "rung3_cell"
    sel["val_argmin"] = min(val_risk, key=val_risk.get)
    oracle_rung = min(test_risk, key=test_risk.get)
    COMPLEXITY = ["rung0_none", "rung1_global", "rung2_series", "rung3_cell", "rung5_instance"]
    curve = [test_risk[k] for k in COMPLEXITY]
    dif = np.diff(curve)
    unimodal = bool(np.all(np.diff(np.sign(dif)[np.sign(dif) != 0]) >= 0))
    from scipy import stats as _st
    _rungs = [k for k in RUNG_ORDER if k in test_risk and k in val_risk]
    _rk = _st.spearmanr([val_risk[k] for k in _rungs], [test_risk[k] for k in _rungs])
    row = dict(seed=seed, val_test_rank_corr=float(_rk[0]), **{f"p_{k}": v for k, v in knobs.items()},
               n_val=p["n_val"], between=p["between"], within=p["within"], snr=p["snr"],
               phi=p["phi"], mis=p["mis"], drift=p["drift"], n_series=p["n_series"],
               V=z["V"], W=z["W"], lam=z["lam"], rho=z["W"] / max(z["V"], 1e-12),
               base_te=base_te, oracle_rung=oracle_rung, unimodal=unimodal,
               resid_R2=float(1 - np.mean((z["Rt"] - z["rt"]) ** 2) / np.var(z["Rt"])),
               oracle_gate_risk=risk(z["Rt"], z["g_or"], z["rt"]))
    for k in RUNG_ORDER:
        row[f"risk_{k}"] = test_risk[k]
        row[f"skill_{k}"] = 1 - test_risk[k] / base_te
        row[f"valrisk_{k}"] = val_risk[k]
    for name, pick in sel.items():
        row[f"sel_{name}"] = pick
        row[f"regret_{name}"] = (test_risk[pick] - test_risk[oracle_rung]) / base_te
        row[f"correct_{name}"] = int(pick == oracle_rung)
    for k in RUNG_ORDER:                                  # fixed-policy comparators
        row[f"regret_fixed_{k}"] = (test_risk[k] - test_risk[oracle_rung]) / base_te
    return row


def grid(mode):
    g = []
    if mode == "quick":
        g += [dict(between=b) for b in (0.0, 0.3, 0.8)]
        g += [dict(n_val=n) for n in (300, 3000)]
        return g
    # full factorial-ish sweep, one knob at a time from the default, plus adversarial corners
    g += [dict(between=b) for b in (0.0, 0.1, 0.3, 0.6, 1.0)]
    g += [dict(within=w) for w in (0.0, 0.5, 1.0, 2.0)]
    g += [dict(n_val=n) for n in (300, 600, 1500, 3000, 8000, 20000)]
    g += [dict(snr=s) for s in (0.02, 0.1, 0.3, 1.0, 3.0)]
    g += [dict(phi=f) for f in (0.0, 0.5, 0.9)]
    g += [dict(mis=m) for m in (0, 2, 4)]
    g += [dict(drift=dr) for dr in (0.0, 0.5, 2.0)]
    g += [dict(n_series=s, n_regimes=r) for s, r in ((4, 2), (12, 3), (40, 5))]
    # adversarial corners
    g += [dict(snr=0.0, between=0.0, name="adv_no_signal"),
          dict(snr=0.02, between=1.0, n_val=300, name="adv_noise_heterogeneity"),
          dict(drift=2.0, n_val=300, name="adv_drift_unseen"),
          dict(within=2.0, n_val=300, snr=0.1, name="adv_within_scarce"),
          dict(mis=4, snr=1.0, name="adv_misspecified_strong_signal")]
    return g


def main(mode="full", seeds=10):
    rows = []
    t0 = time.time()
    G = grid(mode)
    print(f"M3/M5: {len(G)} configurations x {seeds} seeds")
    for i, k in enumerate(G):
        name = k.pop("name", None)
        for s in range(seeds):
            r = run_one(k, s)
            r["config_id"] = i
            r["config_name"] = name or ",".join(f"{a}={b}" for a, b in k.items()) or "default"
            rows.append(r)
        d = pd.DataFrame(rows).query("config_id == @i")
        best = d[[f"skill_{x}" for x in RUNG_ORDER]].mean()
        print(f"[{i+1}/{len(G)}] {d.config_name.iloc[0][:34]:34s} " +
              " ".join(f"{x.split('_')[0]}={best[f'skill_{x}']:+.4f}" for x in RUNG_ORDER) +
              f" | oracle={d.oracle_rung.mode().iloc[0][:12]:12s} rho={d.rho.mean():.2f}"
              f" uni={d.unimodal.mean():.2f} regret_srm1={float(d['regret_srm_c1.0'].mean()):+.4f}"
              f" [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"M3M5_runs_{mode}.csv"), index=False)
    print(f"\nwrote {len(R)} rows")
    # headline summaries
    print("\n=== M5 selection performance (mean regret in units of baseline risk) ===")
    cols = [c for c in R.columns if c.startswith("regret_")]
    print(R[cols].mean().sort_values().round(5).to_string())
    print("\n=== oracle rung distribution ===")
    print(R.oracle_rung.value_counts().to_string())
    print(f"\n=== unimodality rate: {R.unimodal.mean():.3f} ===")
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
