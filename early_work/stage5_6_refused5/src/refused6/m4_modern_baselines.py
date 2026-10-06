"""RE-FUSED-6 / M4: does correction allocation survive when the BASELINE is a modern deep forecaster?

This is the decisive fairness test. Everything so far corrected persistence / rolling mean /
seasonal naive / an operator schedule. A reviewer's first objection is that the whole effect is an
artifact of weak baselines. Here the baseline B is produced by DLinear, NHITS and PatchTST
(neuralforecast, GPU), each trained on TRAIN only, then rolled forward over val+test.

The ladder is then applied to THEIR residuals under the canonical protocol:
    rung0 none | rung1 global | rung2 per-series | rung3 per-series x regime
    rung4 shrunk (empirical Bayes) | rung5 per-instance (validated WLS-bagged estimator)

Predictions: neuralforecast cross_validation with horizon h, step_size h, refit=False, so the model
is fitted once on train and rolled forward - no test information enters the baseline.
Features for the corrector use only information available at the forecast cutoff.

Usage: py -3.10 -u src/refused6/m4_modern_baselines.py [ETTh1 ETTh2 ...]
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

ROOT = r"D:\\REFUSED5"
OUT = os.path.join(ROOT, "results", "refused6")
PUB = os.path.join(ROOT, "data", "public")
os.makedirs(OUT, exist_ok=True)
CLIP, N_SEEDS, N_BOOT, MIN_CELL, N_REGIMES = 1.5, 5, 5, 30, 3
H = 24
INPUT = 168
Z_CLIP = 10.0


def load_ett(name):
    d = pd.read_csv(os.path.join(PUB, name + ".csv"), parse_dates=["date"])
    chans = [c for c in d.columns if c != "date"]
    L = d.melt(id_vars="date", value_vars=chans, var_name="unique_id", value_name="y").dropna()
    return L.rename(columns={"date": "ds"}).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def baseline_predictions(df, name):
    """Fit modern models on the first 70% and roll forward over the remaining 30%."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DLinear, NHITS, PatchTST
    ts = np.sort(df.ds.unique())
    cut = ts[int(0.70 * len(ts))]
    n_windows = int((len(ts) - int(0.70 * len(ts))) // H)
    models = [
        DLinear(h=H, input_size=INPUT, max_steps=400, random_seed=0, scaler_type="standard"),
        NHITS(h=H, input_size=INPUT, max_steps=400, random_seed=0, scaler_type="standard"),
        PatchTST(h=H, input_size=INPUT, max_steps=400, random_seed=0, scaler_type="standard",
                 patch_len=16, stride=8, hidden_size=64, n_heads=4),
    ]
    nf = NeuralForecast(models=models, freq="h")
    t0 = time.time()
    cv = nf.cross_validation(df=df, n_windows=n_windows, step_size=H, refit=False, verbose=False)
    print(f"  [{name}] cross_validation: {len(cv)} rows, {n_windows} windows, {time.time()-t0:.0f}s", flush=True)
    return cv.reset_index() if "unique_id" not in cv.columns else cv


def build_features(df, cv):
    """Causal features known at the cutoff, joined onto each forecast row."""
    hist = df.sort_values(["unique_id", "ds"]).copy()
    g = hist.groupby("unique_id")["y"]
    feat = pd.DataFrame({"unique_id": hist.unique_id, "cutoff": hist.ds,
                         "lag1": g.shift(0), "lag24": g.shift(23), "lag168": g.shift(167),
                         "rm24": g.rolling(24).mean().reset_index(0, drop=True),
                         "rm168": g.rolling(168).mean().reset_index(0, drop=True),
                         "rs24": g.rolling(24).std().reset_index(0, drop=True)})
    out = cv.merge(feat, on=["unique_id", "cutoff"], how="left")
    out["step"] = ((out.ds - out.cutoff).dt.total_seconds() // 3600).astype(int)
    out["hour"] = out.ds.dt.hour
    out["dow"] = out.ds.dt.dayofweek
    out["sid"] = out.unique_id.astype("category").cat.codes
    return out.dropna(subset=["lag1", "lag24", "lag168", "rm24", "rm168", "rs24"])


FEATS = ["lag1", "lag24", "lag168", "rm24", "rm168", "rs24", "step", "hour", "dow", "sid"]


def ls_gate(R, r):
    den = float(np.dot(r, r))
    return float(np.dot(R, r) / den) if den > 1e-12 else 0.0


def wls_gate(F, R, r, seed):
    w = r ** 2
    ok = w > 1e-12
    z = np.clip(R[ok] / r[ok], -Z_CLIP, Z_CLIP)
    m = HGB(max_iter=150, learning_rate=0.08, max_depth=5, random_state=seed)
    m.fit(F[ok], z, sample_weight=w[ok])
    return m


def ladder_on(model_col, D, seed):
    d = D.dropna(subset=[model_col]).copy()
    d["R"] = d.y - d[model_col]
    ts = np.sort(d.cutoff.unique())
    v_end = ts[int(0.5 * len(ts))]
    va, te = d[d.cutoff <= v_end], d[d.cutoff > v_end]
    if min(len(va), len(te)) < 400:
        return None
    # corrector trained on validation-half? No: train on the FIRST half of the rolled span,
    # gates on the same half is leakage; so split val into corrector-fit and gate-fit halves.
    ts_v = np.sort(va.cutoff.unique())
    c_end = ts_v[int(0.5 * len(ts_v))]
    fit, gate = va[va.cutoff <= c_end], va[va.cutoff > c_end]
    if min(len(fit), len(gate)) < 200:
        return None
    corr = HGB(max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed,
               l2_regularization=1.0).fit(fit[FEATS], fit.R)
    rg, rt = corr.predict(gate[FEATS]), corr.predict(te[FEATS])
    Rg, Rt = gate.R.to_numpy(), te.R.to_numpy()
    g1 = ls_gate(Rg, rg)
    rg, rt = g1 * rg, g1 * rt              # D9 reparameterisation
    g1e = ls_gate(Rg, rg)
    sid_g, sid_t = gate.sid.to_numpy(), te.sid.to_numpy()
    lo, hi = np.nanquantile(fit.rs24, [1 / 3, 2 / 3])
    rq_g = np.digitize(gate.rs24.to_numpy(), [lo, hi])
    rq_t = np.digitize(te.rs24.to_numpy(), [lo, hi])
    g2map = {s: (ls_gate(Rg[sid_g == s], rg[sid_g == s]) if (sid_g == s).sum() >= MIN_CELL else g1e)
             for s in np.unique(sid_g)}
    g2 = np.array([g2map.get(s, g1e) for s in sid_t])
    g3map = {}
    for s in np.unique(sid_g):
        for q in range(N_REGIMES):
            k = (sid_g == s) & (rq_g == q)
            g3map[(s, q)] = ls_gate(Rg[k], rg[k]) if k.sum() >= MIN_CELL else g2map.get(s, g1e)
    g3 = np.array([g3map.get((s, q), g1e) for s, q in zip(sid_t, rq_t)])
    Fg, Ft = gate[FEATS].to_numpy(), te[FEATS].to_numpy()
    bw = np.empty((N_BOOT, len(Ft)))
    for b in range(N_BOOT):
        rb = np.random.default_rng(700 + 13 * seed + b)
        idx = rb.integers(0, len(Fg), len(Fg))
        bw[b] = wls_gate(Fg[idx], Rg[idx], rg[idx], seed * 23 + b).predict(Ft)
    g5 = bw.mean(axis=0)
    V, W = float(np.mean(bw.var(axis=0, ddof=1))), float(np.var(g5))
    lam = W / (W + V) if (W + V) > 0 else 0.0
    gates = {"rung0_none": np.zeros(len(te)), "rung1_global": np.full(len(te), g1e),
             "rung2_series": g2, "rung3_cell": g3, "rung5_instance": g5,
             "rung4_shrunk": lam * g5 + (1 - lam) * g2}
    base = Rt ** 2
    rows = []
    for rung, g in gates.items():
        gc = np.clip(g, 0, CLIP)
        sq = (Rt - gc * rt) ** 2
        msq = [1 - sq[sid_t == s].mean() / base[sid_t == s].mean() for s in np.unique(sid_t)]
        rows.append(dict(baseline_model=model_col, rung=rung, seed=seed,
                         skill_sq_macro=float(np.mean(msq)),
                         skill_sq_pooled=float(1 - sq.mean() / base.mean()),
                         base_mse=float(base.mean()), base_mae=float(np.abs(Rt).mean()),
                         frac_clipped=float(np.mean((g < 0) | (g > CLIP))),
                         lam=lam, rho=(W / V if V > 0 else np.nan),
                         resid_R2=float(1 - np.mean((Rt - rt) ** 2) / np.var(Rt)),
                         n_test=len(te), n_gate=len(gate)))
    return rows


def main(datasets):
    allr = []
    for name in datasets:
        df = load_ett(name)
        cv = baseline_predictions(df, name)
        model_cols = [c for c in cv.columns if c in ("DLinear", "NHITS", "PatchTST")]
        D = build_features(df, cv)
        # reference: persistence baseline on the same rows, for comparability
        D["Persistence"] = D["lag1"]
        for mc in model_cols + ["Persistence"]:
            mae = float(np.abs(D.y - D[mc]).mean())
            print(f"  [{name}] baseline {mc:12s} MAE={mae:.4f}", flush=True)
            for seed in range(N_SEEDS):
                r = ladder_on(mc, D, seed)
                if r:
                    for row in r:
                        row["dataset"] = name
                    allr += r
            d = pd.DataFrame(allr).query("dataset == @name and baseline_model == @mc")
            if len(d):
                m = d.groupby("rung").skill_sq_macro.mean()
                print("      ladder: " + " ".join(f"{k.replace('rung','r')[:10]}={m.get(k, float('nan')):+.4f}"
                      for k in ["rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]) +
                      f" | best={m.drop('rung0_none', errors='ignore').idxmax()}", flush=True)
    R = pd.DataFrame(allr)
    R.to_csv(os.path.join(OUT, "M4_modern_baselines.csv"), index=False)
    print(f"\nwrote {len(R)} rows")
    if len(R):
        print("\n=== ladder by baseline model (mean skill over seeds/datasets) ===")
        print(R.pivot_table(index="baseline_model", columns="rung", values="skill_sq_macro").round(4).to_string())
        print("\n=== baseline strength vs correction value ===")
        print(R[R.rung == "rung2_series"].groupby("baseline_model")[["base_mae", "skill_sq_macro", "resid_R2"]].mean().round(4).to_string())
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ETTh1", "ETTh2"])
