"""RE-FUSED-6 / M1: canonical experiment harness.

Single source of truth for every later experiment. Fixes the seven audited defects of RE-FUSED-5:
  D1 experimental unit is ONE (config, seed) run, written in tidy long form
  D2 no pooling of different units; aggregation happens only at analysis time
  D3 seasonal-naive is dropped when h is a multiple of 7 (there it IS persistence)
  D4 datasets enter with matched series/windows; resolution sweep is deferred to M6
  D5 gates are fitted under SQUARED loss (matches the theory) and BOTH squared and MAE skill are logged
  D6 every rung is clipped identically to [0, CLIP]; the clipped fraction is logged
  D7 both macro (per-series mean) and pooled skill are logged per run

Ladder of adaptivity (the RE-FUSED-6 research object):
  rung0 none | rung1 global | rung2 per-series | rung3 per-series x volatility-regime
  rung4 shrunk (empirical Bayes toward rung3) | rung5 per-instance | oracle_instance (upper bound)

Usage:  py -3.10 -u src/refused6/m1_canonical.py [india|ett|nyiso|all]
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

ROOT = r"D:\\REFUSED5"
OUT = os.path.join(ROOT, "results", "refused6")
os.makedirs(OUT, exist_ok=True)

CLIP = 1.5          # identical for every rung (D6)
N_SEEDS = 10        # (W1)
N_BOOT = 5          # bootstrap resamples for gate-variance estimation
N_REGIMES = 3       # rung-3 cells = series x volatility tercile
MIN_CELL = 30       # minimum validation points before a cell gets its own gate

# --------------------------------------------------------------------------------------
# datasets: each returns long-form df with series_id, ts, y, split + causal features
# --------------------------------------------------------------------------------------

def _add_features(d, lags, roll):
    g = d.groupby("series_id")["y"]
    for L in lags:
        d[f"lag{L}"] = g.shift(L)
    for W in roll:
        d[f"rm{W}"] = g.shift(1).rolling(W).mean().reset_index(0, drop=True)
        d[f"rs{W}"] = g.shift(1).rolling(W).std().reset_index(0, drop=True)
    d["diff1"] = d["lag1"] - d[f"lag{lags[1]}"]
    feats = [f"lag{L}" for L in lags] + [f"rm{W}" for W in roll] + [f"rs{W}" for W in roll] + ["diff1"]
    return d, feats


def load_india():
    p = pd.read_parquet(os.path.join(ROOT, "data", "india_model_daily.parquet"))
    d = p.rename(columns={"state_name": "series_id", "date": "ts"})[
        ["series_id", "ts", "y", "sched", "cap", "n_units", "dow", "doy", "split"]].copy()
    d = d.sort_values(["series_id", "ts"]).reset_index(drop=True)
    d, feats = _add_features(d, [1, 2, 3, 7, 14, 28], [7, 28])
    d["dow_sin"] = np.sin(2 * np.pi * d.dow / 7); d["dow_cos"] = np.cos(2 * np.pi * d.dow / 7)
    d["doy_sin"] = np.sin(2 * np.pi * d.doy / 365.25); d["doy_cos"] = np.cos(2 * np.pi * d.doy / 365.25)
    feats += ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "cap", "n_units"]
    return dict(name="india_daily", df=d, feats=feats, season=7,
                horizons=[1, 2, 3, 7], block_days=14,
                baselines={"persistence": ("lag0_self", None), "roll7": ("rm7", None),
                           "snaive7": ("season", 7), "operator_schedule": ("col", "sched")})


def load_ett():
    base = os.path.join(ROOT, "data", "public")
    out = []
    for name in ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]:
        f = os.path.join(base, name)
        if not os.path.exists(f):
            continue
        raw = pd.read_csv(f, parse_dates=["date"])
        chans = [c for c in raw.columns if c != "date"]
        d = raw.melt(id_vars="date", value_vars=chans, var_name="series_id", value_name="y").dropna()
        d = d.rename(columns={"date": "ts"}).sort_values(["series_id", "ts"]).reset_index(drop=True)
        d, feats = _add_features(d, [1, 2, 3, 6, 12, 24], [6, 24])
        d["hour"] = d.ts.dt.hour; d["dow"] = d.ts.dt.dayofweek
        d["h_sin"] = np.sin(2 * np.pi * d.hour / 24); d["h_cos"] = np.cos(2 * np.pi * d.hour / 24)
        feats += ["h_sin", "h_cos", "dow"]
        ts = np.sort(d.ts.unique()); q70, q80 = ts[int(.7 * len(ts))], ts[int(.8 * len(ts))]
        d["split"] = np.where(d.ts <= q70, "train", np.where(d.ts <= q80, "val", "test"))
        season = 24
        out.append(dict(name=name.replace(".csv", ""), df=d, feats=feats, season=season,
                        horizons=[1, 6, 24], block_days=7,
                        baselines={"persistence": ("lag0_self", None), "roll24": ("rm24", None),
                                   "snaive24": ("season", season)}))
    return out


def load_nyiso_1h():
    f = os.path.join(ROOT, "data", "nyiso_1h.parquet")
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f, columns=["zone", "ts", "y", "da", "split"])
    d = d.rename(columns={"zone": "series_id"}).dropna(subset=["y"]).sort_values(["series_id", "ts"])
    d["y"] = d.y.astype("float32"); d["da"] = d.da.astype("float32")
    d = d.reset_index(drop=True)
    d, feats = _add_features(d, [1, 2, 3, 6, 12, 24], [6, 24])
    d["hour"] = d.ts.dt.hour; d["dow"] = d.ts.dt.dayofweek
    d["h_sin"] = np.sin(2 * np.pi * d.hour / 24); d["h_cos"] = np.cos(2 * np.pi * d.hour / 24)
    feats += ["h_sin", "h_cos", "dow", "da"]
    return dict(name="nyiso_1h", df=d, feats=feats, season=24, horizons=[1, 24], block_days=7,
                baselines={"persistence": ("lag0_self", None), "day_ahead": ("col_future", "da")})


# --------------------------------------------------------------------------------------
# baseline construction (all strictly causal or known-in-advance)
# --------------------------------------------------------------------------------------

def build_target_and_baseline(d, h, kind, arg, season):
    """Returns a copy with y_t (target at t+h) and B (baseline for t+h), or None if invalid."""
    x = d.copy()
    x["y_t"] = x.groupby("series_id")["y"].shift(-h)
    if kind == "lag0_self":                       # persistence: last observed value
        x["B"] = x["y"]
    elif kind == "rm7":
        x["B"] = x["rm7"]
    elif kind == "rm24":
        x["B"] = x["rm24"]
    elif kind == "season":                        # seasonal naive at period `arg`
        m = arg * int(np.ceil(h / arg))
        if m == h:                                # D3: this equals persistence -> refuse
            return None
        x["B"] = x.groupby("series_id")["y"].shift(-(h - m))
    elif kind == "col":                           # a column already aligned to t (plan for t+h)
        x["B"] = x.groupby("series_id")[arg].shift(-h)
    elif kind == "col_future":                    # known-in-advance price for delivery at t+h
        x["B"] = x.groupby("series_id")[arg].shift(-h)
    else:
        raise ValueError(kind)
    return x


# --------------------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------------------

def ls_gate(R, rhat):
    """Least-squares gate under squared loss (D5)."""
    den = float(np.dot(rhat, rhat))
    return float(np.dot(R, rhat) / den) if den > 1e-12 else 0.0


def fit_gate_models(F, y_num, y_den, seed):
    n = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(F, y_num)
    dd = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed).fit(F, y_den)
    return n, dd


def regime_labels(x, ref_lo, ref_hi):
    v = x["rs7"].to_numpy() if "rs7" in x.columns else x["rs6"].to_numpy()
    return np.where(v <= ref_lo, 0, np.where(v <= ref_hi, 1, 2))


def run_config(ds, bname, h, seed):
    kind, arg = ds["baselines"][bname]
    x = build_target_and_baseline(ds["df"], h, kind, arg, ds["season"])
    if x is None:
        return None
    F = ds["feats"]
    need = F + ["y_t", "B"]
    x = x.dropna(subset=need)
    tr, va, te = [x[x.split == s] for s in ("train", "val", "test")]
    if min(len(tr), len(va), len(te)) < 400:
        return None
    for f in (tr, va, te):
        f["R"] = f.y_t - f.B

    # corrector (train only)
    sub = min(len(tr), 300_000)
    trs = tr.sample(sub, random_state=seed) if len(tr) > sub else tr
    corr = HGB(max_iter=400, learning_rate=0.06, max_depth=6, random_state=seed,
               l2_regularization=1.0).fit(trs[F], trs.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    Rv, Rt = va.R.to_numpy(), te.R.to_numpy()

    # regime thresholds from TRAIN only
    vcol = "rs7" if "rs7" in F else "rs6"
    lo, hi = np.nanquantile(tr[vcol], [1 / 3, 2 / 3])
    reg_v, reg_t = regime_labels(va, lo, hi), regime_labels(te, lo, hi)
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()

    gates = {}
    gates["rung0_none"] = np.zeros(len(te))
    g1 = ls_gate(Rv, rv)
    gates["rung1_global"] = np.full(len(te), g1)

    g2 = {s: ls_gate(Rv[sid_v == s], rv[sid_v == s]) if (sid_v == s).sum() >= MIN_CELL else g1
          for s in np.unique(sid_v)}
    gates["rung2_series"] = np.array([g2.get(s, g1) for s in sid_t])

    g3 = {}
    for s in np.unique(sid_v):
        for r_ in range(N_REGIMES):
            k = (sid_v == s) & (reg_v == r_)
            g3[(s, r_)] = ls_gate(Rv[k], rv[k]) if k.sum() >= MIN_CELL else g2.get(s, g1)
    gates["rung3_cell"] = np.array([g3.get((s, r_), g2.get(s, g1)) for s, r_ in zip(sid_t, reg_t)])

    num, den = fit_gate_models(va[F].to_numpy(), Rv * rv, rv ** 2, seed)
    g5 = num.predict(te[F].to_numpy()) / np.maximum(den.predict(te[F].to_numpy()), 1e-8)
    gates["rung5_instance"] = g5

    # bootstrap variance of the per-instance gate -> empirical-Bayes shrinkage (rung 4)
    Fva, Fte = va[F].to_numpy(), te[F].to_numpy()
    boots = np.empty((N_BOOT, len(te)))
    for b in range(N_BOOT):
        rb = np.random.default_rng(1000 * seed + b)
        idx = rb.integers(0, len(Fva), len(Fva))
        nb, db = fit_gate_models(Fva[idx], (Rv * rv)[idx], (rv ** 2)[idx], seed * 13 + b)
        boots[b] = nb.predict(Fte) / np.maximum(db.predict(Fte), 1e-8)
    g_bag = boots.mean(axis=0)
    V = float(np.mean(boots.var(axis=0, ddof=1)))
    W = float(np.var(g_bag))
    lam = W / (W + V) if (W + V) > 0 else 0.0
    gates["rung4_shrunk"] = lam * g_bag + (1 - lam) * gates["rung3_cell"]
    gates["rung5b_bagged"] = g_bag
    # oracle per-instance upper bound (uses test residuals; labelled as an oracle everywhere)
    gates["oracle_instance"] = np.where(np.abs(rt) > 1e-9, Rt / np.where(np.abs(rt) > 1e-9, rt, 1), 0.0)

    rows = []
    base_sq = Rt ** 2
    base_ab = np.abs(Rt)
    ts_te = te.ts.to_numpy()
    for rung, g in gates.items():
        frac_clipped = float(np.mean((g < 0) | (g > CLIP)))
        gc = np.clip(g, 0.0, CLIP)                       # D6: identical clipping for all rungs
        err = Rt - gc * rt
        sq, ab = err ** 2, np.abs(err)
        macro_sq, macro_ab = [], []
        for s in np.unique(sid_t):
            k = sid_t == s
            macro_sq.append(1 - sq[k].mean() / base_sq[k].mean())
            macro_ab.append(1 - ab[k].mean() / base_ab[k].mean())
        rows.append(dict(
            dataset=ds["name"], baseline=bname, h=h, seed=seed, rung=rung,
            skill_sq_macro=float(np.mean(macro_sq)), skill_sq_pooled=float(1 - sq.mean() / base_sq.mean()),
            skill_mae_macro=float(np.mean(macro_ab)), skill_mae_pooled=float(1 - ab.mean() / base_ab.mean()),
            degraded_series=int(sum(1 for v in macro_sq if v < 0)), n_series=len(macro_sq),
            frac_clipped=frac_clipped, n_test=int(len(te)), n_val=int(len(va)),
            g_global=g1, var_est=V, var_within=W, lam=lam,
            resid_R2_test=float(1 - np.mean((Rt - rt) ** 2) / np.var(Rt)),
            base_mae=float(base_ab.mean()), base_mse=float(base_sq.mean()),
        ))
    # per-observation squared errors for the block bootstrap, keyed by rung
    err_store = {r: (Rt - np.clip(g, 0, CLIP) * rt) for r, g in gates.items()}
    return rows, dict(ts=ts_te, sid=sid_t, base=Rt, errs=err_store, block_days=ds["block_days"])


# --------------------------------------------------------------------------------------
# block bootstrap over time (W2)
# --------------------------------------------------------------------------------------

def block_bootstrap_diff(store, rung_a, rung_b, n_boot=400, seed=0):
    """CI for the difference in pooled squared-loss skill between two rungs, resampling time blocks."""
    ts = pd.to_datetime(store["ts"])
    block = (ts - ts.min()).days // store["block_days"]
    blocks = np.unique(block)
    ea, eb, base = store["errs"][rung_a] ** 2, store["errs"][rung_b] ** 2, store["base"] ** 2
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        m = np.concatenate([np.flatnonzero(block == p) for p in pick])
        out.append((1 - ea[m].mean() / base[m].mean()) - (1 - eb[m].mean() / base[m].mean()))
    o = np.asarray(out)
    return float(o.mean()), float(np.quantile(o, 0.025)), float(np.quantile(o, 0.975))


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def main(which):
    datasets = []
    if which in ("india", "all"):
        datasets.append(load_india())
    if which in ("ett", "all"):
        datasets += load_ett()
    if which in ("nyiso", "all"):
        n = load_nyiso_1h()
        if n is not None:
            datasets.append(n)
    all_rows, boot_rows = [], []
    t0 = time.time()
    for ds in datasets:
        for bname in ds["baselines"]:
            for h in ds["horizons"]:
                stores = None
                for seed in range(N_SEEDS):
                    res = run_config(ds, bname, h, seed)
                    if res is None:
                        if seed == 0:
                            print(f"  skip {ds['name']} {bname} h={h} (invalid or too small)", flush=True)
                        break
                    rows, store = res
                    all_rows.append(pd.DataFrame(rows))
                    if seed == 0:
                        stores = store
                if stores is not None:
                    for a in ["rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]:
                        m, lo, hi = block_bootstrap_diff(stores, a, "rung3_cell")
                        boot_rows.append(dict(dataset=ds["name"], baseline=bname, h=h,
                                              rung=a, vs="rung3_cell", diff=m, lo=lo, hi=hi))
                    cur = pd.concat(all_rows).query("dataset==@ds['name'] and baseline==@bname and h==@h")
                    piv = cur.groupby("rung").skill_sq_macro.mean()
                    print(f"{ds['name']:10s} {bname:18s} h={h:2d} | " +
                          " ".join(f"{r.split('_')[0]}={piv.get(r, float('nan')):+.4f}" for r in
                                   ["rung0_none", "rung1_global", "rung2_series", "rung3_cell",
                                    "rung4_shrunk", "rung5_instance", "oracle_instance"]) +
                          f" | lam={cur.lam.mean():.2f} rho={cur.var_within.mean()/max(cur.var_est.mean(),1e-9):.2f}"
                          f" [{time.time()-t0:.0f}s]", flush=True)
    if all_rows:
        R = pd.concat(all_rows, ignore_index=True)
        R.to_csv(os.path.join(OUT, f"M1_runs_{which}.csv"), index=False)
        pd.DataFrame(boot_rows).to_csv(os.path.join(OUT, f"M1_blockboot_{which}.csv"), index=False)
        json.dump(dict(clip=CLIP, n_seeds=N_SEEDS, n_boot=N_BOOT, n_regimes=N_REGIMES,
                       min_cell=MIN_CELL, unit="one (dataset,baseline,h,seed) run",
                       primary_metric="skill_sq_macro", secondary=["skill_sq_pooled", "skill_mae_macro",
                                                                    "skill_mae_pooled"]),
                  open(os.path.join(OUT, "M1_canonical_definitions.json"), "w"), indent=2)
        print(f"\nwrote {len(R)} rows -> {OUT}")
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
