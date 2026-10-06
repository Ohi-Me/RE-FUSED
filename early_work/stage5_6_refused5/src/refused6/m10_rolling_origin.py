"""RE-FUSED-6 / M10: rolling-origin re-evaluation of the India ladder (reviewer objection 2.1).

Every India conclusion so far used one chronological split (test 2024-01 .. 2025-04). Here the identical
protocol (D8 WLS estimator, D9 reparameterisation, identical clip, squared-loss macro skill) is re-run at
three origins:
    origin_2022   train <= 2020-12-31, val = 2021, test = 2022
    origin_2023   train <= 2021-12-31, val = 2022, test = 2023
    origin_2024   train <= 2022-12-31, val = 2023, test = 2024-01 .. 2025-04   (canonical)
Question: is the ordering "coarse rung best, per-instance never best" stable across test years?
"""
import os, time, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1c", os.path.join(HERE, "m1_canonical.py"))
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)
OUT = r"D:\\REFUSED5\results\refused6"
CLIP, MIN_CELL, N_BOOT, Z_CLIP = 1.5, 30, 3, 10.0
ORIGINS = {"origin_2022": ("2020-12-31", "2021-12-31", "2022-12-31"),
           "origin_2023": ("2021-12-31", "2022-12-31", "2023-12-31"),
           "origin_2024": ("2022-12-31", "2023-12-31", "2025-12-31")}


def wls(F, R, r, seed):
    w = r ** 2; ok = w > 1e-12
    return HGB(max_iter=150, learning_rate=0.08, max_depth=5, random_state=seed).fit(
        F[ok], np.clip(R[ok] / r[ok], -Z_CLIP, Z_CLIP), sample_weight=w[ok])


def origin_df(ds, origin):
    tr_end, va_end, te_end = ORIGINS[origin]
    base_df = ds["df"].copy()
    base_df["split"] = np.where(base_df.ts <= tr_end, "train",
                                np.where(base_df.ts <= va_end, "val", np.where(base_df.ts <= te_end, "test", "drop")))
    return base_df


def run(ds, origin, bname, h, seed):
    base_df = origin_df(ds, origin)
    kind, arg = ds["baselines"][bname]
    x = m1.build_target_and_baseline(base_df, h, kind, arg, ds["season"])
    if x is None:
        return None
    F = ds["feats"]
    x = x.dropna(subset=F + ["y_t", "B"])
    tr, va, te = [x[x.split == s] for s in ("train", "val", "test")]
    if min(len(tr), len(va), len(te)) < 400:
        return None
    for f in (tr, va, te):
        f["R"] = f.y_t - f.B
    corr = HGB(max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed, l2_regularization=1.0).fit(tr[F], tr.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    Rv, Rt = va.R.to_numpy(), te.R.to_numpy()
    g1 = m1.ls_gate(Rv, rv); rv, rt = g1 * rv, g1 * rt
    g1e = m1.ls_gate(Rv, rv)
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()
    lo, hi = np.nanquantile(tr.rs7, [1 / 3, 2 / 3])
    qv, qt = np.digitize(va.rs7.to_numpy(), [lo, hi]), np.digitize(te.rs7.to_numpy(), [lo, hi])
    g2m = {s: (m1.ls_gate(Rv[sid_v == s], rv[sid_v == s]) if (sid_v == s).sum() >= MIN_CELL else g1e) for s in np.unique(sid_v)}
    g2 = np.array([g2m.get(s, g1e) for s in sid_t])
    g3m = {}
    for s in np.unique(sid_v):
        for q in range(3):
            k = (sid_v == s) & (qv == q)
            g3m[(s, q)] = m1.ls_gate(Rv[k], rv[k]) if k.sum() >= MIN_CELL else g2m.get(s, g1e)
    g3 = np.array([g3m.get((s, q), g1e) for s, q in zip(sid_t, qt)])
    Fv, Ft = va[F].to_numpy(), te[F].to_numpy()
    bw = np.vstack([wls(Fv[i], Rv[i], rv[i], seed * 11 + b).predict(Ft)
                    for b, i in enumerate(np.random.default_rng(seed).integers(0, len(Fv), (N_BOOT, len(Fv))))])
    g5 = bw.mean(axis=0)
    V, W = float(np.mean(bw.var(axis=0, ddof=1))), float(np.var(g5))
    lam = W / (W + V) if (W + V) > 0 else 0.0
    gates = {"rung0_none": np.zeros(len(te)), "rung1_global": np.full(len(te), g1e), "rung2_series": g2,
             "rung3_cell": g3, "rung4_shrunk": lam * g5 + (1 - lam) * g3, "rung5_instance": g5}
    base = Rt ** 2
    rows = []
    for rung, g in gates.items():
        sq = (Rt - np.clip(g, 0, CLIP) * rt) ** 2
        msq = [1 - sq[sid_t == s].mean() / base[sid_t == s].mean() for s in np.unique(sid_t)]
        rows.append(dict(origin=origin, baseline=bname, h=h, seed=seed, rung=rung,
                         skill_sq_macro=float(np.mean(msq)), n_test=len(te), n_val=len(va)))
    return rows


def main():
    ds = m1.load_india()
    rows, t0 = [], time.time()
    for origin in ORIGINS:
        for b in ["persistence", "roll7", "snaive7", "operator_schedule"]:
            for h in [1, 3]:
                for seed in range(5):
                    r = run(ds, origin, b, h, seed)
                    if r:
                        rows += r
                d = pd.DataFrame(rows)
                d = d[(d.origin == origin) & (d.baseline == b) & (d.h == h)]
                if len(d):
                    m = d.groupby("rung").skill_sq_macro.mean()
                    print(f"{origin} {b:18s} h={h} | " + " ".join(f"{k.split('_')[0]}={m[k]:+.4f}" for k in m.index) +
                          f" | best={m.idxmax()} [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "M10_rolling_origin.csv"), index=False)
    print(f"\nwrote {len(R)} rows")
    piv = R.pivot_table(index=["origin", "baseline", "h"], columns="rung", values="skill_sq_macro")
    best = piv.idxmax(axis=1)
    print("\n=== best rung per (origin, baseline, h) ===")
    print(best.to_string())
    print("\n=== best-rung counts by origin ===")
    print(best.groupby(level=0).value_counts().to_string())
    print("\nper-instance best in %d/%d configs" % ((best == "rung5_instance").sum(), len(best)))
    fine = piv["rung5_instance"]
    coarse = piv[["rung1_global", "rung2_series", "rung3_cell"]].max(axis=1)
    gap = fine - coarse
    print("fine - best coarse: mean=%+.4f, fine loses in %d/%d configs (binomial p=%.4f)" % (
        gap.mean(), int((gap < 0).sum()), len(gap),
        stats.binomtest(int((gap < 0).sum()), len(gap), 0.5, alternative="greater").pvalue))
    print(gap.groupby(level=0).agg(["mean", lambda s: int((s < 0).sum()), "size"]).to_string())
    # pre-registered verdict (docs/08, H-M10)
    print("\n=== PRE-REGISTERED VERDICT FOR H-M10 ===")
    ok_all = True
    for o in ORIGINS:
        if o not in best.index.get_level_values(0):
            continue
        b_o, g_o = best.loc[o], gap.loc[o]
        c1 = int((b_o == "rung5_instance").sum()) <= 1
        c2 = int((g_o < 0).sum()) > len(g_o) / 2
        ok_all &= (c1 and c2)
        print(f"  {o}: {'PASS' if c1 else 'FAIL'} (i) instance best in {int((b_o=='rung5_instance').sum())}/{len(b_o)} <= 1;"
              f" {'PASS' if c2 else 'FAIL'} (ii) instance below best coarse in {int((g_o<0).sum())}/{len(g_o)} > half")
    print("  H-M10", "SURVIVES (ordering stable across test years)" if ok_all else "FALSIFIED in at least one origin")

    # ---- part 2: H-M10b - does validation-based rung selection beat the fixed per-series default? ----
    spec5 = importlib.util.spec_from_file_location("m5c", os.path.join(HERE, "m5c_parsimony_selection.py"))
    m5c = importlib.util.module_from_spec(spec5); spec5.loader.exec_module(m5c)
    srows = []
    for origin in ORIGINS:
        ds_o = dict(ds, df=origin_df(ds, origin))
        for b in ["persistence", "roll7", "snaive7", "operator_schedule"]:
            for h in [1, 3]:
                for seed in range(5):
                    r = m5c.real_one(ds_o, b, h, seed, 7)
                    if r:
                        r["origin"] = origin
                        r["fresh"] = origin != "origin_2024"
                        srows.append(r)
        d = pd.DataFrame(srows); d = d[d.origin == origin]
        print(f"[sel] {origin} argmin={d.regret_nested_argmin.mean():.4f} parsimony={d.regret_nested_parsimony.mean():.4f} "
              f"fixed_series={d.regret_fixed_series.mean():.4f} fixed_cell={d.regret_fixed_cell.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(OUT, "M10b_selection_rolling.csv"), index=False)
    cols = [c for c in S.columns if c.startswith("regret_")]
    print("\n=== selection regret by origin (origin_2024 overlaps M5c Part B: in-sample) ===")
    print(S.groupby("origin")[cols].mean().round(4).to_string())
    print("\n=== PRE-REGISTERED VERDICT FOR H-M10b (fresh origins 2022, 2023) ===")
    ok_b = True
    for origin in ["origin_2022", "origin_2023"]:
        P = S[S.origin == origin]
        if not len(P):
            continue
        for sel in ["regret_nested_argmin", "regret_nested_parsimony"]:
            d = P[sel] - P.regret_fixed_series
            p_sel = stats.wilcoxon(P[sel], P.regret_fixed_series, alternative="less").pvalue if (d != 0).sum() > 5 else 1.0
            c = p_sel >= 0.05
            ok_b &= c
            print(f"  {origin} {'PASS' if c else 'FAIL'} {sel[7:]} not significantly better than fixed_series "
                  f"(mean diff {d.mean():+.4f}, p[selector better]={p_sel:.3f})")
    F_ = S[S.fresh]
    c_mean = (F_.regret_fixed_series.mean() <= F_.regret_nested_argmin.mean()) and (F_.regret_fixed_series.mean() <= F_.regret_nested_parsimony.mean())
    ok_b &= c_mean
    print(f"  {'PASS' if c_mean else 'FAIL'} pooled fresh: fixed_series mean regret {F_.regret_fixed_series.mean():.4f} <= argmin "
          f"{F_.regret_nested_argmin.mean():.4f} and parsimony {F_.regret_nested_parsimony.mean():.4f}")
    print("  H-M10b", "SURVIVES (validation-based rung selection does not pay; fixed per-series default)" if ok_b
          else "FALSIFIED (rung selection pays on at least one fresh origin)")
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
