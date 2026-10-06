"""RE-FUSED-6 / M5c: PRE-REGISTERED test of parsimony-rule rung selection (defect D14).

Registered before any result was seen (13 Sep 2026, 01:20, docs/08_defect_lineage.md, D14):

  H-M5c  The nested-split argmin over-selects under noise (loses to a fixed coarse default at
         n_val=300; worst-case regret 0.065). A parsimony rule - choose the COARSEST rung whose
         validation-2 risk is not significantly worse than the best rung, tested on TIME BLOCKS at
         alpha = 0.10 (fixed a priori, same alpha as M7) - should (i) remove the small-validation
         loss, (ii) shrink the worst-case tail, and (iii) not lose materially at large validation.

  Falsified if, on REAL data, parsimony has higher mean regret than the nested argmin, or does not
  reduce worst-case (p95) regret, or loses significantly to the fixed per-series default.

Parts
  A  synthetic grid (same generator as M3/M5b, 5 seeds)  - replication of the phenomenon
  B  real data: India daily (4 baselines x horizons) and ETTh1/ETTh2 (persistence)  - FRESH test

Selectors (all validation-only):  nested_argmin | nested_parsimony | fixed_{none,global,series,cell,
shrunk,instance}.  Oracle rung is chosen with test risk and is used ONLY to measure regret.
The rung ORDER below defines "coarser"; shrinkage sits between per-cell and per-instance.
"""
import os, sys, time, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m35 = _load("m35", "m3_m5_ladder.py")
m1 = _load("m1c", "m1_canonical.py")

OUT = r"D:\\REFUSED5\results\refused6"
ORDER = ["rung0_none", "rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]
CLIP, ALPHA, N_BOOT, MIN_CELL, Z_CLIP = 1.5, 0.10, 3, 30, 10.0


# ----------------------------------------------------------------- shared selection machinery
def ls_gate(R, r):
    den = float(np.dot(r, r))
    return float(np.dot(R, r) / den) if den > 1e-12 else 0.0


def wls_fit(F, R, r, seed):
    w = r ** 2
    ok = w > 1e-12
    z = np.clip(R[ok] / r[ok], -Z_CLIP, Z_CLIP)
    return HGB(max_iter=150, learning_rate=0.08, max_depth=5, random_state=seed).fit(F[ok], z, sample_weight=w[ok])


def fit_rungs(F, R, r, sid, cid, seed):
    """Fit every rung's gate on (F, R, r); returns a predictor g(F_new, sid_new, cid_new) per rung."""
    g1 = ls_gate(R, r)
    g2 = {s: (ls_gate(R[sid == s], r[sid == s]) if (sid == s).sum() >= MIN_CELL else g1) for s in np.unique(sid)}
    g3 = {c: (ls_gate(R[cid == c], r[cid == c]) if (cid == c).sum() >= MIN_CELL else np.nan) for c in np.unique(cid)}
    w0 = wls_fit(F, R, r, seed)
    boots = [wls_fit(F[i], R[i], r[i], seed * 13 + b)
             for b, i in enumerate(np.random.default_rng(seed).integers(0, len(F), (N_BOOT, len(F))))]
    Bp = np.vstack([m.predict(F) for m in boots])
    V, W = float(np.mean(Bp.var(axis=0, ddof=1))), float(np.var(Bp.mean(axis=0)))
    lam = W / (W + V) if (W + V) > 0 else 0.0

    def predict(Fn, sn, cn):
        a2 = np.array([g2.get(s, g1) for s in sn])
        a3 = np.array([g3.get(c, np.nan) for c in cn])
        a3 = np.where(np.isnan(a3), a2, a3)
        bag = np.vstack([m.predict(Fn) for m in boots]).mean(axis=0)
        return {"rung0_none": np.zeros(len(Fn)), "rung1_global": np.full(len(Fn), g1),
                "rung2_series": a2, "rung3_cell": a3,
                "rung4_shrunk": lam * bag + (1 - lam) * a2, "rung5_instance": w0.predict(Fn)}
    return predict


def sq_err(R, g, r):
    return (R - np.clip(g, 0, CLIP) * r) ** 2


def select(err_v2, blocks):
    risk = {k: float(err_v2[k].mean()) for k in ORDER}
    best = min(ORDER, key=risk.get)
    pars = best
    for k in ORDER[:ORDER.index(best)]:              # coarsest first
        d = pd.Series(err_v2[k] - err_v2[best]).groupby(blocks).mean().to_numpy()
        if len(d) < 4:
            pars = k                                  # too little evidence to justify refinement
            break
        sd = d.std(ddof=1)
        if sd <= 0:
            pars = k
            break
        p = 1 - stats.t.cdf(d.mean() / (sd / np.sqrt(len(d))), len(d) - 1)   # H1: k is worse
        if p >= ALPHA:                                # coarser rung not significantly worse
            pars = k
            break
    return best, pars


def record(tag, dataset, config, seed, test_err, err_v2, blocks, base_te, extra=None):
    test_risk = {k: float(test_err[k].mean()) for k in ORDER}
    oracle = min(ORDER, key=test_risk.get)
    best, pars = select(err_v2, blocks)
    row = dict(part=tag, dataset=dataset, config=config, seed=seed, oracle_rung=oracle,
               sel_nested_argmin=best, sel_nested_parsimony=pars, n_blocks=int(len(np.unique(blocks))))
    for nm, pick in [("nested_argmin", best), ("nested_parsimony", pars)]:
        row[f"regret_{nm}"] = (test_risk[pick] - test_risk[oracle]) / base_te
        row[f"correct_{nm}"] = int(pick == oracle)
    for k in ORDER:
        row[f"regret_fixed_{k.split('_', 1)[1]}"] = (test_risk[k] - test_risk[oracle]) / base_te
        row[f"skill_{k}"] = 1 - test_risk[k] / base_te
    if extra:
        row.update(extra)
    return row


# ----------------------------------------------------------------- Part A: synthetic
def part_a(seeds=5):
    rows = []
    for i, k in enumerate(m35.grid("full")):
        k = dict(k); name = k.pop("name", None)
        for s in range(seeds):
            p, tr, va, te = m35.simulate(k, s)
            corr = HGB(max_iter=200, learning_rate=0.08, max_depth=6, random_state=s).fit(tr["F"], tr["R"])
            rv, rt = corr.predict(va["F"]), corr.predict(te["F"])
            order = np.argsort(va["t"]); h1, h2 = order[: len(order) // 2], order[len(order) // 2:]
            pred1 = fit_rungs(va["F"][h1], va["R"][h1], rv[h1], va["sid"][h1], va["cid"][h1], s)
            gv2 = pred1(va["F"][h2], va["sid"][h2], va["cid"][h2])
            err_v2 = {kk: sq_err(va["R"][h2], gv2[kk], rv[h2]) for kk in ORDER}
            blocks = np.arange(len(h2)) // 50
            predf = fit_rungs(va["F"], va["R"], rv, va["sid"], va["cid"], s)
            gt = predf(te["F"], te["sid"], te["cid"])
            test_err = {kk: sq_err(te["R"], gt[kk], rt) for kk in ORDER}
            rows.append(record("A_synthetic", "synthetic", name or ",".join(f"{a}={b}" for a, b in k.items()),
                               s, test_err, err_v2, blocks, float(np.mean(te["R"] ** 2)),
                               extra=dict(n_val=p["n_val"])))
        d = pd.DataFrame(rows).tail(seeds)
        print(f"[A {i+1}] n_val={int(d.n_val.iloc[0]):6d} argmin={d.regret_nested_argmin.mean():+.5f} "
              f"parsimony={d.regret_nested_parsimony.mean():+.5f} fixed_cell={d.regret_fixed_cell.mean():+.5f}", flush=True)
    return rows


# ----------------------------------------------------------------- Part B: real data (fresh test)
def real_one(ds, bname, h, seed, block_days):
    kind, arg = ds["baselines"][bname]
    x = m1.build_target_and_baseline(ds["df"], h, kind, arg, ds["season"])
    if x is None:
        return None
    F = ds["feats"]
    x = x.dropna(subset=F + ["y_t", "B"])
    tr, va, te = [x[x.split == s] for s in ("train", "val", "test")]
    if min(len(tr), len(va), len(te)) < 400:
        return None
    for f in (tr, va, te):
        f["R"] = f.y_t - f.B
    sub = min(len(tr), 200_000)
    trs = tr.sample(sub, random_state=seed) if len(tr) > sub else tr
    corr = HGB(max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed, l2_regularization=1.0).fit(trs[F], trs.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    vcol = "rs7" if "rs7" in F else "rs6"
    lo, hi = np.nanquantile(tr[vcol], [1 / 3, 2 / 3])
    codes = {s: i for i, s in enumerate(sorted(x.series_id.unique()))}
    def ids(df):
        sid = df.series_id.map(codes).to_numpy()
        return sid, sid * 3 + np.digitize(df[vcol].to_numpy(), [lo, hi])
    sid_v, cid_v = ids(va); sid_t, cid_t = ids(te)
    Fv, Ft, Rv, Rt = va[F].to_numpy(), te[F].to_numpy(), va.R.to_numpy(), te.R.to_numpy()
    tsv = pd.to_datetime(va.ts).to_numpy()
    cut = np.sort(np.unique(tsv))[len(np.unique(tsv)) // 2]
    h1, h2 = np.flatnonzero(tsv < cut), np.flatnonzero(tsv >= cut)
    # D9 reparameterisation, fitted on the data each phase is allowed to see
    g1a = ls_gate(Rv[h1], rv[h1]); rv1 = rv * g1a
    pred1 = fit_rungs(Fv[h1], Rv[h1], rv1[h1], sid_v[h1], cid_v[h1], seed)
    gv2 = pred1(Fv[h2], sid_v[h2], cid_v[h2])
    err_v2 = {k: sq_err(Rv[h2], gv2[k], rv1[h2]) for k in ORDER}
    days = (pd.to_datetime(va.ts.to_numpy()[h2]) - pd.Timestamp(cut)).days
    blocks = np.asarray(days) // block_days
    g1f = ls_gate(Rv, rv); rvf, rtf = rv * g1f, rt * g1f
    predf = fit_rungs(Fv, Rv, rvf, sid_v, cid_v, seed)
    gt = predf(Ft, sid_t, cid_t)
    test_err = {k: sq_err(Rt, gt[k], rtf) for k in ORDER}
    return record("B_real", ds["name"], f"{bname}_h{h}", seed, test_err, err_v2, blocks, float(np.mean(Rt ** 2)),
                  extra=dict(baseline=bname, h=h, n_val=int(len(va))))


def part_b(seeds=5):
    rows = []
    india = m1.load_india()
    jobs = [(india, b, h, 7) for b in ["persistence", "roll7", "snaive7", "operator_schedule"] for h in [1, 2, 3, 7]]
    for e in m1.load_ett():
        if e["name"] in ("ETTh1", "ETTh2"):
            jobs += [(e, "persistence", h, 3) for h in [1, 24]]
    for ds, b, h, bd in jobs:
        got = []
        for s in range(seeds):
            r = real_one(ds, b, h, s, bd)
            if r is None:
                break
            got.append(r)
        if got:
            rows += got
            d = pd.DataFrame(got)
            print(f"[B] {ds['name']:11s} {b:18s} h={h:2d} argmin={d.regret_nested_argmin.mean():+.5f} "
                  f"parsimony={d.regret_nested_parsimony.mean():+.5f} fixed_series={d.regret_fixed_series.mean():+.5f} "
                  f"fixed_cell={d.regret_fixed_cell.mean():+.5f} oracle={d.oracle_rung.mode().iloc[0]}", flush=True)
    return rows


def summarise(R, label):
    print(f"\n==================== {label}: {len(R)} runs ====================")
    cols = [c for c in R.columns if c.startswith("regret_")]
    print(R[cols].mean().sort_values().round(5).to_string())
    print("\nexact oracle recovery: argmin=%.3f parsimony=%.3f" % (R.correct_nested_argmin.mean(), R.correct_nested_parsimony.mean()))
    for a, b in [("regret_nested_parsimony", "regret_nested_argmin"), ("regret_nested_parsimony", "regret_fixed_series"),
                 ("regret_nested_parsimony", "regret_fixed_cell"), ("regret_nested_argmin", "regret_fixed_series")]:
        d = R[a] - R[b]
        nz = (d != 0).sum()
        p = stats.wilcoxon(R[a], R[b]).pvalue if nz > 5 else float("nan")
        print(f"  {a.replace('regret_',''):18s} - {b.replace('regret_',''):16s} mean={d.mean():+.5f} lower_in={int((d<0).sum())}/{len(d)} wilcoxon_p={p:.2e}")
    for c in ["regret_nested_argmin", "regret_nested_parsimony", "regret_fixed_series", "regret_fixed_cell"]:
        print(f"  tail {c.replace('regret_',''):18s} p95={R[c].quantile(.95):.5f} max={R[c].max():.5f}")


def main(parts):
    t0 = time.time()
    allr = []
    if "A" in parts:
        a = part_a(); allr += a
        summarise(pd.DataFrame(a), "PART A synthetic")
    if "B" in parts:
        b = part_b(); allr += b
        summarise(pd.DataFrame(b), "PART B real data (fresh test of H-M5c)")
        B = pd.DataFrame(b)
        verdict = []
        verdict.append(("mean regret: parsimony <= argmin", B.regret_nested_parsimony.mean() <= B.regret_nested_argmin.mean()))
        verdict.append(("p95 regret: parsimony < argmin", B.regret_nested_parsimony.quantile(.95) < B.regret_nested_argmin.quantile(.95)))
        d = B.regret_nested_parsimony - B.regret_fixed_series
        p = stats.wilcoxon(B.regret_nested_parsimony, B.regret_fixed_series).pvalue if (d != 0).sum() > 5 else 1.0
        verdict.append(("not significantly worse than fixed per-series", not (d.mean() > 0 and p < 0.05)))
        print("\n=== PRE-REGISTERED VERDICT FOR H-M5c ===")
        for k, v in verdict:
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print("  H-M5c", "SURVIVES" if all(v for _, v in verdict) else "FALSIFIED")
    pd.DataFrame(allr).to_csv(os.path.join(OUT, "M5c_parsimony.csv"), index=False)
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AB")
