"""X1: semi-synthetic phase diagrams for the persistence-adjusted crossing condition (09_theory T3, T4, T8).

Substrate: NYISO zonal load, ISO baseline, development split — real features, a real trained corrector
prediction r, and real residual noise whose sign is flipped per (zone, day) (DV2: removes systematic correlation
with r while keeping scale, tails, heteroskedasticity and within-day dependence; independent val/test draws).

Synthetic residuals with KNOWN, gradually drifting gate heterogeneity:
    R_syn(t) = (0.75 + tau * h_t(z)) * r + kappa * eps,     h_t = cos(w t) h_A + sin(w t) h_B
h_A, h_B have unit signal-weighted energy, are mutually orthogonal and orthogonal to the coarse partition.
The drift speed w is set by the rotation angle phi over one year (phi = 0: stationary). The base gate 0.75 keeps
gates inside the clip interval [0, 1.5] (DV3).
tau^2 = m * dV, with dV the CLUSTER-ROBUST (zone-day clusters) estimation cost of the fine gates (DV3).
Oracle prediction (T4): 2 <h_V, h_T> - ||h_V||^2 - dV, with h_V, h_T the period-average patterns.

Axes:  unit     global -> per zone           (11 cells)
       state    per zone -> zone x hour       (264 cells)
       instance per zone -> per-instance gate (smooth function of features)
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import data, gates as G, ladder, selection as S  # noqa: E402
from refused_gate.paths import RES  # noqa: E402

BASE_GATE = 0.75


def substrate(seed=0, source="nyiso_dev", clean=False):
    """source: nyiso_dev (development) | opsd_confirm (confirmatory grid; blinded loader, TSO baseline).
    clean: deviation DC1b, OPSD plausibility filter (sensitivity analysis after unblinding)."""
    if source == "nyiso_dev":
        ds = data.nyiso_load("dev")
        x = ds["df"].assign(y_t=ds["df"].y, B=ds["df"].iso_fc)
    elif source == "opsd_confirm":
        ds = data.opsd_load("confirm", "2019", clean=clean)
        units = np.sort(ds["df"].sid.unique())
        keep = np.random.default_rng(11).choice(units, 12, replace=False)       # pre-registered 12-unit substrate
        x = ds["df"][ds["df"].sid.isin(keep)].assign(y_t=lambda f: f.y, B=lambda f: f.tso_fc)
    elif source in ("india_daily_dev", "india_daily_confirm"):
        # 34 Indian State control areas, daily drawal (T2), seasonal-naive forecast; the finer "state" axis is
        # area x day of week (238 cells), stored in the column the grid code reads as "hour"
        ds = data.india_state_daily("T2", "long")
        x = ds["df"].assign(y_t=ds["df"].y, B=ds["df"].y_lag7)
        x["hour"] = x["dow"]
    else:
        raise ValueError(source)
    x = x.dropna(subset=ds["feats"] + ["y_t", "B"]).copy()
    x["R"] = x.y_t - x.B
    x["day"] = x.ts.dt.normalize()
    tr, va, te = (x[x.split == s].copy() for s in ("train", "val", "test"))
    corr = ladder._fit_corrector(tr[ds["feats"]].to_numpy(np.float32), tr.R.to_numpy(), ladder.LadderConfig(), seed)
    for f in (va, te):
        f["r"] = corr.predict(f[ds["feats"]].to_numpy(np.float32))
    g1 = G.ls_gate(va.R.to_numpy(), va.r.to_numpy())
    va["r"] *= g1
    te["r"] *= g1
    gs = G.PartitionGate(30).fit(va.R.to_numpy(), va.r.to_numpy(), va.sid.to_numpy())
    rng = np.random.default_rng(seed + 7777)
    t0 = va.day.min()
    for f in (va, te):
        raw = (f.R - gs.predict(f.sid.to_numpy()) * f.r).to_numpy()
        cl = pd.factorize(f.sid.astype(str) + "|" + f.day.astype(str))[0]
        f["cluster"] = cl
        f["eps"] = raw * rng.choice([-1.0, 1.0], size=cl.max() + 1)[cl]
        f["tday"] = (f.day - t0).dt.days.astype(float)
    return va, te, ds["feats"]


def _orth(vecs, w, coarse):
    """Weighted Gram-Schmidt of cell-valued patterns (evaluated per row), orthogonal to coarse-cell constants."""
    out = []
    for v in vecs:
        v = v.copy()
        for c in np.unique(coarse):
            m = coarse == c
            v[m] -= np.sum(w[m] * v[m]) / np.sum(w[m])
        for u in out:
            v -= np.sum(w * v * u) / np.sum(w * u * u) * u
        v /= np.sqrt(np.mean(w * v ** 2))
        out.append(v)
    return out


def cell_patterns(keys_v, keys_t, rv, coarse_v, coarse_t, rng):
    uniq = np.unique(np.concatenate([keys_v, keys_t]))
    A, B = dict(zip(uniq, rng.normal(size=len(uniq)))), dict(zip(uniq, rng.normal(size=len(uniq))))
    av, bv = np.array([A[k] for k in keys_v]), np.array([B[k] for k in keys_v])
    at, bt = np.array([A[k] for k in keys_t]), np.array([B[k] for k in keys_t])
    return _project(av, bv, at, bt, rv, coarse_v, coarse_t)


def feature_patterns(Fv, Ft, rv, coarse_v, coarse_t, rng):
    mu, sd = Fv.mean(0), Fv.std(0) + 1e-9
    k = Fv.shape[1]
    ba, bb = rng.normal(size=k), rng.normal(size=k)
    f = lambda F, b: np.tanh(((F - mu) / sd) @ b / np.sqrt(k))
    return _project(f(Fv, ba), f(Fv, bb), f(Ft, ba), f(Ft, bb), rv, coarse_v, coarse_t)


def _project(av, bv, at, bt, rv, coarse_v, coarse_t):
    """Orthogonalise on validation weights and apply the same linear map to the test-row values."""
    w = rv ** 2
    means_a = {c: np.sum(w[coarse_v == c] * av[coarse_v == c]) / np.sum(w[coarse_v == c]) for c in np.unique(coarse_v)}
    means_b = {c: np.sum(w[coarse_v == c] * bv[coarse_v == c]) / np.sum(w[coarse_v == c]) for c in np.unique(coarse_v)}
    av = av - np.array([means_a[c] for c in coarse_v])
    at = at - np.array([means_a.get(c, 0.0) for c in coarse_t])
    bv = bv - np.array([means_b[c] for c in coarse_v])
    bt = bt - np.array([means_b.get(c, 0.0) for c in coarse_t])
    sa = np.sqrt(np.mean(w * av ** 2))
    av, at = av / sa, at / sa
    proj = np.mean(w * bv * av)
    bv, bt = bv - proj * av, bt - proj * at
    sb = np.sqrt(np.mean(w * bv ** 2))
    return av, bv / sb, at, bt / sb


def clustered_cost(R, r, keys, gvals, clusters):
    """Sum over cells of w(c) * cluster-robust Var(g_c), per-row MSE units."""
    n = len(R)
    e = R - gvals * r
    tot = 0.0
    df = pd.DataFrame({"k": keys, "c": clusters, "re": r * e, "rr": r * r})
    for k, g in df.groupby("k"):
        den = g.rr.sum()
        if den <= 0:
            continue
        s = g.groupby("c").re.sum().to_numpy()
        tot += (den / n) * np.sum(s ** 2) / den ** 2
    return tot


def make_run(va, te, cv, ct, fv, ft, patterns, m, phi, kappa, frac, seed, fine="partition", feats=None):
    rng = np.random.default_rng(seed)
    keep_days = np.sort(va.day.unique())
    if frac < 1:
        keep_days = np.sort(rng.choice(keep_days, int(len(keep_days) * frac), replace=False))
    mv = va.day.isin(keep_days).to_numpy()
    rv, rt = va.r.to_numpy(), te.r.to_numpy()
    av, bv, at, bt = patterns
    omega = phi / 365.0
    tv, tt = va.tday.to_numpy(), te.tday.to_numpy()
    hv = np.cos(omega * tv) * av + np.sin(omega * tv) * bv
    ht = np.cos(omega * tt) * at + np.sin(omega * tt) * bt
    eps_v, eps_t = kappa * va.eps.to_numpy(), kappa * te.eps.to_numpy()
    Fv = va[feats].to_numpy(np.float32) if feats else None
    Ft = te[feats].to_numpy(np.float32) if feats else None

    def fit_pair(R, idx):
        gc = G.PartitionGate(30).fit(R[idx], rv[idx], cv[idx])
        if fine == "partition":
            gf = G.PartitionGate(30).fit(R[idx], rv[idx], fv[idx])
            return gc, gf
        return gc, G.InstanceGate(n_bag=1, seed=seed).fit(Fv[idx], R[idx], rv[idx])

    def pred_fine(gc, gf, keys_c, keys_f, F):
        c = gc.predict(keys_c)
        return c, (gf.predict(keys_f, parent_values=c) if fine == "partition" else gf.predict(F))

    # estimation cost at tau = 0
    R0v = BASE_GATE * rv + eps_v
    if fine == "partition":
        gf0 = G.PartitionGate(30).fit(R0v[mv], rv[mv], fv[mv]).predict(fv[mv])
        gc0 = G.PartitionGate(30).fit(R0v[mv], rv[mv], cv[mv]).predict(cv[mv])
        cl = va.cluster.to_numpy()[mv]
        dV = clustered_cost(R0v[mv], rv[mv], fv[mv], gf0, cl) - clustered_cost(R0v[mv], rv[mv], cv[mv], gc0, cl)
    else:
        R0t = BASE_GATE * rt + eps_t
        gc0, gi0 = fit_pair(R0v, mv)
        c0, f0 = pred_fine(gc0, gi0, ct, ft, Ft)
        dV = -float(np.mean(S.realised_gain(R0t, rt, c0, f0)))
    dV = max(dV, 1e-9)
    tau = np.sqrt(max(m, 0.0) * dV)
    Rv = (BASE_GATE + tau * hv) * rv + eps_v
    Rt = (BASE_GATE + tau * ht) * rt + eps_t
    gc, gf = fit_pair(Rv, mv)
    c_t, f_t = pred_fine(gc, gf, ct, ft, Ft)
    realised = float(np.mean(S.realised_gain(Rt, rt, c_t, f_t)))
    # oracle prediction with period-average patterns (validation weights for ||h_V||, test weights for the product)
    wv, wt = rv[mv] ** 2, rt ** 2
    hV_cell = pd.Series(tau * hv[mv]).groupby(fv[mv] if fine == "partition" else np.arange(mv.sum())).mean()
    if fine == "partition":
        hT_cell = pd.Series(tau * ht).groupby(ft).mean()
        hV_on_t = pd.Series(ft).map(hV_cell).fillna(0).to_numpy()
        hT_on_t = pd.Series(ft).map(hT_cell).fillna(0).to_numpy()
        hV_on_v = pd.Series(fv[mv]).map(hV_cell).fillna(0).to_numpy()
        prod = float(np.mean(wt * hV_on_t * hT_on_t))
        energy = float(np.mean(wv * hV_on_v ** 2))
    else:
        # instance: average pattern coefficients over each period, applied to the test rows' feature patterns
        ca_v, cb_v = np.mean(np.cos(omega * tv[mv])), np.mean(np.sin(omega * tv[mv]))
        ca_t, cb_t = np.mean(np.cos(omega * tt)), np.mean(np.sin(omega * tt))
        hV_t = tau * (ca_v * at + cb_v * bt)
        hT_t = tau * (ca_t * at + cb_t * bt)
        prod = float(np.mean(wt * hV_t * hT_t))
        energy = float(np.mean(wv * (tau * (ca_v * av[mv] + cb_v * bv[mv])) ** 2))
    persistence = prod / energy if energy > 0 else np.nan
    res = dict(m=m, phi=phi, kappa=kappa, frac=frac, seed=seed, tau=float(tau), dV=float(dV), realised=realised,
               energy_V=energy, prod_VT=prod, persistence=persistence, oracle_pred=2 * prod - energy - dV)
    days_s = va.day.to_numpy()[mv]
    # PART v2 (DV5): unclipped blocks, G = 2C - ||h_full||^2; both block designs recorded
    if fine == "partition":
        for mode in ("interleaved", "contiguous"):
            res[f"part_{mode}_G"] = S.part2_partition(Rv[mv], rv[mv], cv[mv], fv[mv], days_s, mode, 6, 30)["G"]
    else:
        ev = rng.choice(mv.sum(), min(15000, mv.sum()), replace=False)
        for mode in ("interleaved", "contiguous"):
            res[f"part_{mode}_G"] = S.part2_instance(Fv[mv], Rv[mv], rv[mv], cv[mv], days_s, mode, 4, ev, seed)["G"]
    kd = np.sort(np.unique(days_s))
    h1 = days_s <= kd[len(kd) // 2 - 1]
    idx_s = np.flatnonzero(mv)
    gc1, gf1 = fit_pair(Rv, idx_s[h1])
    c2, f2 = pred_fine(gc1, gf1, cv[idx_s[~h1]], fv[idx_s[~h1]] if fine == "partition" else None,
                       Fv[idx_s[~h1]] if Fv is not None else None)
    res["nested_G"] = float(np.mean(S.realised_gain(Rv[idx_s[~h1]], rv[idx_s[~h1]], c2, f2)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="partition", choices=["partition", "instance"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--substrate_seed", type=int, default=0)
    ap.add_argument("--source", default="nyiso_dev",
                    choices=["nyiso_dev", "opsd_confirm", "india_daily_dev", "india_daily_confirm"])
    ap.add_argument("--seed_offset", type=int, default=100)
    ap.add_argument("--clean", action="store_true", help="deviation DC1b: OPSD plausibility filter (sensitivity only)")
    a = ap.parse_args()
    if a.source.startswith("india"):
        out = os.path.join(RES, "india", "confirm" if a.source.endswith("confirm") else "dev", "x1")
    else:
        out = os.path.join(RES, "dev" if a.source == "nyiso_dev" else ("confirm_sensitivity" if a.clean else "confirm"),
                           "x1")
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, f"x1_{a.part}_{a.tag}.csv")
    rows = pd.read_csv(dst).to_dict("records") if os.path.exists(dst) else []
    va, te, feats = substrate(a.substrate_seed, a.source, clean=a.clean)
    t0 = time.time()
    phis = [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2, 3 * np.pi / 4]
    confirm = a.source in ("opsd_confirm", "india_daily_confirm")   # pre-registered reduced grid
    if a.part == "partition":
        grid = itertools.product(["unit", "state"], [0.0, 0.5, 1.0, 2.0, 4.0, 8.0], phis,
                                 [1.0, 2.0] if confirm else [0.5, 1.0, 2.0],
                                 [0.25, 1.0] if confirm else [0.1, 0.25, 1.0], range(2 if confirm else a.seeds))
    else:
        grid = itertools.product(["instance"], [0.0, 1.0, 4.0], [0.0, np.pi / 4, np.pi / 2], [1.0, 2.0], [0.25, 1.0],
                                 range(2 if confirm else a.seeds))
    seen = {(r["axis"], r["m"], round(r["phi"], 6), r["kappa"], r["frac"], r["seed"]) for r in rows}
    pat_cache = {}
    for axis, m, phi, kappa, frac, seed in grid:
        key = (axis, m, round(phi, 6), kappa, frac, a.seed_offset + seed)
        if key in seen:
            continue
        if axis == "unit":
            cv, ct = np.array(["all"] * len(va)), np.array(["all"] * len(te))
            fv, ft = va.sid.to_numpy(), te.sid.to_numpy()
        else:
            cv, ct = va.sid.to_numpy(), te.sid.to_numpy()
            fv = np.asarray(G.cell_keys(va.sid, va.hour))
            ft = np.asarray(G.cell_keys(te.sid, te.hour))
        pk = (axis, seed)
        if pk not in pat_cache:
            prng = np.random.default_rng(5000 + a.seed_offset + seed)
            if axis == "instance":
                pat_cache[pk] = feature_patterns(va[feats].to_numpy(np.float32), te[feats].to_numpy(np.float32),
                                                 va.r.to_numpy(), cv, ct, prng)
            else:
                pat_cache[pk] = cell_patterns(fv, ft, va.r.to_numpy(), cv, ct, prng)
        res = make_run(va, te, cv, ct, fv, ft, pat_cache[pk], m, phi, kappa, frac, a.seed_offset + seed,
                       fine="instance" if axis == "instance" else "partition", feats=feats if axis == "instance" else None)
        res["axis"] = axis
        rows.append(res)
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(dst, index=False)
            print(f"  {len(rows)} runs [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(dst, index=False)
    for ax, g in R.groupby("axis"):
        for col in [c for c in g.columns if c.endswith("_G") or c == "oracle_pred"]:
            print(f"{ax:9s} sign accuracy {col:22s} {np.mean(np.sign(g[col]) == np.sign(g.realised)):.3f}")
    print("DONE x1", a.part, len(R), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
