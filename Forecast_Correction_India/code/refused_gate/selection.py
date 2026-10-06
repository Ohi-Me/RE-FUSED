"""Validation-only selection of the adaptivity level (theory T7, T8).

PART - persistence-adjusted refinement test:
    G_hat = 2 C_hat - S_hat - E_hat ; refine iff G_hat > 0
All terms are in units of mean squared error per row (same units as the loss), using clipped gates.
"""
import numpy as np
import pandas as pd

from .gates import CLIP, PartitionGate, InstanceGate


def _fit_table(R, r, keys, min_cell):
    g = PartitionGate(min_cell).fit(R, r, keys)
    return g


def part_partition(R, r, coarse_keys, fine_keys, blocks, min_cell=30, far_frac=0.5, far_min=None):
    """PART for a partition refinement coarse_keys -> fine_keys (fine cells nested in coarse cells).

    blocks: integer block id per row in time order (0..K-1).
    Returns dict with S (heterogeneity energy), C (persistent part), E (estimation cost), G (predicted gain),
    and diagnostics.
    """
    R, r = np.asarray(R, float), np.asarray(r, float)
    coarse_keys, fine_keys, blocks = map(np.asarray, (coarse_keys, fine_keys, blocks))
    n = len(R)
    K = int(blocks.max()) + 1
    fine_u, inv = np.unique(fine_keys, return_inverse=True)
    parent_of = {}
    for fk, ck in zip(fine_keys, coarse_keys):
        parent_of.setdefault(fk, ck)
    w = np.bincount(inv, weights=r * r, minlength=len(fine_u)) / n
    parents = np.array([parent_of[f] for f in fine_u])
    min_blk = max(10, int(min_cell / K))

    def h_vector(mask, mc):
        gc = _fit_table(R[mask], r[mask], coarse_keys[mask], mc)
        gf = _fit_table(R[mask], r[mask], fine_keys[mask], mc)
        cvals = np.clip(gc.predict(parents), *CLIP)
        fvals = np.fromiter((gf.table.get(f, np.nan) for f in fine_u), float, count=len(fine_u))
        fvals = np.where(np.isnan(fvals), np.nan, np.clip(fvals, *CLIP))
        return fvals - cvals

    H = np.vstack([h_vector(blocks == b, min_blk) for b in range(K)])          # K x cells
    far = far_min if far_min is not None else max(1, int(np.ceil(K * far_frac)))
    S_terms, C_terms = [], []
    for b in range(K):
        for b2 in range(K):
            if b == b2:
                continue
            ok = ~np.isnan(H[b]) & ~np.isnan(H[b2])
            p = float(np.sum(w[ok] * H[b, ok] * H[b2, ok]))
            S_terms.append(p)
            if abs(b - b2) >= far:
                C_terms.append(p)
    S = float(np.mean(S_terms)) if S_terms else 0.0
    C = float(np.mean(C_terms)) if C_terms else 0.0
    h_full = h_vector(np.ones(n, bool), min_cell)
    jk = np.vstack([h_vector(blocks != b, min_cell) for b in range(K)])
    jk = np.where(np.isnan(jk), np.nanmean(jk, axis=0, keepdims=True), jk)
    var_jk = (K - 1) / K * np.nansum((jk - np.nanmean(jk, axis=0, keepdims=True)) ** 2, axis=0)
    ok = ~np.isnan(h_full)
    E = float(np.sum(w[ok] * var_jk[ok]))
    return dict(S=S, C=C, E=E, G=2 * C - S - E, K=K, cells=int(len(fine_u)), energy_full=float(np.nansum(w * h_full ** 2)))


def part_instance(F, R, r, coarse_keys, blocks, eval_idx=None, n_bag=1, seed=0, min_cell=30, far_frac=0.5,
                  far_min=None, **hgb):
    """PART for refining a partition gate (coarse_keys) to a per-instance model gate."""
    F, R, r = np.asarray(F), np.asarray(R, float), np.asarray(r, float)
    coarse_keys, blocks = np.asarray(coarse_keys), np.asarray(blocks)
    n = len(R)
    K = int(blocks.max()) + 1
    ev = np.arange(n) if eval_idx is None else np.asarray(eval_idx)
    wv = r[ev] ** 2

    def h_fun(mask, mc):
        gc = PartitionGate(mc).fit(R[mask], r[mask], coarse_keys[mask])
        cvals = np.clip(gc.predict(coarse_keys[ev]), *CLIP)
        gi = InstanceGate(n_bag=n_bag, seed=seed, **hgb).fit(F[mask], R[mask], r[mask])
        return np.clip(gi.predict(F[ev]), *CLIP) - cvals

    min_blk = max(10, int(min_cell / K))
    H = np.vstack([h_fun(blocks == b, min_blk) for b in range(K)])
    far = far_min if far_min is not None else max(1, int(np.ceil(K * far_frac)))
    S_terms, C_terms = [], []
    for b in range(K):
        for b2 in range(K):
            if b != b2:
                p = float(np.mean(wv * H[b] * H[b2]))
                S_terms.append(p)
                if abs(b - b2) >= far:
                    C_terms.append(p)
    S, C = float(np.mean(S_terms)), float(np.mean(C_terms))
    jk = np.vstack([h_fun(blocks != b, min_cell) for b in range(K)])
    var_jk = (K - 1) / K * np.sum((jk - jk.mean(axis=0, keepdims=True)) ** 2, axis=0)
    E = float(np.mean(wv * var_jk))
    return dict(S=S, C=C, E=E, G=2 * C - S - E, K=K)


def _block_design(day, mode, K):
    day = pd.Series(pd.to_datetime(day))
    if mode == "interleaved":
        return (((day - day.min()).dt.days // 7) % K).to_numpy(), 1
    if mode == "year":
        return pd.factorize(day.dt.year, sort=True)[0], 1
    if mode == "contiguous":
        idx = pd.factorize(day, sort=True)[0]
        return np.minimum((idx * K) // (idx.max() + 1), K - 1), int(np.ceil(K / 2))
    raise ValueError(mode)


def _ls_table(R, r, keys, min_n):
    uniq, inv = np.unique(keys, return_inverse=True)
    num = np.bincount(inv, weights=R * r, minlength=len(uniq))
    den = np.bincount(inv, weights=r * r, minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq))
    g = np.where((cnt >= min_n) & (den > 1e-12), num / np.maximum(den, 1e-300), np.nan)
    return dict(zip(uniq, g))


def part2_partition(R, r, coarse_keys, fine_keys, day, mode="interleaved", K=6, min_cell=30):
    """PART v2 (DV5): G = 2 C - ||h_full||^2 with UNCLIPPED block estimates and no jackknife.

    E||h_hat_full||^2 = ||h||^2 + E||eps||^2 and the cross-block mean C estimates the persistent heterogeneity, so
    2C - ||h_hat_full||^2 estimates 2<h_V, h_T> - ||h_V||^2 - E||eps||^2 (theory T4).
    """
    R, r = np.asarray(R, float), np.asarray(r, float)
    coarse_keys, fine_keys = np.asarray(coarse_keys).astype(str), np.asarray(fine_keys).astype(str)
    n = len(R)
    blocks, far = _block_design(day, mode, K)
    Kb = int(blocks.max()) + 1
    fine_u, inv = np.unique(fine_keys, return_inverse=True)
    parent = dict(zip(fine_keys, coarse_keys))
    par = np.array([parent[f] for f in fine_u])
    w = np.bincount(inv, weights=r * r, minlength=len(fine_u)) / n

    def h_vec(mask, mn):
        gc = _ls_table(R[mask], r[mask], coarse_keys[mask], mn)
        gf = _ls_table(R[mask], r[mask], fine_keys[mask], mn)
        return np.array([gf.get(f, np.nan) - gc.get(p, np.nan) for f, p in zip(fine_u, par)])

    H = np.vstack([h_vec(blocks == b, max(5, min_cell // Kb)) for b in range(Kb)])
    h_full = h_vec(np.ones(n, bool), min_cell)
    ok_full = ~np.isnan(h_full)
    energy = float(np.sum(w[ok_full] * h_full[ok_full] ** 2))
    Ct, St = [], []
    for b in range(Kb):
        for b2 in range(Kb):
            if b == b2:
                continue
            ok = ~np.isnan(H[b]) & ~np.isnan(H[b2]) & ok_full
            p = float(np.sum(w[ok] * H[b, ok] * H[b2, ok]))
            St.append(p)
            if abs(b - b2) >= far:
                Ct.append(p)
    S = float(np.mean(St)) if St else 0.0
    C = float(np.mean(Ct)) if Ct else 0.0
    return dict(G=2 * C - energy, C=C, S=S, energy_full=energy, noise=max(energy - S, 0.0), K=Kb, mode=mode)


def part2_instance(F, R, r, coarse_keys, day, mode="interleaved", K=4, eval_idx=None, seed=0, min_cell=30, **hgb):
    """PART v2 for partition -> per-instance model gate: G = 2 C - ||h_full||^2, unclipped model predictions.
    Block models use n/K rows, so C under-states the heterogeneity a full-n model learns (bias toward coarse)."""
    F, R, r = np.asarray(F), np.asarray(R, float), np.asarray(r, float)
    coarse_keys = np.asarray(coarse_keys).astype(str)
    n = len(R)
    ev = np.arange(n) if eval_idx is None else np.asarray(eval_idx)
    w = r[ev] ** 2
    blocks, far = _block_design(day, mode, K)
    Kb = int(blocks.max()) + 1

    def h_fun(mask, mn):
        gc = _ls_table(R[mask], r[mask], coarse_keys[mask], mn)
        cvals = np.array([gc.get(k, np.nan) for k in coarse_keys[ev]])
        gi = InstanceGate(n_bag=1, seed=seed, **hgb).fit(F[mask], R[mask], r[mask])
        return gi.predict(F[ev]) - cvals

    H = np.vstack([h_fun(blocks == b, max(5, min_cell // Kb)) for b in range(Kb)])
    h_full = h_fun(np.ones(n, bool), min_cell)
    ok = ~np.isnan(h_full)
    energy = float(np.mean(np.where(ok, w * h_full ** 2, 0.0)))
    Ct = []
    for b in range(Kb):
        for b2 in range(Kb):
            if b != b2 and abs(b - b2) >= far:
                m = ok & ~np.isnan(H[b]) & ~np.isnan(H[b2])
                Ct.append(float(np.mean(np.where(m, w * H[b] * H[b2], 0.0))))
    C = float(np.mean(Ct)) if Ct else 0.0
    return dict(G=2 * C - energy, C=C, energy_full=energy, K=Kb, mode=mode)


def part2_time(R, r, sid, day, block_days=30):
    """PART-time v2 (DV5): rolling re-estimation vs an EXPANDING static gate (history only), unclipped.
    For block b+1 the comparator is the gate fitted on all blocks <= b (what a static deployed gate knows) and the
    candidate is the gate fitted on block b. G = mean_b sum_s w [2 h_b h'_{b+1} - h_b^2], h_b = g_b - g_<=b,
    h'_{b+1} = g_{b+1} - g_<=b (independent block noise cancels; the shared comparator noise is second order)."""
    R, r = np.asarray(R, float), np.asarray(r, float)
    day = pd.to_datetime(pd.Series(day)).to_numpy()
    b = ((day - day.min()) / np.timedelta64(1, "D")).astype(int) // block_days
    df = pd.DataFrame({"s": np.asarray(sid), "b": b, "Rr": R * r, "rr": r * r})
    blk = df.groupby(["s", "b"])[["Rr", "rr"]].sum().reset_index()
    K = int(b.max()) + 1
    n_rows = df.groupby("b").size()
    terms = []
    for k in range(1, K - 1):
        tot = 0.0
        for s, g in blk.groupby("s"):
            past = g[g.b <= k]
            cur, nxt = g[g.b == k], g[g.b == k + 1]
            if len(cur) == 0 or len(nxt) == 0 or past.rr.sum() <= 0:
                continue
            g_past = past.Rr.sum() / past.rr.sum()
            h_b = cur.Rr.iloc[0] / cur.rr.iloc[0] - g_past
            h_n = nxt.Rr.iloc[0] / nxt.rr.iloc[0] - g_past
            w = nxt.rr.iloc[0] / n_rows[k + 1]
            tot += w * (2 * h_b * h_n - h_b ** 2)
        terms.append(tot)
    return dict(G=float(np.mean(terms)) if terms else 0.0, K=K, block_days=block_days)


def prequential_time(R, r, sid, day, window=None, lam=None, delay=2, burn_in_days=60):
    """Prequential (online replay) validation of a time-adaptive per-series gate (DV6).

    Replays the deployed online rule through the gate-fitting period: for each day D after `burn_in_days`, the
    candidate gate uses residuals of days in (D - delay - window, D - delay] (or exponential forgetting with factor
    `lam`), the comparator is the expanding gate on all days <= D - delay. Returns the mean realised gain per row
    (T1 identity) in MSE units; positive = time adaptivity helps. Only information available at each decision time
    is used, so there is no in-sample optimism and no halving bias for sequential policies.
    """
    R, r = np.asarray(R, float), np.asarray(r, float)
    d = pd.DataFrame({"s": np.asarray(sid), "day": pd.to_datetime(pd.Series(day)).dt.normalize().to_numpy(),
                      "R": R, "r": r})
    d["Rr"], d["rr"] = d.R * d.r, d.r * d.r
    start = d.day.min() + pd.Timedelta(days=burn_in_days)
    gains, count = 0.0, 0
    for s, g in d.groupby("s"):
        daily = g.groupby("day")[["Rr", "rr"]].sum()
        days = daily.index.to_numpy()
        Rr, rr = daily.Rr.to_numpy(), daily.rr.to_numpy()
        cRr, crr = np.cumsum(Rr), np.cumsum(rr)
        if lam is not None:
            eRr, err = np.zeros(len(days)), np.zeros(len(days))
            a = b = 0.0
            for i in range(len(days)):
                a, b = lam * a + Rr[i], lam * b + rr[i]
                eRr[i], err[i] = a, b
        gd_c, gd_e = {}, {}
        for D in days[days >= np.datetime64(start)]:
            j = np.searchsorted(days, D - np.timedelta64(delay, "D"), side="right") - 1
            if j < 0:
                continue
            gd_e[D] = cRr[j] / crr[j] if crr[j] > 0 else np.nan
            if lam is not None:
                gd_c[D] = eRr[j] / err[j] if err[j] > 0 else np.nan
            else:
                i0 = np.searchsorted(days, days[j] - np.timedelta64(window, "D"), side="right") - 1
                num = cRr[j] - (cRr[i0] if i0 >= 0 else 0.0)
                den = crr[j] - (crr[i0] if i0 >= 0 else 0.0)
                gd_c[D] = num / den if den > 1e-12 else np.nan
        m = g.day.isin(list(gd_c))
        if not m.any():
            continue
        gc = np.clip(g.loc[m, "day"].map(gd_e).to_numpy(float), *CLIP)
        gf = np.clip(g.loc[m, "day"].map(gd_c).to_numpy(float), *CLIP)
        ok = ~np.isnan(gc) & ~np.isnan(gf)
        Rm, rm = g.loc[m, "R"].to_numpy()[ok], g.loc[m, "r"].to_numpy()[ok]
        gains += float(np.sum(realised_gain(Rm, rm, gc[ok], gf[ok])))
        count += int(ok.sum())
    return dict(G=gains / max(count, 1), n=count)


def part_time(R, r, sid, day, block_days=30):
    """PART along the time axis (DF6): value of a per-series gate re-estimated on the previous block of
    `block_days` days instead of the static per-series gate fitted on the whole gate-fitting period.

    For consecutive blocks b, b+1 and deviations h_hat_b = g_hat_b - g_static (per series),
        G_hat = mean_b sum_s w_{s,b+1} [ 2 h_hat_{s,b} h_hat_{s,b+1} - h_hat_{s,b}^2 ]
    is unbiased for the expected gain (estimation noise in independent blocks cancels), with w the share of
    signal energy r^2 of series s in block b+1 per row of that block. Units: MSE per row.
    """
    R, r = np.asarray(R, float), np.asarray(r, float)
    sid = np.asarray(sid)
    day = pd.to_datetime(pd.Series(day)).to_numpy()
    b = ((day - day.min()) / np.timedelta64(1, "D")).astype(int) // block_days
    K = int(b.max()) + 1
    df = pd.DataFrame({"s": sid, "b": b, "Rr": R * r, "rr": r * r})
    stat = df.groupby("s")[["Rr", "rr"]].sum()
    g_static = (stat.Rr / stat.rr).to_dict()
    blk = df.groupby(["s", "b"])[["Rr", "rr"]].sum()
    blk["g"] = blk.Rr / blk.rr
    blk["h"] = blk.g - blk.index.get_level_values(0).map(g_static)
    n_rows = df.groupby("b").size()
    terms = []
    for k in range(K - 1):
        if k not in n_rows.index or (k + 1) not in n_rows.index:
            continue
        tot = 0.0
        for s in np.unique(sid):
            if (s, k) not in blk.index or (s, k + 1) not in blk.index:
                continue
            h0, h1 = blk.loc[(s, k), "h"], blk.loc[(s, k + 1), "h"]
            w = blk.loc[(s, k + 1), "rr"] / n_rows[k + 1]
            tot += w * (2 * h0 * h1 - h0 ** 2)
        terms.append(tot)
    return dict(G=float(np.mean(terms)) if terms else 0.0, K=K, block_days=block_days, n_terms=len(terms))


def realised_gain(R, r, g_coarse, g_fine):
    """Exact per-row realised gain of fine over coarse (theory T1): positive = fine better."""
    gc, gf = np.clip(g_coarse, *CLIP), np.clip(g_fine, *CLIP)
    d = gf - gc
    e = R - gc * r
    return 2 * d * r * e - (d * r) ** 2
