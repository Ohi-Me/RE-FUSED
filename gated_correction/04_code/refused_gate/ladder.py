"""The adaptivity-ladder engine (design §4).

run_ladder(frame, feats, state_keys, cfg, seed) fits a corrector on the training split, then every gate on the
gate-fitting set, and returns per-row test outputs (gates for every level) plus validation diagnostics.

Frame columns required: sid (str), ts (datetime64), y_t (target), B (baseline), split in {train,val,test},
feature columns, and one column per state key. Rows with missing y_t/B/features are dropped.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

from . import gates as G
from . import selection as S


@dataclass
class LadderConfig:
    corrector: dict = field(default_factory=lambda: dict(max_iter=300, learning_rate=0.06, max_depth=6,
                                                          l2_regularization=1.0))
    corrector_max_rows: int = 300_000
    min_cell: int = 30
    instance: dict = field(default_factory=lambda: dict(max_iter=150, learning_rate=0.08, max_depth=5))
    n_bag: int = 3
    windows_days: tuple = (30, 90, 180)
    lambdas: tuple = (0.98, 0.995)
    delay_days: int = 2                       # a gate used for target day D may use residuals up to day D - delay
    fixed_share_alpha: float = 0.02
    part_blocks: int = 6
    part_instance_blocks: int = 4
    part_eval_rows: int = 20_000
    do_part: bool = True
    part_version: int = 2                     # 2 = PART v2 + prequential (DV5, DV6); 1 = superseded block PART
    do_instance: bool = True
    do_time: bool = True
    part_block_mode: str = "contiguous"       # "contiguous" | "interleaved" | "year"   (DV1)
    gate_fit: str = "val"                     # "val" | "oof_train" | "oof_train+val"
    gate_fit_fraction: float = 1.0            # random fraction of gate-fitting DAYS (data-budget ladder)
    oof_folds: int = 5


def _fit_corrector(Xtr, ytr, cfg, seed):
    if len(Xtr) > cfg.corrector_max_rows:
        idx = np.random.default_rng(seed).choice(len(Xtr), cfg.corrector_max_rows, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    return HGB(random_state=seed, **cfg.corrector).fit(Xtr, ytr)


def _oof_predictions(tr, feats, cfg, seed):
    """Expanding-window out-of-fold corrector predictions on the training period (first fold is skipped)."""
    days = tr.ts.dt.normalize()
    cuts = np.quantile(days.astype("int64"), np.linspace(0, 1, cfg.oof_folds + 1))
    fold = np.clip(np.searchsorted(cuts[1:-1], days.astype("int64"), side="right"), 0, cfg.oof_folds - 1)
    pred = np.full(len(tr), np.nan)
    X, R = tr[feats].to_numpy(np.float32), tr.R.to_numpy()
    for k in range(1, cfg.oof_folds):
        m_fit, m_pred = fold < k, fold == k
        if m_fit.sum() < 1000 or m_pred.sum() == 0:
            continue
        pred[m_pred] = _fit_corrector(X[m_fit], R[m_fit], cfg, seed + k).predict(X[m_pred])
    return pred


def _daily_online_gate(fit_df, test_df, key_col, policy, cfg):
    """Time-axis gates, re-estimated daily during the test period from all residuals known by then.

    History = gate-fitting rows plus already-realised test rows (target day <= D - delay).
    policy: ("expanding",) | ("rolling", w_days) | ("forget", lam)
    """
    hist = pd.concat([fit_df[[key_col, "day", "Rr", "rr"]], test_df[[key_col, "day", "Rr", "rr"]]], ignore_index=True)
    agg = hist.groupby([key_col, "day"], observed=True)[["Rr", "rr"]].sum().reset_index()
    t_days = np.sort(test_df.day.unique())
    all_days = np.sort(agg.day.unique())
    out = pd.Series(np.nan, index=test_df.index)
    day_index = {d: i for i, d in enumerate(all_days)}
    for k, g in agg.groupby(key_col, observed=True):
        g = g.set_index("day").reindex(all_days, fill_value=0.0)
        Rr, rr = g.Rr.to_numpy(), g.rr.to_numpy()
        if policy[0] == "forget":
            lam = policy[1]
            sRr, srr = np.zeros(len(all_days)), np.zeros(len(all_days))
            a, b = 0.0, 0.0
            for i in range(len(all_days)):
                a = lam * a + Rr[i]
                b = lam * b + rr[i]
                sRr[i], srr[i] = a, b
        else:
            sRr, srr = np.cumsum(Rr), np.cumsum(rr)
        gd = {}
        for D in t_days:
            j = np.searchsorted(all_days, D - np.timedelta64(cfg.delay_days, "D"), side="right") - 1
            if j < 0:
                continue
            if policy[0] == "rolling":
                i0 = np.searchsorted(all_days, all_days[j] - np.timedelta64(policy[1], "D"), side="right") - 1
                num = sRr[j] - (sRr[i0] if i0 >= 0 else 0.0)
                den = srr[j] - (srr[i0] if i0 >= 0 else 0.0)
            else:
                num, den = sRr[j], srr[j]
            gd[D] = num / den if den > 1e-12 else np.nan
        m = test_df[key_col] == k
        out[m] = test_df.loc[m, "day"].map(gd).to_numpy()
    return out.to_numpy()


def _fixed_share(test_df, B_err_col, C_err_col, cfg, eta):
    """Per-series Fixed Share between baseline and gated correction with daily updates and the same delay."""
    w_out = pd.Series(np.nan, index=test_df.index)
    for s, g in test_df.groupby("sid"):
        daily = g.groupby("day")[[B_err_col, C_err_col]].sum()
        days = daily.index.to_numpy()
        lb, lc = daily[B_err_col].to_numpy(), daily[C_err_col].to_numpy()
        w = np.array([0.5, 0.5])
        pending = []
        wd = {}
        for i, D in enumerate(days):
            # apply updates whose losses are known by day D (day <= D - delay)
            while pending and pending[0][0] <= D - np.timedelta64(cfg.delay_days, "D"):
                _, a, b = pending.pop(0)
                v = w * np.exp(-eta * np.array([a, b]))
                v = v / v.sum() if v.sum() > 0 else np.array([0.5, 0.5])
                w = (1 - cfg.fixed_share_alpha) * v + cfg.fixed_share_alpha / 2
            wd[D] = w[1]
            pending.append((D, lb[i], lc[i]))
        w_out[g.index] = g.day.map(wd).to_numpy()
    return w_out.to_numpy()


def part_blocks(day, mode, K):
    """Block ids for PART (DV1). Returns (blocks, far_min).
    contiguous: K equal contiguous spans of days, far pairs = at least ceil(K/2) blocks apart
    interleaved: week index modulo K (every block spans all seasons), all distinct pairs count
    year: calendar year (use with multi-year gate-fitting sets), all distinct pairs count
    """
    day = pd.Series(pd.to_datetime(day))
    if mode == "contiguous":
        idx = pd.factorize(day, sort=True)[0]
        n_days = int(idx.max()) + 1
        return np.minimum((idx * K) // max(1, n_days), K - 1), int(np.ceil(K / 2))
    if mode == "interleaved":
        wk = ((day - day.min()).dt.days // 7).to_numpy()
        return wk % K, 1
    if mode == "year":
        return pd.factorize(day.dt.year, sort=True)[0], 1
    raise ValueError(mode)


def run_ladder(frame, feats, state_keys, cfg: LadderConfig, seed=0, verbose=False):
    t0 = time.time()
    x = frame.dropna(subset=feats + ["y_t", "B"]).copy()
    x["R"] = x.y_t - x.B
    x["day"] = x.ts.dt.normalize().to_numpy()
    tr, va, te = (x[x.split == s].copy() for s in ("train", "val", "test"))
    if min(len(tr), len(va), len(te)) < 400:
        return None
    X = {k: v[feats].to_numpy(np.float32) for k, v in (("tr", tr), ("va", va), ("te", te))}
    corr = _fit_corrector(X["tr"], tr.R.to_numpy(), cfg, seed)
    va["r"], te["r"] = corr.predict(X["va"]), corr.predict(X["te"])

    # ---- gate-fitting set ----
    if cfg.gate_fit == "val":
        fit = va.copy()
    else:
        tr["r"] = _oof_predictions(tr, feats, cfg, seed)
        oof = tr.dropna(subset=["r"])
        fit = oof.copy() if cfg.gate_fit == "oof_train" else pd.concat([oof, va], ignore_index=False)
    if cfg.gate_fit_fraction < 1.0:
        days = np.sort(fit.day.unique())
        keep = np.random.default_rng(10_000 + seed).choice(days, max(2, int(len(days) * cfg.gate_fit_fraction)), replace=False)
        fit = fit[fit.day.isin(keep)]
    g1 = G.ls_gate(fit.R.to_numpy(), fit.r.to_numpy())                  # D9 reparameterisation
    for f_ in (fit, va, te):
        f_["r"] = f_["r"] * g1
    Rf, rf = fit.R.to_numpy(), fit.r.to_numpy()
    Rt, rt = te.R.to_numpy(), te.r.to_numpy()

    gates = {"none": np.zeros(len(te)), "global": np.full(len(te), G.ls_gate(Rf, rf))}
    gs = G.PartitionGate(cfg.min_cell).fit(Rf, rf, fit.sid.to_numpy())
    gates["series"] = gs.predict(te.sid.to_numpy())
    for k in state_keys:
        key_f = G.cell_keys(fit.sid, fit[k])
        key_t = G.cell_keys(te.sid, te[k])
        gates[f"series_x_{k}"] = G.PartitionGate(cfg.min_cell).fit(Rf, rf, key_f).predict(key_t, parent_values=gates["series"])
    Ff, Ft = fit[feats].to_numpy(np.float32), X["te"]
    if cfg.do_instance:
        gi = G.InstanceGate(n_bag=cfg.n_bag, seed=seed, **cfg.instance).fit(Ff, Rf, rf)
        gates["instance"] = gi.predict(Ft)
        # empirical-Bayes shrinkage between per-series and instance using between-bag variance
        bag = np.vstack([m.predict(Ft) for m in gi.models])
        V = float(np.mean(bag.var(axis=0, ddof=1))) if cfg.n_bag > 1 else 0.0
        W = float(np.var(bag.mean(axis=0) - gates["series"]))
        lam = W / (W + V) if (W + V) > 0 else 0.0
        gates["shrunk"] = lam * gates["instance"] + (1 - lam) * gates["series"]

    diag = dict(g1=g1, n_fit=len(fit), n_test=len(te), fit_days=int(fit.day.nunique()))

    # ---- time axis (online re-estimation during the test period) ----
    if cfg.do_time:
        for f_ in (fit, te):
            f_["Rr"] = f_.R * f_.r
            f_["rr"] = f_.r * f_.r
        te_hist = te.copy()
        policies = [("expanding",)] + [("rolling", w) for w in cfg.windows_days] + [("forget", l) for l in cfg.lambdas]
        for pol in policies:
            name = "time_" + "_".join(str(p) for p in pol)
            g_on = _daily_online_gate(fit, te_hist, "sid", pol, cfg)
            gates[name] = np.where(np.isnan(g_on), gates["series"], g_on)
        # Fixed Share between baseline and static per-series correction
        te_hist["eB"] = Rt ** 2
        te_hist["eC"] = (Rt - np.clip(gates["series"], *G.CLIP) * rt) ** 2
        daily_scale = float(fit.assign(e=Rf ** 2).groupby(["sid", "day"]).e.sum().median())
        wfs = _fixed_share(te_hist, "eB", "eC", cfg, eta=1.0 / max(daily_scale, 1e-12))
        gates["time_fixed_share"] = np.clip(gates["series"], *G.CLIP) * np.nan_to_num(wfs, nan=0.5)

    # ---- validation diagnostics: PART and nested hold-out ----
    if cfg.do_part and cfg.part_version == 2:
        # PART v2 (DV5) and prequential validation of time policies (DV6/DF8); validation-only
        sid_f, day_f = fit.sid.to_numpy(), fit.day.to_numpy()
        p2 = {"global->series": S.part2_partition(Rf, rf, np.zeros(len(fit), int), sid_f, day_f, "interleaved", 6,
                                                  cfg.min_cell)}
        for k in state_keys:
            p2[f"series->series_x_{k}"] = S.part2_partition(Rf, rf, sid_f, np.asarray(G.cell_keys(fit.sid, fit[k])),
                                                            day_f, "interleaved", 6, cfg.min_cell)
        if cfg.do_instance:
            ev = np.random.default_rng(seed).choice(len(fit), min(cfg.part_eval_rows, len(fit)), replace=False)
            p2["series->instance"] = S.part2_instance(Ff, Rf, rf, sid_f, day_f, "interleaved", 4, ev, seed,
                                                      cfg.min_cell, **cfg.instance)
        diag["part2"] = p2
        if cfg.do_time:
            pq = {f"expanding->time_rolling_{w}": S.prequential_time(Rf, rf, sid_f, day_f, window=w,
                                                                    delay=cfg.delay_days)["G"] for w in cfg.windows_days}
            pq.update({f"expanding->time_forget_{l}": S.prequential_time(Rf, rf, sid_f, day_f, lam=l,
                                                                        delay=cfg.delay_days)["G"] for l in cfg.lambdas})
            diag["prequential"] = pq
    if cfg.do_part:
        modes = cfg.part_block_mode if isinstance(cfg.part_block_mode, (tuple, list)) else (cfg.part_block_mode,)
        diag["part_modes"] = {}
        for mode in ([] if cfg.part_version == 2 else modes):
            if mode == "year" and fit.day.dt.year.nunique() < 3:
                continue
            blocks, far_min = part_blocks(fit.day, mode, cfg.part_blocks)
            part = {"global->series": S.part_partition(Rf, rf, np.zeros(len(fit), int), fit.sid.to_numpy(), blocks,
                                                       cfg.min_cell, far_min=far_min)}
            for k in state_keys:
                part[f"series->series_x_{k}"] = S.part_partition(Rf, rf, fit.sid.to_numpy(),
                                                                 G.cell_keys(fit.sid, fit[k]), blocks, cfg.min_cell,
                                                                 far_min=far_min)
            if cfg.do_instance:
                bi, far_i = part_blocks(fit.day, mode, cfg.part_instance_blocks)
                ev = np.random.default_rng(seed).choice(len(fit), min(cfg.part_eval_rows, len(fit)), replace=False)
                part["series->instance"] = S.part_instance(Ff, Rf, rf, fit.sid.to_numpy(), bi, eval_idx=ev, n_bag=1,
                                                           seed=seed, min_cell=cfg.min_cell, far_min=far_i,
                                                           **cfg.instance)
            diag["part_modes"][mode] = part
        diag["part"] = diag["part_modes"].get(modes[0], {})
        diag["part_block_mode"] = list(modes)
        if cfg.part_version == 1:
            diag["part_time"] = {f"series->rolling_{bd}": S.part_time(Rf, rf, fit.sid.to_numpy(), fit.day.to_numpy(), bd)
                                 for bd in (30, 90)}
        # nested hold-out on the gate-fitting set: fit on first half of days, score on second half
        days = np.sort(fit.day.unique())
        h1 = fit.day <= days[len(days) // 2 - 1]
        f1, f2 = fit[h1], fit[~h1]
        R1, r1, R2, r2 = f1.R.to_numpy(), f1.r.to_numpy(), f2.R.to_numpy(), f2.r.to_numpy()
        nested = {"none": np.zeros(len(f2)), "global": np.full(len(f2), G.ls_gate(R1, r1))}
        ps = G.PartitionGate(cfg.min_cell).fit(R1, r1, f1.sid.to_numpy())
        nested["series"] = ps.predict(f2.sid.to_numpy())
        for k in state_keys:
            nested[f"series_x_{k}"] = G.PartitionGate(cfg.min_cell).fit(R1, r1, G.cell_keys(f1.sid, f1[k])).predict(
                G.cell_keys(f2.sid, f2[k]), parent_values=nested["series"])
        if cfg.do_instance:
            nested["instance"] = G.InstanceGate(n_bag=1, seed=seed, **cfg.instance).fit(
                f1[feats].to_numpy(np.float32), R1, r1).predict(f2[feats].to_numpy(np.float32))
        diag["nested_risk"] = {k: float(np.mean((R2 - np.clip(v, *G.CLIP) * r2) ** 2)) for k, v in nested.items()}

    out = te[["sid", "ts", "y_t", "B", "r"]].copy()
    for k in state_keys:
        out[k] = te[k].to_numpy()
    for k, v in gates.items():
        out["g_" + k] = np.clip(v, *G.CLIP).astype(np.float32)
    diag["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  ladder seed={seed} fit={len(fit)} test={len(te)} rungs={len(gates)} [{diag['seconds']}s]", flush=True)
    return out, diag


def score(out, rungs=None):
    """Squared-loss skill (macro over series) and pooled skill per rung from a run_ladder output."""
    rungs = rungs or [c[2:] for c in out.columns if c.startswith("g_")]
    R = (out.y_t - out.B).to_numpy()
    base = pd.Series(R ** 2).groupby(out.sid.to_numpy()).mean()
    rows = []
    for k in rungs:
        e = (R - out["g_" + k].to_numpy() * out.r.to_numpy()) ** 2
        ser = pd.Series(e).groupby(out.sid.to_numpy()).mean()
        skill = 1 - ser / base
        rows.append(dict(rung=k, skill_macro=float(skill.mean()), skill_pooled=float(1 - e.mean() / (R ** 2).mean()),
                         degraded=int((skill < 0).sum()), mse=float(e.mean())))
    return pd.DataFrame(rows)
