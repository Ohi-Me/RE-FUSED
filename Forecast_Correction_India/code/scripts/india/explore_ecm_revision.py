"""Exploratory analyses of the all-India hourly study (not pre-registered; added after the confirmatory scoring).

Protocol, settings and status: design/11_ecm_revision_exploratory.md (committed before this script first ran).
Blocks:
  R1  conventional residual corrections from the base forecast's own past errors; R1b the learned gated correction
      applied on top of the validation-selected conventional correction
  R2  error-aware corrector (features extended by the base forecast and its lagged errors)
  R3  refreshed LightGBM base forecast (monthly refits), its ladder and its energy valuation
  R4  rolling-origin check of PART and prequential validation, against the fixed validation window
  R5  reserve recalibration: monthly coverage of the stored quantile tables, online quantile tracking
  R6  sensitivity of the stylised imbalance cost to the settlement assumptions
  R7  block-length sensitivity of the hourly energy intervals
Output: results/india/explore_ecm/<block>/ (an existing output file is never overwritten).
Usage:  python code/scripts/india/explore_ecm_revision.py --blocks R1,R1b,R2,R3,R4,R5,R6,R7   (hpc/explore_ecm.pbs)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

os.environ["REFUSED_GATE_STUDY"] = "india"
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "dev"))
from refused_gate import data, gates as G, ladder, selection as S, stats as ST  # noqa: E402
from refused_gate.paths import PROC, RES  # noqa: E402

OUT = os.path.join(RES, "india", "explore_ecm")
CONF = os.path.join(RES, "india", "confirm")
DEVR = os.path.join(RES, "india", "dev")
SEEDS = (0, 1, 2)
BASES = ["snaive168", "lag48", "lgbm", "chronos", "nhits", "patchtst", "tide", "dlinear"]
LEARNED = set(BASES[2:])
VAL0, TEST0, TEST1 = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-03-01"), pd.Timestamp("2026-09-14")
STATE_KEYS = ["hour_block", "hour"]
CRORE = 1e7
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def out_path(block, name):
    d = os.path.join(OUT, block)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save_df(df, block, name):
    p = out_path(block, name)
    if os.path.exists(p):
        log("exists, kept:", p)
        return
    (df.to_parquet if p.endswith(".parquet") else df.to_csv)(p, index=False)


def save_json(obj, block, name):
    p = out_path(block, name)
    if os.path.exists(p):
        log("exists, kept:", p)
        return
    json.dump(obj, open(p, "w"), indent=1, default=float)


# --------------------------------------------------------------------------------------------------- data
_CACHE = {}


def hourly():
    """The confirmatory hourly frame (guarded by the India pre-registration lock) with every base forecast."""
    if "h" not in _CACHE:
        ds = data.india_hourly("confirm")
        L = pd.read_parquet(os.path.join(PROC, "india_hourly_learned_baselines_confirm_s0.parquet"))
        df = ds["df"].merge(L, on=["sid", "ts"], how="left")
        cols = dict(ds["baselines"])
        cols.update({c[2:]: c for c in L.columns if c.startswith("B_")})
        _CACHE["h"] = (ds, df, cols)
    return _CACHE["h"]


def ladder_frame(df, col, base):
    fr = df.assign(y_t=df.y, B=df[col])
    if base in LEARNED:
        fr = fr[fr[col].notna() | (fr.split != "train")]
    return fr


def ih1(base, seed):
    return pd.read_parquet(os.path.join(CONF, "IH1", f"{base}_s{seed}.parquet"))


def design_cols(out):
    return [c for c in out.columns if c.startswith("g_")]


def skill_by_series(y, B, F, sid):
    sid = np.asarray(sid)
    R2, E2 = pd.Series(np.asarray((y - B) ** 2, float)), pd.Series(np.asarray((y - F) ** 2, float))
    return 1 - E2.groupby(sid).mean() / R2.groupby(sid).mean()


def seed_mean_losses(outs):
    """Per-row squared error of every design, averaged over seeds (rows aligned on sid, ts)."""
    frames = []
    for o in outs:
        R = (o.y_t - o.B).to_numpy()
        L = pd.DataFrame({"sid": o.sid.to_numpy(), "ts": o.ts.to_numpy(), "base": R ** 2})
        for c in design_cols(o):
            L[c[2:]] = (R - o[c].to_numpy() * o.r.to_numpy()) ** 2
        frames.append(L.set_index(["sid", "ts"]))
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    return (sum(f.loc[common] for f in frames) / len(frames)).reset_index()


def daily_ci(diff_by_day, scale=1.0, block=7):
    d = np.asarray(diff_by_day, float)
    c = ST.block_ci(d, block, n_boot=2000, seed=0)
    return dict(mean=c["mean"] * scale, lo=c["lo"] * scale, hi=c["hi"] * scale, p_greater=c["p_greater"],
                p_less=c["p_less"], n_days=c["n"], block=block)


def stationary_ci(d, mean_block=7, n_boot=2000, seed=0, scale=1.0):
    """Stationary bootstrap (Politis and Romano) of the mean of a daily series, geometric block lengths."""
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        new = rng.random(n) < 1.0 / mean_block
        new[0] = True
        starts = rng.integers(0, n, size=n)
        idx = np.empty(n, int)
        cur = 0
        for t in range(n):
            cur = starts[t] if new[t] else (cur + 1) % n
            idx[t] = cur
        means[b] = d[idx].mean()
    return dict(mean=float(d.mean() * scale), lo=float(np.quantile(means, 0.025) * scale),
                hi=float(np.quantile(means, 0.975) * scale), n_days=n, block=f"stationary({mean_block})")


# --------------------------------------------------------------------------------------------------- R1
def _day_hour(ts, v):
    m = pd.DataFrame({"day": ts.dt.normalize().to_numpy(), "hour": ts.dt.hour.to_numpy(), "v": np.asarray(v, float)})
    M = m.pivot_table(index="day", columns="hour", values="v", aggfunc="mean").sort_index()
    return M.reindex(pd.date_range(M.index.min(), M.index.max(), freq="D")).reindex(columns=range(24))


def _back(M, ts):
    """Look up a day x hour matrix at the rows' (day, hour)."""
    s = M.stack(dropna=False)
    idx = pd.MultiIndex.from_arrays([ts.dt.normalize(), ts.dt.hour])
    return s.reindex(idx).to_numpy(dtype=float)


def _rolling_ls(H, ycol, xcols, D, days_back):
    """Least squares on the rows of days D-1-days_back .. D-2 (all known at the end of D-2)."""
    lo, hi = D - pd.Timedelta(days=days_back + 1), D - pd.Timedelta(days=1)
    w = H[(H.ts >= lo) & (H.ts < hi)].dropna(subset=[ycol] + xcols)
    if len(w) < 10 * (len(xcols) + 1):
        return None
    beta, *_ = np.linalg.lstsq(w[xcols].to_numpy(float), w[ycol].to_numpy(float), rcond=None)
    return beta


def conventional(ts, y, B):
    """Conventional corrections of one series on a complete hourly grid (protocol R1). Returns {method: correction}."""
    E = pd.Series(np.asarray(y, float) - np.asarray(B, float))
    ts = pd.Series(pd.to_datetime(ts)).reset_index(drop=True)
    M = _day_hour(ts, E)
    out = {"bc_hod28": _back(M.rolling(28, min_periods=14).mean().shift(2), ts),
           "bc_ewma": _back(M.ewm(alpha=0.1, ignore_na=True, min_periods=7).mean().shift(2), ts)}
    how = pd.DataFrame(np.nan, index=M.index, columns=M.columns)
    for wd in range(7):
        sub = M[M.index.dayofweek == wd]
        how.loc[sub.index] = sub.rolling(8, min_periods=4).mean().shift(1).to_numpy()
    out["bc_how8"] = _back(how.astype(float), ts)
    H = pd.DataFrame({"ts": ts, "E": E.to_numpy(), "how": out["bc_how8"]})
    H["u"] = H.E - H.how
    for L in (48, 72, 168):
        H[f"u{L}"] = H.u.shift(L)
        H[f"E{L}"] = H.E.shift(L)
    H["Em168"] = H.E.shift(48).rolling(168, min_periods=24).mean()
    hours = pd.get_dummies(H.ts.dt.hour, prefix="h", dtype=float)
    H = pd.concat([H, hours], axis=1)
    H["one"] = 1.0
    xar = ["u48", "u72", "u168"]
    xlin = ["one", "E48", "E72", "E168", "Em168"] + list(hours.columns)[1:]
    ar, lin = np.full(len(H), np.nan), np.full(len(H), np.nan)
    day = H.ts.dt.normalize()
    for D in pd.date_range(VAL0 - pd.Timedelta(days=60), day.max(), freq="D"):
        rows = (day == D).to_numpy()
        if not rows.any():
            continue
        b = _rolling_ls(H, "u", xar, D, 56)
        if b is not None:
            ar[rows] = H.loc[rows, xar].to_numpy(float) @ b
        b = _rolling_ls(H, "E", xlin, D, 90)
        if b is not None:
            lin[rows] = H.loc[rows, xlin].to_numpy(float) @ b
    out["bc_how_ar"] = out["bc_how8"] + ar
    out["bc_linreg"] = lin
    return out


_CONV = {}


def conv_frame(base):
    """Conventional corrections of one base forecast for all four series, on the complete hourly grid."""
    if base not in _CONV:
        ds, df, cols = hourly()
        col = cols[base]
        parts = []
        for sid, g in df.groupby("sid"):
            g = g.sort_values("ts").set_index("ts")
            g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h")).rename_axis("ts").reset_index()
            g["sid"] = sid
            corr = conventional(g.ts, g.y, g[col])
            f = g[["sid", "ts", "y"]].assign(B=g[col].to_numpy())
            for k, v in corr.items():
                f[k] = v
            parts.append(f)
        _CONV[base] = pd.concat(parts, ignore_index=True)
    return _CONV[base]


METHODS = ["bc_hod28", "bc_ewma", "bc_how8", "bc_how_ar", "bc_linreg"]


def _selected_correction(cf, picks, base):
    sel = np.zeros(len(cf))
    for sid in cf.sid.unique():
        m = (cf.sid == sid).to_numpy()
        sel[m] = np.nan_to_num(cf.loc[m, picks[(base, sid)]].to_numpy(float), nan=0.0)
    return sel


def block_r1():
    rows, picks, cmp_rows = [], {}, []
    for base in BASES:
        cf = conv_frame(base)
        outs = [ih1(base, s) for s in SEEDS]
        te = outs[0][["sid", "ts"]].merge(cf, on=["sid", "ts"], how="left")
        va = cf[(cf.ts >= VAL0) & (cf.ts < TEST0) & cf.y.notna() & cf.B.notna()]
        for sid in sorted(te.sid.unique()):
            t, v = te[te.sid == sid], va[va.sid == sid]
            base_mse, base_val = float(np.mean((t.y - t.B) ** 2)), float(np.mean((v.y - v.B) ** 2))
            best, best_val = None, np.inf
            for m in METHODS:
                Ft, Fv = t.B + t[m].fillna(0.0), v.B + v[m].fillna(0.0)
                mse_val = float(np.mean((v.y - Fv) ** 2))
                rows.append(dict(base=base, sid=sid, method=m,
                                 skill_test=1 - float(np.mean((t.y - Ft) ** 2)) / base_mse,
                                 skill_val=1 - mse_val / base_val, fallback_test_rows=int(t[m].isna().sum()),
                                 n_test=len(t)))
                if mse_val < best_val:
                    best, best_val = m, mse_val
            picks[(base, sid)] = best
        L = seed_mean_losses(outs).merge(te[["sid", "ts", "y", "B"]], on=["sid", "ts"], how="left")
        cfm = cf.set_index(["sid", "ts"])
        corr = _selected_correction(cf, picks, base)
        csel = pd.Series(corr, index=cfm.index).reindex(pd.MultiIndex.from_frame(L[["sid", "ts"]])).to_numpy()
        L["conv_sel"] = (L.y - L.B - np.nan_to_num(csel)) ** 2
        L["day"] = L.ts.dt.normalize()
        norm = L.groupby("sid").base.transform("sum") * L.sid.nunique()
        for design in ("series", "time_expanding", "time_rolling_30", "time_fixed_share"):
            d = ((L.conv_sel - L[design]) / norm).groupby(L.day).sum()
            ci = daily_ci(d.to_numpy(), scale=len(d))
            cmp_rows.append(dict(base=base, design=design, skill_gap_design_minus_conv=ci["mean"], lo=ci["lo"],
                                 hi=ci["hi"], p_design_better=ci["p_greater"], p_conv_better=ci["p_less"],
                                 n_days=ci["n_days"]))
        log("R1", base, {k[1]: v for k, v in picks.items() if k[0] == base})
    R = pd.DataFrame(rows)
    R["selected"] = [picks[(b, s)] == m for b, s, m in zip(R.base, R.sid, R.method)]
    save_df(R, "R1", "conventional_skill.csv")
    save_df(pd.DataFrame(cmp_rows), "R1", "design_vs_conventional.csv")
    save_json({f"{b}|{s}": m for (b, s), m in picks.items()}, "R1", "selected_method.json")
    return picks


def block_r1b(picks=None):
    if picks is None:
        picks = {tuple(k.split("|")): v for k, v in json.load(open(out_path("R1", "selected_method.json"))).items()}
    ds, df, cols = hourly()
    rows = []
    for base in BASES:
        cf = conv_frame(base)
        cf = cf.assign(Bc=cf.B + _selected_correction(cf, picks, base))
        fr = df.merge(cf[["sid", "ts", "Bc"]], on=["sid", "ts"], how="left")
        fr = ladder_frame(fr, "Bc", "lgbm" if base in LEARNED else base)
        cfg = ladder.LadderConfig(n_bag=3, delay_days=2, do_part=False, do_instance=False)
        for s in SEEDS:
            dst = out_path("R1b", f"{base}_s{s}.parquet")
            if not os.path.exists(dst):
                res = ladder.run_ladder(fr, ds["feats"], STATE_KEYS, cfg, seed=s)
                if res is None:
                    continue
                res[0].to_parquet(dst, index=False)
            o = pd.read_parquet(dst).merge(ih1(base, s)[["sid", "ts", "B"]].rename(columns={"B": "B0"}),
                                           on=["sid", "ts"])
            for c in ["g_none", "g_series", "g_time_expanding"]:
                for sid, v in skill_by_series(o.y_t, o.B0, o.B + o[c] * o.r, o.sid).items():
                    rows.append(dict(base=base, seed=s, sid=sid, design="conv+" + c[2:], skill_vs_original_base=v,
                                     n_rows=int((o.sid == sid).sum())))
        log("R1b", base)
    save_df(pd.DataFrame(rows), "R1b", "skill_gate_on_conventional.csv")


# --------------------------------------------------------------------------------------------------- R2
def block_r2():
    ds, df, cols = hourly()
    rows = []
    for base in BASES:
        col = cols[base]
        long = df[["sid", "ts", "y", col]].rename(columns={col: "Bfc"})
        d, feats = data._hourly_features(long, fc_col="Bfc")
        d["season"] = (d.ts.dt.month % 12) // 3
        d["split"] = np.select([d.ts < VAL0, d.ts < TEST0, d.ts < TEST1], ["train", "val", "test"], "drop")
        d = d[d.split != "drop"]
        fr = d.assign(y_t=d.y, B=d.Bfc)
        if base in LEARNED:
            fr = fr[fr.Bfc.notna() | (fr.split != "train")]
        for s in SEEDS:
            dst = out_path("R2", f"{base}_s{s}.parquet")
            if not os.path.exists(dst):
                res = ladder.run_ladder(fr, feats, STATE_KEYS, ladder.LadderConfig(n_bag=3, delay_days=2), seed=s)
                if res is None:
                    continue
                res[0].to_parquet(dst, index=False)
                json.dump(res[1], open(dst.replace(".parquet", ".json"), "w"), indent=1, default=str)
            o, o0 = pd.read_parquet(dst), ih1(base, s)
            keys = o0[["sid", "ts"]].merge(o[["sid", "ts"]], on=["sid", "ts"])
            for tag, oo in (("error_aware", o), ("original", o0)):
                oo = keys.merge(oo, on=["sid", "ts"])
                for c in design_cols(oo):
                    for sid, v in skill_by_series(oo.y_t, oo.B, oo.B + oo[c] * oo.r, oo.sid).items():
                        rows.append(dict(base=base, seed=s, sid=sid, corrector=tag, design=c[2:], skill=v,
                                         n_rows=len(oo)))
        log("R2", base)
    save_df(pd.DataFrame(rows), "R2", "skill_error_aware.csv")


# --------------------------------------------------------------------------------------------------- R3
def block_r3():
    import make_learned_baselines as M
    import make_learned_baselines_india as MI
    dst = out_path("R3", "lgbm_refreshed_s0.parquet")
    if not os.path.exists(dst):
        p = MI.panel("confirm")
        parts = []
        for m0 in pd.date_range("2025-05-01", "2026-09-01", freq="MS"):
            m1 = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp("2026-09-13"))
            te = (m0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            parts.append(M.lgbm(p, te, m0.strftime("%Y-%m-%d"), m1.strftime("%Y-%m-%d"), 0))
            log("R3 refit for", m0.date(), "trained on data before", te)
        pd.concat(parts, ignore_index=True).rename(columns={"B_lgbm": "B_lgbmref"}).to_parquet(dst, index=False)
    ref = pd.read_parquet(dst)
    ds, df, cols = hourly()
    fr = ladder_frame(df.merge(ref, on=["sid", "ts"], how="left"), "B_lgbmref", "lgbm")
    for s in SEEDS:
        o_dst = out_path("R3", f"ladder_s{s}.parquet")
        if not os.path.exists(o_dst):
            res = ladder.run_ladder(fr, ds["feats"], STATE_KEYS, ladder.LadderConfig(n_bag=3, delay_days=2), seed=s)
            res[0].to_parquet(o_dst, index=False)
            json.dump(res[1], open(o_dst.replace(".parquet", ".json"), "w"), indent=1, default=str)
    outs = [pd.read_parquet(out_path("R3", f"ladder_s{s}.parquet")) for s in SEEDS]
    stale = [ih1("lgbm", s) for s in SEEDS]
    from refused_gate.paths import INDIA_PROC
    pr = pd.read_parquet(os.path.join(INDIA_PROC, "refused_allindia_hourly_prices.parquet"))[
        ["ts", "dam_mcp_rs_mwh", "rtm_mcp_rs_mwh"]]
    summ, rows = {}, []
    for tag, oo in (("refreshed", outs), ("stale", stale)):
        for s, o in zip(SEEDS, oo):
            for c in design_cols(o):
                for sid, v in skill_by_series(o.y_t, o.B, o.B + o[c] * o.r, o.sid).items():
                    rows.append(dict(base=tag, seed=s, sid=sid, design=c[2:], skill=v))
        dem = [o[o.sid == "demand"].sort_values("ts").reset_index(drop=True) for o in oo]
        d0 = dem[0].merge(pr, on="ts", how="left")
        Fc = np.mean([(d.B + d.g_series * d.r).to_numpy() for d in dem], axis=0)
        days = d0.ts.dt.normalize().to_numpy()
        dev = pd.Series(np.abs(d0.y_t - d0.B) - np.abs(d0.y_t - Fc)).groupby(days).sum() / 1000.0
        spread = (d0.rtm_mcp_rs_mwh - d0.dam_mcp_rs_mwh).fillna(0.0)
        imb = pd.Series((Fc - d0.B) * spread).groupby(days).sum() / CRORE
        summ[tag] = dict(
            bias_base_mw=float(np.mean(d0.y_t - d0.B)), bias_corrected_mw=float(np.mean(d0.y_t - Fc)),
            mae_base_mw=float(np.mean(np.abs(d0.y_t - d0.B))), mae_corrected_mw=float(np.mean(np.abs(d0.y_t - Fc))),
            rmse_base_mw=float(np.sqrt(np.mean((d0.y_t - d0.B) ** 2))),
            rmse_corrected_mw=float(np.sqrt(np.mean((d0.y_t - Fc) ** 2))),
            deviation_energy_saved_gwh_yr=daily_ci(dev.to_numpy(), scale=365.0),
            imbalance_saved_rs_crore_yr=daily_ci(imb.to_numpy(), scale=365.0))
        log("R3", tag, json.dumps({k: v for k, v in summ[tag].items() if not isinstance(v, dict)}))
    save_df(pd.DataFrame(rows), "R3", "skill_refreshed_vs_stale.csv")
    save_json(summ, "R3", "energy_refreshed_vs_stale.json")


# --------------------------------------------------------------------------------------------------- R4
REFINEMENTS = [("global->series", None, "sid"), ("series->series_x_hour_block", "sid", "hour_block"),
               ("series->series_x_hour", "sid", "hour")]
POLICIES = [("time_rolling_30", dict(window=30)), ("time_rolling_90", dict(window=90)),
            ("time_rolling_180", dict(window=180)), ("time_forget_0.98", dict(lam=0.98)),
            ("time_forget_0.995", dict(lam=0.995))]


def _keys(f, coarse, fine):
    ck = np.zeros(len(f), int).astype(str) if coarse is None else f[coarse].astype(str).to_numpy()
    fk = f.sid.astype(str).to_numpy() if fine == "sid" else np.asarray(G.cell_keys(f.sid, f[fine])).astype(str)
    return ck, fk


def _realised(W, Mm, rW, rM, coarse, fine):
    """Gain of the fine over the coarse gate on month rows Mm, both fitted on window W (clipped as in the ladder)."""
    RW = W.R.to_numpy()
    if coarse is None:
        gc_m = np.full(len(Mm), G.ls_gate(RW, rW))
        gf_m = G.PartitionGate(30).fit(RW, rW, W.sid.to_numpy()).predict(Mm.sid.to_numpy())
    else:
        gc_m = G.PartitionGate(30).fit(RW, rW, W.sid.to_numpy()).predict(Mm.sid.to_numpy())
        gf_m = G.PartitionGate(30).fit(RW, rW, G.cell_keys(W.sid, W[fine])).predict(
            G.cell_keys(Mm.sid, Mm[fine]), parent_values=gc_m)
    R = Mm.R.to_numpy()
    lc = np.mean((R - np.clip(gc_m, *G.CLIP) * rM) ** 2)
    lf = np.mean((R - np.clip(gf_m, *G.CLIP) * rM) ** 2)
    return float(lc - lf)


def block_r4():
    ds, df, cols = hourly()
    feats = ds["feats"]
    rows, repro = [], []
    months = pd.date_range(TEST0, "2026-09-01", freq="MS")
    for base in BASES:
        x = ladder_frame(df, cols[base], base).dropna(subset=feats + ["y_t", "B"]).copy()
        x["R"] = x.y_t - x.B
        x["day"] = x.ts.dt.normalize()
        tr = x[x.split == "train"]
        ev = x[x.split.isin(["val", "test"])].copy()
        for s in SEEDS:
            corr = ladder._fit_corrector(tr[feats].to_numpy(np.float32), tr.R.to_numpy(), ladder.LadderConfig(), s)
            ev["r0"] = corr.predict(ev[feats].to_numpy(np.float32))
            va = ev[ev.split == "val"]
            g1 = G.ls_gate(va.R.to_numpy(), va.r0.to_numpy())
            st = ih1(base, s).merge(ev[["sid", "ts", "r0"]], on=["sid", "ts"])
            repro.append(dict(base=base, seed=s, max_abs_diff_r=float(np.max(np.abs(st.r - st.r0 * g1))),
                              mean_abs_r=float(np.mean(np.abs(st.r))), rows=len(st)))
            o = ih1(base, s)
            o["day"] = o.ts.dt.normalize()
            o["R"] = o.y_t - o.B
            dg = json.load(open(os.path.join(CONF, "IH1", f"{base}_s{s}.json")))
            for m0 in months:
                m1 = min(m0 + pd.offsets.MonthEnd(0), TEST1 - pd.Timedelta(days=1))
                Mm = ev[(ev.day >= m0) & (ev.day <= m1)]
                om = o[(o.day >= m0) & (o.day <= m1)]
                if len(Mm) == 0 or len(om) == 0:
                    continue
                wend = m0 - pd.Timedelta(days=2)
                wroll = ev[(ev.day >= max(VAL0, wend - pd.Timedelta(days=181))) & (ev.day <= wend)]
                l_exp = np.mean((om.R - om.g_time_expanding * om.r) ** 2)
                for wname, W in (("rolling", wroll), ("fixed", va)):
                    gw = G.ls_gate(W.R.to_numpy(), W.r0.to_numpy())
                    rW, rM = W.r0.to_numpy() * gw, Mm.r0.to_numpy() * gw
                    for name, coarse, fine in REFINEMENTS:
                        ck, fk = _keys(W, coarse, fine)
                        Gp = S.part2_partition(W.R.to_numpy(), rW, ck, fk, W.day.to_numpy(), "interleaved", 6, 30)["G"]
                        rows.append(dict(base=base, seed=s, month=str(m0.date())[:7], window=wname, rule="PART",
                                         refinement=name, predicted=float(Gp),
                                         realised=_realised(W, Mm, rW, rM, coarse, fine)))
                    for pol, kw in POLICIES:
                        if wname == "rolling":
                            Gp = S.prequential_time(W.R.to_numpy(), rW, W.sid.to_numpy(), W.day.to_numpy(), delay=2,
                                                    **kw)["G"]
                        else:
                            Gp = dg.get("prequential", {}).get(f"expanding->{pol}", np.nan)
                        rows.append(dict(base=base, seed=s, month=str(m0.date())[:7], window=wname, rule="prequential",
                                         refinement=pol, predicted=float(Gp),
                                         realised=float(l_exp - np.mean((om.R - om["g_" + pol] * om.r) ** 2))))
        log("R4", base)
    R = pd.DataFrame(rows)
    save_df(R, "R4", "rolling_origin.csv")
    save_df(pd.DataFrame(repro), "R4", "corrector_reproduction.csv")
    cell = R.groupby(["base", "month", "window", "rule", "refinement"], as_index=False)[["predicted", "realised"]].mean()
    cell["agree"] = np.sign(cell.predicted) == np.sign(cell.realised)
    summ = {}
    for (rule, wname), g in cell.groupby(["rule", "window"]):
        summ[f"{rule}|{wname}"] = dict(sign_accuracy=float(g.agree.mean()), n=int(len(g)),
                                       by_month={m: float(h.agree.mean()) for m, h in g.groupby("month")},
                                       by_refinement={m: float(h.agree.mean()) for m, h in g.groupby("refinement")})
    save_json(summ, "R4", "summary.json")
    log("R4 summary", json.dumps({k: (round(v["sign_accuracy"], 3), v["n"]) for k, v in summ.items()}))


# --------------------------------------------------------------------------------------------------- R5
ETAS = (0.005, 0.01, 0.02, 0.05, 0.1)


def _pinball(e, q, tau=0.99):
    u = np.asarray(e, float) - np.asarray(q, float)
    return float(np.mean(u * (tau - (u < 0))))


def _track(P, fname, qname, eta):
    """Online quantile tracking per series with the study's two-day delay (protocol R5). P has a 0..n-1 index."""
    res = np.full(len(P), np.nan)
    for sid, g in P.groupby("sid"):
        g = g.sort_values("ts")
        e = (g.y_t - g[f"F_{fname}"]).to_numpy(float)
        q = g[f"up99_{fname}_{qname}"].to_numpy(float)
        S_s = float(np.mean(np.abs(q)))
        day = g.ts.dt.normalize().to_numpy()
        days = np.unique(day)
        idx = np.searchsorted(days, day)
        off, c = np.zeros(len(days)), np.zeros(len(days))
        for i in range(len(days)):
            if i >= 1:
                off[i] = off[i - 1] + (eta * S_s * c[i - 2] if i >= 2 else 0.0)
            m = idx == i
            c[i] = np.sum((e[m] > q[m] + off[i]).astype(float) - 0.01)
        res[g.index.to_numpy()] = q + off[idx]
    return res


def _reserve_stats(P, fname, reserve):
    e = (P.y_t - P[f"F_{fname}"]).to_numpy(float)
    days = P.ts.dt.normalize().nunique()
    return dict(coverage=float(np.mean(e <= reserve)), mean_reserve_mw=float(np.mean(np.maximum(reserve, 0.0))),
                pinball99=_pinball(e, reserve),
                shortfall_gwh_yr=float(np.sum(np.maximum(0.0, e - reserve)) / 1000.0 * 365.0 / days))


def block_r5():
    conf = [pd.read_parquet(os.path.join(CONF, "prob_energy", f"prob_energy_s{s}.parquet")) for s in SEEDS]
    dev = pd.read_parquet(os.path.join(DEVR, "prob_energy", "prob_energy_s0.parquet")).reset_index(drop=True)
    fnames = [c[2:] for c in conf[0].columns if c.startswith("F_")]
    cov = []
    for s, P in zip(SEEDS, conf):
        mon = P.ts.dt.strftime("%Y-%m")
        for f in fnames:
            e = P.y_t - P[f"F_{f}"]
            for col in [c for c in P.columns if c.startswith(f"up99_{f}_q_")]:
                hit = (e <= P[col]).astype(float)
                for (sid, m), v in hit.groupby([P.sid, mon]).mean().items():
                    cov.append(dict(seed=s, forecast=f, quantiles=col.replace(f"up99_{f}_", ""), sid=sid, month=m,
                                    coverage=v))
    cov = pd.DataFrame(cov).groupby(["forecast", "quantiles", "sid", "month"], as_index=False).coverage.mean()
    save_df(cov, "R5", "monthly_coverage.csv")
    tune = {}
    for eta in ETAS:
        vals = []
        for f in ("lgbm", "corr_series"):
            r = _track(dev, f, "q_zone_hour", eta)
            scale = float(np.mean(np.abs(dev[f"up99_{f}_q_zone_hour"])))
            vals.append(_reserve_stats(dev, f, r)["pinball99"] / scale)
        tune[str(eta)] = float(np.mean(vals))
    eta = float(min(tune, key=tune.get))
    out = dict(eta_grid=tune, eta=eta, chosen_on="development test months 1 Dec 2025 - 28 Feb 2026 (seed 0)")
    for f in ("lgbm", "corr_series"):
        for sid_sel in ("all", "demand", "net_demand"):
            res = {"fixed": [], "tracked": []}
            for P in conf:
                Ps = (P if sid_sel == "all" else P[P.sid == sid_sel]).reset_index(drop=True)
                res["fixed"].append(_reserve_stats(Ps, f, Ps[f"up99_{f}_q_zone_hour"].to_numpy(float)))
                res["tracked"].append(_reserve_stats(Ps, f, _track(Ps, f, "q_zone_hour", eta)))
            out[f"{f}|{sid_sel}"] = {k: {m: float(np.mean([d[m] for d in v])) for m in v[0]} for k, v in res.items()}
    save_json(out, "R5", "online_quantile_tracking.json")
    log("R5 eta", eta, json.dumps(out.get("lgbm|demand")))


# --------------------------------------------------------------------------------------------------- R6 / R7
def _demand_energy_frame():
    conf = [pd.read_parquet(os.path.join(CONF, "prob_energy", f"prob_energy_s{s}.parquet")) for s in SEEDS]
    keep = ["ts", "y_t", "B", "F_lgbm", "dam_mcp_rs_mwh", "rtm_mcp_rs_mwh"]
    d = conf[0][conf[0].sid == "demand"][keep].sort_values("ts").reset_index(drop=True)
    d["F_corr"] = np.mean([c[c.sid == "demand"].sort_values("ts").F_corr_series.to_numpy(float) for c in conf], axis=0)
    d["day"] = d.ts.dt.normalize()
    return d


def _dsm_cost(y, F, dam, rtm):
    """DSM-style buyer cost relative to buying y day-ahead (protocol R6 d): over-drawal at lambda, under 0.9 lambda."""
    lam = np.maximum(dam, rtm)
    over, under = np.maximum(y - F, 0.0), np.maximum(F - y, 0.0)
    return over * (lam - dam) + under * (dam - 0.9 * lam)


def block_r6():
    d = _demand_energy_frame()
    priced = (d.dam_mcp_rs_mwh.notna() & d.rtm_mcp_rs_mwh.notna()).to_numpy()
    n_all, n_pr = d.day.nunique(), d.loc[priced, "day"].nunique()
    spread = d.rtm_mcp_rs_mwh - d.dam_mcp_rs_mwh
    corr = d.F_corr - d.F_lgbm
    out = dict(days_all=int(n_all), days_priced=int(n_pr), mean_correction_mw=float(corr.mean()),
               mean_spread_rs_mwh=float(spread[priced].mean()))
    a = (corr * spread.fillna(0.0) / CRORE).groupby(d.day).sum()
    out["a_preregistered"] = daily_ci(a.to_numpy(), scale=365.0)
    dp = d[priced]
    cp, sp = corr[priced], spread[priced]
    out["b_priced_days_only"] = daily_ci((cp * sp / CRORE).groupby(dp.day).sum().to_numpy(), scale=365.0)
    c = ((np.abs(dp.y_t - dp.F_lgbm) - np.abs(dp.y_t - dp.F_corr)) * dp.dam_mcp_rs_mwh / CRORE).groupby(dp.day).sum()
    out["c_symmetric_dam_price"] = daily_ci(c.to_numpy(), scale=365.0)
    dsm = ((_dsm_cost(dp.y_t, dp.F_lgbm, dp.dam_mcp_rs_mwh, dp.rtm_mcp_rs_mwh)
            - _dsm_cost(dp.y_t, dp.F_corr, dp.dam_mcp_rs_mwh, dp.rtm_mcp_rs_mwh)) / CRORE).groupby(dp.day).sum()
    out["d_dsm_style_buyer"] = daily_ci(dsm.to_numpy(), scale=365.0)
    mon = dp.ts.dt.strftime("%Y-%m")
    out["e_by_month_rs_crore"] = {m: float(v) for m, v in (cp * sp / CRORE).groupby(mon).sum().items()}
    tot = float((cp * sp).sum()) / CRORE * 365.0 / n_pr
    mean_part = float(len(cp) * cp.mean() * sp.mean()) / CRORE * 365.0 / n_pr
    out["f_decomposition_rs_crore_yr"] = dict(total=tot, mean_correction_x_mean_spread=mean_part,
                                              covariance_part=tot - mean_part)
    save_json(out, "R6", "imbalance_sensitivity.json")
    log("R6", json.dumps({k: (round(v["mean"], 1) if isinstance(v, dict) and "mean" in v else v)
                          for k, v in out.items() if not k.startswith("e_")}))


def block_r7():
    d = _demand_energy_frame()
    dev = ((np.abs(d.y_t - d.F_lgbm) - np.abs(d.y_t - d.F_corr)) / 1000.0).groupby(d.day).sum().to_numpy()
    spread = (d.rtm_mcp_rs_mwh - d.dam_mcp_rs_mwh).fillna(0.0)
    imb = ((d.F_corr - d.F_lgbm) * spread / CRORE).groupby(d.day).sum().to_numpy()
    out = {}
    for name, x in (("deviation_energy_saved_gwh_yr", dev), ("imbalance_saved_rs_crore_yr", imb)):
        out[name] = [daily_ci(x, scale=365.0, block=b) for b in (7, 14, 28)] + [stationary_ci(x, 7, scale=365.0)]
    save_json(out, "R7", "block_length_hourly.json")
    log("R7", json.dumps({k: [(round(v["lo"]), round(v["hi"])) for v in vs] for k, vs in out.items()}))


BLOCKS = dict(R1=block_r1, R1b=block_r1b, R2=block_r2, R3=block_r3, R4=block_r4, R5=block_r5, R6=block_r6,
              R7=block_r7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="R1,R1b,R2,R3,R4,R5,R6,R7")
    a = ap.parse_args()
    for b in a.blocks.split(","):
        log("start", b)
        try:
            BLOCKS[b]()
        except Exception as exc:                                  # one failing block must not stop the others
            import traceback
            traceback.print_exc()
            log("FAILED", b, repr(exc))
            continue
        log("done", b)
    log("DONE explore_ecm_revision")


if __name__ == "__main__":
    main()
