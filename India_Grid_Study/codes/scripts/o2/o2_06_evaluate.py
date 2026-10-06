"""O2 step 6: development evaluation (protocol §4–§5, v1.2).

For each target: common evaluation rows (validation + development test, all available models present, y observed);
point metrics (MASE, RMSSE); conformal calibration at C0/C1/C2 (CQR per quantile level for models with quantile
heads, residual conformal around the median otherwise; scaled by the series' MASE scale; cells < 30 fall back to the
parent level); level selection on validation by alternating ISO weeks; rolling conformal (180/90 days of
published residuals); probabilistic metrics on the development
test; DM-HLN and 7-day block-bootstrap tests against the seasonal naive; seed robustness; latency cost (S-lat).

Outputs: results/dev/o2/eval/{point.csv, prob.csv, tests.csv, seeds.csv, latency.csv, selection.json}
         08_tables/dev_o2_summary.md
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import o2data as O  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import RES, TAB  # noqa: E402
from refused.probabilistic import conformal_quantile, crps_from_quantiles  # noqa: E402

OUT = os.path.join(RES, O.PHASE_DIR, "o2")
EV = os.path.join(OUT, "eval")
QN = [f"q{int(round(q * 100)):02d}" for q in O.QUANTILES]
LEV = np.array(O.QUANTILES)
ANCHORS = ["seasonal_naive", "persistence", "ma7"]
LEARNED = ["lgbm", "bilstm", "tft", "patchtst", "chronos"]
CELLS = {"C0": ["H"], "C1": ["H", "sid"], "C2": ["H", "sid", "season"]}
MIN_CELL = 30
DEV_SEL = ({s["tid"]: s for s in json.load(open(os.path.join(RES, "dev", "o2", "eval", "selection.json")))}
           if O.PHASE == "confirm" else {})


def load_preds(tid):
    frames = {}
    for f in glob.glob(os.path.join(OUT, "preds", f"{tid}__*.parquet")):
        name = os.path.basename(f)[len(tid) + 2:-8]
        if name.endswith(("_seeds", "_instant")):
            continue
        d = pd.read_parquet(f)
        d = d[d.split.isin(["validation", "dev_test"])]
        for c in QN:
            if c not in d:
                d[c] = np.nan
        frames[name] = d
    return frames


def common_rows(frames):
    keys = ["sid", "issue_date", "H"]
    base = None
    for name, d in frames.items():
        k = d.loc[d.q50.notna() & d.y.notna(), keys]
        base = k if base is None else base.merge(k, on=keys)
    return base


def has_quantiles(d):
    return d[QN].notna().all(axis=1).mean() > 0.99


def shift_tables(cal, quant):
    """Conformal shift per cell and quantile level, in scaled units."""
    out = {}
    for lvl, keys in CELLS.items():
        rows = []
        for k, g in cal.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            rec = dict(zip(keys, k), n=len(g))
            for j, c in enumerate(QN):
                r = ((g.y - (g[c] if quant else g.q50)) / g.scale_mae).to_numpy()
                rec[c] = conformal_quantile(r, LEV[j])
            rows.append(rec)
        out[lvl] = pd.DataFrame(rows)
    return out


def apply_level(app, tables, level, quant):
    """Calibrated quantiles for rows of `app` at a level with hierarchical fallback to parents."""
    order = {"C0": ["C0"], "C1": ["C1", "C0"], "C2": ["C2", "C1", "C0"]}[level]
    shift = pd.DataFrame(np.nan, index=app.index, columns=QN)
    for lvl in order:
        t = tables[lvl]
        t = t[t.n >= MIN_CELL] if lvl != "C0" else t
        m = app[CELLS[lvl]].merge(t, on=CELLS[lvl], how="left")
        m.index = app.index
        fill = shift[QN[0]].isna()
        shift.loc[fill, QN] = m.loc[fill, QN].to_numpy()
    base = app[QN].to_numpy() if quant else np.repeat(app.q50.to_numpy()[:, None], len(QN), axis=1)
    Q = base + shift.to_numpy() * app.scale_mae.to_numpy()[:, None]
    Q = np.where(np.isfinite(Q), Q, base)
    return np.sort(Q, axis=1)


def rolling_level(allrows, app, level, quant, Ly, window):
    """Rolling conformal: for each row with issue day τ, shifts from residuals whose target date ≤ τ − Ly (published)
    and > τ − Ly − window, in the row's cell at `level`; cells with < 30 residuals fall back to the parent level.
    Scaled units. Rows without any residual history get NaN (excluded from that variant's metrics)."""
    order = {"C0": ["C0"], "C1": ["C1", "C0"], "C2": ["C2", "C1", "C0"]}[level]
    hist = allrows.reset_index(drop=True)
    app = app.reset_index(drop=True)
    R = ((hist.y.to_numpy()[:, None] - (hist[QN].to_numpy() if quant else hist.q50.to_numpy()[:, None]))
         / hist.scale_mae.to_numpy()[:, None])
    hd = hist.target_date.to_numpy().astype("datetime64[D]").astype(np.int64)
    hi = (app.issue_date.to_numpy().astype("datetime64[D]").astype(np.int64) - Ly)
    lo = hi - window
    out = np.full((len(app), len(QN)), np.nan)
    for lvl in order:
        keys = CELLS[lvl]
        need = np.isnan(out[:, 0])
        if not need.any():
            break
        hk = hist[keys].astype(str).agg("|".join, axis=1).to_numpy()
        ak = app[keys].astype(str).agg("|".join, axis=1).to_numpy()
        for key in np.unique(ak[need]):
            hm = np.where(hk == key)[0]
            if len(hm) == 0:
                continue
            order_h = hm[np.argsort(hd[hm])]
            dates = hd[order_h]
            rows = np.where((ak == key) & need)[0]
            a = np.searchsorted(dates, lo[rows], side="right")
            b = np.searchsorted(dates, hi[rows], side="right")
            for r, s0, s1 in zip(rows, a, b):
                n = s1 - s0
                if n == 0 or (lvl != "C0" and n < MIN_CELL):
                    continue
                win = R[order_h[s0:s1]]
                out[r] = [conformal_quantile(win[:, j if quant else 0], LEV[j]) for j in range(len(QN))]
    base = app[QN].to_numpy() if quant else np.repeat(app.q50.to_numpy()[:, None], len(QN), axis=1)
    Q = base + out * app.scale_mae.to_numpy()[:, None]
    return np.sort(np.where(np.isfinite(Q), Q, np.nan), axis=1)


def prob_metrics(d, Q):
    ok = np.isfinite(Q).all(axis=1)
    d, Q = d[ok], Q[ok]
    y = d.y.to_numpy()
    sc = d.scale_mae.to_numpy()
    u = (y[:, None] - Q) / sc[:, None]
    pin = np.maximum(LEV[None, :] * u, (LEV[None, :] - 1) * u)
    cov80 = (y >= Q[:, 1]) & (y <= Q[:, 5])
    cov90 = (y >= Q[:, 0]) & (y <= Q[:, 6])
    per_series = pd.Series(cov90).groupby(d.sid.to_numpy()).mean()
    return dict(n_scored=int(ok.sum()), pinball=float(pin.mean()), crps_approx=float(crps_from_quantiles(y / sc, Q / sc[:, None], LEV).mean()),
                cov80=float(cov80.mean()), cov90=float(cov90.mean()),
                cov90_series_mad=float((per_series - 0.9).abs().mean()),
                width90_scaled=float(((Q[:, 6] - Q[:, 0]) / sc).mean())), pin.mean(axis=1)


def point_metrics(d, pred):
    e = d.y - pred
    g = pd.DataFrame({"sid": d.sid, "H": d.H, "ae": e.abs() / d.scale_mae, "se": e ** 2 / d.scale_mse})
    per = g.groupby(["sid", "H"]).agg(mase=("ae", "mean"), msse=("se", "mean"))
    return dict(mase=float(per.mase.mean()), rmsse=float(np.sqrt(per.msse).mean())), per


def hln(d, h):
    """Diebold–Mariano with HLN small-sample correction; one-sided p for mean(d) < 0 (model better)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    t, _ = S.dm_hac(d, h)
    t *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return float(t), float(st.t.cdf(t, df=n - 1))


def daily_diff(d, loss_a, loss_b):
    x = pd.DataFrame({"date": d.target_date.to_numpy(), "diff": (loss_a - loss_b)})
    return x.groupby("date")["diff"].mean().sort_index().to_numpy()


def evaluate_target(tid):
    frames = load_preds(tid)
    if "seasonal_naive" not in frames:
        return None
    rows = common_rows(frames)
    scales = pd.read_parquet(os.path.join(OUT, "scales.parquet"))
    scales = scales[scales.tid == tid][["sid", "H", "scale_mae", "scale_mse"]]
    D = {}
    for name, d in frames.items():
        d = d.merge(rows, on=["sid", "issue_date", "H"]).merge(scales, on=["sid", "H"])
        d = d[np.isfinite(d.scale_mae) & (d.scale_mae > 0)].sort_values(["sid", "issue_date", "H"]).reset_index(drop=True)
        D[name] = d
    ref = D["seasonal_naive"]
    point, prob, tests, sel = [], [], [], {"tid": tid, "levels": {}, "rows": len(ref)}
    mae_sn = {sp: float((ref[ref.split == sp].y - ref[ref.split == sp].q50).abs().div(ref[ref.split == sp].scale_mae).mean())
              for sp in ("validation", "dev_test")}
    for name, d in D.items():
        quant = has_quantiles(d)
        for split in ("validation", "dev_test"):
            s = d[d.split == split]
            pm, per = point_metrics(s, s.q50)
            rel = float((s.y - s.q50).abs().div(s.scale_mae).mean()) / mae_sn[split]
            point.append(dict(tid=tid, model=name, split=split, **pm, scaled_mae_rel_sn=rel, n=len(s)))
            for H in O.HORIZONS:
                sh = s[s.H == H]
                pmh, _ = point_metrics(sh, sh.q50)
                point.append(dict(tid=tid, model=name, split=split, H=H, **pmh, n=len(sh)))
        # calibration-level selection on validation, alternating ISO weeks
        val = d[d.split == "validation"].copy()
        wk = val.target_date.dt.isocalendar().week.to_numpy() % 2
        scores = {}
        for lvl in CELLS:
            sc = []
            for a in (0, 1):
                cal, ev = val[wk == a], val[wk != a]
                Q = apply_level(ev, shift_tables(cal, quant), lvl, quant)
                sc.append(prob_metrics(ev, Q)[0]["pinball"])
            scores[lvl] = float(np.mean(sc))
        chosen = min(scores, key=scores.get)
        if O.PHASE == "confirm":  # frozen from development (pre-registration); validation scores reported only
            chosen = DEV_SEL[tid]["levels"].get(name, {}).get("chosen", chosen)
        sel["levels"][name] = dict(scores=scores, chosen=chosen, quantile_head=bool(quant))
        tables = shift_tables(val, quant)
        dev = d[d.split == "dev_test"]
        variants = {lvl: apply_level(dev, tables, lvl, quant) for lvl in CELLS}
        Ly = O.lags().get(O.TARGETS[tid]["col"], 1)
        both = d[d.split.isin(["validation", "dev_test"])]
        for W in (180, 90):
            variants[f"roll{W}_{chosen}"] = rolling_level(both, dev, chosen, quant, Ly, W)
        if quant:
            variants["raw"] = np.sort(dev[QN].to_numpy(), axis=1)
        pin_rows = {}
        for v, Q in variants.items():
            m, pr = prob_metrics(dev, Q)
            pin_rows[v] = pr
            prob.append(dict(tid=tid, model=name, variant=v, chosen=(v == chosen), **m, n=len(dev)))
        # D2 pinball comparisons on the development test (daily mean differentials)
        for a, b in (("C1", "C0"), ("C2", "C1")):
            dd = daily_diff(dev, pin_rows[a], pin_rows[b])
            t, p = hln(dd, 1)
            ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
            tests.append(dict(tid=tid, model=name, test=f"pinball_{a}_vs_{b}", H="all", mean_diff=float(dd.mean()),
                              lo=ci["lo"], hi=ci["hi"], t_hln=t, p_one_sided=p, n_days=len(dd)))
    # selection among learned models on validation MASE
    pv = pd.DataFrame(point)
    cand = pv[(pv.split == "validation") & pv.H.isna() & pv.model.isin(LEARNED)]
    sel["selected"] = cand.sort_values("mase").model.iloc[0] if len(cand) else None
    if O.PHASE == "confirm":
        sel["selected"] = DEV_SEL[tid]["selected"]
    # tests vs seasonal naive on the development test, per H: pooled daily mean scaled |e| differential
    devref = ref[ref.split == "dev_test"]
    for name in [m for m in D if m != "seasonal_naive"]:
        dev = D[name][D[name].split == "dev_test"]
        for H in O.HORIZONS:
            a, b = dev[dev.H == H], devref[devref.H == H]
            la = ((a.y - a.q50).abs() / a.scale_mae).to_numpy()
            lb = ((b.y - b.q50).abs() / b.scale_mae).to_numpy()
            dd = daily_diff(a, la, lb)
            t, p = hln(dd, H)
            ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
            per_series = []
            for sid, g in a.groupby("sid"):
                gb = b[b.sid == sid]
                ts, ps = hln((g.y - g.q50).abs().to_numpy() - (gb.y - gb.q50).abs().to_numpy(), H)
                per_series.append(ps)
            per_series = np.array(per_series, float)
            ok = np.isfinite(per_series)
            holm_s = S.holm(per_series[ok]) if ok.any() else []
            tests.append(dict(tid=tid, model=name, test="mase_vs_seasonal_naive", H=H, mean_diff=float(dd.mean()),
                              lo=ci["lo"], hi=ci["hi"], t_hln=t, p_one_sided=p, n_days=len(dd),
                              series_sig_raw=int((per_series[ok] < 0.05).sum()),
                              series_sig_holm=int((np.asarray(holm_s) < 0.05).sum()), n_series=int(ok.sum())))
    return pd.DataFrame(point), pd.DataFrame(prob), pd.DataFrame(tests), sel


def seeds_and_latency(tid, scales):
    out_s, out_l = [], []
    sc = scales[scales.tid == tid][["sid", "H", "scale_mae", "scale_mse"]]
    ref = pd.read_parquet(os.path.join(OUT, "preds", f"{tid}__seasonal_naive.parquet"))
    ref = ref[ref.split == "dev_test"].merge(sc, on=["sid", "H"])
    for f in glob.glob(os.path.join(OUT, "preds", f"{tid}__*_seeds.parquet")):
        name = os.path.basename(f)[len(tid) + 2:-len("_seeds.parquet")]
        d = pd.read_parquet(f)
        d = d[(d.split == "dev_test") & d.y.notna()].merge(sc, on=["sid", "H"])
        for seed, g in d.groupby("seed"):
            j = g.merge(ref[["sid", "issue_date", "H", "q50"]].rename(columns={"q50": "sn"}), on=["sid", "issue_date", "H"])
            j = j[j.sn.notna()]
            pm, _ = point_metrics(j, j.q50)
            dd = daily_diff(j, ((j.y - j.q50).abs() / j.scale_mae).to_numpy(), ((j.y - j.sn).abs() / j.scale_mae).to_numpy())
            t, p = hln(dd, 2)
            out_s.append(dict(tid=tid, model=name, seed=int(seed), mase=pm["mase"], p_vs_sn_H_all=p, n=len(j)))
    fi = os.path.join(OUT, "preds", f"{tid}__lgbm_instant.parquet")
    fs = os.path.join(OUT, "preds", f"{tid}__lgbm_seeds.parquet")
    if os.path.exists(fi) and os.path.exists(fs):
        a = pd.read_parquet(fi)
        b = pd.read_parquet(fs)
        b = b[b.seed == 0]
        j = a[a.split == "dev_test"].merge(b[["sid", "issue_date", "H", "q50"]].rename(columns={"q50": "std"}),
                                           on=["sid", "issue_date", "H"]).merge(sc, on=["sid", "H"])
        j = j[j.y.notna() & j.q50.notna() & j["std"].notna()]
        for H in O.HORIZONS + ["all"]:
            g = j if H == "all" else j[j.H == H]
            mi, _ = point_metrics(g, g.q50)
            ms, _ = point_metrics(g, g["std"])
            dd = daily_diff(g, ((g.y - g.q50).abs() / g.scale_mae).to_numpy(),
                            ((g.y - g["std"]).abs() / g.scale_mae).to_numpy())
            t, p = hln(dd, 2 if H == "all" else H)
            ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
            out_l.append(dict(tid=tid, H=H, mase_instant=mi["mase"], mase_published=ms["mase"],
                              latency_cost_pct=100 * (ms["mase"] / mi["mase"] - 1), mean_diff=float(dd.mean()),
                              lo=ci["lo"], hi=ci["hi"], p_instant_better=p, n=len(g)))
    return out_s, out_l


def main():
    os.makedirs(EV, exist_ok=True)
    tids = sys.argv[1].split(",") if len(sys.argv) > 1 else list(O.TARGETS)
    scales = pd.read_parquet(os.path.join(OUT, "scales.parquet"))
    P, R, T, SEL, SD, LT = [], [], [], [], [], []
    for tid in tids:
        res = evaluate_target(tid)
        if res is None:
            continue
        p, r, t, s = res
        P.append(p), R.append(r), T.append(t), SEL.append(s)
        sd, lt = seeds_and_latency(tid, scales)
        SD += sd
        LT += lt
        print(tid, "evaluated; models:", sorted(p.model.unique()), "selected:", s["selected"], flush=True)
    P, R, T = pd.concat(P), pd.concat(R), pd.concat(T)
    # Holm across targets × H for the selected model's pooled test vs seasonal naive
    selected = {s["tid"]: s["selected"] for s in SEL}
    m = (T.test == "mase_vs_seasonal_naive") & T.apply(lambda r: r.model == selected.get(r.tid), axis=1)
    T.loc[m, "p_holm_family"] = S.holm(T.loc[m, "p_one_sided"].to_numpy())
    P.to_csv(os.path.join(EV, "point.csv"), index=False)
    R.to_csv(os.path.join(EV, "prob.csv"), index=False)
    T.to_csv(os.path.join(EV, "tests.csv"), index=False)
    pd.DataFrame(SD).to_csv(os.path.join(EV, "seeds.csv"), index=False)
    pd.DataFrame(LT).to_csv(os.path.join(EV, "latency.csv"), index=False)
    json.dump(SEL, open(os.path.join(EV, "selection.json"), "w"), indent=1)
    write_summary(P, R, T, SEL, pd.DataFrame(SD), pd.DataFrame(LT))


def write_summary(P, R, T, SEL, SD, LT):
    L = ["# O2 development results (development test FY2024-25 unless stated)", "",
         "Generated by `04_code/scripts/o2/o2_06_evaluate.py`. MASE < 1 beats the seasonal naive at the same lead.", ""]
    for s in SEL:
        tid = s["tid"]
        L += [f"## {tid} — {O.TARGETS[tid]['name']} (selected on validation: **{s['selected']}**)", "",
              "| model | MASE val | MASE dev | rel. SN dev | pinball static | cov80 / cov90 static | pinball roll180 | cov80 / cov90 roll180 | level |",
              "|---|---|---|---|---|---|---|---|---|"]
        pv = P[(P.tid == tid) & P.H.isna()]
        for mdl in ANCHORS + LEARNED:
            if mdl not in set(pv.model):
                continue
            v = pv[(pv.model == mdl) & (pv.split == "validation")].iloc[0]
            dv = pv[(pv.model == mdl) & (pv.split == "dev_test")].iloc[0]
            ch = s["levels"][mdl]["chosen"]
            r = R[(R.tid == tid) & (R.model == mdl) & (R.variant == ch)].iloc[0]
            q = R[(R.tid == tid) & (R.model == mdl) & (R.variant == f"roll180_{ch}")].iloc[0]
            L.append(f"| {mdl} | {v.mase:.3f} | {dv.mase:.3f} | {dv.scaled_mae_rel_sn:.3f} | {r.pinball:.3f} | "
                     f"{r.cov80:.3f} / {r.cov90:.3f} | {q.pinball:.3f} | {q.cov80:.3f} / {q.cov90:.3f} | {ch} |")
        t = T[(T.tid == tid) & (T.model == s["selected"]) & (T.test == "mase_vs_seasonal_naive")]
        L += ["", "Selected model vs seasonal naive (pooled daily mean scaled |e| differential; negative = better):", "",
              "| H | mean diff | 95% block CI | HLN one-sided p | Holm (family) | series significant (Holm) |",
              "|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r.H} | {r.mean_diff:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | {S.fmt_p(r.p_one_sided)} | "
                     f"{S.fmt_p(r.get('p_holm_family', np.nan))} | {r.series_sig_holm}/{r.n_series} |")
        sd = SD[(SD.tid == tid)] if len(SD) else SD
        if len(sd):
            L += ["", "Seeds (development test): " + "; ".join(
                f"{m}: MASE {g.mase.min():.3f}–{g.mase.max():.3f}, seeds with p < 0.05 vs seasonal naive "
                f"{int((g.p_vs_sn_H_all < 0.05).sum())}/{len(g)}" for m, g in sd.groupby("model"))]
        lt = LT[(LT.tid == tid) & (LT.H == "all")] if len(LT) else LT
        if len(lt):
            r = lt.iloc[0]
            L += ["", f"Publication latency (LightGBM seed 0): MASE {r.mase_published:.3f} with measured lags vs "
                      f"{r.mase_instant:.3f} with instant publication → latency cost {r.latency_cost_pct:+.1f}% "
                      f"(p = {S.fmt_p(r.p_instant_better)})."]
        L.append("")
    open(os.path.join(TAB, f"{O.PHASE_DIR}_o2_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
