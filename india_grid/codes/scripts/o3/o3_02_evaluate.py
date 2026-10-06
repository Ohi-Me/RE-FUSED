"""O3 step 2: development evaluation of LA-MCAG (protocol `05_o3_dev_protocol.md` v1.2).

* Online time-adaptive gate (OTG) applied to every variant: Q = B + ĝ(τ)(Q − B), ĝ from the 180 most recent published
  median residuals per series × H (≥ 30, else 1), clipped to [0, 1.5].
* Point (MASE, relative MAE vs seasonal naive) and probabilistic metrics (rolling-180 conformal at level C1 for every
  variant, so calibration is identical across comparisons) on the development test.
* Pairwise tests (daily cross-series mean of scaled |e| and scaled pinball differentials; HLN one-sided; 7-day block
  bootstrap CI): A2, A3, A4, A6, gate ladder (fixed → regime → instance → +OTG), base head vs full model. Ablations
  (3 seeds) are compared with the main variants averaged over the same seeds 0–2.
* L1 source loss: increase in MASE when the RE and weather blocks are missing, instance gates vs fusion.
* F5 price of adaptivity: PART predictions from validation only (part2_partition for fixed → regime, part2_instance for
  → instance, prequential_time for → OTG) against realised development-test gains, per target × H.

Outputs: 06_results/dev/o3/eval/{point.csv, pairs.csv, l1.csv, f5.csv}; 08_tables/dev_o3_summary.md
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
from refused import o2data as O  # noqa: E402
from refused import selection as SEL  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import RES, TAB  # noqa: E402
from refused.report import md_table  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402

O3 = os.path.join(RES, O.PHASE_DIR, "o3")
O2 = os.path.join(RES, O.PHASE_DIR, "o2")
EV = os.path.join(O3, "eval")
QN = E2.QN
BN = ["b" + c[1:] for c in QN]
KEY = ["sid", "issue_date", "H"]
MAIN = ["mcag_fixed", "mcag_regime", "mcag_instance", "fusion_fixed"]
ABL = ["null_context", "permuted_context", "market_only", "no_carbon"]


def otg(d, Ly, window=180, min_n=30):
    """Online gate per series × H from published residuals (target date ≤ τ − Ly)."""
    d = d.sort_values(["sid", "H", "target_date"]).reset_index(drop=True)
    g_all = np.ones(len(d))
    for (_, _), idx in d.groupby(["sid", "H"]).groups.items():
        g = d.loc[idx]
        R = (g.y - g.b50).to_numpy()
        r = (g.q50 - g.b50).to_numpy()
        ok = np.isfinite(R) & np.isfinite(r)
        td = g.target_date.to_numpy().astype("datetime64[D]").astype(np.int64)
        Rr = np.concatenate([[0.0], np.cumsum(np.where(ok, R * r, 0.0))])
        rr = np.concatenate([[0.0], np.cumsum(np.where(ok, r * r, 0.0))])
        nn_ = np.concatenate([[0], np.cumsum(ok.astype(int))])
        hi = g.issue_date.to_numpy().astype("datetime64[D]").astype(np.int64) - Ly
        a = np.searchsorted(td, hi - window, side="right")
        b = np.searchsorted(td, hi, side="right")
        num, den, cnt = Rr[b] - Rr[a], rr[b] - rr[a], nn_[b] - nn_[a]
        gh = np.where((cnt >= min_n) & (den > 1e-12), np.clip(num / np.maximum(den, 1e-300), 0, 1.5), 1.0)
        g_all[np.asarray(idx)] = gh
    out = d.copy()
    Q, B = d[QN].to_numpy(), d[BN].to_numpy()
    out[QN] = np.sort(B + g_all[:, None] * (Q - B), axis=1)
    out["g_otg"] = g_all
    return out


def load(tid, name, seeds=None):
    f = os.path.join(O3, "preds", f"{tid}__{name}{'_seeds' if seeds is not None else ''}.parquet")
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f)
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
        if d.seed.nunique() < len(seeds):
            return None
        cols = [c for c in QN + BN if c in d]
        d = d.groupby(["tid", "sid", "issue_date", "H", "target_date", "split", "season", "y"], dropna=False)[cols] \
            .mean().reset_index()
        d[QN] = np.sort(d[QN].to_numpy(), axis=1)
        d[BN] = np.sort(d[BN].to_numpy(), axis=1)
    return d


def pair_test(a, b, label, tid, metric):
    # comparator columns: its median and its calibrated quantiles (q50 is also in QN, so it must not be listed twice)
    bcols = ["q50"] + ["c" + c for c in QN]
    j = a.merge(b[KEY + bcols].rename(columns={c: c + "_b" for c in bcols}), on=KEY)
    j = j[(j.split == "dev_test") & j.y.notna()]
    out = []
    for H in O.HORIZONS + ["all"]:
        g = j if H == "all" else j[j.H == H]
        if metric == "mae":
            la = ((g.y - g.q50).abs() / g.scale_mae).to_numpy()
            lb = ((g.y - g.q50_b).abs() / g.scale_mae).to_numpy()
        else:
            la = E2.prob_metrics(g, g[["c" + c for c in QN]].to_numpy())[1]
            lb = E2.prob_metrics(g, g[["c" + c + "_b" for c in QN]].to_numpy())[1]
        dd = E2.daily_diff(g, la, lb)
        t, p = E2.hln(dd, 2 if H == "all" else H)
        ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
        out.append(dict(tid=tid, comparison=label, metric=metric, H=H, mean_a=float(np.mean(la)),
                        mean_b=float(np.mean(lb)), mean_diff=float(dd.mean()), lo=ci["lo"], hi=ci["hi"],
                        p_a_better=p, n_days=len(dd)))
    return out


def calibrate(d, Ly):
    """Rolling-180 conformal at C1; adds columns cq05…cq95."""
    both = d[d.split.isin(["validation", "dev_test"])].reset_index(drop=True)
    Q = E2.rolling_level(both, both, "C1", True, Ly, 180)
    for j, c in enumerate(QN):
        both["c" + c] = Q[:, j]
    return both


def main():
    os.makedirs(EV, exist_ok=True)
    scales = pd.read_parquet(os.path.join(O2, "scales.parquet"))
    point, pairs, l1, f5 = [], [], [], []
    for tid in (sys.argv[1].split(",") if len(sys.argv) > 1 else list(O.TARGETS)):
        Ly = O.lags().get(O.TARGETS[tid]["col"], 1)
        sc = scales[scales.tid == tid][["sid", "H", "scale_mae", "scale_mse"]]
        sn = pd.read_parquet(os.path.join(O2, "preds", f"{tid}__seasonal_naive.parquet"))
        sn = sn[sn.split.isin(["validation", "dev_test"])]
        V = {}
        for name in MAIN + ABL:
            d = load(tid, name)
            if d is None:
                continue
            V[name] = d
            V[name + "+otg"] = otg(d, Ly)
            base = d.copy()
            base[QN] = d[BN].to_numpy()
            V[name + ":base"] = base
        if not V:
            continue
        rows = sn.loc[sn.q50.notna() & sn.y.notna(), KEY]
        for d in V.values():
            rows = rows.merge(d.loc[d.q50.notna(), KEY], on=KEY)
        C = {}
        for name, d in V.items():
            d = d.merge(rows, on=KEY).merge(sc, on=["sid", "H"])
            C[name] = calibrate(d, Ly)
        ref = sn.merge(rows, on=KEY).merge(sc, on=["sid", "H"])
        mae_sn = {sp: float(((ref.y - ref.q50).abs() / ref.scale_mae)[ref.split == sp].mean()) for sp in ("validation", "dev_test")}
        for name, d in C.items():
            for sp in ("validation", "dev_test"):
                s = d[d.split == sp]
                pm, _ = E2.point_metrics(s, s.q50)
                pr, _ = E2.prob_metrics(s, s[["c" + c for c in QN]].to_numpy())
                point.append(dict(tid=tid, model=name, split=sp, **pm,
                                  rel_sn=float(((s.y - s.q50).abs() / s.scale_mae).mean()) / mae_sn[sp],
                                  pinball=pr["pinball"], cov80=pr["cov80"], cov90=pr["cov90"], n=len(s)))
        comps = [("A2 instance vs fusion", "mcag_instance", "fusion_fixed"),
                 ("A2+OTG instance vs fusion", "mcag_instance+otg", "fusion_fixed+otg"),
                 ("ladder regime vs fixed", "mcag_regime", "mcag_fixed"),
                 ("ladder instance vs regime", "mcag_instance", "mcag_regime"),
                 ("ladder OTG vs instance", "mcag_instance+otg", "mcag_instance"),
                 ("context vs base head", "mcag_instance+otg", "mcag_instance:base")]
        for label, a, b in comps:
            if a in C and b in C:
                for metric in ("mae", "pinball"):
                    pairs += pair_test(C[a], C[b], label, tid, metric)
        for name, label in [("null_context", "A3 real vs null"), ("permuted_context", "A3 real vs permuted"),
                            ("market_only", "A4 all vs market-only"), ("no_carbon", "A6 all vs no carbon")]:
            if name not in V:
                continue
            m3 = load(tid, "mcag_instance", seeds=[0, 1, 2])
            if m3 is None:
                continue
            for suffix in ("", "+otg"):
                a = otg(m3, Ly) if suffix else m3
                b = V[name + suffix]
                keys = rows
                a = calibrate(a.merge(keys, on=KEY).merge(sc, on=["sid", "H"]), Ly)
                b = calibrate(b.merge(keys, on=KEY).merge(sc, on=["sid", "H"]), Ly)
                for metric in ("mae", "pinball"):
                    pairs += pair_test(a, b, label + suffix, tid, metric)
        # L1 source loss
        for name in ("mcag_instance", "fusion_fixed"):
            dl = load(tid, name + "_L1")
            if dl is None or name not in V:
                l1 = l1
                continue
            j = V[name].merge(dl[KEY + ["q50"]].rename(columns={"q50": "q50_l1"}), on=KEY).merge(sc, on=["sid", "H"])
            j = j[(j.split == "dev_test") & j.y.notna()]
            l1.append(dict(tid=tid, model=name, mase=float(((j.y - j.q50).abs() / j.scale_mae).mean()),
                           mase_source_loss=float(((j.y - j.q50_l1).abs() / j.scale_mae).mean())))
            V[name + "_L1j"] = j
        if "mcag_instance_L1j" in V and "fusion_fixed_L1j" in V:
            a, b = V["mcag_instance_L1j"], V["fusion_fixed_L1j"]
            j = a[KEY + ["target_date", "y", "q50", "q50_l1", "scale_mae"]].merge(
                b[KEY + ["q50", "q50_l1"]], on=KEY, suffixes=("_i", "_f"))
            inc_i = ((j.y - j.q50_l1_i).abs() - (j.y - j.q50_i).abs()) / j.scale_mae
            inc_f = ((j.y - j.q50_l1_f).abs() - (j.y - j.q50_f).abs()) / j.scale_mae
            dd = E2.daily_diff(j, inc_i.to_numpy(), inc_f.to_numpy())
            t, p = E2.hln(dd, 2)
            ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
            l1.append(dict(tid=tid, model="instance − fusion (degradation difference)", mean_diff=float(dd.mean()),
                           lo=ci["lo"], hi=ci["hi"], p_instance_degrades_less=p))
        # F5: PART on validation vs realised development-test gains (MSE, scaled units)
        fx = C.get("mcag_fixed")
        gates = os.path.join(O3, "preds", f"{tid}__mcag_fixed_gates.parquet")
        if fx is not None and os.path.exists(gates):
            gt = pd.read_parquet(gates)
            gt = gt[gt.seed == 0].drop(columns=["seed"])
            stale = [c for c in gt.columns if c.startswith("stale_")]
            for H in O.HORIZONS:
                v = fx[(fx.split == "validation") & (fx.H == H) & fx.y.notna()].merge(
                    gt[["sid", "issue_date", "regime_cell"] + stale], on=["sid", "issue_date"])
                R = ((v.y - v.b50) / v.scale_mae).to_numpy()
                r = ((v.q50 - v.b50) / v.scale_mae).to_numpy()
                doy = v.target_date.dt.dayofyear.to_numpy()
                F = np.column_stack([v[stale].to_numpy(), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
                pred = {
                    "regime vs fixed": SEL.part2_partition(R, r, np.zeros(len(v)).astype(str), v.regime_cell.astype(str),
                                                           v.target_date, mode="interleaved", K=6)["G"],
                    "instance vs fixed": SEL.part2_instance(F, R, r, np.zeros(len(v)).astype(str), v.target_date,
                                                            mode="interleaved", K=4)["G"],
                    "OTG vs fixed": SEL.prequential_time(R, r, v.sid, v.target_date, window=180, delay=Ly)["G"],
                }
                real = {}
                for lab, a in [("regime vs fixed", "mcag_regime"), ("instance vs fixed", "mcag_instance"),
                               ("OTG vs fixed", "mcag_fixed+otg")]:
                    if a not in C:
                        continue
                    j = C[a][(C[a].split == "dev_test") & (C[a].H == H)].merge(
                        fx[(fx.split == "dev_test") & (fx.H == H)][KEY + ["q50"]].rename(columns={"q50": "q50_fx"}), on=KEY)
                    j = j[j.y.notna()]
                    real[lab] = float((((j.y - j.q50_fx) ** 2 - (j.y - j.q50) ** 2) / j.scale_mae ** 2).mean())
                for lab, G in pred.items():
                    f5.append(dict(tid=tid, H=H, refinement=lab, part_G_validation=G, predicted_refine=G > 0,
                                   realised_gain_dev=real.get(lab, np.nan),
                                   agree=(G > 0) == (real.get(lab, np.nan) > 0) if lab in real else np.nan))
        print(tid, "O3 evaluated:", sorted(k for k in V if ":base" not in k), flush=True)
    point, pairs, l1, f5 = map(pd.DataFrame, (point, pairs, l1, f5))
    point.to_csv(os.path.join(EV, "point.csv"), index=False)
    pairs.to_csv(os.path.join(EV, "pairs.csv"), index=False)
    l1.to_csv(os.path.join(EV, "l1.csv"), index=False)
    f5.to_csv(os.path.join(EV, "f5.csv"), index=False)
    summary(point, pairs, l1, f5)


def summary(point, pairs, l1, f5):
    L = ["# O3 development results — LA-MCAG (development test FY2024-25)", "",
         "MASE uses train-period seasonal-naive scales; rel. SN is the scaled MAE ratio to the seasonal naive on the same "
         "rows. Intervals: rolling-180 conformal at C1 for every variant. `+otg` = online time-adaptive gate; "
         "`:base` = base head without context.", ""]
    for tid, g in point[point.split == "dev_test"].groupby("tid"):
        L += [f"## {tid} — {O.TARGETS[tid]['name']}", "", "| variant | MASE | rel. SN | pinball | cov80 | cov90 |",
              "|---|---|---|---|---|---|"]
        for _, r in g.sort_values("mase").iterrows():
            L.append(f"| {r.model} | {r.mase:.3f} | {r.rel_sn:.3f} | {r.pinball:.3f} | {r.cov80:.3f} | {r.cov90:.3f} |")
        pp = pairs[(pairs.tid == tid) & (pairs.H == "all")] if len(pairs) else pairs
        if len(pp):
            L += ["", "| comparison (a vs b) | metric | mean diff (a − b) | 95% block CI | p (a better) |", "|---|---|---|---|---|"]
            for _, r in pp.iterrows():
                L.append(f"| {r.comparison} | {r.metric} | {r.mean_diff:+.4f} | [{r.lo:+.4f}, {r.hi:+.4f}] | {S.fmt_p(r.p_a_better)} |")
        L.append("")
    if len(l1):
        L += ["## L1 source loss (RE and weather blocks missing on the development test)", "", md_table(l1, 4), ""]
    if len(f5):
        ok = f5.dropna(subset=["realised_gain_dev"])
        L += ["## F5 price of adaptivity — PART (validation) vs realised gain (development test)", "",
              f"Sign agreement: {int(ok.agree.sum())}/{len(ok)} cells.", "", md_table(f5, 5), ""]
    open(os.path.join(TAB, f"{O.PHASE_DIR}_o3_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
