"""O4 step 1: context-aware deviation assessment (protocol `02_design/04_o4_o5_dev_protocol.md`, O4).

A = |Δ| · [λ/median(λ) + κ (w₁R + w₂S + w₃C + w₄U)], Δ = OD/UD (GWh), λ = DSM normal rate (bid area, day t).
Ex-ante context at issue day t−1: R regional RE ramp (PSP, data to t−2), S system stress (outage share and coal
capacity share < 7 days at t−3, shortage > 0 at t−2), C carbon intensity (t−3), U width of the calibrated 80 % interval
of the day-ahead drawal forecast (O2 T2, H = 1, rolling-180 conformal at C1) divided by the MASE scale.
R, S components and C are mapped to [0, 1] by their FY2018-23 empirical CDF; U by its FY2023-24 CDF.
Consequences (never inputs): Y₁ intra-day severity, Y₂ shortage > 0, Y₃ All-India % time < 49.9 Hz; composite Y.
Weights and κ: grid on validation (FY2023-24) maximising mean within-M₀-decile Kendall τ(A, Y); frozen and scored on
the development test (FY2024-25). Tests: A7 (A vs M₀), U vs placebos (50 within-State permutations, 50 Gaussian
draws; weights refit for each), regime/season strata, variance decomposition of log(1 + A).

Usage: py -3.10 o4_01_assessment.py [O2 model name for U, default = O2 selection for T2] [--quick]
Outputs: 06_results/dev/o4/{assessment_rows.parquet, weights.json, tests.csv, strata.csv, placebo.csv, variance.csv}
         08_tables/dev_o4_summary.md
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
from refused import o2data as O  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import RES, TAB  # noqa: E402
from refused.report import md_table  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402

OUT = os.path.join(RES, O.PHASE_DIR, "o4")
O2 = os.path.join(RES, O.PHASE_DIR, "o2")
KAPPAS = [0.25, 0.5, 1.0, 2.0]
R3_START = pd.Timestamp("2024-09-16")


def simplex(step=0.1, k=4):
    n = int(round(1 / step))
    return [np.array(c) / n for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n]


def ecdf_map(values, ref):
    ref = np.sort(ref[np.isfinite(ref)])
    return np.where(np.isfinite(values), np.searchsorted(ref, values, side="right") / max(len(ref), 1), np.nan)


def pct_rank(s):
    return s.rank(pct=True, method="average")


def within_decile_tau(score, y, m0):
    """Mean Kendall τ between score and y within deciles of m0 (weighted by decile size)."""
    dec = pd.qcut(pd.Series(m0).rank(method="first"), 10, labels=False).to_numpy()
    taus, ws = [], []
    for d in range(10):
        m = dec == d
        if m.sum() < 20:
            continue
        t = st.kendalltau(score[m], y[m]).statistic
        if np.isfinite(t):
            taus.append(t)
            ws.append(m.sum())
    return float(np.average(taus, weights=ws)) if taus else np.nan


def build(model, drop_outcomes=True):
    P = O.load_panel()
    P = P.sort_values(["entity", "date"]).reset_index(drop=True)
    g = P.groupby("entity")
    sh = lambda c, k: g[c].shift(k)  # noqa: E731
    re_ = P.reg_wind_gwh + P.reg_solar_gwh
    P["re_sum"] = re_
    re_l2, re_l9 = sh("re_sum", 2), sh("re_sum", 9)
    re_m = g["re_sum"].transform(lambda x: x.shift(2).rolling(28, min_periods=10).mean())
    D = pd.DataFrame({"entity": P.entity, "date": P.date, "bid_area": P.bid_area, "region": P.region,
                      "delta": P.dev_od_ud_gwh, "lam": P.prc_dsm_normal,
                      "R_raw": (re_l2 - re_l9).abs() / re_m.replace(0, np.nan),
                      "S_out": sh("gen_outage_share", 3), "S_coal": sh("gen_coal_cap_share_lt7d", 3),
                      "S_short": (sh("dem_energy_shortage_gwh", 2) > 0).astype(float).where(sh("dem_energy_shortage_gwh", 2).notna()),
                      "C_raw": sh("co2_ci_conv_op", 3),
                      "Y1_raw": np.maximum(P.dev_max_od_mw.abs(), P.dev_max_ud_mw.abs().fillna(0))
                      / (P.dem_energy_met_gwh * 1000 / 24).replace(0, np.nan),
                      "Y2_raw": (P.dem_energy_shortage_gwh > 0).astype(float).where(P.dem_energy_shortage_gwh.notna()),
                      "Y3_raw": P.sys_pct_lt_49_9})
    D["split"] = O.split_of(D.date)
    D["season"] = D.date.dt.month.map(O.SEASON)
    D["regime"] = np.where(D.date >= R3_START, "R3", np.where(D.date >= pd.Timestamp("2022-12-05"), "R2", "R1"))
    tr = D.split == "train"
    for c in ["R_raw", "S_out", "S_coal", "C_raw"]:
        D[c.replace("_raw", "") + "_p" if c.endswith("_raw") else c + "_p"] = ecdf_map(D[c].to_numpy(), D.loc[tr, c].to_numpy())
    D["S_p"] = D[["S_out_p", "S_coal_p", "S_short"]].mean(axis=1)  # mean of the available stress components
    # entities without monitored conventional generation have no carbon intensity: carbon context term = 0
    no_gen = D.groupby("entity").C_raw.transform(lambda s: s.notna().mean() < 0.5)
    D.loc[no_gen & D.C_p.isna(), "C_p"] = 0.0
    lam_med = D[tr].groupby("bid_area").lam.median()
    D["lam_n"] = D.lam / D.bid_area.map(lam_med)
    # U from the O2 T2 forecast (issue t−1, H = 1), calibrated with rolling-180 conformal at C1
    pr = pd.read_parquet(os.path.join(O2, "preds", f"T2__{model}.parquet"))
    sc = pd.read_parquet(os.path.join(O2, "scales.parquet"))
    sc = sc[sc.tid == "T2"][["sid", "H", "scale_mae", "scale_mse"]]
    pr = pr[pr.split.isin(["validation", "dev_test"]) & pr.y.notna()].merge(sc, on=["sid", "H"])
    quant = E2.has_quantiles(pr)
    pr = pr.reset_index(drop=True)
    Q = E2.rolling_level(pr, pr, "C1", quant, 1, 180)
    pr["U_raw"] = (Q[:, 5] - Q[:, 1]) / pr.scale_mae
    u = pr[pr.H == 1][["sid", "target_date", "U_raw"]].rename(columns={"sid": "entity", "target_date": "date"})
    D = D.merge(u, on=["entity", "date"], how="left")
    D["U_p"] = ecdf_map(D.U_raw.to_numpy(), D.loc[D.split == "validation", "U_raw"].to_numpy())
    D["M0"] = D.delta.abs() * D.lam_n
    D = D[D.split.isin(["validation", "dev_test"])].copy()
    need = ["delta", "lam_n", "R_p", "S_p", "C_p", "U_p"] + (["Y1_raw", "Y2_raw", "Y3_raw"] if drop_outcomes else [])
    D = D.dropna(subset=need).reset_index(drop=True)
    if not drop_outcomes:  # O5 uses the ex-ante context only
        return D
    for sp in ("validation", "dev_test"):
        m = D.split == sp
        D.loc[m, "Y1"] = D[m].groupby("entity").Y1_raw.transform(pct_rank)
        D.loc[m, "Y2"] = D[m].groupby("entity").Y2_raw.transform(pct_rank)
        D.loc[m, "Y3"] = D[m].Y3_raw.rank(pct=True)
    D["Y"] = D[["Y1", "Y2", "Y3"]].mean(axis=1)
    return D


def assess(D, w, kappa, U=None):
    Uc = D.U_p.to_numpy() if U is None else U
    X = np.column_stack([D.R_p, D.S_p, D.C_p, Uc])
    return D.delta.abs().to_numpy() * (D.lam_n.to_numpy() + kappa * X @ w)


def fit(D, U=None, allow_u=True):
    best = (-np.inf, None, None)
    y, m0 = D.Y.to_numpy(), D.M0.to_numpy()
    for kappa in KAPPAS:
        for w in simplex():
            if not allow_u and w[3] > 0:
                continue
            t = within_decile_tau(assess(D, w, kappa, U), y, m0)
            if t > best[0]:
                best = (t, w, kappa)
    return best


def block_diff(D, a, b, y):
    """Daily differences in within-decile τ are not additive; bootstrap over weeks by resampling dates."""
    dates = np.sort(D.date.unique())
    wk = ((dates - dates[0]) / np.timedelta64(7, "D")).astype(int)
    rng = np.random.default_rng(0)
    n_w = wk.max() + 1
    obs = within_decile_tau(a, y, D.M0.to_numpy()) - within_decile_tau(b, y, D.M0.to_numpy())
    boot = []
    by_week = {k: np.where(np.isin(D.date.to_numpy(), dates[wk == k]))[0] for k in range(n_w)}
    for _ in range(300):
        idx = np.concatenate([by_week[k] for k in rng.integers(0, n_w, n_w)])
        m0 = D.M0.to_numpy()[idx]
        boot.append(within_decile_tau(a[idx], y[idx], m0) - within_decile_tau(b[idx], y[idx], m0))
    boot = np.array(boot)
    return dict(diff=obs, lo=float(np.quantile(boot, 0.025)), hi=float(np.quantile(boot, 0.975)),
                p_one_sided=float((np.sum(boot - boot.mean() >= obs) + 1) / (len(boot) + 1)))


def ci_text(r):
    hi = r.get("hi", np.nan)
    return f"{r['lo']:+.4f}" if not np.isfinite(hi) else f"[{r['lo']:+.4f}, {hi:+.4f}]"


def main():
    quick = "--quick" in sys.argv  # code check only: 3 placebo draws per kind, outputs to a scratch folder
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sel = None if args else json.load(open(os.path.join(O2, "eval", "selection.json")))
    model = args[0] if args else next(x["selected"] for x in sel if x["tid"] == "T2")
    out = os.path.join(OUT, "_quick_check") if quick else OUT
    os.makedirs(out, exist_ok=True)
    D = build(model)
    V, T = D[D.split == "validation"].reset_index(drop=True), D[D.split == "dev_test"].reset_index(drop=True)
    if O.PHASE == "confirm":  # weights and κ frozen from development (pre-registration)
        fw = json.load(open(os.path.join(RES, "dev", "o4", "weights.json")))
        w, kappa = np.array([fw["weights"][k] for k in "RSCU"]), fw["kappa"]
        w0, kappa0 = np.array([fw["weights_noU"][k] for k in "RSCU"]), fw["kappa_noU"]
        y_v, m_v = V.Y.to_numpy(), V.M0.to_numpy()
        tau_v, tau_v0 = within_decile_tau(assess(V, w, kappa), y_v, m_v), within_decile_tau(assess(V, w0, kappa0), y_v, m_v)
    else:
        tau_v, w, kappa = fit(V)
        tau_v0, w0, kappa0 = fit(V, allow_u=False)
    json.dump(dict(model_for_U=model, weights=dict(zip(["R", "S", "C", "U"], map(float, w))), kappa=kappa,
                   val_tau=tau_v, weights_noU=dict(zip(["R", "S", "C", "U"], map(float, w0))), kappa_noU=kappa0,
                   val_tau_noU=tau_v0, n_val=len(V), n_dev=len(T)), open(os.path.join(out, "weights.json"), "w"), indent=1)
    A_T, A0_T, M0_T = assess(T, w, kappa), assess(T, w0, kappa0), T.M0.to_numpy()
    tests = []
    for yname in ["Y", "Y1", "Y2", "Y3"]:
        y = T[yname].to_numpy()
        r = block_diff(T, A_T, M0_T, y)
        tests.append(dict(test="A7 A vs M0", outcome=yname, tau_A=within_decile_tau(A_T, y, M0_T),
                          tau_M0=within_decile_tau(M0_T, y, M0_T), **r))
    y = T.Y.to_numpy()
    r = block_diff(T, A_T, A0_T, y)
    tests.append(dict(test="U gain (A vs A without U)", outcome="Y", tau_A=within_decile_tau(A_T, y, M0_T),
                      tau_M0=within_decile_tau(A0_T, y, M0_T), **r))
    # placebos: refit on validation with placebo U, score gain over no-U on the development test
    rng = np.random.default_rng(1)
    plac = []
    obs_gain = within_decile_tau(A_T, y, M0_T) - within_decile_tau(A0_T, y, M0_T)
    for kind in ("permuted", "gaussian"):
        for i in range(3 if quick else 50):
            if kind == "permuted":
                Uv = V.groupby("entity").U_p.transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
                Ut = T.groupby("entity").U_p.transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
            else:
                Uv = rng.normal(V.U_p.mean(), V.U_p.std(), len(V))
                Ut = rng.normal(V.U_p.mean(), V.U_p.std(), len(T))
            _, wp, kp = fit(V, U=Uv)  # placebo refits use the calibration year in both phases
            plac.append(dict(kind=kind, draw=i, gain=within_decile_tau(assess(T, wp, kp, Ut), y, M0_T)
                             - within_decile_tau(A0_T, y, M0_T)))
    plac = pd.DataFrame(plac)
    for kind, g in plac.groupby("kind"):
        tests.append(dict(test=f"U vs {kind} placebo", outcome="Y", diff=obs_gain, lo=float(g.gain.quantile(0.95)),
                          hi=np.nan, p_one_sided=float((np.sum(g.gain >= obs_gain) + 1) / (len(g) + 1))))
    strata = []
    for col in ("regime", "season"):
        for k, g in T.groupby(col):
            if len(g) < 500:
                continue
            a, m0, yy = assess(g, w, kappa), g.M0.to_numpy(), g.Y.to_numpy()
            strata.append(dict(stratum=col, value=k, n=len(g), tau_A=within_decile_tau(a, yy, m0),
                               tau_M0=within_decile_tau(m0, yy, m0)))
    la = np.log1p(A_T)
    grand = la.mean()
    ss_tot = ((la - grand) ** 2).sum()
    ss_state = T.assign(la=la).groupby("entity").la.agg(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    ss_date = T.assign(la=la).groupby("date").la.agg(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    var = pd.DataFrame([dict(component="State", share=ss_state / ss_tot), dict(component="date", share=ss_date / ss_tot),
                        dict(component="residual (interaction + noise)", share=1 - (ss_state + ss_date) / ss_tot)])
    T.assign(A=A_T, A_noU=A0_T).to_parquet(os.path.join(out, "assessment_rows.parquet"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(out, "tests.csv"), index=False)
    pd.DataFrame(strata).to_csv(os.path.join(out, "strata.csv"), index=False)
    plac.to_csv(os.path.join(out, "placebo.csv"), index=False)
    var.to_csv(os.path.join(out, "variance.csv"), index=False)
    L = ["# O4 development results — context-aware deviation assessment", "",
         f"U from O2 model `{model}` (T2, H = 1, rolling-180 conformal C1). Weights fitted on validation "
         f"(n = {len(V):,}); scored on the development test (n = {len(T):,}).", "",
         f"Weights (R, S, C, U) = {np.round(w, 2).tolist()}, κ = {kappa}; validation within-decile τ = {tau_v:.4f} "
         f"(without U: {tau_v0:.4f}).", "",
         "| test | outcome | τ(A) | τ(comparator) | difference | 95% CI / placebo 95th pct | p (one-sided) |",
         "|---|---|---|---|---|---|---|"]
    for r in tests:
        L.append(f"| {r['test']} | {r['outcome']} | {r.get('tau_A', np.nan):.4f} | {r.get('tau_M0', np.nan):.4f} | "
                 f"{r['diff']:+.4f} | {ci_text(r)} | "
                 f"{S.fmt_p(r['p_one_sided'])} |")
    L += ["", "Strata (development test):", "", md_table(pd.DataFrame(strata), 4), "",
          "Variance decomposition of log(1 + A):", "", md_table(var, 3), ""]
    open(os.path.join(TAB, f"{O.PHASE_DIR}_o4_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
