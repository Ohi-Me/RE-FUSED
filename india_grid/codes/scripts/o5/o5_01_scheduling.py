"""O5 step 1: risk-aware day-ahead drawal scheduling — rule-based arms (protocol `04_o4_o5_dev_protocol.md`, O5 + v1.1).

Stylised no-arbitrage settlement (NOT a replication of actual DSM settlement; addendum v1.2): for schedule S and actual
drawal X (GWh), D = X − S, band b = 0.1·|S|, λ⁺ = max(λ, p), λ⁻ = min(λ, p);
  charges(D) = λ⁺·min(D⁺, b) + 1.2λ⁺·(D⁺ − b)⁺ − [0.9λ⁻·min(D⁻, b) + 0.8λ⁻·(D⁻ − b)⁺],
  regret(S) = p·(S − X) + charges(X − S) ≥ 0  [₹ crore; GWh × ₹/MWh × 1000 / 1e7].
Decisions on day t−1 use the calibrated day-ahead drawal forecast (O2 T2, H = 1, rolling-180 conformal C1; 99
interpolated scenario levels) and published prices (7-day means ending t−14). Realised regret uses day-t prices.

Arms: H_final recorded final schedule (after intra-day revisions: ex-post reference, not a day-ahead competitor);
H_pub latest published recorded schedule (day t−2, a feasible day-ahead rule from operations data); P forecast median; A2 DRO-CVaR (β, ρ constant); A3 volatility-scaled ρ; A4 β(n,t) coupled to
forecast uncertainty U and system stress S (O4 ex-ante terms); ladder rung "A2r" per-regime β (stress state).
Decision arms are projected onto the hard-limit set (FY2018-23 1st–99th percentile of recorded schedules; day-to-day
change ≤ FY2018-23 99th percentile). Tuning on validation by J = ½ mean + ½ CVaR₉₀ of system-wide daily regret.

Usage: py -3.10 o5_01_scheduling.py [O2 model for T2, default = O2 selection] [--quick]
Outputs: 06_results/dev/o5/{decisions.parquet, tuning.csv, metrics.csv, tests.csv, ladder.csv}; 08_tables/dev_o5_summary.md
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
sys.path.insert(0, os.path.join(HERE, "..", "o4"))
from refused import o2data as O  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import RES, TAB  # noqa: E402
from refused.report import md_table  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402
import o4_01_assessment as A4  # noqa: E402

OUT = os.path.join(RES, O.PHASE_DIR, "o5")
O2 = os.path.join(RES, O.PHASE_DIR, "o2")
LEV99 = np.arange(1, 100) / 100
QL = np.array(O.QUANTILES)
CR = 1000 / 1e7  # GWh·₹/MWh → ₹ crore
BETAS, RHOS = [0, 0.25, 0.5, 0.75, 1.0], [0, 0.1, 0.25, 0.5]


def scenarios(Q):
    """(n, 7) quantiles → (n, 99) scenario values by linear interpolation with linear tail extrapolation."""
    out = np.empty((len(Q), 99))
    for i, lv in enumerate(LEV99):
        if lv < QL[0]:
            slope = (Q[:, 1] - Q[:, 0]) / (QL[1] - QL[0])
            out[:, i] = Q[:, 0] - slope * (QL[0] - lv)
        elif lv > QL[-1]:
            slope = (Q[:, -1] - Q[:, -2]) / (QL[-1] - QL[-2])
            out[:, i] = Q[:, -1] + slope * (lv - QL[-1])
        else:
            j = np.searchsorted(QL, lv, side="right") - 1
            j = min(j, len(QL) - 2)
            w = (lv - QL[j]) / (QL[j + 1] - QL[j])
            out[:, i] = Q[:, j] * (1 - w) + Q[:, j + 1] * w
    return np.sort(out, axis=1)


def regret(S_, X, p, lam):
    """No-arbitrage stylised settlement: over-drawal is charged at least max(λ, p) and under-drawal credited at most
    0.9·min(λ, p) within the 10 % band (CERC DSM Regulations 2024, Reg. 8(7): under-drawal receivable @ 90 % of NR at
    50 Hz) and ×1.2 / ×0.8 beyond it, so regret = cost − p·X ≥ 0 and is zero only for S = X."""
    D = X - S_
    b = 0.1 * np.abs(S_)
    Dp, Dm = np.maximum(D, 0), np.maximum(-D, 0)
    lo_, hi_ = np.maximum(lam, p), np.minimum(lam, p)
    over = lo_ * np.minimum(Dp, b) + 1.2 * lo_ * np.maximum(Dp - b, 0)
    under = 0.9 * hi_ * np.minimum(Dm, b) + 0.8 * hi_ * np.maximum(Dm - b, 0)
    return (p * (S_ - X) + over - under) * CR


def decide(sc, p_hat, lam_hat, beta, rho, grid_n=81):
    """Choose S minimising (1−β)·mean + β·CVaR₉₀ of regret over widened scenarios. Arrays broadcast per row."""
    n = len(sc)
    beta = np.broadcast_to(np.asarray(beta, float), (n,))
    rho = np.broadcast_to(np.asarray(rho, float), (n,))
    med = sc[:, 49:50]
    scw = med + (1 + rho[:, None]) * (sc - med)
    grid = np.linspace(scw[:, 0], scw[:, -1], grid_n, axis=1)  # (n, K)
    best = np.empty(n)
    for c0 in range(0, n, 2000):
        sl = slice(c0, c0 + 2000)
        Sg = grid[sl][:, :, None]                     # (m, K, 1)
        Xs = scw[sl][:, None, :]                      # (m, 1, J)
        R = regret(Sg, Xs, p_hat[sl, None, None], lam_hat[sl, None, None])   # (m, K, J)
        mean = R.mean(axis=2)
        cvar = np.sort(R, axis=2)[:, :, -10:].mean(axis=2)
        obj = (1 - beta[sl, None]) * mean + beta[sl, None] * cvar
        best[sl] = grid[sl][np.arange(obj.shape[0]), obj.argmin(axis=1)]
    return best


def project(frame, S_raw):
    """Sequential projection onto [S_min, S_max] and |ΔS| ≤ Δmax per entity (previous day of the same arm)."""
    f = frame.assign(S_raw=S_raw).sort_values(["entity", "date"])
    out = pd.Series(np.nan, index=f.index)
    for e, g in f.groupby("entity"):
        prev, prev_date = None, None
        for idx, r in g.iterrows():
            lo, hi = r.S_min, r.S_max
            if prev is not None and (r.date - prev_date).days == 1:
                lo, hi = max(lo, prev - r.dS_max), min(hi, prev + r.dS_max)
            elif np.isfinite(r.sched_prev):
                lo, hi = max(lo, r.sched_prev - r.dS_max), min(hi, r.sched_prev + r.dS_max)
            s = float(np.clip(r.S_raw, lo, hi)) if lo <= hi else float(np.clip(r.S_raw, r.S_min, r.S_max))
            out[idx] = s
            prev, prev_date = s, r.date
    return out.reindex(frame.index).to_numpy()


def daily_system(frame, S_):
    r = regret(S_, frame.X.to_numpy(), frame.p.to_numpy(), frame.lam.to_numpy())
    return pd.Series(r, index=frame.index).groupby(frame.date).sum().sort_index(), r


def J(daily):
    v = np.sort(daily.to_numpy())
    return 0.5 * v.mean() + 0.5 * v[-max(1, int(np.ceil(0.1 * len(v)))):].mean()


def build(model):
    P = O.load_panel().sort_values(["entity", "date"]).reset_index(drop=True)
    g = P.groupby("entity")
    tr = P.date <= O.TRAIN_END
    env = P[tr].groupby("entity").agg(S_min=("dev_drawal_schedule_gwh", lambda s: s.quantile(0.01)),
                                      S_max=("dev_drawal_schedule_gwh", lambda s: s.quantile(0.99)))
    dS = P[tr].assign(ds=g.dev_drawal_schedule_gwh.diff().abs()).groupby("entity").ds.quantile(0.99).rename("dS_max")
    px = P.groupby(["bid_area", "date"])[["prc_dam_acp", "prc_dsm_normal"]].first().reset_index().sort_values(["bid_area", "date"])
    gb = px.groupby("bid_area")
    px["p_hat"] = gb.prc_dam_acp.transform(lambda s: s.shift(14).rolling(7, min_periods=3).mean())
    px["lam_hat"] = gb.prc_dsm_normal.transform(lambda s: s.shift(14).rolling(7, min_periods=3).mean())
    F = P[["entity", "date", "bid_area", "dev_actual_drawal_gwh", "dev_drawal_schedule_gwh", "prc_dam_acp", "prc_dsm_normal"]].rename(
        columns={"dev_actual_drawal_gwh": "X", "dev_drawal_schedule_gwh": "S_hist", "prc_dam_acp": "p", "prc_dsm_normal": "lam"})
    F["sched_prev"] = g.dev_drawal_schedule_gwh.shift(1)
    F["sched_pub"] = g.dev_drawal_schedule_gwh.shift(2)  # latest recorded schedule published by the decision time
    F["vol28"] = g.dev_od_ud_gwh.transform(lambda s: s.shift(2).rolling(28, min_periods=10).std())
    F = F.merge(px[["bid_area", "date", "p_hat", "lam_hat"]], on=["bid_area", "date"], how="left")
    F = F.join(env, on="entity").join(dS, on="entity")
    vol_med = F[F.date <= O.TRAIN_END].groupby("entity").vol28.median()
    F["vol_ratio"] = (F.vol28 / F.entity.map(vol_med)).clip(0, 2)
    pr = pd.read_parquet(os.path.join(O2, "preds", f"T2__{model}.parquet"))
    sc = pd.read_parquet(os.path.join(O2, "scales.parquet"))
    sc = sc[sc.tid == "T2"][["sid", "H", "scale_mae", "scale_mse"]]
    pr = pr[pr.split.isin(["validation", "dev_test"]) & pr.y.notna()].merge(sc, on=["sid", "H"]).reset_index(drop=True)
    Q = E2.rolling_level(pr, pr, "C1", E2.has_quantiles(pr), 1, 180)
    q = pd.DataFrame(Q, columns=E2.QN)
    q["entity"], q["date"], q["H"] = pr.sid, pr.target_date, pr.H
    q = q[q.H == 1].drop(columns="H")
    F = F.merge(q, on=["entity", "date"], how="inner")
    ctx = A4.build(model, drop_outcomes=False)[["entity", "date", "U_p", "S_p"]]
    F = F.merge(ctx, on=["entity", "date"], how="left")
    F["split"] = O.split_of(F.date)
    F = F.dropna(subset=["X", "p", "lam", "p_hat", "lam_hat", "S_min", "S_max", "dS_max"] + E2.QN).reset_index(drop=True)
    F["U_p"] = F.U_p.fillna(0.5)
    F["S_p"] = F.S_p.fillna(0.5)
    F["stress"] = (F.S_p > 0.5).astype(int)
    return F


def main():
    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = args[0] if args else next(x["selected"] for x in json.load(open(os.path.join(O2, "eval", "selection.json")))
                                      if x["tid"] == "T2")
    out = os.path.join(OUT, "_quick_check") if quick else OUT
    os.makedirs(out, exist_ok=True)
    F = build(model)
    SC = scenarios(F[E2.QN].to_numpy())
    ph, lh = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
    val, dev = (F.split == "validation").to_numpy(), (F.split == "dev_test").to_numpy()
    h1 = val & (F.date.dt.month.between(4, 9)).to_numpy()
    h2 = val & ~h1
    tuning = []

    def evaluate_rule(beta, rho, mask):
        S_raw = decide(SC[mask], ph[mask], lh[mask], beta if np.ndim(beta) == 0 else beta[mask],
                       rho if np.ndim(rho) == 0 else rho[mask])
        fr = F[mask]
        S_ = project(fr, S_raw)
        return J(daily_system(fr, S_)[0]), S_

    grid_a2 = list(itertools.product(BETAS, RHOS))[: (4 if quick else None)]
    res = {}
    for b_, r_ in grid_a2:
        j, _ = evaluate_rule(b_, r_, val)
        tuning.append(dict(arm="A2", beta=b_, rho=r_, J_val=j))
        res[(b_, r_)] = j
    b2, r2 = min(res, key=res.get)
    FROZEN = json.load(open(os.path.join(RES, "dev", "o5", "frozen_params.json"))) if O.PHASE == "confirm" else None
    if FROZEN:
        b2, r2 = FROZEN["A2"]["beta"], FROZEN["A2"]["rho"]
    res3 = {}
    for r0 in RHOS[: (2 if quick else None)]:
        j, _ = evaluate_rule(b2, r0 * F.vol_ratio.fillna(1).to_numpy(), val)
        tuning.append(dict(arm="A3", beta=b2, rho0=r0, J_val=j))
        res3[r0] = j
    r3 = min(res3, key=res3.get) if not FROZEN else FROZEN["A3"]["rho0"]
    res4 = {}
    for b0, b1, bs in list(itertools.product(BETAS, [0, 0.25, 0.5], [0, 0.25, 0.5]))[: (3 if quick else None)]:
        beta = np.clip(b0 + b1 * F.U_p.to_numpy() + bs * F.stress.to_numpy(), 0, 1)
        j, _ = evaluate_rule(beta, r2, val)
        tuning.append(dict(arm="A4", beta0=b0, beta1=b1, beta_stress=bs, rho=r2, J_val=j))
        res4[(b0, b1, bs)] = j
    b0, b1, bs = min(res4, key=res4.get) if not FROZEN else (FROZEN["A4"]["beta0"], FROZEN["A4"]["beta_U"], FROZEN["A4"]["beta_stress"])
    res_r = {}
    for bl, bh in list(itertools.product(BETAS, BETAS))[: (3 if quick else None)]:
        beta = np.where(F.stress.to_numpy() == 1, bh, bl).astype(float)
        j, _ = evaluate_rule(beta, r2, val)
        tuning.append(dict(arm="A2r", beta_low=bl, beta_high=bh, rho=r2, J_val=j))
        res_r[(bl, bh)] = j
    blr, bhr = min(res_r, key=res_r.get) if not FROZEN else (FROZEN["A2r"]["beta_low"], FROZEN["A2r"]["beta_high"])
    if O.PHASE == "dev" and not quick:
        json.dump(dict(A2=dict(beta=b2, rho=r2), A3=dict(rho0=r3), A4=dict(beta0=b0, beta_U=b1, beta_stress=bs),
                       A2r=dict(beta_low=blr, beta_high=bhr), model=model), open(os.path.join(out, "frozen_params.json"), "w"),
                  indent=1)
    arms = {
        "H_final": F.S_hist.to_numpy(),
        "H_pub": project(F, F.sched_pub.fillna(F.q50).to_numpy()),
        "P": project(F, F.q50.to_numpy()),
        "A2": project(F, decide(SC, ph, lh, b2, r2)),
        "A2r": project(F, decide(SC, ph, lh, np.where(F.stress == 1, bhr, blr).astype(float), r2)),
        "A3": project(F, decide(SC, ph, lh, b2, r3 * F.vol_ratio.fillna(1).to_numpy())),
        "A4": project(F, decide(SC, ph, lh, np.clip(b0 + b1 * F.U_p + bs * F.stress, 0, 1).to_numpy(), r2)),
    }
    dec = F[["entity", "date", "split", "X", "p", "lam", "S_hist", "S_min", "S_max", "dS_max"]].copy()
    metrics, daily = [], {}
    for arm, S_ in arms.items():
        dec[f"S_{arm}"] = S_
        dec[f"regret_{arm}"] = regret(S_, F.X.to_numpy(), F.p.to_numpy(), F.lam.to_numpy())
    for sp in ("validation", "dev_test"):
        m = dec.split == sp
        for arm in arms:
            d = dec[m]
            sysd = d.groupby("date")[f"regret_{arm}"].sum().sort_index()
            daily[(sp, arm)] = sysd
            v = np.sort(sysd.to_numpy())
            cv = v[-max(1, int(np.ceil(0.1 * len(v)))):].mean()
            per = d.groupby("entity")[f"regret_{arm}"].agg(["mean", lambda s: np.sort(s.to_numpy())[-max(1, int(np.ceil(0.1 * len(s)))):].mean()])
            dev_e = (d.X - d[f"S_{arm}"]).abs()
            d2 = d.assign(ad=dev_e, aS=d[f"S_{arm}"].abs()).sort_values(["entity", "date"])
            roll = d2.groupby("entity")[["ad", "aS"]].transform(lambda s: s.rolling(7, min_periods=7).sum())
            budget_viol = float(((0.05 * roll.aS - roll.ad) < 0).mean())
            env_ex = float(((d[f"S_{arm}"] < d.S_min - 1e-9) | (d[f"S_{arm}"] > d.S_max + 1e-9)).mean())
            metrics.append(dict(split=sp, arm=arm, system_mean_crore=float(v.mean()), system_cvar90_crore=float(cv),
                                J=0.5 * v.mean() + 0.5 * cv, state_mean_crore=float(per["mean"].mean()),
                                state_cvar90_crore=float(per.iloc[:, 1].mean()),
                                deviation_energy_gwh_per_state_day=float(dev_e.mean()),
                                budget_violation_share=budget_viol, hard_limit_envelope_exceedance_share=env_ex,
                                n_state_days=int(m.sum())))
    tests = []
    for a, b in [("A4", "A2"), ("A4", "H_pub"), ("A4", "P"), ("A2", "H_pub"), ("A2", "P"), ("A3", "A2"), ("A2r", "A2"),
                 ("A4", "A2r"), ("P", "H_pub")]:
        da, db = daily[("dev_test", a)], daily[("dev_test", b)]
        dd = (da - db).dropna().to_numpy()
        t, p = E2.hln(dd, 1)
        ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
        tail = (da > da.quantile(0.9)) | (db > db.quantile(0.9))
        tests.append(dict(a=a, b=b, metric="mean system regret", mean_diff=float(dd.mean()), lo=ci["lo"], hi=ci["hi"],
                          p_a_better=p, n_days=len(dd)))
        dt = (da[tail] - db[tail]).to_numpy()
        ci2 = S.block_ci(dt, 3, n_boot=2000, seed=0) if len(dt) > 10 else dict(mean=np.nan, lo=np.nan, hi=np.nan, p_less=np.nan)
        tests.append(dict(a=a, b=b, metric="tail days (either arm above its 90th pct)", mean_diff=float(np.mean(dt)) if len(dt) else np.nan,
                          lo=ci2["lo"], hi=ci2["hi"], p_a_better=ci2.get("p_less", np.nan), n_days=len(dt)))
    T = pd.DataFrame(tests)
    fam = T.metric == "mean system regret"
    T.loc[fam, "p_holm"] = S.holm(T.loc[fam, "p_a_better"].fillna(1).to_numpy())
    # β-adaptivity ladder: prequential replay on validation (tune on H1, score on H2) vs realised development gains
    ladder = []
    if not quick:
        def tune_on(mask_fit, mask_score):
            best = {}
            for b_ in BETAS:
                best[("const", b_)] = evaluate_rule(b_, r2, mask_fit)[0]
            bc = min((k for k in best if k[0] == "const"), key=best.get)[1]
            rr = {}
            for bl, bh in itertools.product(BETAS, BETAS):
                rr[(bl, bh)] = evaluate_rule(np.where(F.stress == 1, bh, bl).astype(float), r2, mask_fit)[0]
            brg = min(rr, key=rr.get)
            ii = {}
            for c in itertools.product(BETAS, [0, 0.25, 0.5], [0, 0.25, 0.5]):
                ii[c] = evaluate_rule(np.clip(c[0] + c[1] * F.U_p + c[2] * F.stress, 0, 1).to_numpy(), r2, mask_fit)[0]
            bi = min(ii, key=ii.get)
            score = lambda beta: evaluate_rule(beta, r2, mask_score)[0]  # noqa: E731
            return (score(bc), score(np.where(F.stress == 1, brg[1], brg[0]).astype(float)),
                    score(np.clip(bi[0] + bi[1] * F.U_p + bi[2] * F.stress, 0, 1).to_numpy()))
        jc, jr, ji = tune_on(h1, h2)
        ladder = [dict(step="constant → per-regime", predicted_gain_val=jc - jr),
                  dict(step="per-regime → per-instance", predicted_gain_val=jr - ji)]
        mdev = {m_["arm"]: m_["J"] for m_ in metrics if m_["split"] == "dev_test"}
        ladder[0]["realised_gain_dev"] = mdev["A2"] - mdev["A2r"]
        ladder[1]["realised_gain_dev"] = mdev["A2r"] - mdev["A4"]
        for r_ in ladder:
            r_["agree"] = (r_["predicted_gain_val"] > 0) == (r_["realised_gain_dev"] > 0)
    dec.to_parquet(os.path.join(out, "decisions.parquet"), index=False)
    pd.DataFrame(tuning).to_csv(os.path.join(out, "tuning.csv"), index=False)
    M = pd.DataFrame(metrics)
    M.to_csv(os.path.join(out, "metrics.csv"), index=False)
    T.to_csv(os.path.join(out, "tests.csv"), index=False)
    pd.DataFrame(ladder).to_csv(os.path.join(out, "ladder.csv"), index=False)
    L = ["# O5 development results — risk-aware scheduling (stylised settlement; not actual DSM settlement)", "",
         f"Forecast source: O2 `{model}` (T2, H = 1, rolling-180 conformal C1). Chosen on validation: A2 β = {b2}, ρ = {r2}; "
         f"A3 ρ₀ = {r3}; A4 β₀ = {b0}, β_U = {b1}, β_stress = {bs}; A2r β(low, high stress) = ({blr}, {bhr}).", "",
         md_table(M, 4), "", "Development-test paired tests (system-wide daily regret, ₹ crore; "
         "negative = a better):", "", md_table(T, 4), ""]
    if ladder:
        L += ["β adaptivity ladder (prequential validation prediction vs realised development gain in J):", "",
              md_table(pd.DataFrame(ladder), 4), ""]
    if not quick:
        open(os.path.join(TAB, f"{O.PHASE_DIR}_o5_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
