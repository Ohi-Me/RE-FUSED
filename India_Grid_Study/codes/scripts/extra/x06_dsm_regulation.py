"""Exploratory analysis, not pre-registered (TPWRS reviewer request): do the scheduling results survive a settlement
closer to the DSM Regulations 2024 and schedule limits that follow demand growth?

Nothing is refitted. It uses the frame of the confirmatory run (final forecaster, calibrated scenarios, prices as
published before each decision), the frozen solver settings, and re-solves the decision rules under each settlement.

Settlements (daily net energy; the Regulations work per 15-minute block, which daily data cannot reproduce):
  stylised       the paper's model: credit 0.9 / 0.8 x min(lambda, p), band 10 % of |S|, 1.2 x max(lambda, p) beyond
  dsm_nominal    buyer rules of Reg. 8(7) at about 50.00 Hz, scaled to a day: tier 1 = min(10 % of |S|, 100 MW x 24 h),
                 tier 2 = min(15 % of |S|, 200 MW x 24 h); over-drawal charged at lambda in tiers 1 and 2 and 2 x lambda
                 beyond; under-drawal credited at 0.9 x lambda in tier 1, 0.8 x lambda in tier 2, nothing beyond.
                 Credits use lambda itself (not min(lambda, p)), so scheduling above the drawal can earn money.
  dsm_lowfreq    as dsm_nominal with over-drawal in tier 2 at 1.5 x lambda (frequency below 50.00 Hz)
Assumptions (stated in the output): one frequency regime for the whole day; the special limits for buyers below
400 MW and for renewable-rich States are not applied; States with a negative schedule are treated with the same rates.
Schedule limits:
  fixed          the paper's: 1st-99th percentile of recorded schedules in the fitting period, 99th percentile of |dS|
  rolling        the same percentiles over the 365 days ending two days before the target day (known at decision time)
Rules (re-solved for each settlement, projected onto each set of limits): persistence (H_pub), the forecast median P,
A2 (frozen beta = 0, rho = 0.5), R0 (risk-neutral, no widening), one fixed quantile level chosen on the calibration
year, and an adaptive level rule chosen on the calibration year (one level per tercile cell of the price ratio r and
of the band-relative risk, i.e. the forecast probability that the deviation leaves tier 1). Hindsight bounds on the
evaluation period use all 99 quantile levels and the same groups as the paper plus band-relative risk.
Output: results/extra/dsm_regulation.json (never overwrites a confirmatory file).
Usage:  python codes/scripts/extra/x06_dsm_regulation.py      (run on the H100: hpc/extra_dsm_regulation.pbs)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
for sub in ("o2", "o4", "o5", "dev2"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from refused import dev2 as D  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import RES  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402
import d2_09_o5 as R2  # noqa: E402

OUT = os.path.join(RES, "extra")
DAY_MW = 24 / 1000                     # MW sustained over a day -> GWh
SETTLEMENTS = {
    "stylised": None,
    "dsm_nominal": dict(k_over2=1.0),
    "dsm_lowfreq": dict(k_over2=1.5),
}


def regret_s(S_, X, p, lam, name):
    """Settlement cost of schedule S_ relative to an exact schedule (Rs crore). For the DSM variants it can be < 0."""
    if SETTLEMENTS[name] is None:
        return O5.regret(S_, X, p, lam)
    k2 = SETTLEMENTS[name]["k_over2"]
    a = np.abs(S_)
    b1 = np.minimum(0.10 * a, 100 * DAY_MW)
    b2 = np.maximum(np.minimum(0.15 * a, 200 * DAY_MW), b1)
    Dv = X - S_
    Dp, Dm = np.maximum(Dv, 0), np.maximum(-Dv, 0)
    t1p, t2p, t3p = np.minimum(Dp, b1), np.clip(Dp - b1, 0, b2 - b1), np.maximum(Dp - b2, 0)
    t1m, t2m = np.minimum(Dm, b1), np.clip(Dm - b1, 0, b2 - b1)
    over = lam * (t1p + k2 * t2p + 2.0 * t3p)
    under = lam * (0.9 * t1m + 0.8 * t2m)
    return (p * (S_ - X) + over - under) * O5.CR


def decide_s(SC, ph, lh, beta, rho, name, grid_n=81):
    """O5.decide with the settlement of `name` (beta = weight on CVaR90, rho = widening)."""
    med = SC[:, 49:50]
    scw = med + (1 + rho) * (SC - med)
    grid = np.linspace(scw[:, 0], scw[:, -1], grid_n, axis=1)
    best = np.empty(len(SC))
    for c0 in range(0, len(SC), 2000):
        sl = slice(c0, c0 + 2000)
        R = regret_s(grid[sl][:, :, None], scw[sl][:, None, :], ph[sl, None, None], lh[sl, None, None], name)
        obj = (1 - beta) * R.mean(axis=2) + beta * np.sort(R, axis=2)[:, :, -10:].mean(axis=2)
        best[sl] = grid[sl][np.arange(R.shape[0]), obj.argmin(axis=1)]
    return best


def rolling_limits(F):
    """1st/99th percentile of recorded schedules and 99th percentile of |dS| over the 365 days ending at t - 2."""
    P = O5.O.load_panel()[["entity", "date", "dev_drawal_schedule_gwh"]].sort_values(["entity", "date"])
    g = P.groupby("entity").dev_drawal_schedule_gwh
    s2 = g.shift(2)
    P["S_min_r"] = s2.groupby(P.entity).transform(lambda s: s.rolling(365, min_periods=180).quantile(0.01))
    P["S_max_r"] = s2.groupby(P.entity).transform(lambda s: s.rolling(365, min_periods=180).quantile(0.99))
    ds = g.diff().abs().groupby(P.entity).shift(2)
    P["dS_max_r"] = ds.groupby(P.entity).transform(lambda s: s.rolling(365, min_periods=180).quantile(0.99))
    G = F.merge(P[["entity", "date", "S_min_r", "S_max_r", "dS_max_r"]], on=["entity", "date"], how="left")
    G["S_min"] = G.S_min_r.fillna(G.S_min)
    G["S_max"] = G.S_max_r.fillna(G.S_max)
    G["dS_max"] = G.dS_max_r.fillna(G.dS_max)
    return G.drop(columns=["S_min_r", "S_max_r", "dS_max_r"])


def system_daily(F, r, mask):
    return pd.Series(r[mask], index=F.date[mask].to_numpy()).groupby(level=0).sum().sort_index()


def summary(daily):
    v = np.sort(daily.to_numpy())
    cv = v[-max(1, int(np.ceil(0.1 * len(v)))):].mean()
    return dict(mean=float(v.mean()), cvar90=float(cv), J=float(0.5 * v.mean() + 0.5 * cv))


def paired(da, db):
    dd = (da - db).dropna().to_numpy()
    ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
    return dict(mean_diff=float(dd.mean()), lo=ci["lo"], hi=ci["hi"], p_a_better=float(E2.hln(dd, 1)[1]), n_days=len(dd))


def best_by_group(R, groups, fit_mask):
    """Level index per group, chosen by summed cost on fit_mask (groups unseen in fit_mask get the overall best)."""
    overall = int(R[fit_mask].sum(axis=0).argmin())
    ch = np.full(len(R), overall)
    for gv in np.unique(groups[fit_mask]):
        rows = fit_mask & (groups == gv)
        ch[groups == gv] = int(R[rows].sum(axis=0).argmin())
    return ch


def main():
    t0 = time.time()
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    F0 = R2.build_v2(sel["final"], int(sel["calibration_window"])).reset_index(drop=True)
    SC = O5.scenarios(F0[D.QN].to_numpy())
    ph, lh = F0.p_hat.to_numpy(), F0.lam_hat.to_numpy()
    X, p, lam = F0.X.to_numpy(), F0.p.to_numpy(), F0.lam.to_numpy()
    val, test = (F0.split == "validation").to_numpy(), (F0.split == "dev_test").to_numpy()
    fz = D.frozen("o5", "params.json")["params"]
    limits = {"fixed": F0, "rolling": rolling_limits(F0)}
    levels = np.arange(1, 100)                     # scenario column k-1 holds level k %
    r_hat = lh / ph
    cut = lambda x: np.quantile(x[val], [1 / 3, 2 / 3])  # noqa: E731
    res = dict(note="exploratory, not pre-registered; frozen solver, stored forecasts, nothing refitted; hindsight "
                    "values are bounds, not rules",
               assumptions=["daily net energy instead of 15-minute blocks", "one frequency regime per day",
                            "buyer rules applied to every State, including negative schedules",
                            "special limits for buyers below 400 MW and renewable-rich States not applied"],
               n_state_days_test=int(test.sum()), share_negative_schedule_test=float((F0.S_hist[test] < 0).mean()),
               cases={})
    lim_stats = {}
    for ln, F in limits.items():
        lim_stats[ln] = dict(width_median_test=float((F.S_max - F.S_min)[test].median()),
                             recorded_outside_test=float(((F.S_hist < F.S_min) | (F.S_hist > F.S_max))[test].mean()),
                             actual_above_max_test=float((F.X > F.S_max)[test].mean()))
    res["limits"] = lim_stats
    for ln, F in limits.items():
        S_lvl = np.column_stack([O5.project(F, SC[:, k - 1]) for k in levels])
        S_p = O5.project(F, F.q50.to_numpy())
        S_pub_raw = F.sched_pub.fillna(F.q50).to_numpy()
        S_pub = O5.project(F, S_pub_raw)                # as in the confirmatory run (d2_09_o5: H_pub is projected)
        for sn in SETTLEMENTS:
            t1 = time.time()
            a2_raw = decide_s(SC, ph, lh, fz["A2"]["beta"], fz["A2"]["rho"], sn)
            r0_raw = decide_s(SC, ph, lh, 0.0, 0.0, sn)
            S_a2, S_r0 = O5.project(F, a2_raw), O5.project(F, r0_raw)
            # band-relative risk: forecast probability that the deviation from the R0 schedule leaves tier 1
            b1 = np.minimum(0.10 * np.abs(r0_raw), 100 * DAY_MW)
            risk = ((SC > r0_raw[:, None] + b1[:, None]) | (SC < r0_raw[:, None] - b1[:, None])).mean(axis=1)
            groups = {"r (terciles)": np.digitize(r_hat, cut(r_hat)),
                      "forecast uncertainty (terciles)": np.digitize(F.U_p.to_numpy(), cut(F.U_p.to_numpy())),
                      "system stress (on/off)": (F.S_p.to_numpy() > 0.5).astype(int),
                      "band-relative risk (terciles)": np.digitize(risk, cut(risk)),
                      "State": pd.factorize(F.entity)[0], "day": pd.factorize(F.date)[0]}
            R = np.column_stack([regret_s(S_lvl[:, j], X, p, lam, sn) for j in range(len(levels))])
            j_cal = int(R[val].sum(axis=0).argmin())
            adapt_groups = groups["r (terciles)"] * 3 + groups["band-relative risk (terciles)"]
            ch_adapt = best_by_group(R, adapt_groups, val)
            rules = {"H_pub": S_pub, "H_pub (not projected)": S_pub_raw, "P": S_p, "A2": S_a2, "R0": S_r0}
            daily = {k: system_daily(F, regret_s(v, X, p, lam, sn), test) for k, v in rules.items()}
            daily["fixed level (calibration year)"] = system_daily(F, R[:, j_cal], test)
            daily["adaptive level, r x risk (calibration year)"] = system_daily(F, R[np.arange(len(R)), ch_adapt], test)
            per_level = np.array([system_daily(F, R[:, j], test).mean() for j in range(len(levels))])
            j_best = int(per_level.argmin())
            bounds = {}
            for gname, gv in groups.items():
                ch = best_by_group(R, gv, test)
                m = system_daily(F, R[np.arange(len(R)), ch], test).mean()
                bounds[gname] = dict(mean=float(m), gain=float(per_level[j_best] - m),
                                     gain_pct=float(100 * (per_level[j_best] - m) / abs(per_level[j_best])))
            tau_r0 = (SC[test] < r0_raw[test, None]).mean(axis=1)
            case = dict(rules={k: summary(v) for k, v in daily.items()},
                        a2_vs_p=paired(daily["A2"], daily["P"]), a2_vs_pub=paired(daily["A2"], daily["H_pub"]),
                        a2_vs_pub_raw=paired(daily["A2"], daily["H_pub (not projected)"]),
                        r0_vs_a2=paired(daily["R0"], daily["A2"]),
                        adaptive_vs_fixed=paired(daily["adaptive level, r x risk (calibration year)"],
                                                 daily["fixed level (calibration year)"]),
                        fixed_level_calibration=int(levels[j_cal]), best_level_test=int(levels[j_best]),
                        best_level_test_mean=float(per_level[j_best]), bounds=bounds,
                        r0_level_median=float(np.median(tau_r0)),
                        r0_level_iqr=[float(np.percentile(tau_r0, 25)), float(np.percentile(tau_r0, 75))],
                        projection_binding_a2=float(np.mean(np.abs(S_a2 - a2_raw) > 1e-9)),
                        seconds=round(time.time() - t1))
            res["cases"][f"{sn} | {ln} limits"] = case
            print(sn, ln, json.dumps({k: (v if k not in ("bounds", "rules") else None) for k, v in case.items()}),
                  flush=True)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, "dsm_regulation.json"), "w"), indent=1)
    print("DONE", res["seconds"], "s")


if __name__ == "__main__":
    main()
