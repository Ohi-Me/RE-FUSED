"""Additional analysis, not pre-registered: how much any adaptive rule could have earned, and when it would pay.

It uses the frame of the confirmatory run (final forecaster with its calibration window, calibrated forecast
scenarios, prices as published before each decision) and adds rules of the simplest kind: schedule at level tau of
the calibrated forecast distribution, projected onto the operating envelope like every other rule.

1. R0, the parameter-free Bayes rule: the schedule that minimises expected regret under the calibrated distribution
   (beta = 0, no widening). Nothing is tuned. Compared with the frozen rule A2 and the forecast median P.
2. Upper bounds from hindsight. On the evaluation period the best single level for all State-days is found, and then
   the best level for each group of a signal: terciles of forecast uncertainty, stress on or off, terciles of the
   rate-to-price ratio r, each State, each day. The gain of a grouped choice over the single level is the most that
   any rule adapting to that signal could have earned here. These are bounds, not rules.
3. A real rule: one level per State, chosen on the calibration year only, scored on the evaluation period.
4. The bounds of 2 under six settlement designs (band width, penalty and credit multipliers), to show when
   adaptivity would pay.
5. Solve times of the rules on the H100.

It reads only stored results and writes to results/extra/, never into the confirmatory folders.
Outputs: results/extra/settlement_extensions.json
Usage:   python run_all.py stage x_settlement_extensions   (confirmatory environment, on the GPU)
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
LEVELS = np.arange(5, 96, 5)            # quantile levels in per cent; scenario column k-1 holds level k %
DESIGNS = {                             # band, over-drawal in / beyond band, under-drawal credit in / beyond band
    "stylised 2024 settlement": dict(band=0.10, oi=1.0, oo=1.2, ui=0.9, uo=0.8),
    "narrow band (5 %)": dict(band=0.05, oi=1.0, oo=1.2, ui=0.9, uo=0.8),
    "wide band (20 %)": dict(band=0.20, oi=1.0, oo=1.2, ui=0.9, uo=0.8),
    "harsher over-drawal (1.5x beyond band)": dict(band=0.10, oi=1.0, oo=1.5, ui=0.9, uo=0.8),
    "smaller under-drawal credit (0.8 / 0.6)": dict(band=0.10, oi=1.0, oo=1.2, ui=0.8, uo=0.6),
    "no credit haircut in band (1.0)": dict(band=0.10, oi=1.0, oo=1.2, ui=1.0, uo=0.8),
}


def regret_v(S_, X, p, lam, d):
    """Stylised settlement regret for a design d (the base design equals o5_01_scheduling.regret)."""
    Dv = X - S_
    b = d["band"] * np.abs(S_)
    Dp, Dm = np.maximum(Dv, 0), np.maximum(-Dv, 0)
    hi, lo = np.maximum(lam, p), np.minimum(lam, p)
    over = d["oi"] * hi * np.minimum(Dp, b) + d["oo"] * hi * np.maximum(Dp - b, 0)
    under = d["ui"] * lo * np.minimum(Dm, b) + d["uo"] * lo * np.maximum(Dm - b, 0)
    return (p * (S_ - X) + over - under) * O5.CR


def bayes(SC, ph, lh, d, grid_n=81):
    """Risk-neutral schedule under the calibrated scenarios for design d (O5.decide with beta = 0, rho = 0)."""
    grid = np.linspace(SC[:, 0], SC[:, -1], grid_n, axis=1)
    best = np.empty(len(SC))
    for c0 in range(0, len(SC), 2000):
        sl = slice(c0, c0 + 2000)
        R = regret_v(grid[sl][:, :, None], SC[sl][:, None, :], ph[sl, None, None], lh[sl, None, None], d)
        best[sl] = grid[sl][np.arange(R.shape[0]), R.mean(axis=2).argmin(axis=1)]
    return best


def system_daily(F, r, mask):
    return pd.Series(r[mask], index=F.date[mask].to_numpy()).groupby(level=0).sum().sort_index()


def summary(daily):
    v = np.sort(daily.to_numpy())
    cv = v[-max(1, int(np.ceil(0.1 * len(v)))):].mean()
    return dict(mean=float(v.mean()), cvar90=float(cv), J=float(0.5 * v.mean() + 0.5 * cv))


def paired(da, db):
    dd = (da - db).dropna().to_numpy()
    ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
    return dict(mean_diff=float(dd.mean()), lo=ci["lo"], hi=ci["hi"], p_a_better=float(E2.hln(dd, 1)[1]),
                n_days=len(dd))


def grouped_best(R, groups, mask):
    """Hindsight: the best level (column of R) for each group, by summed regret on the rows in mask."""
    chosen = np.full(len(R), int(R[mask].sum(axis=0).argmin()))  # rows outside mask keep the overall best level
    for g in np.unique(groups[mask]):
        rows = mask & (groups == g)
        chosen[groups == g] = int(R[rows].sum(axis=0).argmin())
    return chosen


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    F = R2.build_v2(sel["final"], int(sel["calibration_window"])).reset_index(drop=True)
    SC = O5.scenarios(F[D.QN].to_numpy())
    ph, lh = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
    X, p, lam = F.X.to_numpy(), F.p.to_numpy(), F.lam.to_numpy()
    val, test = (F.split == "validation").to_numpy(), (F.split == "dev_test").to_numpy()
    fz = D.frozen("o5", "params.json")["params"]
    base = DESIGNS["stylised 2024 settlement"]
    res = dict(note="additional analysis, not pre-registered; frozen solver and stored forecasts; hindsight "
                    "choices are upper bounds, not rules", n_state_days_test=int(test.sum()), designs={})

    # 5. solve times on the evaluation period (the fair comparison for an operator: one day-ahead run)
    times = {}
    tt = time.perf_counter()
    S_a2_raw = O5.decide(SC, ph, lh, fz["A2"]["beta"], fz["A2"]["rho"])
    times["A2 (CVaR over widened scenarios)"] = time.perf_counter() - tt
    tt = time.perf_counter()
    S_r0_raw = bayes(SC, ph, lh, base)
    times["R0 (Bayes rule)"] = time.perf_counter() - tt
    tt = time.perf_counter()
    jp = R2.JointProblem(F, SC)
    jp.solve(fz["A5"]["beta"])
    times["A5 (joint scenarios, GPU)"] = time.perf_counter() - tt
    for k, v in list(times.items()):
        times[k] = dict(seconds_all_days=float(v), ms_per_state_day=float(1000 * v / len(F)))
    ppo = json.load(open(os.path.join(RES, "..", "logs", "done", "c_d2_ppo.json")))
    times["A1 (PPO) training, five seeds"] = dict(seconds_all_days=float(ppo["seconds"]), ms_per_state_day=None)
    res["solve_times"] = times

    # schedules of the quantile rules, projected once each (projection does not depend on the settlement design)
    S_lvl = np.column_stack([O5.project(F, SC[:, k - 1]) for k in LEVELS])
    S_a2 = O5.project(F, S_a2_raw)
    S_p = O5.project(F, F.q50.to_numpy())
    S_r0 = O5.project(F, S_r0_raw)
    res["projection_binding_share_a2"] = float(np.mean(np.abs(S_a2 - S_a2_raw) > 1e-9))

    # 1. R0 against A2 and P on the base design
    rules = {"P": S_p, "A2": S_a2, "R0": S_r0}
    daily = {k: system_daily(F, O5.regret(v, X, p, lam), test) for k, v in rules.items()}
    res["rules_test"] = {k: summary(v) for k, v in daily.items()}
    res["r0_vs_a2"] = paired(daily["R0"], daily["A2"])
    res["r0_vs_p"] = paired(daily["R0"], daily["P"])
    tau0 = (SC < S_r0_raw[:, None]).mean(axis=1)
    res["r0_level_test"] = dict(median=float(np.median(tau0[test])),
                                iqr=float(np.subtract(*np.percentile(tau0[test], [75, 25]))))

    # groups for the hindsight bounds (tercile cuts from the calibration year, so the groups are ex-ante)
    r_hat = lh / ph
    cut = lambda x: np.quantile(x[val], [1 / 3, 2 / 3])  # noqa: E731
    groups = {
        "forecast uncertainty (terciles)": np.digitize(F.U_p.to_numpy(), cut(F.U_p.to_numpy())),
        "system stress (on / off)": (F.S_p.to_numpy() > 0.5).astype(int),
        "rate-to-price ratio r (terciles)": np.digitize(r_hat, cut(r_hat)),
        "State": pd.factorize(F.entity)[0],
        "day (upper bound for any system-wide signal)": pd.factorize(F.date)[0],
    }

    # 2 and 4. bounds for every design
    for name, d in DESIGNS.items():
        R = np.column_stack([regret_v(S_lvl[:, j], X, p, lam, d) for j in range(len(LEVELS))])
        per_level = [summary(system_daily(F, R[:, j], test))["mean"] for j in range(len(LEVELS))]
        j_best = int(np.argmin(per_level))
        j_cal = int(np.argmin([R[val, j].sum() for j in range(len(LEVELS))]))
        entry = dict(best_single_level=int(LEVELS[j_best]), best_single_mean=per_level[j_best],
                     calibration_level=int(LEVELS[j_cal]), calibration_level_mean=per_level[j_cal], bounds={},
                     levels=[int(k) for k in LEVELS], mean_by_level=[float(x) for x in per_level])
        for gname, g in groups.items():
            ch = grouped_best(R, g, test)
            m = summary(system_daily(F, R[np.arange(len(R)), ch], test))["mean"]
            entry["bounds"][gname] = dict(mean=m, gain=per_level[j_best] - m,
                                          gain_pct=100 * (per_level[j_best] - m) / per_level[j_best])
        S_bd = O5.project(F, bayes(SC, ph, lh, d)) if name != "stylised 2024 settlement" else S_r0
        tau_d = (SC[test] < bayes(SC[test], ph[test], lh[test], d)[:, None]).mean(axis=1)
        entry["bayes_rule_mean"] = summary(system_daily(F, regret_v(S_bd, X, p, lam, d), test))["mean"]
        entry["bayes_level_iqr"] = float(np.subtract(*np.percentile(tau_d, [75, 25])))
        entry["bayes_level_median"] = float(np.median(tau_d))
        res["designs"][name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "bounds"}), flush=True)

    # 3. a real rule: one level per State chosen on the calibration year, against one level for all
    R = np.column_stack([O5.regret(S_lvl[:, j], X, p, lam) for j in range(len(LEVELS))])
    ent = pd.factorize(F.entity)[0]
    ch_state = np.empty(len(R), dtype=int)
    for e in np.unique(ent):
        rows = val & (ent == e)
        ch_state[ent == e] = int(R[rows].sum(axis=0).argmin()) if rows.any() else int(np.argmin(R[val].sum(axis=0)))
    j_cal = int(np.argmin(R[val].sum(axis=0)))
    d_state = system_daily(F, R[np.arange(len(R)), ch_state], test)
    d_const = system_daily(F, R[:, j_cal], test)
    res["per_state_levels"] = dict(test=summary(d_state), constant_test=summary(d_const),
                                   vs_constant=paired(d_state, d_const), vs_a2=paired(d_state, daily["A2"]),
                                   levels_used=sorted({int(LEVELS[j]) for j in np.unique(ch_state)}))
    # 6. sensitivity of the fixed rule to its own two settings: the development grid of (beta, rho), scored on the
    # evaluation period, next to the setting frozen in development
    grid = []
    for b_ in O5.BETAS:
        for r_ in O5.RHOS:
            S_ = O5.project(F, O5.decide(SC, ph, lh, b_, r_))
            g = summary(system_daily(F, O5.regret(S_, X, p, lam), test))
            grid.append(dict(beta=b_, rho=r_, frozen=(b_ == fz["A2"]["beta"] and r_ == fz["A2"]["rho"]), **g))
    res["a2_grid_test"] = grid
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, "settlement_extensions.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "designs"}, indent=1))


if __name__ == "__main__":
    main()
