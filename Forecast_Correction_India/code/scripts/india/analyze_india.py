"""Pre-registered analysis of the Indian hourly study (design/05_preregistration_india.md).

Reuses the analysis functions of the earlier study (confirm/analyze_confirmatory.py) unchanged, pointed at the Indian
result folders, and adds the parts that differ: IH-5 per forecast and series with relative RMSE, and IH-4/IH-8 with the
LightGBM forecast as the reference and all-India demand met as the energy series (no price is used).
Outputs: results/india/<confirm|dev>/analysis/verdicts.json and the tables behind it.
  --map dev      runs the same code on the development results to check that it executes (not confirmatory)
  --map confirm  the pre-registered analysis
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

os.environ["REFUSED_GATE_STUDY"] = "india"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "confirm"))
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from refused_gate import analysis as A, stats as st  # noqa: E402
from refused_gate.paths import RES  # noqa: E402
import analyze_confirmatory as AC  # noqa: E402

AC.MAPS.update({m: dict(C1=f"india/{m}/IH1", C2=f"india/{m}/IH2", X5=f"india/{m}/prob_energy", X1=f"india/{m}/x1")
                for m in ("dev", "confirm")})
AC.UNIT_OF_BLOCK = {"C1": "INDIA-HOURLY"}
AC.BUDGET_OF_UNIT = {"INDIA-HOURLY": ("C2", "C1", "lgbm")}
ENERGY_SERIES = "demand"


def ih5(mp):
    """Per (forecast, series): relative RMSE of the forecast and the best skill of any ladder level."""
    folder = AC.rpath(mp, "C1")
    rows = []
    for key in AC.configs(folder):
        L = A.row_losses(AC.load_cfg(folder, key))
        for sid, g in L.groupby("sid"):
            base = float(g["base"].mean())
            loss = {c: float(g[c].mean()) for c in g.columns if c not in ("sid", "ts", "base")}
            rows.append(dict(baseline=key, sid=sid, base_rmse=np.sqrt(base), best_skill=1 - min(loss.values()) / base))
    R = pd.DataFrame(rows)
    if R.empty:
        return [], R
    # relative RMSE: divide by the series' mean level on the validation period, read from the forecast frame
    from refused_gate import data
    ds = data.india_hourly(mp if mp == "confirm" else "dev")
    lvl = ds["df"][ds["df"].split == "val"].groupby("sid").y.mean()
    R["rel_rmse"] = R.base_rmse / R.sid.map(lvl)
    rho, p = stats.spearmanr(R.rel_rmse, R.best_skill)
    return [dict(id="H5-strength", family="baseline", estimate=float(rho), n=len(R),
                 p=float(p / 2 if rho > 0 else 1 - p / 2),
                 criterion="Spearman(relative RMSE of the forecast, best correction skill) > 0")], R


def h4_h8_india(mp):
    folder = AC.rpath(mp, "X5")
    files = sorted(glob.glob(os.path.join(folder, "prob_energy_s*.parquet"))) if folder else []
    if not files:
        return []
    frames = [pd.read_parquet(f) for f in files]
    base = frames[0][["sid", "ts", "y_t", "dam_mcp_rs_mwh", "rtm_mcp_rs_mwh"]].copy()
    num = [c for c in frames[0].columns if c.startswith(("F_", "pin99_", "up99_"))]
    avg = sum(f[num].to_numpy(dtype=float) for f in frames) / len(frames)
    D = pd.concat([base.reset_index(drop=True), pd.DataFrame(avg, columns=num)], axis=1)
    D["day"] = D.ts.dt.normalize()
    tests = []

    def daily_diff(X, a, b, per_row=False):
        g = (X[a] - X[b]).groupby(X.day)
        x = (g.mean() if per_row else g.sum()).sort_index().to_numpy()
        return st.block_ci(x, AC.BLOCK_DAYS, 2000, 0)

    for other in ("q_global", "q_zone"):
        ci = daily_diff(D, f"pin99_corr_series_{other}", "pin99_corr_series_q_zone_hour", per_row=True)
        tests.append(dict(id=f"H4a-serieshour-vs-{other.replace('zone', 'series')}", family="probabilistic",
                          estimate=ci["mean"], lo=ci["lo"], hi=ci["hi"], p=ci["p_greater"],
                          criterion="pinball@0.99 lower for series x hour cells"))
    err = D.y_t - D.F_corr_series
    cov_inst = float(np.mean(err <= D.up99_corr_series_q_instance))
    cov_sh = float(np.mean(err <= D.up99_corr_series_q_zone_hour))
    tests.append(dict(id="H4b-instance-undercovers", family="probabilistic", estimate=cov_inst,
                      extra=dict(series_hour=cov_sh), p=np.nan, verdict_rule="threshold",
                      criterion="coverage(instance) < 0.985 and coverage(series x hour) >= 0.985",
                      passed=bool(cov_inst < 0.985 and cov_sh >= 0.985)))
    E = D[D.sid == ENERGY_SERIES].copy()
    E["dev_ref"] = np.abs(E.y_t - E.F_lgbm)
    E["dev_cs"] = np.abs(E.y_t - E.F_corr_series)
    ci = daily_diff(E, "dev_ref", "dev_cs")
    tests.append(dict(id="H8a-deviation-energy", family="energy", estimate=ci["mean"] * 365 / 1000,
                      lo=ci["lo"] * 365 / 1000, hi=ci["hi"] * 365 / 1000, p=ci["p_greater"],
                      criterion="all-India demand: deviation energy lower after correction (GWh/yr)"))
    E["res_ref"] = np.maximum(E.up99_lgbm_q_zone_hour, 0)
    E["res_cs"] = np.maximum(E.up99_corr_series_q_zone_hour, 0)
    ci = daily_diff(E, "res_ref", "res_cs")
    cov_ref = float(np.mean((E.y_t - E.F_lgbm) <= E.up99_lgbm_q_zone_hour))
    cov_cs = float(np.mean((E.y_t - E.F_corr_series) <= E.up99_corr_series_q_zone_hour))
    tests.append(dict(id="H8b-reserve-mw", family="energy", estimate=ci["mean"] / 24, lo=ci["lo"] / 24,
                      hi=ci["hi"] / 24, p=ci["p_greater"], extra=dict(coverage_reference=cov_ref, coverage_corrected=cov_cs),
                      criterion=f"all-India demand: 99% reserve (MW) lower after correction with |coverage difference| "
                                f"<= {AC.COVERAGE_TOL}", coverage_ok=bool(abs(cov_ref - cov_cs) <= AC.COVERAGE_TOL)))
    spread = E.rtm_mcp_rs_mwh - E.dam_mcp_rs_mwh
    imb = ((E.y_t - E.F_lgbm) - (E.y_t - E.F_corr_series)) * spread       # Rs per hour (MW x Rs/MWh x 1 h)
    x = imb.groupby(E.day).sum().sort_index().to_numpy()
    ci = st.block_ci(np.nan_to_num(x), AC.BLOCK_DAYS, 2000, 0)
    tests.append(dict(id="E8c-imbalance-cost", family="energy-estimation", estimate=ci["mean"] * 365 / 1e7,
                      lo=ci["lo"] * 365 / 1e7, hi=ci["hi"] * 365 / 1e7, p=np.nan, verdict_rule="estimate",
                      criterion="estimation only: stylised two-settlement imbalance cost saved, Rs crore per year"))
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="dev", choices=["dev", "confirm"])
    a = ap.parse_args()
    out_dir = os.path.join(RES, "india", a.map, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    extra = {}
    tests = AC.h2(a.map, extra)
    P, T, S, STR = AC.ladder_tables(a.map)
    tests += AC.h3_h6_h7_h5(P, T, S, STR)
    t5, R5 = ih5(a.map)
    tests += t5
    tests += h4_h8_india(a.map)
    tests += AC.h1(a.map)
    tests = AC.finalise(tests)
    for t in tests:                      # the Indian study's ids: IH-...
        t["id"] = "I" + t["id"].replace("H", "H-", 1)
    for name, df in (("IH3_part_refinements", P), ("IH6_time_policies", T), ("IH7_selection", S), ("IH5_strength", R5)):
        df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    pd.DataFrame(extra.get("H2_rows", [])).to_csv(os.path.join(out_dir, "IH2_budget.csv"), index=False)
    json.dump(tests, open(os.path.join(out_dir, "verdicts.json"), "w"), indent=1, default=float)
    for t in tests:
        print(f"{t['id']:36s} {t.get('verdict', '?'):14s} est={t['estimate']:+.5f} p={st.fmt_p(t.get('p', np.nan))} "
              f"p_holm={st.fmt_p(t.get('p_holm', np.nan))}  [{t['criterion']}]")


if __name__ == "__main__":
    main()
