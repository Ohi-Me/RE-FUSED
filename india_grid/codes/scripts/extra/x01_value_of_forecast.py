"""Additional analysis, not pre-registered: what forecast accuracy is worth in settlement cost.

Every day-ahead drawal forecaster (T2) whose forecasts for the evaluation period are stored is passed through the
same scheduling rules as the chain test: P schedules the forecast median, A2 is the risk-aware rule and A5 the
system-tail rule, each with the setting frozen in development (nothing is tuned here, and no new model is fitted).
For each forecaster we record the accuracy of its day-ahead forecast on the scheduled State-days and the system-wide
settlement regret of each rule on the evaluation period. Across forecasters this shows how much a unit of accuracy
is worth to the operator. The final forecaster with its own calibration window must reproduce the confirmatory
result; the script stops if it does not.

The analysis reads only stored results and writes to results/extra/, never into the confirmatory folders.
Outputs: results/extra/value_of_forecast.csv (one row per forecaster and rule) and value_of_forecast.json (slope of
regret on accuracy with a 7-day block-bootstrap interval, rank correlation, reproduction check)
Usage:   python run_all.py stage x_value_of_forecast   (confirmatory environment, on the GPU for the A5 solves)
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
from refused import o2data as O  # noqa: E402
from refused.paths import RES  # noqa: E402
import d2_09_o5 as R2  # noqa: E402  (the round-2 scheduling code that produced the confirmatory results)

OUT = os.path.join(RES, "extra")
N_BOOT, BLOCK, SEED = 2000, 7, 0


def forecasters():
    """Every T2 forecaster of round 2 whose forecasts are stored: the members and the three combinations."""
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    names = list(dict.fromkeys(sel["members"] + ["ens_top", "ens_stack", "ens_online"]))
    return [n for n in names if D.find_member("T2", n)], sel["final"], int(sel["calibration_window"])


def accuracy(F):
    """Day-ahead accuracy of the calibrated forecast on the scheduled State-days, in GWh (evaluation period)."""
    ev = F[F.split == "dev_test"]
    X, Q = ev.X.to_numpy(), ev[D.QN].to_numpy()
    u = X[:, None] - Q
    pin = np.maximum(D.LEV[None, :] * u, (D.LEV[None, :] - 1) * u).mean(axis=1)
    per_day = pd.DataFrame({"date": ev.date.to_numpy(), "ae": np.abs(X - ev.q50.to_numpy()), "pin": pin}).groupby("date").mean()
    return float(np.abs(X - ev.q50.to_numpy()).mean()), float(pin.mean()), per_day


def block_slope(daily_regret, daily_mae):
    """OLS slope of mean regret on mean absolute error across forecasters, re-estimated on 7-day block resamples of
    the evaluation days (the forecasters are the same in every resample)."""
    days = daily_regret.index
    R, E = daily_regret.to_numpy(), daily_mae.to_numpy()          # (days, forecasters)
    x, y = E.mean(axis=0), R.mean(axis=0)
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(SEED)
    n = len(days)
    starts = np.arange(0, n - BLOCK + 1)
    reps = []
    for _ in range(N_BOOT):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, int(np.ceil(n / BLOCK)))])[:n]
        reps.append(np.polyfit(E[idx].mean(axis=0), R[idx].mean(axis=0), 1)[0])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return slope, float(lo), float(hi)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    names, final, window = forecasters()
    rows, reg_daily, mae_daily = [], {}, {}
    runs = [(n, 180) for n in names] + [(final, window)]
    for name, win in runs:
        label = f"{name} (window {win})"
        run = R2.run_source(name, win, full=False)
        mae, pin, per_day = accuracy(run["F"])
        m = run["metrics"]
        for arm in ("P", "A2", "A5"):
            r = m[(m.split == "dev_test") & (m.arm == arm)].iloc[0]
            rows.append(dict(forecaster=name, window=win, is_final=(name == final and win == window), rule=arm,
                             mae_gwh=mae, pinball_gwh=pin, system_mean_crore=r.system_mean_crore,
                             system_cvar90_crore=r.system_cvar90_crore, J=r.J, n_state_days=int(r.n_state_days)))
        if win == 180 or name != final:
            reg_daily[label] = run["daily"][("dev_test", "A2")]
            mae_daily[label] = per_day.ae
        print(f"{label}: MAE {mae:.4f} GWh, A2 mean regret {rows[-2]['system_mean_crore']:.4f} crore/day", flush=True)

    V = pd.DataFrame(rows)
    V.to_csv(os.path.join(OUT, "value_of_forecast.csv"), index=False)

    # the final forecaster with its own window must give the confirmatory A2 and P numbers
    conf = pd.read_csv(os.path.join(D.DEV2, "o5", "metrics.csv"))
    check = {}
    for arm in ("P", "A2", "A5"):
        a = V[(V.is_final) & (V.rule == arm)].system_mean_crore.iloc[0]
        b = conf[(conf.split == "dev_test") & (conf.arm == arm)].system_mean_crore.iloc[0]
        check[arm] = dict(extra=float(a), confirmatory=float(b), same=bool(abs(a - b) < 1e-6))
    if not all(c["same"] for c in check.values()):
        raise SystemExit(f"the final forecaster does not reproduce the confirmatory result: {check}")

    R = pd.DataFrame(reg_daily).dropna()
    E = pd.DataFrame(mae_daily).reindex(R.index)
    R, E = R.loc[E.notna().all(axis=1)], E.dropna()
    slope, lo, hi = block_slope(R, E)
    a2 = V[(V.rule == "A2") & ~V.is_final]
    rho = float(a2[["mae_gwh", "system_mean_crore"]].corr(method="spearman").iloc[0, 1])
    best, worst = a2.loc[a2.mae_gwh.idxmin()], a2.loc[a2.mae_gwh.idxmax()]
    summary = dict(
        note="additional analysis, not pre-registered; frozen rules, stored forecasts, evaluation period only",
        n_forecasters=int(len(a2)), n_days=int(len(R)),
        slope_crore_per_day_per_gwh=slope, slope_lo=lo, slope_hi=hi, spearman_mae_regret=rho,
        best=dict(forecaster=best.forecaster, mae_gwh=float(best.mae_gwh), a2_mean_crore=float(best.system_mean_crore)),
        worst=dict(forecaster=worst.forecaster, mae_gwh=float(worst.mae_gwh), a2_mean_crore=float(worst.system_mean_crore)),
        reproduction=check, seconds=round(time.time() - t0))
    json.dump(summary, open(os.path.join(OUT, "value_of_forecast.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
