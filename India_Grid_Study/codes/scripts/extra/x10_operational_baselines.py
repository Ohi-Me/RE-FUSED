"""Stronger operational baselines for the scheduling rules (exploratory, not pre-registered).

Persistence of the recorded schedule of day t-2 is a weak baseline for an operator who has weeks of schedules and
drawal at hand. Five further baselines use only values published by the decision time and have nothing to fit:
  H7     recorded schedule of the same weekday one week before (t-7)
  WK4    median of the recorded schedules of the same weekday over the four weeks before (t-7, -14, -21, -28)
  M7     mean of the recorded schedules of the seven latest published days (t-2 ... t-8)
  X2     actual drawal of day t-2
  X7     actual drawal of the same weekday one week before
Where a value is missing the baseline falls back to the schedule of day t-2, then to the forecast median, as the
persistence baseline of the paper does. Each is scored projected onto the schedule limits (like every rule) and not
projected (as the States' own schedules are), under the stylised and the DSM-closer settlement, with fixed and with
demand-following limits. The comparison is A2 against (a) the baseline that is best on the calibration year and
(b) the baseline that is best on the evaluation period, which favours the baselines.

Run once on the evaluation period (already opened by the pre-registered study; exploratory). Output:
results/extra/operational_baselines.json.  Usage: REFUSED_PHASE=confirm REFUSED_CONFIRMATORY=1 python x10_...py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
for sub in ("o2", "o4", "o5", "dev2", "extra"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from refused import dev2 as D  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402
import d2_09_o5 as R2  # noqa: E402
import x06_dsm_regulation as X6  # noqa: E402

OUT = X6.OUT
NAMES = ["H7", "WK4", "M7", "X2", "X7"]


def lagged(F):
    P = O5.O.load_panel()[["entity", "date", "dev_drawal_schedule_gwh", "dev_actual_drawal_gwh"]]
    parts = []
    for e, g in P.sort_values(["entity", "date"]).groupby("entity"):
        g = g.set_index("date")[["dev_drawal_schedule_gwh", "dev_actual_drawal_gwh"]].asfreq("D")
        s, a = g.dev_drawal_schedule_gwh, g.dev_actual_drawal_gwh
        d = pd.DataFrame(index=g.index)
        d["H7"] = s.shift(7)
        d["WK4"] = pd.concat([s.shift(7), s.shift(14), s.shift(21), s.shift(28)], axis=1).median(axis=1)
        d["M7"] = s.shift(2).rolling(7, min_periods=3).mean()
        d["X2"] = a.shift(2)
        d["X7"] = a.shift(7)
        d["entity"] = e
        parts.append(d.reset_index())
    B = pd.concat(parts)
    G = F[["entity", "date"]].merge(B, on=["entity", "date"], how="left")
    base = F.sched_pub.fillna(F.q50).to_numpy()
    return {n: G[n].to_numpy(dtype=float) for n in NAMES}, base


def main():
    t0 = time.time()
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    F0 = R2.build_v2(sel["final"], int(sel["calibration_window"])).reset_index(drop=True)
    SC = O5.scenarios(F0[D.QN].to_numpy())
    ph, lh = F0.p_hat.to_numpy(), F0.lam_hat.to_numpy()
    X, p, lam = F0.X.to_numpy(), F0.p.to_numpy(), F0.lam.to_numpy()
    val, test = (F0.split == "validation").to_numpy(), (F0.split == "dev_test").to_numpy()
    fz = D.frozen("o5", "params.json")["params"]
    raw, base = lagged(F0)
    for n in NAMES:
        raw[n] = np.where(np.isnan(raw[n]), base, raw[n])
    res = dict(note="exploratory, not pre-registered; baselines have no parameters; nothing refitted",
               n_state_days_test=int(test.sum()), missing_filled_share={n: float(np.mean(np.isnan(v))) for n, v in
                                                                        lagged(F0)[0].items()},
               cases={})
    limits = {"fixed": F0, "rolling": X6.rolling_limits(F0)}
    for sn in X6.SETTLEMENTS:
        a2_raw = X6.decide_s(SC, ph, lh, fz["A2"]["beta"], fz["A2"]["rho"], sn)
        for ln, F in limits.items():
            rules = {"H_pub": O5.project(F, base), "P": O5.project(F, F.q50.to_numpy()), "A2": O5.project(F, a2_raw)}
            for n in NAMES:
                rules[n] = O5.project(F, raw[n])
                rules[n + " (not projected)"] = raw[n]
            rules["H_pub (not projected)"] = base
            reg = {k: X6.regret_s(v, X, p, lam, sn) for k, v in rules.items()}
            d_test = {k: X6.system_daily(F, r, test) for k, r in reg.items()}
            d_val = {k: X6.system_daily(F, r, val).mean() for k, r in reg.items()}
            case = dict(rules={k: X6.summary(v) for k, v in d_test.items()}, cmp={})
            for proj in ("", " (not projected)"):
                cand = [n + proj for n in NAMES] + ["H_pub" + proj]
                best_val = min(cand, key=lambda k: d_val[k])
                best_test = min(cand, key=lambda k: d_test[k].mean())
                case["cmp"]["vs best on calibration year" + proj] = dict(
                    baseline=best_val, **X6.paired(d_test["A2"], d_test[best_val]))
                case["cmp"]["vs best on evaluation period" + proj] = dict(
                    baseline=best_test, **X6.paired(d_test["A2"], d_test[best_test]))
            res["cases"][f"{sn} | {ln} limits"] = case
            print(sn, ln, {k: round(v["mean"], 2) for k, v in case["rules"].items()}, flush=True)
            for k, v in case["cmp"].items():
                print("   A2", k, v["baseline"], round(v["mean_diff"], 3), [round(v["lo"], 3), round(v["hi"], 3)], flush=True)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, "operational_baselines.json"), "w"), indent=1)
    print("DONE", res["seconds"], "s")


if __name__ == "__main__":
    main()
