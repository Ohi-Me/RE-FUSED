"""Exploratory check after the frozen fusion run (not part of its protocol, written after seeing its result):
is the gain of the fusion variants over F0 a gain from adaptivity, or from F0 being a less well fitted fixed level?

Compares, on the evaluation period and with paired 7-day block-bootstrap intervals, the stored fusion schedules
(results/extra/fusion_tpwrs_confirm_schedules.parquet) with one fixed quantile level chosen on the calibration year
by grid search over all 99 levels, using the real projection onto the schedule limits (as in x06_dsm_regulation.py).
Output: results/extra/fusion_vs_fixed_level.json
Usage:  REFUSED_PHASE=confirm REFUSED_CONFIRMATORY=1 python codes/scripts/extra/x08_fusion_vs_fixed_level.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
for sub in ("o2", "o4", "o5", "dev2", "extra"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from refused import dev2 as D  # noqa: E402
from refused.paths import RES  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402
import d2_09_o5 as R2  # noqa: E402
import x06_dsm_regulation as X6  # noqa: E402


def main():
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    F = R2.build_v2(sel["final"], int(sel["calibration_window"])).reset_index(drop=True)
    SC = O5.scenarios(F[D.QN].to_numpy())
    X, p, lam = F.X.to_numpy(), F.p.to_numpy(), F.lam.to_numpy()
    val, test = (F.split == "validation").to_numpy(), (F.split == "dev_test").to_numpy()
    sch = pd.read_parquet(os.path.join(RES, "extra", "fusion_tpwrs_confirm_schedules.parquet"))
    key = F.loc[test, ["entity", "date"]].reset_index()
    sch = key.merge(sch, on=["entity", "date"], how="left").set_index("index")
    levels = np.arange(1, 100)
    S_lvl = np.column_stack([O5.project(F, SC[:, k - 1]) for k in levels])
    out = {}
    for sname in ("stylised", "dsm_nominal"):
        R = np.column_stack([X6.regret_s(S_lvl[:, j], X, p, lam, sname) for j in range(len(levels))])
        j_cal = int(R[val].sum(axis=0).argmin())
        d_fix = X6.system_daily(F, R[:, j_cal], test)
        res = dict(fixed_level=int(levels[j_cal]), fixed=X6.summary(d_fix))
        for v in ("F0", "F1", "F2", "F3", "F4", "F5"):
            S_v = np.full(len(F), np.nan)
            S_v[test] = sch.loc[F.index[test], f"{sname}_{v}"].to_numpy()
            d_v = X6.system_daily(F, X6.regret_s(S_v, X, p, lam, sname), test)
            res[v] = dict(summary=X6.summary(d_v), vs_fixed_level=X6.paired(d_v, d_fix))
        out[sname] = res
        print(sname, "fixed level", res["fixed_level"], round(res["fixed"]["mean"], 3), round(res["fixed"]["J"], 3))
        for v in ("F0", "F1", "F2", "F3", "F4", "F5"):
            c = res[v]["vs_fixed_level"]
            print(f"  {v} - fixed: {c['mean_diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] p={c['p_a_better']:.4f}"
                  f"  J {res[v]['summary']['J']:.2f}", flush=True)
    out["note"] = ("written after the frozen fusion run; compares the frozen schedules with a grid-chosen fixed level "
                   "on the calibration year; exploratory")
    json.dump(out, open(os.path.join(RES, "extra", "fusion_vs_fixed_level.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
