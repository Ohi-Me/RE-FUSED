"""Additional analysis (not pre-registered): the price of adaptivity on the Indian panel in energy units.

The confirmatory test of family F5 compares refinements of a gated correction in scaled squared error. This script
asks the operator's question on the same forecasts: how much forecast-error energy, in GWh per year summed over all
34 control areas, does each design leave, and how much does a finer gate save or add relative to one fixed gate?

Designs, all read from the stored confirmatory predictions (nothing is refitted):
  base      the base forecast b without correction
  fixed     b + g r with one gate                        (results/confirm/o3/preds/<T>__mcag_fixed)
  regime    one gate per regime cell                      (<T>__mcag_regime)
  instance  one gate per observation                      (<T>__mcag_instance)
  otg       the fixed gate re-estimated online over time  (o3_02_evaluate.otg, as in the F5 test)
  rule:<r>  refinement r used only where the validation-only PART rule chose it (part_decision.csv, refine_rule)

Evaluation period 1 Apr 2025 - 31 Aug 2026 ("dev_test" split of the confirm phase), median forecasts, horizons 1-3.
For T1-T4 (GWh/day) the error energy is the sum over areas of |y - forecast| per day, annualised (x 365.25); for T5
(Rs/MWh, per bid area) the mean absolute error. Differences against the fixed gate are paired by day, with a 7-day
block-bootstrap 95 % interval.

Outputs: results/extra/gate_energy.csv (one row per target x horizon x design), results/extra/gate_energy.json
Usage:   REFUSED_PHASE=confirm python codes/scripts/extra/x05_gate_energy.py     (run on the H100: hpc/extra_gate_energy.pbs)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
sys.path.insert(0, os.path.join(HERE, "..", "o3"))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import RES  # noqa: E402
import o3_02_evaluate as E3  # noqa: E402

KEY = ["sid", "issue_date", "H"]
OUT = os.path.join(RES, "extra")
DAYS = 365.25
REF = {"regime": "regime vs fixed", "instance": "instance vs fixed", "otg": "OTG vs fixed"}


def designs(tid):
    """All designs for one target, aligned on the same rows (inner join on series, issue day and horizon)."""
    E3.O3 = os.path.join(D.DEV1, "o3")
    fx = E3.load(tid, "mcag_fixed")
    if fx is None:
        return None
    Ly = D.lag_of(tid)
    V = {"fixed": fx, "regime": E3.load(tid, "mcag_regime"), "instance": E3.load(tid, "mcag_instance"),
         "otg": E3.otg(fx, Ly)}
    base = fx[KEY + ["target_date", "split", "y", "b50"]].rename(columns={"b50": "pred_base"})
    out = base
    for k, d in V.items():
        if d is None:
            continue
        out = out.merge(d[KEY + ["q50"]].rename(columns={"q50": "pred_" + k}), on=KEY)
    return out[(out.split == "dev_test") & out.y.notna()].reset_index(drop=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rule = pd.read_csv(os.path.join(D.DEV2, "o3", "eval", "part_decision.csv"))
    rows = []
    for tid in O.TARGETS:
        J = designs(tid)
        if J is None:
            print(tid, "no fixed-gate predictions; skipped", flush=True)
            continue
        energy = tid != "T5"
        for H in O.HORIZONS:
            j = J[J.H == H].copy()
            preds = [c for c in j.columns if c.startswith("pred_")]
            for r, lab in REF.items():
                if "pred_" + r not in j:
                    continue
                pick = rule[(rule.tid == tid) & (rule.H == H) & (rule.refinement == lab)]
                use = bool(pick.refine_rule.iloc[0]) if len(pick) else False
                j["pred_rule_" + r] = j["pred_" + r] if use else j["pred_fixed"]
                preds.append("pred_rule_" + r)
            err = {p[5:]: (j.y - j[p]).abs() for p in preds}
            if energy:
                daily = {k: v.groupby(j.target_date).sum().sort_index() for k, v in err.items()}
                scale = DAYS
            else:
                daily = {k: v.groupby(j.target_date).mean().sort_index() for k, v in err.items()}
                scale = 1.0
            for k, s in daily.items():
                diff = (s - daily["fixed"]).to_numpy()
                ci = S.block_ci(diff, 7, n_boot=2000, seed=0) if k != "fixed" else dict(lo=0.0, hi=0.0)
                rows.append(dict(tid=tid, target=O.TARGETS[tid]["name"], H=H, design=k,
                                 unit="GWh per year (all areas)" if energy else "Rs/MWh (mean absolute error)",
                                 error=float(s.mean() * scale), diff_vs_fixed=float(diff.mean() * scale),
                                 lo=float(ci["lo"] * scale), hi=float(ci["hi"] * scale),
                                 n_days=int(len(s)), n_series=int(j.sid.nunique()), n_rows=int(len(j))))
        print(tid, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "gate_energy.csv"), index=False)
    # summary for the paper: per design, summed over T1-T4 at horizon 1 (the day-ahead case) and over all horizons
    summ = {}
    E = R[R.tid != "T5"]
    for H in [1, "all"]:
        e = E if H == "all" else E[E.H == H]
        summ[str(H)] = {d: dict(error=float(g.error.sum()), diff_vs_fixed=float(g.diff_vs_fixed.sum()))
                        for d, g in e.groupby("design")}
    json.dump(dict(note="additional analysis, not pre-registered; GWh per year summed over 34 control areas",
                   period="1 Apr 2025 - 31 Aug 2026", by_horizon=summ,
                   rule=rule[["tid", "H", "refinement", "refine_rule"]].to_dict("records")),
              open(os.path.join(OUT, "gate_energy.json"), "w"), indent=1)
    print(R[R.H == 1].to_string(index=False))


if __name__ == "__main__":
    main()
