"""Additional analysis, not pre-registered: where the settlement puts the best schedule, and why fixed caution is
hard to beat.

In the stylised settlement the regret of a schedule S is piecewise linear in the deviation D = X - S. Per unit, and
in units of the day-ahead price p with r = lambda / p:

    over-drawal (X > S)   inside the 10 % band: max(r, 1) - 1          beyond it: 1.2 max(r, 1) - 1
    under-drawal (X < S)  inside the 10 % band: 1 - 0.9 min(r, 1)      beyond it: 1 - 0.8 min(r, 1)

So the schedule that minimises expected regret (the risk-neutral rule, beta = 0) is a quantile of the forecast
distribution, as in the newsvendor problem, and the level of that quantile depends on the day only through r and the
band, not through how uncertain the forecast is or how stressed the system is. This script checks that on the data:

1. the distribution of r on the evaluation period, as the rule sees it (published prices) and as settled;
2. for every State-day, the risk-neutral schedule from the frozen rule's own solver (beta = 0, no widening) and the
   level tau of the calibrated forecast distribution at which it sits;
3. how much tau varies, and how strongly it follows r, forecast uncertainty (U_p) and system stress (S_p);
4. where the confirmatory schedules of the frozen rule A2 landed: the share of State-days inside the band.

It reads only stored results and writes to results/extra/, never into the confirmatory folders.
Outputs: results/extra/settlement_quantile.csv (per State-day) and settlement_quantile.json (summary)
Usage:   python run_all.py stage x_settlement_quantile   (confirmatory environment)
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
from refused.paths import RES  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402
import d2_09_o5 as R2  # noqa: E402

OUT = os.path.join(RES, "extra")


def marginal_costs(r):
    """Marginal regret per unit of deviation, in units of p (see the module docstring)."""
    hi, lo = np.maximum(r, 1.0), np.minimum(r, 1.0)
    return dict(over_in=hi - 1, over_out=1.2 * hi - 1, under_in=1 - 0.9 * lo, under_out=1 - 0.8 * lo)


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman"))


def summarise(x):
    x = x[np.isfinite(x)]
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(p05=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p95=float(q[4]),
                mean=float(x.mean()), n=int(len(x)))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    F = R2.build_v2(sel["final"], int(sel["calibration_window"]))      # the frame of the confirmatory run
    SC = O5.scenarios(F[D.QN].to_numpy())
    ph, lh = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
    S_star = O5.decide(SC, ph, lh, 0.0, 0.0)                            # risk-neutral schedule, no widening
    tau = (SC < S_star[:, None]).mean(axis=1)                           # its level in the calibrated distribution
    out = pd.DataFrame(dict(entity=F.entity, date=F.date, split=F.split, r_hat=lh / ph, r_settled=F.lam / F.p,
                            U_p=F.U_p, S_p=F.S_p, S_star=S_star, tau=tau))
    out.to_csv(os.path.join(OUT, "settlement_quantile.csv"), index=False)

    res = dict(note="additional analysis, not pre-registered; frozen solver, stored forecasts",
               marginal_costs_at_r1=marginal_costs(np.array(1.0)), splits={})
    for k in res["marginal_costs_at_r1"]:
        res["marginal_costs_at_r1"][k] = float(res["marginal_costs_at_r1"][k])
    for sp in ("validation", "dev_test"):
        o = out[out.split == sp]
        r = o.r_hat.to_numpy()
        res["splits"][sp] = dict(
            r_hat=summarise(r), r_settled=summarise(o.r_settled.to_numpy()),
            share_r_hat_within_10pct_of_1=float(np.mean(np.abs(r - 1) <= 0.1)),
            share_r_hat_at_least_1=float(np.mean(r >= 1)),
            tau=summarise(o.tau.to_numpy()),
            tau_iqr=float(np.subtract(*np.percentile(o.tau, [75, 25]))),
            spearman_tau_r=spearman(o.tau.to_numpy(), r),
            spearman_tau_U=spearman(o.tau.to_numpy(), o.U_p.to_numpy()),
            spearman_tau_S=spearman(o.tau.to_numpy(), o.S_p.to_numpy()),
            n_state_days=int(len(o)))

    # where the confirmatory schedules of the frozen rule A2 landed
    dec = pd.read_parquet(os.path.join(D.DEV2, "o5", "decisions.parquet"))
    for sp in ("validation", "dev_test"):
        d = dec[dec.split == sp]
        dev = (d.X - d.S_A2).abs()
        band = 0.1 * d.S_A2.abs()
        res["splits"][sp]["a2_share_inside_band"] = float((dev <= band).mean())
        res["splits"][sp]["a2_share_over_drawal"] = float((d.X > d.S_A2).mean())
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, "settlement_quantile.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
