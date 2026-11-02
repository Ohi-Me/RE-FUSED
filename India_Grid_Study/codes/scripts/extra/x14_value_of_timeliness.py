"""Additional analysis, not pre-registered (delay study, Nov 2026): what timely publication is worth in settlement.

Every day-ahead drawal forecast (T2) listed below is passed through the frozen rules of the chain test, P (forecast
median) and A2 (risk-aware), with the confirmatory calibration window. A5 is left out, so the script needs no GPU.
Sources: the published LightGBM (lgbm), the instant-publication LightGBM (lgbm_instant, stored S-lat run), the final
forecaster, and the E2 variants of x13 (one source group instant, or all but one), read from results/extra/delay/E2.
Check: the A2 and P mean regret of lgbm must equal results/extra/value_of_forecast.csv; the script stops if not.
Writes results/extra/delay/value_of_timeliness.csv only.
Usage: python codes/scripts/extra/x14_value_of_timeliness.py   (REFUSED_PHASE=confirm, REFUSED_CONFIRMATORY=1)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
for sub in ("o2", "o4", "o5", "dev2"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from refused import dev2 as D  # noqa: E402
from refused.paths import RES  # noqa: E402
import d2_09_o5 as R2  # noqa: E402  (the round-2 scheduling code that produced the confirmatory results)

O5 = R2.O5
OUT = os.path.join(RES, "extra", "delay")
E2DIR = os.path.join(OUT, "E2", "preds")
_find = D.find_member


def find_member(tid, name):
    """E2 variants live in results/extra/delay/E2/preds as T2__<variant>.parquet, named here 'e2_<variant>'."""
    if name.startswith("e2_"):
        f = os.path.join(E2DIR, f"{tid}__{name[3:]}.parquet")
        return f if os.path.exists(f) else None
    return _find(tid, name)


D.find_member = find_member


def cvar(x, a=0.9):
    x = np.sort(np.asarray(x, float))
    return float(x[int(np.floor(a * len(x))):].mean())


def value(name, window, fz):
    F = R2.build_v2(name, window)
    SC = O5.scenarios(F[D.QN].to_numpy())
    ph, lh = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
    # exactly as run_source: project over the whole frame, regret per State-day, then arm_metrics per split
    arms = {"P": O5.project(F, F.q50.to_numpy()),
            "A2": O5.project(F, O5.decide(SC, ph, lh, fz["A2"]["beta"], fz["A2"]["rho"]))}
    dec = F[["entity", "date", "split", "X", "p", "lam", "S_hist", "S_min", "S_max", "dS_max"]].copy()
    for arm, S_ in arms.items():
        dec[f"S_{arm}"] = S_
        dec[f"regret_{arm}"] = O5.regret(S_, F.X.to_numpy(), F.p.to_numpy(), F.lam.to_numpy())
    M, daily = R2.arm_metrics(dec, list(arms))
    dev = F[F.split == "dev_test"]
    mae = float((dev.X - dev.q50).abs().mean())
    rows = []
    for _, m in M[M.split == "dev_test"].iterrows():
        rows.append(dict(source=name, rule=m.arm, system_mean_crore=float(m.system_mean_crore),
                         cvar90_crore=float(m.system_cvar90_crore), J=float(m.J), mae_gwh=mae,
                         days=int(len(daily[("dev_test", m.arm)]))))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    window = int(sel["calibration_window"])
    fz = D.frozen("o5", "params.json")["params"]
    # windows as in x01: 180 days for every single model, the final forecaster with its own window
    names = ["lgbm", "lgbm_instant"]
    names += ["e2_" + os.path.basename(f)[len("T2__"):-len(".parquet")] for f in sorted(glob.glob(os.path.join(E2DIR, "T2__*.parquet")))]
    runs = [(n, 180) for n in names] + [(sel["final"], window)]
    rows = []
    for n, win in runs:
        if find_member("T2", n) is None:
            print("missing forecast:", n, flush=True)
            continue
        rows += [dict(r, window=win) for r in value(n, win, fz)]
        print(n, win, [(r["rule"], round(r["system_mean_crore"], 4)) for r in rows[-2:]], flush=True)
    V = pd.DataFrame(rows)
    # reproduction check against the stored value-of-forecast analysis
    ref = pd.read_csv(os.path.join(RES, "extra", "value_of_forecast.csv"))
    col = "system_mean_crore" if "system_mean_crore" in ref else [c for c in ref.columns if "mean" in c][0]
    src = "source" if "source" in ref else ref.columns[0]
    for arm in ("P", "A2"):
        a = V[(V.source == "lgbm") & (V.rule == arm)].system_mean_crore.iloc[0]
        b = ref[(ref[src] == "lgbm") & (ref.rule == arm)][col].iloc[0]
        print(f"check lgbm {arm}: {a:.6f} vs stored {b:.6f}", flush=True)
        if abs(a - b) > 1e-6:
            sys.exit(f"reproduction failed for lgbm {arm}: {a} vs {b}")
    V.to_csv(os.path.join(OUT, "value_of_timeliness.csv"), index=False)
    print(V.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
