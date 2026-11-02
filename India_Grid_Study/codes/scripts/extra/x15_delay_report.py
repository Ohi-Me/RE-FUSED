"""Report for the delay study (additional analysis, not pre-registered): reads the outputs of x13 (E1, E2) and x14
(E3) and writes results/extra/delay/REPORT.md. Runs locally on pulled results; no model is fitted.
Usage: python codes/scripts/extra/x15_delay_report.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused.paths import RES  # noqa: E402

D = os.path.join(RES, "extra", "delay")
REG = {"T1": 1, "T2": 1, "T3": 2, "T4": 2, "T5": 14}           # registry lag of each target's own series (days)
GR = ["psp", "dsm_prices", "cea_npp", "cea_re", "co2", "imd_weather"]


def load(exp):
    fs = sorted(glob.glob(os.path.join(D, exp, "summary_T*.csv")))
    return pd.concat([pd.read_csv(f).assign(tid=os.path.basename(f)[8:10]) for f in fs], ignore_index=True) if fs else None


def e1(S):
    out = ["## E1. Publication delay as forecast horizon\n",
           "| Target | Own lag (registry) | States | Effective delay d (median, days) | IQR | pub(h) vs inst(h+round d): cells within ±5% |",
           "|---|---|---|---|---|---|"]
    shape = ["\n### Cost of one more day (instant LightGBM, pooled MAE by horizon)\n"]
    for tid, d in S.groupby("tid"):
        p = d[d.model == "published"].pivot(index="sid", columns="H", values="mae")
        i = d[d.model == "instant"].pivot(index="sid", columns="H", values="mae")
        hs = np.array(sorted(i.columns))
        de = []
        for sid in p.index.intersection(i.index):
            c = i.loc[sid].reindex(hs).to_numpy()
            if np.all(np.isfinite(c)) and np.isfinite(p.loc[sid, 1]):
                c = np.maximum.accumulate(c)                  # monotone envelope for interpolation
                de.append(float(np.interp(p.loc[sid, 1], c, hs)) - 1)
        de = np.array(de)
        k = int(round(np.median(de))) if len(de) else 0
        cells = [((p[h] - i[h + k]) / i[h + k]).abs() <= 0.05 for h in p.columns if h + k in i.columns] if k > 0 else []
        share = float(pd.concat(cells, axis=1).values.mean()) if cells else float("nan")
        out.append(f"| {tid} | {REG[tid]} | {len(de)} | {np.median(de):.2f} | {np.percentile(de, 25):.2f}–{np.percentile(de, 75):.2f} | {share:.3f} |")
        m = i.mean()
        shape.append(f"- {tid}: " + ", ".join(f"h{h} {m[h]:.2f}" for h in hs[:8]) +
                     f"; first extra day +{100 * (m[2] / m[1] - 1):.0f}%, then +{100 * (m[3] / m[2] - 1):.0f}%, +{100 * (m[4] / m[3] - 1):.0f}%")
    return out + shape


def e2(S):
    out = ["\n## E2. Value of each source's timeliness (pooled MASE, h = 1-3, evaluation period)\n",
           "| Target | published | instant | " + " | ".join(f"only {g}" for g in GR) + " |",
           "|---|---|---|" + "---|" * len(GR)]
    share = ["\nShare of the full timeliness gain (published − instant) captured by making one source timely:\n"]
    for tid, d in S.groupby("tid"):
        m = d.groupby("model").mase.mean()
        gain = m["published"] - m["instant"]
        out.append(f"| {tid} | {m['published']:.3f} | {m['instant']:.3f} | " +
                   " | ".join(f"{m.get('only_' + g, np.nan):.3f}" for g in GR) + " |")
        share.append(f"- {tid}: " + ", ".join(f"{g} {100 * (m['published'] - m.get('only_' + g, np.nan)) / gain:.0f}%"
                                               for g in GR))
    return out + share


def e3(V):
    out = ["\n## E3. Settlement value of timely publication (T2 drawal, rules P and A2, Rs crore a day)\n",
           "| Source | Rule | MAE (GWh) | Mean regret | CVaR 90 | J | vs lgbm published (crore/day) | per year (crore) |",
           "|---|---|---|---|---|---|---|---|"]
    for rule in ("A2", "P"):
        base = V[(V.source == "lgbm") & (V.rule == rule)].system_mean_crore.iloc[0]
        for _, r in V[V.rule == rule].iterrows():
            dlt = r.system_mean_crore - base
            out.append(f"| {r.source} | {rule} | {r.mae_gwh:.3f} | {r.system_mean_crore:.3f} | {r.cvar90_crore:.3f} | "
                       f"{r.J:.3f} | {dlt:+.3f} | {365 * dlt:+.0f} |")
    return out


if __name__ == "__main__":
    lines = ["# Delay experiments (additional analysis, not pre-registered)\n"]
    s1, s2 = load("E1"), load("E2")
    if s1 is not None:
        lines += e1(s1)
    if s2 is not None:
        lines += e2(s2)
    f3 = os.path.join(D, "value_of_timeliness.csv")
    if os.path.exists(f3):
        lines += e3(pd.read_csv(f3))
    open(os.path.join(D, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
