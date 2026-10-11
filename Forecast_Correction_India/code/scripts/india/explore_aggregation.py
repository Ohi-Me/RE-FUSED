"""Exploratory (not pre-registered): can an online combination of the gate designs do what the validation-only
rules could not, i.e. follow whichever design is paying at the time?

For each forecast and series, the experts are the 14 corrected forecasts B + g_k r (no correction and the 13 gate
designs) stored by the hourly ladders (IH1); nothing is refitted. The weights for target day D use only squared errors
of days up to D - 2 (the publication delay). Combiners:
  EWA(eta)        exponentially weighted average, losses scaled by the mean daily squared error of the uncorrected
                  forecast in the gate-fitting period
  FS(eta, alpha)  Fixed Share: EWA with a share alpha of the weight redistributed each day (tracks a changing best)
  LEAD30          use the single design with the lowest squared error over the last 30 published days
Phase dev:     reads results/india/dev/IH1 (development split), chooses eta (and alpha) by mean skill over all
               forecast x series pairs, writes results/india/explore_aggregation/choice.json and stops.
Phase confirm: reads choice.json (it must exist; it is written before any test file is read) and scores the frozen
               combiners on results/india/confirm/IH1 (test months 1 Mar - 13 Sep 2026, already opened).
Outputs: results/india/explore_aggregation/{choice.json, skill_<phase>.csv}.
Usage: python code/scripts/india/explore_aggregation.py --phase dev|confirm   (run on the H100)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ["REFUSED_GATE_STUDY"] = "india"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate.paths import RES  # noqa: E402

OUT = os.path.join(RES, "india", "explore_aggregation")
ETAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
ALPHAS = (0.01, 0.05)
DELAY = 2


def expert_errors(o):
    """Daily squared error of every expert, per series: dict sid -> (days, E[day, k]), expert names."""
    names = [c for c in o.columns if c.startswith("g_")]
    o = o.assign(day=pd.to_datetime(o.ts).dt.normalize())
    R = (o.y_t - o.B).to_numpy()
    out = {}
    for sid, g in o.groupby("sid"):
        idx = g.index
        e = np.column_stack([(R[idx] - g[n].to_numpy() * g.r.to_numpy()) ** 2 for n in names])
        d = pd.DataFrame(e, index=g.day.to_numpy()).groupby(level=0).sum().sort_index()
        out[sid] = (d.index.to_numpy(), d.to_numpy())
    return out, names


def combine(days, E, method, eta=1.0, alpha=0.0, scale=1.0):
    """Daily weights from losses up to day - DELAY; returns the combined daily squared error per day.
    The combined forecast's error is computed from expert errors via weights on the forecasts, so the squared error of
    the combination needs the forecasts themselves; this function returns the weights."""
    n, K = E.shape
    W = np.full((n, K), 1.0 / K)
    L = E / scale
    for i in range(n):
        past = days < days[i] - np.timedelta64(DELAY - 1, "D")       # days <= D - DELAY
        if not past.any():
            continue
        if method == "LEAD30":
            recent = past & (days >= days[i] - np.timedelta64(DELAY - 1 + 30, "D"))
            k = int(L[recent].sum(axis=0).argmin())
            W[i] = 0.0
            W[i, k] = 1.0
        elif method == "EWA":
            cum = L[past].sum(axis=0)
            w = np.exp(-eta * (cum - cum.min()))
            W[i] = w / w.sum()
        elif method == "FS":
            w = np.full(K, 1.0 / K)
            for j in np.where(past)[0]:
                w = w * np.exp(-eta * (L[j] - L[j].min()))
                w = w / w.sum()
                w = (1 - alpha) * w + alpha / K
            W[i] = w
    return W


def score_file(f, configs, scale_from=None):
    o = pd.read_parquet(f)
    names = [c for c in o.columns if c.startswith("g_")]
    o["day"] = pd.to_datetime(o.ts).dt.normalize()
    rows = []
    for sid, g in o.groupby("sid"):
        R = (g.y_t - g.B).to_numpy()
        P = np.column_stack([g[n].to_numpy() * g.r.to_numpy() for n in names])     # correction of each expert
        dd = g.day.to_numpy()
        days = np.unique(dd)
        E = np.column_stack([pd.Series((R - P[:, k]) ** 2).groupby(dd).sum().reindex(days).to_numpy()
                             for k in range(len(names))])
        scale = float(np.mean(pd.Series(R ** 2).groupby(dd).sum().to_numpy()))
        sse0 = float((R ** 2).sum())
        res = {n[2:]: 1 - float(((R - P[:, k]) ** 2).sum()) / sse0 for k, n in enumerate(names)}
        res["best_of_designs (ex post)"] = max(v for k, v in res.items() if k != "none")
        pos = np.searchsorted(days, dd)
        for cname, (m, eta, alpha) in configs.items():
            W = combine(days, E, m, eta, alpha, scale)
            corr = (W[pos] * P).sum(axis=1)
            res[cname] = 1 - float(((R - corr) ** 2).sum()) / sse0
        for k, v in res.items():
            rows.append(dict(file=os.path.basename(f), sid=sid, method=k, skill=v))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "confirm"], required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    choice_f = os.path.join(OUT, "choice.json")
    if a.phase == "dev":
        cfg = {f"EWA(eta={e})": ("EWA", e, 0.0) for e in ETAS}
        cfg.update({f"FS(eta={e},alpha={al})": ("FS", e, al) for e in ETAS for al in ALPHAS})
        cfg["LEAD30"] = ("LEAD30", 0.0, 0.0)
        files = sorted(glob.glob(os.path.join(RES, "india", "dev", "IH1", "*.parquet")))
    else:
        if not os.path.exists(choice_f):
            raise SystemExit("run --phase dev first: no frozen choice")
        ch = json.load(open(choice_f))
        cfg = {k: tuple(v) for k, v in ch["frozen"].items()}
        files = sorted(glob.glob(os.path.join(RES, "india", "confirm", "IH1", "*.parquet")))
    rows = []
    for f in files:
        rows += score_file(f, cfg)
        print("scored", os.path.basename(f), flush=True)
    df = pd.DataFrame(rows)
    df["forecast"] = df.file.str.replace(r"_s\d+\.parquet$", "", regex=True)
    df.to_csv(os.path.join(OUT, f"skill_{a.phase}_rows.csv"), index=False)
    tab = df.groupby(["forecast", "sid", "method"]).skill.mean().unstack("method")
    tab.to_csv(os.path.join(OUT, f"skill_{a.phase}.csv"))
    mean = tab.mean().sort_values(ascending=False)
    print(mean.round(4).to_string())
    if a.phase == "dev":
        ewa = max((k for k in mean.index if k.startswith("EWA")), key=lambda k: mean[k])
        fs = max((k for k in mean.index if k.startswith("FS")), key=lambda k: mean[k])
        json.dump(dict(note="chosen on the development split only, before any test file was read",
                       mean_dev_skill={k: float(mean[k]) for k in (ewa, fs, "LEAD30")},
                       frozen={ewa: cfg[ewa], fs: cfg[fs], "LEAD30": cfg["LEAD30"]}),
                  open(choice_f, "w"), indent=1)
        print("frozen:", ewa, fs, "LEAD30")


if __name__ == "__main__":
    main()
