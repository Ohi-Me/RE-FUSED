"""Exploratory (not pre-registered): gated correction of the States' own recorded drawal schedules.

Question (reviewer request for the ECM paper): does the gate ladder behave the same when the forecast being corrected
is the State's own schedule rather than a forecast we built?

The State-day panel holds actual drawal X and the recorded schedule S (dev_drawal_schedule_gwh; X - S equals the
published net deviation dev_od_ud_gwh exactly). Grid-India publishes only the final schedule, which includes revisions
made during the day, so two baselines are used and reported separately:
  sched_persist  B = S of the latest published day (t - L0, L0 = 1 + publication delay). Available when the day-ahead
                 forecast is issued, so it is an operational baseline (the persistence rule of Paper 2).
  sched_final    B = S of day t itself. NOT available day-ahead (it contains intra-day revisions); it is a ceiling that
                 asks whether even the final schedule carries a learnable, correctable error.
The corrector sees lags of X, lags of the schedule error X - S (both at the published lag or older), calendar terms
and the area code; never same-day information. Same engine, settings and split as the State-panel ladders:
train < Apr 2024, gate fitting FY2024-25, test 1 Apr 2025 - 31 Aug 2026. Three seeds.

Outputs (never overwritten): results/india/explore_schedule/<baseline>_s<seed>.parquet (per-row test output),
summary.csv (skill per rung, deviation energy GWh/yr with 7-day block-bootstrap intervals against no correction and
against the per-area gate) and run.json (settings, row counts, timings).
Usage: python code/scripts/india/explore_schedule_correction.py      (run on the H100: hpc/explore_schedule.pbs)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

os.environ["REFUSED_GATE_STUDY"] = "india"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import data, ladder  # noqa: E402
from refused_gate.paths import INDIA_PROC, RES  # noqa: E402

OUT = os.path.join(RES, "india", "explore_schedule")
SEEDS = 3
NBOOT, BLOCK = 2000, 7


def frame():
    ds = data.india_state_daily("T2", "long")
    d = ds["df"]
    p = pd.read_parquet(os.path.join(INDIA_PROC, "refused_state_day.parquet"),
                        columns=["entity", "date", "dev_drawal_schedule_gwh"])
    p = p.rename(columns={"entity": "sid", "date": "ts", "dev_drawal_schedule_gwh": "S"})
    p["sid"], p["ts"] = p.sid.astype(str), pd.to_datetime(p.ts)
    d = d.merge(p, on=["sid", "ts"], how="left").sort_values(["sid", "ts"]).reset_index(drop=True)
    L0 = max(ds["delay_days"], 1 + data._india_avail_lag("dev_drawal_schedule_gwh"))
    g = d.groupby("sid")
    d["S_pub"] = g.S.shift(L0)
    d["E"] = d.y - d.S                                   # schedule error of the day (published with the same delay)
    feats = list(ds["feats"])
    for L in sorted({L0, L0 + 1, 7, 14}):
        d[f"E_lag{L}"] = g.E.shift(L)
        feats.append(f"E_lag{L}")
    d["E_roll7"] = g.E.transform(lambda s: s.shift(L0).rolling(7, min_periods=4).mean())
    d["S_pub_feat"] = d.S_pub
    feats += ["E_roll7", "S_pub_feat"]
    return d, feats, ds["state_keys"], ds["delay_days"], L0


def block_ci(x, rng):
    """Mean of daily values with a moving-block bootstrap interval (7-day blocks)."""
    n = len(x)
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, max(1, n - BLOCK + 1), size=(NBOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(NBOOT, -1)[:, :n]
    bs = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def summarise(outs):
    """Pool the seeds: per rung, squared-error skill and deviation energy (GWh per year) with block intervals."""
    rows = []
    rng = np.random.default_rng(20261010)
    for base, frames in outs.items():
        rungs = [c[2:] for c in frames[0].columns if c.startswith("g_")]
        err = {}
        for k in rungs:
            e_abs, e_sq = [], []
            for o in frames:
                R = (o.y_t - o.B).to_numpy()
                res = R - o["g_" + k].to_numpy() * o.r.to_numpy()
                e_abs.append(pd.Series(np.abs(res)).groupby(o.day.to_numpy()).sum())
                e_sq.append(pd.Series(res ** 2).groupby(o.day.to_numpy()).sum())
            err[k] = (pd.concat(e_abs, axis=1).mean(axis=1), pd.concat(e_sq, axis=1).mean(axis=1))
        days = err["none"][0].index
        for k in rungs:
            a, s = err[k][0].reindex(days), err[k][1].reindex(days)
            row = dict(baseline=base, rung=k, n_days=len(days),
                       energy_gwh_yr=float(a.mean() * 365.25), skill=float(1 - s.sum() / err["none"][1].sum()))
            for ref in ("none", "series"):
                diff = (a - err[ref][0].reindex(days)).to_numpy() * 365.25
                m, lo, hi = block_ci(diff, rng)
                row.update({f"d_vs_{ref}": m, f"d_vs_{ref}_lo": lo, f"d_vs_{ref}_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    d, feats, keys, delay, L0 = frame()
    info = dict(L0=L0, delay_days=delay, feats=feats, state_keys=keys, seeds=SEEDS, started=time.strftime("%c"))
    outs = {}
    for base, col in (("sched_persist", "S_pub"), ("sched_final", "S"), ("snaive7", "y_lag7")):
        fr = d.assign(y_t=d.y, B=d[col])
        outs[base] = []
        for s in range(SEEDS):
            f = os.path.join(OUT, f"{base}_s{s}.parquet")
            if os.path.exists(f):
                o = pd.read_parquet(f)
            else:
                t0 = time.time()
                res = ladder.run_ladder(fr, feats, keys, ladder.LadderConfig(n_bag=3, delay_days=L0), seed=s)
                if res is None:
                    print(base, s, "too few rows")
                    continue
                o = res[0] if isinstance(res, tuple) else res["test"] if isinstance(res, dict) else res
                o.to_parquet(f)
                print(f"{base} seed {s}: {len(o)} test rows, {time.time() - t0:.0f}s", flush=True)
            if "day" not in o:
                o["day"] = pd.to_datetime(o.ts).dt.normalize()
            outs[base].append(o)
        info[f"rows_{base}"] = int(fr.dropna(subset=feats + ["y_t", "B"]).shape[0])
    S = summarise(outs)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    info["finished"] = time.strftime("%c")
    json.dump(info, open(os.path.join(OUT, "run.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 200)
    print(S[["baseline", "rung", "skill", "energy_gwh_yr", "d_vs_none", "d_vs_none_lo", "d_vs_none_hi",
             "d_vs_series", "d_vs_series_lo", "d_vs_series_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
