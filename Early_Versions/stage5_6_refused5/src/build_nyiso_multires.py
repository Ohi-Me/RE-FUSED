"""RE-FUSED-5 D3: NYISO testbed at FOUR resolutions from one source (5min/15min/1h/1d),
with the day-ahead price as an operational baseline. Enables the resolution law (N4)."""
import pandas as pd, numpy as np, glob, os, json
D = r"D:\\REFUSED5\data\nyiso"; OUT = r"D:\\REFUSED5\data"
def load(prod, cols):
    fs = sorted(glob.glob(os.path.join(D, prod, "*.parquet")))
    out = []
    for f in fs:
        d = pd.read_parquet(f)
        d.columns = [c.strip() for c in d.columns]
        ren = {c: n for c, n in zip(d.columns, d.columns)}
        out.append(d.rename(columns=ren)[cols])
    return pd.concat(out, ignore_index=True)
rt = load("realtime_zone", ["Time Stamp", "Name", "LBMP ($/MWHr)"])
rt.columns = ["ts", "zone", "lbmp"]
rt["ts"] = pd.to_datetime(rt["ts"], errors="coerce"); rt = rt.dropna(subset=["ts", "lbmp"])
da = load("damlbmp_zone", ["Time Stamp", "Name", "LBMP ($/MWHr)"])
da.columns = ["ts", "zone", "da"]
da["ts"] = pd.to_datetime(da["ts"], errors="coerce"); da = da.dropna(subset=["ts", "da"])
print(f"RT rows={len(rt):,} zones={rt.zone.nunique()} span={rt.ts.min()}..{rt.ts.max()}")
print(f"DA rows={len(da):,} zones={da.zone.nunique()} span={da.ts.min()}..{da.ts.max()}")
rt = rt.sort_values(["zone", "ts"])
step = rt.groupby("zone")["ts"].diff().dt.total_seconds().div(60)
print("RT interval minutes: median", float(step.median()), "| 5-min share",
      round(100 * float((step == 5).mean()), 1), "%")
rep = {"rt_rows": int(len(rt)), "da_rows": int(len(da)), "zones": sorted(rt.zone.unique().tolist())}
for res, rule in [("5min", "5min"), ("15min", "15min"), ("1h", "1h"), ("1d", "1D")]:
    g = (rt.set_index("ts").groupby("zone")["lbmp"].resample(rule)
           .agg(["mean", "max", "min", "std", "count"]).reset_index())
    g.columns = ["zone", "ts", "y", "y_max", "y_min", "y_sd", "n_obs"]
    g = g[g.n_obs > 0]
    dah = da.set_index("ts").groupby("zone")["da"].resample(rule).mean().reset_index()
    m = g.merge(dah, on=["zone", "ts"], how="left")
    m = m.sort_values(["zone", "ts"])
    m["lag1"] = m.groupby("zone")["y"].shift(1)
    m["split"] = np.where(m.ts < "2024-01-01", "train", np.where(m.ts < "2025-01-01", "val", "test"))
    out = os.path.join(OUT, f"nyiso_{res}.parquet"); m.to_parquet(out, index=False)
    v = m.dropna(subset=["lag1", "da"])
    mae_p = float(np.abs(v.y - v.lag1).mean()); mae_d = float(np.abs(v.y - v.da).mean())
    rep[res] = dict(rows=int(len(m)), mae_persistence=round(mae_p, 3), mae_dayahead=round(mae_d, 3),
                    y_mean=round(float(m.y.mean()), 2), y_sd=round(float(m.y.std()), 2),
                    da_better_pct=round(100 * float((np.abs(v.y - v.da) < np.abs(v.y - v.lag1)).mean()), 1))
    print(f"{res:5s} rows={len(m):>9,} | MAE persistence={mae_p:8.3f} day-ahead={mae_d:8.3f} "
          f"| DA better on {rep[res]['da_better_pct']}% of steps | sd(y)={rep[res]['y_sd']}")
json.dump(rep, open(os.path.join(OUT, "D3_nyiso_report.json"), "w"), indent=2)
