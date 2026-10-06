"""RE-FUSED-5 D2: modelling dataset from a BALANCED station panel (no structural jumps),
strict chronological splits, and baseline strength measured per state."""
import pandas as pd, numpy as np, json, os
RAW = r"D:\\Refused0\Data_Preprocessing\dataset\IDP\daily-power-generation.csv"
OUT = r"D:\\REFUSED5\data"
g = pd.read_csv(RAW, usecols=["date","state_name","sector","station_type","power_station",
                              "power_station_unit","monitored_capacity","todays_gen_prgm","todays_gen_act"],
                low_memory=False)
g["date"] = pd.to_datetime(g["date"], errors="coerce"); g = g.dropna(subset=["date","state_name"])
for c in ["todays_gen_prgm","todays_gen_act","monitored_capacity"]:
    g[c] = pd.to_numeric(g[c], errors="coerce")
g["unit_id"] = g.state_name + "|" + g.power_station.astype(str) + "|" + g.power_station_unit.astype(str)
WIN0, WIN1 = pd.Timestamp("2017-09-01"), pd.Timestamp("2025-04-22")
g = g[(g.date >= WIN0) & (g.date <= WIN1)]
ndays = g.date.nunique()
cnt = g.groupby("unit_id")["date"].nunique()
keep = cnt[cnt >= 0.95 * ndays].index
gb = g[g.unit_id.isin(keep)]
print(f"days={ndays} units_total={len(cnt)} units_balanced={len(keep)} ({100*len(keep)/len(cnt):.1f}%) "
      f"rows kept={len(gb)}/{len(g)} ({100*len(gb)/len(g):.1f}%)")
p = gb.groupby(["date","state_name"], as_index=False).agg(
        y=("todays_gen_act","sum"), sched=("todays_gen_prgm","sum"),
        cap=("monitored_capacity","sum"), n_units=("unit_id","nunique"))
# keep states with a stable balanced panel and meaningful size
stab = p.groupby("state_name")["n_units"].agg(["mean","std","min","max"])
size = p.groupby("state_name")["y"].mean()
good = stab[(stab["std"] / stab["mean"] < 0.02)].index.intersection(size[size > 20].index)
p = p[p.state_name.isin(good)].sort_values(["state_name","date"]).reset_index(drop=True)
print(f"states kept: {len(good)} -> {sorted(good)}")
# baselines and calendar features (all causal)
p["lag1"] = p.groupby("state_name")["y"].shift(1)
p["lag7"] = p.groupby("state_name")["y"].shift(7)
p["roll7"] = p.groupby("state_name")["y"].shift(1).rolling(7).mean().reset_index(0, drop=True)
p["dow"] = p.date.dt.dayofweek; p["month"] = p.date.dt.month; p["doy"] = p.date.dt.dayofyear
p["split"] = np.where(p.date <= "2022-12-31", "train", np.where(p.date <= "2023-12-31", "val", "test"))
p = p.dropna(subset=["lag1","lag7","roll7"])
print(p.groupby("split").agg(rows=("y","size"), start=("date","min"), end=("date","max")).to_string())
# baseline quality per split (MAE and MASE with m=7 in-sample scaling from train)
def mase_scale(df):
    tr = df[df.split == "train"].sort_values("date")
    return np.mean(np.abs(tr.y.values[7:] - tr.y.values[:-7]))
rows = []
for st, d in p.groupby("state_name"):
    sc = mase_scale(d)
    te = d[d.split == "test"]
    for nm, col in [("persistence", "lag1"), ("seasonal_naive7", "lag7"), ("roll7", "roll7"), ("operator_schedule", "sched")]:
        e = np.abs(te.y - te[col])
        rows.append(dict(state=st, baseline=nm, MAE=e.mean(), MASE=e.mean()/sc, n=len(te)))
b = pd.DataFrame(rows)
print("\n=== baseline quality on TEST (mean over states)")
print(b.groupby("baseline")[["MAE","MASE"]].mean().round(3).sort_values("MASE").to_string())
print("\n=== per-state MASE (persistence vs schedule)")
piv = b.pivot(index="state", columns="baseline", values="MASE").round(3)
print(piv.to_string())
p.to_parquet(os.path.join(OUT, "india_model_daily.parquet"), index=False)
b.to_csv(os.path.join(OUT, "D2_baselines_test.csv"), index=False)
json.dump(dict(days=int(ndays), units_balanced=int(len(keep)), states=sorted(map(str, good)),
               rows=int(len(p))), open(os.path.join(OUT, "D2_report.json"), "w"), indent=2)
