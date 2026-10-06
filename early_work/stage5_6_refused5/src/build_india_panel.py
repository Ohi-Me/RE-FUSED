"""RE-FUSED-5 D1: rebuild the Indian state-day panel from RAW unit-level sources with provenance.
Key difference from RE-FUSED-4's panel: every field carries an observed/missing flag derived from the
source, nothing is forward-filled, and the operator's own scheduled generation is kept as a baseline."""
import os, json
import numpy as np, pandas as pd
RAW = r"D:\\Refused0\Data_Preprocessing\dataset"
OUT = r"D:\\REFUSED5\data"; os.makedirs(OUT, exist_ok=True)
rep = {}

# ---- 1. generation (unit-level -> state-day): actual AND operator schedule ----
g = pd.read_csv(os.path.join(RAW, "IDP", "daily-power-generation.csv"),
                usecols=["date","state_name","sector","station_type","power_station","power_station_unit",
                         "monitored_capacity","todays_gen_prgm","todays_gen_act"], low_memory=False)
g["date"] = pd.to_datetime(g["date"], errors="coerce")
g = g.dropna(subset=["date","state_name"])
for c in ["monitored_capacity","todays_gen_prgm","todays_gen_act"]:
    g[c] = pd.to_numeric(g[c], errors="coerce")
rep["gen_rows"] = len(g); rep["gen_units"] = int(g["power_station_unit"].nunique())
rep["gen_span"] = [str(g.date.min().date()), str(g.date.max().date())]
rep["gen_act_nan_pct"] = round(100*float(g.todays_gen_act.isna().mean()), 2)
rep["gen_prgm_nan_pct"] = round(100*float(g.todays_gen_prgm.isna().mean()), 2)

gs = g.groupby(["date","state_name"]).agg(
        gen_act=("todays_gen_act","sum"), gen_prgm=("todays_gen_prgm","sum"),
        cap_monitored=("monitored_capacity","sum"), n_units=("power_station_unit","size"),
        n_units_obs=("todays_gen_act","count")).reset_index()
gs["gen_obs_frac"] = gs.n_units_obs / gs.n_units
# thermal / hydro split (station_type) for structure
st = (g.assign(t=g.station_type.str.lower().str.strip())
        .pivot_table(index=["date","state_name"], columns="t", values="todays_gen_act", aggfunc="sum"))
st.columns = [f"gen_{c.replace(' ','_')}" for c in st.columns]
gs = gs.merge(st.reset_index(), on=["date","state_name"], how="left")

# ---- 2. outage (unit-level -> state-day) ----
o = pd.read_csv(os.path.join(RAW, "IDP", "daily-power-outage.csv"),
                usecols=["date","state_name","outage_type","monitored_capacity","cap_under_outage"], low_memory=False)
o["date"] = pd.to_datetime(o["date"], errors="coerce"); o = o.dropna(subset=["date","state_name"])
o["cap_under_outage"] = pd.to_numeric(o["cap_under_outage"], errors="coerce")
os_ = o.groupby(["date","state_name"]).agg(outage_mw=("cap_under_outage","sum"),
                                            outage_rows=("cap_under_outage","size"),
                                            outage_obs=("cap_under_outage","count")).reset_index()
os_["outage_obs_frac"] = os_.outage_obs / os_.outage_rows

# ---- 3. renewables (state-level daily; starts late 2019) ----
r = pd.read_csv(os.path.join(RAW, "IDP", "daily-renewable-energy-generation.csv"),
                usecols=["date","state_name","wind_energy","solar_energy","other_renewable_energy","total_renewable_energy"])
r["date"] = pd.to_datetime(r["date"], errors="coerce"); r = r.dropna(subset=["date","state_name"])
for c in ["wind_energy","solar_energy","other_renewable_energy","total_renewable_energy"]:
    r[c] = pd.to_numeric(r[c], errors="coerce")
r = r.groupby(["date","state_name"], as_index=False).agg(
        wind=("wind_energy","sum"), solar=("solar_energy","sum"),
        other_re=("other_renewable_energy","sum"), re_total=("total_renewable_energy","sum"))
rep["re_span"] = [str(r.date.min().date()), str(r.date.max().date())]

# ---- 4. coal stock (station-level -> state-day; starts late 2018) ----
c = pd.read_csv(os.path.join(RAW, "IDP", "daily-coal-stocks.csv"),
                usecols=["date","state_name","capacity","daily_requirement","daily_receipt","daily_consumption"], low_memory=False)
c["date"] = pd.to_datetime(c["date"], errors="coerce"); c = c.dropna(subset=["date","state_name"])
for col in ["capacity","daily_requirement","daily_receipt","daily_consumption"]:
    c[col] = pd.to_numeric(c[col], errors="coerce")
cs = c.groupby(["date","state_name"], as_index=False).agg(
        coal_cap=("capacity","sum"), coal_req=("daily_requirement","sum"),
        coal_recv=("daily_receipt","sum"), coal_cons=("daily_consumption","sum"))
rep["coal_span"] = [str(cs.date.min().date()), str(cs.date.max().date())]

# ---- 5. merge, with explicit observation flags; NO imputation ----
p = gs.merge(os_, on=["date","state_name"], how="left") \
      .merge(r,  on=["date","state_name"], how="left") \
      .merge(cs, on=["date","state_name"], how="left")
p = p.sort_values(["state_name","date"]).reset_index(drop=True)
for col, flag in [("gen_act","gen_obs"), ("gen_prgm","prgm_obs"), ("outage_mw","outage_obs_f"),
                  ("re_total","re_obs"), ("coal_cons","coal_obs")]:
    p[flag] = p[col].notna().astype(int)
rep["panel_rows"] = len(p); rep["panel_states"] = int(p.state_name.nunique())
rep["panel_span"] = [str(p.date.min().date()), str(p.date.max().date())]
rep["observed_pct"] = {k: round(100*float(p[k].mean()),2) for k in ["gen_obs","prgm_obs","re_obs","coal_obs"]}
# how good is the operator's own schedule as a baseline?
m = p.dropna(subset=["gen_act","gen_prgm"])
m = m[(m.gen_act > 0) & (m.gen_prgm > 0)]
err_prgm = (m.gen_act - m.gen_prgm).abs()
lag1 = m.groupby("state_name")["gen_act"].shift(1)
mask = lag1.notna()
rep["baseline_compare_rows"] = int(mask.sum())
rep["MAE_operator_schedule"] = round(float(err_prgm[mask].mean()), 2)
rep["MAE_persistence_lag1"]  = round(float((m.gen_act - lag1)[mask].abs().mean()), 2)
rep["schedule_beats_persistence_pct"] = round(100*float((err_prgm[mask] < (m.gen_act - lag1)[mask].abs()).mean()), 2)
p.to_parquet(os.path.join(OUT, "india_panel_raw_rebuild.parquet"), index=False)
json.dump(rep, open(os.path.join(OUT, "D1_panel_rebuild_report.json"), "w"), indent=2, default=str)
for k, v in rep.items(): print(f"{k}: {v}")
print("\nstates:", sorted(p.state_name.unique())[:20])
print("\ncoverage by year (gen_obs / re_obs / coal_obs):")
print(p.assign(yr=p.date.dt.year).groupby("yr")[["gen_obs","re_obs","coal_obs"]].mean().round(3).to_string())
