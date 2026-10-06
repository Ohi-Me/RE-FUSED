"""Build step 5: State × fuel × financial-year CO2 emission factors from the CEA CO2 Baseline Database (v22.0).

Inputs : 03_data/raw/cea_co2_database/Baseline_Carbon_Dioxide_Emission_Database_Version_22.0.xlsx, sheet "Data"
           station rows (UNIT_NO = 0): STATE, TYPE, FUEL 1 and, per FY 2017-18 … 2025-26, net generation (GWh),
           absolute emissions (t CO2) and specific emissions (t CO2/MWh)
         03_data/interim/npp_station_day.parquet (daily gross generation by State and fuel, from CEA NPP DGR2)
Outputs: 03_data/interim/co2_ef_state_fuel_fy.parquet
           entity, fuel, fy, emis_t (CEA), netgen_gwh (CEA), ef_net (t/MWh, CEA emissions / CEA net generation),
           npp_gross_gwh (NPP DGR2, same State, fuel and FY), npp_days (dates with DGR2 data in the FY),
           net_gross_ratio, ratio_source, ef_gross (t CO2 per MWh of NPP gross generation)
Method : station-name matching between the two CEA products is ambiguous (e.g. 'ANAPARA "C"' vs 'ANPARA C TPS'), so
         factors are built at State × fuel resolution. CEA factors refer to net generation while NPP reports gross
         generation, so ef_gross = ef_net × (net/gross ratio); the ratio is Σ CEA net / Σ NPP gross over the FYs in
         which NPP is complete (≥ 360 days) for that State and fuel, or the national fuel ratio when the State ratio
         is outside 0.80–1.05 (partial station coverage). Fuel mapping: COAL, LIGN → coal
         (NPP type THERMAL); GAS → gas (THER (GT)); DISL, OIL → diesel (THER (DG)); hydro and nuclear emit 0.
         Leakage: a FY's factors are published after the FY ends, so the panel uses the factor of FY−2 for
         operational features (published before the forecast date) and FY−1 for ex-post accounting (see b07).
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import entities as E  # noqa: E402
from refused.paths import INTERIM, RAW  # noqa: E402

SRC = os.path.join(RAW, "cea_co2_database", "Baseline_Carbon_Dioxide_Emission_Database_Version_22.0.xlsx")
FUEL = {"COAL": "coal", "LIGN": "coal", "GAS": "gas", "DISL": "diesel", "OIL": "diesel"}


def fy_of(d):
    y = d.year if d.month >= 4 else d.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def main():
    x = pd.read_excel(SRC, sheet_name="Data", header=0)
    x.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in x.columns]
    s = x[x.UNIT_NO == 0].copy()
    s["fuel"] = s["FUEL 1"].astype(str).str.strip().map(FUEL)
    s["entity"] = s.STATE.map(E.entity_id)
    print("stations", len(s), "| thermal without State entity:",
          s.loc[s.entity.isna() & s.fuel.notna(), "STATE"].value_counts().to_dict())
    rows = []
    for fy in sorted({m.group(1) for c in s.columns if (m := re.match(r"(\d{4}-\d{2}) Net Generation", c))}):
        g = s.dropna(subset=["entity", "fuel"]).assign(
            gen=pd.to_numeric(s[f"{fy} Net Generation GWh"], errors="coerce"),
            emis=pd.to_numeric(s[next(c for c in s.columns if c.startswith(f"{fy} Absolute Emissions"))], errors="coerce"))
        agg = g.groupby(["entity", "fuel"]).agg(emis_t=("emis", "sum"), netgen_gwh=("gen", "sum")).reset_index()
        agg["fy"] = fy
        rows.append(agg)
    co2 = pd.concat(rows, ignore_index=True)
    co2["ef_net"] = np.where(co2.netgen_gwh > 0, co2.emis_t / (co2.netgen_gwh * 1000), np.nan)
    n = pd.read_parquet(os.path.join(INTERIM, "npp_station_day.parquet"), columns=["date", "entity", "fuel", "act_gwh"])
    n = n.dropna(subset=["entity"])
    n["fy"] = n.date.map(fy_of)
    ng = n.groupby(["entity", "fuel", "fy"]).agg(npp_gross_gwh=("act_gwh", "sum"), npp_days=("date", "nunique"))
    out = co2.merge(ng.reset_index(), on=["entity", "fuel", "fy"], how="outer")
    out = out[out.fuel.isin(["coal", "gas", "diesel"])].sort_values(["entity", "fuel", "fy"]).reset_index(drop=True)
    # net/gross ratio from complete FYs only (NPP DGR2 is missing 13 Mar – 31 May 2020 and before 22 Mar 2018)
    full = out[(out.npp_days >= 360) & (out.netgen_gwh > 0) & (out.npp_gross_gwh > 0)]
    r_sf = full.groupby(["entity", "fuel"]).apply(lambda g: g.netgen_gwh.sum() / g.npp_gross_gwh.sum(),
                                                  include_groups=False).rename("r_state")
    r_f = full.groupby("fuel").apply(lambda g: g.netgen_gwh.sum() / g.npp_gross_gwh.sum(),
                                     include_groups=False).rename("r_national")
    out = out.join(r_sf, on=["entity", "fuel"]).join(r_f, on="fuel")
    ok = out.r_state.between(0.80, 1.05)
    out["net_gross_ratio"] = np.where(ok, out.r_state, out.r_national)
    out["ratio_source"] = np.where(ok, "state_complete_fys", "national_fuel")
    out["ef_gross"] = out.ef_net * out.net_gross_ratio
    out.to_parquet(os.path.join(INTERIM, "co2_ef_state_fuel_fy.parquet"), index=False)
    # national checks
    nat = out.groupby(["fy", "fuel"])[["emis_t", "netgen_gwh", "npp_gross_gwh"]].sum()
    nat["ef_net"] = nat.emis_t / nat.netgen_gwh / 1000
    nat["net_over_gross"] = nat.netgen_gwh / nat.npp_gross_gwh
    print(nat.round(3).to_string())
    print("net/gross ratios by fuel (complete FYs):", r_f.round(3).to_dict(), "| ratio source:",
          out.ratio_source.value_counts().to_dict())
    q = out[(out.fuel == "coal") & (out.netgen_gwh > 1000)]
    print("coal State ef_gross (t/MWh) quantiles:", q.ef_gross.quantile([0, .05, .5, .95, 1]).round(3).to_dict())
    print("State-fuel-FY cells with CEA emissions but no NPP generation:",
          int(((out.emis_t > 0) & ~(out.npp_gross_gwh > 0)).sum()),
          "| with NPP generation but no CEA emissions:", int(((out.npp_gross_gwh > 0) & ~(out.emis_t > 0)).sum()))


if __name__ == "__main__":
    main()
