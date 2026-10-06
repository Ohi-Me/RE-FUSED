"""Build step 3: parse CEA NPP Daily Generation Report sub-report 2 into station-day and State-day tables.

Inputs : 03_data/raw/npp_dgr2/<year>/*.xls
Outputs: 03_data/interim/npp_station_day.parquet  (date, region, state_raw, entity, sector, type, station, cap_mw,
                                                   prog_gwh, act_gwh, coal_days, outage_mw, units, units_out)
         03_data/interim/npp_state_day.parquet    (date, entity, per-type and total capacity, programme, actual,
                                                   outage; coal-weighted stock days; share of coal capacity with
                                                   stock below 7 days)
         03_data/interim/npp_parse_log.csv
Type mapping: THERMAL → coal (includes lignite stations as reported by CEA), THER (GT) → gas, THER (DG) → diesel,
NUCLEAR, HYDRO. Stations in states without a Grid-India control area (A&N Islands, Lakshadweep) and Bhutan imports are
kept in the station table with entity = None and excluded from State aggregates.
Validation: the sum of stations equals the report's STATE TOTAL rows on every date (logged). Station coal-stock days
outside 0–180 are treated as data-entry errors (set to missing before capacity weighting; count printed).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8 import entities as E  # noqa: E402
from refused8 import parse_npp as N  # noqa: E402
from refused8.paths import INTERIM, RAW  # noqa: E402

SRC = os.path.join(RAW, "npp_dgr2")
TYPE = {"THERMAL": "coal", "THER (GT)": "gas", "THER (DG)": "diesel", "NUCLEAR": "nuclear", "HYDRO": "hydro"}


def work(path):
    rel = os.path.relpath(path, SRC).replace("\\", "/")
    try:
        r = N.parse_dgr2(path)
    except Exception as e:
        return rel, None, f"error:{type(e).__name__}"
    if r is None or not r["stations"]:
        return rel, None, "empty"
    if not r["states"]:  # 29–31 Oct 2019: columns shifted and total labels absent, cannot be validated
        return rel, None, "no_state_totals"
    st = pd.DataFrame(r["stations"])
    tot = pd.DataFrame(r["states"])
    s = st.groupby("state_raw").act_gwh.sum()
    t = tot.set_index("state_raw").act_gwh
    chk = float((s.reindex(t.index).fillna(0) - t.fillna(0)).abs().max())
    if chk > 0.05:  # station rows do not reproduce the report's own State totals
        return rel, None, f"total_mismatch|{chk}"
    return rel, (st, tot), f"ok|{chk}"


def main():
    man = pd.read_csv(os.path.join(SRC, "MANIFEST.csv"))
    files = [os.path.join(SRC, p) for p in man.loc[man.status.astype(str) == "200", "path"] if os.path.exists(os.path.join(SRC, p))]
    print("files", len(files), flush=True)
    stations, log = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, (rel, res, status) in enumerate(ex.map(work, files, chunksize=6)):
            st_code, chk = (status.split("|") + [None])[:2]
            log.append(dict(file=rel, status=st_code, max_abs_station_vs_total=chk,
                            date=res[0].date.iloc[0] if res else None, n_stations=len(res[0]) if res else 0))
            if res:
                stations.append(res[0])
            if k % 500 == 0:
                print(k, rel, status, flush=True)
    st = pd.concat(stations, ignore_index=True)
    st["date"] = pd.to_datetime(st.date)
    st["entity"] = st.state_raw.map(E.entity_id)
    st["fuel"] = st.type.map(TYPE).fillna("other")
    st.to_parquet(os.path.join(INTERIM, "npp_station_day.parquet"), index=False)
    s = st.dropna(subset=["entity"])
    piv = s.pivot_table(index=["date", "entity"], columns="fuel", values=["cap_mw", "prog_gwh", "act_gwh", "outage_mw"],
                        aggfunc="sum")
    piv.columns = [f"{v}_{f}" for v, f in piv.columns]
    tot = s.groupby(["date", "entity"])[["cap_mw", "prog_gwh", "act_gwh", "outage_mw"]].sum()
    tot.columns = [c + "_total" for c in tot.columns]
    # coal-stock days outside 0–180 are data-entry errors in the report (e.g. tonnes typed into the days column)
    bad_coal = s.coal_days.notna() & ~s.coal_days.between(0, 180)
    print("station coal-days values outside 0–180 set to missing:", int(bad_coal.sum()))
    s = s.assign(coal_days=s.coal_days.mask(bad_coal))
    coal = s[(s.fuel == "coal") & s.coal_days.notna() & (s.cap_mw > 0)]
    cw = coal.groupby(["date", "entity"]).apply(
        lambda g: pd.Series({"coal_days_capw": np.average(g.coal_days, weights=g.cap_mw),
                             "coal_cap_share_lt7d": g.loc[g.coal_days < 7, "cap_mw"].sum() / g.cap_mw.sum(),
                             "coal_cap_reporting_mw": g.cap_mw.sum()}), include_groups=False)
    # isolated station typos: daily generation > 3× capacity × 24 h and > 2 GWh while the station's 31-day median
    # capacity factor is normal (< 1.2). Systematic bookings of a whole complex under one unit (e.g. Mukerian) do not
    # match because their rolling median is also high; station values are kept, the State-day is flagged.
    s = s.sort_values(["entity", "station", "date"])
    cf = np.where(s.cap_mw > 0, s.act_gwh * 1000 / 24 / s.cap_mw, np.nan)
    s = s.assign(cf=cf)
    s["cf_med"] = s.groupby(["entity", "station"]).cf.transform(lambda x: x.rolling(31, center=True, min_periods=8).median())
    typo = s[(s.cf > 3) & (s.act_gwh > 2) & (s.cf_med < 1.2)]
    ty = typo.groupby(["date", "entity"]).agg(station_typo_n=("station", "size"), station_typo_gwh=("act_gwh", "sum"))
    print("station typos:", len(typo), "State-days flagged:", len(ty))
    out = tot.join(piv).join(cw).join(ty).reset_index()
    out[["station_typo_n", "station_typo_gwh"]] = out[["station_typo_n", "station_typo_gwh"]].fillna(0)
    out.to_parquet(os.path.join(INTERIM, "npp_state_day.parquet"), index=False)
    log = pd.DataFrame(log)
    log.to_csv(os.path.join(INTERIM, "npp_parse_log.csv"), index=False)
    print("station rows", len(st), "state-day rows", len(out), "dates", out.date.nunique(), out.date.min(), out.date.max())
    print("unmapped states", sorted(set(st.loc[st.entity.isna(), "state_raw"].dropna())))
    print(log.status.value_counts().to_dict(), "max station-vs-total discrepancy (GWh):",
          pd.to_numeric(log.max_abs_station_vs_total, errors="coerce").max())


if __name__ == "__main__":
    main()
