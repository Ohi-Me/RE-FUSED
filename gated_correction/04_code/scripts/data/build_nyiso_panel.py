"""Build the NYISO zonal hourly panel used by RE-FUSED (development AND blinded confirmatory rows).

Output 03_data/processed/nyiso_load_panel.parquet with one row per (zone, local hour-beginning):
  load       hourly integrated actual load, MW                      (palIntegrated)
  iso_fc     NYISO day-ahead load forecast: file issued on day D-1, row for hour on day D   (isolf)
  da_lbmp    day-ahead zonal LBMP, $/MWh (hour-beginning)             (damlbmp_zone)
  rt_lbmp    mean of 5-min real-time zonal LBMP in the hour, $/MWh    (realtime_zone; 5-min stamps are
             interval-ending, mapped to hour-beginning by (ts - 5 min).floor('h'))
  da_spin    day-ahead 10-minute spinning reserve price posted for the zone, $/MWh   (damasp)
Timestamps are local prevailing time; on the autumn DST change the repeated hour is averaged, the missing spring
hour is absent. This script reads 2026 raw files but prints only row counts (blinding: no statistics).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate.paths import REFUSED5_DATA, PROC, RAW  # noqa: E402

ZMAP = {"Capitl": "CAPITL", "Centrl": "CENTRL", "Dunwod": "DUNWOD", "Genese": "GENESE", "Hud Vl": "HUD VL",
        "Longil": "LONGIL", "Mhk Vl": "MHK VL", "Millwd": "MILLWD", "N.Y.C.": "N.Y.C.", "North": "NORTH",
        "West": "WEST"}
ZONES = list(ZMAP.values())


def monthly(product, roots):
    files = []
    for r in roots:
        files += glob.glob(os.path.join(r, product, "*.parquet"))
    by_month = {}
    for f in sorted(files):                       # later roots override earlier ones for the same month
        by_month[os.path.basename(f)] = f
    return [by_month[k] for k in sorted(by_month)]


def load_actual():
    frames = []
    for f in monthly("palIntegrated", [os.path.join(RAW, "nyiso")]):
        d = pd.read_parquet(f, columns=["Time Stamp", "Name", "Integrated Load"])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["ts"] = pd.to_datetime(d["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
    d = d[d.Name.isin(ZONES)].rename(columns={"Name": "zone", "Integrated Load": "load"})
    return d.groupby(["zone", "ts"], as_index=False)["load"].mean()


def load_forecast():
    frames = []
    for f in monthly("isolf", [os.path.join(RAW, "nyiso")]):
        d = pd.read_parquet(f)
        d["ts"] = pd.to_datetime(d["Time Stamp"], format="%m/%d/%Y %H:%M")
        d["issue"] = pd.to_datetime(d["issue_date"], format="%Y%m%d")
        # day-ahead: target calendar day == issue day + 1
        d = d[d.ts.dt.normalize() == d.issue + pd.Timedelta(days=1)]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    long = d.melt(id_vars=["ts"], value_vars=list(ZMAP), var_name="zone", value_name="iso_fc")
    long["zone"] = long.zone.map(ZMAP)
    return long.groupby(["zone", "ts"], as_index=False)["iso_fc"].mean()


def load_da_lbmp():
    roots = [os.path.join(REFUSED5_DATA, "nyiso"), os.path.join(RAW, "nyiso")]
    frames = []
    for f in monthly("damlbmp_zone", roots):
        d = pd.read_parquet(f)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["Time Stamp", "Name", "LBMP ($/MWHr)"]])
    d = pd.concat(frames, ignore_index=True)
    d["ts"] = pd.to_datetime(d["Time Stamp"], format="%m/%d/%Y %H:%M")
    d = d[d.Name.isin(ZONES)].rename(columns={"Name": "zone", "LBMP ($/MWHr)": "da_lbmp"})
    return d.groupby(["zone", "ts"], as_index=False)["da_lbmp"].mean()


def load_rt_lbmp():
    roots = [os.path.join(REFUSED5_DATA, "nyiso"), os.path.join(RAW, "nyiso")]
    out = []
    for f in monthly("realtime_zone", roots):
        d = pd.read_parquet(f)
        d.columns = [c.strip() for c in d.columns]
        d = d[d.Name.isin(ZONES)]
        ts = pd.to_datetime(d["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
        d = pd.DataFrame({"zone": d.Name.to_numpy(), "ts": (ts - pd.Timedelta(minutes=5)).dt.floor("h"),
                          "rt_lbmp": d["LBMP ($/MWHr)"].astype(float).to_numpy()})
        out.append(d.groupby(["zone", "ts"], as_index=False)["rt_lbmp"].mean())
    return pd.concat(out, ignore_index=True).groupby(["zone", "ts"], as_index=False)["rt_lbmp"].mean()


def load_da_spin():
    frames = []
    for f in monthly("damasp", [os.path.join(RAW, "nyiso")]):
        d = pd.read_parquet(f)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["Time Stamp", "Name", "10 Min Spinning Reserve ($/MWHr)"]])
    d = pd.concat(frames, ignore_index=True)
    d["ts"] = pd.to_datetime(d["Time Stamp"], format="%m/%d/%Y %H:%M")
    d = d.rename(columns={"Name": "zone", "10 Min Spinning Reserve ($/MWHr)": "da_spin"})
    d = d[d.zone.isin(ZONES)]
    return d.groupby(["zone", "ts"], as_index=False)["da_spin"].mean()


def main():
    parts = {"load": load_actual(), "iso_fc": load_forecast(), "da_lbmp": load_da_lbmp(),
             "rt_lbmp": load_rt_lbmp(), "da_spin": load_da_spin()}
    for k, v in parts.items():
        print(f"{k:8s} rows={len(v):8d} zones={v.zone.nunique()}", flush=True)
    panel = parts["load"]
    for k in ["iso_fc", "da_lbmp", "rt_lbmp", "da_spin"]:
        panel = panel.merge(parts[k], on=["zone", "ts"], how="left")
    panel = panel.sort_values(["zone", "ts"]).reset_index(drop=True)
    panel["zone"] = panel.zone.astype("category")
    for c in ["load", "iso_fc", "da_lbmp", "rt_lbmp", "da_spin"]:
        panel[c] = panel[c].astype("float32")
    dst = os.path.join(PROC, "nyiso_load_panel.parquet")
    panel.to_parquet(dst, index=False)
    dev = panel[panel.ts < "2026-01-01"]
    print("panel rows:", len(panel), "| development rows (<2026):", len(dev),
          "| blinded rows (>=2026):", len(panel) - len(dev))
    print("development missing fraction per column:")
    print(dev[["load", "iso_fc", "da_lbmp", "rt_lbmp", "da_spin"]].isna().mean().round(4).to_string())
    print("wrote", dst)


if __name__ == "__main__":
    main()
