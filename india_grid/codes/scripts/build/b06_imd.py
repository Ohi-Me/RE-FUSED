"""Build step 6: State-day weather from IMD Pune gridded daily data (Govt. of India, India Meteorological Department).

Inputs : 03_data/raw/imd_gridded/{maxtemp,mintemp,rain}/<var>_<year>.grd, 2017–2025
           temperature: 1.0° grid, 31 × 31 (lat 7.5–37.5 N, lon 67.5–97.5 E), missing = 99.9
           rainfall   : 0.25° grid, 129 × 135 (lat 6.5–38.5 N, lon 66.5–100.0 E), missing = −999
           binary float32 little-endian, one field per day, C-order (day, lat, lon); orientation verified on land/sea
           cells and known climatology (Delhi Jan vs Jun Tmax; Mumbai July rainfall).
Outputs: 03_data/interim/imd_state_day.parquet  (date, entity, tmax_c, tmin_c, tmean_c, rain_mm, cdd24, hdd18,
                                                 n_temp_cells, n_rain_cells)
Method : State weather is the mean over the State's main load centres (LOAD_CENTRES: capital and largest demand
         cities, approximate city-centre coordinates). Temperature uses the nearest valid 1° cell (coastal cities
         can fall in a sea cell, in which case the nearest valid land cell within 1.5° is used); rainfall is the
         mean of valid 0.25° cells within ±0.25° of the centre. Degree days: cdd24 = max(tmean − 24, 0),
         hdd18 = max(18 − tmean, 0). This is a documented approximation (no boundary-weighted averaging), chosen
         because it needs no non-government boundary files. IMD gridded data end on 2025-12-31, so weather cannot be
         a required input for 2026 dates.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused.paths import INTERIM, RAW  # noqa: E402

SRC = os.path.join(RAW, "imd_gridded")
YEARS = range(2017, 2026)

LOAD_CENTRES = {  # entity: [(lat, lon), ...]
    "PB": [(30.90, 75.85), (31.63, 74.87), (31.33, 75.58)],
    "HR": [(28.46, 77.03), (28.41, 77.32), (29.39, 76.97)],
    "RJ": [(26.91, 75.79), (26.24, 73.02), (25.18, 75.83)],
    "DL": [(28.61, 77.21)],
    "UP": [(26.85, 80.95), (26.45, 80.33), (28.67, 77.45), (25.32, 82.97)],
    "UK": [(30.32, 78.03), (29.95, 78.16)],
    "HP": [(31.10, 77.17), (30.96, 76.79)],
    "JK": [(34.08, 74.80), (32.73, 74.86)],
    "CH": [(30.73, 76.78)],
    "CG": [(21.25, 81.63), (21.19, 81.38), (22.36, 82.68)],
    "GJ": [(23.03, 72.58), (21.17, 72.83), (22.31, 73.18), (22.30, 70.80)],
    "MP": [(23.26, 77.41), (22.72, 75.86), (23.18, 79.99)],
    "MH": [(19.08, 72.88), (18.52, 73.86), (21.15, 79.09), (20.00, 73.79)],
    "GA": [(15.49, 73.83)],
    "DNHDD": [(20.27, 73.02), (20.40, 72.83)],
    "AP": [(17.69, 83.22), (16.51, 80.65), (13.63, 79.42)],
    "TG": [(17.39, 78.49)],
    "KA": [(12.97, 77.59), (12.30, 76.64), (15.36, 75.12)],
    "KL": [(8.52, 76.94), (9.93, 76.27), (11.26, 75.78)],
    "TN": [(13.08, 80.27), (11.02, 76.96), (9.93, 78.12)],
    "PY": [(11.94, 79.81)],
    "BR": [(25.59, 85.14), (24.79, 85.00), (26.12, 85.39)],
    "DVC": [(23.80, 86.43), (23.52, 87.31)],
    "JH": [(23.34, 85.31), (22.80, 86.20)],
    "OD": [(20.30, 85.82), (20.46, 85.88), (22.26, 84.85)],
    "WB": [(22.57, 88.36), (23.68, 86.98), (26.73, 88.40)],
    "SK": [(27.33, 88.61)],
    "AR": [(27.08, 93.61)],
    "AS": [(26.14, 91.74), (27.47, 94.91)],
    "MN": [(24.82, 93.94)],
    "ML": [(25.58, 91.89)],
    "MZ": [(23.73, 92.72)],
    "NL": [(25.67, 94.11), (25.91, 93.73)],
    "TR": [(23.83, 91.28)],
}


def load(var, year):
    a = np.fromfile(os.path.join(SRC, var, f"{var}_{year}.grd"), dtype="<f4")
    shape = (129, 135) if var == "rain" else (31, 31)
    a = a.reshape(-1, *shape).astype(float)
    a[(a >= 99.8) & (a <= 100.0) if var != "rain" else (a < -900)] = np.nan
    return a


def temp_cell(field_mean, lat, lon):
    """Index of nearest 1° cell with valid data (land), searching within 1.5°."""
    la, lo = np.arange(7.5, 38.5, 1.0), np.arange(67.5, 98.5, 1.0)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    d = np.hypot(LA - lat, LO - lon)
    d[np.isnan(field_mean)] = np.inf
    i = np.unravel_index(np.argmin(d), d.shape)
    return i if d[i] <= 1.5 else None


def rain_cells(lat, lon):
    la, lo = np.arange(6.5, 38.75, 0.25), np.arange(66.5, 100.25, 0.25)
    ii = np.where(np.abs(la - lat) <= 0.25 + 1e-9)[0]
    jj = np.where(np.abs(lo - lon) <= 0.25 + 1e-9)[0]
    return ii, jj


def main():
    out = []
    for y in YEARS:
        tx, tn, rr = load("maxtemp", y), load("mintemp", y), load("rain", y)
        dates = pd.date_range(f"{y}-01-01", periods=tx.shape[0])
        assert tn.shape[0] == rr.shape[0] == len(dates)
        valid_t = np.nanmean(tx, axis=0)
        for ent, pts in LOAD_CENTRES.items():
            txs, tns, rrs = [], [], []
            for lat, lon in pts:
                c = temp_cell(valid_t, lat, lon)
                if c is not None:
                    txs.append(tx[:, c[0], c[1]])
                    tns.append(tn[:, c[0], c[1]])
                ii, jj = rain_cells(lat, lon)
                block = rr[:, ii][:, :, jj].reshape(len(dates), -1)
                if np.isfinite(block).any():
                    rrs.append(np.nanmean(block, axis=1))
            with np.errstate(all="ignore"):
                df = pd.DataFrame({"date": dates, "entity": ent,
                                   "tmax_c": np.nanmean(txs, axis=0) if txs else np.nan,
                                   "tmin_c": np.nanmean(tns, axis=0) if tns else np.nan,
                                   "rain_mm": np.nanmean(rrs, axis=0) if rrs else np.nan,
                                   "n_temp_cells": len(txs), "n_rain_cells": len(rrs)})
            out.append(df)
        print(y, "done", flush=True)
    w = pd.concat(out, ignore_index=True)
    w["tmean_c"] = (w.tmax_c + w.tmin_c) / 2
    w["cdd24"] = (w.tmean_c - 24).clip(lower=0)
    w["hdd18"] = (18 - w.tmean_c).clip(lower=0)
    w.to_parquet(os.path.join(INTERIM, "imd_state_day.parquet"), index=False)
    print("rows", len(w), w.date.min(), w.date.max(), "entities", w.entity.nunique())
    print("missing share:", w[["tmax_c", "tmin_c", "rain_mm"]].isna().mean().round(4).to_dict())
    print("cells used (min over entities):", w.groupby("entity")[["n_temp_cells", "n_rain_cells"]].min().min().to_dict())
    clim = w.assign(m=w.date.dt.month).groupby(["entity", "m"])[["tmax_c", "rain_mm"]].mean().unstack("m")
    print(clim.loc[["DL", "MH", "TN", "AS", "RJ"], [("tmax_c", 1), ("tmax_c", 5), ("rain_mm", 7), ("rain_mm", 12)]].round(1))


if __name__ == "__main__":
    main()
