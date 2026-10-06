"""Build the external confirmatory panels (OPSD / ENTSO-E and UCI electricity) WITHOUT inspecting outcomes.

Blinding rule: this script parses the raw files, applies eligibility rules fixed in advance (below), and writes
processed parquet files. It prints only unit counts, time spans and missing-data fractions — no forecast errors,
no skill, no summary statistics of the target.

OPSD eligibility (fixed before loading): a country / control area / bidding zone column pair
  <X>_load_actual_entsoe_transparency and <X>_load_forecast_entsoe_transparency
with both series >= 95% non-missing over 2015-01-01 .. 2019-12-31 (UTC hours). Gaps <= 3 h interpolated.
Splits (confirmatory): train 2015-2017, validation 2018, test 2019 (primary); 2020-01-01 .. 2020-09-30 second test
period (pandemic shock; labelled separately).

UCI eligibility: clients with no all-zero day between 2012-01-01 and 2014-12-31 after resampling 15-min kW to
hourly mean kW. Splits: train 2012, validation 2013, test 2014.
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate.paths import PROC, RAW  # noqa: E402


def build_opsd():
    f = os.path.join(RAW, "public", "opsd_time_series_60min_singleindex_2020-10-06.csv")
    head = pd.read_csv(f, nrows=0).columns
    act = [c for c in head if c.endswith("_load_actual_entsoe_transparency")]
    units = [c[: -len("_load_actual_entsoe_transparency")] for c in act]
    units = [u for u in units if f"{u}_load_forecast_entsoe_transparency" in head]
    cols = ["utc_timestamp"] + [f"{u}_load_actual_entsoe_transparency" for u in units] + \
           [f"{u}_load_forecast_entsoe_transparency" for u in units]
    d = pd.read_csv(f, usecols=cols, parse_dates=["utc_timestamp"])
    d["utc_timestamp"] = d.utc_timestamp.dt.tz_localize(None)
    d = d[(d.utc_timestamp >= "2015-01-01") & (d.utc_timestamp < "2020-10-01")]
    keep, rows = [], []
    window = (d.utc_timestamp < "2020-01-01")
    for u in units:
        a, fc = f"{u}_load_actual_entsoe_transparency", f"{u}_load_forecast_entsoe_transparency"
        cov_a, cov_f = d.loc[window, a].notna().mean(), d.loc[window, fc].notna().mean()
        if cov_a >= 0.95 and cov_f >= 0.95:
            keep.append(u)
            x = pd.DataFrame({"sid": u, "ts": d.utc_timestamp.to_numpy(),
                              "y": d[a].interpolate(limit=3).to_numpy(dtype=np.float32),
                              "tso_fc": d[fc].interpolate(limit=3).to_numpy(dtype=np.float32)})
            rows.append(x)
        print(f"  {u:14s} actual coverage {cov_a:.3f} forecast coverage {cov_f:.3f} -> {'keep' if u in keep else 'drop'}")
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(os.path.join(PROC, "opsd_load_panel.parquet"), index=False)
    print(f"OPSD: {len(keep)} eligible units of {len(units)}; rows {len(panel)}; span {panel.ts.min()} .. {panel.ts.max()}")


def build_ucl():
    z = zipfile.ZipFile(os.path.join(RAW, "public", "uci_electricityloaddiagrams20112014.zip"))
    name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
    d = pd.read_csv(io.BytesIO(z.read(name)), sep=";", decimal=",", index_col=0, parse_dates=True, low_memory=False)
    d = d[(d.index >= "2012-01-01") & (d.index < "2015-01-01")].astype(np.float32)
    # 15-min stamps mark the end of the interval: shift back 15 min before hourly averaging
    d.index = d.index - pd.Timedelta(minutes=15)
    h = d.resample("h").mean()
    daily_max = h.resample("D").max()
    eligible = [c for c in h.columns if (daily_max[c] > 0).all()]
    long = h[eligible].rename_axis("ts").reset_index().melt(id_vars="ts", var_name="sid", value_name="y")
    long["y"] = long.y.astype(np.float32)
    long.to_parquet(os.path.join(PROC, "uci_load_panel.parquet"), index=False)
    print(f"UCI: {len(eligible)} eligible clients of {h.shape[1]}; rows {len(long)}; span {long.ts.min()} .. {long.ts.max()}")


if __name__ == "__main__":
    build_opsd()
    build_ucl()
