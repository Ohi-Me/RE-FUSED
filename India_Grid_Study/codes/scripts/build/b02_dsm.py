"""Build step 2: parse Grid-India DSM rate files into block-level and daily bid-area price tables.

Inputs : data/raw/grid_india_dsm/<FY>/* (calendar-year ZIPs, daily CSV/XLSX; PX-rate ZIPs are listed but the DSM
         files already contain the exchange ACPs used for DSM, so PX files are not needed)
Outputs: data/interim/dsm_block.parquet  (date, block, bid_area, series, value ₹/MWh)
         data/interim/dsm_daily.parquet  (date, bid_area, regime, and daily statistics per series:
                                            mean, max, std, share of blocks at the ₹10,000/MWh cap)
         data/interim/dsm_parse_log.csv
If the same date appears in several files (e.g. a CY ZIP and a later daily file, or a "Revised" file), the file
listed later on the website (higher manifest row = later retrieval of a later upload) is kept and the choice logged.
Regime: R1 until the first R2 file, R2 afterwards (DSM Regulations 2022 in force from 5 Dec 2022; 2024 Regulations
from 16 Sep 2024 are marked separately as R3 by date).
"""
import datetime as dt
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import parse_dsm as D  # noqa: E402
from refused.paths import INTERIM, RAW  # noqa: E402

SRC = os.path.join(RAW, "grid_india_dsm")
REG2024 = pd.Timestamp("2024-09-16")


def members():
    man = pd.read_csv(os.path.join(SRC, "MANIFEST.csv"))
    man = man[man.status.astype(str) == "200"].reset_index(drop=True)
    for order, rel in enumerate(man.path):
        p = os.path.join(SRC, rel)
        if "PX Rates" in rel or not os.path.exists(p):
            continue
        if rel.lower().endswith(".zip"):
            z = zipfile.ZipFile(p)
            for n in z.namelist():
                if not n.endswith("/") and "PX Rates" not in n:
                    yield order, f"{rel}::{n}", n, z.read(n)
        else:
            yield order, rel, os.path.basename(rel), open(p, "rb").read()


def main():
    log, frames = [], []
    for order, src, name, raw in members():
        try:
            regime, rows = D.parse_file(raw, name)
        except Exception as e:
            regime, rows = f"error:{type(e).__name__}", []
        d = pd.DataFrame(rows)
        date = d.date.iloc[0] if len(d) else None
        log.append(dict(source=src, order=order, regime=regime, date=date, rows=len(d)))
        if len(d):
            d["source"], d["order"], d["regime"] = src, order, regime
            frames.append(d)
    log = pd.DataFrame(log)
    blk = pd.concat(frames, ignore_index=True)
    blk["date"] = pd.to_datetime(blk.date)
    pick = blk.groupby(["date", "source", "order"]).size().rename("n").reset_index().sort_values(["date", "order"])
    pick = pick.drop_duplicates("date", keep="last")
    log["chosen"] = log.source.isin(pick.source)
    blk = blk.merge(pick[["date", "source"]], on=["date", "source"])
    blk.loc[(blk.regime == "R2") & (blk.date >= REG2024), "regime"] = "R3"
    blk[["date", "block", "bid_area", "series", "value", "regime", "source"]].to_parquet(
        os.path.join(INTERIM, "dsm_block.parquet"), index=False)
    g = blk.groupby(["date", "bid_area", "series", "regime"]).value
    daily = pd.DataFrame({"mean": g.mean(), "max": g.max(), "std": g.std(), "n": g.size(),
                          "cap_share": g.apply(lambda s: float(np.mean(s >= 9999.5)))}).reset_index()
    daily.to_parquet(os.path.join(INTERIM, "dsm_daily.parquet"), index=False)
    log.to_csv(os.path.join(INTERIM, "dsm_parse_log.csv"), index=False)
    print("dates", daily.date.nunique(), daily.date.min(), daily.date.max())
    print(daily.groupby(["regime", "series"]).date.agg(["min", "max", "nunique"]).to_string())
    print(log.regime.value_counts().to_dict())


if __name__ == "__main__":
    main()
