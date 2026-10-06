"""Build step 10: hourly all-India day-ahead and real-time market prices from Grid-India's DSM rate files.

The DSM files published by Grid-India carry, for every 15-minute block, the area clearing prices of the day-ahead
market (DAM) and the real-time market (RTM) by bid area, and the national unconstrained market clearing price (bid
area "MCP"). This step takes the MCP of both markets and averages the four blocks of each hour.

Input : data/interim/dsm_block.parquet (build step 2)
Output: data/processed/refused_allindia_hourly_prices.parquet  (ts, dam_mcp_rs_mwh, rtm_mcp_rs_mwh, n_blocks)
Usage : python codes/scripts/build/b10_hourly_prices.py   (run on the H100)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused.paths import INTERIM, PROC  # noqa: E402


def main():
    d = pd.read_parquet(os.path.join(INTERIM, "dsm_block.parquet"))
    d = d[(d.bid_area == "MCP") & d.series.isin(["dam_acp", "rtm_acp"])].copy()
    # block 1..96 (or 0..95) -> start time of the block
    first = int(d.block.min())
    d["ts"] = pd.to_datetime(d.date) + pd.to_timedelta((d.block.astype(int) - first) * 15, unit="m")
    d["hour"] = d.ts.dt.floor("h")
    p = d.pivot_table(index=["hour", "ts"], columns="series", values="value", aggfunc="mean").reset_index()
    h = p.groupby("hour").agg(dam_mcp_rs_mwh=("dam_acp", "mean"), rtm_mcp_rs_mwh=("rtm_acp", "mean"),
                              n_blocks=("ts", "size")).reset_index().rename(columns={"hour": "ts"})
    h.to_parquet(os.path.join(PROC, "refused_allindia_hourly_prices.parquet"), index=False)
    print(f"hourly prices {h.ts.min()} to {h.ts.max()}: {len(h)} hours, blocks per hour min {h.n_blocks.min()}")


if __name__ == "__main__":
    main()
