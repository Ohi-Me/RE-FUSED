"""Out-of-sample learned day-ahead forecasts for the all-India hourly series (demand, net demand, wind, solar).

Same models, settings and information set as for NYISO (make_learned_baselines.py, whose model functions are reused
unchanged): for a target day D every model uses data up to the end of day D-2, which is what is published when a
day-ahead forecast is made on D-1 (the Daily PSP report for a day appears the next day). Learned models are fitted
twice so that their predictions are out of sample wherever they are used:
  development:  fit A train < 2025-03-01 -> Mar-May 2025;  fit B train < 2025-06-01 -> Jun 2025 - Feb 2026
  confirmatory: fit A train < 2025-05-01 -> May-Aug 2025;  fit B train < 2025-09-01 -> Sep 2025 - 13 Sep 2026
Development never reads data from 1 March 2026 onwards; confirmatory mode requires the India pre-registration lock.
Output: 03_data/processed/india_hourly_learned_baselines_<mode>_s<seed>.parquet (sid, ts, B_<model>).
Usage:  python 04_code/scripts/dev/make_learned_baselines_india.py --mode dev   (run on the H100)
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

os.environ["REFUSED_GATE_STUDY"] = "india"   # the Indian study uses the top-level folders

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from refused_gate import guard  # noqa: E402
from refused_gate.data import INDIA_HOURLY_SERIES, INDIA_HOURLY_TEST_START  # noqa: E402
from refused_gate.paths import INDIA_PROC, PROC  # noqa: E402
import make_learned_baselines as M  # noqa: E402


def panel(mode):
    p = pd.read_parquet(os.path.join(INDIA_PROC, "refused_allindia_hourly.parquet"))
    if mode == "confirm":
        guard.require_india_confirmatory()
    else:
        p = p[p.ts < INDIA_HOURLY_TEST_START]
    out = []
    for sid, col in INDIA_HOURLY_SERIES.items():
        g = p[["ts", col]].rename(columns={"ts": "ds", col: "y"}).set_index("ds")
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h")).rename_axis("ds")
        y = g.y.interpolate(limit=3)                         # short telemetry gaps
        y = y.fillna(y.shift(168)).ffill().bfill()           # whole missing days: same hour one week earlier
        out.append(pd.DataFrame({"unique_id": sid, "ds": g.index, "y": y.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def limit_threads(n=6):
    """The cluster stops a job that uses more cores than it requested (8); PyTorch and LightGBM otherwise start one
    thread per core of the node. Added on 6 Oct 2026 after the first confirmatory attempt was stopped (deviation D1)."""
    import pyarrow
    import torch
    pyarrow.set_cpu_count(n)
    pyarrow.set_io_thread_count(1)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def main():
    limit_threads(int(os.environ.get("REFUSED_TORCH_THREADS", "4")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev", choices=["dev", "confirm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="lgbm,chronos,neural")
    a = ap.parse_args()
    df = panel(a.mode)
    if a.mode == "dev":
        fits = [("2025-03-01", "2025-03-01", "2025-05-31"), ("2025-06-01", "2025-06-01", "2026-02-28")]
        span = ("2025-03-01", "2026-02-28")
    else:
        fits = [("2025-05-01", "2025-05-01", "2025-08-31"), ("2025-09-01", "2025-09-01", "2026-09-13")]
        span = ("2025-05-01", "2026-09-13")
    parts, t0, models = [], time.time(), a.models.split(",")
    for train_end, ps, pe in fits:
        if "lgbm" in models:
            parts.append(("lgbm", M.lgbm(df, train_end, ps, pe, a.seed)))
            print(f"  lgbm fit<{train_end} done [{time.time()-t0:.0f}s]", flush=True)
        if "neural" in models:
            parts.append(("neural", M.neural(df, train_end, ps, pe, a.seed)))
            print(f"  neural fit<{train_end} done [{time.time()-t0:.0f}s]", flush=True)
    if "chronos" in models:
        parts.append(("chronos", M.chronos(df, *span)))
        print(f"  chronos done [{time.time()-t0:.0f}s]", flush=True)
    merged = None
    for name in dict.fromkeys(n for n, _ in parts):
        block = pd.concat([p for n, p in parts if n == name], ignore_index=True)
        block = block.groupby(["sid", "ts"], as_index=False).mean(numeric_only=True)
        merged = block if merged is None else merged.merge(block, on=["sid", "ts"], how="outer")
    if a.mode == "dev":
        merged = merged[merged.ts < INDIA_HOURLY_TEST_START]
    dst = os.path.join(PROC, f"india_hourly_learned_baselines_{a.mode}_s{a.seed}.parquet")
    merged.to_parquet(dst, index=False)
    print("wrote", dst, merged.shape, list(merged.columns), f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
