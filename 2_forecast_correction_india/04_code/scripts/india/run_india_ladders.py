"""The adaptivity ladders on Indian data (official Grid-India, CEA and IMD data only).

Blocks (the same engine and settings as the earlier ladders, ladder.run_ladder):
  IH1  all-India hourly: demand, net demand, wind, solar; forecasts snaive168, lag48, lgbm, chronos, nhits,
       patchtst, tide, dlinear; 3 seeds (1 in development)
  IH2  all-India hourly data-budget ladder on the LightGBM forecast: gate-fitting fractions 0.1, 0.25, 0.5 of the
       validation days, and out-of-fold training errors plus validation
  ID1  34 State control areas, daily, targets T1-T4, simple forecasts (seasonal naive 7 days, shortest published lag);
       2 seeds
  ID2  34 State control areas, daily, targets T1-T4, the India study's own day-ahead forecasts (XGBoost, Chronos-2,
       N-HiTS, TiDE, and the final combination); 2 seeds
  ID3  34 State control areas, daily data-budget ladder on T2 with the seasonal-naive forecast
Modes:
  dev      hourly blocks on the development split (never reads data from 1 March 2026 on); daily blocks skipped
  confirm  hourly blocks on the pre-registered split (requires the India pre-registration lock)
  explore  daily blocks (their periods were opened by the India study, so they are exploratory)
Results: 06_results/india/<mode>/<block>/<key>.parquet|json, never overwritten.
Usage:   python 04_code/scripts/india/run_india_ladders.py --mode dev   (run on the H100)
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

os.environ["REFUSED_GATE_STUDY"] = "india"   # the Indian study uses the top-level folders

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import data, ladder  # noqa: E402
from refused_gate.paths import PROC, RES  # noqa: E402

LEARNED_H = ["lgbm", "chronos", "nhits", "patchtst", "tide", "dlinear"]


def hourly_blocks(mode, seeds):
    ds = data.india_hourly(mode)
    df, cols = ds["df"], dict(ds["baselines"])
    lb = os.path.join(PROC, f"india_hourly_learned_baselines_{mode}_s0.parquet")
    if os.path.exists(lb):
        L = pd.read_parquet(lb)
        df = df.merge(L, on=["sid", "ts"], how="left")
        cols.update({c[2:]: c for c in L.columns if c.startswith("B_")})
    for b, col in cols.items():
        fr = df.assign(y_t=df.y, B=df[col])
        if b in LEARNED_H:
            fr = fr[fr[col].notna() | (fr.split != "train")]
        for s in range(seeds):
            yield "IH1", f"{b}_s{s}", fr, ds["feats"], ds["state_keys"], ladder.LadderConfig(n_bag=3, delay_days=2), s
    if "B_lgbm" in df:
        fr = df.assign(y_t=df.y, B=df.B_lgbm)
        fr = fr[fr.B_lgbm.notna() | (fr.split != "train")]
        for s in range(seeds):
            for frac in (0.1, 0.25, 0.5):
                yield "IH2", f"val_frac{frac}_s{s}", fr, ds["feats"], ds["state_keys"], \
                    ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit_fraction=frac, do_time=False), s
            yield "IH2", f"oof_train_val_s{s}", fr, ds["feats"], ds["state_keys"], \
                ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit="oof_train+val", do_time=False), s


def daily_blocks(seeds):
    for tid in data.INDIA_DAILY_TARGETS:
        for design, blk in (("long", "ID1"), ("learned", "ID2")):
            ds = data.india_state_daily(tid, design)
            for b, col in ds["baselines"].items():
                if design == "learned" and not col.startswith("B_"):
                    continue
                fr = ds["df"].assign(y_t=ds["df"].y, B=ds["df"][col])
                for s in range(seeds):
                    yield blk, f"{tid}_{b}_s{s}", fr, ds["feats"], ds["state_keys"], \
                        ladder.LadderConfig(n_bag=3, delay_days=ds["delay_days"]), s
    ds = data.india_state_daily("T2", "long")
    fr = ds["df"].assign(y_t=ds["df"].y, B=ds["df"].y_lag7)
    for s in range(seeds):
        for frac in (0.1, 0.25):
            yield "ID3", f"T2_val_frac{frac}_s{s}", fr, ds["feats"], ds["state_keys"], \
                ladder.LadderConfig(n_bag=3, delay_days=ds["delay_days"], gate_fit_fraction=frac, do_time=False), s
        yield "ID3", f"T2_oof_train_val_s{s}", fr, ds["feats"], ds["state_keys"], \
            ladder.LadderConfig(n_bag=3, delay_days=ds["delay_days"], gate_fit="oof_train+val", do_time=False), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev", choices=["dev", "confirm", "explore"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.mode == "dev":
        gen = hourly_blocks("dev", 1)
    elif a.mode == "confirm":
        gen = hourly_blocks("confirm", 3)
    else:
        gen = daily_blocks(2)
    only = set(a.only.split(",")) if a.only else None
    t0 = time.time()
    for blk, key, fr, feats, keys, cfg, s in gen:
        if only and blk not in only:
            continue
        out_dir = os.path.join(RES, "india", a.mode, blk)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, key + ".parquet")
        if os.path.exists(dst):
            continue
        res = ladder.run_ladder(fr, feats, keys, cfg, seed=s)
        if res is None:
            print("skip (too few rows)", blk, key, flush=True)
            continue
        out, diag = res
        out.to_parquet(dst, index=False)
        diag["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}
        json.dump(diag, open(dst.replace(".parquet", ".json"), "w"), indent=1, default=str)
        print(f"{blk} {key:32s} done [{diag['seconds']}s, total {time.time()-t0:.0f}s]", flush=True)
    print("DONE India ladders", a.mode, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
