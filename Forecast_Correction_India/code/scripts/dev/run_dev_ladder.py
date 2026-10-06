"""Development runner for ladder experiments (resumable; one result per job key).

Usage: py -3.10 run_dev_ladder.py <experiment> [--seeds 3]
Experiments:
  nyiso_baselines   NYISO load, baselines {iso, snaive168, lag48}, gate fit on validation
  nyiso_budget      NYISO load, ISO baseline, gate-fitting budget ladder (fractions of validation; OOF train)
  india             India daily, 4 baselines x h in {1, 3}
  ett               ETTh1/ETTh2, {persistence, snaive24} x h in {1, 24}
Outputs: results/dev/<experiment>/<key>.parquet (per-row test outputs) and <key>.json (diagnostics).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import data, ladder  # noqa: E402
from refused_gate.paths import RES  # noqa: E402


def jobs_nyiso_baselines(seeds):
    ds = data.nyiso_load("dev")
    for b, col in ds["baselines"].items():
        fr = ds["df"].assign(y_t=ds["df"].y, B=ds["df"][col])
        for s in seeds:
            yield f"{b}_s{s}", fr, ds["feats"], ["hour_block", "hour"], ladder.LadderConfig(n_bag=3, delay_days=2), s


def jobs_nyiso_budget(seeds):
    ds = data.nyiso_load("dev")
    fr = ds["df"].assign(y_t=ds["df"].y, B=ds["df"]["iso_fc"])
    for s in seeds:
        for frac in (0.05, 0.1, 0.25, 0.5):
            cfg = ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit="val", gate_fit_fraction=frac, do_time=False)
            yield f"val_frac{frac}_s{s}", fr, ds["feats"], ["hour_block", "hour"], cfg, s
        for gf in ("oof_train", "oof_train+val"):
            cfg = ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit=gf, do_time=False)
            yield f"{gf.replace('+', '_')}_s{s}", fr, ds["feats"], ["hour_block", "hour"], cfg, s


def _vol_tercile(x, col):
    tr = x[x.split == "train"]
    lo, hi = np.nanquantile(tr[col], [1 / 3, 2 / 3])
    return np.digitize(x[col].to_numpy(), [lo, hi])


def jobs_india(seeds):
    ds = data.india_daily("dev")
    for b, spec in ds["baselines"].items():
        for h in (1, 3):
            fr = data.make_baseline(ds["df"], h, spec, season=7)
            if fr is None:
                continue
            fr["vol_tercile"] = _vol_tercile(fr, "rs7")
            for s in seeds:
                cfg = ladder.LadderConfig(n_bag=3, delay_days=h, windows_days=(30, 90, 180), part_eval_rows=8000)
                yield f"{b}_h{h}_s{s}", fr, ds["feats"], ["vol_tercile"], cfg, s


def jobs_ett(seeds):
    for name in ("ETTh1", "ETTh2"):
        ds = data.ett(name, "dev")
        for b, spec in ds["baselines"].items():
            for h in (1, 24):
                fr = data.make_baseline(ds["df"], h, spec, season=24)
                if fr is None:
                    continue
                for s in seeds:
                    cfg = ladder.LadderConfig(n_bag=3, delay_days=1 + int(np.ceil(h / 24)), windows_days=(14, 30, 60))
                    yield f"{name}_{b}_h{h}_s{s}", fr, ds["feats"], ["hour_block"], cfg, s


MODES = ("contiguous", "interleaved")


def jobs_dev2_nyiso(seeds):
    """Dev-2: all baselines incl. learned ones, PART block designs and PART-time in one pass."""
    ds = data.nyiso_load("dev")
    df = ds["df"]
    lb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Other_Countries", "data", "processed",
                      "nyiso_learned_baselines_dev_s0.parquet")
    cols = dict(ds["baselines"])
    if os.path.exists(lb):
        L = pd.read_parquet(lb)
        df = df.merge(L, left_on=["sid", "ts"], right_on=["sid", "ts"], how="left")
        cols.update({c[2:]: c for c in L.columns if c.startswith("B_")})
    for b, col in cols.items():
        fr = df.assign(y_t=df.y, B=df[col])
        if b not in ds["baselines"]:
            fr = fr[fr.ts >= "2022-01-01"]            # learned baselines are out-of-sample from 2022 only
        for s in seeds:
            yield f"{b}_s{s}", fr, ds["feats"], ["hour_block", "hour"], \
                ladder.LadderConfig(n_bag=3, delay_days=2, part_block_mode=MODES), s
    # year-block PART v1 runs on OOF+val were dropped after DV5 (PART v1 superseded; v2 evaluated offline)


def jobs_dev2_india(seeds):
    for key, fr, feats, keys, cfg, s in jobs_india(seeds):
        cfg.part_block_mode = MODES
        yield key, fr, feats, keys, cfg, s


def jobs_dev2_ett(seeds):
    for key, fr, feats, keys, cfg, s in jobs_ett(seeds):
        cfg.part_block_mode = MODES
        yield key, fr, feats, keys, cfg, s


EXPERIMENTS = {"nyiso_baselines": jobs_nyiso_baselines, "nyiso_budget": jobs_nyiso_budget,
               "india": jobs_india, "ett": jobs_ett,
               "dev2_nyiso": jobs_dev2_nyiso, "dev2_india": jobs_dev2_india, "dev2_ett": jobs_dev2_ett}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out_dir = os.path.join(RES, "dev", a.experiment)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for key, fr, feats, keys, cfg, s in EXPERIMENTS[a.experiment](range(a.seeds)):
        dst = os.path.join(out_dir, key + ".parquet")
        if os.path.exists(dst):
            continue
        res = ladder.run_ladder(fr, feats, keys, cfg, seed=s)
        if res is None:
            print("skip", key, flush=True)
            continue
        out, diag = res
        out.to_parquet(dst, index=False)
        diag["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}
        json.dump(diag, open(os.path.join(out_dir, key + ".json"), "w"), indent=1, default=str)
        sc = ladder.score(out).set_index("rung").skill_macro
        best = sc.idxmax()
        print(f"{key:32s} none->global {sc['global']:+.4f} series {sc['series']:+.4f} "
              f"inst {sc.get('instance', np.nan):+.4f} best={best} {sc[best]:+.4f} [{diag['seconds']}s, total {time.time()-t0:.0f}s]",
              flush=True)
    print("DONE", a.experiment, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
