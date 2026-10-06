"""Confirmatory battery XC (pre-registered; runs only after the pre-registration is frozen).

Every configuration below is fixed by design/02_preregistration.md. The script refuses to start unless the blinding
guard is unlocked. Results go to results/confirm/<block>/<key>.parquet|json and are never overwritten.

Blocks
  C1  NYISO load 2026 ladders       baselines iso, snaive168, lag48 and learned (lgbm, chronos, nhits, patchtst, tide,
                                    dlinear); 3 seeds
  C2  NYISO data-budget ladder      ISO baseline; validation fractions 0.1, 0.25, 0.5, 1.0; OOF train + validation; 3 seeds
  C3  OPSD 2019 ladders             baselines tso, snaive168, lag48; 2 seeds
  C4  OPSD 2020 ladders             baselines tso, snaive168, lag48; 2 seeds (pandemic shock period)
  C5  OPSD 2019 data-budget ladder  TSO baseline; fractions 0.1, 0.25, 1.0; OOF train + validation; 2 seeds
  C6  UCI 2014 ladders              60 pre-registered clients; baselines snaive168, lag48; 2 seeds
  C7  UCI 2014 data-budget ladder   snaive168; fractions 0.1, 0.25, 1.0; OOF train + validation; 2 seeds
Separate scripts (same freeze): X5/X8 on NYISO 2026 (x5_x8_prob_energy.py --mode confirm), X1 fresh grid on the OPSD
substrate (x1_phase_diagram.py --source opsd_confirm --seed_offset 900), learned baselines in confirm mode.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import data, guard, ladder  # noqa: E402
from refused_gate.paths import PROC, RES  # noqa: E402

UCI_SUBSET_SEED = 7
UCI_SUBSET_SIZE = 60


def uci_subset(df):
    clients = np.sort(df.sid.unique())
    keep = np.random.default_rng(UCI_SUBSET_SEED).choice(clients, UCI_SUBSET_SIZE, replace=False)
    return df[df.sid.isin(keep)]


def _surrogate_nyiso():
    ds = data.nyiso_load("dev")                      # schema only: dev rows relabelled with confirmatory splits
    d = data._surrogate(ds["df"].drop(columns=["split"]))
    d["split"] = np.select([d.ts < "2024-01-01", d.ts < "2025-01-01"], ["train", "val"], "test")
    ds["df"] = d
    return ds


def blocks(surrogate=False):
    mode = "surrogate" if surrogate else "confirm"
    ny = _surrogate_nyiso() if surrogate else data.nyiso_load("confirm")
    lb = os.path.join(PROC, "nyiso_learned_baselines_confirm_s0.parquet")
    df = ny["df"]
    cols = dict(ny["baselines"])
    if os.path.exists(lb):
        L = pd.read_parquet(lb)
        df = df.merge(L, on=["sid", "ts"], how="left")
        cols.update({c[2:]: c for c in L.columns if c.startswith("B_")})
    for b, col in cols.items():
        fr = df.assign(y_t=df.y, B=df[col])
        if b not in ny["baselines"]:
            fr = fr[fr.ts >= "2023-01-01"]
        for s in range(3):
            yield "C1", f"{b}_s{s}", fr, ny["feats"], ["hour_block", "hour"], ladder.LadderConfig(n_bag=3, delay_days=2), s
    fr = df.assign(y_t=df.y, B=df.iso_fc)
    for s in range(3):
        for frac in (0.1, 0.25, 0.5):
            yield "C2", f"val_frac{frac}_s{s}", fr, ny["feats"], ["hour_block", "hour"], \
                ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit_fraction=frac, do_time=False), s
        yield "C2", f"oof_train_val_s{s}", fr, ny["feats"], ["hour_block", "hour"], \
            ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit="oof_train+val", do_time=False), s
    for period, blk in (("2019", "C3"), ("2020", "C4")):
        op = data.opsd_load(mode, period)
        for b, col in op["baselines"].items():
            fr = op["df"].assign(y_t=op["df"].y, B=op["df"][col])
            for s in range(2):
                yield blk, f"{b}_s{s}", fr, op["feats"], ["hour_block", "hour"], \
                    ladder.LadderConfig(n_bag=3, delay_days=2), s
    op = data.opsd_load(mode, "2019")
    fr = op["df"].assign(y_t=op["df"].y, B=op["df"].tso_fc)
    for s in range(2):
        for frac in (0.1, 0.25):
            yield "C5", f"val_frac{frac}_s{s}", fr, op["feats"], ["hour_block", "hour"], \
                ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit_fraction=frac, do_time=False), s
        yield "C5", f"oof_train_val_s{s}", fr, op["feats"], ["hour_block", "hour"], \
            ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit="oof_train+val", do_time=False), s
    uc = data.uci_load(mode)
    udf = uci_subset(uc["df"])
    for b, col in uc["baselines"].items():
        fr = udf.assign(y_t=udf.y, B=udf[col])
        for s in range(2):
            yield "C6", f"{b}_s{s}", fr, uc["feats"], ["hour_block", "hour"], ladder.LadderConfig(n_bag=3, delay_days=2), s
    fr = udf.assign(y_t=udf.y, B=udf.y_lag168)
    for s in range(2):
        for frac in (0.1, 0.25):
            yield "C7", f"val_frac{frac}_s{s}", fr, uc["feats"], ["hour_block", "hour"], \
                ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit_fraction=frac, do_time=False), s
        yield "C7", f"oof_train_val_s{s}", fr, uc["feats"], ["hour_block", "hour"], \
            ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit="oof_train+val", do_time=False), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated block ids")
    ap.add_argument("--surrogate", action="store_true", help="code test on synthetic targets (no unlock, no real data)")
    ap.add_argument("--max_per_block", type=int, default=0)
    a = ap.parse_args()
    if not a.surrogate:
        guard.require_confirmatory()
    only = set(a.only.split(",")) if a.only else None
    t0 = time.time()
    count = {}
    for blk, key, fr, feats, keys, cfg, s in blocks(a.surrogate):
        if only and blk not in only:
            continue
        count[blk] = count.get(blk, 0) + 1
        if a.max_per_block and count[blk] > a.max_per_block:
            continue
        if a.surrogate:
            cfg.n_bag, cfg.part_eval_rows = 1, 3000
        out_dir = os.path.join(RES, "surrogate" if a.surrogate else "confirm", blk)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, key + ".parquet")
        if os.path.exists(dst):
            continue
        res = ladder.run_ladder(fr, feats, keys, cfg, seed=s)
        if res is None:
            print("skip", blk, key, flush=True)
            continue
        out, diag = res
        out.to_parquet(dst, index=False)
        diag["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}
        json.dump(diag, open(dst.replace(".parquet", ".json"), "w"), indent=1, default=str)
        print(f"{blk} {key:28s} done [{diag['seconds']}s, total {time.time()-t0:.0f}s]", flush=True)
    print("DONE confirmatory ladders", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
