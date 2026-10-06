"""dev2 step 1: XGBoost multi-quantile trees on the GPU (protocol 06, section 2).

Same samples, features, split and z-scale target as LightGBM in round 1. One booster per horizon learns all seven
quantiles at once. Depth 6 or 8 is picked on validation pinball loss with seed 0; then seeds 0-4 are fitted and
averaged quantile by quantile.

Outputs: results/dev2/o2/preds/{tid}__xgb.parquet and _seeds.parquet; results/dev2/o2/logs/xgb_{tid}.json
Usage:   python d2_01_xgb.py [T1 T2 ...] [--smoke]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402

SMOKE = "--smoke" in sys.argv
DEPTHS = (6,) if SMOKE else (6, 8)
SEEDS = range(1) if SMOKE else range(5)


def pinball_z(z, Q):
    u = z[:, None] - Q
    t = np.array(O.QUANTILES)[None, :]
    return float(np.nanmean(np.maximum(t * u, (t - 1) * u)))


def fit_predict(S, H, depth, seed):
    feats = O.feature_columns(S) + ["sid_c"]
    d = S[S.H == H]
    tr = d[(d.split == "train") & (d.target_date < O.EARLY_STOP_START) & d.z.notna()]
    es = d[(d.split == "train") & (d.target_date >= O.EARLY_STOP_START) & d.z.notna()]
    ev = d[d.split.isin(["validation", "dev_test"])]
    dtr = xgb.QuantileDMatrix(tr[feats], tr.z, enable_categorical=True, max_bin=256)
    des = xgb.QuantileDMatrix(es[feats], es.z, ref=dtr, enable_categorical=True)
    params = dict(objective="reg:quantileerror", quantile_alpha=np.array(O.QUANTILES), tree_method="hist",
                  device="cuda", max_depth=depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=50, seed=seed)
    booster = xgb.train(params, dtr, num_boost_round=100 if SMOKE else 3000, evals=[(des, "es")],
                        early_stopping_rounds=50, verbose_eval=False)
    Q = booster.predict(xgb.DMatrix(ev[feats], enable_categorical=True), iteration_range=(0, booster.best_iteration + 1))
    return ev, np.sort(Q, axis=1), int(booster.best_iteration)


def to_frame(ev, Qz):
    pr = ev[D.KEYS].copy()
    Y = ev.base.to_numpy()[:, None] + ev.s.to_numpy()[:, None] * Qz
    for j, c in enumerate(D.QN):
        pr[c] = Y[:, j]
    pr["model"] = "xgb"
    return pr


def run(tid):
    out = D.folder("o2", "preds")
    done = os.path.join(out, f"{tid}__xgb_seeds.parquet")
    if os.path.exists(done) and pd.read_parquet(done, columns=["seed"]).seed.nunique() >= len(SEEDS):
        print(tid, "xgb already done, skipped", flush=True)
        return
    t0 = time.time()
    S = pd.read_parquet(os.path.join(D.DEV1, "o2", "samples", f"{tid}.parquet"))
    S["sid_c"] = S.sid.astype("category")
    log = {"tid": tid, "xgboost": xgb.__version__, "val_pinball_z": {}, "best_iteration": {}}
    choice = {}
    if D.USE_FROZEN:  # depth taken from the development run, not picked again
        fz = D.frozen("o2", "logs", f"xgb_{tid}.json")["depth"]
        choice = {H: int(fz[str(H)]) for H in O.HORIZONS}
    for H in ([] if D.USE_FROZEN else O.HORIZONS):
        scores = {}
        for depth in DEPTHS:
            ev, Qz, it = fit_predict(S, H, depth, 0)
            v = (ev.split == "validation").to_numpy()
            scores[depth] = pinball_z(ev.z.to_numpy()[v], Qz[v])
            log["best_iteration"][f"H{H}_d{depth}_seed0"] = it
        choice[H] = min(scores, key=scores.get)
        log["val_pinball_z"][f"H{H}"] = scores
    log["depth"] = choice
    frames = []
    for seed in SEEDS:
        for H in O.HORIZONS:
            ev, Qz, it = fit_predict(S, H, choice[H], seed)
            log["best_iteration"][f"H{H}_seed{seed}"] = it
            fr = to_frame(ev, Qz)
            fr["seed"] = seed
            frames.append(fr)
        print(f"  {tid} xgb seed {seed} done [{time.time() - t0:.0f}s]", flush=True)
    sd = pd.concat(frames, ignore_index=True)
    sd.to_parquet(done, index=False)
    avg = sd.groupby(D.KEYS, dropna=False)[D.QN].mean().reset_index()
    avg[D.QN] = np.sort(avg[D.QN].to_numpy(), axis=1)
    avg["model"] = "xgb"
    avg.to_parquet(os.path.join(out, f"{tid}__xgb.parquet"), index=False)
    log["seconds"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(D.folder("o2", "logs"), f"xgb_{tid}.json"), "w"), indent=1, default=str)
    print(tid, "xgb done, depth", choice, f"{log['seconds']}s", flush=True)


if __name__ == "__main__":
    tids = [a for a in sys.argv[1:] if not a.startswith("--")] or (["T5"] if SMOKE else list(O.TARGETS))
    for t in tids:
        run(t)
