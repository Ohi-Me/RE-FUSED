"""O2 step 2: LightGBM direct quantile models (protocol §3).

Per target and horizon H: 7 quantile models on the z-scale target z = (y − base)/s, global over series (series id as a
categorical feature). Fitting rows: train target dates before the last 180 train days; early stopping on those 180
days. Config (num_leaves 31 or 63) chosen by validation pinball with seed 0; the chosen config is refit with seeds 0–4
(bagging 0.8). Quantiles are rearranged to be non-decreasing. Seeds are averaged quantile-wise.
S-lat (instant publication) runs the chosen config with seed 0 on the instant samples.

Outputs: 06_results/dev/o2/preds/{tid}__lgbm.parquet, {tid}__lgbm_seeds.parquet, {tid}__lgbm_instant.parquet
         06_results/dev/o2/logs/lgbm_{tid}.json (config choice, best iterations, timings)
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8 import o2data as O  # noqa: E402
from refused8.paths import RES  # noqa: E402

OUT = os.path.join(RES, O.PHASE_DIR, "o2")
KEYS = ["tid", "sid", "issue_date", "H", "target_date", "split", "season", "y"]
QN = [f"q{int(round(q * 100)):02d}" for q in O.QUANTILES]


def pinball_z(z, Q):
    u = z[:, None] - Q
    t = np.array(O.QUANTILES)[None, :]
    return float(np.nanmean(np.maximum(t * u, (t - 1) * u)))


def fit_predict(S, H, num_leaves, seed):
    F = O.feature_columns(S) + ["sid_c"]
    d = S[S.H == H]
    tr = d[(d.split == "train") & (d.target_date < O.EARLY_STOP_START) & d.z.notna()]
    es = d[(d.split == "train") & (d.target_date >= O.EARLY_STOP_START) & d.z.notna()]
    ev = d[d.split.isin(["validation", "dev_test"])]
    Q, iters = [], []
    for q in O.QUANTILES:
        params = dict(objective="quantile", alpha=q, learning_rate=0.05, num_leaves=num_leaves, min_data_in_leaf=50,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=seed, verbose=-1, num_threads=6,
                      max_bin=127, deterministic=True, force_row_wise=True)
        m = lgb.train(params, lgb.Dataset(tr[F], tr.z), 2000, valid_sets=[lgb.Dataset(es[F], es.z)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        Q.append(m.predict(ev[F], num_iteration=m.best_iteration))
        iters.append(int(m.best_iteration))
    Q = np.sort(np.column_stack(Q), axis=1)
    return ev, Q, iters


def to_frame(ev, Qz, model):
    pr = ev[KEYS].copy()
    Y = ev.base.to_numpy()[:, None] + ev.s.to_numpy()[:, None] * Qz
    for j, c in enumerate(QN):
        pr[c] = Y[:, j]
    pr["model"] = model
    return pr


def run(tid):
    if O.is_complete(tid, "lgbm") and os.path.exists(os.path.join(OUT, "preds", f"{tid}__lgbm_instant.parquet")):
        print(tid, "lgbm already complete, skipped", flush=True)
        return
    t0 = time.time()
    S = pd.read_parquet(os.path.join(OUT, "samples", f"{tid}.parquet"))
    S["sid_c"] = S.sid.astype("category")
    log = {"tid": tid, "config_val_pinball_z": {}, "best_iterations": {}}
    choice = {}
    for H in O.HORIZONS:
        scores = {}
        for nl in (31, 63):
            ev, Qz, it = fit_predict(S, H, nl, 0)
            v = (ev.split == "validation").to_numpy()
            scores[nl] = pinball_z(ev.z.to_numpy()[v], Qz[v])
            log["best_iterations"][f"H{H}_nl{nl}_seed0"] = it
        choice[H] = min(scores, key=scores.get)
        log["config_val_pinball_z"][f"H{H}"] = scores
    log["num_leaves"] = choice
    seeds_out = []
    for seed in range(5):
        for H in O.HORIZONS:
            ev, Qz, it = fit_predict(S, H, choice[H], seed)
            log["best_iterations"][f"H{H}_seed{seed}"] = it
            fr = to_frame(ev, Qz, "lgbm")
            fr["seed"] = seed
            seeds_out.append(fr)
        print(f"  {tid} seed {seed} done [{time.time() - t0:.0f}s]", flush=True)
    sd = pd.concat(seeds_out, ignore_index=True)
    sd.to_parquet(os.path.join(OUT, "preds", f"{tid}__lgbm_seeds.parquet"), index=False)
    avg = sd.groupby(KEYS, dropna=False)[QN].mean().reset_index()
    avg[QN] = np.sort(avg[QN].to_numpy(), axis=1)
    avg["model"] = "lgbm"
    avg.to_parquet(os.path.join(OUT, "preds", f"{tid}__lgbm.parquet"), index=False)
    # S-lat: instant publication, chosen config, seed 0
    Si = pd.read_parquet(os.path.join(OUT, "samples", f"{tid}_instant.parquet"))
    Si["sid_c"] = Si.sid.astype(pd.CategoricalDtype(S.sid_c.cat.categories))
    inst = []
    for H in O.HORIZONS:
        ev, Qz, it = fit_predict(Si, H, choice[H], 0)
        inst.append(to_frame(ev, Qz, "lgbm_instant"))
    pd.concat(inst, ignore_index=True).to_parquet(os.path.join(OUT, "preds", f"{tid}__lgbm_instant.parquet"), index=False)
    log["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    json.dump(log, open(os.path.join(OUT, "logs", f"lgbm_{tid}.json"), "w"), indent=1, default=str)
    print(tid, "done", log["num_leaves"], f"{log['seconds']}s", flush=True)


if __name__ == "__main__":
    for tid in (sys.argv[1:] or list(O.TARGETS)):
        run(tid)
