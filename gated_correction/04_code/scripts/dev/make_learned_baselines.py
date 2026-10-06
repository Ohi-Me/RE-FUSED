"""Out-of-sample learned baselines for NYISO zonal load (day-ahead information set).

For a target day D every model uses actual load up to the end of day D-2 (forecast origin 23:00 on D-2,
prediction steps 25..48 = the 24 hours of D). Learned models are fitted twice so that their predictions are
out-of-sample wherever they are used:
  fit A: train < 2022-01-01  -> predictions for 2022-2023 (corrector-training period of the ladder)
  fit B: train < 2024-01-01  -> predictions for 2024-2025 (gate-fitting and development test)
Confirmatory mode (after pre-registration): fit A train < 2023 -> 2023-2024; fit B train < 2025 -> 2025-2026.
Chronos-Bolt-Small is zero-shot: rolling inference over the whole prediction span.

Models (hyperparameters fixed a priori, not tuned): NHITS, PatchTST, TiDE, DLinear (neuralforecast, h=48,
input 336 h, max_steps 500; reduced from 1500 for compute, fixed before any baseline result was seen), LightGBM direct model on lag >= 48 h features and calendar, Chronos-Bolt-Small
(context 2048 h). Output: 03_data/processed/nyiso_learned_baselines_<mode>.parquet with columns sid, ts, B_<model>.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import guard  # noqa: E402
from refused_gate.paths import PROC  # noqa: E402

H, INPUT = 48, 336


def panel(mode):
    p = pd.read_parquet(os.path.join(PROC, "nyiso_load_panel.parquet"), columns=["zone", "ts", "load"])
    if mode == "confirm":
        guard.require_confirmatory()
    else:
        p = p[p.ts < "2026-01-01"]
    p = p.rename(columns={"zone": "unique_id", "ts": "ds", "load": "y"})
    p["unique_id"] = p.unique_id.astype(str)
    out = []
    for s, g in p.groupby("unique_id"):
        idx = pd.date_range(g.ds.min(), g.ds.max(), freq="h")
        g = g.set_index("ds").reindex(idx).rename_axis("ds").reset_index().assign(unique_id=s)
        g["y"] = g.y.interpolate(limit=3).ffill().bfill()          # DST gaps only (<= 1 h)
        out.append(g)
    return pd.concat(out, ignore_index=True)[["unique_id", "ds", "y"]]


def neural(df, train_end, pred_start, pred_end, seed):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DLinear, NHITS, PatchTST, TiDE
    kw = dict(h=H, input_size=INPUT, max_steps=500, random_seed=seed, scaler_type="robust",
              enable_progress_bar=False, enable_model_summary=False, logger=False)
    models = [NHITS(**kw), PatchTST(**kw), TiDE(**kw), DLinear(**kw)]
    nf = NeuralForecast(models=models, freq="h")
    hist = df[df.ds < train_end]
    nf.fit(df=hist, val_size=0)
    days = pd.date_range(pred_start, pred_end, freq="D")
    arrays = {s: (g.ds.to_numpy(), g.y.to_numpy()) for s, g in df.groupby("unique_id")}
    rows = []
    for c0 in range(0, len(days), 30):
        chunk = days[c0:c0 + 30]
        windows = []
        for D in chunk:
            cut = np.datetime64(D - pd.Timedelta(days=1))          # data through 23:00 of D-2
            for s, (ds_, y_) in arrays.items():
                j = np.searchsorted(ds_, cut)
                if j < INPUT:
                    continue
                windows.append(pd.DataFrame({"unique_id": f"{s}|{D.date()}", "ds": ds_[j - INPUT:j], "y": y_[j - INPUT:j]}))
        fc = nf.predict(df=pd.concat(windows, ignore_index=True))
        fc[["sid", "day"]] = fc.unique_id.str.split("|", expand=True)
        fc = fc[fc.ds.dt.normalize() == pd.to_datetime(fc.day)]
        rows.append(fc.drop(columns=["unique_id", "day"]))
        print(f"    neural {chunk[0].date()} ({c0}/{len(days)})", flush=True)
    out = pd.concat(rows, ignore_index=True)
    return out.rename(columns={"ds": "ts", "NHITS": "B_nhits", "PatchTST": "B_patchtst",
                               "TiDE": "B_tide", "DLinear": "B_dlinear"})


def lgbm(df, train_end, pred_start, pred_end, seed):
    import lightgbm as lgb
    d = df.copy()
    g = d.groupby("unique_id").y
    feats = []
    for L in (48, 49, 72, 96, 120, 144, 168, 192, 336):
        d[f"l{L}"] = g.shift(L)
        feats.append(f"l{L}")
    d["rm"] = g.shift(48).groupby(d.unique_id).transform(lambda x: x.rolling(168, min_periods=24).mean())
    d["hour"], d["dow"], d["doy"] = d.ds.dt.hour, d.ds.dt.dayofweek, d.ds.dt.dayofyear
    d["uid"] = d.unique_id.astype("category").cat.codes
    feats += ["rm", "hour", "dow", "doy", "uid"]
    tr = d[(d.ds < train_end)].dropna(subset=feats)
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1).fit(tr[feats], tr.y)
    te = d[(d.ds >= pred_start) & (d.ds < pd.Timestamp(pred_end) + pd.Timedelta(days=1))].dropna(subset=feats)
    return pd.DataFrame({"sid": te.unique_id.to_numpy(), "ts": te.ds.to_numpy(), "B_lgbm": m.predict(te[feats])})


def chronos(df, pred_start, pred_end):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cuda", torch_dtype=torch.float32)
    days = pd.date_range(pred_start, pred_end, freq="D")
    arrays = {s: (g.ds.to_numpy(), g.y.to_numpy().astype(np.float32)) for s, g in df.groupby("unique_id")}
    rows = []
    for c0 in range(0, len(days), 30):
        chunk = days[c0:c0 + 30]
        ctxs, keys = [], []
        for D in chunk:
            cut = np.datetime64(D - pd.Timedelta(days=1))
            for s, (ds_, y_) in arrays.items():
                j = np.searchsorted(ds_, cut)
                ctxs.append(torch.from_numpy(y_[max(0, j - 2048):j].copy()))
                keys.append((s, D))
        _, mean = pipe.predict_quantiles(ctxs, prediction_length=H, quantile_levels=[0.5])
        mean = mean.cpu().numpy()[:, 24:48]
        for (s, D), m in zip(keys, mean):
            rows.append(pd.DataFrame({"sid": s, "ts": pd.date_range(D, periods=24, freq="h"), "B_chronos": m}))
        if c0 % 180 == 0:
            print(f"    chronos {chunk[0].date()} ({c0}/{len(days)})", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev", choices=["dev", "confirm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="lgbm,chronos,neural")
    a = ap.parse_args()
    df = panel(a.mode)
    if a.mode == "dev":
        fits = [("2022-01-01", "2022-01-01", "2023-12-31"), ("2024-01-01", "2024-01-01", "2025-12-31")]
        span = ("2022-01-01", "2025-12-31")
    else:
        fits = [("2023-01-01", "2023-01-01", "2024-12-31"), ("2025-01-01", "2025-01-01", "2026-08-31")]
        span = ("2023-01-01", "2026-08-31")
    parts = []
    t0 = time.time()
    models = a.models.split(",")
    for train_end, ps, pe in fits:
        if "lgbm" in models:
            parts.append(("lgbm", lgbm(df, train_end, ps, pe, a.seed)))
            print(f"  lgbm fit<{train_end} done [{time.time()-t0:.0f}s]", flush=True)
        if "neural" in models:
            parts.append(("neural", neural(df, train_end, ps, pe, a.seed)))
            print(f"  neural fit<{train_end} done [{time.time()-t0:.0f}s]", flush=True)
    if "chronos" in models:
        parts.append(("chronos", chronos(df, *span)))
        print(f"  chronos done [{time.time()-t0:.0f}s]", flush=True)
    merged = None
    for name in dict.fromkeys(n for n, _ in parts):
        block = pd.concat([p for n, p in parts if n == name], ignore_index=True)
        block = block.groupby(["sid", "ts"], as_index=False).mean(numeric_only=True)
        merged = block if merged is None else merged.merge(block, on=["sid", "ts"], how="outer")
    dst = os.path.join(PROC, f"nyiso_learned_baselines_{a.mode}_s{a.seed}.parquet")
    if os.path.exists(dst):
        old = pd.read_parquet(dst)
        keep = [c for c in old.columns if c not in merged.columns or c in ("sid", "ts")]
        merged = old[keep].merge(merged, on=["sid", "ts"], how="outer")
    merged.to_parquet(dst, index=False)
    print("wrote", dst, merged.shape, "columns", list(merged.columns), f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
