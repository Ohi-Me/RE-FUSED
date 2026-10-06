"""O2 step 3: TFT and PatchTST (neuralforecast 3.2.2), protocol §3.

Alignment. For a target with source lag Ly, a forecast cutoff c is the last published target date, so the issue day
is τ = c + Ly and targets τ+1…τ+3 are NF steps Ly+1…Ly+3 (h = Ly + 3). A historical exogenous column with lag Lc is
shifted so that its value at date d is the source value at d + Ly − Lc; at cutoff c this is the value published by τ.
Missing target days carry available_mask = 0 (excluded from the loss); missing exogenous values are forward-filled
(≤ 3 days) and then set to the series' train mean, with source-group missingness masks as extra exogenous columns.

Training: one fit per model × target × seed on data before the first evaluation cutoff; the last 180 days of that
portion are used for early stopping only. Cross-validation windows (step 1 day, no refit) cover validation and
development-test target dates. Seeds 0–4, averaged quantile-wise.

Outputs: 06_results/dev/o2/preds/{tid}__{tft|patchtst}.parquet and ..._seeds.parquet; logs/nf_{model}_{tid}.json
"""
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8 import o2data as O  # noqa: E402
from refused8.paths import RES  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
OUT = os.path.join(RES, O.PHASE_DIR, "o2")
KEYS = ["tid", "sid", "issue_date", "H", "target_date", "split", "season", "y"]
QN = [f"q{int(round(q * 100)):02d}" for q in O.QUANTILES]
GROUP_MASKS = {"m_psp": "dem_energy_met_gwh", "m_npp": "gen_act_gwh_total", "m_re": "re_all_total_re_gwh",
               "m_prc": "prc_dam_acp", "m_wx": "wx_tmax_c", "m_prc_rtm": "prc_rtm_acp", "m_dsm": "prc_dsm_normal"}


def build_df(P, tid):
    t = O.TARGETS[tid]
    L = O.lags()
    Ly = L.get(t["col"], 1)
    D = O.series_frame(P, tid)
    exog = [c for c in (O.ENTITY_EXOG if t["level"] == "entity" else O.AREA_EXOG) if c in D.columns]
    frames = []
    for sid, g in D.groupby("sid"):
        g = g.set_index("date").asfreq("D")
        trn = g[g.index <= O.TRAIN_END]
        f = pd.DataFrame({"unique_id": sid, "ds": g.index})
        f["available_mask"] = g.y.notna().astype(float).to_numpy()
        f["y"] = g.y.ffill().fillna(trn.y.mean()).to_numpy()
        hist = []
        for c in exog:
            if trn[c].notna().sum() < 30:
                continue
            x = g[c].shift(L.get(c, 1) - Ly)
            f[c] = x.ffill(limit=3).fillna(trn[c].mean()).to_numpy()
            hist.append(c)
        for m, c in GROUP_MASKS.items():
            if c in g:
                f[m] = g[c].shift(L.get(c, 1) - Ly).isna().astype(float).to_numpy()
                hist.append(m)
        doy, dow = g.index.dayofyear.to_numpy(), g.index.dayofweek.to_numpy()
        f["f_dow_sin"], f["f_dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
        f["f_doy_sin"], f["f_doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        frames.append((f, hist))
    # keep exogenous columns present for ≥ 80 % of series; series without a column (e.g. carbon intensity for entities
    # without monitored generation) get a constant 0, which carries no information after per-series normalisation
    counts = pd.Series([c for _, h in frames for c in h]).value_counts()
    hist_cols = sorted(counts[counts >= 0.8 * len(frames)].index)
    for f, _ in frames:
        for c in hist_cols:
            if c not in f:
                f[c] = 0.0
    df = pd.concat([f[["unique_id", "ds", "y", "available_mask", "f_dow_sin", "f_dow_cos", "f_doy_sin", "f_doy_cos"]
                      + hist_cols] for f, _ in frames], ignore_index=True)
    df = df[df.ds <= O.DEV_END].reset_index(drop=True)
    return df, hist_cols, Ly


def quantile_of(colname):
    if colname.endswith("-median"):
        return 0.5
    side, lvl = colname.split("-")[-2], float(colname.split("-")[-1])
    return (1 - lvl / 100) / 2 if side == "lo" else (1 + lvl / 100) / 2


def run(model_name, tid, seeds=range(5)):
    if O.is_complete(tid, model_name, len(seeds)):
        print(model_name, tid, "already complete, skipped", flush=True)
        return
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import TFT, PatchTST
    P = O.load_panel()
    df, hist, Ly = build_df(P, tid)
    h = Ly + 3
    first_cut = O.TRAIN_END - pd.Timedelta(days=Ly)          # τ = 31 Mar 2023 → first validation target date
    last_cut = O.DEV_END - pd.Timedelta(days=h)
    n_windows = (last_cut - first_cut).days + 1
    S = pd.read_parquet(os.path.join(OUT, "samples", f"{tid}.parquet"), columns=KEYS)
    S = S[S.split.isin(["validation", "dev_test"])]
    log = {"tid": tid, "model": model_name, "h": h, "Ly": Ly, "n_windows": n_windows, "hist_exog": hist, "seeds": {}}
    t0 = time.time()
    seed_frames = []
    for seed in seeds:
        common = dict(h=h, input_size=O.WINDOW, loss=MQLoss(quantiles=O.QUANTILES), max_steps=1500,
                      early_stop_patience_steps=5, val_check_steps=50, random_seed=seed, batch_size=32,
                      windows_batch_size=512, inference_windows_batch_size=2048, accelerator="gpu", devices=1,
                      enable_progress_bar=False, enable_model_summary=False, logger=False,
                      enable_checkpointing=False)
        if model_name == "tft":
            m = TFT(hidden_size=64, hist_exog_list=hist, futr_exog_list=["f_dow_sin", "f_dow_cos", "f_doy_sin",
                                                                         "f_doy_cos"], scaler_type="robust", **common)
        else:
            m = PatchTST(patch_len=7, stride=3, hidden_size=64, n_heads=4, encoder_layers=3, scaler_type="robust",
                         **common)
        nf = NeuralForecast(models=[m], freq="D")
        cols = ["unique_id", "ds", "y", "available_mask"] + (["f_dow_sin", "f_dow_cos", "f_doy_sin", "f_doy_cos"] + hist
                                                            if model_name == "tft" else [])
        cv = nf.cross_validation(df=df[cols], n_windows=n_windows, step_size=1, val_size=180, refit=False)
        qcols = [c for c in cv.columns if c.startswith(("TFT", "PatchTST"))]
        qmap = {c: f"q{int(round(quantile_of(c) * 100)):02d}" for c in qcols}
        cv = cv.rename(columns=qmap)
        cv["issue_date"] = pd.to_datetime(cv.cutoff) + pd.Timedelta(days=Ly)
        cv["H"] = (pd.to_datetime(cv.ds) - cv.issue_date).dt.days
        cv = cv[cv.H.isin(O.HORIZONS)].rename(columns={"unique_id": "sid", "ds": "target_date"})
        fr = S.merge(cv[["sid", "issue_date", "H"] + QN], on=["sid", "issue_date", "H"], how="inner")
        fr[QN] = np.sort(fr[QN].to_numpy(), axis=1)
        fr["model"], fr["seed"] = model_name, seed
        seed_frames.append(fr)
        log["seeds"][seed] = {"steps_trained": int(getattr(m, "global_step", -1)) if hasattr(m, "global_step") else None,
                              "rows": len(fr), "elapsed_s": round(time.time() - t0, 1)}
        print(f"  {model_name} {tid} seed {seed}: {len(fr)} rows [{time.time() - t0:.0f}s]", flush=True)
    sd = pd.concat(seed_frames, ignore_index=True)
    sd.to_parquet(os.path.join(OUT, "preds", f"{tid}__{model_name}_seeds.parquet"), index=False)
    avg = sd.groupby(KEYS, dropna=False)[QN].mean().reset_index()
    avg[QN] = np.sort(avg[QN].to_numpy(), axis=1)
    avg["model"] = model_name
    avg.to_parquet(os.path.join(OUT, "preds", f"{tid}__{model_name}.parquet"), index=False)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    json.dump(log, open(os.path.join(OUT, "logs", f"nf_{model_name}_{tid}.json"), "w"), indent=1, default=str)
    print(model_name, tid, "done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    model_names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["patchtst", "tft"]
    tids = sys.argv[2].split(",") if len(sys.argv) > 2 else list(O.TARGETS)
    seeds = range(int(sys.argv[3])) if len(sys.argv) > 3 else range(5)
    for mn in model_names:
        for tid in tids:
            run(mn, tid, seeds)
