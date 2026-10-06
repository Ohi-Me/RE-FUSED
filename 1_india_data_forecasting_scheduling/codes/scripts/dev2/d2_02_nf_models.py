"""dev2 step 2: more neural forecasters from neuralforecast 3.2.2 (protocol 06, section 2).

Models: N-HiTS, TiDE, BiTCN and N-BEATSx. They get the same lag-aligned daily frame as TFT in round 1 (build_df in
o2_03_nf.py): past covariates as published by the issue day, missing-source masks, and the calendar as known future
inputs. The input window (28 or 56 days) is picked on validation pinball loss with seed 0, then seeds 0-4 are run
with that window and averaged quantile by quantile.

Outputs: results/dev2/o2/preds/{tid}__{model}.parquet and _seeds.parquet; results/dev2/o2/logs/nf_{model}_{tid}.json
Usage:   python d2_02_nf_models.py nhits,tide,bitcn,nbeatsx T1,T2,T3,T4,T5 [--smoke]
"""
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from o2_03_nf import build_df, quantile_of  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

SMOKE = "--smoke" in sys.argv
WINDOWS = (28,) if SMOKE else (28, 56)
SEEDS = range(1) if SMOKE else range(5)
MAX_STEPS = 20 if SMOKE else 1500
CAL = ["f_dow_sin", "f_dow_cos", "f_doy_sin", "f_doy_cos"]


def make_model(name, h, input_size, hist, seed):
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS, TiDE, BiTCN, NBEATSx
    common = dict(h=h, input_size=input_size, loss=MQLoss(quantiles=O.QUANTILES), max_steps=MAX_STEPS,
                  early_stop_patience_steps=5, val_check_steps=10 if SMOKE else 50, random_seed=seed, batch_size=32,
                  windows_batch_size=512, inference_windows_batch_size=2048, accelerator="gpu", devices=1,
                  enable_progress_bar=False, enable_model_summary=False, logger=False, enable_checkpointing=False,
                  scaler_type="robust", hist_exog_list=hist, futr_exog_list=CAL)
    cls = {"nhits": NHITS, "tide": TiDE, "bitcn": BiTCN, "nbeatsx": NBEATSx}[name]
    return cls(**common)


def one_fit(name, df, hist, Ly, S, input_size, seed):
    """Fit once on data before the first validation cutoff and forecast every validation and test cutoff."""
    from neuralforecast import NeuralForecast
    h = Ly + 3
    first_cut = O.TRAIN_END - pd.Timedelta(days=Ly)
    last_cut = O.DEV_END - pd.Timedelta(days=h)
    n_windows = (last_cut - first_cut).days + 1
    m = make_model(name, h, input_size, hist, seed)
    nf = NeuralForecast(models=[m], freq="D")
    cols = ["unique_id", "ds", "y", "available_mask"] + CAL + hist
    cv = nf.cross_validation(df=df[cols], n_windows=n_windows, step_size=1, val_size=180, refit=False)
    model_col = type(m).__name__
    qcols = [c for c in cv.columns if c.startswith(model_col + "-")]
    cv = cv.rename(columns={c: f"q{int(round(quantile_of(c) * 100)):02d}" for c in qcols})
    cv["issue_date"] = pd.to_datetime(cv.cutoff) + pd.Timedelta(days=Ly)
    cv["H"] = (pd.to_datetime(cv.ds) - cv.issue_date).dt.days
    cv = cv[cv.H.isin(O.HORIZONS)].rename(columns={"unique_id": "sid"})
    fr = S.merge(cv[["sid", "issue_date", "H"] + D.QN], on=["sid", "issue_date", "H"], how="inner")
    fr[D.QN] = np.sort(fr[D.QN].to_numpy(), axis=1)
    return fr, int(getattr(m, "global_step", -1) or -1)


def val_pinball(fr, tid):
    v = fr[(fr.split == "validation") & fr.y.notna()].merge(D.scales(tid), on=["sid", "H"])
    return float(D.pinball_rows(v.y.to_numpy(), v[D.QN].to_numpy(), v.scale_mae.to_numpy()).mean())


def run(name, tid):
    out = D.folder("o2", "preds")
    seeds_file = os.path.join(out, f"{tid}__{name}_seeds.parquet")
    if os.path.exists(seeds_file) and pd.read_parquet(seeds_file, columns=["seed"]).seed.nunique() >= len(SEEDS):
        print(name, tid, "already done, skipped", flush=True)
        return
    t0 = time.time()
    P = O.load_panel()
    df, hist, Ly = build_df(P, tid)
    S = pd.read_parquet(os.path.join(D.DEV1, "o2", "samples", f"{tid}.parquet"), columns=D.KEYS)
    S = S[S.split.isin(["validation", "dev_test"])]
    log = {"tid": tid, "model": name, "Ly": Ly, "hist_exog": hist, "val_pinball": {}, "seeds": {}}
    tried = {}
    # confirmatory phase: the input window is the one picked in development
    windows = (D.frozen("o2", "logs", f"nf_{name}_{tid}.json")["input_size"],) if D.USE_FROZEN else WINDOWS
    for w in windows:
        fr, steps = one_fit(name, df, hist, Ly, S, w, 0)
        tried[w] = fr
        log["val_pinball"][w] = val_pinball(fr, tid)
        print(f"  {name} {tid} window {w}: val pinball {log['val_pinball'][w]:.4f} [{time.time() - t0:.0f}s]", flush=True)
    best_w = min(log["val_pinball"], key=log["val_pinball"].get)
    log["input_size"] = best_w
    frames = []
    for seed in SEEDS:
        fr = tried[best_w] if seed == 0 else one_fit(name, df, hist, Ly, S, best_w, seed)[0]
        fr = fr.assign(model=name, seed=seed)
        frames.append(fr)
        log["seeds"][seed] = dict(rows=len(fr), elapsed_s=round(time.time() - t0, 1))
        print(f"  {name} {tid} seed {seed}: {len(fr)} rows [{time.time() - t0:.0f}s]", flush=True)
    sd = pd.concat(frames, ignore_index=True)
    sd.to_parquet(seeds_file, index=False)
    avg = sd.groupby(D.KEYS, dropna=False)[D.QN].mean().reset_index()
    avg[D.QN] = np.sort(avg[D.QN].to_numpy(), axis=1)
    avg["model"] = name
    avg.to_parquet(os.path.join(out, f"{tid}__{name}.parquet"), index=False)
    log["seconds"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(D.folder("o2", "logs"), f"nf_{name}_{tid}.json"), "w"), indent=1, default=str)
    print(name, tid, "done, window", best_w, f"{log['seconds']}s", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args[0].split(",") if args else ["nhits", "tide", "bitcn", "nbeatsx"]
    tids = args[1].split(",") if len(args) > 1 else (["T5"] if SMOKE else list(O.TARGETS))
    for tid in tids:
        for name in names:
            run(name, tid)
