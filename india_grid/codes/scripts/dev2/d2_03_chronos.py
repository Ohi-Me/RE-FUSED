"""dev2 step 3: pretrained forecasters, zero-shot (protocol 06, section 2).

chronos2      Chronos-2 with covariates. For each cutoff c (last published target day, issue day = c + lag) it gets the
              target history up to c (up to 512 days, missing days as NaN), the same lag-aligned past covariates as TFT,
              and the calendar as known future covariates. Cross-learning (all series of one cutoff forecast together)
              is tried on and off; the better one on validation MASE is kept. With cross-learning on, every call
              holds one cutoff only, so no series can see days that are in the future for another.
chronos_base  Chronos-Bolt-Base, target history only, like Chronos-Bolt-Small in round 1.

Weights come from data/hf_cache (downloaded once by the setup job). No training, no seeds.
Outputs: results/dev2/o2/preds/{tid}__chronos2.parquet, {tid}__chronos_base.parquet; logs/chronos2_{tid}.json
Usage:   python d2_03_chronos.py chronos2,chronos_base T1,T2,T3,T4,T5 [--smoke]
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from o2_03_nf import build_df  # noqa: E402

SMOKE = "--smoke" in sys.argv
CTX = 512


def calendar(dates):
    dates = pd.DatetimeIndex(dates)
    dow, doy = dates.dayofweek.to_numpy(), dates.dayofyear.to_numpy()
    return {"f_dow_sin": np.sin(2 * np.pi * dow / 7), "f_dow_cos": np.cos(2 * np.pi * dow / 7),
            "f_doy_sin": np.sin(2 * np.pi * doy / 365.25), "f_doy_cos": np.cos(2 * np.pi * doy / 365.25)}


def cutoffs(Ly):
    first = O.TRAIN_END - pd.Timedelta(days=Ly)
    last = O.DEV_END - pd.Timedelta(days=1 + Ly)
    cut = pd.date_range(first, last)
    return cut[:5] if SMOKE else cut


def samples_keys(tid):
    S = pd.read_parquet(os.path.join(D.DEV1, "o2", "samples", f"{tid}.parquet"), columns=D.KEYS)
    return S[S.split.isin(["validation", "dev_test"])]


def run_chronos2(tid, pipe):
    out = D.folder("o2", "preds")
    target_file = os.path.join(out, f"{tid}__chronos2.parquet")
    if os.path.exists(target_file):
        print(tid, "chronos2 already done, skipped", flush=True)
        return
    t0 = time.time()
    P = O.load_panel()
    df, hist, Ly = build_df(P, tid)
    h = Ly + 3
    series = {}
    for sid, g in df.groupby("unique_id"):
        g = g.sort_values("ds").reset_index(drop=True)
        series[sid] = dict(ds=pd.DatetimeIndex(g.ds), y=np.where(g.available_mask > 0, g.y, np.nan).astype(np.float32),
                           cov={c: g[c].to_numpy(np.float32) for c in hist},
                           cal={k: v.astype(np.float32) for k, v in calendar(g.ds).items()})
    S = samples_keys(tid)
    log = {"tid": tid, "Ly": Ly, "covariates": hist, "context": CTX, "val_mase": {}}
    results = {}
    # confirmatory phase: cross-learning on or off as picked in development
    modes = (bool(D.frozen("o2", "logs", f"chronos2_{tid}.json")["cross_learning"]),) if D.USE_FROZEN else (False, True)
    for cross in modes:
        rows = []
        for c in cutoffs(Ly):
            inputs, sids = [], []
            future_cal = {k: v.astype(np.float32) for k, v in calendar(pd.date_range(c + pd.Timedelta(days=1), periods=h)).items()}
            for sid, s in series.items():
                i = s["ds"].get_loc(c)
                lo = max(0, i + 1 - CTX)
                past = {k: v[lo:i + 1] for k, v in s["cov"].items()}
                past.update({k: v[lo:i + 1] for k, v in s["cal"].items()})
                inputs.append({"target": s["y"][lo:i + 1], "past_covariates": past, "future_covariates": future_cal})
                sids.append(sid)
            q, _ = pipe.predict_quantiles(inputs, prediction_length=h, quantile_levels=list(O.QUANTILES),
                                          batch_size=4096, cross_learning=cross)
            qa = np.stack([t[0].float().cpu().numpy() for t in q])  # (series, h, 7)
            tau = c + pd.Timedelta(days=Ly)
            for H in O.HORIZONS:
                rows.append(pd.DataFrame({"sid": sids, "issue_date": tau, "H": H,
                                          **{name: qa[:, Ly + H - 1, j] for j, name in enumerate(D.QN)}}))
        pr = S.merge(pd.concat(rows, ignore_index=True), on=["sid", "issue_date", "H"], how="inner")
        pr[D.QN] = np.sort(pr[D.QN].to_numpy(), axis=1)
        v = pr[(pr.split == "validation") & pr.y.notna()].merge(D.scales(tid), on=["sid", "H"])
        log["val_mase"][str(cross)] = D.mase(v, v.q50.to_numpy()) if len(v) else np.nan
        results[cross] = pr
        print(f"  {tid} chronos2 cross_learning={cross}: val MASE {log['val_mase'][str(cross)]:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    best = min((k for k in results), key=lambda k: np.nan_to_num(log["val_mase"][str(k)], nan=np.inf))
    log["cross_learning"] = best
    pr = results[best].assign(model="chronos2")
    pr.to_parquet(target_file, index=False)
    log["seconds"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(D.folder("o2", "logs"), f"chronos2_{tid}.json"), "w"), indent=1, default=str)
    print(tid, "chronos2 done, cross_learning", best, f"{log['seconds']}s", flush=True)


def run_bolt_base(tid, pipe):
    out = D.folder("o2", "preds")
    target_file = os.path.join(out, f"{tid}__chronos_base.parquet")
    if os.path.exists(target_file):
        print(tid, "chronos_base already done, skipped", flush=True)
        return
    t0 = time.time()
    P = O.load_panel()
    Ly = D.lag_of(tid)
    h = Ly + 3
    wide = O.series_frame(P, tid).pivot(index="date", columns="sid", values="y").asfreq("D")
    wide = wide[wide.index <= O.DEV_END]
    sids, arr, dates = list(wide.columns), wide.to_numpy(dtype=np.float32), wide.index
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    rows = []
    for c in cutoffs(Ly):
        i = dates.get_loc(c)
        ctx = [torch.tensor(arr[max(0, i + 1 - CTX): i + 1, j]) for j in range(len(sids))]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=levels)
        q = q.float().cpu().numpy()
        tau = c + pd.Timedelta(days=Ly)
        for H in O.HORIZONS:
            rows.append(pd.DataFrame({"sid": sids, "issue_date": tau, "H": H,
                                      **{f"q{int(round(l * 100)):02d}": q[:, Ly + H - 1, k] for k, l in enumerate(levels)}}))
    pr = samples_keys(tid).merge(pd.concat(rows, ignore_index=True), on=["sid", "issue_date", "H"], how="inner")
    pr["q05"], pr["q95"] = np.nan, np.nan
    pr["model"] = "chronos_base"
    pr.to_parquet(target_file, index=False)
    print(tid, "chronos_base", len(pr), "rows", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    from chronos import BaseChronosPipeline
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args[0].split(",") if args else ["chronos2", "chronos_base"]
    tids = args[1].split(",") if len(args) > 1 else (["T5"] if SMOKE else list(O.TARGETS))
    status = {}
    for name, repo, fn in (("chronos2", "amazon/chronos-2", run_chronos2), ("chronos_base", "amazon/chronos-bolt-base", run_bolt_base)):
        if name not in names:
            continue
        try:
            pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cuda", torch_dtype=torch.float32)
        except Exception as e:  # weights missing on the cluster: record it and go on
            status[name] = f"not available: {e}"
            print(name, "not available:", e, flush=True)
            continue
        status[name] = "ok"
        for tid in tids:
            fn(tid, pipe)
        del pipe
        torch.cuda.empty_cache()
    json.dump(status, open(os.path.join(D.folder("o2", "logs"), "pretrained_status.json"), "w"), indent=1)
