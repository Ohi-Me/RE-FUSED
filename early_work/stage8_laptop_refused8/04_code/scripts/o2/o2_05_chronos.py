"""O2 step 5: Chronos-Bolt-Small zero-shot forecasts (protocol §3).

Weights: amazon/chronos-bolt-small from the local HuggingFace cache (no download; HF_HUB_OFFLINE=1).
For each forecast cutoff c (last published target date) the context is the target history up to c (≤ 512 days,
missing days as NaN) and the prediction length is Ly + 3; issue day τ = c + Ly; horizons H = 1…3 are steps Ly+1…Ly+3.
Chronos-Bolt provides quantiles 0.1…0.9, so q05 and q95 are left missing (the model is calibrated with residual
conformal around its median in the evaluation). Deterministic (no seeds).

Output: 06_results/dev/o2/preds/{tid}__chronos.parquet (q10, q25, q50, q75, q90)
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8 import o2data as O  # noqa: E402
from refused8.paths import RES  # noqa: E402

OUT = os.path.join(RES, O.PHASE_DIR, "o2")
KEYS = ["tid", "sid", "issue_date", "H", "target_date", "split", "season", "y"]
CTX = 512
LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def run(tid, pipe):
    if O.is_complete(tid, "chronos", None):
        print(tid, "chronos already complete, skipped", flush=True)
        return
    t0 = time.time()
    P = O.load_panel()
    Ly = O.lags().get(O.TARGETS[tid]["col"], 1)
    h = Ly + 3
    D = O.series_frame(P, tid)
    wide = D.pivot(index="date", columns="sid", values="y").asfreq("D")
    wide = wide[wide.index <= O.DEV_END]
    sids = list(wide.columns)
    arr = wide.to_numpy(dtype=np.float32)
    dates = wide.index
    first_cut = O.TRAIN_END - pd.Timedelta(days=Ly)
    last_cut = O.DEV_END - pd.Timedelta(days=1 + Ly)
    rows = []
    for c in pd.date_range(first_cut, last_cut):
        i = dates.get_loc(c)
        ctx = [torch.tensor(arr[max(0, i + 1 - CTX): i + 1, j]) for j in range(len(sids))]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=LEVELS)
        q = q.float().cpu().numpy()  # (series, h, levels)
        tau = c + pd.Timedelta(days=Ly)
        for H in O.HORIZONS:
            step = Ly + H - 1
            rows.append(pd.DataFrame({"sid": sids, "issue_date": tau, "H": H,
                                      **{f"q{int(round(l * 100)):02d}": q[:, step, k] for k, l in enumerate(LEVELS)}}))
    pr = pd.concat(rows, ignore_index=True)
    S = pd.read_parquet(os.path.join(OUT, "samples", f"{tid}.parquet"), columns=KEYS)
    pr = S[S.split.isin(["validation", "dev_test"])].merge(pr, on=["sid", "issue_date", "H"], how="inner")
    pr["q05"], pr["q95"] = np.nan, np.nan
    pr["model"] = "chronos"
    pr.to_parquet(os.path.join(OUT, "preds", f"{tid}__chronos.parquet"), index=False)
    print(tid, "chronos", len(pr), "rows", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cuda", torch_dtype=torch.float32)
    for tid in (sys.argv[1].split(",") if len(sys.argv) > 1 else list(O.TARGETS)):
        run(tid, pipe)
