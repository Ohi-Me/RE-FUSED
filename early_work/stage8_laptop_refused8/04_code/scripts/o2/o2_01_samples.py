"""O2 step 1: cache lag-aware samples for T1–T5 (standard and S-lat instant information sets), anchor forecasts and
MASE scales.

Outputs (06_results/dev/o2/):
  samples/{tid}.parquet, samples/{tid}_instant.parquet
  preds/{tid}__{anchor}.parquet  keys + q50 (anchors are point forecasts; intervals come from residual conformal)
  scales.parquet                 tid, sid, H, scale_mae, scale_mse (train seasonal-naive errors)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8 import o2data as O  # noqa: E402
from refused8.paths import RES  # noqa: E402

OUT = os.path.join(RES, O.PHASE_DIR, "o2")
KEYS = ["tid", "sid", "issue_date", "H", "target_date", "split", "season", "y"]


def main():
    for d in ("samples", "preds"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    P = O.load_panel()
    scales = []
    for tid in O.TARGETS:
        for instant in (False, True):
            S = O.tabular_samples(P, tid, instant=instant)
            S.to_parquet(os.path.join(OUT, "samples", f"{tid}{'_instant' if instant else ''}.parquet"), index=False)
            print(tid, "instant" if instant else "standard", S.shape, S.sid.nunique(), "series", flush=True)
            if instant:
                continue
            sc = O.mase_scales(S)
            sc["tid"] = tid
            scales.append(sc)
            for a in ("seasonal_naive", "persistence", "ma7"):
                pr = S[KEYS].copy()
                pr["q50"] = S[f"a_{a}"]
                pr["model"] = a
                pr.to_parquet(os.path.join(OUT, "preds", f"{tid}__{a}.parquet"), index=False)
    pd.concat(scales).to_parquet(os.path.join(OUT, "scales.parquet"), index=False)


if __name__ == "__main__":
    main()
