"""Summarise development ladder runs: skill tables, refinement contrasts (block bootstrap), PART sign accuracy.

Usage: py -3.10 analyze_dev_ladders.py <experiment> [<experiment> ...]
Writes 06_results/dev/analysis/<experiment>_contrasts.csv and _part.csv
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import analysis as A  # noqa: E402
from refused_gate.paths import RES  # noqa: E402

REFINEMENTS = [("global", "series", "global->series"), ("series", "instance", "series->instance")]


def configs(experiment):
    keys = sorted({re.sub(r"_s\d+\.parquet$", "", os.path.basename(f))
                   for f in glob.glob(os.path.join(RES, "dev", experiment, "*.parquet"))})
    return keys


def main():
    out_dir = os.path.join(RES, "dev", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    for exp in sys.argv[1:]:
        crow, prow = [], []
        for key in configs(exp):
            runs = A.load_runs(exp, key)
            if not runs:
                continue
            L = A.row_losses(runs)
            rungs = [c for c in L.columns if c not in ("sid", "ts", "base")]
            state = [r for r in rungs if r.startswith("series_x_")]
            refs = REFINEMENTS + [("series", s, "series->" + s) for s in state]
            refs += [("series", t, "series->" + t) for t in rungs if t.startswith("time_")]
            block = 14 if exp.endswith("india") else 7
            for c, f, name in refs:
                if c in L and f in L:
                    r = A.contrast(L, c, f, block_days=block)
                    r.update(experiment=exp, config=key, refinement=name)
                    crow.append(r)
                    # PART predictions averaged over seeds, per block design
                    for mode_name in ("part", "part_modes"):
                        pass
                    preds = {}
                    for _, _, diag in runs:
                        modes = diag.get("part_modes") or {"contiguous": diag.get("part", {})}
                        for mode, part in modes.items():
                            if name in part:
                                preds.setdefault(mode, []).append(part[name]["G"])
                        if f.startswith("time_rolling_30") and "part_time" in diag:
                            preds.setdefault("part_time30", []).append(diag["part_time"]["series->rolling_30"]["G"])
                        if f.startswith("time_rolling_90") and "part_time" in diag:
                            preds.setdefault("part_time90", []).append(diag["part_time"]["series->rolling_90"]["G"])
                        if "nested_risk" in diag and c in diag["nested_risk"] and f in diag["nested_risk"]:
                            preds.setdefault("nested", []).append(diag["nested_risk"][c] - diag["nested_risk"][f])
                    base_mse = float(L.base.mean())
                    for mode, v in preds.items():
                        prow.append(dict(experiment=exp, config=key, refinement=name, predictor=mode,
                                         pred_gain_skill=float(np.mean(v)) / base_mse, realised_skill=r["gain_skill"],
                                         realised_lo=r["lo"], realised_hi=r["hi"]))
            print(f"{exp:18s} {key:32s} " + " ".join(f"{x['refinement'].split('->')[1][:14]}={x['gain_skill']:+.4f}"
                                                     for x in crow if x["config"] == key), flush=True)
        C, P = pd.DataFrame(crow), pd.DataFrame(prow)
        C.to_csv(os.path.join(out_dir, f"{exp}_contrasts.csv"), index=False)
        P.to_csv(os.path.join(out_dir, f"{exp}_part.csv"), index=False)
        if len(P):
            P["correct"] = np.sign(P.pred_gain_skill) == np.sign(P.realised_skill)
            P["decisive"] = (P.realised_lo > 0) | (P.realised_hi < 0)
            print("\nsign accuracy by predictor (all | decisive realised CI):")
            print(P.groupby("predictor").agg(n=("correct", "size"), acc=("correct", "mean")).round(3).to_string())
            print(P[P.decisive].groupby("predictor").agg(n=("correct", "size"), acc=("correct", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
