"""Sensitivity DC1b (after unblinding): where do sign errors of the crossing condition occur on the semi-synthetic
grid? Breakdown of H1 partition accuracy by |predicted gain| / estimation cost, for the pre-registered grid and the
grid rebuilt on the plausibility-filtered substrate. Output: results/confirm_sensitivity/x1/x1_boundary_breakdown.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate.paths import RES  # noqa: E402

FILES = {"primary": os.path.join(RES, "confirm", "x1", "x1_partition_confirm.csv"),
         "dc1b": os.path.join(RES, "confirm_sensitivity", "x1", "x1_partition_confirm_dc1b.csv")}


def main():
    out = {}
    for tag, f in FILES.items():
        X = pd.read_csv(f)
        dec = X[np.abs(X.oracle_pred) >= 0.5 * X.dV].copy()          # pre-registered non-boundary rule
        dec["ok"] = np.sign(dec.oracle_pred) == np.sign(dec.realised)
        dec["pok"] = np.sign(dec.part_contiguous_G) == np.sign(dec.realised)
        near = np.abs(dec.oracle_pred) < dec.dV
        out[tag] = dict(n=int(len(dec)), oracle_acc=float(dec.ok.mean()), part_acc=float(dec.pok.mean()),
                        near_n=int(near.sum()), near_oracle_acc=float(dec.ok[near].mean()),
                        near_part_acc=float(dec.pok[near].mean()),
                        far_n=int((~near).sum()), far_oracle_acc=float(dec.ok[~near].mean()),
                        far_part_acc=float(dec.pok[~near].mean()))
    path = os.path.join(RES, "confirm_sensitivity", "x1", "x1_boundary_breakdown.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
