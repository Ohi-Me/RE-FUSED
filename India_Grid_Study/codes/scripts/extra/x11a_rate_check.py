"""Check the block-level deviation charge of the CERC DSM Regulations 2024 (codes/refused/dsm2024.py) against the
charges in the official ERPC settlement accounts, block by block (exploratory validation, not pre-registered).

For every State buyer and block the net payable computed from the published frequency, schedule, deviation and
normal rate is compared with payable minus receivable in the account. The buyer category (standard, or schedule up
to 400 MW) is taken from the schedule of the block.
Output: results/extra/dsm_rate_check.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from refused import dsm2024 as R  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(ROOT, "data", "interim", "erpc_dsm_blocks.parquet")
OUT = os.path.join(ROOT, "results", "extra", "dsm_rate_check.json")


def model_charge(D):
    small = 4 * D.sched_mwh.abs() <= 400
    std = R.charge(D.dev_mwh.to_numpy(), D.sched_mwh.to_numpy(), D.nr_rs_mwh.to_numpy(), D.freq_hz.to_numpy())
    sml = R.charge(D.dev_mwh.to_numpy(), D.sched_mwh.to_numpy(), D.nr_rs_mwh.to_numpy(), D.freq_hz.to_numpy(), "small")
    return np.where(small, sml, std)


def main():
    D = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else SRC)
    D["official"] = D.payable_rs.astype(float) - D.receivable_rs.astype(float)
    D["model"] = model_charge(D)
    err = (D.model - D.official).abs()
    tol = np.maximum(1.0, 0.001 * D.official.abs())          # Rs 1 or 0.1 %
    res = dict(note="exploratory validation of the rate function against official ERPC accounts",
               n_blocks=int(len(D)), share_within_tolerance=float((err <= tol).mean()),
               share_within_rs1=float((err <= 1.0).mean()), max_abs_error_rs=float(err.max()),
               median_abs_error_rs=float(err.median()), abs_error_share=float(err.sum() / D.official.abs().sum()),
               by_entity={})
    for e, g in D.groupby("entity"):
        ge = (g.model - g.official).abs()
        res["by_entity"][e] = dict(n=int(len(g)), within=float((ge <= np.maximum(1.0, 0.001 * g.official.abs())).mean()),
                                   official_net_rs=float(g.official.sum()), model_net_rs=float(g.model.sum()),
                                   abs_error_share=float(ge.sum() / g.official.abs().sum()))
    bad = D[err > tol]
    res["mismatch_examples"] = bad.head(15)[["entity", "date", "block", "freq_hz", "sched_mwh", "dev_mwh",
                                             "nr_rs_mwh", "official", "model"]].astype(str).to_dict("records")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "mismatch_examples"}, indent=1))
    print(bad.head(15)[["entity", "date", "block", "freq_hz", "sched_mwh", "dev_mwh", "nr_rs_mwh", "official", "model"]])


if __name__ == "__main__":
    main()
