"""DC1b attribution check: re-run a subset of the pre-registered X1 partition grid WITHOUT the filter and compare it
row by row with the frozen results. If the subset reproduces, differences in the DC1b run are caused by the filter.
Output: results/confirm_sensitivity/x1/x1_reproducibility_check.csv and a printed summary."""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import gates as G, guard  # noqa: E402
from refused_gate.paths import RES  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev"))
import x1_phase_diagram as X  # noqa: E402

guard.require_confirmatory()
OFFSET = 900
frozen = pd.read_csv(os.path.join(RES, "confirm", "x1", "x1_partition_confirm.csv"))
va, te, feats = X.substrate(0, "opsd_confirm", clean=False)
phis = [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2, 3 * np.pi / 4]
rows = []
pat_cache = {}
subset = [c for i, c in enumerate(itertools.product(["unit", "state"], [0.0, 0.5, 1.0, 2.0, 4.0, 8.0], phis, [1.0, 2.0],
                                                      [0.25, 1.0], range(2))) if i % 24 == 0]
for axis, m, phi, kappa, frac, seed in subset:
    if axis == "unit":
        cv, ct = np.array(["all"] * len(va)), np.array(["all"] * len(te))
        fv, ft = va.sid.to_numpy(), te.sid.to_numpy()
    else:
        cv, ct = va.sid.to_numpy(), te.sid.to_numpy()
        fv, ft = np.asarray(G.cell_keys(va.sid, va.hour)), np.asarray(G.cell_keys(te.sid, te.hour))
    pk = (axis, seed)
    if pk not in pat_cache:
        pat_cache[pk] = X.cell_patterns(fv, ft, va.r.to_numpy(), cv, ct, np.random.default_rng(5000 + OFFSET + seed))
    res = X.make_run(va, te, cv, ct, fv, ft, pat_cache[pk], m, phi, kappa, frac, OFFSET + seed, fine="partition")
    res["axis"] = axis
    f = frozen[(frozen.axis == axis) & (frozen.m == m) & np.isclose(frozen.phi, phi) & (frozen.kappa == kappa)
               & (frozen.frac == frac) & (frozen.seed == OFFSET + seed)]
    for col in ("realised", "oracle_pred", "dV", "part_contiguous_G"):
        res[col + "_frozen"] = float(f[col].iloc[0]) if len(f) else np.nan
    rows.append(res)
R = pd.DataFrame(rows)
out = os.path.join(RES, "confirm_sensitivity", "x1", "x1_reproducibility_check.csv")
R.to_csv(out, index=False)
for col in ("realised", "oracle_pred", "dV", "part_contiguous_G"):
    d = np.abs(R[col] - R[col + "_frozen"]) / (np.abs(R[col + "_frozen"]) + 1e-9)
    print(f"{col:20s} max relative difference {d.max():.2e}  median {d.median():.2e}  (n={len(R)})")
print("wrote", out)
