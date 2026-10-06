"""RE-FUSED-6 / M1d: is the ladder conclusion an artifact of the admissible band? (defect D11)

After the D9 reparameterisation the optimal gates centre near 1.05, so the fixed cap [0, 1.5] became
binding (31% of fine-rung values clipped). A reviewer will ask whether "fine rungs lose" is produced
by the cap rather than by estimation cost. This sweeps the cap over {1.0, 1.5, 3.0, inf} on the
rescaled parameterisation and checks whether the ORDERING of rungs is invariant.

Unit: one (baseline, h, seed, cap) run. 4 configs x 5 seeds x 4 caps.
"""
import os, time, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1c_mod", os.path.join(HERE, "m1c_rescaled_corrector.py"))
m1c = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1c)
OUT = r"D:\\REFUSED5\results\refused6"

CAPS = [1.0, 1.5, 3.0, np.inf]
CONFIGS = [("persistence", 1), ("persistence", 7), ("roll7", 3), ("roll7", 7)]
SEEDS = 5

ds = m1c.m1.load_india()
rows, t0 = [], time.time()
for cap in CAPS:
    m1c.CLIP = cap                      # the only thing that changes
    for bname, h in CONFIGS:
        for seed in range(SEEDS):
            for r in m1c.one(ds, bname, h, seed, rescale=True):
                r["cap"] = cap
                rows.append(r)
    d = pd.DataFrame(rows).query("cap == @cap")
    p = d.groupby("rung").skill_sq_macro.mean()
    print(f"cap={cap:<5} | " + "  ".join(f"{k.replace('rung','r')[:12]}={p.get(k, float('nan')):+.4f}"
          for k in ["rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_wls_bagged"]) +
          f" | best={p.idxmax()} [{time.time()-t0:.0f}s]", flush=True)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "M1d_clip_robustness.csv"), index=False)
print("\n=== ordering invariance across caps ===")
piv = R.pivot_table(index="cap", columns="rung", values="skill_sq_macro").round(4)
print(piv.to_string())
print("\nbest rung per cap:", {float(c): piv.loc[c].idxmax() for c in piv.index})
print("\n=== fine minus per-series, per cap (paired over config x seed) ===")
for cap in CAPS:
    a = R[(R.cap == cap) & (R.rung == "rung5_wls_bagged")].set_index(["baseline", "h", "seed"]).skill_sq_macro
    b = R[(R.cap == cap) & (R.rung == "rung2_series")].set_index(["baseline", "h", "seed"]).skill_sq_macro
    j = a.align(b, join="inner")
    t = stats.ttest_rel(j[0], j[1])
    print(f"  cap={cap:<5} diff={float((j[0]-j[1]).mean()):+.5f}  fine wins={int((j[0] > j[1]).sum())}/{len(j[0])}  p={t.pvalue:.2e}")
print("DONE", time.time() - t0)
