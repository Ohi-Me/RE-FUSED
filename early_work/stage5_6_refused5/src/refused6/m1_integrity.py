"""RE-FUSED-6 / M1 integrity checks.

Verifies the canonical outputs before any downstream experiment is allowed to run.
Checks (each prints PASS/FAIL and is counted):
  C1  every planned (dataset, baseline, h) config is present
  C2  every config has exactly N_SEEDS seeds
  C3  every (config, seed) has exactly the full rung set
  C4  no duplicated (config, seed, rung) rows
  C5  seasonal-naive absent wherever h is a multiple of its season (D3)
  C6  rung0 (no correction) has exactly zero skill in all four metric variants
  C7  baseline magnitudes (base_mae/base_mse) identical across rungs within a (config, seed)
  C8  data-dependent counts (n_test, n_val, n_series) identical across seeds within a config
  C9  clipped fraction in [0,1]; rung0 clipped fraction is 0; CLIP matches the definitions file
  C10 macro and pooled aggregations both present, finite, and not silently equal everywhere
  C11 block-bootstrap file covers every config x compared rung, and each CI brackets its point estimate
  C12 metric definitions file exists and declares the canonical unit and primary metric
Exit code is non-zero if any check fails.
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = r"D:\\REFUSED5"
OUT = os.path.join(ROOT, "results", "refused6")
WHICH = sys.argv[1] if len(sys.argv) > 1 else "india"

RUNGS = ["rung0_none", "rung1_global", "rung2_series", "rung3_cell",
         "rung4_shrunk", "rung5_instance", "rung5b_bagged", "oracle_instance"]
PLANNED = {  # must mirror m1_canonical.py
    "india_daily": dict(baselines=["persistence", "roll7", "snaive7", "operator_schedule"],
                        horizons=[1, 2, 3, 7], season=7),
}
METRICS = ["skill_sq_macro", "skill_sq_pooled", "skill_mae_macro", "skill_mae_pooled"]

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))


runs_f = os.path.join(OUT, f"M1_runs_{WHICH}.csv")
boot_f = os.path.join(OUT, f"M1_blockboot_{WHICH}.csv")
defs_f = os.path.join(OUT, "M1_canonical_definitions.json")
if not os.path.exists(runs_f):
    print(f"missing {runs_f}"); sys.exit(2)
R = pd.read_csv(runs_f)
B = pd.read_csv(boot_f) if os.path.exists(boot_f) else None
D = json.load(open(defs_f)) if os.path.exists(defs_f) else {}
N_SEEDS = int(D.get("n_seeds", 10))
CLIP = float(D.get("clip", 1.5))

print(f"\nM1 INTEGRITY CHECKS ({WHICH}) — {len(R)} rows\n" + "=" * 70)

# expected configs: planned grid minus the D3 exclusion
expected = []
for ds, spec in PLANNED.items():
    if ds not in R.dataset.unique():
        continue
    for b in spec["baselines"]:
        for h in spec["horizons"]:
            if b.startswith("snaive") and h % spec["season"] == 0:
                continue
            expected.append((ds, b, h))
got = set(map(tuple, R[["dataset", "baseline", "h"]].drop_duplicates().values))
missing = [e for e in expected if e not in got]
extra = [g for g in got if g not in expected]
check("C1 all planned configs present", not missing, f"missing={missing}" if missing else f"{len(expected)} configs")
check("C1b no unplanned configs", not extra, f"extra={extra}" if extra else "")

seed_counts = R.groupby(["dataset", "baseline", "h"]).seed.nunique()
bad = seed_counts[seed_counts != N_SEEDS]
check(f"C2 every config has {N_SEEDS} seeds", bad.empty, "" if bad.empty else bad.to_dict())

rung_counts = R.groupby(["dataset", "baseline", "h", "seed"]).rung.nunique()
bad = rung_counts[rung_counts != len(RUNGS)]
check(f"C3 every (config,seed) has {len(RUNGS)} rungs", bad.empty,
      "" if bad.empty else f"{len(bad)} offenders, e.g. {list(bad.items())[:3]}")
missing_rungs = sorted(set(RUNGS) - set(R.rung.unique()))
check("C3b rung vocabulary complete", not missing_rungs, f"missing={missing_rungs}" if missing_rungs else "")

dups = R.duplicated(subset=["dataset", "baseline", "h", "seed", "rung"]).sum()
check("C4 no duplicate (config,seed,rung) rows", dups == 0, f"{dups} duplicates")

sn = R[R.baseline.str.startswith("snaive")]
bad_sn = sn[sn.h % PLANNED["india_daily"]["season"] == 0]
check("C5 no seasonal-naive at h multiple of season (D3)", bad_sn.empty,
      "" if bad_sn.empty else f"{len(bad_sn)} rows")

z = R[R.rung == "rung0_none"]
worst = max(abs(z[m]).max() for m in METRICS) if len(z) else np.nan
check("C6 rung0 skill is exactly zero in all metrics", worst == 0, f"max |skill|={worst:.2e}")

g = R.groupby(["dataset", "baseline", "h", "seed"])[["base_mae", "base_mse"]].nunique()
bad = g[(g.base_mae > 1) | (g.base_mse > 1)]
check("C7 baseline magnitude constant across rungs", bad.empty, "" if bad.empty else f"{len(bad)} offenders")

g = R.groupby(["dataset", "baseline", "h"])[["n_test", "n_val", "n_series"]].nunique()
bad = g[(g.n_test > 1) | (g.n_val > 1) | (g.n_series > 1)]
check("C8 data counts identical across seeds", bad.empty, "" if bad.empty else f"{len(bad)} offenders")

ok_range = R.frac_clipped.between(0, 1).all()
r0 = R[R.rung == "rung0_none"].frac_clipped.abs().max()
check("C9 clipped fraction valid; rung0 unclipped; CLIP declared",
      ok_range and r0 == 0 and CLIP == 1.5, f"range_ok={ok_range} rung0={r0} clip={CLIP}")
cl = R[R.rung != "rung0_none"].groupby("rung").frac_clipped.mean().round(4)
print("        clipped fraction by rung:", cl.to_dict())

fin = all(np.isfinite(R[m]).all() for m in METRICS)
same = all(np.allclose(R[a], R[b]) for a, b in [("skill_sq_macro", "skill_sq_pooled"),
                                                 ("skill_mae_macro", "skill_mae_pooled")])
check("C10 both aggregations present, finite, not identical", fin and not same,
      f"finite={fin} macro==pooled everywhere={same}")

if B is None:
    check("C11 block-bootstrap file present", False, "missing")
else:
    need = {(d, b, h, r) for (d, b, h) in expected
            for r in ["rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]}
    have = set(map(tuple, B[["dataset", "baseline", "h", "rung"]].values))
    miss = need - have
    brackets = ((B.lo <= B["diff"]) & (B["diff"] <= B.hi)).all()
    check("C11 bootstrap covers all configs and brackets point estimates",
          not miss and brackets, f"missing={len(miss)} brackets_ok={brackets}")

check("C12 canonical definitions declared",
      bool(D.get("unit")) and D.get("primary_metric") == "skill_sq_macro",
      f"unit={D.get('unit')} primary={D.get('primary_metric')}")

n_fail = sum(1 for _, ok, _ in results if not ok)
print("=" * 70)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
pd.DataFrame(results, columns=["check", "passed", "detail"]).to_csv(
    os.path.join(OUT, f"M1_integrity_{WHICH}.csv"), index=False)
sys.exit(1 if n_fail else 0)
