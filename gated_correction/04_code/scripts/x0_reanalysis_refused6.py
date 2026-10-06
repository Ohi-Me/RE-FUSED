"""X0: re-analysis of RE-FUSED-6 headline claims at the correct inferential unit (audit S1, S2).

RE-FUSED-6 stored run-level skill (config x seed), not per-row losses, so a time-block bootstrap is impossible on
these files; the re-analysis therefore (i) averages seeds within config, (ii) reports sign counts and exact
binomial tests at the config and at the dataset level, and (iii) quantifies how much the run-level p-values
overstated the evidence. Per-row block-bootstrap inference is produced by the RE-FUSED engine re-runs (Dev-1).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from refused_gate.paths import PRIOR, RES  # noqa: E402

P = os.path.join(PRIOR, "results_refused6")
OUT = os.path.join(RES, "x0")
os.makedirs(OUT, exist_ok=True)
rows = []


def add(claim, unit, n, wins, mean_diff, p_run=None, note=""):
    p_unit = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue if n else np.nan
    rows.append(dict(claim=claim, unit=unit, n_units=n, units_supporting=wins, mean_diff=mean_diff,
                     p_unit_sign=p_unit, p_run_level_refused6=p_run, note=note))


# M1: per-series vs per-instance on India
m1 = pd.read_csv(os.path.join(P, "frozen_v1", "M1_runs_india.csv"))
k = ["baseline", "h", "seed"]
d = (m1[m1.rung == "rung2_series"].set_index(k).skill_sq_macro - m1[m1.rung == "rung5_instance"].set_index(k).skill_sq_macro)
dc = d.groupby(level=[0, 1]).mean()
db = dc.groupby(level=0).mean()
add("M1 per-series > per-instance (India)", "config (baseline x h)", len(dc), int((dc > 0).sum()), float(dc.mean()),
    stats.ttest_1samp(d, 0).pvalue, "configs share one dataset and test period")
add("M1 per-series > per-instance (India)", "baseline", len(db), int((db > 0).sum()), float(db.mean()), None,
    "4 baselines on the same series")
# M1: per-series vs shrunk (RE-FUSED-5 reversal)
d2 = (m1[m1.rung == "rung2_series"].set_index(k).skill_sq_macro - m1[m1.rung == "rung4_shrunk"].set_index(k).skill_sq_macro)
dc2 = d2.groupby(level=[0, 1]).mean()
add("M1 per-series > shrunk (India 2024-25)", "config", len(dc2), int((dc2 > 0).sum()), float(dc2.mean()),
    stats.ttest_1samp(d2, 0).pvalue, "M10 later showed this ordering is period-specific")

# M8: fine vs best coarse across datasets
m8 = pd.read_csv(os.path.join(P, "M8_cross_domain.csv"))
if "rung" in m8.columns:
    piv = m8.pivot_table(index=["dataset", "h"], columns="rung", values="skill_sq_macro")
    fine = [c for c in piv.columns if "instance" in c or "wls" in c]
    coarse = [c for c in piv.columns if c in ("rung1_global", "rung2_series", "rung3_cell")]
    gap = piv[fine[0]] - piv[coarse].max(axis=1)
    add("M8 fine < best coarse", "config (dataset x h)", len(gap), int((gap < 0).sum()), float(gap.mean()))
    gd = gap.groupby(level=0).mean()
    add("M8 fine < best coarse", "dataset (mean over h)", len(gd), int((gd < 0).sum()), float(gd.mean()),
        None, "ETTh1/h2/m1/m2 are four files from one physical system family")
else:
    cols = [c for c in m8.columns if "fine_minus" in c]
    g = m8.groupby(["dataset", "h"])[cols[0]].mean()
    add("M8 fine < best coarse", "config (dataset x h)", len(g), int((g < 0).sum()), float(g.mean()))
    gd = g.groupby(level=0).mean()
    add("M8 fine < best coarse", "dataset (mean over h)", len(gd), int((gd < 0).sum()), float(gd.mean()))

# M10: rolling origins
m10 = pd.read_csv(os.path.join(P, "M10_rolling_origin.csv"))
piv = m10.pivot_table(index=["origin", "baseline", "h"], columns="rung", values="skill_sq_macro")
gap = piv["rung5_instance"] - piv[["rung1_global", "rung2_series", "rung3_cell"]].max(axis=1)
add("M10 instance < best coarse", "config (origin x baseline x h)", len(gap), int((gap < 0).sum()), float(gap.mean()))
go = gap.groupby(level=0).mean()
add("M10 instance < best coarse", "origin (test year)", len(go), int((go < 0).sum()), float(go.mean()), None,
    "3 test years of one dataset")

# M9 / M9b: neural gate vs per-series
for f, nv in (("M9_neural_gate.csv", "neural_gate_refit"), ("M9b_neural_tuned.csv", "neural_gate_tuned")):
    m = pd.read_csv(os.path.join(P, f))
    kk = ["dataset", "baseline", "h", "seed"]
    dd = (m[m.gate == "per_series"].set_index(kk).skill_sq_macro - m[m.gate == nv].set_index(kk).skill_sq_macro)
    dcc = dd.groupby(level=[0, 1, 2]).mean()
    add(f"{f[:3]} per-series > {nv}", "config", len(dcc), int((dcc > 0).sum()), float(dcc.mean()),
        stats.wilcoxon(dd).pvalue)
    dds = dcc.groupby(level=0).mean()
    add(f"{f[:3]} per-series > {nv}", "dataset", len(dds), int((dds > 0).sum()), float(dds.mean()))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "x0_unit_level_reanalysis.csv"), index=False)
with open(os.path.join(OUT, "x0_unit_level_reanalysis.md"), "w", encoding="utf-8") as fh:
    fh.write("# X0 - RE-FUSED-6 claims at the correct inferential unit\n\n")
    fh.write("| claim | unit | units | supporting | mean diff | p (sign test, unit level) | p (RE-FUSED-6 run level) | note |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        pr = "" if r["p_run_level_refused6"] is None else "%.1e" % r["p_run_level_refused6"]
        fh.write(f"| {r['claim']} | {r['unit']} | {r['n_units']} | {r['units_supporting']} | {r['mean_diff']:+.4f} | "
                 f"{r['p_unit_sign']:.2g} | {pr} | {r['note']} |\n")
print(R.to_string())
