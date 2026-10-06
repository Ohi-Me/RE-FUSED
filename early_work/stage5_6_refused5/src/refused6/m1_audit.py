"""RE-FUSED-6 / M1 audit: which RE-FUSED-5 findings are confirmed, weakened, reversed, or invalidated.

Uses ONLY the canonical M1 outputs. Every statistic states its experimental unit.
  A. ladder shape        best rung per config; unimodality of the realised risk curve
  B. paired seed tests   rung contrasts within config (unit = seed), Holm-corrected
  C. block bootstrap     time-aware CIs for rung contrasts (unit = time block)
  D. metric sensitivity  does the ranking change between squared loss and MAE? macro vs pooled?
  E. crossing rule       does validation-measurable rho predict the realised best rung?
  F. clipping diagnostic  admissibility of each rung's gate estimator
"""
import os, json, itertools
import numpy as np
import pandas as pd
from scipy import stats

OUT = r"D:\\REFUSED5\results\refused6"
R = pd.read_csv(os.path.join(OUT, "M1_runs_india.csv"))
B = pd.read_csv(os.path.join(OUT, "M1_blockboot_india.csv"))
LADDER = ["rung0_none", "rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]
CFG = ["dataset", "baseline", "h"]
lines = []


def say(s=""):
    print(s)
    lines.append(s)


say("=" * 78)
say("M1 AUDIT (canonical India run: 15 configs x 10 seeds x 8 rungs = %d rows)" % len(R))
say("=" * 78)

# ---------------- A. ladder shape ----------------
say("\nA. LADDER SHAPE  (primary metric: squared-loss skill, macro over series)")
piv = R[R.rung.isin(LADDER)].pivot_table(index=CFG, columns="rung", values="skill_sq_macro")
piv = piv[LADDER]
best = piv.idxmax(axis=1)
say("\nbest rung per config:")
say(best.value_counts().to_string())
uni = []
for _, row in piv.iterrows():
    d = np.diff(row.values)
    s = np.sign(d)
    s = s[s != 0]
    uni.append(bool(np.all(np.diff(s) <= 0)))    # rises then falls = single sign change down
say("\nunimodal (rise-then-fall) configs: %d/%d" % (sum(uni), len(uni)))
say("\nmean skill by rung (over 15 configs):")
say(piv.mean().round(4).to_string())
say("\nfraction of oracle captured by the best feasible rung:")
orc = R[R.rung == "oracle_instance"].groupby(CFG).skill_sq_macro.mean()
frac = (piv.max(axis=1) / orc).round(3)
say("  mean %.3f  min %.3f  max %.3f" % (frac.mean(), frac.min(), frac.max()))

# ---------------- B. paired seed tests ----------------
say("\nB. PAIRED SEED TESTS  (unit = seed, within config; Holm correction across configs)")
contrasts = [("rung2_series", "rung4_shrunk"), ("rung2_series", "rung3_cell"),
             ("rung2_series", "rung5_instance"), ("rung3_cell", "rung4_shrunk"),
             ("rung4_shrunk", "rung5_instance"), ("rung2_series", "rung1_global")]
rows = []
for a, b in contrasts:
    pv, df = [], []
    for cfg, g in R[R.rung.isin([a, b])].groupby(CFG):
        x = g[g.rung == a].sort_values("seed").skill_sq_macro.values
        y = g[g.rung == b].sort_values("seed").skill_sq_macro.values
        if len(x) == len(y) == 10:
            t = stats.ttest_rel(x, y)
            pv.append(t.pvalue); df.append(float(np.mean(x - y)))
    pv = np.array(pv); order = np.argsort(pv)
    holm = np.empty_like(pv); m = len(pv)
    for rank, idx in enumerate(order):
        holm[idx] = min(1.0, pv[idx] * (m - rank))
    rows.append(dict(contrast=f"{a} - {b}", mean_diff=float(np.mean(df)),
                     configs_favouring_first=int(sum(np.array(df) > 0)), n_configs=m,
                     n_sig_holm=int((holm < 0.05).sum()),
                     n_sig_holm_favouring_first=int(((holm < 0.05) & (np.array(df) > 0)).sum())))
say(pd.DataFrame(rows).round(5).to_string(index=False))

# ---------------- C. block bootstrap ----------------
say("\nC. BLOCK BOOTSTRAP  (unit = 14-day time block; all contrasts vs rung3_cell)")
bb = B.copy()
bb["excludes_zero"] = (bb.lo > 0) | (bb.hi < 0)
say(bb.groupby("rung").agg(mean_diff=("diff", "mean"), mean_lo=("lo", "mean"), mean_hi=("hi", "mean"),
                           configs_CI_excludes_zero=("excludes_zero", "sum"),
                           n=("diff", "size")).round(4).to_string())

# ---------------- D. metric sensitivity ----------------
say("\nD. METRIC SENSITIVITY  (does the conclusion depend on the metric or aggregation?)")
for metric in ["skill_sq_macro", "skill_sq_pooled", "skill_mae_macro", "skill_mae_pooled"]:
    p2 = R[R.rung.isin(LADDER)].pivot_table(index=CFG, columns="rung", values=metric)[LADDER]
    bb2 = p2.idxmax(axis=1).value_counts()
    say("  %-18s best-rung counts: %s" % (metric, bb2.to_dict()))
say("\n  rung4_shrunk minus rung2_series, by metric (mean over configs):")
for metric in ["skill_sq_macro", "skill_sq_pooled", "skill_mae_macro", "skill_mae_pooled"]:
    p2 = R[R.rung.isin(LADDER)].pivot_table(index=CFG, columns="rung", values=metric)
    say("    %-18s %+.5f" % (metric, float((p2.rung4_shrunk - p2.rung2_series).mean())))

# ---------------- E. crossing rule ----------------
say("\nE. CROSSING RULE  (is validation-measurable rho predictive of the realised best rung?)")
diag = R[R.rung == "rung4_shrunk"].groupby(CFG).agg(rho=("var_within", "mean"), V=("var_est", "mean"),
                                                    lam=("lam", "mean"))
diag["rho"] = diag.rho / diag.V
tab = pd.concat([diag, best.rename("best_rung"), (piv.rung5_instance - piv.rung3_cell).rename("fine_minus_coarse")], axis=1)
say(tab.round(3).to_string())
say("\n  rho < 1 in %d/%d configs; fine rung (rung5) never best: %s"
    % (int((tab.rho < 1).sum()), len(tab), bool((tab.best_rung != "rung5_instance").all())))
sp = stats.spearmanr(tab.rho, tab.fine_minus_coarse)
say("  spearman(rho, rung5 - rung3) = %+.3f (p=%.4f)  <- tests the RANKING claim" % (sp[0], sp[1]))

# ---------------- F. clipping diagnostic ----------------
say("\nF. GATE ADMISSIBILITY  (fraction of gate values outside [0, 1.5] before clipping)")
say(R.groupby("rung").frac_clipped.mean().round(4).to_string())

json.dump(dict(best_rung_counts=best.value_counts().to_dict(),
               unimodal=f"{sum(uni)}/{len(uni)}",
               mean_skill_by_rung=piv.mean().round(5).to_dict(),
               frac_oracle=float(frac.mean())),
          open(os.path.join(OUT, "M1_audit_summary.json"), "w"), indent=2)
open(os.path.join(OUT, "M1_audit_report.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\nwritten: M1_audit_report.txt / M1_audit_summary.json")
