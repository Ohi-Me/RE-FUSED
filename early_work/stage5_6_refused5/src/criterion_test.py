"""RE-FUSED-5 R7 (core claim test): is the identifiability ratio predictive?
Claim: conditional gating beats static gating iff rho = Var_within(g*) / Var(ghat_est) > 1,
where BOTH quantities are measured on validation only (no test access).
Data: synthetic sweep (S3) + real India configs (R1). Outcome: sign/magnitude prediction accuracy."""
import pandas as pd, numpy as np, json, os
from scipy import stats
OUT = r"D:\\REFUSED5\results"
rows = []
# ---- synthetic: S3 sweep (per-seed records carry var_within, var_est, and both skills) ----
f3j = os.path.join(OUT, "S3_cells.jsonl")
if os.path.exists(f3j):
    for line in open(f3j, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        for r in rec["runs"]:
            rows.append(dict(source="synthetic", config=rec["cell"],
                             var_within=r["var_within"], var_est=r["var_est"],
                             skill_static=r["static"], skill_cond=r["free"], skill_shrunk=r["shrunk"]))
f3 = os.path.join(OUT, "S3_granularity.json")
if os.path.exists(f3) and not rows:
    S = json.load(open(f3))
    for key, runs in S.items():
        for r in runs:
            rows.append(dict(source="synthetic", config=key,
                             var_within=r["var_within"], var_est=r["var_est"],
                             skill_static=r["static"], skill_cond=r["free"], skill_shrunk=r["shrunk"]))
# ---- real: India R1 (aggregate per baseline x horizon) ----
f1c, f1d = os.path.join(OUT, "R1_india_core.csv"), os.path.join(OUT, "R1_india_diag.csv")
if os.path.exists(f1c) and os.path.exists(f1d):
    core = pd.read_csv(f1c); diag = pd.read_csv(f1d)
    piv = core.pivot_table(index=["h","baseline"], columns="strategy", values="skill").reset_index()
    m = piv.merge(diag, on=["h","baseline"])
    for _, r in m.iterrows():
        rows.append(dict(source="india", config=f"{r.baseline}_h{int(r.h)}",
                         var_within=r.var_within, var_est=r.var_est,
                         skill_static=r.static_state, skill_cond=r.free, skill_shrunk=r.shrunk))
d = pd.DataFrame(rows)
if d.empty:
    raise SystemExit("no artifacts yet")
d["rho"] = d.var_within / d.var_est.replace(0, np.nan)
d["delta_cond_static"] = d.skill_cond - d.skill_static
d["delta_shrunk_static"] = d.skill_shrunk - d.skill_static
d = d.dropna(subset=["rho"])
d.to_csv(os.path.join(OUT, "R7_criterion.csv"), index=False)
print(f"configs: {len(d)} ({d.source.value_counts().to_dict()})")
print(f"rho range: {d.rho.min():.2f} .. {d.rho.max():.2f} (median {d.rho.median():.2f})")
for src in d.source.unique():
    s = d[d.source == src]
    pred = (s.rho > 1).astype(int); actual = (s.delta_cond_static > 0).astype(int)
    acc = float((pred == actual).mean())
    corr = float(np.corrcoef(s.rho, s.delta_cond_static)[0, 1]) if len(s) > 2 else np.nan
    sp = stats.spearmanr(s.rho, s.delta_cond_static) if len(s) > 2 else (np.nan, np.nan)
    print(f"\n--- {src}: n={len(s)}")
    print(f"  sign-prediction accuracy of (rho>1): {acc:.3f} | base rate of conditional winning: {actual.mean():.3f}")
    print(f"  corr(rho, skill_cond - skill_static) = {corr:+.3f} | spearman rho={sp[0]:+.3f} p={sp[1]:.4f}")
    print(f"  conditional wins in {int(actual.sum())}/{len(s)} configs; shrunk beats static in "
          f"{int((s.delta_shrunk_static>0).sum())}/{len(s)}")
    # decision-rule value: pick per config using the criterion, vs always-static / always-conditional
    pick = np.where(s.rho > 1, s.skill_cond, s.skill_static)
    print(f"  mean skill — always-static {s.skill_static.mean():.4f} | always-conditional {s.skill_cond.mean():.4f} "
          f"| criterion-selected {pick.mean():.4f} | shrunk {s.skill_shrunk.mean():.4f} "
          f"| oracle-selected {np.maximum(s.skill_static, s.skill_cond).mean():.4f}")
print("\n=== VERDICT (the claim survives only if the criterion is predictive AND shrunk >= static) ===")
allp = (d.rho > 1).astype(int); alla = (d.delta_cond_static > 0).astype(int)
print(f"pooled sign accuracy={float((allp==alla).mean()):.3f}  "
      f"pooled spearman={stats.spearmanr(d.rho, d.delta_cond_static)[0]:+.3f} "
      f"(p={stats.spearmanr(d.rho, d.delta_cond_static)[1]:.4f})")
t = stats.ttest_rel(d.skill_shrunk, d.skill_static)
print(f"shrunk vs static paired t={t.statistic:+.3f} p={t.pvalue:.4f} "
      f"mean diff={d.delta_shrunk_static.mean():+.5f}")
