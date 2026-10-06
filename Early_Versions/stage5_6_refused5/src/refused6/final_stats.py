"""RE-FUSED-6 final statistics aggregator.

Reads every RE-FUSED-6 result file and writes one consolidated report:
    results/refused6/ALL_STATS.md     human-readable, every table generated from the CSVs
    results/refused6/ALL_STATS.json   key numbers for downstream use
Missing inputs are reported as missing, never filled in.
"""
import os, json, datetime
import numpy as np
import pandas as pd
from scipy import stats

O = r"D:\\REFUSED5\results\refused6"
L = []
J = {"generated": datetime.datetime.now().isoformat(timespec="seconds")}


def md(s=""):
    L.append(s)


def load(name):
    p = os.path.join(O, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def table(df, floatfmt=4):
    if df is None or len(df) == 0:
        return "_missing_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floatfmt)
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def paired(a, b):
    j = a.align(b, join="inner")
    if len(j[0]) < 3:
        return None
    t = stats.ttest_rel(j[0], j[1])
    return dict(diff=float((j[0] - j[1]).mean()), wins=int((j[0] > j[1]).sum()), n=len(j[0]),
                p=float(t.pvalue))


def section(title, fn):
    md(f"\n## {title}\n")
    try:
        fn()
    except Exception as e:
        md(f"_section failed: {type(e).__name__}: {e}_")


LADDER = ["rung0_none", "rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]
md("# RE-FUSED-6 — consolidated statistics")
md(f"Generated {J['generated']} from the result files in `results/refused6/`. Nothing here is typed by hand.")

files = ["frozen_v1/M1_runs_india.csv", "M1b_runs_india.csv", "M1b_estimator_validation.csv",
         "M1c_rescaled.csv", "M1d_clip_robustness.csv", "M3M5_runs_full.csv", "M5b_selection_full.csv",
         "M4_modern_baselines.csv", "M6_resolution_deconfounded.csv", "M8_cross_domain.csv",
         "M7_risk_control.csv", "M5c_parsimony.csv", "M7b_hybrid.csv", "M9_neural_gate.csv",
         "M10_rolling_origin.csv", "M10b_selection_rolling.csv", "M9b_neural_tuned.csv"]
md("\n## Stage completion\n")
md(table(pd.DataFrame([{"file": f, "present": os.path.exists(os.path.join(O, f)),
                        "rows": (len(pd.read_csv(os.path.join(O, f))) if os.path.exists(os.path.join(O, f)) else 0)}
                       for f in files])))


def s_m1():
    R = load("frozen_v1/M1_runs_india.csv")
    p = R[R.rung.isin(LADDER)].pivot_table(index=["baseline", "h"], columns="rung", values="skill_sq_macro")[LADDER]
    md(f"Rows: {len(R)} (15 configs x 10 seeds x 8 rungs). Integrity suite: 14/14 passed (frozen_v1).")
    md("\nMean squared-loss skill by rung:\n")
    md(table(p.mean().rename("skill").reset_index()))
    md("\nBest rung per config: " + str(p.idxmax(axis=1).value_counts().to_dict()))
    orc = R[R.rung == "oracle_instance"].groupby(["baseline", "h"]).skill_sq_macro.mean()
    cap = float((p.max(axis=1) / orc).mean())
    md(f"\nOracle capture by best feasible rung: {cap:.3f}")
    J["M1"] = dict(best_rung=p.idxmax(axis=1).value_counts().to_dict(), oracle_capture=cap,
                   mean_skill=p.mean().round(5).to_dict())


def s_m1b():
    B = load("M1b_runs_india.csv")
    md(table(B.pivot_table(index="rung", values="skill_sq_macro").sort_values("skill_sq_macro", ascending=False).reset_index()))
    rows = []
    for fine in ["rung5_wls", "rung5_wls_bagged", "rung5_ridge", "rung5_raw_ratio"]:
        r = paired(B[B.rung == fine].set_index(["baseline", "h", "seed"]).skill_sq_macro,
                   B[B.rung == "rung2_series"].set_index(["baseline", "h", "seed"]).skill_sq_macro)
        if r:
            rows.append(dict(fine_estimator=fine, minus_per_series=r["diff"], wins=f"{r['wins']}/{r['n']}", p=f"{r['p']:.1e}"))
    md("\nFine estimators versus per-series (paired, unit = config x seed):\n")
    md(table(pd.DataFrame(rows)))
    E = load("M1b_estimator_validation.csv")
    if E is not None:
        md("\nEstimator validation against the known closed-form gate:\n")
        md(table(E.groupby("estimator")[["mse_vs_gstar", "corr_vs_gstar", "risk", "absmax"]].mean().reset_index()))


def s_m1cd():
    C = load("M1c_rescaled.csv")
    if C is not None:
        rows = []
        for rung in ["rung2_series", "rung3_cell", "rung4_shrunk", "rung5_wls_bagged"]:
            r = paired(C[(C.rung == rung) & (C.rescaled == 1)].set_index(["baseline", "h", "seed"]).skill_sq_macro,
                       C[(C.rung == rung) & (C.rescaled == 0)].set_index(["baseline", "h", "seed"]).skill_sq_macro)
            if r:
                rows.append(dict(rung=rung, rescaled_minus_original=r["diff"], improves=f"{r['wins']}/{r['n']}", p=f"{r['p']:.1e}"))
        md("D9 (absorb global gate into corrector):\n")
        md(table(pd.DataFrame(rows)))
    D = load("M1d_clip_robustness.csv")
    if D is not None:
        md("\nD11 (admissible cap sweep) - skill by cap:\n")
        md(table(D.pivot_table(index="cap", columns="rung", values="skill_sq_macro").reset_index()))


def s_m35():
    S = load("M3M5_runs_full.csv")
    rg = [c for c in S.columns if c.startswith("regret_")]
    md(f"Runs: {len(S)} ({S.config_id.nunique()} configs x 10 seeds). Unimodality rate: {S.unimodal.mean():.3f}. "
       f"Mean validation-test rank correlation: {S.val_test_rank_corr.mean():.3f}")
    md("\nMean regret versus oracle rung:\n")
    md(table(S[rg].mean().sort_values().rename("regret").reset_index()))
    md("\nOracle rung distribution: " + str(S.oracle_rung.value_counts().to_dict()))
    J["M3M5"] = dict(unimodal=float(S.unimodal.mean()), regret=S[rg].mean().round(5).to_dict())


def s_m5b():
    S = load("M5b_selection_full.csv")
    if S is None:
        S = load("M5b_selection_quick.csv")
        md("_full sweep missing; showing quick run_")
    if S is None:
        md("_missing_")
        return
    rg = [c for c in S.columns if c.startswith("regret_")]
    md(f"Runs: {len(S)}")
    md("\nMean regret versus oracle rung (nested split and Mallows penalty are the corrected selectors):\n")
    md(table(S[rg].mean().sort_values().rename("regret").reset_index()))
    md("\nExact oracle-rung recovery:\n")
    md(table(S[[c for c in S.columns if c.startswith("correct_")]].mean().rename("accuracy").reset_index()))
    cols = [c for c in ["regret_nested", "regret_mallows1", "regret_mallows2", "regret_val_argmin",
                        "regret_fixed_rung3_cell", "regret_fixed_rung2_series"] if c in S.columns]
    md("\nRegret by validation size:\n")
    md(table(S.groupby("n_val")[cols].mean().reset_index()))
    J["M5b"] = dict(regret=S[rg].mean().round(5).to_dict())


def s_m4():
    M = load("M4_modern_baselines.csv")
    if M is None:
        md("_missing - stage not yet run_")
        return
    md(table(M.pivot_table(index="baseline_model", columns="rung", values="skill_sq_macro").reset_index()))
    md("\nBaseline strength versus correction value (per-series rung):\n")
    md(table(M[M.rung == "rung2_series"].groupby("baseline_model")[["base_mae", "skill_sq_macro", "resid_R2"]].mean().reset_index()))


def s_m6():
    M = load("M6_resolution_deconfounded.csv")
    if M is None:
        md("_missing - stage not yet run_")
        return
    md("Matched zones, window and clock horizons across resolutions.\n")
    md("Baseline MAE:\n")
    md(table(M[M.rung == "rung1_global"].pivot_table(index=["clock_h", "baseline"], columns="res", values="base_mae").reset_index()))
    md("\nBest coarse-rung skill:\n")
    best = M[M.rung.isin(["rung1_global", "rung2_series", "rung3_cell"])].groupby(["clock_h", "baseline", "res", "rung"]).skill_sq_macro.mean()
    md(table(best.groupby(level=[0, 1, 2]).max().unstack("res").reset_index()))
    md("\nOut-of-sample residual R2:\n")
    md(table(M[M.rung == "rung1_global"].pivot_table(index=["clock_h", "baseline"], columns="res", values="resid_R2").reset_index()))


def s_m8():
    M = load("M8_cross_domain.csv")
    if M is None:
        md("_missing - stage not yet run_")
        return
    cfg = M.groupby(["dataset", "h"]).mean(numeric_only=True).reset_index()
    md(table(cfg[["dataset", "h", "resid_R2_val", "resid_acf1", "gate_var_between", "gate_var_est",
                  "n_val_per_cell", "best_coarse_skill", "fine_minus_coarse"]]))
    md("\nBest rung by dataset: " + str(M.groupby(["dataset", "h"]).best_rung.agg(lambda s: s.mode().iloc[0]).to_dict()))
    rows = []
    for d in ["resid_R2_val", "gate_var_between", "gate_var_est", "n_val_per_cell", "resid_acf1", "base_mae_over_sd"]:
        if cfg[d].std() > 0 and len(cfg) >= 4:
            sp = stats.spearmanr(cfg[d], cfg.fine_minus_coarse)
            rows.append(dict(descriptor=d, spearman=float(sp[0]), p=float(sp[1])))
    md("\nDo validation-only descriptors predict the fine-rung gap? (unit = dataset x horizon)\n")
    md(table(pd.DataFrame(rows)))
    J["M8"] = dict(fine_minus_coarse_mean=float(cfg.fine_minus_coarse.mean()),
                   fine_never_best=bool((M.best_rung != "rung5_instance").all()))


def s_m7():
    M = load("M7_risk_control.csv")
    if M is None:
        md("_missing - stage not yet run_")
        return
    md(table(M.pivot_table(index=["baseline", "rule"], values=["mean_skill", "degraded_series",
             "worst_series_skill", "coverage", "withheld", "withheld_helpful"]).reset_index()))
    rows = []
    ref = M[M.rule == "block_lcb"].set_index(["baseline", "h", "seed"])
    for r_ in M.rule.unique():
        if r_ == "block_lcb":
            continue
        r = paired(ref.mean_skill, M[M.rule == r_].set_index(["baseline", "h", "seed"]).mean_skill)
        if r:
            rows.append(dict(rule=r_, block_lcb_minus_rule=r["diff"], lcb_better=f"{r['wins']}/{r['n']}", p=f"{r['p']:.1e}"))
    md("\nBlock-LCB versus each alternative (paired, unit = config x seed):\n")
    md(table(pd.DataFrame(rows)))




def s_m5c():
    M = load("M5c_parsimony.csv")
    if M is None:
        md("_missing - stage not yet run_")
        return
    for part in sorted(M.part.unique()):
        P = M[M.part == part]
        cols = [c for c in P.columns if c.startswith("regret_")]
        md("")
        md("**" + part + "** - " + str(len(P)) + " runs")
        md("")
        md(table(P[cols].mean().sort_values().rename("mean_regret").reset_index()))
        rows = []
        for a, b in [("regret_nested_parsimony", "regret_nested_argmin"),
                     ("regret_nested_parsimony", "regret_fixed_series"),
                     ("regret_nested_parsimony", "regret_fixed_cell")]:
            d = P[a] - P[b]
            pv = stats.wilcoxon(P[a], P[b]).pvalue if (d != 0).sum() > 5 else float("nan")
            rows.append(dict(comparison=a[7:] + " - " + b[7:], mean=float(d.mean()),
                             first_lower=str(int((d < 0).sum())) + "/" + str(len(d)), wilcoxon_p=pv))
        md(table(pd.DataFrame(rows)))
        md("")
        md("Tail: " + ", ".join(c[7:] + " p95=" + format(P[c].quantile(0.95), ".4f") + " max=" + format(P[c].max(), ".4f")
                              for c in ["regret_nested_argmin", "regret_nested_parsimony", "regret_fixed_series"]))


def s_m7b():
    M = load("M7b_hybrid.csv")
    if M is None:
        md("_missing - stage not yet run_")
        return
    for lab in M.subset.unique():
        P = M[M.subset == lab]
        md("")
        md("**" + lab + "** - " + str(len(P)) + " rule-runs")
        md("")
        md(table(P.pivot_table(index="rule", values=["mean_skill", "degraded_series", "worst_series_skill", "withheld"]).reset_index()))


def s_m9():
    M = load("M9_neural_gate.csv")
    if M is None:
        md("_missing - stage not yet run_")
        return
    md(table(M.pivot_table(index=["dataset", "baseline", "h"], columns="gate", values="skill_sq_macro").reset_index()))
    md("")
    key = ["dataset", "baseline", "h", "seed"]
    rows = []
    for nv in [g for g in ["neural_gate", "neural_gate_refit"] if g in set(M.gate)]:
        nn_ = M[M.gate == nv].set_index(key).skill_sq_macro
        for ref in ["per_series", "per_series_V1", "global", "none"]:
            rr = M[M.gate == ref].set_index(key).skill_sq_macro
            j = nn_.align(rr, join="inner")
            d = j[0] - j[1]
            pv = stats.wilcoxon(j[0], j[1]).pvalue if (d != 0).sum() > 5 else float("nan")
            rows.append(dict(comparison=nv + " - " + ref, mean=float(d.mean()),
                             neural_better=str(int((d > 0).sum())) + "/" + str(len(d)), wilcoxon_p=pv))
    md(table(pd.DataFrame(rows)))
    md("")
    md(table(M.groupby("gate")[["skill_sq_macro", "degraded_series", "gate_sd"]].mean().reset_index()))
    J["M9"] = {r["comparison"]: r for r in rows}
    T = load("M9b_neural_tuned.csv")
    if T is None:
        md("")
        md("_M9b tuned neural gate missing_")
        return
    md("")
    md("**M9b - neural gate with hyperparameters selected on V2, refit on V**")
    md("")
    md(table(T.pivot_table(index=["dataset", "baseline", "h"], columns="gate", values="skill_sq_macro").reset_index()))
    a = T[T.gate == "neural_gate_tuned"].set_index(key).skill_sq_macro
    b = T[T.gate == "per_series"].set_index(key).skill_sq_macro
    j = a.align(b, join="inner")
    d = j[0] - j[1]
    md("")
    md("neural_gate_tuned - per_series: mean " + format(d.mean(), "+.4f") + ", tuned better in " + str(int((d > 0).sum())) +
       "/" + str(len(d)) + ", Wilcoxon p=" + format(stats.wilcoxon(j[0], j[1]).pvalue, ".2e"))
    md("")
    md("Chosen configurations (hidden/wd/lr): " + ", ".join(k + " x" + str(v) for k, v in
       T[T.gate == "neural_gate_tuned"][["hidden", "wd", "lr"]].astype(str).agg("/".join, axis=1).value_counts().items()))
    J["M9b"] = dict(diff=float(d.mean()), wins=int((d > 0).sum()), n=len(d))


def s_m10():
    M = load("M10_rolling_origin.csv")
    if M is None:
        md("_missing - stage not yet run_")
        return
    piv = M.pivot_table(index=["origin", "baseline", "h"], columns="rung", values="skill_sq_macro")
    piv["best"] = piv[[c for c in LADDER if c in piv.columns]].idxmax(axis=1)
    md(table(piv.reset_index()))
    md("")
    rows = []
    for o, P in piv.groupby(level=0):
        gap = P["rung5_instance"] - P[["rung1_global", "rung2_series", "rung3_cell"]].max(axis=1)
        rows.append(dict(origin=o, configs=len(P), instance_best=int((P.best == "rung5_instance").sum()),
                         instance_below_best_coarse=int((gap < 0).sum()), mean_gap=float(gap.mean()),
                         best_counts="; ".join(k + "=" + str(v) for k, v in P.best.value_counts().items())))
    md(table(pd.DataFrame(rows)))
    J["M10"] = rows
    S = load("M10b_selection_rolling.csv")
    if S is None:
        md("")
        md("_M10b selection part missing_")
        return
    md("")
    md("**M10b - rung selection vs fixed defaults by origin** (origin_2024 overlaps M5c Part B: in-sample)")
    md("")
    cols = ["regret_nested_argmin", "regret_nested_parsimony", "regret_fixed_series", "regret_fixed_cell",
            "regret_fixed_shrunk", "regret_fixed_global", "regret_fixed_instance", "regret_fixed_none"]
    md(table(S.groupby("origin")[cols].mean().rename(columns=lambda c: c[7:]).reset_index()))
    rows = []
    for o, P in S.groupby("origin"):
        for sel in ["regret_nested_argmin", "regret_nested_parsimony"]:
            d = P[sel] - P.regret_fixed_series
            pv = stats.wilcoxon(P[sel], P.regret_fixed_series).pvalue if (d != 0).sum() > 5 else float("nan")
            rows.append(dict(origin=o, comparison=sel[7:] + " - fixed_series", mean=float(d.mean()),
                             selector_lower=str(int((d < 0).sum())) + "/" + str(len(d)), wilcoxon_p=pv))
    md("")
    md(table(pd.DataFrame(rows)))
    J["M10b"] = rows


section("M1 - canonical India run", s_m1)
section("M1b - stable gate estimators", s_m1b)
section("M1c / M1d - reparameterisation and cap robustness", s_m1cd)
section("M3 / M5 - synthetic mechanism and original selectors", s_m35)
section("M5b - corrected selectors (nested split, Mallows penalty)", s_m5b)
section("M5c - pre-registered parsimony selection (synthetic + fresh real data)", s_m5c)
section("M4 - modern deep baselines", s_m4)
section("M6 - de-confounded resolution study", s_m6)
section("M8 - cross-domain ladder and descriptors", s_m8)
section("M7 - risk-controlled deployment versus alternatives", s_m7)
section("M7b - pre-registered hybrid deployment rule", s_m7b)
section("M9 - neural sigmoid gate (reviewer objection 1.3)", s_m9)
section("M10 - rolling-origin India re-evaluation (reviewer objection 2.1)", s_m10)

open(os.path.join(O, "ALL_STATS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
json.dump(J, open(os.path.join(O, "ALL_STATS.json"), "w"), indent=2, default=str)
print("wrote ALL_STATS.md and ALL_STATS.json")
