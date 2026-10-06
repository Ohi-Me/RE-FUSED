"""Score every pre-registered hypothesis (codes/design/02_preregistration.md, v2) from the result files.

This script is part of the frozen pre-registration: its SHA-256 is written into the pre-registration before the
reserved period is opened, so the decision rules cannot move after the results are seen. It only reads files.

Round-1 results come from results/<phase> and round-2 results from results/<phase>2. In the development phase the
same script gives a rehearsal table (development evidence only); in the confirmatory phase it gives the answer.

Outputs: results/<phase>2/hypotheses.csv and results/tables/<phase>_hypotheses.md
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import TAB  # noqa: E402
from refused.report import md_table  # noqa: E402

R1, R2 = D.DEV1, D.DEV2
TIDS = list(O.TARGETS)
ROWS = []


def read(*parts):
    f = os.path.join(*parts)
    return pd.read_csv(f) if os.path.exists(f) else None


def add(hid, family, what, estimate=np.nan, lo=np.nan, hi=np.nan, p=np.nan, rule_ok=None, primary=True, note=""):
    """rule_ok: extra condition that must also hold (a CI side, all seeds, a coverage band); None if only p counts."""
    ROWS.append(dict(id=hid, family=family, what=what, estimate=estimate, lo=lo, hi=hi, p=p, rule_ok=rule_ok,
                     primary=primary and np.isfinite(p), note=note))


def pick(df, **eq):
    if df is None:
        return None
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        m &= df[k].astype(str).to_numpy() == str(v)
    g = df[m]
    return g.iloc[0] if len(g) else None


def family_f1():
    t = read(R2, "o2", "eval", "tests.csv")
    sd = read(R2, "o2", "eval", "seeds.csv")
    pr = read(R2, "o2", "eval", "prob.csv")
    lat = read(R1, "o2", "eval", "latency.csv")
    for i, tid in enumerate(TIDS):
        r = pick(t, tid=tid, test="final_vs_seasonal_naive", H="all")
        seeds = sd[sd.tid == tid] if sd is not None else pd.DataFrame()
        seeds_ok = len(seeds) == 5 and bool((seeds.p_vs_sn < 0.05).all())
        if r is not None:
            add(f"H1{'abcde'[i]}", "F1", f"{tid}: final forecaster beats seasonal naive (scaled |e|, H pooled)",
                r.mean_diff, r.lo, r.hi, r.p_one_sided, rule_ok=seeds_ok, note=f"seeds p<0.05: {int((seeds.p_vs_sn < 0.05).sum()) if len(seeds) else 0}/{len(seeds)}")
        r = pick(t, tid=tid, test="final_vs_round1_selected", H="all")
        if r is not None:
            add(f"H1f-{tid}", "F1", f"{tid}: final forecaster beats the round-1 selected model ({r.comparator})",
                r.mean_diff, r.lo, r.hi, r.p_one_sided)
        c = pr[(pr.tid == tid) & (pr.chosen.astype(str) == "True")] if pr is not None else pd.DataFrame()
        if len(c):
            c = c.iloc[0]
            ok = abs(c.cov80 - 0.8) <= 0.05 and abs(c.cov90 - 0.9) <= 0.05
            add(f"H2-{tid}", "F1", f"{tid}: 80 % / 90 % coverage within ±5 points", c.cov80, c.cov80, c.cov90,
                rule_ok=ok, primary=False, note=f"cov80 {c.cov80:.3f}, cov90 {c.cov90:.3f}")
        r = pick(lat, tid=tid, H="all")
        if r is not None:
            add(f"H3-{tid}", "F1", f"{tid}: publication-latency cost > 0 (LightGBM, % MASE)", r.latency_cost_pct,
                r.lo, r.hi, r.p_instant_better, note="interval: daily scaled-error difference, instant minus published")


def family_f2():
    p1 = read(R1, "o3", "eval", "pairs.csv")
    p2 = read(R2, "o3", "eval", "pairs.csv")
    l1 = read(R2, "o3", "eval", "l1.csv")
    for tid in TIDS:
        r = pick(p2, tid=tid, comparison="A2 sd: instance vs fusion", metric="pinball", H="all")
        if r is not None:
            add(f"H4-{tid}", "F2", f"{tid}: gated (sd) beats matched fusion (sd), pinball", r.mean_diff, r.lo, r.hi, r.p_a_better)
    for tid in ("T1", "T2", "T5"):
        a = pick(p1, tid=tid, comparison="A3 real vs null", metric="pinball", H="all")
        b = pick(p1, tid=tid, comparison="A3 real vs permuted", metric="pinball", H="all")
        if a is not None and b is not None:  # both must hold: the larger p decides
            add(f"H5-{tid}", "F2", f"{tid}: real context beats null and shuffled context, pinball", max(a.mean_diff, b.mean_diff),
                np.nan, max(a.hi, b.hi), max(a.p_a_better, b.p_a_better), note="larger of the two p values")
    for tid in TIDS:
        r = pick(p1, tid=tid, comparison="ladder OTG vs instance", metric="mae", H="all")
        if r is not None:
            add(f"H6-{tid}", "F2", f"{tid}: online time-adaptive gate lowers scaled |e|", r.mean_diff, r.lo, r.hi, r.p_a_better)
    if l1 is not None:
        for _, r in l1[l1.model.astype(str).str.contains("degradation")].iterrows():
            add(f"H7-{r.tid}", "F2", f"{r.tid}: gated model loses less than fusion when RE and weather go missing",
                r.mean_diff, r.lo, r.hi, r.p_instance_degrades_less, rule_ok=bool(r.hi < 0))
    r = pick(p1, tid="T3", comparison="A6 all vs no carbon", metric="pinball", H="all")
    if r is not None:
        add("H8-T3", "F2", "T3: carbon block improves pinball loss", r.mean_diff, r.lo, r.hi, r.p_a_better)
    for tid in ("T1", "T2"):
        r = pick(p1, tid=tid, comparison="A6 all vs no carbon", metric="pinball", H="all")
        if r is not None:
            margin = 0.01 * r.mean_b
            add(f"H8-{tid}", "F2", f"{tid}: carbon block does not worsen pinball (margin 1 %)", r.mean_diff, r.lo, r.hi,
                rule_ok=bool(r.hi < margin), primary=False, note=f"margin {margin:.4f}")


def family_f3():
    t = read(R2, "o4", "tests.csv")
    r = pick(t, test="A vs M0", outcome="Y")
    if r is not None:
        add("H9", "F3", "assessment A ranks consequences better than M0 within M0 deciles", r["diff"], r.lo, r.hi,
            r.p_one_sided, rule_ok=bool(r.lo > 0))
    r = pick(t, test="forecast information vs shuffled placebo", outcome="Y")
    if r is not None:
        add("H10", "F3", "forecast information beats the shuffled placebo (95th percentile)", r["diff"], np.nan, r.lo, r.p_one_sided)


def family_f4():
    t = read(R2, "o5", "tests.csv")
    for hid, a, b in (("H11a", "A4s", "A2"), ("H11b", "A4s", "H_pub"), ("H11c", "A4s", "P"), ("H11d", "A5", "A2"), ("H11e", "A5", "P")):
        r = pick(t, a=a, b=b, metric="tail days (either arm above its 90th pct)")
        if r is not None:
            add(hid, "F4", f"{a} has lower tail-day system regret than {b}", r.mean_diff, r.lo, r.hi, r.p_a_better,
                rule_ok=bool(r.hi < 0))
    pt = read(R2, "o5", "ppo_tests.csv")
    for hid, b in (("H11f", "A2"), ("H11g", "P")):
        r = pick(pt, b=b)
        if r is not None:
            add(hid, "F4", f"PPO (A1) has lower mean system regret than {b}", r.mean_diff, r.lo, r.hi, r.p_a_better)
    m = read(R2, "o5", "metrics.csv")
    if m is not None:
        dm = m[(m.split == "dev_test") & (m.arm != "H_final")]
        worst = float(dm.hard_limit_envelope_exceedance_share.max())
        # PPO decisions are checked against the same limits
        pf, df_ = os.path.join(R2, "o5", "ppo_decisions.parquet"), os.path.join(R2, "o5", "decisions.parquet")
        pm = read(R2, "o5", "ppo_metrics.csv")
        if os.path.exists(pf) and os.path.exists(df_) and pm is not None:
            seed = int(pm[pm.selected_on_validation.astype(str) == "True"].seed.iloc[0])
            pp = pd.read_parquet(pf)
            pp = pp[(pp.seed == seed) & (pp.split == "dev_test")].merge(
                pd.read_parquet(df_)[["entity", "date", "S_min", "S_max"]], on=["entity", "date"])
            if len(pp):
                worst = max(worst, float(((pp.S_A1 < pp.S_min - 1e-9) | (pp.S_A1 > pp.S_max + 1e-9)).mean()))
        add("H12", "F4", "no decision arm breaks the hard limits", worst, rule_ok=worst == 0.0, primary=False,
            note="largest share of State-days outside the limits over all arms")
    ct = read(R2, "o5", "chain_tests.csv")
    r = pick(ct, a="A2 final", b="A2 round1", metric="mean system regret")
    if r is not None:
        add("H14", "F4", "the chain with the final forecaster lowers mean system regret (arm A2) vs the round-1 forecaster",
            r.mean_diff, r.lo, r.hi, r.p_a_better)


def family_f5():
    pdx = read(R2, "o3", "eval", "part_decision.csv")
    if pdx is None:
        return
    for ref, g in pdx.dropna(subset=["realised_gain_dev"]).groupby("refinement"):
        k = int(((g.G_val > 0) == (g.realised_gain_dev > 0)).sum())
        p = st.binomtest(k, len(g), 0.5, alternative="greater").pvalue
        add(f"H13-{ref}", "F5", f"PART sign agreement > 1/2 ({ref})", k / len(g), p=p, note=f"{k}/{len(g)} cells")


def by_adjust(p):
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    c = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    adj = np.empty(m)
    run = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, p[i] * m * c / rank)
        adj[i] = run
    return np.minimum(adj, 1.0)


def main():
    for fam in (family_f1, family_f2, family_f3, family_f4, family_f5):
        fam()
    H = pd.DataFrame(ROWS)
    H["p_holm"] = np.nan
    for fam, g in H[H.primary].groupby("family"):
        H.loc[g.index, "p_holm"] = S.holm(g.p.to_numpy())
    prim = H.primary
    H.loc[prim, "p_by"] = by_adjust(H.loc[prim, "p"].to_numpy())

    def decide(r):
        if r.primary:
            ok = r.p_holm < 0.05 and (r.rule_ok is None or bool(r.rule_ok))
        else:
            ok = bool(r.rule_ok)
        return "supported" if ok else "not supported"
    H["decision"] = H.apply(decide, axis=1)
    H.to_csv(os.path.join(D.folder(), "hypotheses.csv"), index=False)
    phase = "confirmatory evaluation 1 Apr 2025 – 31 Aug 2026" if O.PHASE == "confirm" else \
        "development rehearsal (FY2024-25, already seen; not a test)"
    L = [f"# Pre-registered hypotheses — {phase}", "",
         "Holm within family, Benjamini–Yekutieli across all primary tests. A hypothesis is supported when its Holm-"
         "adjusted p < 0.05 and its extra rule holds (CI side, all seeds, coverage band, zero violations).", "",
         f"Supported: {int((H.decision == 'supported').sum())} of {len(H)}.", "",
         md_table(H[["id", "family", "what", "estimate", "lo", "hi", "p", "p_holm", "p_by", "decision", "note"]], 4), ""]
    open(os.path.join(TAB, f"{O.PHASE_DIR}_hypotheses.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
