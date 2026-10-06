"""dev2 step 10: collect the round-1 and round-2 evidence for N1-N12 into one table (results/NOVELTY_EVIDENCE.md).

It only reads result files. Numbers are copied, not recomputed, and a missing file shows as "not run".
Development-test evidence only; the confirmatory test is the reserved period.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import dev2 as D  # noqa: E402
from refused.paths import ROOT, RES  # noqa: E402

R1, R2 = D.DEV1, D.DEV2


def read_csv(*parts):
    f = os.path.join(*parts)
    return pd.read_csv(f) if os.path.exists(f) else None


def fmt(x, nd=3):
    try:
        return f"{float(x):+.{nd}f}" if abs(float(x)) < 1e4 else f"{float(x):.3g}"
    except (TypeError, ValueError):
        return str(x)


def pval(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "–"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def pairs_line(df, label, metric="mae"):
    if df is None:
        return "not run"
    g = df[(df.comparison == label) & (df.metric == metric) & (df.H.astype(str) == "all")]
    return "; ".join(f"{r.tid} {fmt(r.mean_diff, 4)} (p {pval(r.p_a_better)})" for r in g.itertuples()) or "not run"


def main():
    rows = []
    ck = open(os.path.join(ROOT, "data", "processed", "CHECKSUM.txt")).read().strip()
    ver = os.path.join(RES, "rebuild", "VERIFY.md")
    rows.append(("N1", "Official-data State-day panel with provenance", ck.split("content_sha256=")[1][:16] + "…",
                 ("rebuild identical" if os.path.exists(ver) and "identical" in open(ver).read() else "rebuild not checked"), "–"))
    lat = read_csv(R1, "o2", "eval", "latency.csv")
    rows.append(("N2", "Publication-latency cost (LightGBM, MASE)",
                 "; ".join(f"{r.tid} +{r.latency_cost_pct:.1f}% (p {pval(r.p_instant_better)})" for r in lat[lat.H.astype(str) == "all"].itertuples())
                 if lat is not None else "not run", "same data; not re-run", "–"))
    s1 = json.load(open(os.path.join(R1, "o2", "eval", "selection.json")))
    p1 = read_csv(R1, "o2", "eval", "point.csv")
    r1_txt = "; ".join(f"{s['tid']} {s['selected']} {p1[(p1.tid == s['tid']) & (p1.model == s['selected']) & (p1.split == 'dev_test') & p1.H.isna()].mase.iloc[0]:.3f}"
                       for s in s1)
    sel2 = os.path.join(R2, "o2", "eval", "selection_dev2.json")
    p2 = read_csv(R2, "o2", "eval", "point.csv")
    t2 = read_csv(R2, "o2", "eval", "tests.csv")
    if os.path.exists(sel2) and p2 is not None:
        s2 = json.load(open(sel2))
        parts = []
        for s in s2:
            m = p2[(p2.tid == s["tid"]) & (p2.model == s["final"]) & (p2.split == "dev_test")].mase.iloc[0]
            v = t2[(t2.tid == s["tid"]) & (t2.test == "final_vs_round1_selected") & (t2.H.astype(str) == "all")] if t2 is not None else None
            extra = f", vs round 1 {fmt(v.mean_diff.iloc[0], 3)} (p {pval(v.p_one_sided.iloc[0])})" if v is not None and len(v) else ""
            parts.append(f"{s['tid']} {s['final']} {m:.3f}{extra}")
        r2_txt = "; ".join(parts)
    else:
        r2_txt = "not run"
    rows.append(("N3", "Forecast accuracy: selected / final forecaster, dev MASE", r1_txt, r2_txt, "–"))
    pr1 = read_csv(R1, "o3", "eval", "pairs.csv")
    pr2 = read_csv(R2, "o3", "eval", "pairs.csv")
    rows.append(("N4", "LA-MCAG instance gates vs matched fusion (scaled MAE diff)", pairs_line(pr1, "A2 instance vs fusion"),
                 pairs_line(pr2, "A2 sd: instance vs fusion"), "–"))
    rows.append(("N5", "Online time-adaptive gate vs instance (scaled MAE diff)", pairs_line(pr1, "ladder OTG vs instance"), "not re-run", "–"))
    rows.append(("N6", "Real vs null context (pinball diff)", pairs_line(pr1, "A3 real vs null", "pinball"), "not re-run", "–"))
    l1a, l1b = read_csv(R1, "o3", "eval", "l1.csv"), read_csv(R2, "o3", "eval", "l1.csv")

    def l1_line(df):
        if df is None:
            return "not run"
        g = df[df.model.astype(str).str.contains("degradation")]
        return "; ".join(f"{r.tid} {fmt(r.mean_diff, 4)} [{fmt(r.lo, 4)}, {fmt(r.hi, 4)}]" for r in g.itertuples()) or "not run"
    rows.append(("N7", "Source loss: extra error of gated minus fusion (negative = gated holds up better)", l1_line(l1a), l1_line(l1b), "–"))
    rows.append(("N8", "Carbon block (GCAL): all vs no carbon (scaled MAE diff)", pairs_line(pr1, "A6 all vs no carbon"), "not re-run", "–"))
    f5 = read_csv(R1, "o3", "eval", "f5.csv")
    pd2 = read_csv(R2, "o3", "eval", "part_decision.csv")
    f5_txt = (f"sign agreement {int(f5.dropna(subset=['realised_gain_dev']).agree.sum())}/{len(f5.dropna(subset=['realised_gain_dev']))}"
              if f5 is not None else "not run")
    pd_txt = (f"gain collected: rule {pd2.realised_gain_dev[pd2.refine_rule].sum():+.4f}, always {pd2.realised_gain_dev.sum():+.4f}, "
              f"never 0; clear-cell agreement {int(((pd2.G_val > 0) == (pd2.realised_gain_dev > 0))[pd2.clear].sum())}/{int(pd2.clear.sum())}"
              if pd2 is not None else "not run")
    rows.append(("N9", "Price of adaptivity (PART)", f5_txt, pd_txt, "–"))
    o41, o42 = read_csv(R1, "o4", "tests.csv"), read_csv(R2, "o4", "tests.csv")

    def o4_line(df):
        if df is None:
            return "not run"
        a = df[(df.test.str.contains("vs M0")) & (df.outcome == "Y")]
        b = df[df.test.str.contains("placebo")]
        return (f"A vs M0 {fmt(a['diff'].iloc[0], 4)} (p {pval(a.p_one_sided.iloc[0])})" if len(a) else "") + \
            ("; placebo p " + ", ".join(pval(x) for x in b.p_one_sided) if len(b) else "")
    rows.append(("N10", "Context-aware deviation assessment", o4_line(o41), o4_line(o42), "–"))
    t51, t52 = read_csv(R1, "o5", "tests.csv"), read_csv(R2, "o5", "tests.csv")

    def o5_line(df, pairs):
        if df is None:
            return "not run"
        out = []
        for a, b in pairs:
            g = df[(df.a == a) & (df.b == b) & (df.metric == "mean system regret")]
            h = df[(df.a == a) & (df.b == b) & (df.metric != "mean system regret")]
            if len(g):
                out.append(f"{a}-{b} mean {fmt(g.mean_diff.iloc[0], 2)} (p {pval(g.p_a_better.iloc[0])}), tail {fmt(h.mean_diff.iloc[0], 2) if len(h) else '–'}")
        return "; ".join(out) or "not run"
    rows.append(("N11", "Risk-aware scheduling (crore per day, negative = better)", o5_line(t51, [("A4", "A2"), ("A2", "H_pub")]),
                 o5_line(t52, [("A5", "A2"), ("A4s", "A2"), ("A5", "H_pub")]), "–"))
    ch = read_csv(R2, "o5", "chain.csv")
    if ch is not None:
        c = ch[(ch.split == "dev_test") & ch.arm.isin(["A5", "A2", "P"])].pivot_table(index="arm", columns="source", values="J")
        ch_txt = "; ".join(f"{arm}: " + ", ".join(f"{src} J {c.loc[arm, src]:.2f}" for src in c.columns) for arm in c.index)
    else:
        ch_txt = "not run"
    rows.append(("N12", "Chain: better forecaster inside scheduling (dev J)", "not run", ch_txt, "reserved-period run pending"))
    L = ["# N1–N12 evidence (development data only)", "",
         "Round 1 = first H100 run; round 2 = protocol `codes/design/06_dev2_protocol.md`. All numbers are copied from the "
         "result files named in `NOVELTY_MAP.md`. Claims are tested later on the reserved period.", "",
         "| N | What is tested | Round 1 (dev test) | Round 2 (dev test) | Note |", "|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    name = "NOVELTY_EVIDENCE.md" if D.TAG == "dev2" else f"NOVELTY_EVIDENCE_{D.TAG}.md"
    open(os.path.join(RES, name), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
