"""Descriptive breakdowns of the confirmatory results by State, season and horizon.

These are not pre-registered tests and nothing here changes a decision: the 52 checks were scored once by
c01_hypotheses.py and stay as they are. This script only shows where the headline numbers come from, which a
reviewer will ask for: which States the forecaster does well on, whether the gain survives every season, and how
the scheduling regret is spread across States.

Outputs: results/audit/strata/{accuracy_state,accuracy_season,accuracy_horizon,regret_state}.csv
         results/paper/<phase>/tables/{t_state_accuracy,t_season_accuracy,t_state_regret}.tex
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from refused.paths import RES, PAPER  # noqa: E402

OUT = os.path.join(RES, "audit", "strata")
TAB = os.path.join(PAPER, O.PHASE_DIR, "tables")
TNAME = {t: O.TARGETS[t]["name"] for t in O.TARGETS}
ARMS = ["H_pub", "P", "A2", "A4s", "A5", "A5c"]


def final_models():
    f = os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json")
    return {s["tid"]: s["final"] for s in json.load(open(f))} if os.path.exists(f) else {}


def scored(tid, model):
    """Scaled absolute errors of one model on the evaluation split, with the State, season and horizon kept."""
    d = D.load_member(tid, model)
    if d is None:
        return None
    d = d[d.split == "dev_test"].merge(D.scales(tid), on=["sid", "H"])
    d = d[d.y.notna() & d.q50.notna()].copy()
    d["ae"] = (d.y - d.q50).abs() / d.scale_mae
    return d


def accuracy():
    fin = final_models()
    by_state, by_season, by_h = [], [], []
    for tid in O.TARGETS:
        f, n = scored(tid, fin.get(tid, "")), scored(tid, "seasonal_naive")
        if f is None or n is None:
            continue
        for keys, sink in ((["sid"], by_state), (["season"], by_season), (["H"], by_h)):
            a = f.groupby(keys).ae.mean().rename("mase_final")
            b = n.groupby(keys).ae.mean().rename("mase_naive")
            d = pd.concat([a, b], axis=1).reset_index()
            d.insert(0, "model", fin.get(tid))
            d.insert(0, "tid", tid)
            d["n_days"] = f.groupby(keys).size().to_numpy()
            d["gain_pct"] = 100 * (1 - d.mase_final / d.mase_naive)
            sink.append(d)
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for name, frames in (("accuracy_state", by_state), ("accuracy_season", by_season), ("accuracy_horizon", by_h)):
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.to_csv(os.path.join(OUT, name + ".csv"), index=False)
            out[name] = d
            print(name, len(d), "rows", flush=True)
    return out


def regret_by_state():
    f = os.path.join(D.DEV2, "o5", "decisions.parquet")
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f)
    d = d[d.split == "dev_test"]
    rows = []
    for ent, g in d.groupby("entity"):
        r = dict(entity=ent, days=int(g.date.nunique()))
        for arm in ARMS:
            c = f"regret_{arm}"
            if c in g:
                r[f"mean_{arm}"] = float(g[c].mean())
        if "mean_H_pub" in r and "mean_A2" in r:
            r["saving_A2_vs_pub"] = r["mean_H_pub"] - r["mean_A2"]
            r["saving_pct"] = 100 * r["saving_A2_vs_pub"] / r["mean_H_pub"] if r["mean_H_pub"] else np.nan
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("mean_H_pub", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "regret_state.csv"), index=False)
    print("regret_state", len(out), "States", flush=True)
    return out


def tex_tables(acc, reg):
    os.makedirs(TAB, exist_ok=True)

    def head(stem, caption, label, colspec, header, rows, source, note=None):
        b = ["% written by codes/scripts/audit/a01_strata.py from " + source + "; do not edit",
             "\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}", "\\small",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
        b += [" & ".join(r) + " \\\\" for r in rows]
        b += ["\\bottomrule", "\\end{tabular}"]
        if note:
            b += ["", "\\vspace{2pt}", "{\\footnotesize " + note + "}"]
        b += ["\\end{table}", ""]
        with open(os.path.join(TAB, stem + ".tex"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(b))
        print("wrote", stem, flush=True)

    if "accuracy_season" in acc:
        d = acc["accuracy_season"]
        seasons = sorted(set(d.season))
        rows = []
        for tid in O.TARGETS:
            g = d[d.tid == tid].set_index("season")
            if not len(g):
                continue
            rows.append([f"{tid} {TNAME[tid]}"] + [("--" if not np.isfinite(g.gain_pct.get(s, np.nan)) else f"{g.gain_pct.get(s, np.nan):.1f}") for s in seasons])
        head("t_season_accuracy", "Accuracy gain over the seasonal naive by season on the confirmatory period "
             "(per cent reduction in scaled absolute error). Descriptive; not a pre-registered test.",
             "tab:seasonacc", "l" + "r" * len(seasons), ["Target"] + [s.title() for s in seasons], rows,
             "results/audit/strata/accuracy_season.csv")
    if "accuracy_state" in acc:
        d = acc["accuracy_state"]
        t2 = d[d.tid == "T2"].sort_values("gain_pct", ascending=False)
        if len(t2):
            pick = pd.concat([t2.head(5), t2.tail(5)])
            rows = [[r.sid, f"{r.mase_final:.3f}", f"{r.mase_naive:.3f}", f"{r.gain_pct:+.1f}"]
                    for _, r in pick.iterrows()]
            head("t_state_accuracy", "Best and worst five State control areas for the drawal forecast (T2) on the "
                 "confirmatory period. Descriptive; not a pre-registered test.", "tab:stateacc", "lrrr",
                 ["State", "MASE final", "MASE naive", "Gain (\\%)"], rows,
                 "results/audit/strata/accuracy_state.csv",
                 f"Median gain across all {len(t2)} control areas: {t2.gain_pct.median():+.1f}\\%.")
    if reg is not None and len(reg):
        pick = reg.head(10)
        rows = [[r.entity, f"{r.mean_H_pub:.3f}", f"{r.mean_P:.3f}", f"{r.mean_A2:.3f}", f"{r.saving_pct:+.1f}"]
                for _, r in pick.iterrows()]
        head("t_state_regret", "The ten States with the largest settlement regret under the persistence schedule, "
             "and what the decision rules would have cost instead (Rs crore per day). Descriptive.",
             "tab:stateregret", "lrrrr",
             ["State", "Persistence", "Forecast median", "Fixed rule (A2)", "Saving (\\%)"], rows,
             "results/audit/strata/regret_state.csv",
             f"Across all {len(reg)} control areas the fixed rule lowers mean daily regret by "
             f"{100 * (1 - reg.mean_A2.sum() / reg.mean_H_pub.sum()):.1f}\\%.")


if __name__ == "__main__":
    acc = accuracy()
    reg = regret_by_state()
    tex_tables(acc, reg)
    print("strata done")
