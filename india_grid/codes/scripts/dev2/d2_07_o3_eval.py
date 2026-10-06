"""dev2 step 7: LA-MCAG round-2 checks (protocol 06, section 4).

Part A (N7) staleness-matched source dropout: mcag_instance_sd and fusion_fixed_sd against each other (A2) and against
their round-1 versions, and the source-loss test H7 (T1, T2): does the gated model lose less than fusion when the RE and
weather blocks go missing on the development test?

Part B (N9) PART as a decision rule: for each target x horizon x refinement (fixed -> regime, fixed -> instance,
fixed -> online gate) we compute G on the validation year as in round 1, plus a series-bootstrap 80 % interval. We
refine only when the lower end is above zero. The rule is scored by the realised development-test gain it collects,
next to "always refine" and "never refine", and by sign agreement over all cells and over cells whose realised gain is
clearly not zero.

Outputs: results/dev2/o3/eval/{variants.csv, pairs.csv, l1.csv, part_decision.csv}; results/tables/dev2_o3_summary.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
sys.path.insert(0, os.path.join(HERE, "..", "o3"))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from refused import selection as SEL  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import TAB  # noqa: E402
from refused.report import md_table  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402
import o3_02_evaluate as E3  # noqa: E402

SMOKE = "--smoke" in sys.argv
N_BOOT = 3 if SMOKE else 50
KEY = ["sid", "issue_date", "H"]
O3_ROUND1 = os.path.join(D.DEV1, "o3")
O3_ROUND2 = os.path.join(D.DEV2, "o3")


def load_from(folder, tid, name):
    E3.O3 = folder  # E3.load reads from this module-level folder
    return E3.load(tid, name)


def part_a(tid):
    Ly = D.lag_of(tid)
    sc = D.scales(tid)
    V = {}
    for folder, names in ((O3_ROUND1, ["mcag_instance", "fusion_fixed"]), (O3_ROUND2, ["mcag_instance_sd", "fusion_fixed_sd"])):
        for n in names:
            d = load_from(folder, tid, n)
            if d is not None:
                V[n] = d
    if len(V) < 2:
        return [], [], []
    rows = None
    for d in V.values():
        k = d.loc[d.q50.notna() & d.y.notna(), KEY]
        rows = k if rows is None else rows.merge(k, on=KEY)
    C = {n: E3.calibrate(d.merge(rows, on=KEY).merge(sc, on=["sid", "H"]), Ly) for n, d in V.items()}
    variants, pairs, l1 = [], [], []
    for n, d in C.items():
        for sp in ("validation", "dev_test"):
            s = d[d.split == sp]
            pm, _ = E2.point_metrics(s, s.q50)
            pr, _ = E2.prob_metrics(s, s[["c" + c for c in D.QN]].to_numpy())
            variants.append(dict(tid=tid, model=n, split=sp, **pm, pinball=pr["pinball"], cov80=pr["cov80"],
                                 cov90=pr["cov90"], n=len(s)))
    for a, b, label in (("mcag_instance_sd", "fusion_fixed_sd", "A2 sd: instance vs fusion"),
                        ("mcag_instance_sd", "mcag_instance", "sd vs round 1 (instance)"),
                        ("fusion_fixed_sd", "fusion_fixed", "sd vs round 1 (fusion)")):
        if a in C and b in C:
            for metric in ("mae", "pinball"):
                pairs += E3.pair_test(C[a], C[b], label, tid, metric)
    if tid in ("T1", "T2"):
        J = {}
        for name, folder in (("mcag_instance_sd", O3_ROUND2), ("fusion_fixed_sd", O3_ROUND2)):
            dl = load_from(folder, tid, name + "_L1")
            if dl is None or name not in V:
                continue
            j = V[name].merge(dl[KEY + ["q50"]].rename(columns={"q50": "q50_l1"}), on=KEY).merge(sc, on=["sid", "H"])
            j = j[(j.split == "dev_test") & j.y.notna()]
            l1.append(dict(tid=tid, model=name, mase=float(((j.y - j.q50).abs() / j.scale_mae).mean()),
                           mase_source_loss=float(((j.y - j.q50_l1).abs() / j.scale_mae).mean())))
            J[name] = j
        if len(J) == 2:
            a, b = J["mcag_instance_sd"], J["fusion_fixed_sd"]
            j = a[KEY + ["target_date", "y", "q50", "q50_l1", "scale_mae"]].merge(b[KEY + ["q50", "q50_l1"]], on=KEY, suffixes=("_i", "_f"))
            inc_i = ((j.y - j.q50_l1_i).abs() - (j.y - j.q50_i).abs()) / j.scale_mae
            inc_f = ((j.y - j.q50_l1_f).abs() - (j.y - j.q50_f).abs()) / j.scale_mae
            dd = E2.daily_diff(j, inc_i.to_numpy(), inc_f.to_numpy())
            ci = S.block_ci(dd, 7, n_boot=2000, seed=0)
            l1.append(dict(tid=tid, model="instance_sd − fusion_sd (degradation difference)", mean_diff=float(dd.mean()),
                           lo=ci["lo"], hi=ci["hi"], p_instance_degrades_less=E2.hln(dd, 2)[1]))
    return variants, pairs, l1


def g_values(v, gt, stale, Ly, idx=None):
    """PART G for the three refinements on validation rows v (optionally a bootstrap row index)."""
    if idx is not None:
        v = v.iloc[idx].reset_index(drop=True)
    R = ((v.y - v.b50) / v.scale_mae).to_numpy()
    r = ((v.q50 - v.b50) / v.scale_mae).to_numpy()
    doy = v.target_date.dt.dayofyear.to_numpy()
    F = np.column_stack([v[stale].to_numpy(), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    zeros = np.zeros(len(v)).astype(str)
    return {"regime vs fixed": SEL.part2_partition(R, r, zeros, v.regime_cell.astype(str), v.target_date, mode="interleaved", K=6)["G"],
            "instance vs fixed": SEL.part2_instance(F, R, r, zeros, v.target_date, mode="interleaved", K=4)["G"],
            "OTG vs fixed": SEL.prequential_time(R, r, v.sid, v.target_date, window=180, delay=Ly)["G"]}


def part_b(tid):
    Ly = D.lag_of(tid)
    sc = D.scales(tid)
    E3.O3 = O3_ROUND1
    fx = E3.load(tid, "mcag_fixed")
    gates = os.path.join(O3_ROUND1, "preds", f"{tid}__mcag_fixed_gates.parquet")
    if fx is None or not os.path.exists(gates):
        return []
    fx = fx.merge(sc, on=["sid", "H"])
    refined = {"regime vs fixed": E3.load(tid, "mcag_regime"), "instance vs fixed": E3.load(tid, "mcag_instance"),
               "OTG vs fixed": E3.otg(E3.load(tid, "mcag_fixed"), Ly)}
    gt = pd.read_parquet(gates)
    gt = gt[gt.seed == 0].drop(columns=["seed"])
    stale = [c for c in gt.columns if c.startswith("stale_")]
    rng = np.random.default_rng(0)
    out = []
    for H in O.HORIZONS:
        v = fx[(fx.split == "validation") & (fx.H == H) & fx.y.notna()].merge(
            gt[["sid", "issue_date", "regime_cell"] + stale], on=["sid", "issue_date"]).reset_index(drop=True)
        G = g_values(v, gt, stale, Ly)
        sids = v.sid.unique()
        by_sid = {s: np.where(v.sid.to_numpy() == s)[0] for s in sids}
        boot = {k: [] for k in G}
        for b in range(N_BOOT):
            pick = rng.choice(sids, len(sids), replace=True)
            idx = np.concatenate([by_sid[s] for s in pick])
            gb = g_values(v, gt, stale, Ly, idx)
            for k in G:
                boot[k].append(gb[k])
        for lab, d in refined.items():
            if d is None:
                continue
            j = d[(d.split == "dev_test") & (d.H == H)].merge(
                fx[(fx.split == "dev_test") & (fx.H == H)][KEY + ["q50"]].rename(columns={"q50": "q50_fx"}), on=KEY)
            j = j.merge(sc, on=["sid", "H"])
            j = j[j.y.notna()]
            gain_rows = (((j.y - j.q50_fx) ** 2 - (j.y - j.q50) ** 2) / j.scale_mae ** 2).to_numpy()
            daily = pd.Series(gain_rows).groupby(j.target_date.to_numpy()).mean().sort_index().to_numpy()
            ci = S.block_ci(daily, 7, n_boot=500 if SMOKE else 2000, seed=0)
            lower = float(np.quantile(boot[lab], 0.10))
            out.append(dict(tid=tid, H=H, refinement=lab, G_val=G[lab], G_lower80=lower, G_upper80=float(np.quantile(boot[lab], 0.90)),
                            refine_rule=lower > 0, refine_sign=G[lab] > 0, realised_gain_dev=float(gain_rows.mean()),
                            gain_lo=ci["lo"], gain_hi=ci["hi"], clear=bool(ci["lo"] > 0 or ci["hi"] < 0)))
    return out


def summary(variants, pairs, l1, part):
    L = ["# O3 round 2 — staleness-matched dropout and PART as a decision rule", "",
         "Development test FY2024-25, reported only (protocol `codes/design/06_dev2_protocol.md`, section 4).", ""]
    if len(variants):
        L += ["## Variants (development test)", "", md_table(variants[variants.split == "dev_test"].drop(columns=["split"]), 4), ""]
    if len(pairs):
        L += ["## Paired tests (all horizons pooled)", "", md_table(pairs[pairs.H == "all"], 4), ""]
    if len(l1):
        L += ["## H7 source loss with staleness-matched dropout (RE and weather blocks missing)", "", md_table(l1, 4), ""]
    if len(part):
        tot = dict(rule=float((part.realised_gain_dev * part.refine_rule).sum()), sign_only=float((part.realised_gain_dev * part.refine_sign).sum()),
                   always=float(part.realised_gain_dev.sum()), never=0.0,
                   oracle=float(part.realised_gain_dev.clip(lower=0).sum()))
        agree_all = ((part.G_val > 0) == (part.realised_gain_dev > 0))
        clear = part[part.clear]
        agree_clear = ((clear.G_val > 0) == (clear.realised_gain_dev > 0))
        L += ["## PART decision rule (N9)", "",
              "Total realised development-test gain collected (MSE, scaled units; higher is better): "
              + ", ".join(f"{k} {v:+.4f}" for k, v in tot.items()) + ".", "",
              f"Sign agreement, all cells: {int(agree_all.sum())}/{len(part)}; cells with a clear realised gain: "
              f"{int(agree_clear.sum())}/{len(clear)}.", "", md_table(part, 4), ""]
    open(os.path.join(TAB, f"{D.TAG}_o3_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    return L


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tids = args[0].split(",") if args else (["T5"] if SMOKE else list(O.TARGETS))
    V, P, L1, PB = [], [], [], []
    for tid in tids:
        v, p, l1 = part_a(tid)
        V += v
        P += p
        L1 += l1
        PB += part_b(tid)
        print(tid, "O3 round-2 checks done", flush=True)
    V, P, L1, PB = map(pd.DataFrame, (V, P, L1, PB))
    ev = D.folder("o3", "eval")
    V.to_csv(os.path.join(ev, "variants.csv"), index=False)
    P.to_csv(os.path.join(ev, "pairs.csv"), index=False)
    L1.to_csv(os.path.join(ev, "l1.csv"), index=False)
    PB.to_csv(os.path.join(ev, "part_decision.csv"), index=False)
    print("\n".join(summary(V, P, L1, PB)[:40]))


if __name__ == "__main__":
    main()
