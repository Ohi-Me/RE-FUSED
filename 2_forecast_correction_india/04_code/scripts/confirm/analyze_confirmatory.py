"""Pre-registered confirmatory analysis (frozen with 02_design/02_preregistration.md).

Computes every confirmatory test H1-H8 from the confirmatory result files and writes
  06_results/confirm/analysis/verdicts.json      one entry per test: estimate, CI, p, adjusted p, verdict
  06_results/confirm/analysis/*.csv              the underlying tables
The same code can be run on development outputs (--map dev) to check that it executes; those numbers are NOT
confirmatory and are written to 06_results/dev/analysis_confirm_dryrun/.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import analysis as A, stats as st  # noqa: E402
from refused_gate.paths import RES  # noqa: E402

MAPS = {
    "confirm": dict(C1="confirm/C1", C2="confirm/C2", C3="confirm/C3", C4="confirm/C4", C5="confirm/C5",
                    C6="confirm/C6", C7="confirm/C7", X5="confirm/x5_x8", X1="confirm/x1"),
    "sensitivity": dict(C1="confirm/C1", C2="confirm/C2", C3="confirm_sensitivity/C3", C4="confirm_sensitivity/C4",
                        C5="confirm_sensitivity/C5", C6="confirm/C6", C7="confirm/C7", X5="confirm/x5_x8",
                        X1="confirm_sensitivity/x1"),  # DC1b
    "dev": dict(C1="dev/dev2_nyiso", C2="dev/nyiso_budget", C3="dev/dev2_india", C6="dev/dev2_ett",
                X5="dev/x5_x8", X1="dev/x1"),
}
UNIT_OF_BLOCK = {"C1": "NYISO-2026", "C3": "OPSD-2019", "C4": "OPSD-2020", "C6": "UCI-2014"}
BUDGET_OF_UNIT = {"NYISO-2026": ("C2", "C1", "iso"), "OPSD-2019": ("C5", "C3", "tso"), "UCI-2014": ("C7", "C6", "snaive168")}
BLOCK_DAYS = 7
STATIC = ["global", "series", "series_x_hour_block", "series_x_hour", "instance"]
TIME = ["time_rolling_30", "time_rolling_90", "time_rolling_180", "time_forget_0.98", "time_forget_0.995"]
UPDATING_MARGIN = -0.002          # H6b non-inferiority margin (pooled skill)
COVERAGE_TOL = 0.005              # H8b reliability tolerance


def rpath(mp, blk):
    return os.path.join(RES, MAPS[mp][blk]) if blk in MAPS[mp] else None


def configs(folder):
    return sorted({re.sub(r"_s\d+\.parquet$", "", os.path.basename(f)) for f in glob.glob(os.path.join(folder, "*.parquet"))})


def load_cfg(folder, key):
    runs = []
    for f in sorted(glob.glob(os.path.join(folder, key + "_s*.parquet"))):
        runs.append((f, pd.read_parquet(f), json.load(open(f[:-8] + ".json"))))
    return runs


def pooled_loss(L, rung):
    return float(L[rung].mean())


def daily_contrast(L, coarse, fine):
    return A.contrast(L, coarse, fine, block_days=BLOCK_DAYS)


# ------------------------------------------------------------------ H2 mechanism (data budget)
def h2(mp, out):
    rows = []
    for unit, (bblk, fblk, base) in BUDGET_OF_UNIT.items():
        bf, ff = rpath(mp, bblk), rpath(mp, fblk)
        if not bf or not os.path.isdir(bf):
            continue
        levels = {"frac0.1": load_cfg(bf, "val_frac0.1"), "frac0.25": load_cfg(bf, "val_frac0.25"),
                  "frac0.5": load_cfg(bf, "val_frac0.5"), "full": load_cfg(ff, base) if ff else [],
                  "oof_val": load_cfg(bf, "oof_train_val")}
        levels = {k: v for k, v in levels.items() if v}
        if "frac0.1" not in levels or "oof_val" not in levels:
            continue
        gains = {}
        for k, runs in levels.items():
            L = A.row_losses(runs)
            gains[k] = L.set_index(["sid", "ts"])
        lo, hi = gains["frac0.1"], gains["oof_val"]
        idx = lo.index.intersection(hi.index)
        g_lo = (lo.loc[idx, "series"] - lo.loc[idx, "instance"])
        g_hi = (hi.loc[idx, "series"] - hi.loc[idx, "instance"])
        d = (g_hi - g_lo).reset_index()
        d["day"] = pd.to_datetime(d.ts).dt.normalize()
        per_day = d.groupby("day")[0].agg(["sum", "size"]).sort_index()
        daily = (per_day["sum"] / per_day["size"]).to_numpy() / float(lo.loc[idx, "base"].mean())
        ci = st.block_ci(daily, BLOCK_DAYS, 2000, 0)
        order = [k for k in ["frac0.1", "frac0.25", "frac0.5", "full", "oof_val"] if k in gains]
        levels_gain = [float((gains[k]["series"] - gains[k]["instance"]).mean() / gains[k]["base"].mean()) for k in order]
        rho = stats.spearmanr(range(len(order)), levels_gain).correlation if len(order) > 2 else np.nan
        rows.append(dict(unit=unit, delta=ci["mean"], lo=ci["lo"], hi=ci["hi"], p=ci["p_greater"], se=ci["se"],
                         spearman_budget=rho, levels=dict(zip(order, levels_gain))))
    out["H2_rows"] = rows
    tests = []
    for r in rows:
        tests.append(dict(id=f"H2-{r['unit']}", family="mechanism", estimate=r["delta"], lo=r["lo"], hi=r["hi"], p=r["p"],
                          criterion="delta > 0 (p < 0.05)"))
    if len(rows) >= 2:
        m = st.dersimonian_laird([r["delta"] for r in rows], [max(r["se"], 1e-12) for r in rows])
        tests.append(dict(id="H2-meta", family="mechanism", estimate=m["effect"], lo=m["lo"], hi=m["hi"],
                          p=float(m["p"] / 2 if m["effect"] > 0 else 1 - m["p"] / 2), criterion="pooled delta > 0",
                          extra=dict(I2=m["I2"], k=m["k"])))
    return tests


# ------------------------------------------------------------------ H3/H7 static selection, H6 time axis, H5
def ladder_tables(mp):
    part_rows, time_rows, sel_rows, strength = [], [], [], []
    for blk, unit in UNIT_OF_BLOCK.items():
        folder = rpath(mp, blk)
        if not folder or not os.path.isdir(folder):
            continue
        for key in configs(folder):
            runs = load_cfg(folder, key)
            L = A.row_losses(runs)
            base = float(L["base"].mean())
            loss = {c: pooled_loss(L, c) for c in L.columns if c not in ("sid", "ts", "base")}
            diags = [d for _, _, d in runs]
            strength.append(dict(unit=unit, baseline=key, base_rmse=float(np.sqrt(base)),
                                 best_skill=1 - min(loss.values()) / base))
            # partition / instance refinements
            has_part2 = all("part2" in d for d in diags)
            has_preq = all("prequential" in d for d in diags)
            refs = [("global", "series")] + [("series", c) for c in STATIC if c.startswith("series_x_")] + [("series", "instance")]
            refs = refs if has_part2 else []
            for c, f in refs:
                if c not in loss or f not in loss:
                    continue
                name = f"{c}->{f}"
                preds = [d["part2"][name]["G"] for d in diags if "part2" in d and name in d["part2"]]
                nested = [d["nested_risk"][c] - d["nested_risk"][f] for d in diags if "nested_risk" in d
                          and c in d["nested_risk"] and f in d["nested_risk"]]
                real = (loss[c] - loss[f]) / base
                part_rows.append(dict(unit=unit, baseline=key, refinement=name, realised=real,
                                      pred=np.mean(preds) / base if preds else np.nan,
                                      nested=np.mean(nested) / base if nested else np.nan))
            # time policies vs expanding (H6a) and updating (H6b)
            if "time_expanding" in loss and has_preq:
                for pol in TIME:
                    if pol not in loss:
                        continue
                    name = f"expanding->{pol}"
                    preds = [d["prequential"][name] for d in diags if "prequential" in d and name in d["prequential"]]
                    time_rows.append(dict(unit=unit, baseline=key, policy=pol, realised=(loss["time_expanding"] - loss[pol]) / base,
                                          pred=np.mean(preds) / base if preds else np.nan,
                                          updating=(loss["series"] - loss["time_expanding"]) / base))
            # static-level selection regret (H7)
            if not has_part2:
                continue
            cands = [c for c in STATIC if c in loss]
            oracle = min(loss[c] for c in cands)
            choice = "global"
            p2 = {}
            for d in diags:
                for k, v in d.get("part2", {}).items():
                    p2.setdefault(k, []).append(v["G"])
            p2 = {k: np.mean(v) for k, v in p2.items()}
            if p2.get("global->series", -1) > 0:
                choice = "series"
                state = {k.split("->")[1]: v for k, v in p2.items() if k.startswith("series->series_x_") and v > 0}
                if state:
                    choice = max(state, key=state.get)
                if p2.get("series->instance", -1) > 0 and p2["series->instance"] > max(state.values(), default=0):
                    choice = "instance"
            nr = {}
            for d in diags:
                for k, v in d.get("nested_risk", {}).items():
                    nr.setdefault(k, []).append(v)
            nested_choice = min((c for c in cands if c in nr), key=lambda c: np.mean(nr[c])) if nr else "series"
            sel_rows.append(dict(unit=unit, baseline=key, oracle=min(cands, key=loss.get), part_choice=choice,
                                 nested_choice=nested_choice,
                                 regret_part=(loss[choice] - oracle) / base, regret_nested=(loss[nested_choice] - oracle) / base,
                                 regret_series=(loss["series"] - oracle) / base, regret_global=(loss["global"] - oracle) / base))
    return pd.DataFrame(part_rows), pd.DataFrame(time_rows), pd.DataFrame(sel_rows), pd.DataFrame(strength)


def decision_regret(pred, real):
    return np.where(pred > 0, np.maximum(-real, 0), np.maximum(real, 0))


def one_sided_wilcoxon_less(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.abs(d) > 1e-15]
    if len(d) < 6:
        return np.nan
    return float(stats.wilcoxon(d, alternative="less").pvalue)


def h3_h6_h7_h5(P, T, S, STR):
    tests = []
    if len(P):
        Pp = P.dropna(subset=["pred"])
        correct = int((np.sign(Pp.pred) == np.sign(Pp.realised)).sum())
        tests.append(dict(id="H3a-sign", family="selection", estimate=correct / len(Pp), n=len(Pp),
                          p=float(stats.binomtest(correct, len(Pp), 0.5, alternative="greater").pvalue),
                          criterion="sign accuracy > 0.5"))
        rp = decision_regret(Pp.pred.to_numpy(), Pp.realised.to_numpy())
        never = np.maximum(Pp.realised.to_numpy(), 0)
        tests.append(dict(id="H3b-regret-vs-never", family="selection", estimate=float(rp.mean() - never.mean()),
                          p=one_sided_wilcoxon_less(rp, never), criterion="PART regret < never-refine regret"))
        Pn = Pp.dropna(subset=["nested"])
        if len(Pn):
            rn = decision_regret(Pn.nested.to_numpy(), Pn.realised.to_numpy())
            rpn = decision_regret(Pn.pred.to_numpy(), Pn.realised.to_numpy())
            tests.append(dict(id="H3c-regret-vs-nested", family="selection", estimate=float(rpn.mean() - rn.mean()),
                              p=one_sided_wilcoxon_less(rpn, rn), criterion="PART regret < nested hold-out regret"))
    if len(T):
        Tp = T.dropna(subset=["pred"])
        correct = int((np.sign(Tp.pred) == np.sign(Tp.realised)).sum())
        tests.append(dict(id="H6a-sign", family="time", estimate=correct / len(Tp), n=len(Tp),
                          p=float(stats.binomtest(correct, len(Tp), 0.5, alternative="greater").pvalue),
                          criterion="prequential sign accuracy > 0.5"))
        rq = decision_regret(Tp.pred.to_numpy(), Tp.realised.to_numpy())
        always, never = np.maximum(-Tp.realised.to_numpy(), 0), np.maximum(Tp.realised.to_numpy(), 0)
        tests.append(dict(id="H6a-regret-vs-always", family="time", estimate=float(rq.mean() - always.mean()),
                          p=one_sided_wilcoxon_less(rq, always), criterion="prequential regret < always-discount"))
        tests.append(dict(id="H6a-regret-vs-never", family="time", estimate=float(rq.mean() - never.mean()),
                          p=one_sided_wilcoxon_less(rq, never), criterion="prequential regret < never-discount"))
        upd = T.groupby(["unit", "baseline"]).updating.first()
        share = float((upd >= UPDATING_MARGIN).mean())
        tests.append(dict(id="H6b-updating-free", family="time", estimate=share, n=len(upd), p=np.nan,
                          criterion=f"share of configs with expanding - frozen >= {UPDATING_MARGIN} is >= 0.8",
                          verdict_rule="threshold"))
    if len(S):
        tests.append(dict(id="H7a-part-vs-nested", family="selection", estimate=float(S.regret_part.mean() - S.regret_nested.mean()),
                          p=one_sided_wilcoxon_less(S.regret_part, S.regret_nested), criterion="PART selection regret < nested"))
        tests.append(dict(id="H7b-part-vs-fixed-series", family="selection",
                          estimate=float(S.regret_part.mean() - S.regret_series.mean()),
                          p=one_sided_wilcoxon_less(S.regret_part, S.regret_series), criterion="PART regret < fixed per-series"))
    ny = STR[STR.unit == "NYISO-2026"] if len(STR) else STR
    if len(ny) >= 5:
        rho, p = stats.spearmanr(ny.base_rmse, ny.best_skill)
        tests.append(dict(id="H5-strength", family="baseline", estimate=float(rho), n=len(ny),
                          p=float(p / 2 if rho > 0 else 1 - p / 2), criterion="Spearman(baseline RMSE, best correction skill) > 0"))
    return tests


# ------------------------------------------------------------------ H4 / H8 probabilistic and energy
def h4_h8(mp):
    folder = rpath(mp, "X5")
    files = sorted(glob.glob(os.path.join(folder, "prob_energy_s*.parquet"))) if folder else []
    if not files:
        return []
    frames = [pd.read_parquet(f) for f in files]
    base = frames[0][["sid", "ts", "y_t", "da_lbmp", "rt_lbmp", "da_spin"]].copy()
    num = [c for c in frames[0].columns if c.startswith(("F_", "pin99_", "up99_"))]
    avg = sum(f[num].to_numpy(dtype=float) for f in frames) / len(frames)
    D = pd.concat([base.reset_index(drop=True), pd.DataFrame(avg, columns=num)], axis=1)
    D["day"] = D.ts.dt.normalize()
    tests = []

    def daily_diff(a, b, per_row=False):
        g = (D[a] - D[b]).groupby(D.day)
        x = (g.mean() if per_row else g.sum()).sort_index().to_numpy()
        return st.block_ci(x, BLOCK_DAYS, 2000, 0)

    for other in ("q_global", "q_zone"):
        ci = daily_diff(f"pin99_corr_series_{other}", "pin99_corr_series_q_zone_hour", per_row=True)
        tests.append(dict(id=f"H4a-zonehour-vs-{other}", family="probabilistic", estimate=ci["mean"], lo=ci["lo"], hi=ci["hi"],
                          p=ci["p_greater"], criterion="pinball@0.99 lower for zone x hour cells"))
    err = D.y_t - D.F_corr_series
    cov_inst = float(np.mean(err <= D.up99_corr_series_q_instance))
    cov_zh = float(np.mean(err <= D.up99_corr_series_q_zone_hour))
    tests.append(dict(id="H4b-instance-undercovers", family="probabilistic", estimate=cov_inst, extra=dict(zone_hour=cov_zh),
                      p=np.nan, criterion="coverage(instance) < 0.985 and coverage(zone x hour) >= 0.985", verdict_rule="threshold",
                      passed=bool(cov_inst < 0.985 and cov_zh >= 0.985)))
    D["dev_iso"] = np.abs(D.y_t - D.F_iso)
    D["dev_cs"] = np.abs(D.y_t - D.F_corr_series)
    ci = daily_diff("dev_iso", "dev_cs")
    n_days = D.day.nunique()
    tests.append(dict(id="H8a-deviation-energy", family="energy", estimate=ci["mean"] * 365 / 1000, lo=ci["lo"] * 365 / 1000,
                      hi=ci["hi"] * 365 / 1000, p=ci["p_greater"], criterion="deviation energy lower after correction (GWh/yr)"))
    D["res_iso"] = np.maximum(D.up99_iso_q_zone_hour, 0)
    D["res_cs"] = np.maximum(D.up99_corr_series_q_zone_hour, 0)
    ci = daily_diff("res_iso", "res_cs")
    cov_iso = float(np.mean((D.y_t - D.F_iso) <= D.up99_iso_q_zone_hour))
    tests.append(dict(id="H8b-reserve-mw", family="energy", estimate=ci["mean"] / 24, lo=ci["lo"] / 24, hi=ci["hi"] / 24,
                      p=ci["p_greater"], extra=dict(coverage_iso=cov_iso, coverage_corrected=cov_zh),
                      criterion=f"99% reserve (MW, summed over zones) lower after correction with |coverage difference| <= {COVERAGE_TOL}",
                      coverage_ok=bool(abs(cov_iso - cov_zh) <= COVERAGE_TOL)))
    imb_iso = (D.y_t - D.F_iso) * (D.rt_lbmp - D.da_lbmp)
    imb_cs = (D.y_t - D.F_corr_series) * (D.rt_lbmp - D.da_lbmp)
    x = (imb_iso - imb_cs).groupby(D.day).sum().sort_index().to_numpy()
    ci = st.block_ci(np.nan_to_num(x), BLOCK_DAYS, 2000, 0)
    tests.append(dict(id="E8c-imbalance-cost", family="energy-estimation", estimate=ci["mean"] * 365 / 1e6, lo=ci["lo"] * 365 / 1e6,
                      hi=ci["hi"] * 365 / 1e6, p=np.nan, criterion="estimation only (no directional hypothesis), M$/yr saved",
                      verdict_rule="estimate"))
    return tests


# ------------------------------------------------------------------ H1 synthetic
def h1(mp):
    folder = rpath(mp, "X1")
    tests = []
    for part in ("partition", "instance"):
        files = glob.glob(os.path.join(folder, f"x1_{part}_*.csv")) if folder else []
        if not files:
            continue
        X = pd.concat([pd.read_csv(f) for f in files])
        dec = X[np.abs(X.oracle_pred) >= 0.5 * X.dV]
        if part == "instance":
            # DV8: for the instance axis dV is the realised loss at tau = 0, so m = 0 is tautological; excluded
            dec = dec[dec.m > 0]
        for col, thr in (("oracle_pred", 0.85), ("part_contiguous_G", 0.75)):
            if col in dec:
                acc = float(np.mean(np.sign(dec[col]) == np.sign(dec.realised)))
                tests.append(dict(id=f"H1-{part}-{col}", family="synthetic", estimate=acc, n=len(dec), p=np.nan,
                                  criterion=f"sign accuracy >= {thr} on non-boundary configurations", verdict_rule="threshold",
                                  passed=bool(acc >= thr)))
    return tests


EXPECTED = [("H2", "gt", 0.0), ("H3a", "gt", 0.5), ("H6a-sign", "gt", 0.5), ("H3b", "lt", 0.0), ("H3c", "lt", 0.0),
            ("H6a-regret", "lt", 0.0), ("H7", "lt", 0.0), ("H5", "gt", 0.0), ("H4a", "gt", 0.0), ("H8a", "gt", 0.0),
            ("H8b", "gt", 0.0)]


def direction_ok(t):
    """Pre-registered rule: the point estimate must have the predicted sign (prereg §3)."""
    for prefix, op, ref in EXPECTED:
        if t["id"].startswith(prefix):
            return t["estimate"] > ref if op == "gt" else t["estimate"] < ref
    return True


def finalise(tests):
    fams = {}
    for t in tests:
        if t.get("p") is not None and np.isfinite(t.get("p", np.nan)):
            fams.setdefault(t["family"], []).append(t)
    for fam, ts in fams.items():
        adj = st.holm([t["p"] for t in ts])
        for t, a in zip(ts, adj):
            t["p_holm"] = float(a)
    allp = [t for t in tests if t.get("p") is not None and np.isfinite(t.get("p", np.nan))]
    by = st.benjamini_yekutieli([t["p"] for t in allp]) if allp else []
    for t, a in zip(allp, by):
        t["p_by"] = float(a)
    for t in tests:
        rule = t.get("verdict_rule", "p")
        if rule == "threshold":
            if "passed" in t:
                t["verdict"] = "SUPPORTED" if t["passed"] else "NOT SUPPORTED"
            elif t["id"] == "H6b-updating-free":
                t["verdict"] = "SUPPORTED" if t["estimate"] >= 0.8 else "NOT SUPPORTED"
        elif rule == "estimate":
            t["verdict"] = "ESTIMATE"
        else:
            ok = t.get("p_holm", 1.0) < 0.05 and direction_ok(t)
            if t["id"].startswith("H8b"):
                ok = ok and t.get("coverage_ok", False)
            t["verdict"] = "SUPPORTED" if ok else "NOT SUPPORTED"
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="confirm", choices=["confirm", "dev", "sensitivity"])
    a = ap.parse_args()
    out_dir = {"confirm": os.path.join(RES, "confirm", "analysis"), "sensitivity": os.path.join(RES, "confirm_sensitivity", "analysis"),
               "dev": os.path.join(RES, "dev", "analysis_confirm_dryrun")}[a.map]
    os.makedirs(out_dir, exist_ok=True)
    extra = {}
    tests = h2(a.map, extra)
    P, T, S, STR = ladder_tables(a.map)
    for name, df in (("H3_part_refinements", P), ("H6_time_policies", T), ("H7_selection", S), ("H5_strength", STR)):
        df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    pd.DataFrame(extra.get("H2_rows", [])).to_csv(os.path.join(out_dir, "H2_budget.csv"), index=False)
    tests += h3_h6_h7_h5(P, T, S, STR)
    tests += h4_h8(a.map)
    tests += h1(a.map)
    tests = finalise(tests)
    json.dump(tests, open(os.path.join(out_dir, "verdicts.json"), "w"), indent=1, default=float)
    for t in tests:
        est = t["estimate"]
        print(f"{t['id']:34s} {t.get('verdict','?'):14s} est={est:+.5f} p={st.fmt_p(t.get('p', np.nan))} "
              f"p_holm={st.fmt_p(t.get('p_holm', np.nan))}  [{t['criterion']}]")


if __name__ == "__main__":
    main()
