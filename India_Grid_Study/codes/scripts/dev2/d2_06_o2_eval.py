"""dev2 step 6: pick the final forecaster per target and score everything (protocol 06, sections 1 and 3).

Final forecaster = lowest cross-fitted validation MASE among the best single member, ens_top, ens_stack and
ens_online (ties within 0.5 % go to lower validation pinball loss). Calibration: rolling conformal at level C1; the
window (90, 180, 365 days) for the final forecaster is picked on validation pinball loss, all other models use 180
days as in round 1. Development-test numbers are only reported.

Tests on the development test: final vs seasonal naive (per horizon, HLN one-sided, 7-day block bootstrap, Holm over
targets x horizons), final vs the round-1 selected model, final vs the best single member, and the same test for
each seed of the final forecaster (seed-level members recombined with the frozen ensemble settings).

Outputs: results/dev2/o2/eval/{point,prob,tests,seeds}.csv, selection_dev2.json; results/tables/dev2_o2_summary.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
sys.path.insert(0, HERE)
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import TAB  # noqa: E402
from refused.report import md_table  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402
import d2_05_ensemble as ENS  # noqa: E402

SMOKE = "--smoke" in sys.argv
ENSEMBLES = ["ens_top", "ens_stack", "ens_online"]
WINDOWS = [90, 180, 365]


def round1_selected():
    f = os.path.join(D.DEV1, "o2", "eval", "selection.json")
    return {s["tid"]: s["selected"] for s in json.load(open(f))} if os.path.exists(f) else {}


def choose_final(tid, log, frames, rows_val):
    cf = log["crossfit_val_mase"]
    best = min(cf, key=cf.get)
    ties = [c for c in cf if cf[c] <= cf[best] * 1.005]
    if len(ties) > 1:
        pin = {}
        for c in ties:
            name = log["best_single"] if c == "best_single" else c
            v = frames[name].merge(rows_val, on=["sid", "issue_date", "H"])
            pin[c] = float(D.pinball_rows(v.y.to_numpy(), v[D.QN].to_numpy(), v.scale_mae.to_numpy()).mean())
        best = min(ties, key=pin.get)
    return best, (log["best_single"] if best == "best_single" else best)


def pooled_test(a, b, H):
    la = ((a.y - a.q50).abs() / a.scale_mae).to_numpy()
    lb = ((b.y - b.q50).abs() / b.scale_mae).to_numpy()
    dd = E2.daily_diff(a, la, lb)
    t, p = E2.hln(dd, 2 if H == "all" else H)
    ci = S.block_ci(dd, 7, n_boot=500 if SMOKE else 2000, seed=0)
    return dict(mean_diff=float(dd.mean()), lo=ci["lo"], hi=ci["hi"], p_one_sided=p, n_days=len(dd))


def seed_frames(tid, log, final_key, seeds):
    """The final forecaster rebuilt from one seed of each member, with the ensemble settings kept as frozen."""
    out = {}
    for seed in seeds:
        if final_key == "best_single":
            d = D.load_member(tid, log["best_single"], seed)
            if d is None:
                continue
            out[seed] = d
            continue
        base, arr, used = D.member_stack(tid, log["members"], seed)
        if base is None or used != log["members"]:
            continue
        if final_key == "ens_top":
            idx = [used.index(m) for m in log["ens_top"]["members"]]
            Q = np.sort(arr[:, idx, :].mean(axis=1), axis=1)
        elif final_key == "ens_stack":
            Q = np.full((len(base), arr.shape[2]), np.nan)
            for H in O.HORIZONS:
                w = np.array([log["ens_stack"]["weights"][str(H)][m] for m in used])
                h = (base.H == H).to_numpy()
                Q[h] = np.sort((arr[h] * w[None, :, None]).sum(axis=1), axis=1)
        else:
            e = log["ens_online"]
            Q, _ = ENS.online_forecast(base, arr, D.lag_of(tid), e["eta"], e["gamma"], e["pool"], D.regions(tid))
        fr = base[D.KEYS].copy()
        fr[D.QN] = Q
        out[seed] = fr
    return out


def evaluate(tid, r1_sel):
    log = json.load(open(os.path.join(D.DEV2, "o2", "logs", f"ensemble_{tid}.json")))
    Ly = D.lag_of(tid)
    names = log["members"] + ENSEMBLES + ["seasonal_naive"]
    frames = {n: D.load_member(tid, n) for n in names}
    frames = {n: d for n, d in frames.items() if d is not None}
    key = ["sid", "issue_date", "H"]
    rows = None
    for d in frames.values():
        k = d.loc[d.q50.notna() & d.y.notna(), key]
        rows = k if rows is None else rows.merge(k, on=key)
    counts = {n: int((d.q50.notna() & d.y.notna()).sum()) for n, d in frames.items()}
    print(tid, "rows per member:", counts, "-> common rows:", len(rows), flush=True)
    if len(rows) < 50:
        print(tid, "too few common rows, skipped", flush=True)
        return None
    sc = D.scales(tid)
    F = {n: d.merge(rows, on=key).merge(sc, on=["sid", "H"]).sort_values(["sid", "issue_date", "H"]).reset_index(drop=True)
         for n, d in frames.items()}
    rows_val = F["seasonal_naive"].loc[F["seasonal_naive"].split == "validation", key + ["y", "scale_mae"]]
    frozen_window = None
    if D.USE_FROZEN:  # confirmatory phase: the final forecaster and its window come from the development run
        fz = next(x for x in D.frozen("o2", "eval", "selection_dev2.json") if x["tid"] == tid)
        final_key, final_name, frozen_window = fz["final_kind"], fz["final"], int(fz["calibration_window"])
    else:
        final_key, final_name = choose_final(tid, log, F, rows_val[key])
    ref = F["seasonal_naive"]
    point, prob, tests = [], [], []
    mae_sn = {sp: float(((ref.y - ref.q50).abs() / ref.scale_mae)[ref.split == sp].mean()) for sp in ("validation", "dev_test")}
    for n, d in F.items():
        for sp in ("validation", "dev_test"):
            s = d[d.split == sp]
            pm, _ = E2.point_metrics(s, s.q50)
            point.append(dict(tid=tid, model=n, split=sp, **pm, rel_sn=float(((s.y - s.q50).abs() / s.scale_mae).mean()) / mae_sn[sp],
                              n=len(s), is_final=(n == final_name)))
    # calibration: 180 days for everyone, window picked on validation for the final forecaster
    chosen_window = 180
    for n, d in F.items():
        if n == "seasonal_naive" or not E2.has_quantiles(d):
            continue
        wins = ([frozen_window] if frozen_window else WINDOWS) if n == final_name else [180]
        Qs = {W: E2.rolling_level(d, d, "C1", True, Ly, W) for W in wins}
        if n == final_name and frozen_window:
            chosen_window = frozen_window
        elif n == final_name:
            val = (d.split == "validation").to_numpy()
            ok = val & np.all([np.isfinite(Q).all(axis=1) for Q in Qs.values()], axis=0)
            vp = {W: float(D.pinball_rows(d.y.to_numpy()[ok], Qs[W][ok], d.scale_mae.to_numpy()[ok]).mean()) for W in wins}
            chosen_window = min(vp, key=vp.get)
            tests.append(dict(tid=tid, model=n, test="calibration_window_val_pinball", **{f"W{W}": v for W, v in vp.items()},
                              chosen=chosen_window))
        dev = (d.split == "dev_test").to_numpy()
        for W, Q in Qs.items():
            m, _ = E2.prob_metrics(d[dev], Q[dev])
            prob.append(dict(tid=tid, model=n, window=W, chosen=(n == final_name and W == chosen_window), **m, n=int(dev.sum())))
    # tests on the development test
    fin = F[final_name][F[final_name].split == "dev_test"]
    comparators = [("seasonal_naive", "final_vs_seasonal_naive")]
    if r1_sel.get(tid) in F and r1_sel.get(tid) != final_name:
        comparators.append((r1_sel[tid], "final_vs_round1_selected"))
    if log["best_single"] in F and log["best_single"] != final_name:
        comparators.append((log["best_single"], "final_vs_best_single"))
    for comp, label in comparators:
        other = F[comp][F[comp].split == "dev_test"]
        for H in O.HORIZONS + ["all"]:
            a = fin if H == "all" else fin[fin.H == H]
            b = other if H == "all" else other[other.H == H]
            tests.append(dict(tid=tid, model=final_name, test=label, comparator=comp, H=H, **pooled_test(a, b, H)))
    seeds = []
    for seed, fr in seed_frames(tid, log, final_key, range(1 if SMOKE else 5)).items():
        j = fr[fr.split == "dev_test"].merge(sc, on=["sid", "H"]).merge(
            ref[ref.split == "dev_test"][key + ["q50"]].rename(columns={"q50": "sn"}), on=key)
        j = j[j.y.notna() & j.q50.notna()]
        pm, _ = E2.point_metrics(j, j.q50)
        dd = E2.daily_diff(j, ((j.y - j.q50).abs() / j.scale_mae).to_numpy(), ((j.y - j.sn).abs() / j.scale_mae).to_numpy())
        seeds.append(dict(tid=tid, model=final_name, seed=seed, mase=pm["mase"], p_vs_sn=E2.hln(dd, 2)[1], n=len(j)))
    sel = dict(tid=tid, final=final_name, final_kind=final_key, crossfit_val_mase=log["crossfit_val_mase"],
               calibration_window=chosen_window, members=log["members"], round1_selected=r1_sel.get(tid),
               ensemble_settings={k: log[k] for k in ("ens_top", "ens_stack", "ens_online")})
    print(tid, "final:", final_name, "window", chosen_window, flush=True)
    return pd.DataFrame(point), pd.DataFrame(prob), pd.DataFrame(tests), pd.DataFrame(seeds), sel


def summary(P, R, T, SD, SEL):
    L = ["# O2 round 2 — base models, combinations and the final forecaster", "",
         "Development test FY2024-25 is reported only; every choice was made on the validation year "
         "(protocol `codes/design/06_dev2_protocol.md`). MASE < 1 beats the seasonal naive.", ""]
    for s in SEL:
        tid = s["tid"]
        if not len(P[P.tid == tid]):
            continue
        L += [f"## {tid} — {O.TARGETS[tid]['name']} (final: **{s['final']}**, calibration window {s['calibration_window']} days)", "",
              "Cross-fitted validation MASE: " + ", ".join(f"{k} {v:.4f}" for k, v in s["crossfit_val_mase"].items()), "",
              "| model | MASE val | MASE dev | RMSSE dev | rel. SN dev | pinball dev (C1, 180 d) | cov80 / cov90 dev |",
              "|---|---|---|---|---|---|---|"]
        p = P[P.tid == tid]
        r = R[(R.tid == tid) & (R.window == 180)].set_index("model") if len(R) else pd.DataFrame()
        for n, g in p.groupby("model"):
            v, d = g[g.split == "validation"].iloc[0], g[g.split == "dev_test"].iloc[0]
            pin = r.pinball.get(n, np.nan) if n in r.index else np.nan
            cov = f"{r.cov80[n]:.3f} / {r.cov90[n]:.3f}" if n in r.index else "–"
            mark = " **(final)**" if n == s["final"] else ""
            L.append(f"| {n}{mark} | {v.mase:.3f} | {d.mase:.3f} | {d.rmsse:.3f} | {d.rel_sn:.3f} | {pin:.3f} | {cov} |")
        t = T[(T.tid == tid) & T.test.str.startswith("final_vs")] if len(T) else T
        if len(t):
            L += ["", "| test (dev) | H | mean diff | 95% block CI | p (final better) |", "|---|---|---|---|---|"]
            for _, x in t.iterrows():
                L.append(f"| {x.test.replace('final_vs_', 'vs ')} ({x.comparator}) | {x.H} | {x.mean_diff:+.4f} | "
                         f"[{x.lo:+.4f}, {x.hi:+.4f}] | {S.fmt_p(x.p_one_sided)} |")
        sd = SD[SD.tid == tid] if len(SD) else SD
        if len(sd):
            L += ["", f"Seeds of the final forecaster: MASE {sd.mase.min():.3f}–{sd.mase.max():.3f}; "
                      f"seeds with p < 0.05 vs seasonal naive: {int((sd.p_vs_sn < 0.05).sum())}/{len(sd)}"]
        L.append("")
    open(os.path.join(TAB, f"{D.TAG}_o2_summary.md"), "w", encoding="utf-8").write("\n".join(L))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tids = args[0].split(",") if args else (["T5"] if SMOKE else list(O.TARGETS))
    r1 = round1_selected()
    P, R, T, SD, SEL = [], [], [], [], []
    for tid in tids:
        if not os.path.exists(os.path.join(D.DEV2, "o2", "logs", f"ensemble_{tid}.json")):
            print(tid, "no ensemble log, skipped", flush=True)
            continue
        res = evaluate(tid, r1)
        if res is None:
            continue
        p, r, t, sd, s = res
        P.append(p), R.append(r), T.append(t), SD.append(sd), SEL.append(s)
    if not P:
        sys.exit("no target could be evaluated")
    P, R, T, SD = (pd.concat(x) if any(len(y) for y in x) else pd.DataFrame() for x in (P, R, T, SD))
    m = (T.test == "final_vs_seasonal_naive") & (T.H != "all") if len(T) else []
    if len(T) and np.any(m):
        T.loc[m, "p_holm_family"] = S.holm(T.loc[m, "p_one_sided"].fillna(1).to_numpy())
    ev = D.folder("o2", "eval")
    P.to_csv(os.path.join(ev, "point.csv"), index=False)
    R.to_csv(os.path.join(ev, "prob.csv"), index=False)
    T.to_csv(os.path.join(ev, "tests.csv"), index=False)
    SD.to_csv(os.path.join(ev, "seeds.csv"), index=False)
    json.dump(SEL, open(os.path.join(ev, "selection_dev2.json"), "w"), indent=1, default=str)
    summary(P, R, T, SD, SEL)


if __name__ == "__main__":
    main()
