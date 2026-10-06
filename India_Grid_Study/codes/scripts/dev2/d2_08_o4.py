"""dev2 step 8: deviation assessment, round 2 (protocol 06, section 5).

U and the forecast median F now come from the final T2 forecaster of round 2 (H = 1, rolling conformal C1 with its
chosen window). Three forms are fitted on the validation year and compared by two-fold ISO-week cross-validation of
the within-M0-decile Kendall tau against the consequence outcome Y:

O4a  round-1 form        A = |d| * [lam + k * w.(R, S, C, U)]
O4b  split deviation     A = lam * [a*|F - S| + (1 - a)*|X - F|] * [1 + k * w.(R, S, C, U)]
                         |F - S| is the part of the deviation that was already visible when the schedule was set,
                         |X - F| is the surprise part
O4c  learned multiplier  A = M0 * exp(k * s), s from a GPU XGBoost pairwise-ranking model (queries = M0 deciles) with a
                         non-decreasing effect of R, S, C and U, plus the two split shares

The chosen form is scored on the development test (reported only): A vs M0 on Y, Y1, Y2, Y3; the gain from forecast
information (U, F) over the same form without it; and placebos where U and F are shuffled within State and the form is
refitted (50 draws each). Placebo refits of O4a/O4b use a coarser grid, and the real gain is recomputed on that grid.

Outputs: results/dev2/o4/{choice.json, cv.csv, tests.csv, placebo.csv, rows.parquet}; results/tables/dev2_o4_summary.md
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
sys.path.insert(0, os.path.join(HERE, "..", "o4"))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import TAB  # noqa: E402
from refused.report import md_table  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402
import o4_01_assessment as A4  # noqa: E402

SMOKE = "--smoke" in sys.argv
N_PLACEBO = 3 if SMOKE else 50
A_GRID = np.round(np.arange(0, 1.01, 0.1), 2)


def final_t2():
    f = os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json")
    s = next((x for x in json.load(open(f)) if x["tid"] == "T2"), None) if os.path.exists(f) else None
    return (s["final"], int(s["calibration_window"])) if s else ("lgbm", 180)


def build_v2(model, window):
    """Round-1 O4 frame (context terms, outcomes) with U and F from the round-2 forecaster."""
    base = A4.build("lgbm", drop_outcomes=False)  # context terms, M0, raw outcomes; U columns are replaced below
    base = base.drop(columns=["U_raw", "U_p"])
    P = O.load_panel()[["entity", "date", "dev_actual_drawal_gwh", "dev_drawal_schedule_gwh"]]
    pr = D.load_member("T2", model)
    pr = pr[pr.y.notna()].merge(D.scales("T2"), on=["sid", "H"]).sort_values(["sid", "issue_date", "H"]).reset_index(drop=True)
    Q = E2.rolling_level(pr, pr, "C1", True, 1, window)
    pr["U_raw"] = (Q[:, 5] - Q[:, 1]) / pr.scale_mae
    pr["F_med"] = Q[:, 3]
    pr["width_gwh"] = Q[:, 5] - Q[:, 1]
    u = pr[(pr.H == 1) & np.isfinite(pr.U_raw)][["sid", "target_date", "U_raw", "F_med", "width_gwh"]].rename(columns={"sid": "entity", "target_date": "date"})
    Dd = base.merge(u, on=["entity", "date"], how="inner").merge(P, on=["entity", "date"], how="left")
    Dd["U_p"] = A4.ecdf_map(Dd.U_raw.to_numpy(), Dd.loc[Dd.split == "validation", "U_raw"].to_numpy())
    Dd = Dd.dropna(subset=["delta", "lam_n", "R_p", "S_p", "C_p", "U_p", "F_med", "dev_actual_drawal_gwh",
                           "dev_drawal_schedule_gwh", "Y1_raw", "Y2_raw", "Y3_raw"]).reset_index(drop=True)
    Dd["known"] = (Dd.F_med - Dd.dev_drawal_schedule_gwh).abs()
    Dd["surprise"] = (Dd.dev_actual_drawal_gwh - Dd.F_med).abs()
    Dd["share_known"] = Dd.known / (Dd.known + Dd.surprise + 1e-9)
    # surprise measured in widths of the 80 % forecast interval (both in GWh)
    Dd["surprise_z"] = Dd.surprise / Dd.width_gwh.clip(lower=1e-6)
    for sp in ("validation", "dev_test"):
        m = Dd.split == sp
        Dd.loc[m, "Y1"] = Dd[m].groupby("entity").Y1_raw.transform(A4.pct_rank)
        Dd.loc[m, "Y2"] = Dd[m].groupby("entity").Y2_raw.transform(A4.pct_rank)
        Dd.loc[m, "Y3"] = Dd[m].Y3_raw.rank(pct=True)
    Dd["Y"] = Dd[["Y1", "Y2", "Y3"]].mean(axis=1)
    return Dd


def ctx(d, U=None):
    return np.column_stack([d.R_p, d.S_p, d.C_p, d.U_p.to_numpy() if U is None else U])


def tau(d, score):
    return A4.within_decile_tau(score, d.Y.to_numpy(), d.M0.to_numpy())


# --- O4a and O4b: grid search ------------------------------------------------------------------------------------
def score_a(d, w, k, U=None):
    return d.delta.abs().to_numpy() * (d.lam_n.to_numpy() + k * ctx(d, U) @ w)


def score_b(d, a, w, k, U=None, F=None):
    Fm = d.F_med.to_numpy() if F is None else F
    known = np.abs(Fm - d.dev_drawal_schedule_gwh.to_numpy())
    surprise = np.abs(d.dev_actual_drawal_gwh.to_numpy() - Fm)
    return d.lam_n.to_numpy() * (a * known + (1 - a) * surprise) * (1 + k * ctx(d, U) @ w)


def fit_grid(d, form, step=0.1, allow_forecast=True, U=None, F=None):
    best = (-np.inf, None)
    ws = A4.simplex(step)
    a_grid = A_GRID if step == 0.1 else np.array([0, 0.25, 0.5, 0.75, 1.0])
    for k in A4.KAPPAS:
        for w in ws:
            if not allow_forecast and w[3] > 0:
                continue
            if form == "O4a":
                t = tau(d, score_a(d, w, k, U))
                if t > best[0]:
                    best = (t, dict(w=w, k=k))
            else:
                for a in (a_grid if allow_forecast else [1.0]):
                    # without forecast information the split cannot be used: a = 1 and F = S gives |d| again
                    s = score_b(d, a, w, k, U, F) if allow_forecast else d.delta.abs().to_numpy() * d.lam_n.to_numpy() * (1 + k * ctx(d) @ w)
                    t = tau(d, s)
                    if t > best[0]:
                        best = (t, dict(a=a, w=w, k=k))
    return best


def apply_grid(d, form, p, U=None, F=None, allow_forecast=True):
    if form == "O4a":
        return score_a(d, p["w"], p["k"], U)
    if not allow_forecast:
        return d.delta.abs().to_numpy() * d.lam_n.to_numpy() * (1 + p["k"] * ctx(d) @ p["w"])
    return score_b(d, p["a"], p["w"], p["k"], U, F)


# --- O4c: learned ranking multiplier -----------------------------------------------------------------------------
FEATS = ["R_p", "S_p", "C_p", "U_p", "share_known", "surprise_z"]


def deciles(m0):
    return pd.qcut(pd.Series(m0).rank(method="first"), 10, labels=False).to_numpy()


def fit_c(d, depth, allow_forecast=True, U=None, F=None, seed=0):
    X = feats_c(d, U, F, allow_forecast)
    q = deciles(d.M0.to_numpy())
    order = np.argsort(q, kind="stable")
    dm = xgb.DMatrix(X[order], label=d.Y.to_numpy()[order], qid=q[order])
    mono = "(" + ",".join(["1", "1", "1", "1", "0", "0"][:X.shape[1]]) + ")"
    params = dict(objective="rank:pairwise", tree_method="hist", device="cuda", max_depth=depth, eta=0.05,
                  monotone_constraints=mono, seed=seed, subsample=0.8)
    return xgb.train(params, dm, num_boost_round=30 if SMOKE else 300)


def feats_c(d, U=None, F=None, allow_forecast=True):
    cols = [d.R_p.to_numpy(), d.S_p.to_numpy(), d.C_p.to_numpy()]
    if allow_forecast:
        Uc = d.U_p.to_numpy() if U is None else U
        if F is None:
            sk, sz = d.share_known.to_numpy(), d.surprise_z.to_numpy()
        else:
            known = np.abs(F - d.dev_drawal_schedule_gwh.to_numpy())
            surprise = np.abs(d.dev_actual_drawal_gwh.to_numpy() - F)
            sk = known / (known + surprise + 1e-9)
            sz = surprise / np.clip(d.width_gwh.to_numpy(), 1e-6, None)
        cols += [Uc, sk, sz]
    return np.column_stack(cols)


def score_c(d, model, k, U=None, F=None, allow_forecast=True):
    s = model.predict(xgb.DMatrix(feats_c(d, U, F, allow_forecast)))
    s = (s - s.mean()) / (s.std() + 1e-9)
    return d.M0.to_numpy() * np.exp(k * s)


def fit_c_full(d, allow_forecast=True, U=None, F=None):
    fold = D.iso_week_fold(d.date)
    best = (-np.inf, None)
    for depth in (2, 3):
        for k in A4.KAPPAS:
            t = []
            for a in (0, 1):
                fa, fb = d[fold == a].reset_index(drop=True), d[fold != a].reset_index(drop=True)
                ia, ib = fold == a, fold != a
                m = fit_c(fa, depth, allow_forecast, None if U is None else U[ia], None if F is None else F[ia])
                t.append(tau(fb, score_c(fb, m, k, None if U is None else U[ib], None if F is None else F[ib], allow_forecast)))
            if np.mean(t) > best[0]:
                best = (float(np.mean(t)), dict(depth=depth, k=k))
    model = fit_c(d, best[1]["depth"], allow_forecast, U, F)
    return best[0], best[1], model


# --- main ---------------------------------------------------------------------------------------------------------
def cv_grid(V, form):
    fold = D.iso_week_fold(V.date)
    t = []
    for a in (0, 1):
        fa, fb = V[fold == a].reset_index(drop=True), V[fold != a].reset_index(drop=True)
        _, p = fit_grid(fa, form)
        t.append(tau(fb, apply_grid(fb, form, p)))
    return float(np.mean(t))


def main():
    model, window = final_t2()
    Dd = build_v2(model, window)
    V = Dd[Dd.split == "validation"].reset_index(drop=True)
    T = Dd[Dd.split == "dev_test"].reset_index(drop=True)
    if SMOKE:
        V, T = V.iloc[:3000].reset_index(drop=True), T.iloc[:3000].reset_index(drop=True)
    fz = D.frozen("o4", "choice.json") if D.USE_FROZEN else None
    if fz:
        # confirmatory phase: form and settings from the development run; the model is fitted again on the
        # calibration year with those settings (the no-forecast comparator and placebos use the same settings)
        cv, form = {}, fz["chosen"]
        if form == "O4c":
            pc = dict(depth=int(fz["params"]["depth"]), k=float(fz["params"]["k"]))
            mc = fit_c(V, pc["depth"])
    else:
        cv = {"O4a": cv_grid(V, "O4a"), "O4b": cv_grid(V, "O4b")}
        tc, pc, mc = fit_c_full(V)
        cv["O4c"] = tc
        form = max(cv, key=cv.get)
    y = T.Y.to_numpy()
    M0_T = T.M0.to_numpy()
    if form == "O4c" and fz:
        p0, m0 = pc, fit_c(V, pc["depth"], allow_forecast=False)
        A_T = score_c(T, mc, pc["k"])
        A0_T = score_c(T, m0, p0["k"], allow_forecast=False)
        params = dict(**pc)
    elif form == "O4c":
        _, p0, m0 = fit_c_full(V, allow_forecast=False)
        A_T = score_c(T, mc, pc["k"])
        A0_T = score_c(T, m0, p0["k"], allow_forecast=False)
        params = dict(**pc)
    else:
        if fz:
            p = {k: (np.array(v) if isinstance(v, list) else v) for k, v in fz["params"].items()}
        else:
            _, p = fit_grid(V, form)
        _, p0 = fit_grid(V, form, allow_forecast=False)
        A_T, A0_T = apply_grid(T, form, p), apply_grid(T, form, p0, allow_forecast=False)
        params = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in p.items()}
    tests = []
    for yname in ("Y", "Y1", "Y2", "Y3"):
        r = A4.block_diff(T, A_T, M0_T, T[yname].to_numpy())
        tests.append(dict(test="A vs M0", outcome=yname, tau_A=A4.within_decile_tau(A_T, T[yname].to_numpy(), M0_T),
                          tau_M0=A4.within_decile_tau(M0_T, T[yname].to_numpy(), M0_T), **r))
    r = A4.block_diff(T, A_T, A0_T, y)
    tests.append(dict(test="forecast information gain (A vs A without U, F)", outcome="Y", tau_A=tau(T, A_T), tau_M0=tau(T, A0_T), **r))
    # placebos: shuffle U and F within State in both years, refit, score the gain over the no-forecast form
    rng = np.random.default_rng(1)
    shuffle = lambda d, col: d.groupby("entity")[col].transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()  # noqa: E731
    plac = []
    if form == "O4c":
        real_gain = tau(T, A_T) - tau(T, A0_T)
    else:
        _, pc_real = fit_grid(V, form, step=0.25)
        real_gain = tau(T, apply_grid(T, form, pc_real)) - tau(T, A0_T)
    for i in range(N_PLACEBO):
        Uv, Ut, Fv, Ft = shuffle(V, "U_p"), shuffle(T, "U_p"), shuffle(V, "F_med"), shuffle(T, "F_med")
        if form == "O4c" and fz:
            mp = fit_c(V, pc["depth"], U=Uv, F=Fv)
            gain = tau(T, score_c(T, mp, pc["k"], Ut, Ft)) - tau(T, A0_T)
        elif form == "O4c":
            _, pp, mp = fit_c_full(V, U=Uv, F=Fv)
            gain = tau(T, score_c(T, mp, pp["k"], Ut, Ft)) - tau(T, A0_T)
        else:
            _, pp = fit_grid(V, form, step=0.25, U=Uv, F=Fv if form == "O4b" else None)
            gain = tau(T, apply_grid(T, form, pp, Ut, Ft if form == "O4b" else None)) - tau(T, A0_T)
        plac.append(dict(draw=i, gain=gain))
    plac = pd.DataFrame(plac)
    tests.append(dict(test="forecast information vs shuffled placebo", outcome="Y", diff=real_gain,
                      lo=float(plac.gain.quantile(0.95)), hi=np.nan,
                      p_one_sided=float((np.sum(plac.gain >= real_gain) + 1) / (len(plac) + 1))))
    out = D.folder("o4")
    json.dump(dict(forecast_model=model, calibration_window=window, cv_tau=cv, chosen=form, params=params,
                   n_val=len(V), n_dev=len(T)), open(os.path.join(out, "choice.json"), "w"), indent=1, default=str)
    pd.DataFrame([dict(form=k, cv_tau=v) for k, v in cv.items()]).to_csv(os.path.join(out, "cv.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(out, "tests.csv"), index=False)
    plac.to_csv(os.path.join(out, "placebo.csv"), index=False)
    T.assign(A=A_T, A_noF=A0_T).to_parquet(os.path.join(out, "rows.parquet"), index=False)
    L = ["# O4 round 2 — deviation assessment", "",
         f"Forecast source: `{model}` (T2, H = 1, rolling conformal C1, {window}-day window). Fitted on validation "
         f"(n = {len(V):,}); development test (n = {len(T):,}) reported only.", "",
         "Two-fold week cross-validated within-decile tau on validation: " + ", ".join(f"{k} {v:.4f}" for k, v in cv.items())
         + f". Chosen: **{form}** with {params}.", "",
         "| test | outcome | tau(A) | tau(comparator) | difference | 95% CI / placebo 95th pct | p (one-sided) |",
         "|---|---|---|---|---|---|---|"]
    for r in tests:
        L.append(f"| {r['test']} | {r['outcome']} | {r.get('tau_A', np.nan):.4f} | {r.get('tau_M0', np.nan):.4f} | "
                 f"{r['diff']:+.4f} | {A4.ci_text(r)} | {S.fmt_p(r['p_one_sided'])} |")
    open(os.path.join(TAB, f"{D.TAG}_o4_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
