"""dev2 step 9: risk-aware scheduling, round 2 (protocol 06, section 6).

Same stylised no-arbitrage settlement, hard limits, published prices and J = 1/2 mean + 1/2 CVaR90 of system-wide
daily regret as in round 1 (o5_01_scheduling.py). Changes:

* forecast source: the final T2 forecaster of round 2 (H = 1, rolling conformal C1 with its chosen window); U from it
* A4s  smooth coupling per State-day: beta = sigmoid(b0 + bU*(U - 0.5) + bS*(S - 0.5)), tuned on validation J
* A5   system-tail scheduling: all State schedules of a day are chosen together to minimise
       (1 - beta)*mean + beta*CVaR90 of the SYSTEM regret over joint scenarios. Joint scenarios keep the dependence
       between States: scenario j re-uses the forecast-quantile ranks (PIT values) that all States had together on one
       of the 180 most recent published days (target date <= decision day - 2), mapped onto today's forecast quantiles.
       Solved for all days at once with Adam on the GPU.
* A5c  A5 with beta for the day coupled to the mean U and mean S across States (same smooth form as A4s)
* chain check: P, A2 and A5 re-tuned with the round-1 selected forecaster, to see what the better forecaster adds

All tuning uses the validation year only; the development test is reported only.
Outputs: results/dev2/o5/{decisions.parquet, tuning.csv, metrics.csv, tests.csv, chain.csv}; results/tables/dev2_o5_summary.md
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
for sub in ("o2", "o4", "o5"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import TAB  # noqa: E402
from refused.report import md_table  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402
import o4_01_assessment as A4  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402

SMOKE = "--smoke" in sys.argv
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B0, BU, BS = [-4, -2, -1, 0, 1], [0, 1, 2, 4], [0, 1, 2, 4]
BETAS5 = [0, 0.25, 0.5, 0.75, 1.0]
N_HIST = 180
STEPS = 30 if SMOKE else 300


def final_t2():
    f = os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json")
    r1 = json.load(open(os.path.join(D.DEV1, "o2", "eval", "selection.json")))
    r1_t2 = next(x["selected"] for x in r1 if x["tid"] == "T2")
    s = next((x for x in json.load(open(f)) if x["tid"] == "T2"), None) if os.path.exists(f) else None
    return (s["final"], int(s["calibration_window"]), r1_t2) if s else (r1_t2, 180, r1_t2)


def build_v2(model, window):
    """Round-1 O5 frame (prices, limits, stress) with forecast quantiles and U from `model`."""
    F = O5.build("lgbm")
    pr = D.load_member("T2", model)
    pr = pr[pr.y.notna()].merge(D.scales("T2"), on=["sid", "H"]).sort_values(["sid", "issue_date", "H"]).reset_index(drop=True)
    Q = E2.rolling_level(pr, pr, "C1", True, 1, window)
    q = pd.DataFrame(Q, columns=D.QN)
    q["entity"], q["date"], q["H"] = pr.sid, pr.target_date, pr.H
    q["U_raw"] = (Q[:, 5] - Q[:, 1]) / pr.scale_mae.to_numpy()
    q = q[q.H == 1].drop(columns="H")
    F = F.drop(columns=D.QN + ["U_p"]).merge(q, on=["entity", "date"], how="inner")
    F["U_p"] = A4.ecdf_map(F.U_raw.to_numpy(), F.loc[F.split == "validation", "U_raw"].to_numpy())
    F["U_p"] = F.U_p.fillna(0.5)
    F = F.dropna(subset=D.QN).sort_values(["date", "entity"]).reset_index(drop=True)
    if SMOKE:  # 60 validation days and 60 test days are enough to check the code
        days = pd.Series(np.sort(F.date.unique()))
        keep = pd.concat([days[days <= O.VAL_END].iloc[:60], days[days > O.VAL_END].iloc[:60]])
        F = F[F.date.isin(keep)].reset_index(drop=True)
    return F


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------------------------------------------------------------ A5: joint scenarios on GPU
def regret_t(S_, X, p, lam):
    D_ = X - S_
    b = 0.1 * S_.abs()
    Dp, Dm = torch.clamp(D_, min=0), torch.clamp(-D_, min=0)
    hi, lo = torch.maximum(lam, p), torch.minimum(lam, p)
    over = hi * torch.minimum(Dp, b) + 1.2 * hi * torch.clamp(Dp - b, min=0)
    under = 0.9 * lo * torch.minimum(Dm, b) + 0.8 * lo * torch.clamp(Dm - b, min=0)
    return (p * (S_ - X) + over - under) * O5.CR


class JointProblem:
    """Everything A5 needs, laid out as (days, States) arrays once, so each beta setting is one GPU solve."""

    def __init__(self, F, SC, seed=0):
        self.dates = np.sort(F.date.unique())
        self.ents = sorted(F.entity.unique())
        di = {d: i for i, d in enumerate(self.dates)}
        ei = {e: i for i, e in enumerate(self.ents)}
        self.r = F.date.map(di).to_numpy()
        self.c = F.entity.map(ei).to_numpy()
        nd, ne = len(self.dates), len(self.ents)
        self.mask = np.zeros((nd, ne), bool)
        self.mask[self.r, self.c] = True
        sc = np.full((nd, ne, 99), np.nan)
        sc[self.r, self.c] = SC
        pit = np.full((nd, ne), np.nan)
        X = F.X.to_numpy()
        pit[self.r, self.c] = np.clip((SC < X[:, None]).sum(axis=1) / 100.0, 0.01, 0.99)
        rng = np.random.default_rng(seed)
        day_num = (pd.to_datetime(self.dates) - pd.Timestamp(self.dates[0])).days.to_numpy()
        levels = np.empty((nd, N_HIST, ne))
        for i in range(nd):
            hist = np.where(day_num <= day_num[i] - 2)[0][-N_HIST:]
            lv = rng.uniform(0.01, 0.99, (N_HIST, ne))
            if len(hist) >= 30:
                take = hist[rng.integers(0, len(hist), N_HIST)] if len(hist) < N_HIST else hist
                ph = pit[take]
                lv = np.where(np.isfinite(ph), ph, lv)
            levels[i] = lv
        idx = np.clip(np.round(levels * 100).astype(int) - 1, 0, 98)          # (days, J, States)
        sc0 = np.nan_to_num(sc)
        self.Xs = np.empty((nd, N_HIST, ne), np.float32)
        cols = np.arange(ne)[None, :]
        for i in range(nd):  # scenario value of each State = its forecast quantile at the borrowed rank
            self.Xs[i] = sc0[i][cols, idx[i]]
        self.med = np.nan_to_num(sc[:, :, 49])
        self.spread = np.nan_to_num(sc[:, :, 89] - sc[:, :, 9]) + 1e-3
        p = np.zeros((nd, ne))
        lam = np.zeros((nd, ne))
        p[self.r, self.c], lam[self.r, self.c] = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
        self.p, self.lam = p, lam

    def solve(self, beta_day):
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
        Xs, med, spread, m = t(self.Xs), t(self.med), t(self.spread), t(self.mask.astype(float))
        p, lam = t(self.p)[:, None, :], t(self.lam)[:, None, :]
        beta = t(np.broadcast_to(np.asarray(beta_day, float), (len(self.dates),)).copy())
        delta = torch.zeros_like(med, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=0.05)
        k = max(1, int(np.ceil(0.1 * N_HIST)))
        for _ in range(STEPS):
            S_ = (med + delta * spread)[:, None, :]
            sysr = (regret_t(S_, Xs, p, lam) * m[:, None, :]).sum(dim=2)          # (days, J)
            cvar = torch.topk(sysr, k, dim=1).values.mean(dim=1)
            obj = ((1 - beta) * sysr.mean(dim=1) + beta * cvar).sum()
            opt.zero_grad()
            obj.backward()
            opt.step()
        S_best = (med + delta * spread).detach().cpu().numpy()
        return S_best[self.r, self.c]


# ------------------------------------------------------------------------------------------------------ scoring
def arm_metrics(dec, arms):
    metrics, daily = [], {}
    for sp in ("validation", "dev_test"):
        m = dec.split == sp
        d = dec[m]
        for arm in arms:
            sysd = d.groupby("date")[f"regret_{arm}"].sum().sort_index()
            daily[(sp, arm)] = sysd
            v = np.sort(sysd.to_numpy())
            cv = v[-max(1, int(np.ceil(0.1 * len(v)))):].mean()
            env_ex = float(((d[f"S_{arm}"] < d.S_min - 1e-9) | (d[f"S_{arm}"] > d.S_max + 1e-9)).mean())
            metrics.append(dict(split=sp, arm=arm, system_mean_crore=float(v.mean()), system_cvar90_crore=float(cv),
                                J=0.5 * v.mean() + 0.5 * cv, deviation_energy_gwh_per_state_day=float((d.X - d[f"S_{arm}"]).abs().mean()),
                                hard_limit_envelope_exceedance_share=env_ex, n_state_days=int(m.sum())))
    return pd.DataFrame(metrics), daily


def paired(daily, pairs):
    tests = []
    for a, b in pairs:
        if ("dev_test", a) not in daily or ("dev_test", b) not in daily:
            continue
        da, db = daily[("dev_test", a)], daily[("dev_test", b)]
        dd = (da - db).dropna().to_numpy()
        ci = S.block_ci(dd, 7, n_boot=500 if SMOKE else 2000, seed=0)
        tests.append(dict(a=a, b=b, metric="mean system regret", mean_diff=float(dd.mean()), lo=ci["lo"], hi=ci["hi"],
                          p_a_better=E2.hln(dd, 1)[1], n_days=len(dd)))
        tail = (da > da.quantile(0.9)) | (db > db.quantile(0.9))
        dt = (da[tail] - db[tail]).to_numpy()
        ci2 = S.block_ci(dt, 3, n_boot=500 if SMOKE else 2000, seed=0) if len(dt) > 10 else dict(lo=np.nan, hi=np.nan, p_less=np.nan)
        tests.append(dict(a=a, b=b, metric="tail days (either arm above its 90th pct)", mean_diff=float(np.mean(dt)) if len(dt) else np.nan,
                          lo=ci2["lo"], hi=ci2["hi"], p_a_better=ci2.get("p_less", np.nan), n_days=len(dt)))
    T = pd.DataFrame(tests)
    if len(T):
        fam = T.metric == "mean system regret"
        T.loc[fam, "p_holm"] = S.holm(T.loc[fam, "p_a_better"].fillna(1).to_numpy())
    return T


def run_source(model, window, full=True):
    """Tune and run the arms for one forecast source. full=False runs only P, A2 and A5 (chain check)."""
    F = build_v2(model, window)
    SC = O5.scenarios(F[D.QN].to_numpy())
    ph, lh = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
    val = (F.split == "validation").to_numpy()
    tuning = []
    # confirmatory phase: every arm uses the setting picked in development (each "grid" below has one entry)
    fz = D.frozen("o5", "params.json")["params"] if D.USE_FROZEN else None

    def score(S_raw, mask):
        fr = F[mask]
        return O5.J(O5.daily_system(fr, O5.project(fr, S_raw[mask]))[0])

    # A2: constant beta and rho (round-1 grid), re-tuned for this forecaster
    grid2 = {}
    grid_a2 = [(fz["A2"]["beta"], fz["A2"]["rho"])] if fz else list(itertools.product(O5.BETAS, O5.RHOS))[: (4 if SMOKE else None)]
    for b_, r_ in grid_a2:
        grid2[(b_, r_)] = score(O5.decide(SC, ph, lh, b_, r_), val)
        tuning.append(dict(source=model, arm="A2", beta=b_, rho=r_, J_val=grid2[(b_, r_)]))
    b2, r2 = min(grid2, key=grid2.get)
    arms = {"H_final": F.S_hist.to_numpy(), "H_pub": O5.project(F, F.sched_pub.fillna(F.q50).to_numpy()),
            "P": O5.project(F, F.q50.to_numpy()), "A2": O5.project(F, O5.decide(SC, ph, lh, b2, r2))}
    params = dict(A2=dict(beta=b2, rho=r2))
    jp = JointProblem(F, SC)
    grid5 = {}
    for b_ in ([fz["A5"]["beta"]] if fz else (BETAS5[:2] if SMOKE else BETAS5)):
        S5 = jp.solve(b_)
        grid5[b_] = score(S5, val)
        tuning.append(dict(source=model, arm="A5", beta=b_, J_val=grid5[b_]))
    b5 = min(grid5, key=grid5.get)
    arms["A5"] = O5.project(F, jp.solve(b5))
    params["A5"] = dict(beta=b5)
    if full:
        grid4 = {}
        combos = list(itertools.product(B0, BU, BS))[: (3 if SMOKE else None)]
        combos4 = [(fz["A4s"]["b0"], fz["A4s"]["bU"], fz["A4s"]["bS"])] if fz else combos
        combos5c = [(fz["A5c"]["b0"], fz["A5c"]["bU"], fz["A5c"]["bS"])] if fz else combos
        for b0, bu, bs in combos4:
            beta = sigmoid(b0 + bu * (F.U_p.to_numpy() - 0.5) + bs * (F.S_p.to_numpy() - 0.5))
            grid4[(b0, bu, bs)] = score(O5.decide(SC, ph, lh, beta, r2), val)
            tuning.append(dict(source=model, arm="A4s", b0=b0, bU=bu, bS=bs, rho=r2, J_val=grid4[(b0, bu, bs)]))
        b0, bu, bs = min(grid4, key=grid4.get)
        arms["A4s"] = O5.project(F, O5.decide(SC, ph, lh, sigmoid(b0 + bu * (F.U_p - 0.5) + bs * (F.S_p - 0.5)).to_numpy(), r2))
        params["A4s"] = dict(b0=b0, bU=bu, bS=bs, rho=r2)
        day = F.groupby("date")[["U_p", "S_p"]].mean().reindex(jp.dates)
        grid5c = {}
        for c0, cu, cs in combos5c:
            beta_day = sigmoid(c0 + cu * (day.U_p.to_numpy() - 0.5) + cs * (day.S_p.to_numpy() - 0.5))
            grid5c[(c0, cu, cs)] = score(jp.solve(beta_day), val)
            tuning.append(dict(source=model, arm="A5c", b0=c0, bU=cu, bS=cs, J_val=grid5c[(c0, cu, cs)]))
        c0, cu, cs = min(grid5c, key=grid5c.get)
        arms["A5c"] = O5.project(F, jp.solve(sigmoid(c0 + cu * (day.U_p.to_numpy() - 0.5) + cs * (day.S_p.to_numpy() - 0.5))))
        params["A5c"] = dict(b0=c0, bU=cu, bS=cs)
    dec = F[["entity", "date", "split", "X", "p", "lam", "S_hist", "S_min", "S_max", "dS_max"]].copy()
    for arm, S_ in arms.items():
        dec[f"S_{arm}"] = S_
        dec[f"regret_{arm}"] = O5.regret(S_, F.X.to_numpy(), F.p.to_numpy(), F.lam.to_numpy())
    M, daily = arm_metrics(dec, list(arms))
    return dict(F=F, dec=dec, metrics=M.assign(source=model), daily=daily, tuning=tuning, params=params)


def main():
    t0 = time.time()
    model, window, r1 = final_t2()
    main_run = run_source(model, window, full=True)
    tests = paired(main_run["daily"], [("A5", "A2"), ("A5", "P"), ("A5", "H_pub"), ("A4s", "A2"), ("A4s", "H_pub"),
                                       ("A5c", "A5"), ("A4s", "P"), ("A2", "P"), ("P", "H_pub")])
    chain = [main_run["metrics"]]
    tuning = list(main_run["tuning"])
    chain_tests = pd.DataFrame()
    if r1 != model:
        other = run_source(r1, 180, full=False)
        chain.append(other["metrics"])
        tuning += other["tuning"]
        # N12: the same arm with the round-2 forecaster against the round-1 forecaster, day by day
        both = {("dev_test", f"{arm} final"): main_run["daily"][("dev_test", arm)] for arm in ("P", "A2", "A5")}
        both.update({("dev_test", f"{arm} round1"): other["daily"][("dev_test", arm)] for arm in ("P", "A2", "A5")})
        chain_tests = paired(both, [(f"{arm} final", f"{arm} round1") for arm in ("P", "A2", "A5")])
    out = D.folder("o5")
    main_run["dec"].to_parquet(os.path.join(out, "decisions.parquet"), index=False)
    pd.DataFrame(tuning).to_csv(os.path.join(out, "tuning.csv"), index=False)
    main_run["metrics"].to_csv(os.path.join(out, "metrics.csv"), index=False)
    tests.to_csv(os.path.join(out, "tests.csv"), index=False)
    CH = pd.concat(chain)
    CH.to_csv(os.path.join(out, "chain.csv"), index=False)
    chain_tests.to_csv(os.path.join(out, "chain_tests.csv"), index=False)
    json.dump(dict(forecast_model=model, calibration_window=window, round1_model=r1, params=main_run["params"],
                   seconds=round(time.time() - t0)), open(os.path.join(out, "params.json"), "w"), indent=1, default=str)
    L = ["# O5 round 2 — scheduling with the round-2 forecaster (stylised settlement)", "",
         f"Forecast source: `{model}` (window {window} days). Tuned on validation: {json.dumps(main_run['params'])}.", "",
         md_table(main_run["metrics"], 4), "", "Development-test paired tests (system-wide daily regret, crore; negative = a better):", "",
         md_table(tests, 4), "", "Chain check — same arms with the round-1 selected forecaster:", "",
         md_table(CH[CH.arm.isin(["P", "A2", "A5"])], 4), ""]
    open(os.path.join(TAB, f"{D.TAG}_o5_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
