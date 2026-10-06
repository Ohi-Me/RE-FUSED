"""dev2 step 5: combine the forecasters (protocol 06, section 3).

Members: the round-1 learned models, the round-2 models, and the better LA-MCAG variant on validation MASE. They are
lined up on the rows they all cover (validation and development test). Three ways to combine the quantiles:

ens_top     equal-weight average of the k best members (k picked by two-fold ISO-week split of the validation year)
ens_stack   one weight per member for each horizon, weights >= 0 and summing to 1, fitted on validation pinball loss
            with a pull towards equal weights (strength picked by the same week split); fitted on the GPU with torch
ens_online  lag-aware online weights: each issue day the weights come from losses already published by that day
            (target date <= issue day - publication lag), w ~ exp(-eta * discounted loss sum). eta, the forgetting
            factor and pooling (all series or by region) are picked on the validation year; the run then carries on
            through the development test with those settings fixed

Nothing here looks at development-test scores. For the final choice (next step) we also store cross-fitted
validation MASE: best single member, ens_top and ens_stack fitted on one half of the weeks and scored on the other.

Outputs: results/dev2/o2/preds/{tid}__ens_top|ens_stack|ens_online.parquet
         results/dev2/o2/logs/ensemble_{tid}.json, ens_online_weights_{tid}.parquet
Usage:   python d2_05_ensemble.py [T1,T2,...] [--smoke]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402

SMOKE = "--smoke" in sys.argv
ROUND1 = ["lgbm", "bilstm", "tft", "patchtst", "chronos"]
ROUND2 = ["xgb", "nhits", "tide", "bitcn", "nbeatsx", "chronos2", "chronos_base"]
ETAS = [0.5, 1.0, 2.0, 4.0]
GAMMAS = [1.0, 0.99, 0.97]
POOLS = ["all", "region"]
LAMBDAS = [0.0, 0.1, 1.0]
MED = D.QN.index("q50")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def o3_member(tid):
    """The LA-MCAG variant with the lower validation MASE (round 1 or staleness-matched dropout)."""
    best, best_score, scores = None, np.inf, {}
    for name in ("mcag_instance", "mcag_instance_sd"):
        d = D.load_member(tid, name)
        if d is None:
            continue
        v = d[(d.split == "validation") & d.y.notna()].merge(D.scales(tid), on=["sid", "H"])
        scores[name] = D.mase(v, v.q50.to_numpy())
        if scores[name] < best_score:
            best, best_score = name, scores[name]
    return best, scores


def mase_on(base, q50, mask):
    return D.mase(base[mask], q50[mask])


def pinball_on(base, Q, mask):
    f = base[mask]
    return float(D.pinball_rows(f.y.to_numpy(), Q[mask], f.scale_mae.to_numpy()).mean())


# ----------------------------------------------------------------------------------------------------- ens_top
def rank_members(base, arr, mask):
    scores = np.array([mase_on(base, arr[:, m, MED], mask) for m in range(arr.shape[1])])
    return np.argsort(scores), scores


def top_k(arr, order, k):
    return np.sort(arr[:, order[:k], :].mean(axis=1), axis=1)


def pick_k(base, arr, fit, score, ks):
    order, _ = rank_members(base, arr, fit)
    res = {k: mase_on(base, top_k(arr, order, k)[:, MED], score) for k in ks}
    return res, order


# --------------------------------------------------------------------------------------------------- ens_stack
def fit_stack(arr, y, scale, lam, steps=400):
    X = torch.tensor(arr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    st = torch.tensor(scale, dtype=torch.float32, device=DEV)
    tau = torch.tensor(O.QUANTILES, dtype=torch.float32, device=DEV)
    M = X.shape[1]
    theta = torch.zeros(M, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    for _ in range(40 if SMOKE else steps):
        w = torch.softmax(theta, 0)
        Q = (X * w[None, :, None]).sum(1)
        u = (yt[:, None] - Q) / st[:, None]
        loss = torch.maximum(tau * u, (tau - 1) * u).mean() + lam * ((w - 1.0 / M) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(theta, 0).detach().cpu().numpy()


def stack_forecast(base, arr, fit_mask, lam):
    """Fit weights per horizon on fit_mask rows; return combined quantiles for all rows and the weights."""
    Q = np.full((len(base), arr.shape[2]), np.nan)
    weights = {}
    for H in O.HORIZONS:
        h = (base.H == H).to_numpy()
        m = fit_mask & h
        w = fit_stack(arr[m], base.y.to_numpy()[m], base.scale_mae.to_numpy()[m], lam)
        weights[H] = w
        Q[h] = np.sort((arr[h] * w[None, :, None]).sum(axis=1), axis=1)
    return Q, weights


# -------------------------------------------------------------------------------------------------- ens_online
def online_forecast(base, arr, Ly, eta, gamma, pool, region_of):
    """Lag-aware exponentiated-gradient weights. Only losses with target date <= issue day - lag are used."""
    n, M, _ = arr.shape
    y, scale = base.y.to_numpy(), base.scale_mae.to_numpy()
    has_y = np.isfinite(y)
    loss = np.full((n, M), np.nan)
    for m in range(M):
        loss[has_y, m] = D.pinball_rows(y[has_y], arr[has_y, m, :], scale[has_y])
    group = base.sid.map(region_of).fillna("?").to_numpy() if pool == "region" else np.array(["all"] * n)
    day0 = base.target_date.min()
    tday = (base.target_date - day0).dt.days.to_numpy()
    cut = ((base.issue_date - day0).dt.days - Ly).to_numpy()
    W = np.full((n, M), 1.0 / M)
    trace = []
    for H in O.HORIZONS:
        for g in np.unique(group):
            rows = np.where((base.H.to_numpy() == H) & (group == g))[0]
            lr = rows[has_y[rows]]
            if len(lr) == 0:
                continue
            daily = pd.DataFrame(loss[lr], index=tday[lr]).groupby(level=0).mean().sort_index()
            days, L = daily.index.to_numpy(), daily.to_numpy()
            C = np.zeros_like(L)
            acc = np.zeros(M)
            for i in range(len(days)):
                gap = days[i] - days[i - 1] if i else 0
                acc = acc * (gamma ** gap) + L[i]
                C[i] = acc
            q = np.searchsorted(days, cut[rows], side="right") - 1
            ok = q >= 0
            Cq = np.zeros((len(rows), M))
            Cq[ok] = C[q[ok]] * (gamma ** (cut[rows][ok] - days[q[ok]]))[:, None]
            z = -eta * (Cq - Cq.min(axis=1, keepdims=True))
            w = np.exp(z)
            W[rows] = w / w.sum(axis=1, keepdims=True)
            first_issue = base.issue_date.to_numpy()[rows]
            trace.append(pd.DataFrame(W[rows], columns=[f"w{m}" for m in range(M)]).assign(
                issue_date=first_issue, H=H, group=g).drop_duplicates(["issue_date", "H", "group"]))
    Q = np.sort((arr * W[:, :, None]).sum(axis=1), axis=1)
    return Q, pd.concat(trace, ignore_index=True) if trace else pd.DataFrame()


# -------------------------------------------------------------------------------------------------------- main
def run(tid):
    t0 = time.time()
    Ly = D.lag_of(tid)
    # confirmatory phase: members, k, pull strength and online settings come from the development run; the weights
    # themselves are fitted again on the calibration year by the same procedure
    fl = D.frozen("o2", "logs", f"ensemble_{tid}.json") if D.USE_FROZEN else None
    if fl:
        o3_name, o3_scores, names = fl["o3_member"], {}, fl["members"]
    else:
        o3_name, o3_scores = o3_member(tid)
        names = ROUND1 + ROUND2 + ([o3_name] if o3_name else [])
    base, arr, used = D.member_stack(tid, names)
    if base is None or len(used) < 2:
        print(tid, "not enough members, skipped", flush=True)
        return
    region_of = D.regions(tid)
    val = ((base.split == "validation") & base.y.notna()).to_numpy()
    fold = D.iso_week_fold(base.target_date)
    fA, fB = val & (fold == 0), val & (fold == 1)
    M = len(used)
    ks = [fl["ens_top"]["k"]] if fl else [k for k in range(2, min(6, M) + 1)]
    log = dict(tid=tid, members=used, o3_member=o3_name, o3_val_mase=o3_scores, frozen=bool(fl), rows=len(base),
               val_rows=int(val.sum()), dev_rows=int(((base.split == "dev_test") & base.y.notna()).sum()))
    order_full, member_scores = rank_members(base, arr, val)
    log["member_val_mase"] = dict(zip(used, map(float, member_scores)))

    # best single member, cross-fitted: pick on one half of the weeks, score on the other
    q50_cf = np.full(len(base), np.nan)
    for fit, score in ((fA, fB), (fB, fA)):
        o, _ = rank_members(base, arr, fit)
        q50_cf[score] = arr[score, o[0], MED]
    cf = {"best_single": mase_on(base, q50_cf, val)}
    log["best_single"] = used[order_full[0]]

    # ens_top: k by two-fold week split, then refit the ranking on the whole validation year
    cv_k = {k: [] for k in ks}
    q50_cf = np.full(len(base), np.nan)
    for fit, score in ((fA, fB), (fB, fA)):
        res, o = pick_k(base, arr, fit, score, ks)
        for k in ks:
            cv_k[k].append(res[k])
        k_in, _ = min(((k, mase_on(base, top_k(arr, o, k)[:, MED], fit)) for k in ks), key=lambda x: x[1])
        q50_cf[score] = top_k(arr, o, k_in)[score, MED]
    k_best = min(ks, key=lambda k: np.mean(cv_k[k]))
    Q_top = top_k(arr, order_full, k_best)
    cf["ens_top"] = mase_on(base, q50_cf, val)
    log["ens_top"] = dict(k=k_best, cv_mase={k: float(np.mean(v)) for k, v in cv_k.items()},
                          members=[used[i] for i in order_full[:k_best]])

    # ens_stack: pull strength by two-fold week split (pinball), weights refit on the whole validation year
    cv_l = {}
    for lam in ([fl["ens_stack"]["lambda_"]] if fl else LAMBDAS):
        s = []
        for fit, score in ((fA, fB), (fB, fA)):
            Qf, _ = stack_forecast(base, arr, fit, lam)
            s.append(pinball_on(base, Qf, score))
        cv_l[lam] = float(np.mean(s))
    lam_best = min(cv_l, key=cv_l.get)
    Q_stack, w_stack = stack_forecast(base, arr, val, lam_best)
    q50_cf = np.full(len(base), np.nan)
    for fit, score in ((fA, fB), (fB, fA)):
        Qf, _ = stack_forecast(base, arr, fit, lam_best)
        q50_cf[score] = Qf[score, MED]
    cf["ens_stack"] = mase_on(base, q50_cf, val)
    log["ens_stack"] = dict(lambda_=lam_best, cv_pinball=cv_l,
                            weights={H: dict(zip(used, map(float, w))) for H, w in w_stack.items()})

    # ens_online: settings picked on the validation-year run (it only ever uses published losses)
    grid = {}
    best_key, best_val, best_Q, best_trace = None, None, None, None
    if fl:
        e = fl["ens_online"]
        settings = [(e["eta"], e["gamma"], e["pool"])]
    else:
        settings = [(eta, gamma, pool) for eta in (ETAS[:1] if SMOKE else ETAS)
                    for gamma in (GAMMAS[:2] if SMOKE else GAMMAS) for pool in POOLS]
    for eta, gamma, pool in settings:
        Qo, trace = online_forecast(base, arr, Ly, eta, gamma, pool, region_of)
        sc = (mase_on(base, Qo[:, MED], val), pinball_on(base, Qo, val))
        grid[f"eta={eta},gamma={gamma},pool={pool}"] = dict(val_mase=sc[0], val_pinball=sc[1])
        better = best_val is None or sc[0] < best_val[0] * 0.995 or \
            (sc[0] <= best_val[0] * 1.005 and sc[1] < best_val[1])
        if better:
            best_key, best_val, best_Q, best_trace = (eta, gamma, pool), sc, Qo, trace
    cf["ens_online"] = best_val[0]
    log["ens_online"] = dict(eta=best_key[0], gamma=best_key[1], pool=best_key[2], grid=grid)
    log["crossfit_val_mase"] = cf

    out = D.folder("o2", "preds")
    for name, Q in (("ens_top", Q_top), ("ens_stack", Q_stack), ("ens_online", best_Q)):
        fr = base[D.KEYS].copy()
        fr[D.QN] = Q
        fr["model"] = name
        fr.to_parquet(os.path.join(out, f"{tid}__{name}.parquet"), index=False)
    if best_trace is not None and len(best_trace):
        best_trace.rename(columns={f"w{m}": used[m] for m in range(M)}).to_parquet(
            os.path.join(D.folder("o2", "logs"), f"ens_online_weights_{tid}.parquet"), index=False)
    log["seconds"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(D.folder("o2", "logs"), f"ensemble_{tid}.json"), "w"), indent=1, default=str)
    print(f"{tid} ensembles done: members {M}, k={k_best}, lambda={lam_best}, online {best_key}; "
          f"cross-fitted val MASE {json.dumps({k: round(v, 4) for k, v in cf.items()})} [{log['seconds']}s]", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tids = args[0].split(",") if args else (["T5"] if SMOKE else list(O.TARGETS))
    for t in tids:
        run(t)
