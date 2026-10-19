"""Source fusion for day-ahead scheduling (scheduling follow-up 2; protocol codes/design/10_source_fusion_protocol.md).

Two schedules are available at decision time: the settlement-aware forecast rule A2 (from the calibrated forecast
distribution) and the State's own recorded schedule of day t-2 (H_pub), which carries what the State knows about its
contracts and plans. Neither is best every year. This script blends them, S = w A2 + (1 - w) H_pub, with the weight
chosen in four ways, and scores each blend under the same settlements as the paper:

  G0  the better single source on the fitting year (no blending)
  G1  one weight for all States, chosen on the fitting year (grid 0, 0.05, ..., 1)
  G2  one weight per State, chosen on the fitting year
  G3  online weights per State: exponentially weighted average over 11 blends (w = 0, 0.1, ..., 1) with Fixed Share,
      each blend's loss known only two days later (publication delay); eta and the share rate chosen on the fitting year
  G4  weight w(z) from a logistic model of decision-time features (the paper's features plus the two sources' recent
      relative regret, lagged two days), trained on the settlement cost of the fitting year

Every variant is scored projected onto the schedule limits (as the paper's rules) and not projected (as the State's
schedule actually is); each version is tuned on its own objective.

Phases (REFUSED_PHASE): dev fits FY2023-24 and scores FY2024-25; confirm fits FY2024-25 and scores 1 Apr 2025 -
31 Aug 2026 and is refused unless the protocol's SHA-256 equals codes/design/SOURCE_FUSION_SHA256.txt.
Outputs: results/extra/source_fusion_<phase>.json (never overwritten).
Usage:   python codes/scripts/extra/x09_source_fusion.py    (hpc/extra_source_fusion.pbs, CPU queue)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
for sub in ("o2", "o4", "o5", "dev2", "extra"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from refused import dev2 as D  # noqa: E402
from refused.paths import RES  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402
import d2_09_o5 as R2  # noqa: E402
import x06_dsm_regulation as X6  # noqa: E402
import x07_fusion_scheduler as X7  # noqa: E402

PHASE = os.environ.get("REFUSED_PHASE", "dev")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROTO = os.path.join(ROOT, "codes", "design", "10_source_fusion_protocol.md")
LOCK = os.path.join(ROOT, "codes", "design", "SOURCE_FUSION_SHA256.txt")
OUT = os.path.join(RES, "extra")
WGRID = np.round(np.arange(0, 1.0001, 0.05), 2)
BLENDS = np.round(np.arange(0, 1.0001, 0.1), 1)
ETAS = (0.1, 0.3, 1.0, 3.0, 10.0)
SHARES = (0.0, 0.01, 0.05, 0.2)
SEEDS = (0, 1, 2)
LAG = 2                                              # a day's settlement is known two days later


def guard():
    if PHASE != "confirm":
        return
    if not os.path.exists(LOCK):
        raise SystemExit("protocol not frozen: no " + LOCK)
    want = open(LOCK).read().split()[0]
    have = hashlib.sha256(open(PROTO, "rb").read()).hexdigest()
    if want != have:
        raise SystemExit(f"protocol hash mismatch: {have} != {want}")


def online(F, Rb, fitm, eta, share):
    """Fixed Share over the blends, per State, losses scaled by the State's mean fitting-year loss; returns w per row."""
    w_out = np.empty(len(F))
    K = Rb.shape[1]
    for _, idx in F.groupby("entity").indices.items():
        idx = idx[np.argsort(F.date.to_numpy()[idx])]
        scale = max(np.abs(Rb[idx][fitm[idx]]).mean(), 1e-9) if fitm[idx].any() else max(np.abs(Rb[idx]).mean(), 1e-9)
        logw = np.zeros(K)
        for j, i in enumerate(idx):
            if j >= LAG:                              # the loss of day j - LAG is now published
                logw = logw - eta * Rb[idx[j - LAG]] / scale
                p = np.exp(logw - logw.max())
                p /= p.sum()
                p = (1 - share) * p + share / K
                logw = np.log(p)
            p = np.exp(logw - logw.max())
            p /= p.sum()
            w_out[i] = float(p @ BLENDS)
    return w_out


def learned_weight(Z, A, H, X, p, lam, smin, smax, fitm, days, sname, project, seed):
    torch.manual_seed(seed)
    T = lambda a: torch.tensor(np.asarray(a, np.float32))  # noqa: E731
    z, a, h, x, pt, lt, lo, hi = map(T, (Z, A, H, X, p, lam, smin, smax))
    order = np.argsort(days[fitm])
    cut = days[fitm][order[int(0.8 * fitm.sum())]]
    tr, es = torch.tensor(fitm & (days < cut)), torch.tensor(fitm & (days >= cut))
    lin = torch.nn.Linear(z.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.01, weight_decay=1e-4)
    best, state, bad = np.inf, None, 0
    for _ in range(800):
        w = torch.sigmoid(lin(z).squeeze(1))
        S_ = w * a + (1 - w) * h
        if project:
            S_ = torch.minimum(torch.maximum(S_, lo), hi)
        R = X7.regret_t(S_, x, pt, lt, sname)
        opt.zero_grad()
        R[tr].mean().backward()
        opt.step()
        v = float(R[es].mean())
        if v < best - 1e-6:
            best, bad, state = v, 0, {k: t.detach().clone() for k, t in lin.state_dict().items()}
        else:
            bad += 1
            if bad > 60:
                break
    lin.load_state_dict(state)
    with torch.no_grad():
        return torch.sigmoid(lin(z).squeeze(1)).numpy().astype(float)


def main():
    guard()
    t0 = time.time()
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    F = R2.build_v2(sel["final"], int(sel["calibration_window"])).reset_index(drop=True)
    SC = O5.scenarios(F[D.QN].to_numpy())
    Zdf, _ = X7.features(F, SC)
    fitm, test = (F.split == "validation").to_numpy(), (F.split == "dev_test").to_numpy()
    X, p, lam = F.X.to_numpy(), F.p.to_numpy(), F.lam.to_numpy()
    ph, lh = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
    smin, smax = F.S_min.to_numpy(), F.S_max.to_numpy()
    days = F.date.values.astype("datetime64[D]").astype(np.int64)
    ent = F.entity.to_numpy()
    fz = D.frozen("o5", "params.json")["params"]
    H = F.sched_pub.fillna(F.q50).to_numpy()
    res = dict(phase=PHASE, protocol_sha256=hashlib.sha256(open(PROTO, "rb").read()).hexdigest(),
               note="exploratory follow-up; see protocol", n_fit=int(fitm.sum()), n_test=int(test.sum()), settlements={})
    for sname in ("stylised", "dsm_nominal"):
        A = X6.decide_s(SC, ph, lh, fz["A2"]["beta"], fz["A2"]["rho"], sname)      # A2 before projection
        out = {}
        for project in (True, False):
            tag = "projected" if project else "not projected"
            fin = (lambda s: O5.project(F, s)) if project else (lambda s: s)
            R = lambda s: X6.regret_s(fin(s), X, p, lam, sname)  # noqa: E731
            rA, rH = R(A), R(H)
            daily = {"A2": X6.system_daily(F, rA, test), "H_pub": X6.system_daily(F, rH, test)}
            info = {}
            # G0: better single source on the fitting year
            g0 = "A2" if rA[fitm].sum() <= rH[fitm].sum() else "H_pub"
            daily["G0"], info["G0"] = daily[g0], dict(source=g0)
            # G1: one weight
            costs = [R(w * A + (1 - w) * H)[fitm].sum() for w in WGRID]
            w1 = float(WGRID[int(np.argmin(costs))])
            daily["G1"], info["G1"] = X6.system_daily(F, R(w1 * A + (1 - w1) * H), test), dict(w=w1)
            # G2: one weight per State
            Rg = np.stack([R(w * A + (1 - w) * H) for w in WGRID], axis=1)
            w2 = np.full(len(F), w1)
            per = {}
            for e in np.unique(ent):
                m = fitm & (ent == e)
                if m.any():
                    w2[ent == e] = WGRID[int(Rg[m].sum(axis=0).argmin())]
                    per[e] = float(w2[ent == e][0])
            daily["G2"], info["G2"] = X6.system_daily(F, R(w2 * A + (1 - w2) * H), test), dict(w_by_state=per)
            # G3: online Fixed Share over blends; eta and share chosen on the fitting year
            Rb = np.stack([R(b * A + (1 - b) * H) for b in BLENDS], axis=1)
            best, arg = np.inf, None
            for eta in ETAS:
                for sh in SHARES:
                    w3 = online(F, Rb, fitm, eta, sh)
                    c = R(w3 * A + (1 - w3) * H)[fitm].sum()
                    if c < best:
                        best, arg = c, (eta, sh)
            w3 = online(F, Rb, fitm, *arg)
            daily["G3"] = X6.system_daily(F, R(w3 * A + (1 - w3) * H), test)
            info["G3"] = dict(eta=arg[0], share=arg[1], w_median=float(np.median(w3[test])),
                              w_iqr=[float(np.percentile(w3[test], 25)), float(np.percentile(w3[test], 75))])
            # G4: learned weight from decision-time features plus lagged relative regret of the two sources
            rel = pd.Series((rH - rA), index=F.index)
            g = F.assign(e=rel).sort_values(["entity", "date"]).groupby("entity").e
            Zx = Zdf.copy()
            Zx["rel7"] = g.transform(lambda s: s.shift(LAG).rolling(7, min_periods=3).mean()).reindex(F.index).fillna(0)
            Zx["rel28"] = g.transform(lambda s: s.shift(LAG).rolling(28, min_periods=7).mean()).reindex(F.index).fillna(0)
            mu, sd = Zx[fitm].mean(), Zx[fitm].std().replace(0, 1)
            Zn = ((Zx - mu) / sd).fillna(0.0).to_numpy()
            w4 = np.mean([learned_weight(Zn, A, H, X, p, lam, smin, smax, fitm, days, sname, project, s)
                          for s in SEEDS], axis=0)
            daily["G4"] = X6.system_daily(F, R(w4 * A + (1 - w4) * H), test)
            info["G4"] = dict(w_median=float(np.median(w4[test])),
                              w_iqr=[float(np.percentile(w4[test], 25)), float(np.percentile(w4[test], 75))])
            summ = {k: X6.summary(v) for k, v in daily.items()}
            comp = {}
            for v in ("G1", "G2", "G3", "G4"):
                for b in ("A2", "H_pub", "G0"):
                    comp[f"{v} vs {b}"] = X6.paired(daily[v], daily[b])
            prim = [f"{v} vs G0" for v in ("G1", "G2", "G3", "G4")]
            ps = [comp[k]["p_a_better"] for k in prim]
            order = np.argsort(ps)
            holm = np.empty(len(ps))
            for i, j in enumerate(order):
                holm[j] = min(1.0, max((len(ps) - i) * ps[j], holm[order[i - 1]] if i else 0))
            for k, hv in zip(prim, holm):
                comp[k]["p_holm"] = float(hv)
            out[tag] = dict(summary=summ, comparisons=comp, variants=info)
            print(sname, tag, json.dumps({k: round(v["mean"], 3) for k, v in summ.items()}), flush=True)
            print(json.dumps({k: [round(v["mean_diff"], 3), round(v["lo"], 3), round(v["hi"], 3), v.get("p_holm")]
                              for k, v in comp.items()}), flush=True)
        res["settlements"][sname] = out
    res["seconds"] = round(time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"source_fusion_{PHASE}.json")
    if os.path.exists(dst):
        dst = dst.replace(".json", time.strftime("_%Y%m%d_%H%M.json"))
    json.dump(res, open(dst, "w"), indent=1)
    print("DONE", res["seconds"], "s ->", dst)


if __name__ == "__main__":
    main()
