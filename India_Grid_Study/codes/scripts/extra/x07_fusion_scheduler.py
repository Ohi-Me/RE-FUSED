"""Settlement-aware fusion scheduler (TPWRS follow-up; protocol codes/design/09_fusion_tpwrs_protocol.md).

A level tau = 0.01 + 0.98 sigmoid(f(z)) is learned from decision-time features z and the schedule is the calibrated
forecast quantile at that level (variant F5 adds a scaled bias term). theta minimises the realised settlement regret
on the fitting year; the rule is then projected onto the schedule limits and scored like every other rule.

Phases (environment variable REFUSED_PHASE):
  dev      fit on FY2023-24, evaluate on FY2024-25 (development frames, already read in development)
  confirm  fit on FY2024-25, evaluate on 1 Apr 2025 - 31 Aug 2026; refused unless the protocol file's SHA-256 equals
           codes/design/FUSION_TPWRS_SHA256.txt (exploratory: that period was opened by the pre-registered study)
Outputs: results/extra/fusion_tpwrs_<phase>.json and fusion_tpwrs_<phase>_schedules.parquet (never overwritten).
Usage:   python codes/scripts/extra/x07_fusion_scheduler.py     (run on the H100: hpc/extra_fusion_tpwrs.pbs)
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
from refused import stats as S  # noqa: E402
from refused.paths import RES  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402
import d2_09_o5 as R2  # noqa: E402
import x06_dsm_regulation as X6  # noqa: E402

PHASE = os.environ.get("REFUSED_PHASE", "dev")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROTO = os.path.join(ROOT, "codes", "design", "09_fusion_tpwrs_protocol.md")
LOCK = os.path.join(ROOT, "codes", "design", "FUSION_TPWRS_SHA256.txt")
OUT = os.path.join(RES, "extra")
SEEDS = (0, 1, 2)
VARIANTS = {"F0": ("const", []), "F1": ("logit", ["r_hat"]), "F2": ("logit", ["r_hat", "risk"]),
            "F3": ("logit", "all"), "F4": ("mlp", "all"), "F5": ("mlp_bias", "all")}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def guard():
    if PHASE != "confirm":
        return
    if not os.path.exists(LOCK):
        raise SystemExit("protocol not frozen: no " + LOCK)
    want = open(LOCK).read().split()[0]
    have = hashlib.sha256(open(PROTO, "rb").read()).hexdigest()
    if want != have:
        raise SystemExit(f"protocol hash mismatch: {have} != {want}")


def features(F, SC):
    q10, q50, q90 = SC[:, 9], SC[:, 49], SC[:, 89]
    scale = np.maximum((q90 - q10) / 2, 1e-3)
    Z = pd.DataFrame(index=F.index)
    Z["r_hat"] = F.lam_hat / F.p_hat
    Z["risk"] = (np.abs(SC - q50[:, None]) > 0.1 * np.abs(q50)[:, None]).mean(axis=1)
    Z["spread"] = (q90 - q10) / np.maximum(np.abs(q50), 1e-3)
    Z["skew"] = (q90 + q10 - 2 * q50) / np.maximum(q90 - q10, 1e-6)
    Z["U"], Z["Z"] = F.U_p.to_numpy(), F.S_p.to_numpy()
    err = pd.Series((F.X.to_numpy() - q50) / scale, index=F.index)
    g = F.assign(e=err).sort_values(["entity", "date"]).groupby("entity").e
    Z["bias7"] = g.transform(lambda s: s.shift(2).rolling(7, min_periods=3).mean()).reindex(F.index).fillna(0.0)
    dow, mon = F.date.dt.dayofweek, F.date.dt.month
    Z["dow_s"], Z["dow_c"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    Z["mon_s"], Z["mon_c"] = np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)
    Z = pd.concat([Z, pd.get_dummies(F.entity, prefix="a", dtype=float)], axis=1)
    return Z, scale


def regret_t(S_, X, p, lam, sname):
    """torch version of x06.regret_s (Rs crore)."""
    if X6.SETTLEMENTS[sname] is None:
        b = 0.1 * S_.abs()
        Dp, Dm = torch.relu(X - S_), torch.relu(S_ - X)
        hi, lo = torch.maximum(lam, p), torch.minimum(lam, p)
        over = hi * (torch.minimum(Dp, b) + 1.2 * torch.relu(Dp - b))
        under = lo * (0.9 * torch.minimum(Dm, b) + 0.8 * torch.relu(Dm - b))
    else:
        k2 = X6.SETTLEMENTS[sname]["k_over2"]
        a = S_.abs()
        b1 = torch.clamp(0.10 * a, max=100 * X6.DAY_MW)
        b2 = torch.maximum(torch.clamp(0.15 * a, max=200 * X6.DAY_MW), b1)
        Dp, Dm = torch.relu(X - S_), torch.relu(S_ - X)
        over = lam * (torch.minimum(Dp, b1) + k2 * torch.minimum(torch.relu(Dp - b1), b2 - b1) +
                      2.0 * torch.relu(Dp - b2))
        under = lam * (0.9 * torch.minimum(Dm, b1) + 0.8 * torch.minimum(torch.relu(Dm - b1), b2 - b1))
    return (p * (S_ - X) + over - under) * O5.CR


class Policy(torch.nn.Module):
    def __init__(self, kind, d):
        super().__init__()
        self.kind = kind
        if kind == "const":
            self.c = torch.nn.Parameter(torch.zeros(1))
        elif kind == "logit":
            self.lin = torch.nn.Linear(d, 1)
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
                                           torch.nn.ReLU(), torch.nn.Linear(32, 2 if kind == "mlp_bias" else 1))

    def forward(self, z):
        if self.kind == "const":
            return self.c.expand(len(z)), torch.zeros(len(z), device=z.device)
        if self.kind == "logit":
            return self.lin(z).squeeze(1), torch.zeros(len(z), device=z.device)
        o = self.net(z)
        return o[:, 0], (0.5 * torch.tanh(o[:, 1]) if self.kind == "mlp_bias" else torch.zeros(len(z), device=z.device))


def quantile_at(SC_t, tau):
    """Linear interpolation of the 99 scenario quantiles (column k-1 = level k %) at levels tau in [0.01, 0.99]."""
    x = tau * 100 - 1
    lo = torch.clamp(x.floor().long(), 0, 97)
    w = (x - lo.float()).clamp(0, 1)
    return SC_t.gather(1, lo[:, None]).squeeze(1) * (1 - w) + SC_t.gather(1, (lo + 1)[:, None]).squeeze(1) * w


def fit(kind, Zfit, SCfit, scale, X, p, lam, smin, smax, days, sname, seed):
    torch.manual_seed(seed)
    n = len(Zfit)
    order = np.argsort(days)
    cut = days[order[int(0.8 * n)]]
    tr, es = days < cut, days >= cut
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEV)  # noqa: E731
    z, sc, c, Xt, pt, lt = T(Zfit), T(SCfit), T(scale), T(X), T(p), T(lam)
    lo_, hi_ = T(smin), T(smax)
    m = Policy(kind, z.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    trm, esm = torch.tensor(tr, device=DEV), torch.tensor(es, device=DEV)
    for ep in range(600):
        f, b = m(z)
        tau = 0.01 + 0.98 * torch.sigmoid(f)
        S_ = torch.minimum(torch.maximum(quantile_at(sc, tau) + c * b, lo_), hi_)
        R = regret_t(S_, Xt, pt, lt, sname)
        loss = R[trm].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = float(R[esm].mean())
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad > 60:
                break
    m.load_state_dict(best_state)
    return m, dict(epochs=ep + 1, es_regret=best)


def apply(m, Z, SC, scale):
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEV)  # noqa: E731
    with torch.no_grad():
        f, b = m(T(Z))
        tau = 0.01 + 0.98 * torch.sigmoid(f)
        S_ = quantile_at(T(SC), tau) + T(scale) * b
    return S_.cpu().numpy().astype(float), tau.cpu().numpy().astype(float)


def main():
    guard()
    t0 = time.time()
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    F = R2.build_v2(sel["final"], int(sel["calibration_window"])).reset_index(drop=True)
    SC = O5.scenarios(F[D.QN].to_numpy())
    Zdf, scale = features(F, SC)
    fitm, test = (F.split == "validation").to_numpy(), (F.split == "dev_test").to_numpy()
    mu, sd = Zdf[fitm].mean(), Zdf[fitm].std().replace(0, 1)
    Zn = ((Zdf - mu) / sd).fillna(0.0)
    X, p, lam = F.X.to_numpy(), F.p.to_numpy(), F.lam.to_numpy()
    ph, lh = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
    days = F.date.values.astype("datetime64[D]").astype(np.int64)
    fz = D.frozen("o5", "params.json")["params"]
    res = dict(phase=PHASE, protocol_sha256=hashlib.sha256(open(PROTO, "rb").read()).hexdigest(),
               note="exploratory follow-up; see protocol", n_fit=int(fitm.sum()), n_test=int(test.sum()), settlements={})
    sched = F.loc[test, ["entity", "date"]].copy()
    for sname in ("stylised", "dsm_nominal"):
        daily, out = {}, {}
        base = {"H_pub": O5.project(F, F.sched_pub.fillna(F.q50).to_numpy()),      # as in the confirmatory run
                "H_pub (not projected)": F.sched_pub.fillna(F.q50).to_numpy(), "P": O5.project(F, F.q50.to_numpy()),
                "A2": O5.project(F, X6.decide_s(SC, ph, lh, fz["A2"]["beta"], fz["A2"]["rho"], sname)),
                "R0": O5.project(F, X6.decide_s(SC, ph, lh, 0.0, 0.0, sname))}
        for k, v in base.items():
            daily[k] = X6.system_daily(F, X6.regret_s(v, X, p, lam, sname), test)
        for vid, (kind, cols) in VARIANTS.items():
            cols = list(Zn.columns) if cols == "all" else cols
            Zc = Zn[cols].to_numpy() if cols else np.zeros((len(F), 1))
            runs, S_seeds, taus = [], [], []
            for s in SEEDS:
                m, info = fit(kind, Zc[fitm], SC[fitm], scale[fitm], X[fitm], p[fitm], lam[fitm],
                              F.S_min.to_numpy()[fitm], F.S_max.to_numpy()[fitm], days[fitm], sname, s)
                S_raw, tau = apply(m, Zc, SC, scale)
                S_seeds.append(O5.project(F, S_raw))
                taus.append(tau)
                runs.append(info)
            S_v = np.mean(S_seeds, axis=0)             # schedule averaged over seeds, then scored
            daily[vid] = X6.system_daily(F, X6.regret_s(S_v, X, p, lam, sname), test)
            sched[f"{sname}_{vid}"] = S_v[test]
            tt = np.mean(taus, axis=0)[test]
            out[vid] = dict(runs=runs, tau_median=float(np.median(tt)),
                            tau_iqr=[float(np.percentile(tt, 25)), float(np.percentile(tt, 75))])
        summ = {k: X6.summary(v) for k, v in daily.items()}
        comp = {}
        for vid in ("F1", "F2", "F3", "F4", "F5"):
            comp[f"{vid} vs F0"] = X6.paired(daily[vid], daily["F0"])
        ps = [comp[k]["p_a_better"] for k in comp]
        order = np.argsort(ps)
        holm = np.empty(len(ps))
        for i, j in enumerate(order):
            holm[j] = min(1.0, max((len(ps) - i) * ps[j], holm[order[i - 1]] if i else 0))
        for k, h in zip(comp, holm):
            comp[k]["p_holm"] = float(h)
        for vid in VARIANTS:
            comp[f"{vid} vs A2"] = X6.paired(daily[vid], daily["A2"])
            comp[f"{vid} vs P"] = X6.paired(daily[vid], daily["P"])
        res["settlements"][sname] = dict(summary=summ, comparisons=comp, variants=out)
        print(sname, json.dumps({k: round(v["mean"], 3) for k, v in summ.items()}), flush=True)
        print(json.dumps({k: [round(v["mean_diff"], 3), round(v["lo"], 3), round(v["hi"], 3), v.get("p_holm")]
                          for k, v in comp.items() if "F0" in k}), flush=True)
    res["seconds"] = round(time.time() - t0)
    dst = os.path.join(OUT, f"fusion_tpwrs_{PHASE}.json")
    if os.path.exists(dst):
        dst = dst.replace(".json", time.strftime("_%Y%m%d_%H%M.json"))
    json.dump(res, open(dst, "w"), indent=1)
    sched.to_parquet(dst.replace(".json", "_schedules.parquet"))
    print("DONE", res["seconds"], "s ->", dst)


if __name__ == "__main__":
    main()
