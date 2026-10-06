"""X3: capacity axis of the per-instance gate (audit E2; 09_theory T3/T4b).

Separates *instance dependence* from *estimator capacity*: the instance gate's realised gain over the per-series
gate is measured for a grid of capacities (tree depth x boosting iterations, plus a linear-in-features gate) and
three gate-fitting budgets (25% of validation days, full validation year, out-of-fold train + validation).
Datasets: NYISO load (ISO baseline) and India (roll7, h = 1) — development splits only.
Output: 06_results/dev/x3/x3_capacity.csv with realised gain (skill units) and its day-block bootstrap CI.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import data, gates as G, ladder, selection as S, stats as st  # noqa: E402
from refused_gate.paths import RES  # noqa: E402

CAPS = [("linear", None, None)] + [("hgb", d, it) for d, it in itertools.product([1, 3, 6], [25, 150, 400])]


def prepare(name, seed):
    if name == "nyiso_iso":
        ds = data.nyiso_load("dev")
        fr = ds["df"].assign(y_t=ds["df"].y, B=ds["df"].iso_fc)
    else:
        ds = data.india_daily("dev")
        fr = data.make_baseline(ds["df"], 1, ds["baselines"]["roll7"], season=7)
    x = fr.dropna(subset=ds["feats"] + ["y_t", "B"]).copy()
    x["R"] = x.y_t - x.B
    x["day"] = x.ts.dt.normalize()
    return x, ds["feats"]


def main():
    out = os.path.join(RES, "dev", "x3")
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, "x3_capacity.csv")
    rows = pd.read_csv(dst).to_dict("records") if os.path.exists(dst) else []
    done = {(r["dataset"], r["budget"], r["cap"], r["seed"]) for r in rows}
    t0 = time.time()
    for name in ("nyiso_iso", "india_roll7_h1"):
        x, feats = prepare(name, 0)
        cfg = ladder.LadderConfig()
        for seed in range(2):
            tr, va, te = (x[x.split == s].copy() for s in ("train", "val", "test"))
            corr = ladder._fit_corrector(tr[feats].to_numpy(np.float32), tr.R.to_numpy(), cfg, seed)
            va["r"], te["r"] = corr.predict(va[feats].to_numpy(np.float32)), corr.predict(te[feats].to_numpy(np.float32))
            tr["r"] = ladder._oof_predictions(tr, feats, cfg, seed)
            budgets = {"val25": va[va.day.isin(np.random.default_rng(seed).choice(np.sort(va.day.unique()),
                                                                                    va.day.nunique() // 4, replace=False))],
                       "val": va, "oof_train+val": pd.concat([tr.dropna(subset=["r"]), va])}
            for bname, fit in budgets.items():
                g1 = G.ls_gate(fit.R.to_numpy(), fit.r.to_numpy())
                Rf, rf = fit.R.to_numpy(), fit.r.to_numpy() * g1
                Rt, rt = te.R.to_numpy(), te.r.to_numpy() * g1
                gs = G.PartitionGate(30).fit(Rf, rf, fit.sid.to_numpy()).predict(te.sid.to_numpy())
                Ff, Ft = fit[feats].to_numpy(np.float32), te[feats].to_numpy(np.float32)
                base = float(np.mean(Rt ** 2))
                for kind, depth, iters in CAPS:
                    cap = "linear" if kind == "linear" else f"d{depth}_it{iters}"
                    if (name, bname, cap, seed) in done:
                        continue
                    if kind == "linear":
                        w = rf ** 2
                        ok = w > 1e-12
                        mu, sd = Ff.mean(0), Ff.std(0) + 1e-9
                        m = Ridge(alpha=10.0).fit((Ff[ok] - mu) / sd, np.clip(Rf[ok] / rf[ok], -10, 10), sample_weight=w[ok])
                        gi = m.predict((Ft - mu) / sd)
                    else:
                        gi = G.InstanceGate(n_bag=1, seed=seed, max_depth=depth, max_iter=iters).fit(Ff, Rf, rf).predict(Ft)
                    gain = S.realised_gain(Rt, rt, gs, gi)
                    daily = pd.Series(gain).groupby(te.day.to_numpy()).sum()
                    n_per_day = te.groupby("day").size().mean()
                    ci = st.block_ci(daily.to_numpy() / (n_per_day * base), 7 if name.startswith("nyiso") else 14, 1000, seed)
                    rows.append(dict(dataset=name, budget=bname, n_fit=len(fit), cap=cap, seed=seed,
                                     gain_skill=ci["mean"], lo=ci["lo"], hi=ci["hi"]))
                    done.add((name, bname, cap, seed))
                pd.DataFrame(rows).to_csv(dst, index=False)
                print(f"  {name} seed {seed} {bname:14s} n_fit={len(fit):7d} [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    print(R.groupby(["dataset", "budget", "cap"]).gain_skill.mean().unstack("cap").round(4).to_string())
    print("DONE x3", round(time.time() - t0))


if __name__ == "__main__":
    main()
