"""Block-level (15-minute) check of the scheduling rules with the official settlement of six Eastern-Region control areas
(exploratory, not pre-registered; the pre-registered results are unchanged).

Data: ERPC block-wise settlement accounts (data/interim/erpc_dsm_blocks.parquet, b11_erpc_blocks.py): per block the
system frequency, scheduled and actual drawal, deviation, normal rate and the official charge. The rate function
codes/refused/dsm2024.py reproduces the official charges (x11a_rate_check.py).

B  How well do daily settlement models reproduce the official daily deviation charge of a State?
   M1 the paper's stylised model; M2 the buyer rules of the Regulations applied to daily energy at 50.00 Hz (volume
   limits as daily energy); M3 M2 with the day's blocks settled at their own frequency and normal rate, the daily
   deviation spread evenly over the blocks. Official = sum of the 96 block charges.
C  Do the paper's rules rank the same under block-level settlement? Each rule's daily schedule is spread over the
   blocks in proportion to the State's recorded block schedule of that day; every block is settled with the
   official frequency, normal rate and actual drawal. Regret = p (S - X) + net deviation charge, per day, summed
   over the six areas. Rules: forecast median P, persistence H_pub, A2, R0, fixed level and adaptive level (r x
   band-relative risk) chosen on the calibration year, as in x06.
Output: results/extra/block_settlement.json.  Run: REFUSED_PHASE=confirm REFUSED_CONFIRMATORY=1 python x11_...py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
for sub in ("o2", "o4", "o5", "dev2", "extra"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from refused import dev2 as D  # noqa: E402
from refused import dsm2024 as R  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402
import d2_09_o5 as R2  # noqa: E402
import x06_dsm_regulation as X6  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BLK = os.path.join(ROOT, "data", "interim", "erpc_dsm_blocks.parquet")
OUT = os.path.join(X6.OUT, "block_settlement.json")
ER = ["BR", "JH", "OD", "WB", "DVC", "SK"]
CR = 1e-7                                    # Rs -> Rs crore


def block_charges(dev, sched, nr, f):
    small = 4 * np.abs(sched) <= 400
    return np.where(small, R.charge(dev, sched, nr, f, "small"), R.charge(dev, sched, nr, f))


def daily_models(Bd, p):
    """Daily deviation charge (Rs) under M1-M3 from daily totals; Bd has one row per entity-day."""
    S, X, lam = Bd.sched.to_numpy(), Bd.actual.to_numpy(), Bd.nr_mean.to_numpy()
    Dp, Dm = np.maximum(X - S, 0), np.maximum(S - X, 0)
    b = 0.1 * np.abs(S)
    hi, lo = np.maximum(lam, p), np.minimum(lam, p)
    m1 = hi * (np.minimum(Dp, b) + 1.2 * np.maximum(Dp - b, 0)) - lo * (0.9 * np.minimum(Dm, b) + 0.8 * np.maximum(Dm - b, 0))
    b1 = np.minimum(0.10 * np.abs(S), 100 * 24)                         # MWh per day
    b2 = np.maximum(np.minimum(0.15 * np.abs(S), 200 * 24), b1)
    t = lambda d: (np.minimum(d, b1), np.clip(np.minimum(d, b2) - b1, 0, None), np.maximum(d - b2, 0))  # noqa: E731
    o1, o2, o3 = t(Dp)
    u1, u2, _ = t(Dm)
    m2 = lam * (o1 + o2 + o3) - lam * (0.9 * u1 + 0.8 * u2)
    return m1, m2


def main():
    t0 = time.time()
    B = pd.read_parquet(BLK)
    B = B[B.entity.isin(ER)].copy()
    B["official"] = B.payable_rs.astype(float) - B.receivable_rs.astype(float)
    sel = next(x for x in json.load(open(os.path.join(D.DEV2, "o2", "eval", "selection_dev2.json"))) if x["tid"] == "T2")
    F = R2.build_v2(sel["final"], int(sel["calibration_window"])).reset_index(drop=True)
    test, val = (F.split == "dev_test").to_numpy(), (F.split == "validation").to_numpy()
    res = dict(note="exploratory; official ERPC block settlement; pre-registered results unchanged",
               blocks=int(len(B)), period=[str(B.date.min().date()), str(B.date.max().date())])

    # daily totals of the official accounts, merged with the panel (p, panel drawal for a consistency check)
    g = B.groupby(["entity", "date"])
    Bd = g.agg(sched=("sched_mwh", "sum"), actual=("actual_mwh", "sum"), official=("official", "sum"),
               nr_mean=("nr_rs_mwh", "mean"), n=("block", "size")).reset_index()
    Bd = Bd[Bd.n == 96]
    Fd = F[["entity", "date", "X", "S_hist", "p", "split"]].copy()
    Fd["date"] = pd.to_datetime(Fd.date)
    Bd = Bd.merge(Fd, on=["entity", "date"], how="inner")
    Bd = Bd[Bd.split == "dev_test"]
    res["state_days"] = int(len(Bd))
    res["panel_vs_account_drawal"] = dict(
        corr=float(np.corrcoef(Bd.X * 1000, Bd.actual)[0, 1]),
        median_abs_rel_diff=float(np.median(np.abs(Bd.X * 1000 - Bd.actual) / np.abs(Bd.actual))))

    # B: daily models against the official daily charge
    m1, m2 = daily_models(Bd, Bd.p.to_numpy())
    Bm = B.merge(Bd[["entity", "date", "sched", "actual"]], on=["entity", "date"])
    Bm["dev_even"] = (Bm.actual - Bm.sched) / 96.0                  # daily deviation spread evenly
    Bm["m3"] = block_charges(Bm.dev_even.to_numpy(), Bm.sched_mwh.to_numpy(), Bm.nr_rs_mwh.to_numpy(), Bm.freq_hz.to_numpy())
    m3 = Bm.groupby(["entity", "date"]).m3.sum().reindex(pd.MultiIndex.from_frame(Bd[["entity", "date"]])).to_numpy()
    off = Bd.official.to_numpy()
    gross = B.assign(a=B.dev_mwh.abs()).groupby(["entity", "date"]).a.sum()
    netd = (Bd.actual - Bd.sched).abs().to_numpy()
    res["intra_day"] = dict(median_gross_over_net_deviation=float(np.median(
        gross.reindex(pd.MultiIndex.from_frame(Bd[["entity", "date"]])).to_numpy() / np.maximum(netd, 1e-6))))
    res["daily_models"] = {}
    for name, m in (("M1 stylised (paper)", m1), ("M2 Regulations, daily at 50.00 Hz", m2),
                    ("M3 M2 with block frequency and rate", m3)):
        res["daily_models"][name] = dict(
            corr=float(np.corrcoef(m, off)[0, 1]), sign_agreement=float(np.mean(np.sign(m) == np.sign(off))),
            mean_official_rs_cr=float(off.mean() * CR), mean_model_rs_cr=float(m.mean() * CR),
            mae_over_mean_abs_official=float(np.mean(np.abs(m - off)) / np.mean(np.abs(off))))
        print(name, res["daily_models"][name], flush=True)

    # C: block-level scoring of the rules for the six areas
    SC = O5.scenarios(F[D.QN].to_numpy())
    ph, lh = F.p_hat.to_numpy(), F.lam_hat.to_numpy()
    X, p, lam = F.X.to_numpy(), F.p.to_numpy(), F.lam.to_numpy()
    fz = D.frozen("o5", "params.json")["params"]
    levels = np.arange(1, 100)
    S_lvl = np.column_stack([O5.project(F, SC[:, k - 1]) for k in levels])
    Rl = np.column_stack([X6.regret_s(S_lvl[:, j], X, p, lam, "stylised") for j in range(len(levels))])
    j_cal = int(Rl[val].sum(axis=0).argmin())
    r0_raw = X6.decide_s(SC, ph, lh, 0.0, 0.0, "stylised")
    b1 = np.minimum(0.10 * np.abs(r0_raw), 100 * X6.DAY_MW)
    risk = ((SC > r0_raw[:, None] + b1[:, None]) | (SC < r0_raw[:, None] - b1[:, None])).mean(axis=1)
    cut = lambda x: np.quantile(x[val], [1 / 3, 2 / 3])  # noqa: E731
    grp = np.digitize(lh / ph, cut(lh / ph)) * 3 + np.digitize(risk, cut(risk))
    ch = X6.best_by_group(Rl, grp, val)
    rules = {"P": O5.project(F, F.q50.to_numpy()), "H_pub": O5.project(F, F.sched_pub.fillna(F.q50).to_numpy()),
             "A2": O5.project(F, X6.decide_s(SC, ph, lh, fz["A2"]["beta"], fz["A2"]["rho"], "stylised")),
             "R0": O5.project(F, r0_raw), "fixed level": S_lvl[:, j_cal],
             "adaptive level": S_lvl[np.arange(len(F)), ch]}
    key = F[["entity", "date"]].copy()
    key["date"] = pd.to_datetime(key.date)
    sel_rows = test & key.entity.isin(ER).to_numpy()
    shape = B.merge(Bd[["entity", "date", "sched"]], on=["entity", "date"])
    w = np.where(np.abs(shape.sched) > 1.0, shape.sched_mwh / shape.sched, 1.0 / 96)
    shape = shape.assign(w=w)
    # the panel's drawal (PSP reports) and the accounts' metered drawal can differ in level; "rescaled" maps each
    # rule's schedule into account units with the day's ratio of account schedule to the panel's recorded schedule
    ratio = Bd.assign(k=np.where(np.abs(Bd.S_hist) > 1e-3, Bd.sched / (1000.0 * Bd.S_hist), 1.0))[["entity", "date", "k"]]
    ratio["k"] = ratio.k.clip(0.8, 1.25)
    res["schedule_ratio_account_over_panel"] = dict(median=float(ratio.k.median()),
                                                    iqr=[float(ratio.k.quantile(0.25)), float(ratio.k.quantile(0.75))])
    res["block_level"] = {}
    for variant in ("as is", "rescaled"):
        daily = {}
        for name, S in rules.items():
            d = key[sel_rows].assign(S_mwh=S[sel_rows] * 1000.0, p=p[sel_rows])
            m = shape.merge(d, on=["entity", "date"]).merge(ratio, on=["entity", "date"])
            Sb = m.w * m.S_mwh * (m.k if variant == "rescaled" else 1.0)
            dev = m.actual_mwh - Sb
            chg = block_charges(dev.to_numpy(), Sb.to_numpy(), m.nr_rs_mwh.to_numpy(), m.freq_hz.to_numpy())
            m = m.assign(reg=(m.p * (Sb - m.actual_mwh) + chg) * CR)          # p in Rs/MWh, energy in MWh
            daily[name] = m.groupby("date").reg.sum().sort_index()
        bl = dict(rules={k: X6.summary(v) for k, v in daily.items()},
                  cmp={f"A2 vs {k}": X6.paired(daily["A2"], daily[k]) for k in rules if k != "A2"},
                  adaptive_vs_fixed=X6.paired(daily["adaptive level"], daily["fixed level"]),
                  days=int(len(daily["A2"])))
        res["block_level"][variant] = bl
        print(variant, json.dumps({k: round(v["mean"], 3) for k, v in bl["rules"].items()}), flush=True)
        for k, v in list(bl["cmp"].items()) + [("adaptive vs fixed", bl["adaptive_vs_fixed"])]:
            print("  ", k, round(v["mean_diff"], 3), [round(v["lo"], 3), round(v["hi"], 3)], flush=True)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(OUT, "w"), indent=1)
    print("DONE", res["seconds"], "s")


if __name__ == "__main__":
    main()
