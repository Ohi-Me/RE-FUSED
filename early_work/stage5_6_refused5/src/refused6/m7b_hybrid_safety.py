"""RE-FUSED-6 / M7b: PRE-REGISTERED test of the hybrid deployment rule (H-M7b, docs/08_defect_lineage.md).

clip90_lcb = deploy in a series iff the one-sided t lower bound on validation block skill is > 0
             (series-level protection against drift, Proposition 7), and when deployed clip the
             correction at the validation 90th percentile of |correction| (instance-level protection,
             Proposition 1).

FRESH stationary data: ETTh1, ETTh2 (persistence), NYISO hourly (persistence, day-ahead).
IN-SAMPLE (labelled): India daily, including the drifting operator schedule.
alpha = 0.10 and the 90th percentile are carried over from M7 unchanged.
"""
import os, time, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1c", os.path.join(HERE, "m1_canonical.py"))
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)
OUT = r"D:\\REFUSED5\results\refused6"
CLIP, N_ENS, ALPHA, MIN_CELL = 1.5, 4, 0.10, 30
RULES = ["always", "never", "block_lcb", "crc_quantile", "clip90", "clip90_lcb", "unc_selective"]


def run(ds, bname, h, seed, freq, subset_label):
    kind, arg = ds["baselines"][bname]
    x = m1.build_target_and_baseline(ds["df"], h, kind, arg, ds["season"])
    if x is None:
        return None
    F = ds["feats"]
    x = x.dropna(subset=F + ["y_t", "B"])
    tr, va, te = [x[x.split == s] for s in ("train", "val", "test")]
    if min(len(tr), len(va), len(te)) < 400:
        return None
    for f in (tr, va, te):
        f["R"] = f.y_t - f.B
    sub = min(len(tr), 200_000)
    trs = tr.sample(sub, random_state=seed) if len(tr) > sub else tr
    corr = HGB(max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed,
               l2_regularization=1.0).fit(trs[F], trs.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    ens_v, ens_t = [], []
    esub = min(len(trs), 100_000)
    for b in range(N_ENS):
        rb = np.random.default_rng(3100 + 19 * seed + b)
        idx = rb.integers(0, len(trs), esub)
        mb = HGB(max_iter=150, learning_rate=0.08, max_depth=6, random_state=seed * 7 + b).fit(
            trs[F].iloc[idx], trs.R.iloc[idx])
        ens_v.append(mb.predict(va[F])); ens_t.append(mb.predict(te[F]))
    sd_v, sd_t = np.std(ens_v, axis=0), np.std(ens_t, axis=0)
    Rv, Rt = va.R.to_numpy(), te.R.to_numpy()
    g1 = m1.ls_gate(Rv, rv)
    rv, rt = g1 * rv, g1 * rt
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()
    blk_v = pd.to_datetime(va.ts).dt.to_period(freq).astype(str).to_numpy()
    per = {r_: [] for r_ in RULES}
    withheld = {r_: 0 for r_ in RULES}
    for s in np.unique(sid_t):
        kv, kt = sid_v == s, sid_t == s
        if kv.sum() < MIN_CELL or kt.sum() < 10:
            continue
        g = float(np.clip(m1.ls_gate(Rv[kv], rv[kv]), 0, CLIP))
        base_t = np.mean(Rt[kt] ** 2)
        full_t = 1 - np.mean((Rt[kt] - g * rt[kt]) ** 2) / base_t
        bs = []
        for bb in np.unique(blk_v[kv]):
            k = kv & (blk_v == bb)
            if k.sum() >= 10 and np.mean(Rv[k] ** 2) > 0:
                bs.append(1 - np.mean((Rv[k] - g * rv[k]) ** 2) / np.mean(Rv[k] ** 2))
        bs = np.array(bs)
        if len(bs) >= 3 and bs.std(ddof=1) > 0:
            lcb_ok = bool(bs.mean() - stats.t.ppf(1 - ALPHA, len(bs) - 1) * bs.std(ddof=1) / np.sqrt(len(bs)) > 0)
            kq = int(np.ceil((len(bs) + 1) * (1 - ALPHA))) - 1
            crc_ok = bool(np.sort(-bs)[min(kq, len(bs) - 1)] <= 0)
        else:
            lcb_ok = crc_ok = False
        tau = float(np.quantile(np.abs(g * rv[kv]), 0.90))
        clip_t = 1 - np.mean((Rt[kt] - np.clip(g * rt[kt], -tau, tau)) ** 2) / base_t
        best_q, best_val = 1.0, -np.inf
        for q in (0.25, 0.5, 0.75, 0.9, 1.0):
            use = sd_v[kv] <= np.quantile(sd_v[kv], q)
            v = 1 - np.mean((Rv[kv] - np.where(use, g * rv[kv], 0)) ** 2) / np.mean(Rv[kv] ** 2)
            if v > best_val:
                best_val, best_q = v, q
        use_t = sd_t[kt] <= np.quantile(sd_v[kv], best_q)
        unc_t = 1 - np.mean((Rt[kt] - np.where(use_t, g * rt[kt], 0)) ** 2) / base_t
        vals = {"always": full_t, "never": 0.0, "block_lcb": full_t if lcb_ok else 0.0,
                "crc_quantile": full_t if crc_ok else 0.0, "clip90": clip_t,
                "clip90_lcb": clip_t if lcb_ok else 0.0, "unc_selective": unc_t}
        for r_, v in vals.items():
            per[r_].append(v)
        withheld["block_lcb"] += int(not lcb_ok)
        withheld["clip90_lcb"] += int(not lcb_ok)
        withheld["crc_quantile"] += int(not crc_ok)
    rows = []
    for r_ in RULES:
        v = np.array(per[r_])
        if len(v):
            rows.append(dict(subset=subset_label, dataset=ds["name"], baseline=bname, h=h, seed=seed, rule=r_,
                             mean_skill=float(v.mean()), degraded_series=int((v < 0).sum()),
                             worst_series_skill=float(v.min()), n_series=len(v), withheld=withheld[r_]))
    return rows


def main():
    jobs = []
    india = m1.load_india()
    jobs += [(india, b, h, s, "M", "in_sample_india") for b in ["persistence", "roll7", "operator_schedule"]
             for h in [1, 3] for s in range(3)]
    for e in m1.load_ett():
        if e["name"] in ("ETTh1", "ETTh2"):
            jobs += [(e, "persistence", h, s, "W", "fresh_stationary") for h in [1, 24] for s in range(5)]
    ny = m1.load_nyiso_1h()
    if ny is not None:
        jobs += [(ny, b, h, s, "M", "fresh_stationary") for b in ["persistence", "day_ahead"]
                 for h in [1, 24] for s in range(3)]
    rows, t0 = [], time.time()
    for i, (ds, b, h, s, fq, lab) in enumerate(jobs):
        r = run(ds, b, h, s, fq, lab)
        if r:
            rows += r
        if s == 0 or i == len(jobs) - 1:
            d = pd.DataFrame(rows)
            if len(d):
                q = d[(d.dataset == ds["name"]) & (d.baseline == b) & (d.h == h)].groupby("rule")[["mean_skill", "degraded_series"]].mean()
                print(f"[{i+1}/{len(jobs)}] {lab:17s} {ds['name']:11s} {b:18s} h={h:2d} | " +
                      " ".join(f"{r_}={q.loc[r_,'mean_skill']:+.4f}({q.loc[r_,'degraded_series']:.1f})" for r_ in RULES if r_ in q.index) +
                      f" [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "M7b_hybrid.csv"), index=False)
    print(f"\nwrote {len(R)} rows")
    for lab in R.subset.unique():
        P = R[R.subset == lab]
        print(f"\n=== {lab}: mean over configs x seeds ===")
        print(P.pivot_table(index="rule", values=["mean_skill", "degraded_series", "worst_series_skill", "withheld"]).round(4).to_string())
    F_ = R[R.subset == "fresh_stationary"]
    key = ["dataset", "baseline", "h", "seed"]
    a = F_[F_.rule == "clip90_lcb"].set_index(key)
    lcb = F_[F_.rule == "block_lcb"].set_index(key)
    clp = F_[F_.rule == "clip90"].set_index(key)
    j = a.mean_skill.align(lcb.mean_skill, join="inner")
    d = j[0] - j[1]
    p_lcb_better = stats.wilcoxon(j[1], j[0], alternative="greater").pvalue if (d != 0).sum() > 5 else 1.0
    c1 = (d.mean() >= 0) and (p_lcb_better >= 0.05)
    jd = a.degraded_series.align(clp.degraded_series, join="inner")
    c2 = jd[0].mean() <= jd[1].mean()
    I_ = R[(R.subset == "in_sample_india") & (R.baseline == "operator_schedule")]
    c3 = I_[I_.rule == "clip90_lcb"].degraded_series.mean() <= I_[I_.rule == "block_lcb"].degraded_series.mean() + 0.5
    print("\n=== PRE-REGISTERED VERDICT FOR H-M7b ===")
    print(f"  {'PASS' if c1 else 'FAIL'}  (i)  fresh: clip90_lcb >= block_lcb (diff {d.mean():+.5f}, p[lcb better]={p_lcb_better:.3f})")
    print(f"  {'PASS' if c2 else 'FAIL'}  (ii) fresh: clip90_lcb degraded {jd[0].mean():.2f} <= clip90 {jd[1].mean():.2f}")
    print(f"  {'PASS' if c3 else 'FAIL'}  (iii) in-sample drift: clip90_lcb degraded <= block_lcb + 0.5")
    print("  H-M7b", "SURVIVES" if c1 else "FALSIFIED (criterion i)")
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
