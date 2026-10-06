"""RE-FUSED-6 / M7: risk-controlled deployment versus established safety mechanisms.

Question: is RE-FUSED's block lower-confidence-bound deployment rule a genuinely new safety principle,
or does it reproduce known selective-prediction / conformal behaviour?

Setting: India daily panel, persistence (stationary residual), roll7, and the operator schedule
(drifting bias). The per-series gate (the empirically best rung) is fitted on validation with the
D9 reparameterisation; each rule then decides, per series or per instance, whether to apply it.

Rules (all use VALIDATION ONLY):
  always          apply the per-series correction everywhere
  never           baseline only
  block_lcb       (RE-FUSED) deploy in a series iff one-sided t lower bound on monthly-block skill > 0
  crc_quantile    conformal-risk-control style: deploy iff the finite-sample (1-alpha) order statistic
                  of per-block loss difference (corrected - baseline) is <= 0
  clip90          CRC-paper-style firewall: always deploy but clip |correction| at the validation
                  90th percentile of |r_hat| in that series
  unc_selective   instance-level selective correction: apply only where the ensemble std of the
                  corrector is below a threshold chosen on validation (quantile grid)

Test metrics per run: mean skill over series, degraded-series count, worst-series skill,
withheld series, withheld-but-would-have-helped, instance coverage.
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
CLIP, N_SEEDS, N_ENS, ALPHA, MIN_CELL = 1.5, 5, 4, 0.10, 30
RULES = ["always", "never", "block_lcb", "crc_quantile", "clip90", "unc_selective"]


def block_skill(R, r, g, months):
    out = []
    for m in np.unique(months):
        k = months == m
        if k.sum() < 10:
            continue
        base = np.mean(R[k] ** 2)
        if base <= 0:
            continue
        out.append(1 - np.mean((R[k] - g * r[k]) ** 2) / base)
    return np.array(out)


def run(ds, bname, h, seed):
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
    corr = HGB(max_iter=400, learning_rate=0.06, max_depth=6, random_state=seed,
               l2_regularization=1.0).fit(tr[F], tr.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    # small bootstrap ensemble for epistemic uncertainty of the corrector (train only)
    ens_v, ens_t = [], []
    for b in range(N_ENS):
        rb = np.random.default_rng(3000 + 17 * seed + b)
        idx = rb.integers(0, len(tr), len(tr))
        mb = HGB(max_iter=150, learning_rate=0.08, max_depth=6, random_state=seed * 7 + b).fit(
            tr[F].iloc[idx], tr.R.iloc[idx])
        ens_v.append(mb.predict(va[F])); ens_t.append(mb.predict(te[F]))
    sd_v, sd_t = np.std(ens_v, axis=0), np.std(ens_t, axis=0)
    Rv, Rt = va.R.to_numpy(), te.R.to_numpy()
    g1 = m1.ls_gate(Rv, rv)
    rv, rt = g1 * rv, g1 * rt                          # D9 reparameterisation
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()
    mon_v = pd.to_datetime(va.ts).dt.to_period("M").astype(str).to_numpy()
    rows = []
    per_series = {r_: [] for r_ in RULES}
    withheld = {r_: 0 for r_ in RULES}
    withheld_helpful = {r_: 0 for r_ in RULES}
    applied_pts = {r_: 0 for r_ in RULES}
    for s in np.unique(sid_t):
        kv, kt = sid_v == s, sid_t == s
        if kv.sum() < MIN_CELL or kt.sum() < 10:
            continue
        g = float(np.clip(m1.ls_gate(Rv[kv], rv[kv]), 0, CLIP))
        base_t = np.mean(Rt[kt] ** 2)
        skill_full = 1 - np.mean((Rt[kt] - g * rt[kt]) ** 2) / base_t
        bs = block_skill(Rv[kv], rv[kv], g, mon_v[kv])
        # --- rules ---
        dec = {"always": True, "never": False}
        if len(bs) >= 3:
            lcb = bs.mean() - stats.t.ppf(1 - ALPHA, len(bs) - 1) * bs.std(ddof=1) / np.sqrt(len(bs))
            dec["block_lcb"] = bool(lcb > 0)
            loss_diff = np.sort(-bs)                   # corrected minus baseline, normalised
            kq = int(np.ceil((len(bs) + 1) * (1 - ALPHA))) - 1
            dec["crc_quantile"] = bool(loss_diff[min(kq, len(bs) - 1)] <= 0)
        else:
            dec["block_lcb"] = False
            dec["crc_quantile"] = False
        active = g > 1e-9                              # a zero gate deploys nothing (vacuous)
        for r_ in ["always", "never", "block_lcb", "crc_quantile"]:
            sk = skill_full if dec[r_] else 0.0
            per_series[r_].append(sk)
            applied_pts[r_] += int(kt.sum()) if (dec[r_] and active) else 0
            if not dec[r_] and r_ != "never":
                withheld[r_] += 1
                withheld_helpful[r_] += int(skill_full > 0)
        # clip90 firewall
        tau = float(np.quantile(np.abs(g * rv[kv]), 0.90))
        corr_t = np.clip(g * rt[kt], -tau, tau)
        per_series["clip90"].append(1 - np.mean((Rt[kt] - corr_t) ** 2) / base_t)
        applied_pts["clip90"] += int(kt.sum()) if active else 0
        # uncertainty-selective (threshold chosen on validation only)
        best_q, best_val = 1.0, -np.inf
        for q in (0.25, 0.5, 0.75, 0.9, 1.0):
            thr = np.quantile(sd_v[kv], q)
            use = sd_v[kv] <= thr
            val_sk = 1 - np.mean((Rv[kv] - np.where(use, g * rv[kv], 0)) ** 2) / np.mean(Rv[kv] ** 2)
            if val_sk > best_val:
                best_val, best_q = val_sk, q
        thr = np.quantile(sd_v[kv], best_q)
        use_t = sd_t[kt] <= thr
        per_series["unc_selective"].append(
            1 - np.mean((Rt[kt] - np.where(use_t, g * rt[kt], 0)) ** 2) / base_t)
        applied_pts["unc_selective"] += int(use_t.sum()) if active else 0
    n_pts = int(sum(1 for _ in sid_t))
    for r_ in RULES:
        v = np.array(per_series[r_])
        if len(v) == 0:
            continue
        rows.append(dict(baseline=bname, h=h, seed=seed, rule=r_, mean_skill=float(v.mean()),
                         degraded_series=int((v < 0).sum()), n_series=len(v),
                         worst_series_skill=float(v.min()), withheld=withheld[r_],
                         withheld_helpful=withheld_helpful[r_],
                         coverage=float(applied_pts[r_] / max(n_pts, 1))))
    return rows


def main():
    ds = m1.load_india()
    allr, t0 = [], time.time()
    for bname in ["persistence", "roll7", "operator_schedule"]:
        for h in [1, 3, 7]:
            for seed in range(N_SEEDS):
                r = run(ds, bname, h, seed)
                if r:
                    allr += r
            d = pd.DataFrame(allr).query("baseline == @bname and h == @h")
            if len(d):
                p = d.groupby("rule")[["mean_skill", "degraded_series", "worst_series_skill", "coverage"]].mean()
                print(f"{bname:18s} h={h} | " + "  ".join(
                    f"{r_}={p.loc[r_,'mean_skill']:+.4f}({p.loc[r_,'degraded_series']:.1f})"
                    for r_ in RULES if r_ in p.index) + f" [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(allr)
    R.to_csv(os.path.join(OUT, "M7_risk_control.csv"), index=False)
    print(f"\nwrote {len(R)} rows")
    print("\n=== accuracy-safety summary by baseline (mean over h x seeds) ===")
    print(R.pivot_table(index=["baseline", "rule"], values=["mean_skill", "degraded_series",
          "worst_series_skill", "coverage", "withheld", "withheld_helpful"]).round(4).to_string())
    print("\n=== paired comparison vs block_lcb (unit = config x seed) ===")
    key = ["baseline", "h", "seed"]
    ref = R[R.rule == "block_lcb"].set_index(key)
    for r_ in RULES:
        if r_ == "block_lcb":
            continue
        o = R[R.rule == r_].set_index(key)
        j = ref.mean_skill.align(o.mean_skill, join="inner")
        jd = ref.degraded_series.align(o.degraded_series, join="inner")
        t = stats.ttest_rel(j[0], j[1])
        print(f"  block_lcb - {r_:14s} skill={float((j[0]-j[1]).mean()):+.5f} p={t.pvalue:.2e} | "
              f"degraded: lcb={jd[0].mean():.2f} vs {r_}={jd[1].mean():.2f}")
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
