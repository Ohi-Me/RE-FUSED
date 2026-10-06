"""RE-FUSED-6 / M1c: closes defect D9 (corrector scale miscalibration).

D9: ~21-23% of fine-rung gate values exceed the admissible cap even with the validated WLS
estimator, because the l2-regularised corrector systematically under-predicts residual magnitude.
Every gate estimator is then working off-centre, which handicaps the fine rungs.

Fix (reparameterisation, not a new model): absorb the global least-squares gate into the corrector,
    r_tilde = g1 * r_hat,   g1 = <R, r_hat> / <r_hat, r_hat>   estimated on VALIDATION only.
By construction the optimal global gate on r_tilde is 1, so finer gates estimate multiplicative
deviations around 1 and the admissible band [0, 1.5] is centred on the right scale.

If the fine rungs improve materially, part of the measured "price of adaptivity" was scale
miscalibration. If they do not, the price-of-adaptivity result is robust to gate parameterisation.

Subset used (targeted confound test, not a full re-run): persistence and roll7 at h = 1, 3, 7.
"""
import os, json, time, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1c_base", os.path.join(HERE, "m1_canonical.py"))
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)
OUT = r"D:\\REFUSED5\results\refused6"
CLIP, N_SEEDS, N_BOOT, MIN_CELL = m1.CLIP, m1.N_SEEDS, m1.N_BOOT, m1.MIN_CELL
Z_CLIP = 10.0


def wls_gate(F, R, r, seed):
    w = r ** 2
    ok = w > 1e-12
    z = np.clip(R[ok] / r[ok], -Z_CLIP, Z_CLIP)
    m = HGB(max_iter=200, learning_rate=0.07, max_depth=5, random_state=seed)
    m.fit(F[ok], z, sample_weight=w[ok])
    return m


def one(ds, bname, h, seed, rescale):
    kind, arg = ds["baselines"][bname]
    x = m1.build_target_and_baseline(ds["df"], h, kind, arg, ds["season"])
    F = ds["feats"]
    x = x.dropna(subset=F + ["y_t", "B"])
    tr, va, te = [x[x.split == s] for s in ("train", "val", "test")]
    for f in (tr, va, te):
        f["R"] = f.y_t - f.B
    corr = HGB(max_iter=400, learning_rate=0.06, max_depth=6, random_state=seed,
               l2_regularization=1.0).fit(tr[F], tr.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    Rv, Rt = va.R.to_numpy(), te.R.to_numpy()
    g1 = m1.ls_gate(Rv, rv)
    if rescale:                      # D9 fix: absorb the global gate into the corrector
        rv, rt = g1 * rv, g1 * rt
        g1_eff = m1.ls_gate(Rv, rv)  # equals 1 up to numerical error, by construction
    else:
        g1_eff = g1
    Fva, Fte = va[F].to_numpy(), te[F].to_numpy()
    vcol = "rs7" if "rs7" in F else "rs6"
    lo, hi = np.nanquantile(tr[vcol], [1 / 3, 2 / 3])
    reg_v, reg_t = m1.regime_labels(va, lo, hi), m1.regime_labels(te, lo, hi)
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()
    g2map = {s: (m1.ls_gate(Rv[sid_v == s], rv[sid_v == s]) if (sid_v == s).sum() >= MIN_CELL else g1_eff)
             for s in np.unique(sid_v)}
    g2 = np.array([g2map.get(s, g1_eff) for s in sid_t])
    g3map = {}
    for s in np.unique(sid_v):
        for r_ in range(m1.N_REGIMES):
            k = (sid_v == s) & (reg_v == r_)
            g3map[(s, r_)] = m1.ls_gate(Rv[k], rv[k]) if k.sum() >= MIN_CELL else g2map.get(s, g1_eff)
    g3 = np.array([g3map.get((s, r_), g1_eff) for s, r_ in zip(sid_t, reg_t)])
    bw = np.empty((N_BOOT, len(Fte)))
    for b in range(N_BOOT):
        rb = np.random.default_rng(5100 + 89 * seed + b)
        idx = rb.integers(0, len(Fva), len(Fva))
        bw[b] = wls_gate(Fva[idx], Rv[idx], rv[idx], seed * 19 + b).predict(Fte)
    g5 = bw.mean(axis=0)
    V, W = float(np.mean(bw.var(axis=0, ddof=1))), float(np.var(g5))
    lam = W / (W + V) if (W + V) > 0 else 0.0
    gates = {"rung1_global": np.full(len(te), g1_eff), "rung2_series": g2, "rung3_cell": g3,
             "rung5_wls_bagged": g5, "rung4_shrunk": lam * g5 + (1 - lam) * g2}
    base = Rt ** 2
    out = []
    for rung, g in gates.items():
        frac = float(np.mean((g < 0) | (g > CLIP)))
        gc = np.clip(g, 0, CLIP)
        sq = (Rt - gc * rt) ** 2
        msq = [1 - sq[sid_t == s].mean() / base[sid_t == s].mean() for s in np.unique(sid_t)]
        out.append(dict(baseline=bname, h=h, seed=seed, rescaled=int(rescale), rung=rung,
                        skill_sq_macro=float(np.mean(msq)), frac_clipped=frac,
                        gate_mean=float(np.mean(gc)), gate_absmax=float(np.max(np.abs(g))),
                        g1=g1, lam=lam, rho=(W / V if V > 0 else np.nan)))
    return out


def main():
    ds = m1.load_india()
    rows, t0 = [], time.time()
    for bname in ["persistence", "roll7"]:
        for h in [1, 3, 7]:
            for rescale in (False, True):
                for seed in range(N_SEEDS):
                    rows += one(ds, bname, h, seed, rescale)
            d = pd.DataFrame(rows).query("baseline == @bname and h == @h")
            p = d.pivot_table(index="rung", columns="rescaled", values="skill_sq_macro")
            c = d.pivot_table(index="rung", columns="rescaled", values="frac_clipped")
            print(f"{bname:12s} h={h} | " + "  ".join(
                f"{r.replace('rung','r')[:12]}: {p.loc[r,0]:+.4f}->{p.loc[r,1]:+.4f} (clip {c.loc[r,0]:.2f}->{c.loc[r,1]:.2f})"
                for r in ["rung2_series", "rung5_wls_bagged"]) + f" [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "M1c_rescaled.csv"), index=False)
    print("\n=== D9 TEST: effect of absorbing the global gate (mean over 6 configs x 10 seeds) ===")
    piv = R.pivot_table(index="rung", columns="rescaled", values=["skill_sq_macro", "frac_clipped", "gate_mean"])
    print(piv.round(4).to_string())
    print("\n=== paired test per rung (unit = config x seed) ===")
    for rung in ["rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_wls_bagged"]:
        a = R[(R.rung == rung) & (R.rescaled == 0)].set_index(["baseline", "h", "seed"]).skill_sq_macro
        b = R[(R.rung == rung) & (R.rescaled == 1)].set_index(["baseline", "h", "seed"]).skill_sq_macro
        j = a.align(b, join="inner")
        t = stats.ttest_rel(j[1], j[0])
        print(f"  {rung:18s} rescaled - original = {float((j[1]-j[0]).mean()):+.5f}  "
              f"improves in {int((j[1] > j[0]).sum())}/{len(j[0])}  p={t.pvalue:.2e}")
    # does the fine rung close the gap to per-series after rescaling?
    for rs in (0, 1):
        a = R[(R.rung == "rung5_wls_bagged") & (R.rescaled == rs)].set_index(["baseline", "h", "seed"]).skill_sq_macro
        b = R[(R.rung == "rung2_series") & (R.rescaled == rs)].set_index(["baseline", "h", "seed"]).skill_sq_macro
        j = a.align(b, join="inner")
        print(f"  fine - per-series (rescaled={rs}): {float((j[0]-j[1]).mean()):+.5f} "
              f"wins={int((j[0] > j[1]).sum())}/{len(j[0])}")
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
