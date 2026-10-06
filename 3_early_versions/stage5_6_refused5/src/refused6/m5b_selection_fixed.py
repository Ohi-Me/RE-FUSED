"""RE-FUSED-6 / M5b: correct the rung-selection procedure (fixes my own methodological error).

Diagnosis from M3/M5: validation risk barely ranks the rungs (mean val-test rank correlation 0.126)
and the SRM selector's regret was worst in the MIDDLE of the n_val range (0.049 at n_val=1500)
precisely where the oracle rung had moved finer. Root cause: the gates were FITTED and EVALUATED on
the same validation data, so validation risk is optimistically biased in favour of finer rungs, and a
generic sqrt(log M / n) penalty mis-corrects that bias.

Two principled fixes, both validation-only:

  nested      split validation in two by TIME: fit gates on V1, select the rung on V2.
              Unbiased by construction, no penalty needed, no theory needed.

  mallows_k   keep fitting on all of V, but add the Proposition-1 estimation term
                  pen(k) = sum_c p_c * Var(R*rhat | c) / (n_c * E[rhat^2 | c])
              which is exactly the quantity the bias equals to first order. Reported at
              multiplier 1 (bias correction) and 2 (Mallows-style Cp).

Comparators: unpenalised validation argmin (the biased baseline), the old sqrt-penalty SRM,
fixed policies, and the oracle rung. Selection never touches test data.

Usage: py -3.10 -u src/refused6/m5b_selection_fixed.py [quick|full]
"""
import os, sys, time, importlib.util, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m35", os.path.join(HERE, "m3_m5_ladder.py"))
m35 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m35)

OUT = r"D:\\REFUSED5\results\refused6"
CLIP = m35.CLIP
LADDER = ["rung0_none", "rung1_global", "rung2_series", "rung3_cell", "rung4_shrunk", "rung5_instance"]


def risk(R, g, r):
    return float(np.mean((R - np.clip(g, 0, CLIP) * r) ** 2))


def cell_gates(R, r, cid, fallback, min_cell=30):
    out = {}
    for c in np.unique(cid):
        k = cid == c
        out[c] = m35.ls_gate(R[k], r[k]) if k.sum() >= min_cell else fallback
    return out


def prop1_penalty(R, r, cid):
    """sum_c p_c * Var(R*r|c) / (n_c * E[r^2|c]) : the first-order optimism of a per-cell gate."""
    tot, n = 0.0, len(R)
    for c in np.unique(cid):
        k = cid == c
        nc = int(k.sum())
        if nc < 5:
            continue
        er2 = float(np.mean(r[k] ** 2))
        if er2 <= 1e-12:
            continue
        v = float(np.var(R[k] * r[k], ddof=1))
        tot += (nc / n) * v / (nc * er2)
    return tot


def one(knobs, seed):
    p, tr, va, te = m35.simulate(knobs, seed)
    # corrector on train only
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    corr = HGB(max_iter=200, learning_rate=0.08, max_depth=6, random_state=seed).fit(tr["F"], tr["R"])
    rv, rt = corr.predict(va["F"]), corr.predict(te["F"])
    Rv, Rt = va["R"], te["R"]
    base_te = float(np.mean(Rt ** 2))

    # ---------- gates fitted on ALL of validation (as before) ----------
    g1 = m35.ls_gate(Rv, rv)
    g2m = cell_gates(Rv, rv, va["sid"], g1)
    g3m = cell_gates(Rv, rv, va["cid"], g1)
    w0 = m35.wls_gate(va["F"], Rv, rv, seed)
    g5_te, g5_va = w0.predict(te["F"]), w0.predict(va["F"])
    boots = np.vstack([m35.wls_gate(va["F"][i], Rv[i], rv[i], seed * 41 + b).predict(te["F"])
                       for b, i in enumerate(np.random.default_rng(seed).integers(
                           0, len(va["F"]), (m35.N_BOOT, len(va["F"]))))])
    V, W = float(np.mean(boots.var(axis=0, ddof=1))), float(np.var(boots.mean(axis=0)))
    lam = W / (W + V) if (W + V) > 0 else 0.0
    gt = {"rung0_none": np.zeros(len(Rt)), "rung1_global": np.full(len(Rt), g1),
          "rung2_series": np.array([g2m.get(s, g1) for s in te["sid"]]),
          "rung3_cell": np.array([g3m.get(c, g1) for c in te["cid"]]),
          "rung5_instance": g5_te}
    gt["rung4_shrunk"] = lam * boots.mean(axis=0) + (1 - lam) * gt["rung3_cell"]
    gv = {"rung0_none": np.zeros(len(Rv)), "rung1_global": np.full(len(Rv), g1),
          "rung2_series": np.array([g2m.get(s, g1) for s in va["sid"]]),
          "rung3_cell": np.array([g3m.get(c, g1) for c in va["cid"]]),
          "rung5_instance": g5_va}
    gv["rung4_shrunk"] = lam * g5_va + (1 - lam) * gv["rung3_cell"]
    test_risk = {k: risk(Rt, g, rt) for k, g in gt.items()}
    val_risk = {k: risk(Rv, g, rv) for k, g in gv.items()}

    # ---------- NESTED: fit gates on V1 (early half), select on V2 (late half) ----------
    ordr = np.argsort(va["t"])
    half = len(ordr) // 2
    i1, i2 = ordr[:half], ordr[half:]
    g1n = m35.ls_gate(Rv[i1], rv[i1])
    g2n = cell_gates(Rv[i1], rv[i1], va["sid"][i1], g1n)
    g3n = cell_gates(Rv[i1], rv[i1], va["cid"][i1], g1n)
    wn = m35.wls_gate(va["F"][i1], Rv[i1], rv[i1], seed + 5)
    g5n_v2 = wn.predict(va["F"][i2])
    nested_risk = {
        "rung0_none": risk(Rv[i2], np.zeros(len(i2)), rv[i2]),
        "rung1_global": risk(Rv[i2], np.full(len(i2), g1n), rv[i2]),
        "rung2_series": risk(Rv[i2], np.array([g2n.get(s, g1n) for s in va["sid"][i2]]), rv[i2]),
        "rung3_cell": risk(Rv[i2], np.array([g3n.get(c, g1n) for c in va["cid"][i2]]), rv[i2]),
        "rung5_instance": risk(Rv[i2], g5n_v2, rv[i2]),
    }
    nested_risk["rung4_shrunk"] = risk(
        Rv[i2], lam * g5n_v2 + (1 - lam) * np.array([g3n.get(c, g1n) for c in va["cid"][i2]]), rv[i2])

    # ---------- Proposition-1 plug-in penalties ----------
    pen = {"rung0_none": 0.0,
           "rung1_global": prop1_penalty(Rv, rv, np.zeros(len(Rv), dtype=int)),
           "rung2_series": prop1_penalty(Rv, rv, va["sid"]),
           "rung3_cell": prop1_penalty(Rv, rv, va["cid"]),
           "rung5_instance": prop1_penalty(Rv, rv, np.arange(len(Rv)) // 25)}   # ~25 pts per local cell
    pen["rung4_shrunk"] = lam ** 2 * pen["rung5_instance"] + (1 - lam) ** 2 * pen["rung3_cell"]

    sel = {"val_argmin": min(val_risk, key=val_risk.get),
           "nested": min(nested_risk, key=nested_risk.get),
           "mallows1": min(LADDER, key=lambda k: val_risk[k] + pen[k]),
           "mallows2": min(LADDER, key=lambda k: val_risk[k] + 2 * pen[k]),
           "srm_sqrt": min(LADDER, key=lambda k: val_risk[k] + float(np.mean(Rv ** 2))
                           * np.sqrt(np.log(6) / len(Rv)))}
    oracle = min(test_risk, key=test_risk.get)
    row = dict(seed=seed, n_val=p["n_val"], between=p["between"], within=p["within"], snr=p["snr"],
               phi=p["phi"], mis=p["mis"], drift=p["drift"], oracle_rung=oracle, lam=lam,
               base_te=base_te)
    for k in LADDER:
        row[f"risk_{k}"] = test_risk[k]
        row[f"pen_{k}"] = pen[k]
    for nm, pick in sel.items():
        row[f"sel_{nm}"] = pick
        row[f"regret_{nm}"] = (test_risk[pick] - test_risk[oracle]) / base_te
        row[f"correct_{nm}"] = int(pick == oracle)
    for k in LADDER:
        row[f"regret_fixed_{k}"] = (test_risk[k] - test_risk[oracle]) / base_te
    return row


def main(mode):
    grid = m35.grid("full" if mode == "full" else "quick")
    rows, t0 = [], time.time()
    for i, k in enumerate(grid):
        k = dict(k); k.pop("name", None)
        for s in range(10 if mode == "full" else 5):
            rows.append(one(k, s))
        d = pd.DataFrame(rows).tail(10 if mode == "full" else 5)
        print(f"[{i+1}/{len(grid)}] n_val={int(d.n_val.iloc[0]):6d} "
              f"nested={d.regret_nested.mean():+.5f} mallows1={d.regret_mallows1.mean():+.5f} "
              f"mallows2={d.regret_mallows2.mean():+.5f} val_argmin={d.regret_val_argmin.mean():+.5f} "
              f"fixed_cell={d.regret_fixed_rung3_cell.mean():+.5f} oracle={d.oracle_rung.mode().iloc[0][:12]} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"M5b_selection_{mode}.csv"), index=False)
    print(f"\nwrote {len(R)} rows")
    rg = [c for c in R.columns if c.startswith("regret_")]
    print("\n=== mean regret vs oracle rung (lower is better) ===")
    print(R[rg].mean().sort_values().round(5).to_string())
    print("\n=== selection accuracy (exact oracle-rung recovery) ===")
    print(R[[c for c in R.columns if c.startswith("correct_")]].mean().round(3).to_string())
    print("\n=== regret by validation size ===")
    print(R.groupby("n_val")[["regret_nested", "regret_mallows1", "regret_mallows2",
                              "regret_val_argmin", "regret_fixed_rung3_cell"]].mean().round(5).to_string())
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quick")
