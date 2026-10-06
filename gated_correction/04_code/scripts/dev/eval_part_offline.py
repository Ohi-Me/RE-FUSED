"""Offline development evaluation of PART designs (DV5) against realised test gains.

For each development ladder run, rebuild its gate-fitting set exactly as the engine does (same seed, corrector,
D9 rescale), compute PART variants on validation only, and pair them with the realised test gain computed from the
stored per-row outputs. Output: 06_results/dev/analysis/part_offline_<experiment>.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from refused_gate import gates as G, ladder, selection as S  # noqa: E402
from refused_gate.paths import RES  # noqa: E402
import run_dev_ladder as RDL  # noqa: E402


def realised(out, coarse, fine):
    R = (out.y_t - out.B).to_numpy()
    r = out.r.to_numpy()
    base = float(np.mean(R ** 2))
    gain = np.mean((R - out["g_" + coarse].to_numpy() * r) ** 2) - np.mean((R - out["g_" + fine].to_numpy() * r) ** 2)
    return float(gain / base), base


def main():
    exp = sys.argv[1]
    do_instance = "--instance" in sys.argv
    rows = []
    t0 = time.time()
    for key, fr, feats, keys, cfg, seed in RDL.EXPERIMENTS[exp](range(3)):
        f_out = os.path.join(RES, "dev", exp, key + ".parquet")
        if not os.path.exists(f_out):
            continue
        out = pd.read_parquet(f_out)
        x = fr.dropna(subset=feats + ["y_t", "B"]).copy()
        x["R"] = x.y_t - x.B
        x["day"] = x.ts.dt.normalize()
        tr, va = x[x.split == "train"], x[x.split == "val"].copy()
        corr = ladder._fit_corrector(tr[feats].to_numpy(np.float32), tr.R.to_numpy(), cfg, seed)
        va["r"] = corr.predict(va[feats].to_numpy(np.float32))
        va["r"] *= G.ls_gate(va.R.to_numpy(), va.r.to_numpy())
        R, r, sid, day = va.R.to_numpy(), va.r.to_numpy(), va.sid.to_numpy(), va.day.to_numpy()
        _, base = realised(out, "global", "series")
        refinements = [("global", "series", np.zeros(len(va), int), sid)]
        refinements += [("series", f"series_x_{k}", sid, np.asarray(G.cell_keys(va.sid, va[k]))) for k in keys]
        for coarse, fine, ck, fk in ([] if "--time-only" in sys.argv else refinements):
            real, _ = realised(out, coarse, fine)
            for mode in ("interleaved", "contiguous"):
                p = S.part2_partition(R, r, ck, fk, day, mode, 6, cfg.min_cell)
                rows.append(dict(experiment=exp, config=key.rsplit("_s", 1)[0], seed=seed, refinement=f"{coarse}->{fine}",
                                 predictor=f"part2_{mode}", pred=p["G"] / base, realised=real))
        for bd, fine in ((30, "time_rolling_30"), (90, "time_rolling_90")):
            if "g_" + fine in out:
                real, _ = realised(out, "time_expanding", fine)     # DF8: matched comparator
                p = S.part2_time(R, r, sid, day, bd)
                rows.append(dict(experiment=exp, config=key.rsplit("_s", 1)[0], seed=seed, refinement=f"expanding->{fine}",
                                 predictor="part2_time", pred=p["G"] / base, realised=real))
                q = S.prequential_time(R, r, sid, day, window=bd, delay=cfg.delay_days)
                rows.append(dict(experiment=exp, config=key.rsplit("_s", 1)[0], seed=seed, refinement=f"expanding->{fine}",
                                 predictor="prequential", pred=q["G"] / base, realised=real))
        for lam, fine in ((0.98, "time_forget_0.98"), (0.995, "time_forget_0.995")):
            if "g_" + fine in out:
                real, _ = realised(out, "time_expanding", fine)     # DF8: matched comparator
                q = S.prequential_time(R, r, sid, day, lam=lam, delay=cfg.delay_days)
                rows.append(dict(experiment=exp, config=key.rsplit("_s", 1)[0], seed=seed, refinement=f"expanding->{fine}",
                                 predictor="prequential", pred=q["G"] / base, realised=real))
        if do_instance and "g_instance" in out:
            real, _ = realised(out, "series", "instance")
            ev = np.random.default_rng(seed).choice(len(va), min(15000, len(va)), replace=False)
            p = S.part2_instance(va[feats].to_numpy(np.float32), R, r, sid, day, "interleaved", 4, ev, seed, **cfg.instance)
            rows.append(dict(experiment=exp, config=key.rsplit("_s", 1)[0], seed=seed, refinement="series->instance",
                             predictor="part2_interleaved", pred=p["G"] / base, realised=real))
        print(f"  {key} [{time.time()-t0:.0f}s]", flush=True)
    D = pd.DataFrame(rows)
    dst = os.path.join(RES, "dev", "analysis", f"part_offline_{exp}{'_time' if '--time-only' in sys.argv else ''}.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    D.to_csv(dst, index=False)
    D["correct"] = np.sign(D.pred) == np.sign(D.realised)
    print(D.groupby(["predictor"]).correct.agg(["size", "mean"]).round(3).to_string())
    print(D.groupby(["refinement", "predictor"]).agg(n=("correct", "size"), acc=("correct", "mean"),
                                                     pred=("pred", "mean"), real=("realised", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
