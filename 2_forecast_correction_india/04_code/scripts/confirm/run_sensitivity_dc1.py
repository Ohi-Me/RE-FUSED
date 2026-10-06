"""Deviation DC1 — sensitivity analysis after unblinding (NOT the pre-registered analysis).

Re-runs the OPSD confirmatory blocks C3 (2019), C4 (2020) and C5 (2019 data budget) with the plausibility filter
`refused_gate.data.opsd_plausibility_mask` (training-period quantile bounds), using exactly the frozen ladder configurations.
Outputs: 06_results/confirm_sensitivity/C3, C4, C5 (C1, C2, C6, C7 are symlinked conceptually: the analysis map reuses
the pre-registered NYISO and UCI results unchanged).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused_gate import data, guard, ladder  # noqa: E402
from refused_gate.paths import RES  # noqa: E402


def jobs():
    for period, blk in (("2019", "C3"), ("2020", "C4")):
        op = data.opsd_load("confirm", period, clean=True)
        for b, col in op["baselines"].items():
            fr = op["df"].assign(y_t=op["df"].y, B=op["df"][col])
            for s in range(2):
                yield blk, f"{b}_s{s}", fr, op["feats"], ["hour_block", "hour"], ladder.LadderConfig(n_bag=3, delay_days=2), s
    op = data.opsd_load("confirm", "2019", clean=True)
    fr = op["df"].assign(y_t=op["df"].y, B=op["df"].tso_fc)
    for s in range(2):
        for frac in (0.1, 0.25):
            yield "C5", f"val_frac{frac}_s{s}", fr, op["feats"], ["hour_block", "hour"], \
                ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit_fraction=frac, do_time=False), s
        yield "C5", f"oof_train_val_s{s}", fr, op["feats"], ["hour_block", "hour"], \
            ladder.LadderConfig(n_bag=3, delay_days=2, gate_fit="oof_train+val", do_time=False), s


def main():
    guard.require_confirmatory()
    t0 = time.time()
    for blk, key, fr, feats, keys, cfg, s in jobs():
        out_dir = os.path.join(RES, "confirm_sensitivity", blk)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, key + ".parquet")
        if os.path.exists(dst):
            continue
        out, diag = ladder.run_ladder(fr, feats, keys, cfg, seed=s)
        out.to_parquet(dst, index=False)
        diag["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}
        diag["deviation"] = "DC1 plausibility filter"
        json.dump(diag, open(dst.replace(".parquet", ".json"), "w"), indent=1, default=str)
        print(f"{blk} {key:24s} done [{diag['seconds']}s, total {time.time()-t0:.0f}s]", flush=True)
    print("DONE DC1 sensitivity", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
