"""Additional analysis, not pre-registered (delay study, Nov 2026): publication delay as forecast horizon.

E1  Horizon curve. LightGBM with the protocol settings (seed 0) on the published information set and on instant
    publication, horizons 1-14, every target. Tests whether the error with real delays at horizon h equals the
    instant-publication error at h + d, and estimates d per series.
E2  Value of each source's timeliness. Horizons 1-3, every target: the published information set with one source
    group made instant ("only_<g>"), and with every group instant except one ("all_but_<g>").

The samples come from the same `o2data.tabular_samples` as the confirmatory run; only the availability lags change.
For horizons 1-3 the configuration chosen in the confirmatory run (logs/lgbm_<tid>.json) is reused, and the instant
fit must reproduce the stored S-lat predictions (results/confirm/o2/preds/<tid>__lgbm_instant.parquet); the script
records the difference. Longer horizons choose num_leaves by validation pinball, as the protocol does.
Reads the panel in the confirmatory phase (REFUSED_PHASE=confirm, REFUSED_CONFIRMATORY=1); writes only to
results/extra/delay/, never into the confirmatory folders.
Usage: python codes/scripts/extra/x13_delay_experiments.py E1|E2 [T1 T2 ...]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
from refused import o2data as O  # noqa: E402
from refused.paths import RES  # noqa: E402
import o2_02_lgbm as LG  # noqa: E402  (fit_predict, to_frame, pinball_z: the protocol's own LightGBM code)

OUT = os.path.join(RES, "extra", "delay")
# source groups by column prefix (FIELD_REGISTRY.csv source): lags 1, 14, 2, 2, 2, 1 days
GROUPS = {"psp": ("dem_", "dev_", "reg_", "nat_", "sys_"), "dsm_prices": ("prc_",), "cea_npp": ("gen_",),
          "cea_re": ("re_all_",), "co2": ("co2_",), "imd_weather": ("wx_",)}
REG_LAGS = O.lags(False)


def lags_with(instant_groups):
    """Registry lags with the columns of the given source groups set to 0."""
    pre = tuple(p for g in instant_groups for p in GROUPS[g])
    return {k: (0 if k.startswith(pre) else v) for k, v in REG_LAGS.items()}


def samples(P, tid, instant_groups, horizons):
    """o2data.tabular_samples with patched lags and horizons (restored afterwards)."""
    old_lags, old_h = O.lags, O.HORIZONS
    L = lags_with(instant_groups)
    O.lags, O.HORIZONS = (lambda instant=False: L), list(horizons)
    try:
        S = O.tabular_samples(P, tid)
    finally:
        O.lags, O.HORIZONS = old_lags, old_h
    S["sid_c"] = S.sid.astype("category")
    return S


def fit_variant(S, horizons, choice, name):
    out = []
    for H in horizons:
        ev, Qz, _ = LG.fit_predict(S, H, choice[H], 0)
        out.append(LG.to_frame(ev, Qz, name))
    return pd.concat(out, ignore_index=True)


def choose_config(S, H):
    scores = {}
    for nl in (31, 63):
        ev, Qz, _ = LG.fit_predict(S, H, nl, 0)
        v = (ev.split == "validation").to_numpy()
        scores[nl] = LG.pinball_z(ev.z.to_numpy()[v], Qz[v])
    return min(scores, key=scores.get)


def summarise(pred, scales):
    d = pred[(pred.split == "dev_test") & pred.y.notna()].merge(scales, on=["sid", "H"], how="left")
    d["ae"] = (d.y - d.q50).abs()
    g = d.groupby(["model", "sid", "H"])
    return pd.DataFrame({"mae": g.ae.mean(), "mase": g.apply(lambda x: x.ae.mean() / x.scale_mae.iloc[0]),
                         "n": g.size()}).reset_index()


def run(exp, tid, P):
    t0 = time.time()
    os.makedirs(os.path.join(OUT, exp, "preds"), exist_ok=True)
    log = {"exp": exp, "tid": tid, "phase": O.PHASE}
    conf = json.load(open(os.path.join(RES, O.PHASE_DIR, "o2", "logs", f"lgbm_{tid}.json")))
    choice = {int(k): v for k, v in conf["num_leaves"].items()}       # horizons 1-3, as chosen in the confirmatory run
    horizons = list(range(1, 15)) if exp == "E1" else [1, 2, 3]
    S_pub = samples(P, tid, [], horizons)
    scales = O.mase_scales(S_pub)
    for H in horizons:
        if H not in choice:
            choice[H] = choose_config(S_pub, H)
    log["num_leaves"] = choice
    variants = {"published": []}
    if exp == "E1":
        variants["instant"] = list(GROUPS)
    else:
        variants.update({f"only_{g}": [g] for g in GROUPS})
        variants.update({f"all_but_{g}": [x for x in GROUPS if x != g] for g in GROUPS})
        variants["instant"] = list(GROUPS)
    preds = []
    for name, groups in variants.items():
        S = S_pub if not groups else samples(P, tid, groups, horizons)
        pr = fit_variant(S, horizons, choice, name)
        pr.to_parquet(os.path.join(OUT, exp, "preds", f"{tid}__{name}.parquet"), index=False)
        preds.append(pr)
        print(f"  {exp} {tid} {name} done [{time.time() - t0:.0f}s]", flush=True)
    # reproduction check: the instant fit at horizons 1-3 against the stored S-lat predictions
    ref = pd.read_parquet(os.path.join(RES, O.PHASE_DIR, "o2", "preds", f"{tid}__lgbm_instant.parquet"))
    mine = next(p for p in preds if p.model.iloc[0] == "instant")
    k = ["sid", "issue_date", "H"]
    m = ref[k + ["q50"]].merge(mine[mine.H <= 3][k + ["q50"]], on=k, suffixes=("_ref", "_new"))
    log["reproduce_lgbm_instant"] = dict(rows=int(len(m)), max_abs_diff_q50=float((m.q50_ref - m.q50_new).abs().max()))
    S_all = pd.concat(preds, ignore_index=True)
    summarise(S_all, scales).to_csv(os.path.join(OUT, exp, f"summary_{tid}.csv"), index=False)
    log["seconds"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(OUT, exp, f"log_{tid}.json"), "w"), indent=1, default=str)
    print(exp, tid, "done", log["reproduce_lgbm_instant"], f"{log['seconds']}s", flush=True)


if __name__ == "__main__":
    exp = sys.argv[1]
    assert exp in ("E1", "E2"), "first argument must be E1 or E2"
    P = O.load_panel()
    for tid in (sys.argv[2:] or list(O.TARGETS)):
        run(exp, tid, P)
