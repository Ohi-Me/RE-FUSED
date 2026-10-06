"""dev2 step 11: PPO scheduling arm (A1) with the round-2 forecaster, on the GPU.

Same environment, reward and replay as round 1 (o5_02_ppo.py): one episode is 28 days of one State from the
validation year, the action picks a quantile level of the calibrated forecast, and the reward is minus the realised
regret scaled by that State's mean |regret| under the plain forecast arm. What changes is the forecast source (the
final T2 forecaster of round 2) and that the policy network runs on the GPU. Five seeds; the seed is chosen on
validation J, and the chosen policy is compared with the round-2 arms on the development test.

Outputs: results/dev2/o5/{ppo_decisions.parquet, ppo_metrics.csv, ppo_tests.csv}
Usage:   python d2_11_ppo.py [--steps N] [--seeds K] [--smoke]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
for sub in ("o2", "o4", "o5"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
sys.path.insert(0, HERE)
from refused import dev2 as D  # noqa: E402
from refused import stats as S  # noqa: E402
from refused.paths import TAB  # noqa: E402
from refused.report import md_table  # noqa: E402
import o2_06_evaluate as E2  # noqa: E402
import o5_01_scheduling as O5  # noqa: E402
import o5_02_ppo as P1  # noqa: E402
import d2_09_o5 as O5v2  # noqa: E402

SMOKE = "--smoke" in sys.argv


def main():
    t0 = time.time()
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else (2000 if SMOKE else 200_000)
    n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else (1 if SMOKE else 5)
    model, window, _ = O5v2.final_t2()
    F = O5v2.build_v2(model, window)
    SC = O5.scenarios(F[D.QN].to_numpy())
    sc = D.scales("T2")
    scale = sc[sc.H == 1].set_index("sid").scale_mae.to_dict()
    out = D.folder("o5")
    dec = pd.read_parquet(os.path.join(out, "decisions.parquet"))
    pn = dec[dec.split == "validation"].groupby("entity").regret_P.apply(lambda s: s.abs().mean()).clip(lower=1e-3).to_dict()
    norm = {e: pn.get(e, 1.0) for e in F.entity.unique()}
    val = F[F.split == "validation"]
    results, metrics = [], []
    for seed in range(n_seeds):
        ts = time.time()
        env = P1.SchedEnv(val, SC, scale, norm, rng=np.random.default_rng(seed))
        policy = PPO("MlpPolicy", env, seed=seed, verbose=0, device=os.environ.get("REFUSED_PPO_DEVICE", "cuda"))
        policy.learn(total_timesteps=steps)
        rp = P1.replay(policy, F, SC, scale, norm).assign(seed=seed)
        rp = rp.merge(F[["entity", "date", "split"]], on=["entity", "date"])
        results.append(rp)
        for sp in ("validation", "dev_test"):
            sysd = rp[rp.split == sp].groupby("date").regret_A1.sum()
            metrics.append(dict(seed=seed, split=sp, system_mean_crore=float(sysd.mean()),
                                system_cvar90_crore=float(np.sort(sysd.to_numpy())[-max(1, int(np.ceil(0.1 * len(sysd)))):].mean()),
                                J=O5.J(sysd), seconds=round(time.time() - ts, 1)))
        print(f"PPO seed {seed}: " + ", ".join(f"{m['split']} J={m['J']:.3f}" for m in metrics[-2:]), flush=True)
    R = pd.concat(results, ignore_index=True)
    M = pd.DataFrame(metrics)
    best = int(M[M.split == "validation"].sort_values("J").seed.iloc[0])
    M["selected_on_validation"] = M.seed == best
    R.to_parquet(os.path.join(out, "ppo_decisions.parquet"), index=False)
    M.to_csv(os.path.join(out, "ppo_metrics.csv"), index=False)
    # the chosen policy against the round-2 arms on the development test
    a1 = R[(R.seed == best) & (R.split == "dev_test")].groupby("date").regret_A1.sum().sort_index()
    tests = []
    for arm in [c[len("regret_"):] for c in dec.columns if c.startswith("regret_")]:
        other = dec[dec.split == "dev_test"].groupby("date")[f"regret_{arm}"].sum().sort_index()
        dd = (a1 - other).dropna().to_numpy()
        if len(dd) < 20:
            continue
        ci = S.block_ci(dd, 7, n_boot=500 if SMOKE else 2000, seed=0)
        tests.append(dict(a="A1 (PPO)", b=arm, metric="mean system regret", mean_diff=float(dd.mean()), lo=ci["lo"],
                          hi=ci["hi"], p_a_better=E2.hln(dd, 1)[1], n_days=len(dd)))
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(out, "ppo_tests.csv"), index=False)
    L = ["# O5 round 2 — PPO arm (A1) with the round-2 forecaster", "",
         f"Forecast source `{model}` (window {window} days), {steps:,} steps per seed, {n_seeds} seeds, policy on the GPU. "
         f"Seed chosen on validation J: {best}.", "", md_table(M, 4), "",
         "Development test, PPO against the other arms (negative = PPO better):", "", md_table(T, 4), ""]
    open(os.path.join(TAB, f"{D.TAG}_o5_ppo_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"PPO round 2 done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
