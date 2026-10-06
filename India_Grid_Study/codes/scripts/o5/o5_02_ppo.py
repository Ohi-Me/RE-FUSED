"""O5 step 2: arm A1 — PPO scheduling policy without a risk term (protocol O5 + addendum v1.1).

Environment: one episode = 28 consecutive decision days of one State sampled from the validation year (FY2023-24).
Observation on day t−1 (published information only):
  calibrated forecast quantiles of X(t) relative to the median, scaled by the State's MASE scale (6);
  median relative to the previous own schedule (1);
  mean absolute deviation of the own schedule over published days t−8…t−2 (1);
  rolling 7-day budget before the decision, scaled (1);
  U and system-stress percentiles from O4 (2); day-of-week sine/cosine of t (2); published price ratio p̂/λ̂ (1).
Action a ∈ [−1, 1] → quantile level ℓ = 0.05 + 0.45(a + 1) → S = scenario value at ℓ, projected onto the hard-limit set.
Reward = −regret(t) / (mean |regret| of the P arm for that State on validation).
PPO (Stable-Baselines3 2.7.1, MlpPolicy defaults), 200k steps, seeds 0–4. Each trained policy is replayed
deterministically over all States and days of validation (selection by J) and the development test.

Usage: py -3.10 o5_02_ppo.py [O2 model] [--steps N] [--seeds K] [--quick]   (reads results/dev/o5/decisions.parquet and O5 frame)
Outputs: results/dev/o5/ppo_decisions.parquet, ppo_metrics.csv
"""
import os
import sys
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import o5_01_scheduling as O5  # noqa: E402

OUT = O5.OUT


def lev_to_s(sc, lev):
    i = np.clip(lev * 100 - 1, 0, 98)
    lo = np.floor(i).astype(int)
    hi = np.minimum(lo + 1, 98)
    w = i - lo
    return sc[lo] * (1 - w) + sc[hi] * w


class SchedEnv(gym.Env):
    def __init__(self, F, SC, scale, norm, episodes=True, rng=None):
        super().__init__()
        self.F, self.SC, self.scale, self.norm = F, SC, scale, norm
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.rng = rng or np.random.default_rng(0)
        self.by_entity = {e: g.sort_values("date").index.to_numpy() for e, g in F.groupby("entity")}
        self.starts = [(e, k) for e, idx in self.by_entity.items() for k in range(len(idx) - 28)
                       if (F.date[idx[k + 27]] - F.date[idx[k]]).days == 27]

    def _obs(self):
        i = self.idx[self.t]
        r = self.F.loc[i]
        sc = self.scale[r.entity]
        q = r[O5.E2.QN].to_numpy(float)
        prev = self.prev_S if self.prev_S is not None else r.sched_prev if np.isfinite(r.sched_prev) else q[3]
        hist = self.hist
        pub = hist[:-1][-7:] if len(hist) > 1 else []
        mad = np.mean([abs(x - s) for x, s in pub]) / sc if len(pub) else 0.0
        last7 = hist[-7:]
        budget = (0.05 * sum(abs(s) for _, s in last7) - sum(abs(x - s) for x, s in last7)) / sc if last7 else 0.0
        dow = r.date.dayofweek
        return np.array([*((np.delete(q, 3) - q[3]) / sc), (q[3] - prev) / sc, mad, budget, r.U_p, r.S_p,
                         np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                         r.p_hat / max(r.lam_hat, 1.0)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        e, k = self.starts[self.rng.integers(len(self.starts))]
        self.idx, self.t, self.prev_S, self.prev_date, self.hist = self.by_entity[e][k:k + 28], 0, None, None, []
        return self._obs(), {}

    def step(self, action):
        i = self.idx[self.t]
        r = self.F.loc[i]
        lev = 0.05 + 0.45 * (float(np.clip(action[0], -1, 1)) + 1)
        s_raw = lev_to_s(self.SC[i], lev)
        lo, hi = r.S_min, r.S_max
        ref = self.prev_S if self.prev_S is not None else (r.sched_prev if np.isfinite(r.sched_prev) else None)
        if ref is not None:
            lo, hi = max(lo, ref - r.dS_max), min(hi, ref + r.dS_max)
        s = float(np.clip(s_raw, lo, hi)) if lo <= hi else float(np.clip(s_raw, r.S_min, r.S_max))
        reg = float(O5.regret(s, r.X, r.p, r.lam))
        self.hist.append((r.X, s))
        self.prev_S = s
        self.t += 1
        done = self.t >= len(self.idx)
        obs = self._obs() if not done else np.zeros(14, np.float32)
        return obs, -reg / self.norm[r.entity], done, False, {"S": s, "regret": reg}


def replay(model, F, SC, scale, norm):
    """Deterministic sequential replay over all States and days in F."""
    out = []
    for e, g in F.groupby("entity"):
        idx = g.sort_values("date").index.to_numpy()
        env = SchedEnv(F, SC, scale, norm)
        env.idx, env.t, env.prev_S, env.hist = idx, 0, None, []
        obs = env._obs()
        for _ in range(len(idx)):
            a, _ = model.predict(obs, deterministic=True)
            i = env.idx[env.t]
            obs, _, done, _, info = env.step(a)
            out.append(dict(entity=e, date=F.date[i], S_A1=info["S"], regret_A1=info["regret"]))
            if done:
                break
    return pd.DataFrame(out)


def main():
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 200_000
    n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 5
    out = os.path.join(OUT, "_quick_check") if "--quick" in sys.argv else OUT
    dec = pd.read_parquet(os.path.join(out, "decisions.parquet"))
    skip = {sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a in ("--steps", "--seeds")}
    model_name = [a for a in sys.argv[1:] if not a.startswith("--") and a not in skip]
    import json
    mdl = model_name[0] if model_name else next(x["selected"] for x in json.load(open(os.path.join(O5.O2, "eval", "selection.json")))
                                                if x["tid"] == "T2")
    F = O5.build(mdl)
    SC = O5.scenarios(F[O5.E2.QN].to_numpy())
    sc = pd.read_parquet(os.path.join(O5.O2, "scales.parquet"))
    scale = sc[(sc.tid == "T2") & (sc.H == 1)].set_index("sid").scale_mae.to_dict()
    val = F[F.split == "validation"]
    pn = dec[dec.split == "validation"].groupby("entity").regret_P.apply(lambda s: s.abs().mean()).clip(lower=1e-3).to_dict()
    norm = {e: pn.get(e, 1.0) for e in F.entity.unique()}
    results, metrics = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        env = SchedEnv(val, SC, scale, norm, rng=np.random.default_rng(seed))
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, device=os.environ.get("REFUSED_PPO_DEVICE", "cpu"))
        model.learn(total_timesteps=steps)
        rp = replay(model, F, SC, scale, norm)
        rp["seed"] = seed
        rp = rp.merge(F[["entity", "date", "split"]], on=["entity", "date"])
        results.append(rp)
        for sp in ("validation", "dev_test"):
            sysd = rp[rp.split == sp].groupby("date").regret_A1.sum()
            metrics.append(dict(seed=seed, split=sp, system_mean_crore=float(sysd.mean()),
                                system_cvar90_crore=float(np.sort(sysd.to_numpy())[-int(np.ceil(0.1 * len(sysd))):].mean()),
                                J=O5.J(sysd), seconds=round(time.time() - t0, 1)))
        print(f"PPO seed {seed}: " + ", ".join(f"{m['split']} J={m['J']:.3f}" for m in metrics[-2:]), flush=True)
    R = pd.concat(results, ignore_index=True)
    M = pd.DataFrame(metrics)
    best = int(M[M.split == "validation"].sort_values("J").seed.iloc[0])
    M["selected_on_validation"] = M.seed == best
    R.to_parquet(os.path.join(out, "ppo_decisions.parquet"), index=False)
    M.to_csv(os.path.join(out, "ppo_metrics.csv"), index=False)
    print(M.round(4).to_string())


if __name__ == "__main__":
    main()
