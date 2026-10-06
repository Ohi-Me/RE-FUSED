"""RE-FUSED-6 / M9b: robustness of H-M9 to neural-gate hyperparameters (reviewer objection: "under-tuned competitor").

neural_gate_tuned: a grid of 8 configurations (hidden width x weight decay x learning rate) is trained on V1 with
early stopping on V2; the configuration with the lowest V2 loss is retrained on all of V for its early-stopped
epoch count. Selection uses validation data only. Same datasets, seeds, corrector and scoring as M9.
"""
import os, time, itertools, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1c", os.path.join(HERE, "m1_canonical.py"))
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)
OUT = r"D:\\REFUSED5\results\refused6"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP, MIN_CELL = 1.5, 30
GRID = list(itertools.product([32, 128], [1e-5, 1e-3], [1e-3, 3e-3]))   # hidden, weight decay, lr


def train(Xa, Ra, ra, Xb, Rb, rb, seed, hidden, wd, lr, fixed_epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-8
    s = float(np.std(Ra)) + 1e-8
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=DEV)
    xa, Ra_, ra_ = T((Xa - mu) / sd), T(Ra / s), T(ra / s)
    if not fixed_epochs:
        xb, Rb_, rb_ = T((Xb - mu) / sd), T(Rb / s), T(rb / s)
    net = nn.Sequential(nn.Linear(Xa.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best, state, bad, best_ep = np.inf, None, 0, 0
    n = len(xa)
    for ep in range(fixed_epochs if fixed_epochs else 200):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            g = CLIP * torch.sigmoid(net(xa[idx]).squeeze(-1))
            loss = ((Ra_[idx] - g * ra_[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if fixed_epochs:
            continue
        net.eval()
        with torch.no_grad():
            vl = float(((Rb_ - CLIP * torch.sigmoid(net(xb).squeeze(-1)) * rb_) ** 2).mean())
        if vl < best - 1e-9:
            best, bad, best_ep = vl, 0, ep + 1
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    if not fixed_epochs:
        net.load_state_dict(state)
    net.eval()
    def predict(X):
        with torch.no_grad():
            return (CLIP * torch.sigmoid(net(T((X - mu) / sd)).squeeze(-1))).cpu().numpy()
    return predict, best_ep, best


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
    sub = min(len(tr), 200_000)
    trs = tr.sample(sub, random_state=seed) if len(tr) > sub else tr
    corr = HGB(max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed, l2_regularization=1.0).fit(trs[F], trs.R)
    rv, rt = corr.predict(va[F]), corr.predict(te[F])
    Rv, Rt = va.R.to_numpy(), te.R.to_numpy()
    g1 = m1.ls_gate(Rv, rv)
    rv, rt = g1 * rv, g1 * rt
    series = sorted(x.series_id.unique())
    oh = lambda df: np.stack([(df.series_id.to_numpy() == s).astype(np.float32) for s in series], axis=1)
    Xv = np.hstack([va[F].to_numpy(np.float32), oh(va)])
    Xt = np.hstack([te[F].to_numpy(np.float32), oh(te)])
    tsv = pd.to_datetime(va.ts).to_numpy()
    cut = np.sort(np.unique(tsv))[len(np.unique(tsv)) // 2]
    a, b = tsv < cut, tsv >= cut
    trials = []
    for hidden, wd, lr in GRID:
        _, ep, vl = train(Xv[a], Rv[a], rv[a], Xv[b], Rv[b], rv[b], seed, hidden, wd, lr)
        trials.append((vl, hidden, wd, lr, ep))
    vl, hidden, wd, lr, ep = min(trials)
    pred, _, _ = train(Xv, Rv, rv, None, None, None, seed, hidden, wd, lr, fixed_epochs=max(ep, 1))
    g_nn = pred(Xt)
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()
    mp = {s: m1.ls_gate(Rv[sid_v == s], rv[sid_v == s]) if (sid_v == s).sum() >= MIN_CELL else m1.ls_gate(Rv, rv)
          for s in np.unique(sid_v)}
    gates = {"per_series": np.array([mp.get(s, 1.0) for s in sid_t]), "neural_gate_tuned": g_nn}
    base = Rt ** 2
    rows = []
    for name, g in gates.items():
        sq = (Rt - np.clip(g, 0, CLIP) * rt) ** 2
        msq = [1 - sq[sid_t == s].mean() / base[sid_t == s].mean() for s in np.unique(sid_t)]
        rows.append(dict(dataset=ds["name"], baseline=bname, h=h, seed=seed, gate=name,
                         skill_sq_macro=float(np.mean(msq)), degraded_series=int(sum(v < 0 for v in msq)),
                         hidden=hidden, wd=wd, lr=lr, epochs=ep))
    return rows


def main():
    jobs = []
    india = m1.load_india()
    jobs += [(india, b, h) for b in ["persistence", "roll7", "snaive7"] for h in [1, 3]]
    for e in m1.load_ett():
        if e["name"] in ("ETTh1", "ETTh2"):
            jobs += [(e, "persistence", h) for h in [1, 24]]
    rows, t0 = [], time.time()
    for ds, b, h in jobs:
        for seed in range(5):
            r = run(ds, b, h, seed)
            if r:
                rows += r
        d = pd.DataFrame(rows)
        d = d[(d.dataset == ds["name"]) & (d.baseline == b) & (d.h == h)]
        m = d.groupby("gate").skill_sq_macro.mean()
        chosen = d[d.gate == "neural_gate_tuned"][["hidden", "wd", "lr"]].astype(str).agg("/".join, axis=1).value_counts().to_dict()
        print(f"{ds['name']:11s} {b:12s} h={h:2d} | " + " ".join(f"{k}={m[k]:+.4f}" for k in m.index) +
              f" | chosen={chosen} [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "M9b_neural_tuned.csv"), index=False)
    key = ["dataset", "baseline", "h", "seed"]
    nn_ = R[R.gate == "neural_gate_tuned"].set_index(key).skill_sq_macro
    ps = R[R.gate == "per_series"].set_index(key).skill_sq_macro
    j = nn_.align(ps, join="inner"); d = j[0] - j[1]
    p_nn = stats.wilcoxon(j[0], j[1], alternative="greater").pvalue if (d != 0).sum() > 5 else 1.0
    p2 = stats.wilcoxon(j[0], j[1]).pvalue if (d != 0).sum() > 5 else float("nan")
    cfg = R.pivot_table(index=["dataset", "baseline", "h"], columns="gate", values="skill_sq_macro")
    n_cfg = int((cfg["neural_gate_tuned"] > cfg["per_series"]).sum())
    print(cfg.round(4).to_string())
    print(f"\nneural_gate_tuned - per_series mean={d.mean():+.5f} tuned better in {int((d>0).sum())}/{len(d)} two-sided p={p2:.2e}")
    print("\n=== PRE-REGISTERED VERDICT FOR H-M9b ===")
    c1, c2 = p_nn >= 0.05, n_cfg <= len(cfg) / 2
    print(f"  {'PASS' if c1 else 'FAIL'}  (i)  tuned neural gate NOT significantly better than per-series (p[neural better]={p_nn:.3f})")
    print(f"  {'PASS' if c2 else 'FAIL'}  (ii) tuned neural gate beats per-series in {n_cfg}/{len(cfg)} configs <= half")
    print("  H-M9b", "SURVIVES" if (c1 and c2) else "FALSIFIED")
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
