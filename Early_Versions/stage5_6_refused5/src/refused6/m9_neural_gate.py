"""RE-FUSED-6 / M9: does the dominance of coarse gates hold against the field's default NEURAL gate?

Reviewer objection 1.3: the fine gate in RE-FUSED-6 is gradient-boosted (ratio / WLS), but recent residual
correction work uses a bounded sigmoid gate trained end-to-end. Here:

  neural_gate   g(x) = 1.5 * sigmoid(MLP([standardised features, series one-hot]))
                trained on V1 with Adam + weight decay, early stopping on V2 (validation only)
compared, on identical runs, with:
  per_series    least-squares gate per series on all validation data
  per_series_V1 the same, fitted on V1 only (matched data budget to the neural gate's training set)
  global, none

The neural gate is scale-invariant in (R, r_hat), so both are divided by std(R on V1) for training stability.
"""
import os, time, importlib.util, warnings
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


def train_gate(Xa, Ra, ra, Xb, Rb, rb, seed, fixed_epochs=None):
    """Early stopping on (Xb, Rb, rb); with fixed_epochs, train that many epochs on Xa and ignore Xb."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-8
    s = float(np.std(Ra)) + 1e-8
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=DEV)
    xa, Ra_, ra_ = T((Xa - mu) / sd), T(Ra / s), T(ra / s)
    if not fixed_epochs:
        xb, Rb_, rb_ = T((Xb - mu) / sd), T(Rb / s), T(rb / s)
    net = nn.Sequential(nn.Linear(Xa.shape[1], 64), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
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
            gb = CLIP * torch.sigmoid(net(xb).squeeze(-1))
            vl = float(((Rb_ - gb * rb_) ** 2).mean())
        if vl < best - 1e-9:
            best, bad, best_ep = vl, 0, ep + 1
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    if not fixed_epochs:
        net.load_state_dict(state)
    else:
        best_ep = fixed_epochs
    net.eval()
    def predict(X):
        with torch.no_grad():
            return (CLIP * torch.sigmoid(net(T((X - mu) / sd)).squeeze(-1))).cpu().numpy()
    return predict, best_ep


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
    predict, epochs = train_gate(Xv[a], Rv[a], rv[a], Xv[b], Rv[b], rv[b], seed)
    g_nn = predict(Xt)
    predict_all, _ = train_gate(Xv, Rv, rv, None, None, None, seed, fixed_epochs=max(epochs, 1))
    g_nn_refit = predict_all(Xt)
    sid_v, sid_t = va.series_id.to_numpy(), te.series_id.to_numpy()
    def per_series(mask):
        mp = {s: m1.ls_gate(Rv[mask & (sid_v == s)], rv[mask & (sid_v == s)])
              if (mask & (sid_v == s)).sum() >= MIN_CELL else m1.ls_gate(Rv[mask], rv[mask]) for s in np.unique(sid_v)}
        return np.array([mp.get(s, 1.0) for s in sid_t])
    gates = {"none": np.zeros(len(te)), "global": np.full(len(te), m1.ls_gate(Rv, rv)),
             "per_series": per_series(np.ones(len(va), bool)), "per_series_V1": per_series(a),
             "neural_gate": g_nn, "neural_gate_refit": g_nn_refit}
    base = Rt ** 2
    rows = []
    for name, g in gates.items():
        sq = (Rt - np.clip(g, 0, CLIP) * rt) ** 2
        msq = [1 - sq[sid_t == s].mean() / base[sid_t == s].mean() for s in np.unique(sid_t)]
        rows.append(dict(dataset=ds["name"], baseline=bname, h=h, seed=seed, gate=name,
                         skill_sq_macro=float(np.mean(msq)), degraded_series=int(sum(v < 0 for v in msq)),
                         gate_sd=float(np.std(np.clip(g, 0, CLIP))), epochs=epochs))
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
        if len(d):
            m = d.groupby("gate").skill_sq_macro.mean()
            print(f"{ds['name']:11s} {b:12s} h={h:2d} | " + " ".join(f"{k}={m[k]:+.4f}" for k in m.index) +
                  f" | nn_epochs={d[d.gate=='neural_gate'].epochs.mean():.0f} [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "M9_neural_gate.csv"), index=False)
    print(f"\nwrote {len(R)} rows on {DEV}")
    print(R.pivot_table(index=["dataset", "baseline", "h"], columns="gate", values="skill_sq_macro").round(4).to_string())
    key = ["dataset", "baseline", "h", "seed"]
    # the verdict is taken against the STRONGER neural variant (conservative for H-M9)
    strong = R[R.gate.isin(["neural_gate", "neural_gate_refit"])].groupby("gate").skill_sq_macro.mean().idxmax()
    print(f"  stronger neural variant: {strong}")
    res = {}
    for nv in ["neural_gate", "neural_gate_refit"]:
        nn_ = R[R.gate == nv].set_index(key).skill_sq_macro
        for ref in ["per_series", "per_series_V1", "global"]:
            rr = R[R.gate == ref].set_index(key).skill_sq_macro
            j = nn_.align(rr, join="inner"); d = j[0] - j[1]
            p = stats.wilcoxon(j[0], j[1]).pvalue if (d != 0).sum() > 5 else float("nan")
            p_nn = stats.wilcoxon(j[0], j[1], alternative="greater").pvalue if (d != 0).sum() > 5 else 1.0
            res[(nv, ref)] = (d.mean(), p_nn)
            print(f"  {nv} - {ref:14s} mean={d.mean():+.5f} neural better in {int((d>0).sum())}/{len(d)} wilcoxon_p={p:.2e} p[neural better]={p_nn:.3f}")
    cfg = R.pivot_table(index=["dataset", "baseline", "h"], columns="gate", values="skill_sq_macro")
    n_cfg_nn = int((cfg[strong] > cfg["per_series"]).sum())
    # pre-registered verdict (docs/08, H-M9)
    c1 = res[(strong, "per_series")][1] >= 0.05
    c2 = n_cfg_nn <= len(cfg) / 2
    print("\n=== PRE-REGISTERED VERDICT FOR H-M9 ===")
    print(f"  {'PASS' if c1 else 'FAIL'}  (i)  {strong} NOT significantly better than per-series (p[neural better]={res[(strong, 'per_series')][1]:.3f} >= 0.05)")
    print(f"  {'PASS' if c2 else 'FAIL'}  (ii) {strong} beats per-series in {n_cfg_nn}/{len(cfg)} configs <= half")
    print("  H-M9", "SURVIVES (coarse dominance transfers to the neural gate family)" if (c1 and c2)
          else "FALSIFIED (dominance claim must be restricted to boosted gates)")
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
