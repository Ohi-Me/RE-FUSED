"""Gate estimators along the unit, state and time axes (theory T2-T5).

All gates act on a fixed corrector prediction r and residual R = Y - B.
Callers apply the identical clip CLIP to every gate before scoring (RE-FUSED-6 D6).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

CLIP = (0.0, 1.5)
Z_CLIP = 10.0


def ls_gate(R, r, w=None):
    if w is None:
        den = float(np.dot(r, r))
        return float(np.dot(R, r) / den) if den > 1e-12 else 0.0
    den = float(np.sum(w * r * r))
    return float(np.sum(w * R * r) / den) if den > 1e-12 else 0.0


def clip(g):
    return np.clip(g, *CLIP)


def cell_keys(*arrays):
    """Combine several id arrays into one string key per row."""
    arrays = [np.asarray(a).astype(str) for a in arrays]
    out = arrays[0]
    for a in arrays[1:]:
        out = np.char.add(np.char.add(out, "|"), a)
    return out


class PartitionGate:
    """Least-squares gate constant on cells. Cells with fewer than min_cell rows use the parent value."""

    def __init__(self, min_cell=30):
        self.min_cell = min_cell

    def fit(self, R, r, keys):
        keys = np.asarray(keys)
        self.global_ = ls_gate(R, r)
        uniq, inv = np.unique(keys, return_inverse=True)
        num = np.bincount(inv, weights=R * r, minlength=len(uniq))
        den = np.bincount(inv, weights=r * r, minlength=len(uniq))
        cnt = np.bincount(inv, minlength=len(uniq))
        self.table = {u: nu / de for u, nu, de, c in zip(uniq, num, den, cnt) if c >= self.min_cell and de > 1e-12}
        return self

    def predict(self, keys, parent_values=None):
        keys = np.asarray(keys)
        out = np.fromiter((self.table.get(k, np.nan) for k in keys), float, count=len(keys))
        miss = np.isnan(out)
        if miss.any():
            out[miss] = np.asarray(parent_values, float)[miss] if parent_values is not None else self.global_
        return out


def _wls_model(F, R, r, seed, max_iter=150, learning_rate=0.08, max_depth=5, max_leaf_nodes=31,
               l2_regularization=0.0, min_samples_leaf=20):
    w = r ** 2
    ok = w > 1e-12
    z = np.clip(R[ok] / r[ok], -Z_CLIP, Z_CLIP)
    return HGB(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
               l2_regularization=l2_regularization, min_samples_leaf=min_samples_leaf,
               random_state=seed).fit(F[ok], z, sample_weight=w[ok])


class InstanceGate:
    """Per-instance gate: bagged weighted least squares on z = R / r with weights r^2 (RE-FUSED-6 D8)."""

    def __init__(self, n_bag=3, seed=0, **hgb):
        self.n_bag, self.seed, self.hgb = n_bag, seed, hgb

    def fit(self, F, R, r):
        rng = np.random.default_rng(self.seed)
        n = len(F)
        self.models = []
        for b in range(self.n_bag):
            idx = rng.integers(0, n, n) if self.n_bag > 1 else np.arange(n)
            self.models.append(_wls_model(F[idx], R[idx], r[idx], self.seed * 13 + b, **self.hgb))
        return self

    def predict(self, F):
        return np.mean([m.predict(F) for m in self.models], axis=0)


class NeuralGate:
    """Bounded neural sigmoid gate g(x) = 1.5 * sigmoid(MLP(x)) (RE-FUSED-6 M9/M9b). Hyperparameters fixed a priori
    from the configuration most often selected in M9b (hidden 128, weight decay 1e-3, lr 3e-3). Early stopping on the
    last 25% of fit days, then refit on all fit rows for the selected number of epochs."""

    def __init__(self, seed=0, hidden=128, weight_decay=1e-3, lr=3e-3, max_epochs=200, patience=10):
        self.seed, self.hidden, self.wd, self.lr = seed, hidden, weight_decay, lr
        self.max_epochs, self.patience = max_epochs, patience

    def _train(self, X, R, r, epochs=None, Xv=None, Rv=None, rv=None):
        import torch
        import torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)
        x, R_, r_ = T((X - self.mu) / self.sd), T(R / self.s), T(r / self.s)
        net = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_ep, bad, state = np.inf, 1, 0, None
        if Xv is not None:
            xv, Rv_, rv_ = T((Xv - self.mu) / self.sd), T(Rv / self.s), T(rv / self.s)
        for ep in range(epochs or self.max_epochs):
            net.train()
            perm = torch.randperm(len(x), device=dev)
            for i in range(0, len(x), 2048):
                idx = perm[i:i + 2048]
                loss = ((R_[idx] - CLIP[1] * torch.sigmoid(net(x[idx]).squeeze(-1)) * r_[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if Xv is None:
                continue
            net.eval()
            with torch.no_grad():
                vl = float(((Rv_ - CLIP[1] * torch.sigmoid(net(xv).squeeze(-1)) * rv_) ** 2).mean())
            if vl < best - 1e-9:
                best, best_ep, bad = vl, ep + 1, 0
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if state is not None:
            net.load_state_dict(state)
        return net, best_ep

    def fit(self, F, R, r, days):
        F = np.asarray(F, np.float32)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-8
        self.s = float(np.std(R)) + 1e-8
        d = np.asarray(days)
        cut = np.sort(np.unique(d))[int(0.75 * len(np.unique(d)))]
        a = d < cut
        _, ep = self._train(F[a], R[a], r[a], Xv=F[~a], Rv=R[~a], rv=r[~a])
        self.net, _ = self._train(F, R, r, epochs=max(ep, 1))
        self.epochs = ep
        return self

    def predict(self, F):
        import torch
        dev = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor((np.asarray(F, np.float32) - self.mu) / self.sd, device=dev)
            return (CLIP[1] * torch.sigmoid(self.net(x).squeeze(-1))).cpu().numpy()


def ewls_gate(R, r, age_days, lam):
    """Exponential-forgetting LS gate; weight lam ** age (age in days before the end of the fit window)."""
    return ls_gate(R, r, w=np.power(lam, np.asarray(age_days, float)))


def fixed_share_weights(loss_base, loss_corr, eta, alpha):
    """Online Fixed Share (Herbster & Warmuth 1998) over two experts.
    Returns the weight on the corrected forecast used at each step, computed before that step's loss."""
    w = np.array([0.5, 0.5])
    out = np.empty(len(loss_base))
    for t in range(len(loss_base)):
        out[t] = w[1]
        v = w * np.exp(-eta * np.array([loss_base[t], loss_corr[t]]))
        s = v.sum()
        v = v / s if s > 0 else np.array([0.5, 0.5])
        w = (1 - alpha) * v + alpha / 2
    return out
