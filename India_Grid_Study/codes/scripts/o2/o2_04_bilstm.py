"""O2 step 4: BiLSTM quantile model (protocol §3; the proposal's recurrent baseline).

Inputs per window (cutoff c = last published target date, issue day τ = c + Ly): 14 days of the lag-aligned frame
built for TFT (`o2_03_nf.build_df`): target (centred on the mean of its last 7 available values in the window and scaled by the series' train std,
`refused.windows`), target availability mask,
historical exogenous columns (normalised per series on train) and source-group missingness masks; plus the target
days' calendar (day-of-week and day-of-year, sine/cosine) and a learned series embedding.
Architecture: 2-layer bidirectional LSTM (hidden 64, dropout 0.2) → [last state, mean state, embedding (8), calendar]
→ MLP (128) → 3 horizons × 7 quantiles. Loss: pinball, masked where the target is missing.
Training windows: all target days before the last 180 train days; early stopping (patience 6, ≤ 60 epochs) on
windows whose target days fall in those 180 days. Evaluation windows: validation and development-test issue days.
Seeds 0–4, averaged quantile-wise.

Outputs: results/dev/o2/preds/{tid}__bilstm.parquet and _seeds.parquet; logs/bilstm_{tid}.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from refused import o2data as O  # noqa: E402
from refused.paths import RES  # noqa: E402
from refused.windows import target_window  # noqa: E402
from o2_03_nf import build_df  # noqa: E402

OUT = os.path.join(RES, O.PHASE_DIR, "o2")
KEYS = ["tid", "sid", "issue_date", "H", "target_date", "split", "season", "y"]
QN = [f"q{int(round(q * 100)):02d}" for q in O.QUANTILES]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows(tid):
    P = O.load_panel()
    df, hist, Ly = build_df(P, tid)
    cal = ["f_dow_sin", "f_dow_cos", "f_doy_sin", "f_doy_cos"]
    X, T, M, C, SID, TAU, MU, SD = [], [], [], [], [], [], [], []
    sids = sorted(df.unique_id.unique())
    for k, sid in enumerate(sids):
        g = df[df.unique_id == sid].sort_values("ds").reset_index(drop=True)
        tr = g[g.ds <= O.TRAIN_END]
        avail = g.available_mask.to_numpy()
        y = g.y.to_numpy()
        mu = tr.y[tr.available_mask > 0].mean()
        sd = tr.y[tr.available_mask > 0].std() or 1.0
        ex = g[hist].to_numpy(dtype=float)
        emu, esd = tr[hist].mean().to_numpy(), tr[hist].std().replace(0, 1).fillna(1).to_numpy()
        exn = np.nan_to_num((ex - emu) / esd)
        feats = np.column_stack([np.zeros(len(g)), avail, exn]).astype(np.float32)
        calv = g[cal].to_numpy(dtype=np.float32)
        n = len(g)
        for i in range(O.WINDOW - 1, n - (Ly + 3)):
            sl = slice(i - O.WINDOW + 1, i + 1)
            yw, base = target_window(y, avail, sl, sd, mu)
            x = feats[sl].copy()
            x[:, 0] = yw
            X.append(x)
            idx = [i + Ly + H for H in O.HORIZONS]
            T.append((y[idx] - base) / sd)
            M.append(avail[idx])
            C.append(calv[idx].ravel())
            SID.append(k)
            TAU.append(g.ds.iloc[i] + pd.Timedelta(days=Ly))
            MU.append(base)
            SD.append(sd)
    W = dict(X=np.stack(X), T=np.array(T, np.float32), M=np.array(M, np.float32), C=np.array(C, np.float32),
             SID=np.array(SID), TAU=pd.to_datetime(TAU), MU=np.array(MU), SD=np.array(SD))
    return W, sids, Ly


class BiLSTMQ(nn.Module):
    def __init__(self, n_in, n_series, n_cal, hidden=64, emb=8):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=2, dropout=0.2, bidirectional=True, batch_first=True)
        self.emb = nn.Embedding(n_series, emb)
        self.head = nn.Sequential(nn.Linear(4 * hidden + emb + n_cal, 128), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(128, len(O.HORIZONS) * len(O.QUANTILES)))

    def forward(self, x, sid, cal):
        h, _ = self.lstm(x)
        z = torch.cat([h[:, -1], h.mean(1), self.emb(sid), cal], 1)
        return self.head(z).view(-1, len(O.HORIZONS), len(O.QUANTILES))


def pinball_loss(q, t, m):
    tau = torch.tensor(O.QUANTILES, device=q.device).view(1, 1, -1)
    u = t.unsqueeze(-1) - q
    loss = torch.maximum(tau * u, (tau - 1) * u).mean(-1)
    return (loss * m).sum() / m.sum().clamp(min=1)


def run(tid, seeds=range(5)):
    if O.is_complete(tid, "bilstm", len(seeds)):
        print("bilstm", tid, "already complete, skipped", flush=True)
        return
    t0 = time.time()
    W, sids, Ly = make_windows(tid)
    last_target = W["TAU"] + pd.Timedelta(days=3)
    first_target = W["TAU"] + pd.Timedelta(days=1)
    tr = np.where(last_target < O.EARLY_STOP_START)[0]
    es = np.where((first_target >= O.EARLY_STOP_START) & (last_target <= O.TRAIN_END))[0]
    ev = np.where((W["TAU"] >= O.TRAIN_END) & (first_target <= O.DEV_END))[0]
    to = lambda a, idx: torch.tensor(a[idx], device=DEV)  # noqa: E731
    S = pd.read_parquet(os.path.join(OUT, "samples", f"{tid}.parquet"), columns=KEYS)
    S = S[S.split.isin(["validation", "dev_test"])]
    log = {"tid": tid, "n_train": len(tr), "n_es": len(es), "n_eval": len(ev), "seeds": {}}
    frames = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = BiLSTMQ(W["X"].shape[2], len(sids), W["C"].shape[1]).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xes, Tes, Mes, Ses, Ces = (to(W[k], es) for k in ("X", "T", "M", "SID", "C"))
        best, best_state, bad, epochs = np.inf, None, 0, 0
        rng = np.random.default_rng(seed)
        for epoch in range(60):
            model.train()
            perm = rng.permutation(tr)
            for b in range(0, len(perm), 256):
                idx = perm[b:b + 256]
                opt.zero_grad()
                loss = pinball_loss(model(to(W["X"], idx), to(W["SID"], idx), to(W["C"], idx)),
                                    to(W["T"], idx), to(W["M"], idx))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(pinball_loss(model(Xes, Ses, Ces), Tes, Mes))
            epochs = epoch + 1
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 6:
                    break
        model.load_state_dict(best_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for b in range(0, len(ev), 4096):
                idx = ev[b:b + 4096]
                preds.append(model(to(W["X"], idx), to(W["SID"], idx), to(W["C"], idx)).cpu().numpy())
        Qn = np.sort(np.concatenate(preds), axis=2)
        Q = W["MU"][ev][:, None, None] + W["SD"][ev][:, None, None] * Qn
        rows = []
        for hi, H in enumerate(O.HORIZONS):
            r = pd.DataFrame(Q[:, hi, :], columns=QN)
            r["sid"] = [sids[i] for i in W["SID"][ev]]
            r["issue_date"] = W["TAU"][ev]
            r["H"] = H
            rows.append(r)
        fr = S.merge(pd.concat(rows), on=["sid", "issue_date", "H"], how="inner")
        fr["model"], fr["seed"] = "bilstm", seed
        frames.append(fr)
        log["seeds"][seed] = dict(epochs=epochs, best_es_pinball_z=best, elapsed_s=round(time.time() - t0, 1))
        print(f"  bilstm {tid} seed {seed}: epochs {epochs}, es pinball {best:.4f} [{time.time() - t0:.0f}s]", flush=True)
    sd = pd.concat(frames, ignore_index=True)
    sd.to_parquet(os.path.join(OUT, "preds", f"{tid}__bilstm_seeds.parquet"), index=False)
    avg = sd.groupby(KEYS, dropna=False)[QN].mean().reset_index()
    avg[QN] = np.sort(avg[QN].to_numpy(), axis=1)
    avg["model"] = "bilstm"
    avg.to_parquet(os.path.join(OUT, "preds", f"{tid}__bilstm.parquet"), index=False)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    json.dump(log, open(os.path.join(OUT, "logs", f"bilstm_{tid}.json"), "w"), indent=1)
    print("bilstm", tid, "done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for tid in (sys.argv[1].split(",") if len(sys.argv) > 1 else list(O.TARGETS)):
        run(tid, range(int(sys.argv[2])) if len(sys.argv) > 2 else range(5))
