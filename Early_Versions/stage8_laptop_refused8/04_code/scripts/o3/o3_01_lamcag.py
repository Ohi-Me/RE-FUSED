"""O3 step 1: LA-MCAG — latency-aware multi-context adaptive gating (protocol `02_design/05_o3_dev_protocol.md` v1.1).

Variants (argument 2): mcag_fixed, mcag_regime, mcag_instance, fusion_fixed; ablations (instance gates):
null_context, permuted_context, market_only, no_carbon.
Each run writes 06_results/dev/o3/preds/{tid}__{variant}.parquet (seed-averaged; q05…q95 final quantiles and b05…b95
base-head quantiles without context, used by the online time-adaptive gate) and _seeds.parquet, the learned gates
on evaluation windows (…_gates.parquet), a log, and for mcag_instance / fusion_fixed the L1 source-loss predictions
(…__{variant}_L1.parquet: R and W blocks missing on all development-test windows).
Usage: py -3.10 o3_01_lamcag.py T1,T2 mcag_fixed,mcag_instance [n_seeds]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "o2"))
from refused8 import o2data as O  # noqa: E402
from refused8.paths import RES  # noqa: E402
from refused8.windows import context_window, target_window  # noqa: E402
from o2_03_nf import build_df  # noqa: E402

# round 2 writes its O3 variants to results/dev2/o3 (set by the dev2 job); round 1 keeps results/<phase>/o3
OUT = os.environ.get("REFUSED8_O3_OUT") or os.path.join(RES, O.PHASE_DIR, "o3")
MAX_EPOCHS = int(os.environ.get("REFUSED8_O3_MAX_EPOCHS", "60"))
KEYS = ["tid", "sid", "issue_date", "H", "target_date", "split", "season", "y"]
QN = [f"q{int(round(q * 100)):02d}" for q in O.QUANTILES]
HQ = len(O.HORIZONS) * len(O.QUANTILES)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STALE_DROP = np.log1p(30.0)

BLOCKS = {  # block: (columns, key mask for staleness)
    "market": (["prc_dam_acp", "prc_dsm_normal", "prc_rtm_acp", "m_prc", "m_dsm", "m_prc_rtm"], ["m_prc", "m_dsm"]),
    "supply_npp": (["gen_act_gwh_total", "gen_prog_gwh_total", "gen_act_gwh_hydro", "m_npp"], ["m_npp"]),
    "supply_re": (["re_all_total_re_gwh", "re_all_wind_gwh", "re_all_solar_gwh", "m_re"], ["m_re"]),
    "system": (["dem_energy_met_gwh", "dem_max_demand_mw", "dem_energy_shortage_gwh", "dev_drawal_schedule_gwh",
                "dev_od_ud_gwh", "dev_actual_drawal_gwh", "reg_energy_met_gwh", "nat_energy_met_gwh", "reg_wind_gwh",
                "reg_solar_gwh", "reg_hydro_gwh", "sys_pct_lt_49_9", "sys_fvi", "gen_outage_share",
                "gen_coal_days_capw", "m_psp"], ["m_psp"]),
    "environment": (["wx_tmax_c", "wx_tmin_c", "wx_rain_mm", "wx_cdd24", "m_wx"], ["m_wx"]),
    "carbon": (["co2_ci_conv_op"], ["m_npp"]),
}
MASK_COLS = {"m_psp", "m_npp", "m_re", "m_prc", "m_wx", "m_prc_rtm", "m_dsm"}


def days_since_observed(mask):
    out = np.empty(len(mask))
    c = 30.0
    for i, m in enumerate(mask):
        c = 0.0 if m < 0.5 else min(c + 1.0, 365.0)
        out[i] = c
    return out


def make_windows(tid, variant):
    P = O.load_panel()
    df, hist, Ly = build_df(P, tid)
    blocks = {}
    for b, (cols, keys) in BLOCKS.items():
        c = [x for x in cols if x in hist]
        k = [x for x in keys if x in hist]
        if [x for x in c if x not in MASK_COLS] and k:
            blocks[b] = (c, k[0])
    if variant == "market_only":
        blocks = {b: v for b, v in blocks.items() if b == "market"}
    if variant == "no_carbon":
        blocks = {b: v for b, v in blocks.items() if b != "carbon"}
    names = list(blocks)
    cal = ["f_dow_sin", "f_dow_cos", "f_doy_sin", "f_doy_cos"]
    XT, XB, ST, T, M, C, SID, TAU, MU, SD, VOL = [], {b: [] for b in names}, [], [], [], [], [], [], [], [], []
    sids = sorted(df.unique_id.unique())
    raw_mask_idx = {b: [j for j, col in enumerate(blocks[b][0]) if col in MASK_COLS] for b in names}
    for k, sid in enumerate(sids):
        g = df[df.unique_id == sid].sort_values("ds").reset_index(drop=True)
        tr = g[g.ds <= O.TRAIN_END]
        avail = g.available_mask.to_numpy()
        y = g.y.to_numpy()
        mu, sd = tr.y[tr.available_mask > 0].mean(), (tr.y[tr.available_mask > 0].std() or 1.0)
        feats = {}
        for b in names:
            cols, key = blocks[b]
            arr = []
            for col in cols:
                v = g[col].to_numpy(dtype=float)
                if col in MASK_COLS:
                    arr.append(v)
                else:
                    m_, s_ = tr[col].mean(), tr[col].std()
                    arr.append(np.nan_to_num((v - m_) / (s_ if s_ and np.isfinite(s_) else 1.0)))
            feats[b] = np.column_stack(arr).astype(np.float32)
        stale = np.column_stack([np.log1p(days_since_observed(g[blocks[b][1]].to_numpy())) for b in names]).astype(np.float32)
        # ex-ante price volatility state: trailing 28-day CV of the lag-aligned DAM price vs the series' train median
        pc = "prc_dam_acp" if "prc_dam_acp" in g else ("prc_rtm_acp" if "prc_rtm_acp" in g else None)
        if tid == "T5":
            pser = pd.Series(np.where(avail > 0, y, np.nan))
        else:
            pser = pd.Series(g[pc].to_numpy(dtype=float)) if pc else pd.Series(np.nan, index=g.index)
        cv = (pser.rolling(28, min_periods=10).std() / pser.rolling(28, min_periods=10).mean()).to_numpy()
        thr = np.nanmedian(cv[(g.ds <= O.TRAIN_END).to_numpy()]) if np.isfinite(cv).any() else 0.0
        vol = (np.nan_to_num(cv, nan=thr) > thr).astype(int)
        calv = g[cal].to_numpy(dtype=np.float32)
        season_idx = g.ds.dt.month.map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3}).to_numpy()
        for i in range(O.WINDOW - 1, len(g) - (Ly + 3)):
            sl = slice(i - O.WINDOW + 1, i + 1)
            yw, base = target_window(y, avail, sl, sd, mu)
            XT.append(np.column_stack([yw, avail[sl]]))
            for b in names:
                XB[b].append(context_window(feats[b][sl], raw_mask_idx[b]))
            ST.append(stale[i])
            idx = [i + Ly + H for H in O.HORIZONS]
            T.append((y[idx] - base) / sd)
            M.append(avail[idx])
            C.append(calv[idx].ravel())
            SID.append(k)
            TAU.append(g.ds.iloc[i] + pd.Timedelta(days=Ly))
            MU.append(base)
            SD.append(sd)
            VOL.append(season_idx[i + Ly + 1] * 2 + vol[i])
    W = dict(XT=np.stack(XT).astype(np.float32), XB={b: np.stack(XB[b]) for b in names}, ST=np.stack(ST),
             T=np.array(T, np.float32), M=np.array(M, np.float32), C=np.array(C, np.float32), SID=np.array(SID),
             TAU=pd.to_datetime(TAU), MU=np.array(MU), SD=np.array(SD), REG=np.array(VOL))
    n_num = {b: len(blocks[b][0]) - len(raw_mask_idx[b]) for b in names}
    mask_idx = {b: list(range(2 * n_num[b], 2 * n_num[b] + len(raw_mask_idx[b]))) for b in names}
    return W, sids, Ly, names, mask_idx


class LAMCAG(nn.Module):
    def __init__(self, block_dims, n_series, n_cal, gate, n_regimes=8, fusion_hidden=None):
        super().__init__()
        self.gate, self.K = gate, len(block_dims)
        self.tenc = nn.LSTM(2, 64, num_layers=2, dropout=0.2, bidirectional=True, batch_first=True)
        self.emb = nn.Embedding(n_series, 8)
        self.zmlp = nn.Sequential(nn.Linear(256 + 8 + n_cal, 128), nn.ReLU(), nn.Dropout(0.2))
        self.base = nn.Linear(128, HQ)
        self.cenc = nn.ModuleList([nn.GRU(d, 32, batch_first=True) for d in block_dims])
        if gate == "fusion":
            self.fuse = nn.Sequential(nn.Linear(128 + 33 * self.K, fusion_hidden), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(fusion_hidden, HQ))
        else:
            self.corr = nn.ModuleList([nn.Sequential(nn.Linear(128 + 33, 64), nn.ReLU(), nn.Linear(64, HQ))
                                       for _ in block_dims])
            if gate == "fixed":
                self.theta = nn.Parameter(torch.zeros(self.K))
            elif gate == "regime":
                self.theta = nn.Parameter(torch.zeros(self.K, n_regimes))
            else:
                self.gmlp = nn.ModuleList([nn.Sequential(nn.Linear(128 + 33, 16), nn.ReLU(), nn.Linear(16, 1))
                                           for _ in block_dims])

    def forward(self, xt, xb, stale, sid, cal, reg):
        h, _ = self.tenc(xt)
        z = self.zmlp(torch.cat([h[:, -1], h.mean(1), self.emb(sid), cal], 1))
        q = self.base(z)
        cs = [torch.cat([self.cenc[k](xb[k])[1][-1], stale[:, k:k + 1]], 1) for k in range(self.K)]
        gates = None
        if self.gate == "fusion":
            corr_total = self.fuse(torch.cat([z] + cs, 1))
            q = q + corr_total
        else:
            gl, corr_total = [], 0
            for k in range(self.K):
                r = self.corr[k](torch.cat([z, cs[k]], 1))
                if self.gate == "fixed":
                    g = 1.5 * torch.sigmoid(self.theta[k]).expand(len(z), 1)
                elif self.gate == "regime":
                    g = 1.5 * torch.sigmoid(self.theta[k, reg]).unsqueeze(1)
                else:
                    g = 1.5 * torch.sigmoid(self.gmlp[k](torch.cat([z, cs[k]], 1)))
                gl.append(g)
                corr_total = corr_total + g * r
            q = q + corr_total
            gates = torch.cat(gl, 1)
        return q.view(-1, len(O.HORIZONS), len(O.QUANTILES)), gates, self.base(z).view(-1, len(O.HORIZONS), len(O.QUANTILES))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fusion_width(block_dims, n_series, n_cal):
    target = n_params(LAMCAG(block_dims, n_series, n_cal, "instance"))
    best = min(range(16, 4096, 4), key=lambda w: abs(n_params(LAMCAG(block_dims, n_series, n_cal, "fusion", fusion_hidden=w)) - target))
    return best, target, n_params(LAMCAG(block_dims, n_series, n_cal, "fusion", fusion_hidden=best))


def pinball_loss(q, t, m):
    tau = torch.tensor(O.QUANTILES, device=q.device).view(1, 1, -1)
    u = t.unsqueeze(-1) - q
    return (torch.maximum(tau * u, (tau - 1) * u).mean(-1) * m).sum() / m.sum().clamp(min=1)


def batch(W, names, mask_idx, idx, drop=None, rng=None, force_missing=(), stale_override=None, null=False, perm=None,
          stale_random=False):
    src = idx if perm is None else perm[idx]
    xt = torch.tensor(W["XT"][idx], device=DEV)
    xb = [torch.tensor(W["XB"][b][src], device=DEV) for b in names]
    st = torch.tensor(W["ST"][src], device=DEV)
    for k, b in enumerate(names):
        if null or b in force_missing:
            miss = torch.ones(len(idx), dtype=torch.bool, device=DEV)
        elif drop:
            miss = torch.tensor(rng.random(len(idx)) < drop, device=DEV)
        else:
            continue
        if miss.any():
            xb[k][miss] = 0.0
            if mask_idx[b]:
                xb[k][miss.nonzero().squeeze(1)[:, None], :, torch.tensor(mask_idx[b], device=DEV)[None, :]] = 1.0
            if stale_override is not None and b in force_missing:
                st[miss, k] = float(stale_override[k])
            elif stale_random and drop:
                # dev2 (_sd variants): a dropped source is between 1 day and a year old, so the model also learns
                # what long outages look like (log-uniform, like the staleness scale itself)
                days = np.exp(rng.uniform(0.0, np.log(365.0), int(miss.sum())))
                st[miss, k] = torch.tensor(np.log1p(days), device=DEV, dtype=torch.float32)
            else:
                st[miss, k] = STALE_DROP
    return (xt, xb, st, torch.tensor(W["SID"][idx], device=DEV), torch.tensor(W["C"][idx], device=DEV),
            torch.tensor(W["REG"][idx], device=DEV))


def run(tid, variant, seeds):
    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    fpath = os.path.join(OUT, "preds", f"{tid}__{variant}_seeds.parquet")
    if os.path.exists(fpath) and pd.read_parquet(fpath, columns=["seed"]).seed.nunique() >= len(seeds):
        print(tid, variant, "already complete, skipped", flush=True)
        return
    t0 = time.time()
    stale_random = variant.endswith("_sd")
    base_variant = variant[:-3] if stale_random else variant
    W, sids, Ly, names, mask_idx = make_windows(tid, base_variant)
    gate = {"mcag_fixed": "fixed", "mcag_regime": "regime", "fusion_fixed": "fusion"}.get(base_variant, "instance")
    dims = [W["XB"][b].shape[2] for b in names]
    fw, p_inst, p_fus = fusion_width(dims, len(sids), W["C"].shape[1])
    first_t, last_t = W["TAU"] + pd.Timedelta(days=1), W["TAU"] + pd.Timedelta(days=3)
    tr = np.where(last_t < O.EARLY_STOP_START)[0]
    es = np.where((first_t >= O.EARLY_STOP_START) & (last_t <= O.TRAIN_END))[0]
    ev = np.where((W["TAU"] >= O.TRAIN_END) & (first_t <= O.DEV_END))[0]
    perm = None
    if variant == "permuted_context":
        rng0 = np.random.default_rng(12345)
        perm = np.arange(len(W["TAU"]))
        for s in np.unique(W["SID"]):
            for part in (tr, es, ev):
                ids = part[W["SID"][part] == s]
                perm[ids] = rng0.permutation(ids)
    null = variant == "null_context"
    S = pd.read_parquet(os.path.join(RES, O.PHASE_DIR, "o2", "samples", f"{tid}.parquet"), columns=KEYS)
    S = S[S.split.isin(["validation", "dev_test"])]
    log = dict(tid=tid, variant=variant, blocks=names, block_dims=dims, params_instance=p_inst,
               params=(p_fus if gate == "fusion" else n_params(LAMCAG(dims, len(sids), W["C"].shape[1], gate))),
               fusion_hidden=fw, n_train=len(tr), n_es=len(es), n_eval=len(ev), seeds={})
    frames, gates_out, l1_frames = [], [], []
    dev_start = np.datetime64(O.VAL_END)
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = LAMCAG(dims, len(sids), W["C"].shape[1], gate, fusion_hidden=fw).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_state, bad, epochs = np.inf, None, 0, 0
        for epoch in range(MAX_EPOCHS):
            model.train()
            order = rng.permutation(tr)
            for b0 in range(0, len(order), 512):
                idx = order[b0:b0 + 512]
                xt, xb, st, sid, cal, reg = batch(W, names, mask_idx, idx, drop=0.15, rng=rng, null=null, perm=perm,
                                                  stale_random=stale_random)
                q, _, _ = model(xt, xb, st, sid, cal, reg)
                loss = pinball_loss(q, torch.tensor(W["T"][idx], device=DEV), torch.tensor(W["M"][idx], device=DEV))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vl, nn_ = 0.0, 0
                for b0 in range(0, len(es), 4096):
                    idx = es[b0:b0 + 4096]
                    q, _, _ = model(*batch(W, names, mask_idx, idx, null=null, perm=perm))
                    vl += float(pinball_loss(q, torch.tensor(W["T"][idx], device=DEV),
                                             torch.tensor(W["M"][idx], device=DEV))) * len(idx)
                    nn_ += len(idx)
            vl /= max(nn_, 1)
            epochs = epoch + 1
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 6:
                    break
        model.load_state_dict(best_state)
        model.eval()

        def predict(force=(), stale_days=None):
            Qs, Gs, Bs = [], [], []
            with torch.no_grad():
                for b0 in range(0, len(ev), 4096):
                    idx = ev[b0:b0 + 4096]
                    so = None
                    if stale_days is not None:
                        so = stale_days[b0:b0 + 4096]
                    args = batch(W, names, mask_idx, idx, null=null, perm=perm, force_missing=force)
                    if so is not None:
                        for k, b in enumerate(names):
                            if b in force:
                                args[2][:, k] = torch.tensor(so, device=DEV, dtype=torch.float32)
                    q, g, bq = model(*args)
                    Qs.append(q.cpu().numpy())
                    Bs.append(bq.cpu().numpy())
                    if g is not None:
                        Gs.append(g.cpu().numpy())
            return np.sort(np.concatenate(Qs), axis=2), (np.concatenate(Gs) if Gs else None), np.concatenate(Bs)

        def to_frame(Qn, Bn=None):
            Q = W["MU"][ev][:, None, None] + W["SD"][ev][:, None, None] * Qn
            B = None if Bn is None else W["MU"][ev][:, None, None] + W["SD"][ev][:, None, None] * np.sort(Bn, axis=2)
            rows = []
            for hi, H in enumerate(O.HORIZONS):
                r = pd.DataFrame(Q[:, hi, :], columns=QN)
                if B is not None:
                    for j, c in enumerate(QN):
                        r["b" + c[1:]] = B[:, hi, j]
                r["sid"] = [sids[i] for i in W["SID"][ev]]
                r["issue_date"] = W["TAU"][ev]
                r["H"] = H
                rows.append(r)
            return S.merge(pd.concat(rows), on=["sid", "issue_date", "H"], how="inner")

        Qn, G, Bn = predict()
        fr = to_frame(Qn, Bn)
        fr["model"], fr["seed"] = variant, seed
        frames.append(fr)
        base_med = W["MU"][ev] + W["SD"][ev] * Bn[:, :, 3].mean(1)
        gdf = pd.DataFrame({"sid": [sids[i] for i in W["SID"][ev]], "issue_date": W["TAU"][ev], "seed": seed,
                            "regime_cell": W["REG"][ev], "base_q50_h1": W["MU"][ev] + W["SD"][ev] * Bn[:, 0, 3],
                            "corr_q50_h1_norm": Qn[:, 0, 3] - np.sort(Bn, axis=2)[:, 0, 3],
                            **{f"stale_{b}": W["ST"][ev][:, k] for k, b in enumerate(names)}})
        if G is not None:
            for k, b in enumerate(names):
                gdf[f"gate_{b}"] = G[:, k]
        gates_out.append(gdf)
        if base_variant in ("mcag_instance", "fusion_fixed") and tid in ("T1", "T2"):
            force = tuple(b for b in ("supply_re", "environment") if b in names)
            days = (W["TAU"][ev].to_numpy().astype("datetime64[D]") - dev_start.astype("datetime64[D]")).astype(float)
            stale_days = np.log1p(np.clip(days, 1, 365)).astype(np.float32)
            Ql, _, Bl = predict(force=force, stale_days=stale_days)
            fl = to_frame(Ql, Bl)
            fl["model"], fl["seed"] = f"{variant}_L1", seed
            l1_frames.append(fl)
        log["seeds"][seed] = dict(epochs=epochs, best_es=best, elapsed_s=round(time.time() - t0, 1))
        print(f"  {tid} {variant} seed {seed}: epochs {epochs} es {best:.4f} [{time.time() - t0:.0f}s]", flush=True)

    def save(frs, name):
        sd = pd.concat(frs, ignore_index=True)
        sd.to_parquet(os.path.join(OUT, "preds", f"{tid}__{name}_seeds.parquet"), index=False)
        BQ = [c for c in sd.columns if c.startswith("b") and c[1:].isdigit()]
        avg = sd.groupby(KEYS, dropna=False)[QN + BQ].mean().reset_index()
        avg[QN] = np.sort(avg[QN].to_numpy(), axis=1)
        if BQ:
            avg[BQ] = np.sort(avg[BQ].to_numpy(), axis=1)
        avg["model"] = name
        avg.to_parquet(os.path.join(OUT, "preds", f"{tid}__{name}.parquet"), index=False)

    if l1_frames:
        save(l1_frames, f"{variant}_L1")
    pd.concat(gates_out).to_parquet(os.path.join(OUT, "preds", f"{tid}__{variant}_gates.parquet"), index=False)
    save(frames, variant)
    json.dump(log, open(os.path.join(OUT, "logs", f"{tid}__{variant}.json"), "w"), indent=1, default=str)
    print(tid, variant, "done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    tids = sys.argv[1].split(",")
    variants = sys.argv[2].split(",")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    for v in variants:
        for tid in tids:
            run(tid, v, range(n))
