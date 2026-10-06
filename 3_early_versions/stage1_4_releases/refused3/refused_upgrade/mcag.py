"""
mcag.py — MCAG-v2: carbon-aware gating rebuilt for the upgrade harness.

Relationship to the original notebook
-------------------------------------
The classes `MCAGGate` and `MCAGRegimeGate` below are faithful re-implementations
of Cell 3.2's `MCAG` and `MCAG_Regime`. They are COPIES, written here so the new
variants can be built without importing, patching or re-running any original
cell. The published `PatchTST+MCAG` result stands untouched.

`PatchTSTMCAG_Replica` reproduces the original architecture (patch → transformer
encoder → carbon gate → linear head) so the original modelling choice can be
scored on the upgrade's evaluation grid. `MCAGv2` is the new model: the same
carbon-gating idea, but wrapped in the two corrections Section U3.2 showed are
decisive on this panel (instance normalisation and a target-lag input), plus the
six requested extensions.

The six MCAG-v2 extensions
--------------------------
  A. state embedding            — a learned vector per state, added to every patch
  B. regime embedding           — the six train-only regime flags, projected and added
  C. cross-state lagged input   — the national mean at the origin as an extra channel
  D. improved carbon/price fusion — separate carbon and price gates combined by a
                                    learned mixing weight, instead of one gate over
                                    a concatenated block
  E. multi-task prediction      — auxiliary heads on the other targets, so the shared
                                  encoder is regularised by related signals
  F. best practical combination — whichever subset validation selects

Every input is drawn from the forecast origin or earlier, or is a deterministic
calendar attribute; the U1.7 perturbation proof covers the same series.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .panel import DATE, STATE, CalendarPanel

CARBON_EXOG: List[str] = ["carbon_intensity", "carbon_budget_pressure", "renewable_ratio"]
PRICE_EXOG: List[str] = ["avg_market_price", "supply_demand_gap"]


# ------------------------------------------------------------------- dataset
class MCAGWindows(Dataset):
    """Windows carrying everything the MCAG variants can consume.

    Returns per sample:
      x     (L, C)  target channel + carbon channels + price channels + calendar
      xs    (L, 1)  cross-state national mean at the origin (lagged)
      y     (H,)    normalised targets for the primary series
      yaux  (H, A)  normalised auxiliary targets (multi-task)
      sid   ()      state index
      reg   (R,)    regime flags at the forecast origin
      norm  (2,)    (offset, scale) used to invert the normalisation
      t     ()      origin index into `dates`
    """

    def __init__(self, cal: CalendarPanel, target: str, lookback: int,
                 horizons: Sequence[int], split: str,
                 carbon_exog: Sequence[str] = tuple(CARBON_EXOG),
                 price_exog: Sequence[str] = tuple(PRICE_EXOG),
                 regime_cols: Sequence[str] = (),
                 aux_targets: Sequence[str] = ()):
        self.lookback, self.horizons = lookback, tuple(horizons)
        self.max_h = max(horizons)
        self.dates, self.states = cal.dates, cal.states
        self.target = target

        self.Y = cal.wide(target).to_numpy(float)
        self.carbon = [cal.wide(c).to_numpy(float) for c in carbon_exog if c in cal.long.columns]
        self.price = [cal.wide(c).to_numpy(float) for c in price_exog if c in cal.long.columns]
        self.aux = [cal.wide(c).to_numpy(float) for c in aux_targets]
        self.n_carbon, self.n_price = len(self.carbon), len(self.price)

        if regime_cols:
            self.reg = np.stack([cal.wide(c).to_numpy(float) for c in regime_cols], axis=-1)
        else:
            self.reg = None
        self.n_regime = len(regime_cols)

        from .deep import window_index, gather_windows

        S = cal.split_of().to_numpy(dtype=object)
        L, H = lookback, self.max_h
        self.index = window_index(self.Y, S, L, H, split)

        d = pd.DatetimeIndex(self.dates)
        cal_feats = np.stack([
            np.sin(2 * np.pi * d.dayofweek.to_numpy() / 7.0),
            np.cos(2 * np.pi * d.dayofweek.to_numpy() / 7.0),
            np.sin(2 * np.pi * d.month.to_numpy() / 12.0),
            np.cos(2 * np.pi * d.month.to_numpy() / 12.0),
            np.sin(2 * np.pi * d.dayofyear.to_numpy() / 365.25),
            np.cos(2 * np.pi * d.dayofyear.to_numpy() / 365.25),
        ], axis=1).astype("f4")
        self._n_cal = cal_feats.shape[1]

        # national mean over states, computed per date from observed values only
        with np.errstate(invalid="ignore"):
            national = np.nanmean(self.Y, axis=1)

        if not self.index:
            self.x = torch.zeros(0, L, self.n_channels)
            self.xs = torch.zeros(0, L, 1)
            self.y = torch.zeros(0, H)
            self.yaux = torch.zeros(0, H, len(self.aux))
            self.sid = torch.zeros(0, dtype=torch.long)
            self.regv = torch.zeros(0, self.n_regime)
            self.norm = torch.zeros(0, 2)
            self.t = torch.zeros(0, dtype=torch.long)
            return

        idx_s = np.array([s for s, _ in self.index])
        idx_t = np.array([t for _, t in self.index])

        def _z(M):
            w = gather_windows(M, idx_s, idx_t, L)
            mu = np.nanmean(w, axis=1, keepdims=True)
            sd = np.nanstd(w, axis=1, keepdims=True)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            return np.nan_to_num((w - mu) / sd)

        win = gather_windows(self.Y, idx_s, idx_t, L)
        offset = win[:, -1]
        scale = win.std(axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)

        chans = [((win - offset[:, None]) / scale[:, None])[:, :, None]]
        for E in self.carbon:
            chans.append(_z(E)[:, :, None])
        for E in self.price:
            chans.append(_z(E)[:, :, None])
        rows = idx_t[:, None] + np.arange(-L + 1, 1)[None, :]
        chans.append(cal_feats[rows])

        nat_w = national[rows]
        nmu = np.nanmean(nat_w, axis=1, keepdims=True)
        nsd = np.nanstd(nat_w, axis=1, keepdims=True)
        nsd = np.where(np.isfinite(nsd) & (nsd > 1e-8), nsd, 1.0)

        tgt = np.stack([self.Y[idx_t + k, idx_s] for k in range(1, H + 1)], axis=1)

        if self.aux:
            aux_list = []
            for A in self.aux:
                aw = gather_windows(A, idx_s, idx_t, L)
                ao = aw[:, -1]
                ao = np.where(np.isfinite(ao), ao, np.nanmean(aw, axis=1))
                asd = np.nanstd(aw, axis=1)
                asd = np.where(np.isfinite(asd) & (asd > 1e-8), asd, 1.0)
                at = np.stack([A[idx_t + k, idx_s] for k in range(1, H + 1)], axis=1)
                aux_list.append(np.nan_to_num((at - ao[:, None]) / asd[:, None]))
            yaux = np.stack(aux_list, axis=-1).astype("f4")
        else:
            yaux = np.zeros((len(idx_s), H, 0), dtype="f4")

        self.x = torch.from_numpy(np.concatenate(chans, axis=2).astype("f4"))
        self.xs = torch.from_numpy(np.nan_to_num((nat_w - nmu) / nsd)[:, :, None].astype("f4"))
        self.y = torch.from_numpy(((tgt - offset[:, None]) / scale[:, None]).astype("f4"))
        self.yaux = torch.from_numpy(yaux)
        self.sid = torch.from_numpy(idx_s.astype(np.int64))
        self.regv = torch.from_numpy(
            (self.reg[idx_t, idx_s] if self.reg is not None
             else np.zeros((len(idx_s), 0))).astype("f4"))
        self.norm = torch.from_numpy(np.stack([offset, scale], axis=1).astype("f4"))
        self.t = torch.from_numpy(idx_t.astype(np.int64))

    @property
    def n_channels(self) -> int:
        return 1 + self.n_carbon + self.n_price + getattr(self, "_n_cal", 6)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        return (self.x[i], self.xs[i], self.y[i], self.yaux[i], self.sid[i],
                self.regv[i], self.norm[i], self.t[i])


# ---------------------------------------------------------------- gate copies
class MCAGGate(nn.Module):
    """Faithful copy of the original Cell 3.2 `MCAG`: a sigmoid gate over the
    time-averaged carbon context, multiplied into the hidden representation."""

    def __init__(self, nc: int, d: int):
        super().__init__()
        self.fc = nn.Linear(nc, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, h, c):
        g = torch.sigmoid(self.ln(self.fc(c.mean(1))))
        return h * g.unsqueeze(1)


class MCAGRegimeGate(nn.Module):
    """Faithful copy of the original `MCAG_Regime`: soft routing over 3 gates."""

    def __init__(self, nc: int, d: int, n_regimes: int = 3):
        super().__init__()
        self.gates = nn.ModuleList([MCAGGate(nc, d) for _ in range(n_regimes)])
        self.router = nn.Linear(nc, n_regimes)

    def forward(self, h, c):
        alpha = torch.softmax(self.router(c.mean(1)), dim=-1)
        gated = torch.stack([g(h, c) for g in self.gates], dim=1)
        return (alpha.unsqueeze(-1).unsqueeze(-1) * gated).sum(1)


class CarbonPriceFusionGate(nn.Module):
    """Extension D — separate carbon and price gates, combined by a learned mix.

    The original gate concatenates carbon and price signals and learns one gate
    over the union, which forces a single multiplicative response to two drivers
    that act on different timescales. Here each driver gets its own gate and a
    context-dependent weight decides how much of each to apply.
    """

    def __init__(self, n_carbon: int, n_price: int, d: int):
        super().__init__()
        self.carbon_gate = MCAGGate(max(n_carbon, 1), d)
        self.price_gate = MCAGGate(max(n_price, 1), d)
        self.mix = nn.Sequential(nn.Linear(max(n_carbon + n_price, 1), d // 2),
                                 nn.ReLU(), nn.Linear(d // 2, 2))

    def forward(self, h, c_carbon, c_price):
        hc = self.carbon_gate(h, c_carbon)
        hp = self.price_gate(h, c_price)
        ctx = torch.cat([c_carbon.mean(1), c_price.mean(1)], dim=-1)
        w = torch.softmax(self.mix(ctx), dim=-1)
        return w[:, 0:1, None] * hc + w[:, 1:2, None] * hp


# ------------------------------------------------------------------- models
class PatchTSTBackbone(nn.Module):
    """Patch → linear projection → transformer encoder (as in the original)."""

    def __init__(self, lookback: int, n_feat: int, d: int = 64,
                 patch_len: int = 4, stride: int = 2, layers: int = 2, heads: int = 4):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patch = (lookback - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len * n_feat, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.d = d

    def forward(self, x):
        B = x.shape[0]
        patches = [x[:, i * self.stride:i * self.stride + self.patch_len, :].reshape(B, -1)
                   for i in range(self.n_patch)]
        return self.enc(self.proj(torch.stack(patches, 1)))


@dataclass
class MCAGv2Config:
    lookback: int = 28
    n_out: int = 3
    d: int = 64
    state_emb: bool = False
    regime_emb: bool = False
    cross_state: bool = False
    fusion: bool = False
    multitask: int = 0          # number of auxiliary targets
    n_states: int = 18
    n_regimes: int = 6


class MCAGv2(nn.Module):
    """PatchTST backbone + carbon gating, with the six requested extensions.

    Forecasts in instance-normalised space, so a zero output is the persistence
    forecast and every extension is measured against that floor.
    """

    def __init__(self, cfg: MCAGv2Config, n_channels: int, n_carbon: int, n_price: int):
        super().__init__()
        self.cfg = cfg
        self.n_carbon, self.n_price = n_carbon, n_price
        in_ch = n_channels + (1 if cfg.cross_state else 0)
        self.backbone = PatchTSTBackbone(cfg.lookback, in_ch, d=cfg.d)
        d = cfg.d

        if cfg.fusion:
            self.gate = CarbonPriceFusionGate(n_carbon, n_price, d)
        else:
            self.gate = MCAGGate(max(n_carbon + n_price, 1), d)

        if cfg.state_emb:
            self.state_emb = nn.Embedding(cfg.n_states, d)
        if cfg.regime_emb:
            self.regime_proj = nn.Sequential(nn.Linear(cfg.n_regimes, d), nn.ReLU(),
                                             nn.Linear(d, d))
        flat = d * self.backbone.n_patch
        self.head = nn.Linear(flat, cfg.n_out)
        if cfg.multitask:
            self.aux_head = nn.Linear(flat, cfg.n_out * cfg.multitask)

    def forward(self, x, xs, sid, reg):
        if self.cfg.cross_state:
            x = torch.cat([x, xs], dim=-1)
        h = self.backbone(x)                                   # (B, P, d)

        c_carbon = x[:, :, 1:1 + self.n_carbon]
        c_price = x[:, :, 1 + self.n_carbon:1 + self.n_carbon + self.n_price]
        P = h.shape[1]
        if self.cfg.fusion:
            h = self.gate(h, c_carbon[:, :P] if c_carbon.shape[1] >= P else c_carbon,
                          c_price[:, :P] if c_price.shape[1] >= P else c_price)
        else:
            c = torch.cat([c_carbon, c_price], dim=-1)
            h = self.gate(h, c[:, :P] if c.shape[1] >= P else c)

        if self.cfg.state_emb:
            h = h + self.state_emb(sid).unsqueeze(1)
        if self.cfg.regime_emb and reg.shape[-1] > 0:
            h = h + self.regime_proj(reg).unsqueeze(1)

        flat = h.reshape(h.shape[0], -1)
        out = self.head(flat)
        if self.cfg.multitask:
            aux = self.aux_head(flat).reshape(-1, self.cfg.n_out, self.cfg.multitask)
            return out, aux
        return out, None


class PatchTSTMCAG_Replica(nn.Module):
    """The ORIGINAL architecture, reproduced for reference scoring.

    Same shape as Cell 3.2's `PatchTST_MCAG`, but fed by the upgrade's
    calendar-correct, per-state windows so it can be scored on the same grid as
    everything else. Used only as a comparison point; the published result is
    produced by the untouched original cells.
    """

    def __init__(self, lookback: int, n_feat: int, n_carbon: int, n_out: int,
                 n_targets: int = 1, d: int = 32):
        super().__init__()
        self.backbone = PatchTSTBackbone(lookback, n_feat, d=d, patch_len=4,
                                         stride=2, layers=2, heads=4)
        self.mcag = MCAGGate(max(n_carbon, 1), d)
        self.n_out, self.n_targets = n_out, n_targets
        self.head = nn.Linear(d * self.backbone.n_patch, n_out * n_targets)

    def forward(self, x, c):
        h = self.backbone(x)
        P = h.shape[1]
        cp = c[:, :P] if c.shape[1] >= P else c[:, :1].expand(-1, P, -1)
        h = self.mcag(h, cp)
        return self.head(h.reshape(h.shape[0], -1)).reshape(-1, self.n_out, self.n_targets)


# ----------------------------------------------------------------- training
def train_mcag(model: nn.Module, ds_tr, ds_va, cfg, aux_weight: float = 0.3):
    """Train an `MCAGv2`. Same protocol as `deep.train_model`: AdamW, cosine
    schedule, early stopping on VALIDATION loss, best weights restored.

    With multi-task enabled the auxiliary loss is down-weighted so the primary
    target still drives selection — the auxiliary heads are a regulariser, not
    a second objective to trade against.
    """
    import copy
    from torch.utils.data import DataLoader
    from .deep import DEVICE, set_seed

    set_seed(cfg.seed)
    model = model.to(DEVICE)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    best, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "valid": []}
    for ep in range(cfg.epochs):
        model.train()
        tot = n = 0
        for x, xs, y, yaux, sid, reg, norm, t in dl_tr:
            x, xs, y = x.to(DEVICE), xs.to(DEVICE), y.to(DEVICE)
            sid, reg, yaux = sid.to(DEVICE), reg.to(DEVICE), yaux.to(DEVICE)
            opt.zero_grad()
            out, aux = model(x, xs, sid, reg)
            loss = F.l1_loss(out, y)
            if aux is not None and yaux.shape[-1] > 0:
                if aux.shape != yaux.shape:
                    raise ValueError(
                        f"multi-task head emits {tuple(aux.shape)} but the dataset "
                        f"supplies {tuple(yaux.shape)} auxiliary targets; these would "
                        f"silently broadcast. Set MCAGv2Config.multitask to the number "
                        f"of aux_targets passed to MCAGWindows.")
                loss = loss + aux_weight * F.l1_loss(aux, yaux)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        hist["train"].append(tot / max(n, 1))

        model.eval()
        tot = n = 0
        with torch.no_grad():
            for x, xs, y, yaux, sid, reg, norm, t in dl_va:
                x, xs, y = x.to(DEVICE), xs.to(DEVICE), y.to(DEVICE)
                sid, reg = sid.to(DEVICE), reg.to(DEVICE)
                out, _ = model(x, xs, sid, reg)
                l = F.l1_loss(out, y)          # selection on the PRIMARY target only
                tot += l.item() * len(y); n += len(y)
        va = tot / max(n, 1)
        hist["valid"].append(va)
        if va < best - 1e-6:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best


@torch.no_grad()
def predict_mcag(model: nn.Module, ds, dates, states, target: str,
                 horizons: Sequence[int], batch_size: int = 1024) -> pd.DataFrame:
    """Canonical long prediction frame, de-normalised to real units."""
    from torch.utils.data import DataLoader
    from .deep import DEVICE

    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    rows = []
    for x, xs, y, yaux, sid, reg, norm, t in dl:
        out, _ = model(x.to(DEVICE), xs.to(DEVICE), sid.to(DEVICE), reg.to(DEVICE))
        out = out.cpu().numpy()
        off = norm[:, 0].numpy()[:, None]
        sc = norm[:, 1].numpy()[:, None]
        yhat = out * sc + off
        ytrue = y.numpy() * sc + off
        s_np, t_np = sid.numpy(), t.numpy()
        for j, h in enumerate(horizons):
            rows.append(pd.DataFrame({
                DATE: dates[t_np + h],
                STATE: [states[i] for i in s_np],
                "target": target, "horizon": h,
                "y": ytrue[:, j], "yhat": yhat[:, j],
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


__all__ = [
    "CARBON_EXOG", "PRICE_EXOG", "MCAGWindows", "MCAGGate", "MCAGRegimeGate",
    "CarbonPriceFusionGate", "PatchTSTBackbone", "MCAGv2Config", "MCAGv2",
    "PatchTSTMCAG_Replica", "train_mcag", "predict_mcag",
]
