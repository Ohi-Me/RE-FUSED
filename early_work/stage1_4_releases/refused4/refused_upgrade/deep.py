"""
deep.py — PyTorch harness for the Part II deep time-series models.

Why the models are implemented here rather than pulled from a library
--------------------------------------------------------------------
`darts` / `neuralforecast` would drag a large dependency tree (and in several
combinations a different torch build) into an environment whose existing results
must stay reproducible. The architectures needed here — N-BEATS, N-HiTS,
TSMixer, plus a faithful replica of the notebook's PatchTST+MCAG — are each
compact, so they are implemented directly against the installed torch. Nothing
in the original notebook is imported, patched or re-run.

Instance normalisation (the single most important design choice)
----------------------------------------------------------------
Every model here forecasts in a **window-normalised** space:

    x_norm = (x − last_observed_value) / window_scale
    ŷ      = last_observed_value + window_scale · ŷ_norm

This does two things that matter on this panel:

  * it removes the train→test level shift that makes direct level prediction
    hopeless for a fixed-range learner (Section U3 measured this), so the deep
    models are structurally comparable to the Section U4 hybrids rather than
    being set up to fail;
  * it puts all 18 states on one scale. Generation ranges from ~0.7 (Puducherry)
    to ~400 (Uttar Pradesh); without per-window scaling a global model would be
    fitted almost entirely to the largest states.

Because the offset is the *last observed value*, a model that outputs zero is
exactly the persistence forecast. The deep models therefore start from the same
floor as the Section U4 hybrids, and any improvement is genuinely learned.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .panel import DATE, STATE, CalendarPanel, Splits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------- data
@dataclass
class WindowSpec:
    lookback: int = 56
    horizons: Tuple[int, ...] = (1, 2, 3)
    calendar: bool = True

    @property
    def max_h(self) -> int:
        return max(self.horizons)


def window_index(Y: np.ndarray, S: np.ndarray, lookback: int, max_h: int,
                 split: str) -> List[Tuple[int, int]]:
    """Admissible (state, origin) pairs: full observed window + observed targets.

    A cumulative observation count makes the "is this window gap-free?" test O(1)
    per candidate instead of O(lookback).
    """
    obs = np.isfinite(Y)
    cobs = np.cumsum(obs.astype(np.int32), axis=0)
    idx: List[Tuple[int, int]] = []
    n_t, n_s = Y.shape
    for s in range(n_s):
        for t in range(lookback - 1, n_t - max_h):
            if (cobs[t, s] - (cobs[t - lookback, s] if t - lookback >= 0 else 0)) != lookback:
                continue
            if not obs[t + 1:t + 1 + max_h, s].all():
                continue
            if S[t + max_h, s] != split:
                continue
            idx.append((s, t))
    return idx


def gather_windows(M: np.ndarray, idx_s: np.ndarray, idx_t: np.ndarray,
                   lookback: int) -> np.ndarray:
    """Vectorised (N, lookback) gather from a (T, S) matrix.

    Slicing one window at a time inside `__getitem__` makes the DataLoader the
    bottleneck — it dominated training time by an order of magnitude before this
    was hoisted into a single fancy-index in `__init__`.
    """
    rows = idx_t[:, None] + np.arange(-lookback + 1, 1)[None, :]
    return M[rows, idx_s[:, None]]


class PanelWindows(Dataset):
    """Sliding windows over the calendar grid, built per state.

    A sample is admissible only when the whole lookback window and every target
    step are observed, so no imputed-by-construction value is ever fabricated to
    fill a window. The forecast origin is the last index of the window; targets
    are the following `max_h` days, which keeps the sample strictly causal.

    All windows are materialised once in `__init__`; `__getitem__` is then a
    pure tensor index.
    """

    def __init__(self, y_wide: pd.DataFrame, spec: WindowSpec,
                 split_wide: pd.DataFrame, split: str,
                 exog_wide: Optional[Dict[str, pd.DataFrame]] = None):
        self.spec = spec
        self.states = list(y_wide.columns)
        self.dates = y_wide.index
        Y = y_wide.to_numpy(float)                      # (n_dates, n_states)
        self.Y = Y
        self.exog = ([e.to_numpy(float) for e in exog_wide.values()]
                     if exog_wide else [])
        S = split_wide.to_numpy(dtype=object)
        L, H = spec.lookback, spec.max_h
        self.index = window_index(Y, S, L, H, split)

        if not self.index:
            self.X = torch.zeros(0, L, 1)
            self.y = torch.zeros(0, H)
            self.sid = torch.zeros(0, dtype=torch.long)
            self.norm = torch.zeros(0, 2)
            self.t = torch.zeros(0, dtype=torch.long)
            return

        idx_s = np.array([s for s, _ in self.index])
        idx_t = np.array([t for _, t in self.index])

        win = gather_windows(Y, idx_s, idx_t, L)                    # (N, L)
        tgt = np.stack([Y[idx_t + k, idx_s] for k in range(1, H + 1)], axis=1)

        # instance normalisation anchored on the last observed value
        offset = win[:, -1]
        scale = win.std(axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)

        chans = [((win - offset[:, None]) / scale[:, None])[:, :, None]]
        for E in self.exog:
            e = gather_windows(E, idx_s, idx_t, L)
            mu = np.nanmean(e, axis=1, keepdims=True)
            sd = np.nanstd(e, axis=1, keepdims=True)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            chans.append(np.nan_to_num((e - mu) / sd)[:, :, None])
        if spec.calendar:
            d = pd.DatetimeIndex(self.dates)
            cal = np.stack([
                np.sin(2 * np.pi * d.dayofweek.to_numpy() / 7.0),
                np.cos(2 * np.pi * d.dayofweek.to_numpy() / 7.0),
                np.sin(2 * np.pi * d.month.to_numpy() / 12.0),
                np.cos(2 * np.pi * d.month.to_numpy() / 12.0),
                np.sin(2 * np.pi * d.dayofyear.to_numpy() / 365.25),
                np.cos(2 * np.pi * d.dayofyear.to_numpy() / 365.25),
            ], axis=1).astype("f4")
            rows = idx_t[:, None] + np.arange(-L + 1, 1)[None, :]
            chans.append(cal[rows])                                  # (N, L, 6)

        self.X = torch.from_numpy(np.concatenate(chans, axis=2).astype("f4"))
        self.y = torch.from_numpy(((tgt - offset[:, None]) / scale[:, None]).astype("f4"))
        self.sid = torch.from_numpy(idx_s.astype(np.int64))
        self.norm = torch.from_numpy(np.stack([offset, scale], axis=1).astype("f4"))
        self.t = torch.from_numpy(idx_t.astype(np.int64))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i], self.sid[i], self.norm[i], self.t[i]


# ------------------------------------------------------------------ models
class NBeatsBlock(nn.Module):
    def __init__(self, in_dim, theta, width, out_dim, layers=4):
        super().__init__()
        seq = []
        d = in_dim
        for _ in range(layers):
            seq += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.fc = nn.Sequential(*seq)
        self.back = nn.Linear(width, in_dim)
        self.fore = nn.Linear(width, out_dim)

    def forward(self, x):
        h = self.fc(x)
        return self.back(h), self.fore(h)


class NBeats(nn.Module):
    """N-BEATS: a residual stack of fully-connected blocks (Oreshkin et al., 2020).

    Generic architecture: each block subtracts what it can explain from the
    backcast and adds its forecast, so later blocks see only the unexplained part.
    """

    def __init__(self, lookback, n_out, n_blocks=4, width=256, n_feat=1, n_states=18):
        super().__init__()
        self.lookback = lookback
        self.in_dim = lookback * n_feat
        self.blocks = nn.ModuleList([
            NBeatsBlock(self.in_dim, width, width, n_out) for _ in range(n_blocks)])

    def forward(self, X, sid=None):
        x = X.flatten(1)
        fc = 0.0
        for b in self.blocks:
            back, fore = b(x)
            x = x - back
            fc = fc + fore
        return fc


class NHiTS(nn.Module):
    """N-HiTS: N-BEATS with multi-rate input pooling (Challu et al., 2023).

    Each stack pools the input at a different rate before the MLP and expands its
    forecast back up, so different stacks specialise on different frequencies —
    which suits a series carrying both weekly and seasonal structure.
    """

    def __init__(self, lookback, n_out, pool_sizes=(8, 4, 1), width=256,
                 n_feat=1, n_blocks_per_stack=2, n_states=18):
        super().__init__()
        self.lookback, self.n_out = lookback, n_out
        self.stacks = nn.ModuleList()
        self.pools = list(pool_sizes)
        for p in self.pools:
            pooled = math.ceil(lookback / p) * n_feat
            blocks = nn.ModuleList([
                NBeatsBlock(pooled, width, width, max(1, n_out)) for _ in range(n_blocks_per_stack)])
            self.stacks.append(blocks)
        self.n_feat = n_feat

    def forward(self, X, sid=None):
        B = X.shape[0]
        residual = X
        total = 0.0
        for p, blocks in zip(self.pools, self.stacks):
            xp = X.transpose(1, 2)                       # (B, C, L)
            if p > 1:
                xp = F.avg_pool1d(xp, kernel_size=p, stride=p, ceil_mode=True)
            xp = xp.transpose(1, 2).flatten(1)
            for b in blocks:
                back, fore = b(xp)
                xp = xp - back
                total = total + fore
        return total


class TSMixer(nn.Module):
    """TSMixer: alternating time-mixing and feature-mixing MLPs (Chen et al., 2023).

    An all-MLP alternative to attention: one MLP mixes across time steps, another
    across channels, with residual connections and normalisation between.
    """

    def __init__(self, lookback, n_out, n_feat=1, n_blocks=4, hidden=128,
                 dropout=0.1, n_states=18):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "t_norm": nn.LayerNorm(n_feat),
                "t_mix": nn.Sequential(nn.Linear(lookback, hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, lookback)),
                "f_norm": nn.LayerNorm(n_feat),
                "f_mix": nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, n_feat)),
            }))
        self.head = nn.Linear(lookback * n_feat, n_out)

    def forward(self, X, sid=None):
        h = X                                            # (B, L, C)
        for blk in self.blocks:
            z = blk["t_norm"](h).transpose(1, 2)         # (B, C, L)
            z = blk["t_mix"](z).transpose(1, 2)
            h = h + z
            z = blk["f_norm"](h)
            h = h + blk["f_mix"](z)
        return self.head(h.flatten(1))


# ------------------------------------------------------------------ training
@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    loss: str = "l1"
    seed: int = 42
    verbose: bool = False


def train_model(model: nn.Module, ds_tr: Dataset, ds_va: Dataset,
                cfg: TrainConfig) -> Tuple[nn.Module, Dict[str, list]]:
    """Standard loop: Adam + cosine schedule, early stopping on VALIDATION loss."""
    set_seed(cfg.seed)
    model = model.to(DEVICE)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    lossfn = F.l1_loss if cfg.loss == "l1" else F.mse_loss

    best, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "valid": []}
    for ep in range(cfg.epochs):
        model.train()
        tot = n = 0
        for X, y, sid, norm, _ in dl_tr:
            X, y, sid = X.to(DEVICE), y.to(DEVICE), sid.to(DEVICE)
            opt.zero_grad()
            out = model(X, sid)
            loss = lossfn(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        tr_loss = tot / max(n, 1)

        model.eval()
        tot = n = 0
        with torch.no_grad():
            for X, y, sid, norm, _ in dl_va:
                X, y, sid = X.to(DEVICE), y.to(DEVICE), sid.to(DEVICE)
                loss = lossfn(model(X, sid), y)
                tot += loss.item() * len(y); n += len(y)
        va_loss = tot / max(n, 1)
        hist["train"].append(tr_loss); hist["valid"].append(va_loss)

        if va_loss < best - 1e-6:
            best, best_state, bad = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if cfg.verbose:
            print(f"      ep {ep+1:3d}  train={tr_loss:.5f}  valid={va_loss:.5f}"
                  + ("  <-best" if bad == 0 else f"  p{bad}/{cfg.patience}"))
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model: nn.Module, ds: Dataset, spec: WindowSpec, dates: pd.DatetimeIndex,
            states: Sequence[str], target: str, batch_size: int = 1024) -> pd.DataFrame:
    """Emit the canonical long prediction frame, de-normalised to real units."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    rows = []
    for X, y, sid, norm, t in dl:
        out = model(X.to(DEVICE), sid.to(DEVICE)).cpu().numpy()
        offset = norm[:, 0].numpy()[:, None]
        scale = norm[:, 1].numpy()[:, None]
        yhat = out * scale + offset
        ytrue = y.numpy() * scale + offset
        s_np, t_np = sid.numpy(), t.numpy()
        for j, h in enumerate(spec.horizons):
            k = h - 1
            rows.append(pd.DataFrame({
                DATE: dates[t_np + h],
                STATE: [states[i] for i in s_np],
                "target": target, "horizon": h,
                "y": ytrue[:, k], "yhat": yhat[:, k],
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_datasets(cal: CalendarPanel, target: str, spec: WindowSpec,
                   exog: Optional[Sequence[str]] = None):
    """(train, valid, test) window datasets for one target."""
    y_wide = cal.wide(target)
    split_wide = cal.split_of()
    ex = {c: cal.wide(c) for c in (exog or [])}
    return tuple(PanelWindows(y_wide, spec, split_wide, sp, ex)
                 for sp in ("train", "valid", "test"))


__all__ = [
    "DEVICE", "set_seed", "WindowSpec", "PanelWindows", "NBeats", "NHiTS",
    "TSMixer", "TrainConfig", "train_model", "predict", "build_datasets",
]
