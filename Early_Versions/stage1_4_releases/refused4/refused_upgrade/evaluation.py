"""
evaluation.py — one evaluation protocol shared by every upgrade experiment.

Canonical prediction frame
--------------------------
Every model in every section emits the same long frame:

    date | state_name | target | horizon | y | yhat | model

That single convention is what makes the leaderboard, the regime breakdown, the
state-wise table, the Diebold-Mariano tests and the ensembles all operate on
identical, aligned evaluation points. No model is ever scored on a different
subset than its competitors: `Scorer.align` intersects the evaluation grid
across whichever models are being compared.

Metric conventions
------------------
* MASE / RMSSE use each state's own TRAIN history with m=7 and are then
  macro-averaged (unweighted) across the 18 states — identical to the code path
  that produced the published `results/tables/baseline_metrics.csv`. This is the
  headline convention.
* A POOLED variant is also reported. The original notebook scored its deep
  models pooled (one scale for the whole panel) but scored the baselines
  per-state, so its published "MASE 3.4748 vs naive 0.494" compared two
  different scalings. Reporting both makes that discrepancy visible instead of
  inheriting it.
* MAPE is reported twice: `MAPE` is the textbook definition, and
  `MAPE_floored` replicates the original notebook's floored denominator
  max(|y|, 0.1 * sigma_train) so the new numbers can be placed next to the
  published 28.20%.

MAPE is never restated as "accuracy".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from refused_fixes.metrics import mase as _mase, rmsse as _rmsse, smape as _smape
from refused_fixes.metrics import diebold_mariano as _dm

from .panel import DATE, STATE, TARGETS, OBSERVABLE_TARGETS

PRED_COLS = [DATE, STATE, "target", "horizon", "y", "yhat"]
SEASONAL_M = 7


# --------------------------------------------------------------- raw metrics
def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mape(y, p, eps: float = 1e-9) -> float:
    """Textbook MAPE (%). Undefined at y == 0, so those points are excluded."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.abs(y) > eps
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100.0)


def mape_floored(y, p, floor: float) -> float:
    """The original notebook's MAPE: denominator max(|y|, 0.1 * sigma_train)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.maximum(np.abs(y), floor)
    return float(np.mean(np.abs((y - p) / denom)) * 100.0)


def wql(y, q_preds: np.ndarray, taus: Sequence[float]) -> float:
    """Weighted quantile loss — total pinball loss normalised by sum(|y|)."""
    y = np.asarray(y, float)
    tot = 0.0
    for i, t in enumerate(taus):
        e = y - np.asarray(q_preds)[:, i]
        tot += np.sum(np.maximum(t * e, (t - 1.0) * e))
    denom = np.sum(np.abs(y)) * len(taus)
    return float(2.0 * tot / denom) if denom > 0 else float("nan")


# ------------------------------------------------------------------- scorer
@dataclass
class Scorer:
    """Holds the frozen TRAIN-side quantities every metric needs.

    `hist` — per (state, target) training history for the MASE/RMSSE scale.
    `pooled_hist` — per target, the whole training column (original convention).
    `floors` — per target, 0.1 * sigma_train, for the floored MAPE replica.
    """
    hist: Dict[Tuple[str, str], np.ndarray]
    pooled_hist: Dict[str, np.ndarray]
    floors: Dict[str, float]
    states: List[str]
    scales: Dict[Tuple[str, str], float] = field(default_factory=dict)

    @classmethod
    def from_panel(cls, df: pd.DataFrame, targets: Sequence[str] = tuple(TARGETS)) -> "Scorer":
        """Build from the panel. History = the ORIGINAL train period
        (`_split == 'train'`, i.e. fit + validation), matching baseline_metrics.csv."""
        tr = df[df["_split"] == "train"]
        hist, pooled, floors, scales = {}, {}, {}, {}
        for t in targets:
            pooled[t] = tr[t].dropna().to_numpy(float)
            floors[t] = float(tr[t].std() * 0.1)
            for s, g in tr.groupby(STATE, sort=False):
                h = g.sort_values(DATE)[t].dropna().to_numpy(float)
                hist[(s, t)] = h
                scales[(s, t)] = cls._mase_denom(h, SEASONAL_M)
        return cls(hist=hist, pooled_hist=pooled, floors=floors,
                   states=sorted(tr[STATE].unique()), scales=scales)

    @staticmethod
    def _mase_denom(y_train: np.ndarray, m: int = SEASONAL_M) -> float:
        """The MASE denominator: in-sample mean absolute seasonal difference."""
        yt = np.asarray(y_train, float)
        if len(yt) <= m:
            m = 1
        s = float(np.mean(np.abs(yt[m:] - yt[:-m])))
        if s == 0:
            s = float(np.mean(np.abs(np.diff(yt)))) or 1.0
        return s

    def scale_for(self, states: Sequence[str], target: str) -> np.ndarray:
        """Per-row MASE denominator — the natural loss weighting for this panel.

        Training a *global* model on raw levels optimises pooled absolute error,
        but the headline metric macro-averages per-state MASE. On an 18-state
        panel spanning three orders of magnitude (Puducherry ~0.7 vs Uttar
        Pradesh ~368 GWh) those two objectives disagree violently: a model can
        cut pooled MAE and still post a macro MASE above 1 because the smallest
        state dominates the average.

        Dividing each state's target by its own MASE denominator makes the
        training loss *equal* the evaluation metric up to a constant. The scale
        is estimated on TRAIN only, so it carries no test information.
        """
        return np.array([self.scales.get((s, target), 1.0) for s in states], dtype=float)

    # ---------------------------------------------------------------- scoring
    def _per_state(self, g: pd.DataFrame, target: str) -> Optional[Dict[str, float]]:
        h = self.hist.get((g.name if hasattr(g, "name") else None, target))
        return None

    def score_group(self, sub: pd.DataFrame, target: str) -> Dict[str, float]:
        """Metrics for one (target, ...) slice: macro across states + pooled."""
        rows = []
        for st, g in sub.groupby(STATE, sort=False):
            h = self.hist.get((st, target))
            if h is None or len(h) <= SEASONAL_M + 1 or len(g) < 3:
                continue
            y, p = g["y"].to_numpy(float), g["yhat"].to_numpy(float)
            rows.append({
                "MAE": mae(y, p), "RMSE": rmse(y, p),
                "MAPE": mape(y, p), "sMAPE": _smape(y, p),
                "MAPE_floored": mape_floored(y, p, self.floors.get(target, 1e-9)),
                "MASE": _mase(y, p, h, m=SEASONAL_M),
                "RMSSE": _rmsse(y, p, h, m=SEASONAL_M),
            })
        y_all, p_all = sub["y"].to_numpy(float), sub["yhat"].to_numpy(float)
        ph = self.pooled_hist.get(target, np.array([1.0, 2.0]))
        out = {
            "n_obs": int(len(sub)), "n_states": len(rows),
            "MASE_pooled": _mase(y_all, p_all, ph, m=SEASONAL_M),
            "RMSSE_pooled": _rmsse(y_all, p_all, ph, m=SEASONAL_M),
            "MAE_pooled": mae(y_all, p_all), "RMSE_pooled": rmse(y_all, p_all),
            "MAPE_floored_pooled": mape_floored(y_all, p_all, self.floors.get(target, 1e-9)),
        }
        if rows:
            agg = pd.DataFrame(rows).mean()
            out.update({k: float(agg[k]) for k in
                        ["MAE", "RMSE", "MAPE", "sMAPE", "MAPE_floored", "MASE", "RMSSE"]})
            out["MASE_worst_state"] = float(pd.DataFrame(rows)["MASE"].max())
        return out

    def by_target_horizon(self, pred: pd.DataFrame) -> pd.DataFrame:
        """Metrics for every (model, target, horizon) cell."""
        recs = []
        keys = ["model", "target", "horizon"] if "model" in pred.columns else ["target", "horizon"]
        for k, sub in pred.groupby(keys, sort=False):
            k = k if isinstance(k, tuple) else (k,)
            rec = dict(zip(keys, k))
            rec.update(self.score_group(sub, rec["target"]))
            recs.append(rec)
        return pd.DataFrame(recs)

    def by_target(self, pred: pd.DataFrame) -> pd.DataFrame:
        """Metrics per (model, target), pooling horizons 1..3."""
        recs = []
        keys = ["model", "target"] if "model" in pred.columns else ["target"]
        for k, sub in pred.groupby(keys, sort=False):
            k = k if isinstance(k, tuple) else (k,)
            rec = dict(zip(keys, k))
            rec.update(self.score_group(sub, rec["target"]))
            recs.append(rec)
        return pd.DataFrame(recs)

    def headline(self, pred: pd.DataFrame,
                 targets: Sequence[str] = tuple(OBSERVABLE_TARGETS)) -> pd.DataFrame:
        """One row per model: the publication headline.

        MASE / RMSSE are averaged over the observable targets (macro-across-states,
        pooled over h=1..3). `avg_MAPE_floored_6t` reproduces the original
        notebook's 6-target average so it can sit beside the published 28.20%.
        """
        bt = self.by_target(pred)
        recs = []
        keys = ["model"] if "model" in bt.columns else []
        groups = bt.groupby("model", sort=False) if keys else [("(single)", bt)]
        for m, g in groups:
            obs = g[g["target"].isin(targets)]
            rec = {"model": m}
            for col in ["MASE", "RMSSE", "MAE", "RMSE", "MAPE", "sMAPE", "MAPE_floored"]:
                rec[col] = float(obs[col].mean()) if len(obs) else np.nan
            rec["MASE_pooled"] = float(obs["MASE_pooled"].mean()) if len(obs) else np.nan
            rec["avg_MAPE_floored_6t"] = float(g["MAPE_floored"].mean())
            rec["avg_MAPE_6t"] = float(g["MAPE"].mean())
            rec["n_obs"] = int(g["n_obs"].sum())
            recs.append(rec)
        return (pd.DataFrame(recs)
                .sort_values("MASE", kind="mergesort")
                .reset_index(drop=True))

    def by_state(self, pred: pd.DataFrame,
                 targets: Sequence[str] = tuple(OBSERVABLE_TARGETS)) -> pd.DataFrame:
        """State-wise error for the given targets."""
        recs = []
        sub0 = pred[pred["target"].isin(targets)]
        keys = [c for c in ["model", "target", STATE] if c in sub0.columns]
        for k, g in sub0.groupby(keys, sort=False):
            k = k if isinstance(k, tuple) else (k,)
            rec = dict(zip(keys, k))
            h = self.hist.get((rec[STATE], rec["target"]))
            if h is None:
                continue
            y, p = g["y"].to_numpy(float), g["yhat"].to_numpy(float)
            rec.update({"n_obs": len(g), "MAE": mae(y, p), "RMSE": rmse(y, p),
                        "MAPE": mape(y, p), "sMAPE": _smape(y, p),
                        "MASE": _mase(y, p, h, m=SEASONAL_M),
                        "RMSSE": _rmsse(y, p, h, m=SEASONAL_M)})
            recs.append(rec)
        return pd.DataFrame(recs)

    def by_regime(self, pred: pd.DataFrame, regime_map: pd.DataFrame,
                  regime_cols: Sequence[str],
                  targets: Sequence[str] = tuple(OBSERVABLE_TARGETS)) -> pd.DataFrame:
        """Error conditional on each regime flag (flags may overlap)."""
        sub = pred[pred["target"].isin(targets)].merge(
            regime_map, on=[DATE, STATE], how="left")
        recs = []
        keys = [c for c in ["model", "target"] if c in sub.columns]
        for rc in regime_cols:
            if rc not in sub.columns:
                continue
            sel = sub[sub[rc] == 1]
            if not len(sel):
                for k, _ in sub.groupby(keys, sort=False):
                    k = k if isinstance(k, tuple) else (k,)
                    rec = dict(zip(keys, k)); rec["regime"] = rc; rec["n_obs"] = 0
                    recs.append(rec)
                continue
            for k, g in sel.groupby(keys, sort=False):
                k = k if isinstance(k, tuple) else (k,)
                rec = dict(zip(keys, k)); rec["regime"] = rc
                rec.update(self.score_group(g, rec["target"]))
                recs.append(rec)
        return pd.DataFrame(recs)


# ---------------------------------------------------------------- alignment
def align(preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Intersect several models onto identical evaluation points and stack them.

    Guarantees every model in a comparison is scored on exactly the same
    (date, state, target, horizon) rows.
    """
    keys = [DATE, STATE, "target", "horizon"]
    common = None
    for name, p in preds.items():
        idx = pd.MultiIndex.from_frame(p[keys].drop_duplicates())
        common = idx if common is None else common.intersection(idx)
    frames = []
    for name, p in preds.items():
        q = p.set_index(keys)
        q = q.loc[q.index.isin(common)].reset_index()
        q["model"] = name
        frames.append(q[PRED_COLS + ["model"]])
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------- significance
def dm_panel(pred_a: pd.DataFrame, pred_b: pd.DataFrame,
             target: str, loss: str = "mae",
             horizons: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """Diebold-Mariano on a panel, done the defensible way.

    The loss differential is averaged ACROSS STATES for each date, producing a
    single daily time series of length ~n_test_days. DM is then applied to that
    series. Pooling all state-days into one long vector would treat 18
    contemporaneous, cross-correlated observations as independent draws and
    overstate significance.

    dm < 0 => `pred_a` has the smaller loss (a is better).
    """
    keys = [DATE, STATE, "target", "horizon"]
    a = pred_a[pred_a["target"] == target]
    b = pred_b[pred_b["target"] == target]
    if horizons is not None:
        a = a[a["horizon"].isin(horizons)]
        b = b[b["horizon"].isin(horizons)]
    m = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if len(m) < 20:
        return {"n": len(m), "DM": np.nan, "p": np.nan, "better": None}
    ea = np.abs(m["y_a"] - m["yhat_a"]) if loss == "mae" else (m["y_a"] - m["yhat_a"]) ** 2
    eb = np.abs(m["y_a"] - m["yhat_b"]) if loss == "mae" else (m["y_a"] - m["yhat_b"]) ** 2
    d = (pd.DataFrame({DATE: m[DATE], "d": ea.to_numpy() - eb.to_numpy()})
         .groupby(DATE)["d"].mean().sort_index())
    n = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    if gamma0 <= 0 or n < 10:
        return {"n": n, "DM": np.nan, "p": np.nan, "better": None}
    from scipy import stats as _st
    dm = dbar / np.sqrt(gamma0 / n)
    p = float(2.0 * (1.0 - _st.t.cdf(abs(dm), df=n - 1)))
    return {"n": int(n), "DM": float(dm), "p": p,
            "better": ("a" if dm < 0 else "b"), "mean_loss_diff": float(dbar)}


def block_bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean,
                       block: int = 14, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Moving-block bootstrap CI for a statistic of a serially-dependent series.

    Returns (point_estimate, lo, hi).
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < block * 2:
        return float(stat(v)) if n else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([v[s:s + block] for s in starts])[:n]
        boots[i] = stat(sample)
    return (float(stat(v)),
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def daily_loss_series(pred: pd.DataFrame, target: str, loss: str = "mae") -> np.ndarray:
    """Cross-state mean loss per date — the unit the bootstrap resamples."""
    s = pred[pred["target"] == target]
    e = np.abs(s["y"] - s["yhat"]) if loss == "mae" else (s["y"] - s["yhat"]) ** 2
    return (pd.DataFrame({DATE: s[DATE], "e": e.to_numpy()})
            .groupby(DATE)["e"].mean().sort_index().to_numpy())


# ----------------------------------------------------------------- registry
@dataclass
class Registry:
    """Collects every model's predictions and metadata for the whole upgrade."""
    out_dir: Path
    preds: Dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: Dict[str, dict] = field(default_factory=dict)

    def add(self, name: str, pred: pd.DataFrame, section: str = "", notes: str = "",
            **extra) -> pd.DataFrame:
        missing = [c for c in PRED_COLS if c not in pred.columns]
        if missing:
            raise ValueError(f"{name}: prediction frame missing {missing}")
        p = pred[PRED_COLS].copy()
        p = p[p["y"].notna() & p["yhat"].notna()].reset_index(drop=True)
        # Label columns are left as object rather than categorical on purpose:
        # they hold references to interned strings (~8 bytes/row), so the whole
        # registry is only a few hundred MB even at ~30 models, and categoricals
        # would introduce empty-group and category-mismatch hazards in the
        # groupby/merge paths that every metric goes through.
        self.preds[name] = p
        self.meta[name] = {"section": section, "notes": notes,
                           "n_obs": int(len(p)), **extra}
        return p

    def get(self, *names: str) -> pd.DataFrame:
        return align({n: self.preds[n] for n in names})

    def stacked(self, names: Optional[Iterable[str]] = None) -> pd.DataFrame:
        names = list(names) if names is not None else list(self.preds)
        return align({n: self.preds[n] for n in names})

    def save_metrics(self, scorer: "Scorer", filename: str,
                     names: Optional[Iterable[str]] = None) -> pd.DataFrame:
        st = self.stacked(names)
        tbl = scorer.headline(st)
        tbl.to_csv(self.out_dir / "tables" / filename, index=False)
        return tbl

    def save_predictions(self, name: str) -> Path:
        p = self.out_dir / "tables" / f"pred_{name.replace('/', '_')}.csv.gz"
        self.preds[name].to_csv(p, index=False, compression="gzip")
        return p

    def dump_meta(self, filename: str = "model_registry.json") -> Path:
        p = self.out_dir / "tables" / filename
        p.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        return p

    # ------------------------------------------------------------ checkpoint
    def checkpoint(self, filename: str = "_registry.pkl") -> Path:
        """Persist every registered prediction so a long run can be resumed.

        Sections U3-U11 are expensive; checkpointing after each one means a
        failure late in the sequence does not cost the earlier fits.
        """
        import pickle
        p = self.out_dir / filename
        with p.open("wb") as fh:
            pickle.dump({"preds": self.preds, "meta": self.meta}, fh, protocol=4)
        return p

    def restore(self, filename: str = "_registry.pkl", only_missing: bool = True) -> int:
        """Reload a checkpoint. Returns how many models were added."""
        import pickle
        p = self.out_dir / filename
        if not p.exists():
            return 0
        with p.open("rb") as fh:
            blob = pickle.load(fh)
        n = 0
        for k, v in blob.get("preds", {}).items():
            if only_missing and k in self.preds:
                continue
            self.preds[k] = v
            self.meta[k] = blob.get("meta", {}).get(k, {})
            n += 1
        return n


__all__ = [
    "PRED_COLS", "SEASONAL_M", "mae", "rmse", "mape", "mape_floored", "wql",
    "Scorer", "align", "dm_panel", "block_bootstrap_ci", "daily_loss_series",
    "Registry",
]
