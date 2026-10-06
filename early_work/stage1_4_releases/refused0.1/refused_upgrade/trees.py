"""
trees.py — gradient-boosted tabular forecasters under one fixed protocol.

Protocol (identical for LightGBM / XGBoost / CatBoost, and reused verbatim by
the hybrid residual models in Section U4)
--------------------------------------------------------------------------
  1. fit on TRAIN(fit) with VALIDATION used only for early stopping;
  2. record the selected number of boosting rounds;
  3. refit on TRAIN(fit) + VALIDATION with that fixed round count;
  4. predict TEST once.

Step 3 matters: the final model must see the 12 months immediately preceding the
test period, otherwise it is handicapped relative to the classical baselines,
which are all fitted on the full training history. Step 1 is what keeps the
round count honest — the test split is never consulted.

A note on extrapolation
-----------------------
A tree ensemble predicts a weighted average of training leaf values, so it can
never emit a value outside the range of the training targets. On this panel the
mean market price roughly doubles from TRAIN to TEST, which makes direct level
prediction structurally incapable of tracking the test period. `fit_predict`
therefore reports `oob_extrapolation_rate` — the share of test targets lying
outside the training target range — so the failure mode is measured rather than
inferred. Section U4 removes the problem by learning a residual instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_LGB = dict(objective="regression", metric="l1", learning_rate=0.06,
                   num_leaves=63, min_data_in_leaf=40, feature_fraction=0.85,
                   bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
                   verbosity=-1, num_threads=0)

DEFAULT_XGB = dict(objective="reg:absoluteerror", learning_rate=0.09,
                   max_depth=6, min_child_weight=20, subsample=0.85,
                   colsample_bytree=0.85, reg_lambda=1.0, verbosity=0)

DEFAULT_CAT = dict(loss_function="MAE", learning_rate=0.10, depth=6,
                   l2_leaf_reg=3.0, verbose=0, allow_writing_files=False,
                   boosting_type="Plain", bootstrap_type="Bernoulli", subsample=0.85)

# On the metric-normalised target the validation curve is very flat for several
# targets, so early stopping fires late and the round cap dominates runtime.
# Rounds are capped and the learning rates raised to compensate: on the measured
# configurations this reaches the same validation loss for roughly a third of the
# cost. `best_rounds` is reported per fit in U3.4 so the cap can be checked.
MAX_ROUNDS = 700
EARLY_STOP = 50


@dataclass
class FitResult:
    """Everything one (model, target, horizon) fit produces."""
    test_pred: np.ndarray
    valid_pred: np.ndarray
    best_rounds: int
    importance: Optional[pd.Series] = None
    extras: Dict[str, float] = field(default_factory=dict)


def _matrices(feat: pd.DataFrame, fcols: Sequence[str], y_col: str = "y"):
    tr = feat[feat["split"] == "train"]
    va = feat[feat["split"] == "valid"]
    te = feat[feat["split"] == "test"]
    return (tr[list(fcols)], tr[y_col], va[list(fcols)], va[y_col],
            te[list(fcols)], te[y_col])


def extrapolation_rate(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Share of test targets outside the training target range."""
    lo, hi = float(np.nanmin(y_train)), float(np.nanmax(y_train))
    y = np.asarray(y_test, float)
    return float(np.mean((y < lo) | (y > hi)))


def fit_predict(kind: str, feat: pd.DataFrame, fcols: Sequence[str],
                params: Optional[dict] = None, seed: int = 42,
                y_col: str = "y", cat_features: Optional[Sequence[str]] = None
                ) -> FitResult:
    """Fit one gradient-boosted model under the fixed four-step protocol."""
    Xtr, ytr, Xva, yva, Xte, yte = _matrices(feat, fcols, y_col)
    kind = kind.lower()

    if kind in ("lgb", "lightgbm"):
        import lightgbm as lgb
        p = {**DEFAULT_LGB, **(params or {}), "seed": seed}
        dtr = lgb.Dataset(Xtr, ytr)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        booster = lgb.train(p, dtr, num_boost_round=MAX_ROUNDS,
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        best = int(booster.best_iteration or MAX_ROUNDS)
        vpred = booster.predict(Xva, num_iteration=best)
        full = lgb.Dataset(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
        final = lgb.train(p, full, num_boost_round=best)
        tpred = final.predict(Xte)
        imp = pd.Series(final.feature_importance("gain"), index=list(fcols))

    elif kind in ("xgb", "xgboost"):
        import xgboost as xgb
        p = {**DEFAULT_XGB, **(params or {}), "seed": seed}
        dtr = xgb.DMatrix(Xtr, ytr)
        dva = xgb.DMatrix(Xva, yva)
        bst = xgb.train(p, dtr, num_boost_round=MAX_ROUNDS,
                        evals=[(dva, "valid")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        best = int(bst.best_iteration + 1)
        vpred = bst.predict(dva, iteration_range=(0, best))
        dfull = xgb.DMatrix(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
        final = xgb.train(p, dfull, num_boost_round=best)
        tpred = final.predict(xgb.DMatrix(Xte))
        _sc = final.get_score(importance_type="gain")
        imp = pd.Series({c: _sc.get(c, 0.0) for c in fcols})

    elif kind in ("cat", "catboost"):
        from catboost import CatBoostRegressor, Pool
        p = {**DEFAULT_CAT, **(params or {}), "random_seed": seed}
        cf = list(cat_features or [])
        ptr = Pool(Xtr, ytr, cat_features=cf)
        pva = Pool(Xva, yva, cat_features=cf)
        m = CatBoostRegressor(iterations=MAX_ROUNDS, **p)
        m.fit(ptr, eval_set=pva, early_stopping_rounds=EARLY_STOP, verbose=False)
        best = int(m.get_best_iteration() or MAX_ROUNDS) + 1
        vpred = m.predict(Xva)
        m2 = CatBoostRegressor(iterations=best, **p)
        m2.fit(Pool(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]), cat_features=cf),
               verbose=False)
        tpred = m2.predict(Xte)
        imp = pd.Series(m2.get_feature_importance(), index=list(fcols))

    else:
        raise ValueError(f"unknown model kind {kind!r}")

    return FitResult(
        test_pred=np.asarray(tpred, float),
        valid_pred=np.asarray(vpred, float),
        best_rounds=best,
        importance=imp.sort_values(ascending=False),
        extras={"oob_extrapolation_rate": extrapolation_rate(
            pd.concat([ytr, yva]).to_numpy(float), yte.to_numpy(float))},
    )


def fit_predict_normalized(kind: str, feat: pd.DataFrame, fcols: Sequence[str],
                           scale: np.ndarray, base: Optional[np.ndarray] = None,
                           params: Optional[dict] = None, seed: int = 42,
                           cat_features: Optional[Sequence[str]] = None
                           ) -> Tuple[FitResult, np.ndarray]:
    """Fit on the scale-normalised (and optionally base-removed) target.

        z = (y - base) / scale        <- what the model learns
        yhat = z_hat * scale + base   <- what is scored

    `scale` is the per-row MASE denominator (see `Scorer.scale_for`), estimated
    on TRAIN only. Learning `z` rather than `y` aligns the training loss with the
    macro-averaged MASE the leaderboard reports, and removes the cross-state
    scale spread that otherwise lets one small state dominate the metric.

    Returns (fit, test_yhat, test_rows, valid_yhat, valid_rows), all in original
    units. Callers must build prediction frames from the returned row slices
    rather than re-slicing the input: rows with a non-finite normalised target
    are dropped here, so the two are not guaranteed to line up.

    The VALIDATION predictions come from the early-stopping fit — a model that
    has never seen the validation block. They are what Sections U9 and U10 use to
    learn ensemble weights and conformal quantiles, which is precisely why those
    weights carry no test information.
    """
    b = np.zeros(len(feat)) if base is None else np.asarray(base, float)
    s = np.asarray(scale, float)
    s = np.where(np.isfinite(s) & (s > 0), s, 1.0)

    work = feat.copy()
    work["_z"] = (work["y"].to_numpy(float) - b) / s
    work["_base"] = b
    work["_scale"] = s
    work = work[np.isfinite(work["_z"])].reset_index(drop=True)

    fit = fit_predict(kind, work, fcols, params=params, seed=seed, y_col="_z",
                      cat_features=cat_features)

    def _denorm(split_name, z):
        rows = work[work["split"] == split_name].reset_index(drop=True)
        if len(z) != len(rows):                          # defensive: must never fire
            raise RuntimeError(f"{split_name}: {len(z)} preds vs {len(rows)} rows")
        return z * rows["_scale"].to_numpy() + rows["_base"].to_numpy(), rows

    te_yhat, te_rows = _denorm("test", fit.test_pred)
    va_yhat, va_rows = _denorm("valid", fit.valid_pred)
    return fit, te_yhat, te_rows, va_yhat, va_rows


def pred_frame_from_fit(feat: pd.DataFrame, fit: FitResult, target: str,
                        horizon: int, split: str = "test",
                        base: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Assemble the canonical prediction frame from a fit.

    `base` (optional) is added back to the model output — that is how the
    Section U4 hybrids turn a predicted residual into a level forecast.
    """
    from .panel import DATE, STATE
    sub = feat[feat["split"] == split]
    yhat = fit.test_pred if split == "test" else fit.valid_pred
    yhat = np.asarray(yhat, float)
    if base is not None:
        yhat = yhat + np.asarray(base, float)
    return pd.DataFrame({
        DATE: sub[DATE].to_numpy(),
        STATE: sub[STATE].to_numpy(),
        "target": target, "horizon": horizon,
        "y": sub["y"].to_numpy(float), "yhat": yhat,
    })


__all__ = ["DEFAULT_LGB", "DEFAULT_XGB", "DEFAULT_CAT", "MAX_ROUNDS", "EARLY_STOP",
           "FitResult", "fit_predict", "fit_predict_normalized", "pred_frame_from_fit",
           "extrapolation_rate"]
