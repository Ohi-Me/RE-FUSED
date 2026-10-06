"""Dataset adapters. Every adapter returns a dict with a long frame (sid, ts, y, split, features) plus metadata.

Blinding: mode="confirm" calls guard.require_confirmatory() before any confirmatory row is read.
Development adapters drop every row that belongs to a confirmatory period.
"""
import os

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import guard
from .paths import REFUSED5_DATA, PROC

CONFIRM_START_NYISO = pd.Timestamp("2026-01-01")


def _lags(d, col, lags, roll=(), prefix=None, min_lag_for_roll=1):
    p = prefix or col
    g = d.groupby("sid", observed=True)[col]
    feats = []
    for L in lags:
        d[f"{p}_lag{L}"] = g.shift(L)
        feats.append(f"{p}_lag{L}")
    for W in roll:
        s = g.shift(min_lag_for_roll)
        d[f"{p}_rm{W}"] = s.groupby(d["sid"], observed=True).transform(lambda x: x.rolling(W, min_periods=max(2, W // 2)).mean())
        d[f"{p}_rs{W}"] = s.groupby(d["sid"], observed=True).transform(lambda x: x.rolling(W, min_periods=max(2, W // 2)).std())
        feats += [f"{p}_rm{W}", f"{p}_rs{W}"]
    return feats


def nyiso_load(mode="dev"):
    """NYISO zonal hourly load with the ISO day-ahead forecast as the operational baseline.

    Information set for a target hour on day D: everything up to the end of day D-2 (lags >= 48 h), plus the
    ISO forecast issued on D-1. Development: train < 2024, val 2024, test 2025. Confirmatory: train < 2025,
    val 2025, test 2026-01..2026-08.
    """
    p = pd.read_parquet(os.path.join(PROC, "nyiso_load_panel.parquet"))
    if mode == "confirm":
        guard.require_confirmatory()
        bounds = ("2025-01-01", "2026-01-01", "2026-09-01")
    elif mode == "dev":
        p = p[p.ts < CONFIRM_START_NYISO]
        bounds = ("2024-01-01", "2025-01-01", "2026-01-01")
    else:
        raise ValueError(mode)
    d = p.rename(columns={"zone": "sid", "load": "y"}).copy()
    d["sid"] = d.sid.astype(str)
    d = d.sort_values(["sid", "ts"]).reset_index(drop=True)
    # regular hourly grid per zone so that shift(L) means L hours
    full = []
    for s, g in d.groupby("sid"):
        idx = pd.date_range(g.ts.min(), g.ts.max(), freq="h")
        full.append(g.set_index("ts").reindex(idx).rename_axis("ts").reset_index().assign(sid=s))
    d = pd.concat(full, ignore_index=True)
    d["err_iso"] = d.y - d.iso_fc
    feats = ["iso_fc"]
    feats += _lags(d, "y", [48, 72, 168], roll=(), prefix="y")
    feats += _lags(d, "err_iso", [48, 72, 168], roll=(168,), prefix="eiso", min_lag_for_roll=48)
    d["hour"] = d.ts.dt.hour
    d["dow"] = d.ts.dt.dayofweek
    d["doy_sin"] = np.sin(2 * np.pi * d.ts.dt.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.ts.dt.dayofyear / 365.25)
    hol = USFederalHolidayCalendar().holidays(d.ts.min(), d.ts.max())
    d["holiday"] = d.ts.dt.normalize().isin(hol).astype(int)
    feats += ["hour", "dow", "doy_sin", "doy_cos", "holiday"]
    d["sid_code"] = d.sid.astype("category").cat.codes
    feats += ["sid_code"]
    d["hour_block"] = d.hour // 4                         # 6 blocks of 4 hours: state axis key
    d["season"] = ((d.ts.dt.month % 12) // 3)             # 0 DJF, 1 MAM, 2 JJA, 3 SON
    tr, va, te = map(pd.Timestamp, bounds)
    d["split"] = np.select([d.ts < tr, d.ts < va, d.ts < te], ["train", "val", "test"], "drop")
    d = d[d.split != "drop"]
    baselines = {"iso": "iso_fc", "snaive168": "y_lag168", "lag48": "y_lag48"}
    return dict(name="nyiso_load", df=d, feats=feats, baselines=baselines, state_keys=["hour_block", "hour", "season"],
                block_days=7, horizon_h=24, freq="h", energy=True)


# Non-overlapping OPSD units (most granular level per country; aggregates that duplicate sub-areas are excluded).
OPSD_UNITS = ["AT", "BE", "BG", "CH", "CZ", "DE_50hertz", "DE_amprion", "DE_tennet", "DE_transnetbw", "DK_1", "DK_2",
              "EE", "ES", "FI", "FR", "GB_GBN", "GB_NIR", "GR", "HR", "HU", "IE", "IT_CNOR", "IT_CSUD", "IT_NORD",
              "IT_SARD", "IT_SICI", "IT_SUD", "LT", "LU", "LV", "ME", "NL", "NO_1", "NO_2", "NO_3", "NO_4", "NO_5",
              "PL", "PT", "RO", "RS", "SE_1", "SE_2", "SE_3", "SE_4", "SI", "SK"]


def _surrogate(d, seed=0):
    """Replace the target (and forecast columns) by a synthetic seasonal random walk with the same schema.
    Used only to test confirmatory code paths while the data remain blinded."""
    rng = np.random.default_rng(seed)
    out = []
    for s, g in d.groupby("sid", observed=True):
        n = len(g)
        hours = pd.to_datetime(g.ts).dt.hour.to_numpy()
        y = 1000 + 100 * np.sin(2 * np.pi * hours / 24) + np.cumsum(rng.normal(scale=2, size=n))
        g = g.copy()
        g["y"] = y.astype(np.float32)
        for c in ("tso_fc", "iso_fc"):
            if c in g:
                g[c] = (y + rng.normal(scale=30, size=n)).astype(np.float32)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _hourly_features(d, fc_col=None):
    d = d.sort_values(["sid", "ts"]).reset_index(drop=True)
    full = []
    for s, g in d.groupby("sid"):
        idx = pd.date_range(g.ts.min(), g.ts.max(), freq="h")
        full.append(g.set_index("ts").reindex(idx).rename_axis("ts").reset_index().assign(sid=s))
    d = pd.concat(full, ignore_index=True)
    feats = []
    if fc_col:
        d["err_fc"] = d.y - d[fc_col]
        feats.append(fc_col)
        feats += _lags(d, "err_fc", [48, 72, 168], roll=(168,), prefix="efc", min_lag_for_roll=48)
    feats += _lags(d, "y", [48, 72, 168], roll=(168,), prefix="y", min_lag_for_roll=48)
    d["hour"] = d.ts.dt.hour
    d["dow"] = d.ts.dt.dayofweek
    d["doy_sin"] = np.sin(2 * np.pi * d.ts.dt.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.ts.dt.dayofyear / 365.25)
    d["sid_code"] = d.sid.astype("category").cat.codes
    feats += ["hour", "dow", "doy_sin", "doy_cos", "sid_code"]
    d["hour_block"] = d.hour // 4
    return d, feats


def opsd_plausibility_mask(p):
    """Deviation DC1 (sensitivity only): flag physically implausible values using TRAINING-period data only.
    A value of actual load or TSO forecast is implausible if it lies outside
    [0.5 x the area's 0.1% quantile, 1.5 x its 99.9% quantile] of actual load in 2015-2017."""
    tr = p[p.ts < "2018-01-01"].groupby("sid").y.quantile([0.001, 0.999]).unstack()
    lo = p.sid.map(tr[0.001]) * 0.5
    hi = p.sid.map(tr[0.999]) * 1.5
    return {c: (p[c] < lo) | (p[c] > hi) for c in ("y", "tso_fc")}


def opsd_load(mode="confirm", test_period="2019", clean=False):
    """OPSD / ENTSO-E load with the TSO day-ahead forecast (confirmatory; blinded).
    mode: "confirm" (requires unlock) | "surrogate" (synthetic target, real schema, for code tests).
    clean: apply the DC1 plausibility filter (sensitivity analysis after unblinding; not the pre-registered analysis)."""
    p = pd.read_parquet(os.path.join(PROC, "opsd_load_panel.parquet"))
    p = p[p.sid.isin(OPSD_UNITS)]
    if clean:
        m = opsd_plausibility_mask(p)
        p = p.copy()
        p.loc[m["y"], "y"] = np.nan
        p.loc[m["tso_fc"], "tso_fc"] = np.nan
    if mode == "confirm":
        guard.require_confirmatory()
    elif mode == "surrogate":
        p = _surrogate(p)
    else:
        raise ValueError(mode)
    d, feats = _hourly_features(p, "tso_fc")
    if test_period == "2019":
        bounds = ("2018-01-01", "2019-01-01", "2020-01-01")
    elif test_period == "2020":
        bounds = ("2019-01-01", "2020-01-01", "2020-10-01")
    else:
        raise ValueError(test_period)
    tr, va, te = map(pd.Timestamp, bounds)
    d["split"] = np.select([d.ts < tr, d.ts < va, d.ts < te], ["train", "val", "test"], "drop")
    d = d[(d.split != "drop") & (d.ts >= "2015-01-01")]
    d["country"] = d.sid.str.split("_").str[0]
    return dict(name=f"opsd_{test_period}", df=d, feats=feats, state_keys=["hour_block", "hour"], block_days=7,
                baselines={"tso": "tso_fc", "snaive168": "y_lag168", "lag48": "y_lag48"}, energy=False)


def uci_load(mode="confirm"):
    """UCI electricity clients (hourly mean kW), day-ahead information set (lags >= 48 h). Confirmatory; blinded."""
    p = pd.read_parquet(os.path.join(PROC, "uci_load_panel.parquet"))
    p = p[p.ts >= "2012-01-01"]
    if mode == "confirm":
        guard.require_confirmatory()
    elif mode == "surrogate":
        p = _surrogate(p)
    else:
        raise ValueError(mode)
    d, feats = _hourly_features(p, None)
    tr, va, te = map(pd.Timestamp, ("2013-01-01", "2014-01-01", "2015-01-01"))
    d["split"] = np.select([d.ts < tr, d.ts < va, d.ts < te], ["train", "val", "test"], "drop")
    d = d[d.split != "drop"]
    return dict(name="uci", df=d, feats=feats, state_keys=["hour_block", "hour"], block_days=7,
                baselines={"snaive168": "y_lag168", "lag48": "y_lag48"}, energy=False)


def _add_basic(d, lags, roll):
    g = d.groupby("sid", observed=True)["y"]
    feats = []
    for L in lags:
        d[f"lag{L}"] = g.shift(L)
        feats.append(f"lag{L}")
    for W in roll:
        s = g.shift(1)
        d[f"rm{W}"] = s.groupby(d["sid"], observed=True).transform(lambda x: x.rolling(W).mean())
        d[f"rs{W}"] = s.groupby(d["sid"], observed=True).transform(lambda x: x.rolling(W).std())
        feats += [f"rm{W}", f"rs{W}"]
    d["diff1"] = d["lag1"] - d[f"lag{lags[1]}"]
    return feats + ["diff1"]


def india_daily(mode="dev"):
    """India daily generation panel from RE-FUSED-5/6 (development only; no confirmatory period exists)."""
    if mode != "dev":
        raise ValueError("India has no confirmatory period")
    p = pd.read_parquet(os.path.join(REFUSED5_DATA, "india_model_daily.parquet"))
    d = p.rename(columns={"state_name": "sid", "date": "ts"})[["sid", "ts", "y", "sched", "cap", "n_units", "dow", "doy", "split"]].copy()
    d = d.sort_values(["sid", "ts"]).reset_index(drop=True)
    feats = _add_basic(d, [1, 2, 3, 7, 14, 28], [7, 28])
    d["dow_sin"] = np.sin(2 * np.pi * d.dow / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d.dow / 7)
    d["doy_sin"] = np.sin(2 * np.pi * d.doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.doy / 365.25)
    feats += ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "cap", "n_units"]
    d["vol_tercile"] = np.nan                                # filled by the engine from train thresholds
    return dict(name="india_daily", df=d, feats=feats, state_keys=["vol_tercile"], block_days=14, freq="D",
                baselines={"persistence": ("lag", 0), "roll7": ("col", "rm7"), "snaive7": ("season", 7),
                           "operator_schedule": ("colshift", "sched")}, energy=False)


def ett(name, mode="dev"):
    if mode != "dev":
        raise ValueError("ETT has no confirmatory period")
    raw = pd.read_csv(os.path.join(REFUSED5_DATA, "public", f"{name}.csv"), parse_dates=["date"])
    chans = [c for c in raw.columns if c != "date"]
    d = raw.melt(id_vars="date", value_vars=chans, var_name="sid", value_name="y").dropna()
    d = d.rename(columns={"date": "ts"}).sort_values(["sid", "ts"]).reset_index(drop=True)
    step = 24 if name.startswith("ETTh") else 96
    feats = _add_basic(d, [1, 2, 3, 6, 12, 24], [6, 24])
    d["hour"] = d.ts.dt.hour
    d["dow"] = d.ts.dt.dayofweek
    d["h_sin"] = np.sin(2 * np.pi * d.hour / 24)
    d["h_cos"] = np.cos(2 * np.pi * d.hour / 24)
    feats += ["h_sin", "h_cos", "dow"]
    ts = np.sort(d.ts.unique())
    q70, q80 = ts[int(0.7 * len(ts))], ts[int(0.8 * len(ts))]
    d["split"] = np.where(d.ts <= q70, "train", np.where(d.ts <= q80, "val", "test"))
    d["hour_block"] = d.hour // 4
    return dict(name=name, df=d, feats=feats, state_keys=["hour_block"], block_days=7, freq="h", season=step,
                baselines={"persistence": ("lag", 0), "snaive24": ("season", 24)}, energy=False)


def make_baseline(d, h, spec, season=None):
    """Target y_t = y at t+h and baseline B for t+h, strictly causal; returns None when a baseline is degenerate."""
    x = d.copy()
    g = x.groupby("sid", observed=True)
    x["y_t"] = g["y"].shift(-h)
    kind, arg = spec
    if kind == "lag":
        x["B"] = x["y"]
    elif kind == "col":
        x["B"] = x[arg]
    elif kind == "season":
        m = arg * int(np.ceil(h / arg))
        if m == h:
            return None                                    # RE-FUSED-6 D3: equals persistence
        x["B"] = g["y"].shift(-(h - m))
    elif kind == "colshift":
        x["B"] = g[arg].shift(-h)
    else:
        raise ValueError(kind)
    return x


# ---------------------------------------------------------------------------------------------------------------
# Indian data (official Grid-India, CEA and IMD data only, built by the India programme)
# ---------------------------------------------------------------------------------------------------------------
INDIA_HOURLY_SERIES = {"demand": "demand_met_mw", "net_demand": "net_demand_met_mw", "wind": "wind_mw",
                       "solar": "solar_mw"}
INDIA_HOURLY_TEST_START = pd.Timestamp("2026-03-01")


def india_hourly(mode="dev"):
    """All-India hourly series from Grid-India's Daily PSP TimeSeries sheet (SCADA, 4 Nov 2024 onwards): demand met,
    net demand met (demand minus wind and solar), wind and solar generation, in MW. Each quantity is one series.

    Information set for a target hour on day D: values up to the end of day D-2 (lags >= 48 h); the report for a day
    is published the next day, so this is what is known when a day-ahead forecast is made on D-1.
    Development: train < 2025-06, val 2025-06..11, test 2025-12..2026-02 (the test months below are never read).
    Confirmatory (pre-registered, blinded): train < 2025-09, val 2025-09..2026-02, test 2026-03..2026-09-13.
    """
    from .paths import INDIA_PROC
    p = pd.read_parquet(os.path.join(INDIA_PROC, "refused_allindia_hourly.parquet"))
    if mode == "confirm":
        guard.require_india_confirmatory()
        bounds = ("2025-09-01", "2026-03-01", "2026-09-14")
    elif mode == "dev":
        p = p[p.ts < INDIA_HOURLY_TEST_START]
        bounds = ("2025-06-01", "2025-12-01", "2026-03-01")
    else:
        raise ValueError(mode)
    long = p.melt(id_vars="ts", value_vars=list(INDIA_HOURLY_SERIES.values()), var_name="col", value_name="y")
    long["sid"] = long.col.map({v: k for k, v in INDIA_HOURLY_SERIES.items()})
    d, feats = _hourly_features(long[["sid", "ts", "y"]], None)
    d["season"] = ((d.ts.dt.month % 12) // 3)
    tr, va, te = map(pd.Timestamp, bounds)
    d["split"] = np.select([d.ts < tr, d.ts < va, d.ts < te], ["train", "val", "test"], "drop")
    d = d[d.split != "drop"]
    return dict(name="india_hourly", df=d, feats=feats, state_keys=["hour_block", "hour"], block_days=7, freq="h",
                baselines={"snaive168": "y_lag168", "lag48": "y_lag48"}, energy=True)


INDIA_DAILY_TARGETS = {"T1": "dem_energy_met_gwh", "T2": "dev_actual_drawal_gwh", "T3": "gen_act_gwh_total",
                       "T4": "re_all_total_re_gwh"}
INDIA_LEARNED = ["xgb", "chronos2", "nhits", "tide", "ens_final"]


def _india_avail_lag(col):
    from .paths import INDIA_PROC
    fr = pd.read_csv(os.path.join(INDIA_PROC, "..", "docs", "FIELD_REGISTRY.csv"))
    v = fr.loc[fr.column == col, "avail_lag_days"]
    return int(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else 1


def india_state_daily(target="T2", design="long"):
    """Daily State-control-area series of the official State-day panel (34 areas), one target at a time.

    The forecast for day D is issued on D-1 and may use a value only once it was published (availability delay from
    FIELD_REGISTRY.csv), so the shortest lag is 1 + delay days. These periods were opened by the India study, so
    every run on this adapter is exploratory (not pre-registered).
      design="long":    simple forecasts (seasonal naive 7 days, shortest available lag); train < Apr 2024,
                        val FY2024-25, test Apr 2025 - Aug 2026 (the India study's own split)
      design="learned": the India study's stored day-ahead forecasts (median, horizon 1) as baselines; the corrector
                        learns on FY2024-25, gates are fitted on Apr-Sep 2025, test Oct 2025 - Aug 2026
    """
    from .paths import INDIA_PROC, INDIA_RES
    col = INDIA_DAILY_TARGETS[target]
    p = pd.read_parquet(os.path.join(INDIA_PROC, "refused_state_day.parquet"), columns=["entity", "date", col])
    d = p.rename(columns={"entity": "sid", "date": "ts", col: "y"}).copy()
    d["sid"] = d.sid.astype(str)
    d["ts"] = pd.to_datetime(d.ts)
    d = d.sort_values(["sid", "ts"]).reset_index(drop=True)
    L0 = 1 + _india_avail_lag(col)
    feats = _lags(d, "y", sorted({L0, L0 + 1, 7, 14, 28}), roll=(7, 28), prefix="y", min_lag_for_roll=L0)
    d["dow"] = d.ts.dt.dayofweek
    d["weekend"] = (d.dow >= 5).astype(int)
    d["doy_sin"] = np.sin(2 * np.pi * d.ts.dt.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.ts.dt.dayofyear / 365.25)
    d["sid_code"] = d.sid.astype("category").cat.codes
    feats += ["dow", "doy_sin", "doy_cos", "sid_code"]
    d["season"] = ((d.ts.dt.month % 12) // 3)
    baselines = {"snaive7": "y_lag7", f"lag{L0}": f"y_lag{L0}"}
    if design == "long":
        bounds = ("2024-04-01", "2025-04-01", "2026-09-01")
    elif design == "learned":
        for m in INDIA_LEARNED:
            src = "ens_" + ("stack" if target in ("T1", "T2") else "top") if m == "ens_final" else m
            if target == "T4" and m == "ens_final":
                src = "chronos2"           # the final forecaster for renewable generation is Chronos-2 alone
            f = os.path.join(INDIA_RES, "confirm2", "o2", "preds", f"{target}__{src}.parquet")
            if not os.path.exists(f):
                continue
            q = pd.read_parquet(f, columns=["sid", "H", "target_date", "q50"])
            q = q[q.H == 1].rename(columns={"target_date": "ts", "q50": "B_" + m}).drop(columns="H")
            q["sid"] = q.sid.astype(str)
            d = d.merge(q, on=["sid", "ts"], how="left")
            baselines[m] = "B_" + m
        bounds = ("2025-04-01", "2025-10-01", "2026-09-01")
        d = d[d.ts >= "2024-04-01"]
    else:
        raise ValueError(design)
    tr, va, te = map(pd.Timestamp, bounds)
    d["split"] = np.select([d.ts < tr, d.ts < va, d.ts < te], ["train", "val", "test"], "drop")
    d = d[d.split != "drop"]
    return dict(name=f"india_daily_{target}_{design}", df=d, feats=feats, state_keys=["season", "weekend"],
                block_days=14, freq="D", baselines=baselines, energy=True, delay_days=L0)
