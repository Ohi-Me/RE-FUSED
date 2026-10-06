"""O2 data layer (protocol `02_design/03_o2_dev_protocol.md`).

* `load_panel`       blinded panel loader (reserved rows need the confirmatory guard)
* `series_frame`     complete daily frame per series for a target (entity or DSM bid area)
* `tabular_samples`  lag-aware samples (series, issue day τ, H) with target, anchors and features
* `mase_scales`      train-period seasonal-naive error scales per (series, H)

Information set: a column with availability lag L is used up to date τ − L (FIELD_REGISTRY.csv); target-day calendar
is known; `instant=True` sets L = 0 for every column (latency sensitivity S-lat).
"""
import os

import numpy as np
import pandas as pd

from . import guard
from .paths import DOCS, PROC

RESERVED_START = pd.Timestamp("2025-04-01")
# Phase switch (environment variable REFUSED8_PHASE). "dev" (default): fit on FY2018-23, calibrate/select on FY2023-24,
# evaluate on FY2024-25. "confirm": the same pipeline with every boundary moved forward one year — fit on FY2018-24,
# calibration and online-gate warm-up on FY2024-25, evaluation on the reserved period 1 Apr 2025 – 31 Aug 2026 (the
# split labels keep their roles: "validation" = calibration year, "dev_test" = evaluation period). Confirm requires the
# blinding guard (pre-registration hash, git tag, REFUSED8_CONFIRMATORY=1).
PHASE = os.environ.get("REFUSED8_PHASE", "dev")
if PHASE not in ("dev", "confirm"):
    raise ValueError(f"REFUSED8_PHASE must be dev or confirm, not {PHASE}")
PHASE_DIR = PHASE
if PHASE == "dev":
    TRAIN_END, VAL_END, DEV_END = pd.Timestamp("2023-03-31"), pd.Timestamp("2024-03-31"), pd.Timestamp("2025-03-31")
else:
    TRAIN_END, VAL_END, DEV_END = pd.Timestamp("2024-03-31"), pd.Timestamp("2025-03-31"), pd.Timestamp("2026-08-31")
EARLY_STOP_START = TRAIN_END - pd.Timedelta(days=179)  # last 180 train days: early stopping only
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
HORIZONS = [1, 2, 3]
WINDOW = 14
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer", 6: "monsoon", 7: "monsoon",
          8: "monsoon", 9: "monsoon", 10: "post", 11: "post"}

TARGETS = {
    "T1": dict(col="dem_energy_met_gwh", level="entity", name="energy met"),
    "T2": dict(col="dev_actual_drawal_gwh", level="entity", name="actual drawal"),
    "T3": dict(col="gen_act_gwh_total", level="entity", name="conventional generation", min_mean=1.0),
    "T4": dict(col="re_all_total_re_gwh", level="entity", name="RE generation", min_mean=1.0),
    "T5": dict(col="prc_dam_acp", level="bid_area", name="DAM price"),
}

ENTITY_EXOG = ["dem_energy_met_gwh", "dem_max_demand_mw", "dem_energy_shortage_gwh", "dev_drawal_schedule_gwh",
               "dev_od_ud_gwh", "dev_actual_drawal_gwh", "reg_wind_gwh", "reg_solar_gwh", "reg_hydro_gwh",
               "reg_energy_met_gwh", "nat_energy_met_gwh", "sys_pct_lt_49_9", "sys_fvi",
               "gen_act_gwh_total", "gen_prog_gwh_total", "gen_outage_share", "gen_act_gwh_hydro", "gen_coal_days_capw",
               "co2_ci_conv_op", "re_all_total_re_gwh", "re_all_wind_gwh", "re_all_solar_gwh",
               "prc_dam_acp", "prc_dsm_normal", "wx_tmax_c", "wx_tmin_c", "wx_rain_mm", "wx_cdd24"]
AREA_EXOG = ["prc_rtm_acp", "prc_dsm_normal", "dem_energy_met_gwh", "dev_actual_drawal_gwh", "reg_wind_gwh",
             "reg_solar_gwh", "nat_energy_met_gwh", "sys_pct_lt_49_9", "gen_outage_share", "wx_tmax_c", "wx_cdd24"]


def load_panel(allow_reserved=None):
    P = pd.read_parquet(os.path.join(PROC, "refused8_state_day.parquet"))
    if allow_reserved is None:
        allow_reserved = PHASE == "confirm"
    if allow_reserved:
        guard.require_confirmatory()
        return P
    return P[P.date < RESERVED_START].reset_index(drop=True)


def lags(instant=False):
    fr = pd.read_csv(os.path.join(DOCS, "FIELD_REGISTRY.csv"))
    L = dict(zip(fr.column, fr.avail_lag_days.fillna(0).astype(int)))
    return {k: 0 for k in L} if instant else L


def split_of(target_date):
    d = pd.to_datetime(target_date)
    return np.select([d <= TRAIN_END, d <= VAL_END, d <= DEV_END], ["train", "validation", "dev_test"], "beyond")


def series_frame(P, tid):
    """Daily frame with columns sid, date, y and exogenous columns, complete date range per series."""
    t = TARGETS[tid]
    if t["level"] == "entity":
        cols = sorted(set(ENTITY_EXOG) | {t["col"]})
        D = P[["entity", "date", "region"] + cols].rename(columns={"entity": "sid"})
    else:
        agg = {c: "mean" for c in ["prc_dam_acp", "prc_rtm_acp", "prc_dsm_normal", "reg_wind_gwh", "reg_solar_gwh",
                                   "nat_energy_met_gwh", "sys_pct_lt_49_9", "gen_outage_share", "wx_tmax_c", "wx_cdd24"]}
        agg.update({"dem_energy_met_gwh": lambda s: s.sum(min_count=1),
                    "dev_actual_drawal_gwh": lambda s: s.sum(min_count=1)})
        D = P.groupby(["bid_area", "date"]).agg({**agg, "region": "first"}).reset_index().rename(columns={"bid_area": "sid"})
    D = D.rename(columns={t["col"]: "y"}) if t["col"] not in (ENTITY_EXOG if t["level"] == "entity" else AREA_EXOG) \
        else D.assign(y=D[t["col"]])
    if tid == "T4":
        D.loc[D.date > pd.Timestamp("2025-11-18"), "y"] = np.nan
    if "min_mean" in t:
        m = D[D.date <= TRAIN_END].groupby("sid").y.mean()
        D = D[D.sid.isin(m[m >= t["min_mean"]].index)]
    return D.sort_values(["sid", "date"]).reset_index(drop=True)


def tabular_samples(P, tid, instant=False):
    """One row per (sid, issue day τ, H). Columns: keys, y, anchors (raw units), normalisation (base, s), z-target and
    features (normalised). Anchors: persistence (last available), seasonal naive (same weekday as the target, most
    recent available week), MA7 (mean of the last 7 available values)."""
    t = TARGETS[tid]
    L = lags(instant)
    Ly = L.get(t["col"], 1)
    D = series_frame(P, tid)
    exog = ENTITY_EXOG if t["level"] == "entity" else AREA_EXOG
    exog = [c for c in exog if c in D.columns]
    trn = D[D.date <= TRAIN_END]
    s_y = trn.groupby("sid").y.std().clip(lower=1e-3)
    mu_x = trn.groupby("sid")[exog].mean()
    sd_x = trn.groupby("sid")[exog].std().replace(0, np.nan)
    out = []
    for sid, g in D.groupby("sid", sort=True):
        g = g.set_index("date").asfreq("D")
        g["sid"] = sid
        tau = g.index
        f = {}
        y = g.y
        ylag = pd.concat({k: y.shift(Ly + k) for k in range(WINDOW)}, axis=1)  # value at τ − Ly − k
        base = ylag.iloc[:, :7].mean(axis=1)
        s = float(s_y[sid])
        for k in range(WINDOW):
            f[f"y_l{k}"] = (ylag[k] - base) / s
        f["y_base_z"] = (base - trn.loc[trn.sid == sid, "y"].mean()) / s
        f["y_std7"] = ylag.iloc[:, :7].std(axis=1) / s
        f["y_nobs14"] = ylag.notna().sum(axis=1)
        for c in exog:
            Lc = L.get(c, 1)
            xs = g[c].shift(Lc)
            m, sd = mu_x.loc[sid, c], sd_x.loc[sid, c]
            if not np.isfinite(sd):
                continue
            last, m7 = (xs - m) / sd, (xs.rolling(7, min_periods=3).mean() - m) / sd
            f[f"x_{c}_last"], f[f"x_{c}_m7"], f[f"x_{c}_d7"] = last, m7, last - m7
        f["cal_issue_dow"] = pd.Series(tau.dayofweek, index=tau)
        f = pd.DataFrame(f, index=tau)
        rows = []
        for H in HORIZONS:
            td = tau + pd.Timedelta(days=H)
            new = {}
            new["H"] = H
            new["target_date"] = td
            new["y"] = y.reindex(td).to_numpy()
            k = int(np.ceil((H + Ly) / 7))
            sn = y.reindex(td - pd.Timedelta(days=7 * k)).to_numpy()
            new["a_seasonal_naive"] = sn
            new["a_persistence"] = ylag[0].to_numpy()
            new["a_ma7"] = base.to_numpy()
            new["y_sn_z"] = (sn - base.to_numpy()) / s
            new["y_sn2_z"] = (y.reindex(td - pd.Timedelta(days=7 * (k + 1))).to_numpy() - base.to_numpy()) / s
            new["y_ly_z"] = (y.reindex(td - pd.Timedelta(days=364)).to_numpy() - base.to_numpy()) / s
            new["cal_dow"] = td.dayofweek
            new["cal_doy_sin"] = np.sin(2 * np.pi * td.dayofyear / 365.25)
            new["cal_doy_cos"] = np.cos(2 * np.pi * td.dayofyear / 365.25)
            new["cal_month"] = td.month
            new["base"] = base.to_numpy()
            new["s"] = s
            r = pd.concat([f, pd.DataFrame(new, index=tau)], axis=1)
            rows.append(r)
        r = pd.concat(rows)
        r.index.name = "issue_date"
        r = r.reset_index()
        r.insert(0, "sid", sid)
        r["region"] = g.region.dropna().iloc[0] if g.region.notna().any() else None
        out.append(r)
    S = pd.concat(out, ignore_index=True)
    S = S[S.target_date <= DEV_END]
    S["split"] = split_of(S.target_date)
    S["season"] = S.target_date.dt.month.map(SEASON)
    S["z"] = (S.y - S.base) / S.s
    S["tid"] = tid
    S = S[S.y_nobs14 >= 7]  # at least half of the target history available
    return S.reset_index(drop=True)


def feature_columns(S):
    return [c for c in S.columns if c.startswith(("y_l", "y_base", "y_std", "y_nobs", "y_sn", "y_ly", "x_", "cal_"))]


def mase_scales(S):
    tr = S[(S.split == "train") & S.y.notna() & S.a_seasonal_naive.notna()]
    e = (tr.y - tr.a_seasonal_naive)
    g = pd.DataFrame({"sid": tr.sid, "H": tr.H, "ae": e.abs(), "se": e ** 2}).groupby(["sid", "H"])
    return pd.DataFrame({"scale_mae": g.ae.mean(), "scale_mse": g.se.mean()}).reset_index()


def is_complete(tid, model, n_seeds=5):
    """True if the averaged prediction file exists and (for seeded models) its seed file holds n_seeds seeds."""
    from .paths import RES
    d = os.path.join(RES, PHASE_DIR, "o2", "preds")
    if not os.path.exists(os.path.join(d, f"{tid}__{model}.parquet")):
        return False
    if n_seeds is None:
        return True
    f = os.path.join(d, f"{tid}__{model}_seeds.parquet")
    return os.path.exists(f) and pd.read_parquet(f, columns=["seed"]).seed.nunique() >= n_seeds
