"""Build step 8: technical-validation statistics for the O1 dataset (inputs to the data descriptor).

Recomputes, from raw manifests, API listings, interim tables and the processed panel, every validation number reported
in `03_data/docs/SOURCE_REGISTRY.md` and the data descriptor, and writes them to `03_data/docs/validation.json`.
The pilot copy (raw/legacy_refused0, India Data Portal files) is used only as an external cross-check.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8 import entities as E  # noqa: E402
from refused8.paths import DOCS, INTERIM, PROC, RAW  # noqa: E402

WINDOW = ("2018-04-01", "2026-08-31")
BLIND_END = pd.Timestamp("2025-03-31")  # value comparisons (pilot copy, climatology) never use reserved-period values
V = {}


def manifests():
    out = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*", "MANIFEST.csv"))):
        m = pd.read_csv(f)
        m["status"] = m.status.astype(str)
        m = m.drop_duplicates("url", keep="last")
        src = os.path.basename(os.path.dirname(f))
        out[src] = dict(files_ok=int((m.status == "200").sum()), not_available=int((m.status != "200").sum()),
                        bytes_ok=int(m.loc[m.status == "200", "bytes"].sum()))
    leg = pd.read_csv(os.path.join(RAW, "legacy_refused0", "MANIFEST_LEGACY.csv"))
    out["legacy_refused0"] = dict(files_ok=len(leg), not_available=0, bytes_ok=int(leg.bytes.sum()))
    return out


def psp():
    st = pd.read_parquet(os.path.join(INTERIM, "psp_state_day.parquet"))
    log = pd.read_csv(os.path.join(INTERIM, "psp_parse_log.csv"))
    have = set(pd.DatetimeIndex(st.date.unique()))
    miss = [d for d in pd.date_range(*WINDOW) if d not in have]
    L = []
    for f in glob.glob(os.path.join(RAW, "grid_india_psp", "LISTING_psp_*.json")):
        L += json.load(open(f))
    li = pd.DataFrame(L)
    shared_titles = li.groupby("FilePath").Title_.transform("nunique") > 1
    li["created"] = pd.to_datetime(li.CreatedOn, format="%d-%m-%Y %H:%M", errors="coerce")
    li["rep"] = pd.to_datetime(li.Field1, format="%d-%m-%Y", errors="coerce")
    shared_paths = set(li.loc[shared_titles, "FilePath"])
    gap_shared = 0
    for dday in miss:  # report records for the reporting day dday + 1 (listing Field1 = reporting date)
        recs = li[li.rep == dday + pd.Timedelta(days=1)]
        if len(recs) and recs.FilePath.isin(shared_paths).all():
            gap_shared += 1
    recent = li[li.rep >= "2025-04-01"]
    lag_h = (recent.created - recent.rep).dt.total_seconds() / 3600
    # PDF vs XLS agreement on dates where both formats were parsed
    both = st.groupby("date").source_file.first()
    return dict(state_rows=len(st), dates=int(st.date.nunique()), first=str(st.date.min().date()),
                last=str(st.date.max().date()), entities=int(st.entity.nunique()),
                missing_dates_window=len(miss), missing_dates_shared_path=gap_shared,
                listing_records=len(li), listing_filepaths_shared_by_titles=int(li.loc[shared_titles, "FilePath"].nunique()),
                titles_on_shared_paths=int(shared_titles.sum()),
                largest_shared_path_titles=int(li.groupby("FilePath").Title_.nunique().max()),
                files_parsed_ok=int((log.status == "ok").sum()),
                files_few_states=int(log.status.str.startswith("few").sum()),
                files_not_psp=int((log.status == "not_psp_or_no_date").sum()),
                date_reassigned_by_cover_letter=int(log.date_reassigned_to.notna().sum()) if "date_reassigned_to" in log else 0,
                duplicate_uploads_dropped=int(log.dropped_duplicate_content.sum()) if "dropped_duplicate_content" in log else 0,
                publication_lag_hours_median=float(lag_h.median()), publication_lag_hours_p90=float(lag_h.quantile(0.9)),
                publication_lag_n=int(lag_h.notna().sum()), xls_share_of_chosen=float(both.str.lower().str.endswith(("xls", "xlsx")).mean()))


def dsm():
    d = pd.read_parquet(os.path.join(INTERIM, "dsm_daily.parquet"))
    li = pd.DataFrame(json.load(open(os.path.join(RAW, "grid_india_dsm", "LISTING_dsm.json"))))
    daily = li[li.Title_.str.match(r"DSM Rate \d{2}-\d{2}-\d{4}")].copy()
    daily["data_date"] = pd.to_datetime(daily.Title_.str[-10:], format="%d-%m-%Y", errors="coerce")
    daily["created"] = pd.to_datetime(daily.CreatedOn, format="%d-%m-%Y %H:%M", errors="coerce")
    lag = (daily.created - daily.data_date).dt.total_seconds() / 86400
    lag = lag[daily.data_date >= "2025-01-01"]
    reg = d.groupby("regime").date.agg(["min", "max", "nunique"])
    return dict(dates=int(d.date.nunique()), first=str(d.date.min().date()), last=str(d.date.max().date()),
                bid_areas=int(d[d.bid_area.str.match(r"^[NEWSA]\d$")].bid_area.nunique()),
                regimes={k: dict(first=str(v["min"].date()), last=str(v["max"].date()), dates=int(v["nunique"]))
                         for k, v in reg.iterrows()},
                daily_file_lag_days_median=float(lag.median()), daily_file_lag_days_min=float(lag.min()),
                daily_file_lag_days_max=float(lag.max()), daily_file_lag_n=int(lag.notna().sum()))


def npp():
    s = pd.read_parquet(os.path.join(INTERIM, "npp_state_day.parquet"))
    log = pd.read_csv(os.path.join(INTERIM, "npp_parse_log.csv"))
    miss = pd.date_range("2018-03-22", s.date.max()).difference(pd.DatetimeIndex(s.date.unique()))
    x = pd.read_csv(os.path.join(RAW, "legacy_refused0", "IDP", "daily-power-generation.csv"),
                    usecols=["date", "state_name", "todays_gen_act"])
    x["date"] = pd.to_datetime(x.date)
    x["entity"] = x.state_name.map(E.entity_id)
    a = x.dropna(subset=["entity"]).groupby(["date", "entity"]).todays_gen_act.sum().rename("idp")
    j = pd.concat([a, s.set_index(["date", "entity"]).act_gwh_total.rename("npp")], axis=1, join="inner")
    j = j[j.index.get_level_values(0) <= BLIND_END]
    d = (j.idp - j.npp).abs()
    stn = pd.read_parquet(os.path.join(INTERIM, "npp_station_day.parquet"), columns=["date"])
    return dict(station_rows=len(stn), state_rows=len(s), dates=int(s.date.nunique()), first=str(s.date.min().date()),
                last=str(s.date.max().date()), entities=int(s.entity.nunique()), missing_dates=len(miss),
                missing_2020_lockdown=int(((miss >= "2020-03-13") & (miss <= "2020-05-31")).sum()),
                files_no_state_totals=int((log.status == "no_state_totals").sum()),
                max_abs_station_vs_total_gwh=float(pd.to_numeric(log.max_abs_station_vs_total, errors="coerce").max()),
                idp_overlap_rows=len(j), idp_first=str(j.index.get_level_values(0).min().date()),
                idp_last=str(j.index.get_level_values(0).max().date()),
                idp_share_within_1pct=float((d <= 0.01 * j.npp.abs().clip(lower=1)).mean()),
                idp_share_within_0p01gwh=float((d <= 0.01).mean()))


def cea_re():
    r = pd.read_parquet(os.path.join(INTERIM, "cea_re_state_day.parquet"))
    log = pd.read_csv(os.path.join(INTERIM, "cea_re_parse_log.csv"))
    x = pd.read_csv(os.path.join(RAW, "legacy_refused0", "IDP", "daily-renewable-energy-generation.csv"))
    x["date"] = pd.to_datetime(x.date)
    x["entity"] = x.state_name.map(E.entity_id)
    x = x.dropna(subset=["entity"]).groupby(["date", "entity"]).total_renewable_energy.sum().rename("idp")
    j = r.set_index(["date", "entity"]).join(x, how="inner").dropna(subset=["all_total_re_gwh", "idp"])
    j = j[j.index.get_level_values(0) <= BLIND_END]
    d = (j.all_total_re_gwh - j.idp).abs()
    lay = pd.concat([r.all_layout.value_counts(), r.ctrl_layout.value_counts()], axis=1, keys=["all", "ctrl"]).fillna(0)
    return dict(rows=len(r), dates=int(r.date.nunique()), first=str(r.date.min().date()), last=str(r.date.max().date()),
                ctrl_first=str(r.loc[r.ctrl_total_re_gwh.notna(), "date"].min().date()),
                ctrl_dates=int(r.loc[r.ctrl_total_re_gwh.notna(), "date"].nunique()), entities=int(r.entity.nunique()),
                files=len(log), date_mismatch_files=int(log.date_mismatch.sum()),
                duplicate_table_files=int((log.status == "duplicate_tables").sum()),
                inconsistent_rows_all=int(r.all_sum_flag.sum()), inconsistent_rows_ctrl=int(r.ctrl_sum_flag.sum()),
                layout_counts={k: {c: int(v) for c, v in row.items()} for k, row in lay.iterrows()},
                idp_overlap_rows=len(j), idp_share_within_1pct=float((d <= 0.01 * j.idp.abs().clip(lower=1)).mean()))


def co2():
    c = pd.read_parquet(os.path.join(INTERIM, "co2_ef_state_fuel_fy.parquet"))
    coal = c[(c.fuel == "coal") & (c.netgen_gwh > 1000)]
    full = c[(c.npp_days >= 360) & (c.netgen_gwh > 0) & (c.npp_gross_gwh > 0)]
    ratio = full.groupby("fuel").apply(lambda g: g.netgen_gwh.sum() / g.npp_gross_gwh.sum(), include_groups=False)
    nat = c[c.fuel == "coal"].groupby("fy").apply(lambda g: g.emis_t.sum() / g.netgen_gwh.sum() / 1000, include_groups=False)
    return dict(cells=len(c), entities=int(c.entity.nunique()), net_gross_ratio={k: float(v) for k, v in ratio.items()},
                coal_ef_gross_min=float(coal.ef_gross.min()), coal_ef_gross_max=float(coal.ef_gross.max()),
                coal_ef_gross_median=float(coal.ef_gross.median()),
                national_coal_ef_net={k: float(v) for k, v in nat.dropna().items()})


def imd():
    w = pd.read_parquet(os.path.join(INTERIM, "imd_state_day.parquet"))
    clim = w[w.date <= BLIND_END].assign(m=lambda d: d.date.dt.month).groupby(["entity", "m"]).tmax_c.mean()
    rain = np.fromfile(os.path.join(RAW, "imd_gridded", "rain", "rain_2023.grd"), dtype="<f4").reshape(-1, 129, 135)
    mumbai = float(rain[181:212, int(round((19.0 - 6.5) / 0.25)), int(round((72.75 - 66.5) / 0.25))].sum())
    return dict(rows=len(w), first=str(w.date.min().date()), last=str(w.date.max().date()),
                missing_share=float(w[["tmax_c", "tmin_c", "rain_mm"]].isna().mean().max()),
                delhi_tmax_jan=float(clim.loc[("DL", 1)]), delhi_tmax_may=float(clim.loc[("DL", 5)]),
                mumbai_cell_rain_jul2023_mm=mumbai)


def panel():
    P = pd.read_parquet(os.path.join(PROC, "refused8_state_day.parquet"))
    fr = pd.read_csv(os.path.join(DOCS, "FIELD_REGISTRY.csv"))
    ck = open(os.path.join(PROC, "CHECKSUM.txt")).read().split("content_sha256=")[1].strip()
    qc = {c: int(P[c].sum()) for c in P.columns if c.startswith("qc_")}
    blocks = fr.groupby("block").agg(train_dev=("missing_share_train_to_devtest", "mean"),
                                     reserved=("missing_share_reserved", "mean"))
    return dict(rows=len(P), columns=P.shape[1], entities=int(P.entity.nunique()), first=str(P.date.min().date()),
                last=str(P.date.max().date()), days=int(P.date.nunique()), nonmissing_values=int(P.notna().sum().sum()),
                rows_with_energy_met=int(P.dem_energy_met_gwh.notna().sum()),
                splits={k: int(v) for k, v in P.split.value_counts().items()}, qc_flag_counts=qc,
                blocks_missing={k: dict(train_dev=float(v.train_dev), reserved=float(v.reserved)) for k, v in blocks.iterrows()},
                block_column_counts={k: int(v) for k, v in fr.block.value_counts().items()}, content_sha256=ck)


def main():
    V["manifests"] = manifests()
    V["raw_total_bytes_official"] = int(sum(v["bytes_ok"] for k, v in V["manifests"].items() if k != "legacy_refused0"))
    V["raw_files_ok_official"] = int(sum(v["files_ok"] for k, v in V["manifests"].items() if k != "legacy_refused0"))
    for name, fn in [("psp", psp), ("dsm", dsm), ("npp", npp), ("cea_re", cea_re), ("co2", co2), ("imd", imd),
                     ("panel", panel)]:
        V[name] = fn()
        print(name, "ok", flush=True)
    json.dump(V, open(os.path.join(DOCS, "validation.json"), "w"), indent=1)
    print(json.dumps({k: V[k] for k in ("raw_total_bytes_official", "raw_files_ok_official")}, indent=1))


if __name__ == "__main__":
    main()
