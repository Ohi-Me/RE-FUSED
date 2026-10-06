"""Build step 7: the RE-FUSED-8 State-day panel (proposal O1).

Inputs : interim tables from b01–b06 (Grid-India PSP and DSM; CEA NPP DGR2, daily RE, CO2 database; IMD gridded)
Outputs: 03_data/processed/refused8_state_day.parquet  one row per (entity, date), 2018-04-01 → 2026-08-31, 34 entities
         03_data/processed/refused8_entities.csv       entity registry (region, DSM bid area, pilot-18 flag)
         03_data/docs/FIELD_REGISTRY.csv             every column: block, source, unit, availability lag, QC rule
         03_data/docs/QC_REPORT.md                   plausibility exclusions, missingness, cross-source checks
         03_data/processed/CHECKSUM.txt              content hash of the panel (row-order independent)
Rules  : source tables are never modified here; values failing a plausibility rule are set to missing in the panel
         and counted in `qc_*` flags. Targets and consequence outcomes are never imputed. Structural zeros (a fuel with
         no monitored capacity in the State on a reporting date) are written as 0.
Blinding: the QC report prints only counts and shares for dates ≥ 2025-04-01 (no target statistics).
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8 import entities as E  # noqa: E402
from refused8.paths import DOCS, INTERIM, PROC  # noqa: E402

START, END = pd.Timestamp("2018-04-01"), pd.Timestamp("2026-08-31")
RESERVED = pd.Timestamp("2025-04-01")
FUELS = ["coal", "gas", "diesel", "hydro", "nuclear"]

# field registry: column -> (block, source, unit, availability lag in days after the data date, note)
REG = {}


def reg(cols, block, source, unit, lag, note=""):
    for c in ([cols] if isinstance(cols, str) else cols):
        REG[c] = dict(column=c, block=block, source=source, unit=unit, avail_lag_days=lag, note=note)


def fy_start_year(d):
    return np.where(d.dt.month >= 4, d.dt.year, d.dt.year - 1)


def rolling_median(s, by, window=29):
    return s.groupby(by).transform(lambda x: x.rolling(window, center=True, min_periods=10).median())


def main():
    ent = E.table()
    idx = pd.MultiIndex.from_product([ent.entity, pd.date_range(START, END)], names=["entity", "date"])
    P = pd.DataFrame(index=idx).reset_index().merge(ent[["entity", "region", "bid_area", "pilot18"]], on="entity")
    qc = {}

    # ---------------------------------------------------------------- G1 PSP: State demand and deviation
    ps = pd.read_parquet(os.path.join(INTERIM, "psp_state_day.parquet"))
    ps = ps.rename(columns={"energy_met_gwh": "dem_energy_met_gwh", "max_demand_mw": "dem_max_demand_mw",
                            "peak_shortage_mw": "dem_peak_shortage_mw", "energy_shortage_gwh": "dem_energy_shortage_gwh",
                            "drawal_schedule_gwh": "dev_drawal_schedule_gwh", "od_ud_gwh": "dev_od_ud_gwh",
                            "max_od_mw": "dev_max_od_mw", "max_ud_mw": "dev_max_ud_mw"})
    cols = [c for c in ps.columns if c.startswith(("dem_", "dev_"))]
    P = P.merge(ps[["entity", "date"] + cols], on=["entity", "date"], how="left")
    P["avail_psp"] = P.dem_energy_met_gwh.notna()
    reg(["dem_energy_met_gwh", "dem_energy_shortage_gwh"], "demand", "Grid-India PSP §C", "GWh", 1)
    reg(["dem_max_demand_mw", "dem_peak_shortage_mw"], "demand", "Grid-India PSP §C", "MW", 1,
        "peak shortage = shortage at the time of maximum demand")
    reg(["dev_drawal_schedule_gwh", "dev_od_ud_gwh"], "deviation", "Grid-India PSP §C", "GWh", 1,
        "OD(+)/UD(−) = actual − scheduled drawal, daily net")
    reg(["dev_max_od_mw", "dev_max_ud_mw"], "deviation", "Grid-India PSP §C", "MW", 1, "max UD printed from 2023")
    P = P.sort_values(["entity", "date"]).reset_index(drop=True)
    med_em = rolling_median(P.dem_energy_met_gwh, P.entity)
    med_md = rolling_median(P.dem_max_demand_mw, P.entity)
    rules = {
        "qc_energy_met": (P.dem_energy_met_gwh <= 0) & (med_em > 1) | (P.dem_energy_met_gwh > 5 * med_em.clip(lower=1)),
        "qc_max_demand": (P.dem_max_demand_mw > 5 * med_md.clip(lower=20)) | (P.dem_max_demand_mw < 0)
                         | (P.dem_energy_met_gwh * 1000 / 24 > 1.5 * P.dem_max_demand_mw.clip(lower=1)),
        "qc_energy_shortage": (P.dem_energy_shortage_gwh < 0)
                              | (P.dem_energy_shortage_gwh > P.dem_energy_met_gwh.clip(lower=0) + 1),
        "qc_od_ud": P.dev_od_ud_gwh.abs() > P.dem_energy_met_gwh.clip(lower=0) + P.dev_drawal_schedule_gwh.abs() + 1,
    }
    target = {"qc_energy_met": ["dem_energy_met_gwh"], "qc_max_demand": ["dem_max_demand_mw"],
              "qc_energy_shortage": ["dem_energy_shortage_gwh"], "qc_od_ud": ["dev_od_ud_gwh"]}
    for k, m in rules.items():
        m = m.fillna(False)
        P[k] = m
        for c in target[k]:
            P.loc[m, c] = np.nan
        qc[k] = int(m.sum())
    # actual drawal = schedule + OD/UD
    P["dev_actual_drawal_gwh"] = P.dev_drawal_schedule_gwh + P.dev_od_ud_gwh
    P["dev_abs_od_ud_gwh"] = P.dev_od_ud_gwh.abs()
    reg(["dev_actual_drawal_gwh", "dev_abs_od_ud_gwh"], "deviation", "derived (PSP)", "GWh", 1)
    reg(list(rules), "qc", "b07 rules", "flag", 1, "value set to missing when flagged")

    # ---------------------------------------------------------------- G1 PSP: regional and national context
    rg = pd.read_parquet(os.path.join(INTERIM, "psp_region_day.parquet"))
    items = {"wind_gen_gwh": "reg_wind_gwh", "solar_gen_gwh": "reg_solar_gwh", "hydro_gen_gwh": "reg_hydro_gwh",
             "energy_met_gwh": "reg_energy_met_gwh", "energy_shortage_gwh": "reg_energy_shortage_gwh",
             "ir_odud_gwh": "reg_ir_od_ud_gwh", "outage_total_mw": "reg_outage_total_mw"}
    r = rg[rg.item.isin(items) & (rg.section.isin(["A", "E", "F"]))].pivot_table(
        index=["date", "region"], columns="item", values="value", aggfunc="first").rename(columns=items).reset_index()
    nat = r[r.region == "ALL"].drop(columns="region").rename(columns=lambda c: c.replace("reg_", "nat_"))
    P = P.merge(r[r.region != "ALL"], on=["date", "region"], how="left").merge(nat, on="date", how="left")
    reg([c for c in P.columns if c.startswith(("reg_", "nat_")) and c.endswith("gwh")], "regional", "Grid-India PSP §A/§E",
        "GWh", 1, "region of the entity / All-India")
    reg([c for c in P.columns if c.startswith(("reg_", "nat_")) and c.endswith("_mw")], "regional", "Grid-India PSP §F",
        "MW", 1)
    fq = pd.read_parquet(os.path.join(INTERIM, "psp_freq_day.parquet")).drop(columns="source_file")
    bands = ["pct_lt_49_7", "pct_49_7_49_8", "pct_49_8_49_9", "pct_49_9_50_05", "pct_gt_50_05"]
    bad = (fq[bands + ["pct_lt_49_9"]].lt(0) | fq[bands + ["pct_lt_49_9"]].gt(100)).any(axis=1) \
        | (fq[bands].sum(axis=1) - 100).abs().gt(2) | ~fq.fvi.between(0, 5)
    qc["qc_frequency_days"] = int(bad.sum())
    fq.loc[bad, [c for c in fq.columns if c != "date"]] = np.nan
    fq = fq.rename(columns={c: "sys_" + c for c in fq.columns if c != "date"})
    P = P.merge(fq, on="date", how="left")
    reg([c for c in P.columns if c.startswith("sys_")], "system stress", "Grid-India PSP §B", "% of time / index", 1,
        "All-India; day excluded when bands do not sum to 100 ± 2 or FVI ∉ [0, 5]")

    # ---------------------------------------------------------------- G2 prices by bid area
    ds = pd.read_parquet(os.path.join(INTERIM, "dsm_daily.parquet"))
    acp = ds[ds.series.isin(["acp_daily", "dam_acp"])].groupby(["date", "bid_area"]).agg(
        prc_dam_acp=("mean", "mean"), prc_dam_acp_max=("max", "max"), prc_regime=("regime", "first"))
    rtm = ds[ds.series == "rtm_acp"].groupby(["date", "bid_area"]).agg(prc_rtm_acp=("mean", "mean"))
    dsm = ds[ds.series.isin(["charge_at_50_00", "normal_rate"])].groupby(["date", "bid_area"]).agg(
        prc_dsm_normal=("mean", "mean"), prc_dsm_normal_max=("max", "max"), prc_dsm_cap_share=("cap_share", "mean"))
    pr = acp.join(rtm, how="outer").join(dsm, how="outer").reset_index()
    badp = pr[["prc_dam_acp", "prc_rtm_acp", "prc_dsm_normal"]].le(0) | pr[["prc_dam_acp", "prc_rtm_acp", "prc_dsm_normal"]].gt(50000)
    qc["qc_price_values"] = int(badp.sum().sum())
    for c in badp:
        pr.loc[badp[c], c] = np.nan
    P = P.merge(pr, on=["date", "bid_area"], how="left")
    reg(["prc_dam_acp", "prc_dam_acp_max", "prc_rtm_acp"], "price", "Grid-India DSM files", "₹/MWh", 14,
        "DAM/RTM area clearing price of the entity's bid area (R1: daily average ACP column); Grid-India uploads the "
        "files in batches 8–11 days after the date, so features use a 14-day lag")
    reg(["prc_dsm_normal", "prc_dsm_normal_max", "prc_dsm_cap_share"], "price", "Grid-India DSM files", "₹/MWh / share",
        14, "R1: charge for deviation at 50.00 Hz; R2/R3: normal rate of charges for deviation")
    reg("prc_regime", "price", "Grid-India DSM files", "category", 14, "R1 ≤ 2022-12-04; R2; R3 ≥ 2024-09-16")

    # ---------------------------------------------------------------- C1 NPP conventional generation
    n = pd.read_parquet(os.path.join(INTERIM, "npp_state_day.parquet"))
    ncols = {c: "gen_" + c for c in n.columns if c not in ("date", "entity")}
    n = n.rename(columns=ncols)
    P = P.merge(n, on=["entity", "date"], how="left")
    npp_dates = set(n.date)
    P["avail_npp"] = P.date.isin(npp_dates)
    for f in FUELS:
        for v in ["act_gwh", "prog_gwh", "cap_mw", "outage_mw"]:
            c = f"gen_{v}_{f}"
            if c in P:
                P.loc[P.avail_npp & P[c].isna(), c] = 0.0  # structural zero: no monitored capacity of this fuel
    for tot in ["act_gwh_total", "prog_gwh_total", "cap_mw_total", "outage_mw_total"]:
        P.loc[P.avail_npp & P[f"gen_{tot}"].isna(), f"gen_{tot}"] = 0.0
    # State generation above capacity × 24 h with a 15 % overload margin (hydro units routinely run ~10 % above
    # rating), or an isolated station typo flagged in b03: actual generation of that State-day is set to missing
    badg = P.gen_act_gwh_total > P.gen_cap_mw_total * 24 / 1000 * 1.15 + 0.5
    P["qc_npp_capacity"] = badg.fillna(False)
    P["qc_npp_station_typo"] = P.gen_station_typo_n.fillna(0) > 0
    bad_gen = P.qc_npp_capacity | P.qc_npp_station_typo
    P.loc[bad_gen, [c for c in P.columns if c.startswith("gen_act_gwh")] ] = np.nan
    qc["qc_npp_capacity"] = int(P.qc_npp_capacity.sum())
    qc["qc_npp_station_typo"] = int(P.qc_npp_station_typo.sum())
    P = P.drop(columns=["gen_station_typo_n", "gen_station_typo_gwh"])
    reg(["qc_npp_capacity", "qc_npp_station_typo"], "qc", "b03/b07 rules", "flag", 2,
        "State actual generation set to missing")
    P["gen_outage_share"] = np.where(P.gen_cap_mw_total > 0, P.gen_outage_mw_total / P.gen_cap_mw_total, np.nan)
    P["gen_conv_dev_gwh"] = P.gen_act_gwh_total - P.gen_prog_gwh_total  # proposal's generation deviation (secondary)
    reg([c for c in P.columns if c.startswith("gen_") and "gwh" in c], "conventional generation", "CEA NPP DGR2", "GWh", 2,
        "gross; stations monitored by CEA (≥ 25 MW); published d+1 evening, later at weekends")
    reg([c for c in P.columns if c.startswith("gen_") and c.endswith(tuple(f"mw_{f}" for f in FUELS + ["total"]))],
        "conventional generation", "CEA NPP DGR2", "MW", 2)
    reg(["gen_coal_days_capw", "gen_coal_cap_share_lt7d", "gen_coal_cap_reporting_mw", "gen_outage_share"],
        "system stress", "CEA NPP DGR2", "days / share / MW", 2)

    # ---------------------------------------------------------------- C5 CEA RE
    re_ = pd.read_parquet(os.path.join(INTERIM, "cea_re_state_day.parquet"))
    keep = [c for c in re_.columns if c.startswith(("all_", "ctrl_")) and c.endswith("gwh")]
    re_ = re_[["entity", "date"] + keep].rename(columns={c: "re_" + c for c in keep})
    P = P.merge(re_, on=["entity", "date"], how="left")
    P["avail_re"] = P.re_all_total_re_gwh.notna()
    reg([c for c in P.columns if c.startswith("re_all")], "renewables", "CEA daily RE report (State+ISGS)", "GWh", 2,
        "published to 2025-11-18")
    reg([c for c in P.columns if c.startswith("re_ctrl")], "renewables", "CEA daily RE report (State control area)",
        "GWh", 2, "from 2020-06-01 to 2025-11-18")

    # ---------------------------------------------------------------- C4 carbon layer (GCAL)
    ef = pd.read_parquet(os.path.join(INTERIM, "co2_ef_state_fuel_fy.parquet"))
    ef["fy0"] = ef.fy.str[:4].astype(int)
    first = ef.dropna(subset=["ef_gross"]).groupby(["entity", "fuel"]).fy0.min()
    P["fy0"] = fy_start_year(P.date)
    for lag, tag in [(2, "op"), (1, "acc")]:
        emis = np.zeros(len(P))
        filled = np.zeros(len(P), dtype=bool)
        anyval = np.zeros(len(P), dtype=bool)
        for f in ["coal", "gas", "diesel"]:
            e = ef[ef.fuel == f][["entity", "fy0", "ef_gross"]].dropna()
            key = pd.DataFrame({"entity": P.entity, "fy0": P.fy0 - lag})
            fb = first.xs(f, level="fuel") if f in first.index.get_level_values("fuel") else pd.Series(dtype=float)
            key["fy_used"] = np.maximum(key.fy0, key.entity.map(fb).fillna(key.fy0))  # earliest FY if FY−lag not in v22
            m = key.merge(e.rename(columns={"fy0": "fy_used"}), on=["entity", "fy_used"], how="left")
            factor = m.ef_gross.to_numpy()
            gen = P[f"gen_act_gwh_{f}"].to_numpy()
            contrib = np.where(np.isnan(factor), 0.0, gen * factor * 1000)
            emis = emis + np.nan_to_num(contrib)
            filled |= (key.fy_used != key.fy0).to_numpy() & (np.nan_to_num(gen) > 0)
            anyval |= ~np.isnan(gen)
        P[f"co2_emis_t_{tag}"] = np.where(anyval & P.avail_npp, emis, np.nan)
        P[f"qc_co2_factor_filled_{tag}"] = filled
    P["co2_ci_conv_op"] = np.where(P.gen_act_gwh_total > 0, P.co2_emis_t_op / (P.gen_act_gwh_total * 1000), np.nan)
    gen_all = P.gen_act_gwh_total + P.re_all_total_re_gwh
    P["co2_ci_gen_acc"] = np.where(gen_all > 0, P.co2_emis_t_acc / (gen_all * 1000), np.nan)
    # consumption-based intensity: imports at the region's generation intensity of the day, exports at the State's own
    g = P.groupby(["region", "date"])
    reg_ci = (g.co2_emis_t_acc.transform("sum") / (g.gen_act_gwh_total.transform("sum")
                                                    + g.re_all_total_re_gwh.transform("sum")) / 1000)
    d = P.dev_actual_drawal_gwh
    e_cons = P.co2_emis_t_acc + np.where(d > 0, d * reg_ci * 1000, d * P.co2_ci_gen_acc * 1000)
    P["co2_ci_cons_acc"] = np.where(P.dem_energy_met_gwh > 0, e_cons / (P.dem_energy_met_gwh * 1000), np.nan)
    reg(["co2_emis_t_op", "co2_emis_t_acc"], "carbon", "CEA CO2 database v22 × CEA NPP DGR2", "t CO2", 2,
        "op: factor of FY−2 (published before the date); acc: FY−1 (ex-post accounting)")
    reg(["co2_ci_conv_op", "co2_ci_gen_acc", "co2_ci_cons_acc"], "carbon", "derived", "t CO2/MWh", 2,
        "conv: conventional generation only; gen: incl. CEA RE (to 2025-11-18); cons: consumption-based with net drawal")
    reg(["qc_co2_factor_filled_op", "qc_co2_factor_filled_acc"], "qc", "b07", "flag", 2,
        "earliest available FY factor used (FY2017-18) because FY−lag precedes the database")

    # ---------------------------------------------------------------- I1 weather
    w = pd.read_parquet(os.path.join(INTERIM, "imd_state_day.parquet"))
    w = w.rename(columns={c: "wx_" + c for c in w.columns if c not in ("date", "entity")})
    badw = w.wx_tmax_c < w.wx_tmin_c
    qc["qc_weather_tmax_lt_tmin"] = int(badw.sum())
    w.loc[badw, ["wx_tmax_c", "wx_tmin_c", "wx_tmean_c", "wx_cdd24", "wx_hdd18"]] = np.nan
    P = P.merge(w.drop(columns=["wx_n_temp_cells", "wx_n_rain_cells"]), on=["entity", "date"], how="left")
    reg([c for c in P.columns if c.startswith("wx_")], "weather", "IMD Pune gridded (load centres)", "°C / mm", 1,
        "observed weather; IMD final gridded product ends 2025-12-31")

    # ---------------------------------------------------------------- calendar and availability
    P["cal_dow"] = P.date.dt.dayofweek
    P["cal_month"] = P.date.dt.month
    P["cal_doy"] = P.date.dt.dayofyear
    P["cal_fy"] = P.fy0
    P["split"] = np.select([P.date < "2023-04-01", P.date < "2024-04-01", P.date < RESERVED],
                           ["train", "validation", "dev_test"], "reserved")
    reg(["cal_dow", "cal_month", "cal_doy", "cal_fy"], "calendar", "derived", "integer", 0)
    reg(["avail_psp", "avail_npp", "avail_re"], "availability", "derived", "flag", 0)
    reg(["entity", "date", "region", "bid_area", "pilot18", "split"], "key", "registry", "-", 0)
    P = P.drop(columns=["fy0"])

    # ---------------------------------------------------------------- write
    os.makedirs(PROC, exist_ok=True)
    P = P.sort_values(["entity", "date"]).reset_index(drop=True)
    P.to_parquet(os.path.join(PROC, "refused8_state_day.parquet"), index=False)
    ent.to_csv(os.path.join(PROC, "refused8_entities.csv"), index=False)
    fr = pd.DataFrame([REG.get(c, dict(column=c, block="?", source="?", unit="?", avail_lag_days=np.nan, note=""))
                       for c in P.columns])
    fr["missing_share_train_to_devtest"] = [float(P.loc[P.split != "reserved", c].isna().mean()) for c in P.columns]
    fr["missing_share_reserved"] = [float(P.loc[P.split == "reserved", c].isna().mean()) for c in P.columns]
    fr.to_csv(os.path.join(DOCS, "FIELD_REGISTRY.csv"), index=False)
    # content hash of the panel as read back from the parquet file (the in-memory frame can hold NaN payloads with
    # other bit patterns, which the file round trip normalises; users can only verify the file as read)
    h = hashlib.sha256(pd.util.hash_pandas_object(pd.read_parquet(os.path.join(PROC, "refused8_state_day.parquet")),
                                                  index=False).values.tobytes()).hexdigest()
    with open(os.path.join(PROC, "CHECKSUM.txt"), "w") as fh:
        fh.write(f"refused8_state_day.parquet rows={len(P)} cols={P.shape[1]} content_sha256={h}\n")
    print("panel", P.shape, "checksum", h[:16])
    print("unregistered columns:", fr.loc[fr.block == "?", "column"].tolist())
    print("QC exclusions:", qc)
    write_qc_report(P, fr, qc)


def write_qc_report(P, fr, qc):
    dev = P[P.split != "reserved"]
    lines = ["# RE-FUSED-8 panel QC report", "",
             f"Panel: {len(P):,} rows × {P.shape[1]} columns; {P.entity.nunique()} entities; "
             f"{P.date.min().date()} → {P.date.max().date()}. Reserved period (≥ {RESERVED.date()}): counts only.", "",
             "## Plausibility exclusions (values set to missing)", "", "| rule | rows/days |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in qc.items()]
    lines += ["", "## Source availability by split (share of entity-days)", "", "| split | PSP | NPP | CEA RE |",
              "|---|---|---|---|"]
    for s, g in P.groupby("split", sort=False):
        lines.append(f"| {s} | {g.avail_psp.mean():.3f} | {g.avail_npp.mean():.3f} | {g.avail_re.mean():.3f} |")
    # cross-source check on development data only: energy met vs in-State generation + actual drawal
    x = dev.dropna(subset=["dem_energy_met_gwh", "gen_act_gwh_total", "re_all_total_re_gwh", "dev_actual_drawal_gwh"])
    x = x[x.dem_energy_met_gwh > 5]
    ratio = (x.gen_act_gwh_total + x.re_all_total_re_gwh + x.dev_actual_drawal_gwh) / x.dem_energy_met_gwh
    by = ratio.groupby(x.entity).median().round(2)
    lines += ["", "## Cross-source check (train–dev_test only)", "",
              "Ratio (NPP conventional gross + CEA RE + PSP actual drawal) / PSP energy met, median by entity. Values "
              "above 1 are expected where in-State generation is gross and part of it is scheduled to other States "
              "through ISGS or bilateral contracts; values far from 1 identify entities whose generation is not "
              "represented by the in-State reports (for use as a documented limitation, not as an exclusion). Days with energy "
              "met ≤ 5 GWh are omitted, so small north-eastern States have few or no rows; DVC is negative because its "
              "stations are listed under West Bengal and Jharkhand in the NPP report.", "",
              "| entity | median ratio | n |", "|---|---|---|"]
    lines += [f"| {e} | {v} | {int((x.entity == e).sum())} |" for e, v in by.items()]
    lines += ["", "## Missingness by block (share of cells missing)", "", "| block | train–dev_test | reserved |",
              "|---|---|---|"]
    for b, g in fr.groupby("block"):
        lines.append(f"| {b} | {g.missing_share_train_to_devtest.mean():.3f} | {g.missing_share_reserved.mean():.3f} |")
    # longest run of missing target values per entity (development data)
    runs = []
    for e, g in dev.groupby("entity"):
        m = g.dem_energy_met_gwh.isna().to_numpy()
        best = cur = 0
        for v in m:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        runs.append((e, best))
    lines += ["", "## Longest consecutive gap in energy met (train–dev_test)", "",
              ", ".join(f"{e}: {r}" for e, r in runs), ""]
    open(os.path.join(DOCS, "QC_REPORT.md"), "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
