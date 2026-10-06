"""Build step 9: the 15-minute all-India series from the "TimeSeries" sheet of Grid-India's Daily PSP reports.

Grid-India added this sheet to the Daily Power Supply Position report on 4 November 2024. It gives, for each
15-minute block of the day, the instantaneous all-India SCADA values of frequency, demand met, nuclear, wind, solar,
hydro (excluding Bhutan), gas, thermal and other generation, net demand met (demand met minus wind and solar), total
generation and net transnational exchange (import positive).

Inputs : data/raw/grid_india_psp/<FY>/*.xls (MANIFEST.csv lists every file with its URL and SHA-256)
Outputs: data/processed/refused_allindia_15min.parquet  one row per 15-minute block
         data/processed/refused_allindia_hourly.parquet  hourly means of the blocks (at least 3 valid blocks)
         data/interim/psp_timeseries_parse_log.csv       one row per file: status, data date, reporting date
         data/docs/ALLINDIA_15MIN_QC.md                  coverage and quality-control summary
Rules  : the sheet carries its "Date of Reporting"; the data date is the day before it (the report for day D is
         published on D+1). It must agree with the date in the file name; files that disagree are logged and the
         sheet's own date is used. One report per data date: identical content is used once; otherwise the larger
         file is kept and the choice is logged. Quality flags, all computed per column without looking at any later
         period: missing block, non-numeric value, value outside a physical range, and "frozen" telemetry (the same
         value for 8 or more consecutive blocks while the column is normally varying). Flagged values are set to
         missing in the hourly table; the 15-minute table keeps the reported value and the flag.
Usage  : python codes/scripts/build/b09_psp_timeseries.py   (run on the H100: python run_all.py stage build_b09_timeseries)
"""
import datetime as dt
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused.paths import DATA, INTERIM, PROC, RAW  # noqa: E402

SRC = os.path.join(RAW, "grid_india_psp")
COLS = ["freq_hz", "demand_met_mw", "nuclear_mw", "wind_mw", "solar_mw", "hydro_mw", "gas_mw", "thermal_mw",
        "others_mw", "net_demand_met_mw", "total_generation_mw", "net_exchange_mw"]
# physical plausibility ranges for all-India values (MW, Hz); wide on purpose, they only catch garbage values
RANGE = {"freq_hz": (49.0, 51.0), "demand_met_mw": (80_000, 400_000), "nuclear_mw": (0, 15_000),
         "wind_mw": (0, 60_000), "solar_mw": (0, 200_000), "hydro_mw": (0, 80_000), "gas_mw": (0, 30_000),
         "thermal_mw": (50_000, 350_000), "others_mw": (0, 30_000), "net_demand_met_mw": (40_000, 400_000),
         "total_generation_mw": (80_000, 450_000), "net_exchange_mw": (-20_000, 20_000)}
FREEZE_BLOCKS = 8
# header patterns (upper case); order matters where one header contains another
HEADERS = [("storage_demand_mw", r"^STORAGE DEMAND"), ("storage_gen_mw", r"^STORAGE \("),
           ("net_demand_met_mw", r"^NET DEMAND"), ("net_exchange_mw", r"TRANS\w*TIONAL"),
           ("freq_hz", r"^FREQUENCY"), ("demand_met_mw", r"^DEMAND MET"), ("nuclear_mw", r"^NUCLEAR"),
           ("wind_mw", r"^WIND"), ("solar_mw", r"^SOLAR"), ("hydro_mw", r"^HYDRO"), ("gas_mw", r"^GAS"),
           ("thermal_mw", r"^THERMAL"), ("others_mw", r"^OTHERS"), ("total_generation_mw", r"^TOTAL GENERATION")]
# present only from 1 June 2026, when Grid-India began to report pumped-storage and battery storage separately;
# from then "demand met" includes storage load and "hydro" excludes pumped-storage generation
OPTIONAL = ["storage_demand_mw", "storage_gen_mw"]
RANGE.update({"storage_demand_mw": (0, 30_000), "storage_gen_mw": (0, 30_000)})


def file_date(name):
    m = re.match(r"(\d\d)\.(\d\d)\.(\d\d)", name)
    return dt.date(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


def work(path):
    rel = os.path.relpath(path, SRC).replace("\\", "/")
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        x = pd.ExcelFile(path)
        if "TimeSeries" not in x.sheet_names:
            return dict(file=rel, status="no_timeseries_sheet", sha256=sha), None
        d = pd.read_excel(x, "TimeSeries", header=None)
    except Exception as e:  # noqa: BLE001
        return dict(file=rel, status=f"error:{type(e).__name__}", sha256=sha), None
    rep = None
    for v in d.iloc[0].tolist():
        if isinstance(v, (dt.datetime, pd.Timestamp)):
            rep = pd.Timestamp(v).date()
        elif isinstance(v, str):
            try:
                rep = pd.to_datetime(v.strip(), format="%d-%b-%Y").date()
            except ValueError:
                pass
    head = [re.sub(r"\s+", " ", str(x)).upper() for x in d.iloc[2].tolist()]
    if not head or "TIME" not in head[0]:
        return dict(file=rel, status="unexpected_layout", sha256=sha), None
    # map columns by their header, not their position: from 1 June 2026 Grid-India added storage columns
    pos = {}
    for i, h in enumerate(head):
        for key, pat in HEADERS:
            if key not in pos and re.search(pat, h):
                pos[key] = i
                break
    missing = [c for c in COLS if c not in pos]
    if missing:
        return dict(file=rel, status="unexpected_layout:" + ",".join(missing), sha256=sha), None
    body = d.iloc[4:].copy()
    body = body[body[0].astype(str).str.match(r"^\s*\d{1,2}:\d{2}")]
    if len(body) == 0:
        return dict(file=rel, status="no_rows", sha256=sha), None
    hm = body[0].astype(str).str.extract(r"^\s*(\d{1,2}):(\d{2})")
    out = pd.DataFrame({"time": hm[0].str.zfill(2) + ":" + hm[1]})
    for c in COLS + OPTIONAL:
        out[c] = pd.to_numeric(body[pos[c]], errors="coerce").to_numpy() if c in pos else np.nan
    out["layout"] = "with_storage" if "storage_demand_mw" in pos else "original"
    fd = file_date(os.path.basename(path))
    date = (rep - dt.timedelta(days=1)) if rep else fd
    status = "ok" if (rep is None or fd is None or date == fd) else "date_mismatch_used_sheet_date"
    out["date"] = pd.Timestamp(date)
    out["reported_on"] = pd.Timestamp(rep) if rep else pd.NaT
    out["source_file"] = rel
    return dict(file=rel, status=status, data_date=str(date), reported_on=str(rep), file_name_date=str(fd),
                rows=len(out), sha256=sha, content=hashlib.sha256(out[COLS].to_csv(index=False).encode()).hexdigest(),
                size=len(raw)), out


def flags(df):
    """Per-column quality flags on the 15-minute table (sorted by time)."""
    df = df.sort_values("ts").reset_index(drop=True)
    for c in COLS:
        lo, hi = RANGE[c]
        v = df[c]
        bad = v.isna() | (v < lo) | (v > hi)
        run = (v != v.shift()).cumsum()
        runlen = v.groupby(run).transform("size")
        varies = v.rolling(96, min_periods=24).std() > 0
        frozen = (runlen >= FREEZE_BLOCKS) & varies.fillna(True) & (v != 0)
        df["flag_" + c] = np.select([v.isna(), bad, frozen], ["missing", "out_of_range", "frozen"], "")
    return df


def main():
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(SRC) for f in fs if f.lower().endswith((".xls", ".xlsx")))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("REFUSED_NCPUS", "8"))) as ex:
        res = list(ex.map(work, files, chunksize=8))
    log = pd.DataFrame([r for r, _ in res])
    parts = {r["file"]: o for r, o in res if o is not None}
    ok = log[log.file.isin(parts)].copy()
    # one report per data date: drop identical content, otherwise keep the larger file
    ok = ok.sort_values(["data_date", "size"], ascending=[True, False])
    keep, choice = [], {}
    for date, g in ok.groupby("data_date"):
        keep.append(g.file.iloc[0])
        for f in g.file.iloc[1:]:
            choice[f] = ("dropped_duplicate_content" if g.content.iloc[0] == g[g.file == f].content.iloc[0]
                         else f"dropped_other_report_for_date_kept:{g.file.iloc[0]}")
    log["used"] = log.file.isin(keep)
    log["choice"] = log.file.map(choice).fillna("")
    os.makedirs(INTERIM, exist_ok=True)
    log.to_csv(os.path.join(INTERIM, "psp_timeseries_parse_log.csv"), index=False)
    df = pd.concat([parts[f] for f in keep], ignore_index=True)
    df["ts"] = df["date"] + pd.to_timedelta(df["time"] + ":00")
    df = df.drop_duplicates("ts")
    full = pd.DataFrame({"ts": pd.date_range(df.ts.min().normalize(), df.ts.max().normalize() + pd.Timedelta("23:45:00"),
                                             freq="15min")})
    df = full.merge(df, on="ts", how="left")
    df["date"] = df.ts.dt.normalize()
    df = flags(df)
    df.to_parquet(os.path.join(PROC, "refused_allindia_15min.parquet"), index=False)
    clean = df.copy()
    for c in COLS:
        clean.loc[clean["flag_" + c] != "", c] = np.nan
    clean["hour"] = clean.ts.dt.floor("h")
    agg = clean.groupby("hour")[COLS + OPTIONAL].agg(["mean", "count"])
    hourly = pd.DataFrame({"ts": agg.index})
    for c in COLS + OPTIONAL:
        hourly[c] = np.where(agg[(c, "count")].to_numpy() >= 3, agg[(c, "mean")].to_numpy(), np.nan)
    hourly.to_parquet(os.path.join(PROC, "refused_allindia_hourly.parquet"), index=False)
    # QC summary
    days = df.date.nunique()
    covered = df.groupby("date").demand_met_mw.apply(lambda s: s.notna().any()).sum()
    lines = ["# All-India 15-minute series: coverage and quality control", "",
             f"Built by `codes/scripts/build/b09_psp_timeseries.py` from {log.used.sum()} Daily PSP reports "
             f"({len(log)} XLS files read). Data dates {df.date.min().date()} to {df.date.max().date()}: "
             f"{days} calendar days, {covered} with data.", "",
             "| Column | Missing blocks | Out of range | Frozen | Mean | Min | Max |", "|---|---|---|---|---|---|---|"]
    for c in COLS:
        f = df["flag_" + c]
        lines.append(f"| {c} | {(f == 'missing').sum()} | {(f == 'out_of_range').sum()} | {(f == 'frozen').sum()} | "
                     f"{df[c].mean():,.0f} | {df[c].min():,.0f} | {df[c].max():,.0f} |")
    lay = df.groupby("layout").date.nunique().to_dict() if "layout" in df else {}
    lines += ["", "Sheet layouts (days): " + ", ".join(f"{k} {v}" for k, v in lay.items()) + ". From 1 June 2026 the "
              "sheet adds storage demand (pumped storage and battery charging, included in demand met) and storage "
              "generation (pumped storage and batteries, no longer counted in hydro). Net demand met is defined the "
              "same way throughout (demand met minus wind and solar)."]
    lines += ["", "Parse outcomes: " + ", ".join(f"{k} {v}" for k, v in log.status.value_counts().items()) + ".",
              "Choices for dates with more than one report: " + (", ".join(f"{k} {v}" for k, v in
                                                                         pd.Series(list(choice.values())).str.split(":").str[0].value_counts().items())
                                                                  or "none") + ".", "",
              "Source: Grid Controller of India, Daily Power Supply Position report, sheet TimeSeries (SCADA, "
              "instantaneous values at the start of each block). Grid-India notes that the data are operational "
              "SCADA values that may contain telemetry errors, and that demand met and RE generation are those "
              "incident on the transmission system (distribution-connected and behind-the-meter generation excluded)."]
    open(os.path.join(DATA, "docs", "ALLINDIA_15MIN_QC.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
