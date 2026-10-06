"""Build step 4: parse CEA Daily Renewable Generation Reports into a State-day RE table.

Inputs : 03_data/raw/cea_daily_re/<year>/<YYYY-MM-DD>_<name>.(xlsx|pdf)  (MANIFEST.csv; date = listing date)
Outputs: 03_data/interim/cea_re_state_day.parquet
           date, entity, and for each scope s in {all, ctrl}: {s}_wind_gwh, {s}_solar_gwh, {s}_other_re_gwh,
           {s}_total_re_gwh, {s}_sum_flag; state_raw, source_file
           all  = all RE located in the State incl. inter-State generating stations (State+ISGS table)
           ctrl = RE in the State's own control area (State Control Area table, from May 2020)
         03_data/interim/cea_re_parse_log.csv
Rules  : one file per report date, XLSX preferred over PDF, then the file with most State rows. The date printed in
         the report is used (the website listing date is wrong for some files, e.g. a "30 Sept" report listed on
         30 Oct 2021); disagreements are logged. Row check (parser): wind + solar + other = total within 0.05 GWh,
         or wind + solar = total for files without an "other RES" column; rows passing neither are set to NaN and
         flagged in {s}_sum_flag, with the recognised layout in {s}_layout. Labels that are not Grid-India control areas (A&N Islands,
         Lakshadweep, region lines) are logged and excluded.
"""
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import entities as E  # noqa: E402
from refused import parse_cea_re as C  # noqa: E402
from refused.paths import INTERIM, RAW  # noqa: E402

SRC = os.path.join(RAW, "cea_daily_re")
NUM = ["wind_gwh", "solar_gwh", "other_re_gwh", "total_re_gwh"]


def work(rel):
    p = os.path.join(SRC, rel)
    try:
        r = C.parse_xlsx(p) if rel.lower().endswith(".xlsx") else C.parse_pdf(p)
    except Exception as e:
        return rel, None, f"error:{type(e).__name__}"
    n = sum(1 for x in r["rows"] if x["scope"] == "all")
    if r["title_swapped"] == "duplicate_tables":
        return rel, r, "duplicate_tables"
    return rel, r, "ok" if n >= 20 or len(r["rows"]) >= 20 else f"few_states:{len(r['rows'])}"


def main():
    man = pd.read_csv(os.path.join(SRC, "MANIFEST.csv"))
    rels = sorted({p for p in man.loc[man.status.astype(str) == "200", "path"] if os.path.exists(os.path.join(SRC, p))})
    print("files", len(rels), flush=True)
    log, rows = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for rel, r, status in ex.map(work, rels, chunksize=8):
            ldate = re.search(r"(\d{4}-\d{2}-\d{2})", rel).group(1)
            rdate = str(r["date"]) if r and r["date"] else None
            scopes = sorted({x["scope"] for x in r["rows"]}) if r else []
            log.append(dict(file=rel, status=status, n_rows=len(r["rows"]) if r else 0, scopes="+".join(scopes),
                            title_swapped=r["title_swapped"] if r else None, listing_date=ldate, report_date=rdate,
                            fmt=rel.rsplit(".", 1)[-1].lower()))
            if r and rdate:
                rows += [dict(x, date=rdate, source_file=rel) for x in r["rows"]]
    log = pd.DataFrame(log)
    log["date_mismatch"] = log.report_date.notna() & (log.report_date != log.listing_date)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.date)
    df["fmt"] = df.source_file.str.rsplit(".", n=1).str[-1].str.lower()
    df["has_ctrl"] = df.groupby("source_file").scope.transform(lambda s: (s == "ctrl").any())
    pick = (df[df.scope == "all"].groupby(["date", "source_file", "fmt"]).size().rename("n").reset_index()
            .sort_values(["date", "fmt", "n"], ascending=[True, False, False]).drop_duplicates("date"))
    log["chosen"] = log.file.isin(pick.source_file)
    df = df.merge(pick[["date", "source_file"]], on=["date", "source_file"])
    df["entity"] = df.state_raw.map(E.entity_id)
    unmapped = df.loc[df.entity.isna(), "state_raw"].value_counts()
    df = df.dropna(subset=["entity"])
    g = df.groupby(["date", "entity", "scope"], as_index=False).agg(
        {**{c: (lambda s: s.sum(min_count=1)) for c in NUM}, "state_raw": lambda s: "+".join(sorted(set(s))),
         "source_file": "first"})
    lay = df.groupby(["date", "entity", "scope"]).layout.agg(lambda s: "+".join(sorted(set(s)))).rename("layout")
    g = g.merge(lay.reset_index(), on=["date", "entity", "scope"])
    g["sum_flag"] = g.layout.str.contains("inconsistent")  # values set to NaN by the parser
    wide = g.pivot_table(index=["date", "entity"], columns="scope", values=NUM + ["sum_flag", "layout"],
                         aggfunc="first")
    wide.columns = [f"{s}_{v}" for v, s in wide.columns]
    meta = g[g.scope == "all"].set_index(["date", "entity"])[["state_raw", "source_file"]]
    out = wide.join(meta).reset_index()
    for c in [c for c in out.columns if c.endswith("sum_flag")]:
        out[c] = out[c].astype("boolean")
    out.to_parquet(os.path.join(INTERIM, "cea_re_state_day.parquet"), index=False)
    log.to_csv(os.path.join(INTERIM, "cea_re_parse_log.csv"), index=False)
    print("rows", len(out), "dates", out.date.nunique(), out.date.min(), out.date.max(), "entities", out.entity.nunique())
    print("ctrl coverage:", out.loc[out.ctrl_total_re_gwh.notna(), "date"].agg(["min", "max", "nunique"]).to_dict())
    print("unmapped labels:", unmapped.to_dict())
    print(log.status.str.split(":").str[0].value_counts().to_dict(), "| title/magnitude swaps:",
          int((log.title_swapped == True).sum()), "| date mismatches:", int(log.date_mismatch.sum()),  # noqa: E712
          "| sum flags all/ctrl:", int(out.all_sum_flag.sum()), int(out.ctrl_sum_flag.sum()))


if __name__ == "__main__":
    main()
