"""Generate 10_manuscript/data_descriptor/numbers.tex from 03_data/docs/validation.json (O1 data descriptor).

Every number in the data-descriptor text is a macro defined here; numbers_manifest.json maps each macro to its source
field so the claims map can be checked automatically.
Usage: py -3.10 make_numbers_data.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8.paths import DOCS, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "10_manuscript", "data_descriptor")
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
M, MAN = {}, {}


def mname(s):
    s = "".join(p[:1].upper() + p[1:] for p in re.split(r"[^A-Za-z0-9]+", s) if p)
    return "".join(DIG.get(c, c) for c in s)


def put(name, value, field, fmt=None):
    if fmt == "int":
        text = f"{int(value):,}"
    elif fmt == "pct1":
        text = f"{100 * value:.1f}"
    elif fmt == "pct2":
        text = f"{100 * value:.2f}"
    elif isinstance(fmt, str) and fmt.startswith("f"):
        text = f"{value:.{int(fmt[1:])}f}"
    else:
        text = str(value)
    M[mname(name)] = text
    MAN[mname(name)] = dict(value=text, source="03_data/docs/validation.json", field=field)


def main():
    V = json.load(open(os.path.join(DOCS, "validation.json")))
    put("raw files official", V["raw_files_ok_official"], "raw_files_ok_official", "int")
    put("raw GB official", V["raw_total_bytes_official"] / 1e9, "raw_total_bytes_official", "f2")
    for src, d in V["manifests"].items():
        put(f"{src} files ok", d["files_ok"], f"manifests.{src}.files_ok", "int")
        put(f"{src} not available", d["not_available"], f"manifests.{src}.not_available", "int")
        put(f"{src} MB", d["bytes_ok"] / 1e6, f"manifests.{src}.bytes_ok", "f1")
    p = V["psp"]
    for k, f in [("state_rows", "int"), ("dates", "int"), ("entities", "int"), ("missing_dates_window", "int"), ("missing_dates_shared_path", "int"),
                 ("listing_records", "int"), ("listing_filepaths_shared_by_titles", "int"), ("titles_on_shared_paths", "int"),
                 ("largest_shared_path_titles", "int"), ("date_reassigned_by_cover_letter", "int"),
                 ("publication_lag_hours_median", "f1"), ("publication_lag_hours_p90", "f1"), ("publication_lag_n", "int")]:
        put(f"psp {k}", p[k], f"psp.{k}", f)
    put("psp first", p["first"], "psp.first")
    put("psp last", p["last"], "psp.last")
    put("psp missing pct window", p["missing_dates_window"] / 3075, "psp.missing_dates_window / 3075 days", "pct1")
    d = V["dsm"]
    for k, f in [("dates", "int"), ("bid_areas", "int"), ("daily_file_lag_days_median", "f1"),
                 ("daily_file_lag_days_min", "f1"), ("daily_file_lag_days_max", "f1"), ("daily_file_lag_n", "int")]:
        put(f"dsm {k}", d[k], f"dsm.{k}", f)
    put("dsm first", d["first"], "dsm.first")
    put("dsm last", d["last"], "dsm.last")
    for r, v in d["regimes"].items():
        put(f"dsm {r} first", v["first"], f"dsm.regimes.{r}.first")
        put(f"dsm {r} last", v["last"], f"dsm.regimes.{r}.last")
    n = V["npp"]
    for k, f in [("station_rows", "int"), ("state_rows", "int"), ("dates", "int"), ("entities", "int"),
                 ("missing_dates", "int"), ("missing_2020_lockdown", "int"), ("files_no_state_totals", "int"),
                 ("idp_overlap_rows", "int"), ("idp_share_within_1pct", "pct1"), ("idp_share_within_0p01gwh", "pct1")]:
        put(f"npp {k}", n[k], f"npp.{k}", f)
    put("npp first", n["first"], "npp.first")
    put("npp last", n["last"], "npp.last")
    put("npp idp last", n["idp_last"], "npp.idp_last")
    put("npp max station vs total exp", f"{n['max_abs_station_vs_total_gwh']:.1e}".replace("e-", r"\times10^{-") + "}",
        "npp.max_abs_station_vs_total_gwh")
    c = V["cea_re"]
    for k, f in [("rows", "int"), ("dates", "int"), ("ctrl_dates", "int"), ("entities", "int"), ("files", "int"),
                 ("date_mismatch_files", "int"), ("duplicate_table_files", "int"), ("inconsistent_rows_all", "int"),
                 ("inconsistent_rows_ctrl", "int"), ("idp_overlap_rows", "int"), ("idp_share_within_1pct", "pct1")]:
        put(f"re {k}", c[k], f"cea_re.{k}", f)
    for k in ("first", "last", "ctrl_first"):
        put(f"re {k}", c[k], f"cea_re.{k}")
    no_other = c["layout_counts"].get("no_other_col", {}).get("all", 0)
    put("re no other col rows", no_other, "cea_re.layout_counts.no_other_col.all", "int")
    q = V["co2"]
    put("co2 cells", q["cells"], "co2.cells", "int")
    put("co2 ratio coal", q["net_gross_ratio"]["coal"], "co2.net_gross_ratio.coal", "f3")
    put("co2 ratio gas", q["net_gross_ratio"]["gas"], "co2.net_gross_ratio.gas", "f3")
    put("co2 coal ef min", q["coal_ef_gross_min"], "co2.coal_ef_gross_min", "f2")
    put("co2 coal ef max", q["coal_ef_gross_max"], "co2.coal_ef_gross_max", "f2")
    put("co2 coal ef median", q["coal_ef_gross_median"], "co2.coal_ef_gross_median", "f2")
    w = V["imd"]
    put("imd rows", w["rows"], "imd.rows", "int")
    put("imd delhi jan", w["delhi_tmax_jan"], "imd.delhi_tmax_jan", "f1")
    put("imd delhi may", w["delhi_tmax_may"], "imd.delhi_tmax_may", "f1")
    put("imd mumbai jul", w["mumbai_cell_rain_jul2023_mm"], "imd.mumbai_cell_rain_jul2023_mm", "int")
    P = V["panel"]
    for k, f in [("rows", "int"), ("columns", "int"), ("entities", "int"), ("days", "int"), ("nonmissing_values", "int"),
                 ("rows_with_energy_met", "int")]:
        put(f"panel {k}", P[k], f"panel.{k}", f)
    put("panel nonmissing million", P["nonmissing_values"] / 1e6, "panel.nonmissing_values", "f2")
    put("panel first", P["first"], "panel.first")
    put("panel last", P["last"], "panel.last")
    put("panel sha short", P["content_sha256"][:16], "panel.content_sha256")
    for s, v in P["splits"].items():
        put(f"panel split {s}", v, f"panel.splits.{s}", "int")
    for k, v in P["qc_flag_counts"].items():
        put(f"panel {k}", v, f"panel.qc_flag_counts.{k}", "int")
    for b, v in P["blocks_missing"].items():
        put(f"panel miss {b} dev", v["train_dev"], f"panel.blocks_missing.{b}.train_dev", "pct1")
        put(f"panel miss {b} reserved", v["reserved"], f"panel.blocks_missing.{b}.reserved", "pct1")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by 04_code/scripts/paper/make_numbers_data.py from 03_data/docs/validation.json; do not edit\n")
        for k in sorted(M):
            fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    json.dump(MAN, open(os.path.join(OUT, "numbers_manifest.json"), "w"), indent=1)
    print(len(M), "macros written")


if __name__ == "__main__":
    main()
