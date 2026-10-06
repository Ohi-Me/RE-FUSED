"""Build step 1: parse every Grid-India Daily PSP report into tidy interim tables.

Inputs : data/raw/grid_india_psp/<FY>/* (MANIFEST.csv lists every file with its SHA-256)
Outputs: data/interim/psp_state_day.parquet   (date, entity, state_raw, region, 8 State variables, source_file)
         data/interim/psp_region_day.parquet  (date, region, section, item, value, source_file)
         data/interim/psp_freq_day.parquet    (date, FVI and % time in frequency bands, source_file)
         data/interim/psp_parse_log.csv       (file, status, date, n_states, n_region_items)
Rules  : one report per data date. If two reports with different figures carry the same table-page "Date of
         Reporting" and one cover letter implies a different, otherwise uncovered date, that report is moved to the
         cover-letter date (`date_reassigned_to`). The same report uploaded under two dates is used once
         (`dropped_duplicate_content`). If several files still map to the same date, XLS is preferred over PDF, then
         the file with most States; the choice is logged. DD and DNH (reported separately before their merger) are
         summed into DNHDD for additive variables; max-type variables take the sum of the two maxima (upper bound,
         flagged in `dnhdd_combined`).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused import entities as E  # noqa: E402
from refused import parse_psp as P  # noqa: E402
from refused.paths import INTERIM, RAW  # noqa: E402

SRC = os.path.join(RAW, "grid_india_psp")
ADDITIVE = ["energy_met_gwh", "drawal_schedule_gwh", "od_ud_gwh", "energy_shortage_gwh", "peak_shortage_mw",
            "max_demand_mw", "max_od_mw", "max_ud_mw"]


def work(path):
    rel = os.path.relpath(path, SRC).replace("\\", "/")
    try:
        res = P.parse_xls(path, E.entity_id) if path.lower().endswith((".xls", ".xlsx")) else P.parse_pdf(path, E.entity_id)
    except Exception as e:  # unreadable or unexpected layout
        return rel, None, f"error:{type(e).__name__}"
    if res is None:
        return rel, None, "not_psp_or_no_date"
    if len(res["state"]) < 25:
        return rel, res, f"few_states:{len(res['state'])}"
    return rel, res, "ok"


def main():
    man = pd.read_csv(os.path.join(SRC, "MANIFEST.csv"))
    files = [os.path.join(SRC, p) for p in man.loc[man.status.astype(str) == "200", "path"]]
    files = [f for f in files if os.path.exists(f)]
    print("files", len(files), flush=True)
    log, states, regions, freqs, metas = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, (rel, res, status) in enumerate(ex.map(work, files, chunksize=8)):
            n_s = len(res["state"]) if res else 0
            date = res["state"][0]["date"] if res and res["state"] else (res["freq"][0]["date"] if res and res["freq"] else None)
            log.append(dict(file=rel, status=status, date=date, n_states=n_s, n_region=len(res["region"]) if res else 0,
                            fmt="xls" if rel.lower().endswith(("xls", "xlsx")) else "pdf"))
            if res and status in ("ok",) or (res and status.startswith("few_states")):
                for r in res["state"]:
                    states.append(dict(r, source_file=rel))
                for r in res["region"]:
                    regions.append(dict(r, source_file=rel))
                for r in res["freq"]:
                    freqs.append(dict(r, source_file=rel))
                mt = res.get("meta") or {"table_date": date, "letter_date": None}
                metas.append(dict(file=rel, **mt))
            if k % 500 == 0:
                print(k, rel, status, flush=True)
    log = pd.DataFrame(log)
    st = pd.DataFrame(states)
    st["date"] = pd.to_datetime(st.date)
    regions, freqs = pd.DataFrame(regions), pd.DataFrame(freqs)
    # ---- date collisions: two different reports whose table page carries the same "Date of Reporting"
    fp = st.sort_values("state_raw").groupby("source_file").apply(
        lambda g: hash(tuple(zip(g.state_raw, g.energy_met_gwh.round(2), g.od_ud_gwh.round(2)))), include_groups=False)
    meta = pd.DataFrame(metas).set_index("file")
    meta["fp"] = fp
    meta["table_date"] = pd.to_datetime(meta.table_date)
    meta["letter_date"] = pd.to_datetime(meta.letter_date)
    meta = meta.dropna(subset=["fp"])
    covered = set(meta.table_date)
    reassign = {}
    for d, g in meta.groupby("table_date"):
        if g.fp.nunique() < 2:
            continue
        for f, r in g.iterrows():
            if pd.notna(r.letter_date) and r.letter_date != d and r.letter_date not in covered:
                reassign[f] = r.letter_date
                covered.add(r.letter_date)
    for df_ in (st, regions, freqs):
        df_["date"] = pd.to_datetime(df_.date)
        m = df_.source_file.isin(reassign)
        df_.loc[m, "date"] = df_.loc[m, "source_file"].map(reassign)
    log["date_reassigned_to"] = log.file.map(reassign)
    # identical content under different dates (the same report uploaded twice): keep the file whose cover letter
    # agrees with its table date; the other copy is not used
    meta["final_date"] = [reassign.get(f, d) for f, d in zip(meta.index, meta.table_date)]
    dup = meta[meta.duplicated("fp", keep=False)]
    drop = set()
    for _, g in dup.groupby("fp"):
        if g.final_date.nunique() > 1:
            agree = g[(g.letter_date == g.final_date) | g.letter_date.isna()]
            keep = agree.index[0] if len(agree) else g.index[0]
            drop |= set(g.index) - {keep}
    log["dropped_duplicate_content"] = log.file.isin(drop)
    st = st[~st.source_file.isin(drop)]
    print("date collisions resolved by cover letter:", len(reassign), "| duplicate uploads dropped:", len(drop), flush=True)
    st["fmt"] = st.source_file.str.lower().str.endswith(("xls", "xlsx")).map({True: "xls", False: "pdf"})
    # one source file per date: prefer xls, then the most states parsed
    pick = (st.groupby(["date", "source_file", "fmt"]).size().rename("n").reset_index()
            .sort_values(["date", "fmt", "n"], ascending=[True, False, False]).drop_duplicates("date"))
    log["chosen"] = log.file.isin(pick.source_file)
    st = st.merge(pick[["date", "source_file"]], on=["date", "source_file"])
    st["entity"] = st.state_raw.map(E.entity_id)
    unmapped = sorted(set(st.loc[st.entity.isna(), "state_raw"]))
    st = st.dropna(subset=["entity"])
    comb = st.groupby(["date", "entity"]).size().rename("n_raw").reset_index()
    agg = st.groupby(["date", "entity"], as_index=False).agg(
        {**{c: (lambda s: s.sum(min_count=1)) for c in ADDITIVE}, "state_raw": lambda s: "+".join(sorted(s)),
         "source_file": "first"})
    agg = agg.merge(comb, on=["date", "entity"])
    agg["dnhdd_combined"] = agg.n_raw > 1
    agg["region"] = agg.entity.map(lambda e: E.ENTITIES[e][1])
    agg["bid_area"] = agg.entity.map(lambda e: E.ENTITIES[e][2])
    rg = regions.merge(pick[["date", "source_file"]], on=["date", "source_file"])
    fq = freqs.merge(pick[["date", "source_file"]], on=["date", "source_file"])
    os.makedirs(INTERIM, exist_ok=True)
    agg.drop(columns=["n_raw"]).to_parquet(os.path.join(INTERIM, "psp_state_day.parquet"), index=False)
    rg.to_parquet(os.path.join(INTERIM, "psp_region_day.parquet"), index=False)
    fq.to_parquet(os.path.join(INTERIM, "psp_freq_day.parquet"), index=False)
    log.to_csv(os.path.join(INTERIM, "psp_parse_log.csv"), index=False)
    print("states rows", len(agg), "dates", agg.date.nunique(), agg.date.min(), agg.date.max(),
          "entities", agg.entity.nunique(), "unmapped labels", unmapped[:20], flush=True)
    print(log.status.str.split(":").str[0].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
