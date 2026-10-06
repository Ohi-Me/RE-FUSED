"""Acquire Grid-India public reports through the website's public file API (https://webapi.grid-india.in/api/v1/).

Products:
  psp  Daily PSP report (FileType DAILY_PSP_REPORT). For each report date the XLS version is taken when published,
       otherwise the PDF. Stored as 03_data/raw/grid_india_psp/<FY>/<file>.
  dsm  DSM rates page (FileType F_PG00403): calendar-year ZIPs, daily CSV/XLSX files and weekly PX-rate ZIPs.
       Stored as 03_data/raw/grid_india_dsm/<FY>/<file>.
A listing of every API record is saved next to the files (LISTING_<product>.json) for provenance.

Usage: py -3.10 grid_india.py psp --from 2017-18 --to 2026-27
       py -3.10 grid_india.py dsm
"""
import argparse
import json
import os
import re
import sys
from urllib.parse import unquote

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8.fetch import Fetcher  # noqa: E402

API = "https://webapi.grid-india.in/api/v1/"
CDN = "https://webcdn.grid-india.in/"
MONTHS = ["04", "05", "06", "07", "08", "09", "10", "11", "12", "01", "02", "03"]


def fy_range(a, b):
    y0, y1 = int(a[:4]), int(b[:4])
    return [f"{y}-{str(y + 1)[2:]}" for y in range(y0, y1 + 1)]


def list_files(f, ftype, fy, month):
    js = f.post_json(API + "file", {"_source": "GRDW", "_type": ftype, "_fileDate": fy, "_month": month})
    return js.get("retData") or []


def psp(args):
    f = Fetcher("grid_india_psp", delay=0.5)
    listing = []
    for fy in fy_range(args.frm, args.to):
        for m in MONTHS:
            rows = list_files(f, "DAILY_PSP_REPORT", fy, m)
            listing += rows
            by_title = {}
            for r in rows:
                by_title.setdefault(r["Title_"].strip(), []).append(r)
            n_ok = 0
            for title, recs in sorted(by_title.items()):
                xls = [r for r in recs if re.search(r"\.xlsx?$", r["FilePath"], re.I)]
                pick = xls[0] if xls else recs[0]
                name = unquote(pick["FilePath"].split("/")[-1])
                if f.get(CDN + pick["FilePath"], f"{fy}/{name}", note=f"{title}|{pick.get('Field1')}|{pick['MimeType']}"):
                    n_ok += 1
            print(f"psp {fy}-{m}: {len(by_title)} report dates, {n_ok} files on disk", flush=True)
    json.dump(listing, open(os.path.join(f.dir, f"LISTING_psp_{args.frm}_{args.to}.json"), "w"), indent=0)


def dsm(args):
    f = Fetcher("grid_india_dsm", delay=0.5)
    per = f.post_json(API + "file/get-period", {"_source": "GRDW", "_fileType": "F_PG00403"}).get("retData") or []
    listing = []
    for p in per:
        fy = p["PeriodYear"]
        rows = list_files(f, "F_PG00403", fy, "")
        listing += rows
        for r in rows:
            name = unquote(r["FilePath"].split("/")[-1])
            f.get(CDN + r["FilePath"], f"{fy}/{name}", note=f"{r['Title_']}|{r.get('Field1')}|{r['MimeType']}")
        print(f"dsm {fy}: {len(rows)} files", flush=True)
    json.dump(listing, open(os.path.join(f.dir, "LISTING_dsm.json"), "w"), indent=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("product", choices=["psp", "dsm"])
    ap.add_argument("--from", dest="frm", default="2017-18")
    ap.add_argument("--to", default="2026-27")
    a = ap.parse_args()
    {"psp": psp, "dsm": dsm}[a.product](a)
