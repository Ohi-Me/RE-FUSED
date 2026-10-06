"""Acquire CEA daily reports from the National Power Portal (https://npp.gov.in/publishedReports), Govt. of India.

Reports (XLS versions):
  dgr2   unit-wise generation (monitored capacity, programmed and actual generation)
  dgr6   hydro reservoir levels
  dgr10  daily maintenance report (coal, lignite, nuclear): planned and forced maintenance by unit
  coal   daily coal stock report (dailyCoal1)
URL patterns (verified 14 Sep 2026):
  https://npp.gov.in/public-reports/cea/daily/dgr/DD-MM-YYYY/dgrN-YYYY-MM-DD.xls
  https://npp.gov.in/public-reports/cea/daily/fuel/DD-MM-YYYY/dailyCoal1-YYYY-MM-DD.xls(x)   (.xlsx for most dates)
The DGR archive starts on 22 Mar 2018 (found by bisection). Days without a published file are recorded as 404.

Usage: py -3.10 npp.py --start 2018-03-22 --end 2026-09-11 [--reports dgr2,dgr10,dgr6,coal]
"""
import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8.fetch import Fetcher  # noqa: E402

BASE = "https://npp.gov.in/public-reports/cea/daily"


def url_for(rep, d):
    if rep == "coal":
        return f"{BASE}/fuel/{d:%d-%m-%Y}/dailyCoal1-{d:%Y-%m-%d}.xls"
    return f"{BASE}/dgr/{d:%d-%m-%Y}/{rep}-{d:%Y-%m-%d}.xls"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-03-22")
    ap.add_argument("--end", default="2026-09-11")
    ap.add_argument("--reports", default="dgr2,dgr10,dgr6,coal")
    a = ap.parse_args()
    reps = a.reports.split(",")
    fetchers = {r: Fetcher(f"npp_{r}", delay=0.35) for r in reps}
    d, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    n = 0
    while d <= end:
        for r in reps:
            got = fetchers[r].get(url_for(r, d), f"{d.year}/{d:%Y-%m-%d}_{r}.xls", note=r, min_bytes=2000)
            if not got and r == "coal":  # the coal archive publishes .xlsx for most dates (verified via the date form)
                fetchers[r].get(url_for(r, d) + "x", f"{d.year}/{d:%Y-%m-%d}_{r}.xlsx", note=r, min_bytes=2000)
        n += 1
        if n % 30 == 0:
            got = {r: sum(1 for x in fetchers[r].done.values() if x["status"] == "200") for r in reps}
            print(f"{d} done; files on record {got}", flush=True)
        d += dt.timedelta(days=1)
    print("NPP_DONE", flush=True)


if __name__ == "__main__":
    main()
