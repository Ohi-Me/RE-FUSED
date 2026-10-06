"""Acquire Central Electricity Authority (CEA, Govt. of India) publications.

  re    Daily Renewable Generation Report (State-wise wind, solar, other RE). Listing from the CEA website's public
        table endpoint https://cea.nic.in/wp-admin/admin-ajax.php?action=getpostsfordatatables (as used by
        https://cea.nic.in/daily-renewable-generation-report/). XLSX taken when published, otherwise PDF.
        Coverage on 14 Sep 2026: 1 May 2019 – 18 Nov 2025.
  co2   CO2 Baseline Database for the Indian Power Sector: every database file (zip/xlsx) and user guide linked from
        https://cea.nic.in/cdm-co2-baseline-database/ (versions up to 22.0, Sep 2026).

Usage: py -3.10 cea.py re|co2
"""
import argparse
import json
import os
import re
import sys
from urllib.parse import unquote

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8.fetch import Fetcher  # noqa: E402


def daily_re():
    f = Fetcher("cea_daily_re", delay=0.5)
    t = f.s.get("https://cea.nic.in/wp-admin/admin-ajax.php?action=getpostsfordatatables", timeout=180,
                headers={"Referer": "https://cea.nic.in/daily-renewable-generation-report/?lang=en",
                         "X-Requested-With": "XMLHttpRequest"}).text
    rows = json.loads(t[t.find("{"):])["data"]
    json.dump(rows, open(os.path.join(f.dir, "LISTING_daily_re.json"), "w"), indent=0)
    for r in sorted(rows, key=lambda x: x["date"]):
        cands = [x.split("^")[0] for x in (r.get("link", ""), r.get("excerpt", ""))]
        xlsx = [u for u in cands if u.lower().endswith(".xlsx")]
        pdf = [u for u in cands if u.lower().endswith(".pdf")]
        got = None
        for u in xlsx + pdf:
            name = unquote(u.split("/")[-1])
            got = f.get(u, f"{r['date'][:4]}/{r['date']}_{name}", note=r["title"], min_bytes=2000)
            if got:
                break
        if r["date"].endswith("-01"):
            print(r["date"], "ok" if got else "missing", flush=True)
    print("CEA_RE_DONE", flush=True)


def co2():
    f = Fetcher("cea_co2_database", delay=1.0)
    html = f.s.get("https://cea.nic.in/cdm-co2-baseline-database/?lang=en", timeout=120).text
    links = sorted(set(re.findall(r'href="(https://(?:www\.)?cea\.nic\.in/wp-content/uploads/[^"]+\.(?:zip|xlsx|xls|pdf))"', html)))
    keep = [u for u in links if re.search(r"baseline|CO2|co2|User_Guide|user_guide|database|version|emission", u)]
    for u in keep:
        name = unquote(u.split("/")[-1])
        f.get(u, name, note="CEA CO2 baseline database page")
        print(name, flush=True)
    print("CEA_CO2_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("product", choices=["re", "co2"])
    {"re": daily_re, "co2": co2}[ap.parse_args().product]()
