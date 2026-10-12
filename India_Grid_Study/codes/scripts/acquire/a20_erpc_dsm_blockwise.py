"""Download the block-wise deviation settlement data of the Eastern Regional Power Committee (ERPC).

ERPC publishes, for every week, the "DSM, SRAS, TRAS and SCUC Settlement Account" with a "Data Files" workbook
(DSM_Blockwise_Data_<from>-<to>.xlsx): one sheet per regional entity, one row per 15-minute block, with frequency,
actual and scheduled drawal or injection, deviation, the charges payable or receivable and the normal rate.
This script reads the listing pages, keeps the weekly accounts (and revised accounts) whose period overlaps the
requested dates, downloads each data workbook once, and writes a manifest (URL, title, file, size, SHA-256, UTC time).
Downloaded files are data: they are opened only with openpyxl in read-only mode by the parser.

Usage:  python codes/scripts/acquire/a20_erpc_dsm_blockwise.py [--from 2025-03-31] [--to 2026-08-31] [--pages 97]
Output: data/raw/erpc_dsm/*.xlsx and data/raw/erpc_dsm/MANIFEST.csv
"""
import csv
import datetime as dt
import hashlib
import html
import os
import re
import sys
import time

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "data", "raw", "erpc_dsm")
LIST = "https://www.erpc.gov.in/dsm-rras-fras-and-sced-accts"
UA = {"User-Agent": "Mozilla/5.0 (research data collection; RE-FUSED)"}
ITEM = re.compile(r'<div class="pdf-title">\s*(.*?)\s*</div>.*?<div class="meeting-links">(.*?)</div>', re.S)
LINK = re.compile(r'<a href="([^"]+)"[^>]*>\s*(.*?)\s*</a>', re.S)
PERIOD = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s*to\s*(\d{2})\.(\d{2})\.(\d{4})")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def listing(pages):
    rows = []
    for p in range(1, pages + 1):
        r = requests.get(LIST, params={"page": p}, headers=UA, timeout=120)
        r.raise_for_status()
        for title, links in ITEM.findall(r.text):
            title = html.unescape(re.sub(r"\s+", " ", title)).strip()
            m = PERIOD.search(title)
            if not m or "DSM" not in title.upper():
                continue
            d0 = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            d1 = dt.date(int(m.group(6)), int(m.group(5)), int(m.group(4)))
            for href, label in LINK.findall(links):
                if "Data File" in label:
                    rows.append(dict(title=title, start=d0, end=d1, url=html.unescape(href), page=p))
        time.sleep(1)
    return rows


def main():
    d_from = dt.date.fromisoformat(arg("--from", "2025-03-31"))
    d_to = dt.date.fromisoformat(arg("--to", "2026-08-31"))
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in listing(int(arg("--pages", "97"))) if r["end"] >= d_from and r["start"] <= d_to]
    print(len(rows), "data files in range")
    man_path = os.path.join(OUT, "MANIFEST.csv")
    done = set()
    if os.path.exists(man_path):
        done = {r["url"] for r in csv.DictReader(open(man_path, encoding="utf-8")) if r["status"] == "200"}
    new = not os.path.exists(man_path)
    with open(man_path, "a", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        if new:
            w.writerow(["title", "start", "end", "url", "file", "status", "bytes", "sha256", "retrieved_utc"])
        for r in rows:
            if r["url"] in done:
                continue
            try:
                resp = requests.get(r["url"], headers=UA, timeout=300)
                status = resp.status_code
                cd = resp.headers.get("Content-Disposition", "")
                fn = re.search(r'filename="?([^";]+)', cd)
                name = fn.group(1) if fn else f"erpc_{r['start']}_{r['end']}.bin"
                if "Revised" in r["title"]:
                    name = "Revised_" + name
                ok = status == 200 and resp.content[:2] == b"PK"      # an xlsx is a zip archive
                path = os.path.join(OUT, name)
                if ok:
                    open(path, "wb").write(resp.content)
                w.writerow([r["title"], r["start"], r["end"], r["url"], name if ok else "", status if ok else f"{status}-not-xlsx",
                            len(resp.content), hashlib.sha256(resp.content).hexdigest(),
                            dt.datetime.utcnow().isoformat(timespec="seconds")])
                mf.flush()
                print(status, name, len(resp.content) // 1024, "KB", flush=True)
            except requests.RequestException as e:
                w.writerow([r["title"], r["start"], r["end"], r["url"], "", f"error {type(e).__name__}", 0, "", ""])
                print("error", r["title"], e, flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
