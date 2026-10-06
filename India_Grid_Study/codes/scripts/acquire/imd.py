"""Acquire IMD Pune gridded daily data (India Meteorological Department, Govt. of India).

  rain     0.25° x 0.25° daily rainfall (mm), 135 x 129 grid, 6.5–38.5°N, 66.5–100°E   form rainfall.php, field rain
  maxtemp  1° x 1° daily maximum temperature (°C), 31 x 31 grid, 7.5–37.5°N, 67.5–97.5°E  form maxtemp.php, field maxtemp
  mintemp  1° x 1° daily minimum temperature (°C)                                          form mintemp.php, field mintemp
Pages: https://www.imdpune.gov.in/cmpg/Griddata/{Rainfall_25_Bin,Max_1_Bin,Min_1_Bin}.html (years listed to 2025 on
14 Sep 2026). Files are binary; the manifest records form, year, bytes and SHA-256.

Usage: py -3.10 imd.py --years 2017-2025
"""
import argparse
import datetime as dt
import hashlib
import os
import sys
import time

import requests
import urllib3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused.fetch import USER_AGENT, Fetcher  # noqa: E402

urllib3.disable_warnings()
FORMS = {"rain": ("rainfall.php", "rain"), "maxtemp": ("maxtemp.php", "maxtemp"), "mintemp": ("mintemp.php", "mintemp")}
BASE = "https://www.imdpune.gov.in/cmpg/Griddata/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2017-2025")
    ap.add_argument("--vars", default="maxtemp,mintemp,rain")
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    f = Fetcher("imd_gridded", delay=3.0)
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Referer": BASE})
    for var in a.vars.split(","):
        action, field = FORMS[var]
        for y in range(y0, y1 + 1):
            key = f"{BASE}{action}?{field}={y}"
            if f.have(key):
                continue
            rel = f"{var}/{var}_{y}.grd"
            for attempt in range(5):
                try:
                    r = s.post(BASE + action, data={field: str(y)}, timeout=600, verify=False)
                    if r.status_code == 200 and len(r.content) > 100_000:
                        dst = os.path.join(f.dir, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        open(dst, "wb").write(r.content)
                        f._record(dict(url=key, path=rel, status="200", bytes=len(r.content),
                                       sha256=hashlib.sha256(r.content).hexdigest(),
                                       retrieved_utc=dt.datetime.utcnow().isoformat(timespec="seconds"),
                                       note=f"POST {action} {field}={y}; content-type {r.headers.get('Content-Type')}"))
                        print(var, y, len(r.content), flush=True)
                        break
                    print(var, y, "status", r.status_code, len(r.content), flush=True)
                except requests.RequestException as e:
                    print(var, y, "error", type(e).__name__, flush=True)
                time.sleep(10 * (attempt + 1))
            time.sleep(3)
    print("IMD_DONE", flush=True)


if __name__ == "__main__":
    main()
