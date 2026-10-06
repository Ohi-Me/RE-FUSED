"""Fetch NYISO MIS monthly archives (public, http://mis.nyiso.com/public/).

Products:
  isolf          ISO 7-day hourly zonal load forecast, one file per issue day
  palIntegrated  hourly integrated actual zonal load
  damlbmp_zone   day-ahead zonal LBMP (hourly)
  realtime_zone  real-time zonal LBMP (5-min)
  damasp         day-ahead ancillary service prices (hourly)
Resumable: a month already stored as parquet is skipped. NYISO's Legal Notice grants no redistribution
licence, so raw files are NOT part of the release; this script regenerates them.
Usage: py -3.10 fetch_nyiso.py [start YYYYMM] [end YYYYMM] [products comma-separated]
"""
import io, os, sys, time, zipfile, urllib.request
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "Other_Countries", "data", "raw", "nyiso")
PATHS = {"isolf": "csv/isolf/{t}01isolf_csv.zip", "palIntegrated": "csv/palIntegrated/{t}01palIntegrated_csv.zip",
         "damlbmp_zone": "csv/damlbmp/{t}01damlbmp_zone_csv.zip", "realtime_zone": "csv/realtime/{t}01realtime_zone_csv.zip",
         "damasp": "csv/damasp/{t}01damasp_csv.zip"}


def months(a, b):
    y, m = divmod(a, 100)
    while y * 100 + m <= b:
        yield f"{y}{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 201901
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 202608
    prods = sys.argv[3].split(",") if len(sys.argv) > 3 else list(PATHS)
    log = open(os.path.join(OUT, "_fetch.log"), "a", encoding="utf-8") if os.makedirs(OUT, exist_ok=True) is None else None
    for p in prods:
        os.makedirs(os.path.join(OUT, p), exist_ok=True)
        for t in months(start, end):
            dst = os.path.join(OUT, p, f"{t}.parquet")
            if os.path.exists(dst):
                continue
            url = "http://mis.nyiso.com/public/" + PATHS[p].format(t=t)
            for attempt in range(3):
                try:
                    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=180).read()))
                    frames = []
                    for n in sorted(z.namelist()):
                        if not n.endswith(".csv"):
                            continue
                        f = pd.read_csv(z.open(n))
                        f.columns = [c.strip() for c in f.columns]
                        if p == "isolf":
                            f["issue_date"] = n[:8]
                        frames.append(f)
                    pd.concat(frames, ignore_index=True).to_parquet(dst, index=False)
                    msg = f"OK   {p} {t} files={len(frames)}"
                    break
                except Exception as e:
                    msg = f"FAIL {p} {t} attempt={attempt+1} {type(e).__name__}: {str(e)[:80]}"
                    time.sleep(5)
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            time.sleep(0.5)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
