"""Fetch NYISO zonal prices: 5-min real-time (realtime_zone) + hourly day-ahead (damlbmp_zone).
Resumable: skips months already stored. Saves one parquet per month per product."""
import urllib.request, zipfile, io, os, sys, time
import pandas as pd
OUT = r"D:\\REFUSED5\data\nyiso"
PRODUCTS = {"realtime_zone": "csv/realtime", "damlbmp_zone": "csv/damlbmp"}
START, END = (2019, 1), (2025, 12)
os.makedirs(OUT, exist_ok=True)
def months(a, b):
    y, m = a
    while (y, m) <= b:
        yield y, m
        m += 1
        if m == 13: y, m = y + 1, 1
log = open(os.path.join(OUT, "_fetch.log"), "a", encoding="utf-8")
for prod, path in PRODUCTS.items():
    d = os.path.join(OUT, prod); os.makedirs(d, exist_ok=True)
    for y, m in months(START, END):
        tag = f"{y}{m:02d}"
        dst = os.path.join(d, f"{tag}.parquet")
        if os.path.exists(dst):
            continue
        url = f"http://mis.nyiso.com/public/{path}/{tag}01{prod}_csv.zip"
        try:
            raw = urllib.request.urlopen(url, timeout=120).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            frames = [pd.read_csv(z.open(n)) for n in z.namelist() if n.endswith(".csv")]
            df = pd.concat(frames, ignore_index=True)
            df.columns = [c.strip() for c in df.columns]
            df.to_parquet(dst, index=False)
            msg = f"OK   {prod} {tag} files={len(frames)} rows={len(df)}"
        except Exception as e:
            msg = f"FAIL {prod} {tag} {type(e).__name__} {str(e)[:80]}"
        print(msg); log.write(msg + "\n"); log.flush()
        time.sleep(0.8)
log.close()
print("DONE")
