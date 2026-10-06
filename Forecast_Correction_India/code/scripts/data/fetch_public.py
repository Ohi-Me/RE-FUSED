"""Fetch public benchmark datasets used only in RE-FUSED (never touched by RE-FUSED-6).

OPSD time series 2020-10-06, 60-min single index (ENTSO-E load actual + TSO day-ahead forecast);
attribution: Open Power System Data (2020), https://doi.org/10.25832/time_series/2020-10-06
UCI ElectricityLoadDiagrams20112014 (CC BY 4.0); Trindade (2015), https://doi.org/10.24432/C58C86
Writes SHA-256 checksums next to each file.
"""
import hashlib, os, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "Other_Countries", "data", "raw", "public")
FILES = {
    "opsd_time_series_60min_singleindex_2020-10-06.csv":
        "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv",
    "uci_electricityloaddiagrams20112014.zip":
        "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
}
os.makedirs(OUT, exist_ok=True)
for name, url in FILES.items():
    dst = os.path.join(OUT, name)
    if not os.path.exists(dst):
        tmp = dst + ".part"
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    open(dst + ".sha256", "w").write(h + "  " + name + "\n")
    print("OK", name, os.path.getsize(dst), h[:16], flush=True)
print("DONE")
