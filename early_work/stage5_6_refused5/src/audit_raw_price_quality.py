"""RE-FUSED-5 audit A2: per-year quality + schema variants of the raw IEX block archive."""
import glob, os, json, collections
import numpy as np, pandas as pd
RAW = r"D:\\Refused0\Data_Preprocessing\cleaned\prices of energy"
DSET= r"D:\\Refused0\Data_Preprocessing\dataset\prices of energy"
OUT = r"D:\\REFUSED5\audit"
schemas = collections.Counter(); rows=[]
for f in sorted(glob.glob(os.path.join(RAW,"*","*.csv"))):
    yr, mon = f.split(os.sep)[-2], os.path.basename(f)[:-4]
    try: d = pd.read_csv(f)
    except Exception as e:
        rows.append(dict(year=yr, month=mon, err=str(e)[:40])); continue
    schemas[tuple(d.columns)] += 1
    mcpcol = next((c for c in d.columns if "mcp" in c.lower()), None)
    delcol = next((c for c in d.columns if "deliv" in c.lower() or c.lower()=="date"), None)
    mcp = pd.to_numeric(d[mcpcol], errors="coerce") if mcpcol else pd.Series(dtype=float)
    dt  = pd.to_datetime(d[delcol], format="%d-%m-%Y", errors="coerce") if delcol else pd.Series(dtype="datetime64[ns]")
    if delcol is not None and dt.isna().mean() > 0.5:
        dt = pd.to_datetime(d[delcol], errors="coerce", dayfirst=True)
    rows.append(dict(year=yr, month=mon, n=len(d), mcp_col=mcpcol, date_col=delcol,
                     mcp_nan_pct=round(100*float(mcp.isna().mean()),1) if len(mcp) else None,
                     date_nan_pct=round(100*float(dt.isna().mean()),1) if len(dt) else None,
                     days=int(dt.dt.date.nunique()) if len(dt) else 0,
                     mcp_mean=round(float(mcp.mean()),1) if len(mcp) else None,
                     mcp_max=round(float(mcp.max()),1) if len(mcp) else None))
q = pd.DataFrame(rows)
q.to_csv(os.path.join(OUT,"A2_raw_price_by_file.csv"), index=False)
print("=== schema variants:", len(schemas))
for s,c in schemas.most_common(): print(f"  x{c}: {list(s)}")
print("\n=== per-year summary")
qq = q.dropna(subset=["n"]).copy(); qq["year"]=qq.year.astype(int)
agg = qq.groupby("year").agg(files=("month","count"), rows=("n","sum"), days=("days","sum"),
                              mcp_nan_pct=("mcp_nan_pct","mean"), mcp_mean=("mcp_mean","mean"), mcp_max=("mcp_max","max"))
print(agg.round(1).to_string())
print("\n=== files with >50% NaN mcp"); print(q[q.mcp_nan_pct>50][["year","month","n","mcp_nan_pct"]].to_string(index=False))
d2 = sorted(glob.glob(os.path.join(DSET,"*","*.csv")))
print(f"\n=== untouched 'dataset' copy: {len(d2)} files; sample head")
if d2:
    import itertools
    with open(d2[0]) as fh: print(d2[0].split(os.sep)[-2:], "|", "".join(itertools.islice(fh,2)).strip()[:200])
