"""RE-FUSED-5 audit A3: untouched 'dataset' IEX archive vs damaged 'cleaned' copy."""
import glob, os, collections, json
import numpy as np, pandas as pd
ROOTS = {"dataset(original)": r"D:\\Refused0\Data_Preprocessing\dataset\prices of energy",
         "cleaned":           r"D:\\Refused0\Data_Preprocessing\cleaned\prices of energy"}
OUT = r"D:\\REFUSED5\audit"; summary={}
for tag, root in ROOTS.items():
    schemas = collections.Counter(); recs=[]
    for f in sorted(glob.glob(os.path.join(root,"*","*.csv"))):
        try: d = pd.read_csv(f)
        except Exception: continue
        schemas[tuple(c.strip() for c in d.columns)] += 1
        mcol = next((c for c in d.columns if "mcp" in c.lower()), None)
        dcol = next((c for c in d.columns if "deliv" in c.lower()), None)
        if mcol is None or dcol is None: continue
        mcp = pd.to_numeric(d[mcol].astype(str).str.replace(",","").str.strip('"'), errors="coerce")
        dt  = pd.to_datetime(d[dcol].astype(str).str.strip('"'), errors="coerce", dayfirst=True)
        recs.append(dict(year=int(f.split(os.sep)[-2]), n=len(d), nan=float(mcp.isna().mean()),
                         days=int(dt.dt.date.nunique()), mean=float(mcp.mean()), mx=float(mcp.max())))
    df = pd.DataFrame(recs)
    summary[tag] = dict(files=int(len(df)), rows=int(df.n.sum()), days=int(df.days.sum()),
                        mcp_nan_pct=round(100*float((df.nan*df.n).sum()/df.n.sum()),2),
                        mcp_mean=round(float(df["mean"].mean()),1), mcp_max=round(float(df.mx.max()),1),
                        schemas=len(schemas))
    print(f"=== {tag}: {summary[tag]}")
    for s,c in schemas.most_common(3): print(f"    x{c}: {list(s)[:9]}")
    if not df.empty:
        print(df.groupby("year").agg(files=("n","count"), rows=("n","sum"), days=("days","sum"),
              nan_pct=("nan", lambda s: round(100*float(s.mean()),1)), mean=("mean","mean")).round(1).to_string())
json.dump(summary, open(os.path.join(OUT,"A3_raw_archive_compare.json"),"w"), indent=2)
