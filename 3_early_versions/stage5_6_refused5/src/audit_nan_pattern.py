"""RE-FUSED-5 audit A4: what are the NaN MCP rows, and how many fully-observed days exist?"""
import glob, os, pandas as pd, numpy as np
ROOT = r"D:\\Refused0\Data_Preprocessing\dataset\prices of energy"
f23 = os.path.join(ROOT, "2023", "aug.csv")
d = pd.read_csv(f23)
mcp_raw = d["MCP (Rs/MWh)"]
print("2023/aug rows:", len(d), "| dtype:", mcp_raw.dtype)
print("distinct raw MCP values (first 12):", list(pd.unique(mcp_raw.astype(str)))[:12])
bad = d[pd.to_numeric(mcp_raw, errors="coerce").isna()]
print("NaN-MCP rows:", len(bad), "-> sample:")
print(bad.head(3).to_string(index=False))
print("hours present in NaN rows:", sorted(bad["Hour"].unique())[:10], "... n_hours:", bad["Hour"].nunique())
print("hours present in OK rows:", sorted(d[~d.index.isin(bad.index)]["Hour"].unique())[:10])
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "*", "*.csv"))):
    d = pd.read_csv(f)
    m = pd.to_numeric(d["MCP (Rs/MWh)"].astype(str).str.replace(",", ""), errors="coerce")
    dt = pd.to_datetime(d["Delivery"].astype(str), errors="coerce", dayfirst=True)
    t = pd.DataFrame({"date": dt.dt.date, "ok": m.notna()})
    g = t.groupby("date")["ok"].agg(["sum", "size"])
    rows.append(pd.DataFrame(dict(year=int(f.split(os.sep)[-2]), date=g.index,
                                  ok=g["sum"].values, tot=g["size"].values)))
a = pd.concat(rows)
a["complete"] = a.ok >= 96
a["partial"] = (a.ok > 0) & (a.ok < 96)
print("\n=== days by year: complete(>=96 obs) / partial / empty")
print(a.groupby("year").agg(days=("date", "nunique"), complete=("complete", "sum"),
                            partial=("partial", "sum"), empty=("ok", lambda s: int((s == 0).sum())),
                            mean_obs_blocks=("ok", "mean")).round(1).to_string())
print("\nTOTAL usable complete days:", int(a.complete.sum()), "| partial:", int(a.partial.sum()))
a.to_csv(r"D:\\REFUSED5\audit\A4_raw_day_coverage.csv", index=False)
