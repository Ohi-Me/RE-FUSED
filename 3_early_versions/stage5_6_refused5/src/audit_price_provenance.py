"""RE-FUSED-5 audit A1: provenance of avg_market_price vs raw 15-min IEX blocks."""
import glob, os, json, re
import numpy as np, pandas as pd

PANEL = r"D:\\Refused4\data\full_dataset.csv"
RAW   = r"D:\\Refused0\Data_Preprocessing\cleaned\prices of energy"
OUT   = r"D:\\REFUSED5\audit"
os.makedirs(OUT, exist_ok=True)
rep = {}

# ---------- panel ----------
cols = ["date","state_name","avg_market_price","market_price_imputed_flag","total_generation_mwh"]
p = pd.read_csv(PANEL, usecols=cols, parse_dates=["date"])
rep["panel_rows"], rep["panel_states"] = len(p), int(p.state_name.nunique())
rep["panel_date_range"] = [str(p.date.min().date()), str(p.date.max().date())]

# Is the price identical across states on the same date?
g = p.groupby("date")["avg_market_price"].agg(["nunique","mean","std","count"])
rep["dates_total"] = int(len(g))
rep["dates_price_identical_across_states"] = int((g["nunique"] == 1).sum())
rep["pct_dates_price_identical"] = round(100*(g["nunique"] == 1).mean(), 2)
rep["mean_within_date_std"] = float(np.nanmean(g["std"]))
# effective sample size implication
rep["price_rows_vs_unique_price_days"] = [int(len(p)), int(g.shape[0])]

# imputation flag by year / split
p["yr"] = p.date.dt.year
flag = p.groupby("yr")["market_price_imputed_flag"].mean().mul(100).round(2)
rep["price_imputed_pct_by_year"] = {int(k): float(v) for k, v in flag.items()}
test = p[p.date >= "2024-01-01"]
rep["price_imputed_pct_test"] = round(100*test.market_price_imputed_flag.mean(), 2)
rep["price_at_cap_10000_pct_test"] = round(100*(test.avg_market_price >= 9999.99).mean(), 2)

# ---------- raw 15-min blocks ----------
files = sorted(glob.glob(os.path.join(RAW, "*", "*.csv")))
rep["raw_files"] = len(files)
frames = []
for f in files:
    try:
        d = pd.read_csv(f)
    except Exception as e:
        continue
    if "mcp_rs/mwh" not in d.columns or "delivery" not in d.columns:
        continue
    d = d[["delivery", "mcp_rs/mwh"] + [c for c in ["mcv_mwh","purchase_bid_mwh","sell_bid_mwh"] if c in d.columns]]
    d["date"] = pd.to_datetime(d["delivery"], format="%d-%m-%Y", errors="coerce")
    frames.append(d)
raw = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
raw = raw.rename(columns={"mcp_rs/mwh": "mcp"})
raw["mcp"] = pd.to_numeric(raw["mcp"], errors="coerce")
rep["raw_blocks"] = int(len(raw))
rep["raw_date_range"] = [str(raw.date.min().date()), str(raw.date.max().date())]
rep["raw_mcp_nan_pct"] = round(100*raw.mcp.isna().mean(), 3)

daily = raw.groupby("date")["mcp"].agg(blocks="size", mean="mean", std="std", mn="min", mx="max",
                                        p95=lambda s: s.quantile(0.95)).reset_index()
rep["raw_days"] = int(len(daily))
rep["raw_days_with_96_blocks"] = int((daily.blocks == 96).sum())
rep["raw_blocks_per_day_median"] = float(daily.blocks.median())
rep["raw_daily_mcp_mean"] = round(float(daily["mean"].mean()), 1)
rep["raw_intraday_std_mean"] = round(float(daily["std"].mean()), 1)
rep["raw_intraday_range_mean"] = round(float((daily.mx - daily.mn).mean()), 1)
# calendar gaps in raw coverage
full = pd.date_range(raw.date.min(), raw.date.max(), freq="D")
missing = sorted(set(full) - set(daily.date))
rep["raw_missing_days"] = len(missing)
rep["raw_missing_examples"] = [str(m.date()) for m in missing[:10]]

# ---------- panel vs raw ----------
pp = p.groupby("date")["avg_market_price"].mean().reset_index().rename(columns={"avg_market_price":"panel"})
m = pp.merge(daily[["date","mean","mx","p95"]], on="date", how="left")
both = m.dropna(subset=["mean"])
rep["dates_overlap_panel_raw"] = int(len(both))
for name, col in [("raw_daily_mean","mean"), ("raw_daily_max","mx"), ("raw_daily_p95","p95")]:
    d = both["panel"] - both[col]
    rep[f"panel_vs_{name}"] = {"corr": round(float(both["panel"].corr(both[col])), 4),
                                "MAE": round(float(d.abs().mean()), 1),
                                "exact_match_pct": round(100*float((d.abs() < 1e-6).mean()), 2)}
# panel dates with no raw coverage (these must be the imputed ones)
no_raw = m[m["mean"].isna()]
rep["panel_dates_without_raw_price"] = int(len(no_raw))
rep["panel_dates_without_raw_examples"] = [str(x.date()) for x in no_raw.date.head(8)]
test_no_raw = no_raw[no_raw.date >= "2024-01-01"]
rep["test_dates_without_raw_price"] = int(len(test_no_raw))
rep["test_dates_total"] = int((pp.date >= "2024-01-01").sum())

daily.to_csv(os.path.join(OUT, "raw_daily_mcp.csv"), index=False)
with open(os.path.join(OUT, "A1_price_provenance.json"), "w") as fh:
    json.dump(rep, fh, indent=2, default=str)
for k, v in rep.items():
    print(f"{k}: {v}")
