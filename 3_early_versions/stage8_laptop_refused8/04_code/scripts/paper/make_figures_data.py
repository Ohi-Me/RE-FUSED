"""Figures for the O1 data descriptor (10_manuscript/data_descriptor/figures/*.pdf and .png).

Palette and marks follow the validated reference palette (categorical slots in fixed order, single-hue sequential blue
ramp, hairline recessive grid, 2 px lines, no text in series colours). Blinding: figures that show target values use
dates up to 2025-03-31 only; the coverage figure uses availability flags only.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8 import entities as E  # noqa: E402
from refused8.paths import DOCS, INTERIM, PROC, RAW, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "10_manuscript", "data_descriptor", "figures")
BLIND_END = pd.Timestamp("2025-03-31")
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": SURF, "axes.titlecolor": INK, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "legend.frameon": False, "savefig.dpi": 300, "axes.axisbelow": True})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_axis_off()
    cols = [("Official\nsources", ["Grid-India\nPSP reports", "Grid-India\nDSM rate files",
                                  "CEA NPP\ngeneration", "CEA daily\nRE reports", "CEA CO$_2$\ndatabase",
                                  "IMD gridded\nweather"]),
            ("Raw layer", ["scripted\ndownload", "URL, time,\nbytes, SHA-256", "per-file\nmanifests"]),
            ("Parsing", ["layout-aware\nparsers", "date and\nscope checks", "parse logs"]),
            ("Interim\ntables", ["State, region,\nstation-day", "bid-area\nprices",
                                "emission factors,\nweather"]),
            ("Panel (O1)", ["State-day,\n107 columns", "QC flags\n(never imputed)",
                            "field registry\nwith lags", "content\nchecksum"])]
    x0, w, gap = 0.0, 0.175, 0.03
    for i, (title, items) in enumerate(cols):
        x = x0 + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0.0), w, 1.0, fc="#f0efec" if i != 4 else "#cde2fb", ec="none",
                                   transform=ax.transAxes))
        ax.text(x + w / 2, 0.92, title, ha="center", va="center", weight="bold", color=INK, fontsize=8,
                transform=ax.transAxes, linespacing=1.0)
        for k, it in enumerate(items):
            ax.text(x + w / 2, 0.76 - k * 0.135, it, ha="center", va="center", color=INK2, fontsize=6.6,
                    transform=ax.transAxes, linespacing=1.0)
        if i < len(cols) - 1:
            ax.annotate("", xy=(x + w + gap * 0.95, 0.5), xytext=(x + w + gap * 0.05, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    save(fig, "fig1_pipeline")


def fig_coverage():
    P = pd.read_parquet(os.path.join(PROC, "refused8_state_day.parquet"),
                        columns=["date", "entity", "avail_psp", "avail_npp", "avail_re", "prc_dam_acp", "wx_tmax_c",
                                 "re_ctrl_total_re_gwh"])
    P["price"], P["weather"], P["re_ctrl"] = P.prc_dam_acp.notna(), P.wx_tmax_c.notna(), P.re_ctrl_total_re_gwh.notna()
    rows = [("Grid-India PSP (demand, deviation)", "avail_psp"), ("Grid-India DSM files (prices)", "price"),
            ("CEA NPP (conventional generation)", "avail_npp"), ("CEA RE, State + ISGS", "avail_re"),
            ("CEA RE, State control area", "re_ctrl"), ("IMD gridded weather", "weather")]
    m = P.groupby(P.date.dt.to_period("M"))[[r[1] for r in rows]].mean()
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    im = ax.imshow(m.T.to_numpy(), aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK2)
    idx = [i for i, p in enumerate(m.index) if p.month == 4]
    ax.set_xticks(idx, [f"FY{str(m.index[i].year)[2:]}-{str(m.index[i].year + 1)[2:]}" for i in idx], rotation=0)
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    rs = [i for i, p in enumerate(m.index) if p == pd.Period("2025-04", "M")][0]
    ax.axvline(rs - 0.5, color=INK, lw=1.0)
    ax.text(rs + 0.5, -0.9, "reserved (blinded) →", color=INK2, fontsize=7, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("share of State-days available", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "fig2_coverage")


def fig_lags():
    L = []
    for f in glob.glob(os.path.join(RAW, "grid_india_psp", "LISTING_psp_*.json")):
        L += json.load(open(f))
    li = pd.DataFrame(L)
    li["created"] = pd.to_datetime(li.CreatedOn, format="%d-%m-%Y %H:%M", errors="coerce")
    li["rep"] = pd.to_datetime(li.Field1, format="%d-%m-%Y", errors="coerce")
    lag_h = ((li.created - li.rep).dt.total_seconds() / 3600)[li.rep >= "2025-04-01"].dropna()
    d = pd.DataFrame(json.load(open(os.path.join(RAW, "grid_india_dsm", "LISTING_dsm.json"))))
    d = d[d.Title_.str.match(r"DSM Rate \d{2}-\d{2}-\d{4}")].copy()
    d["data_date"] = pd.to_datetime(d.Title_.str[-10:], format="%d-%m-%Y", errors="coerce")
    d["created"] = pd.to_datetime(d.CreatedOn, format="%d-%m-%Y %H:%M", errors="coerce")
    lag_d = ((d.created - d.data_date).dt.total_seconds() / 86400)[d.data_date >= "2025-01-01"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    ax = axes[0]
    ax.hist(lag_h.clip(upper=96), bins=np.arange(0, 97, 3), color=S1, rwidth=0.85)
    med = lag_h.median()
    ax.axvline(med, color=INK, lw=1)
    ax.text(med + 2, ax.get_ylim()[1] * 0.9, f"median {med:.1f} h", color=INK, fontsize=7.5)
    ax.set_xlabel("hours after the end of the data day (capped at 96)")
    ax.set_ylabel("reports")
    ax.set_title("Grid-India PSP report (n = %d)" % len(lag_h), loc="left")
    ax = axes[1]
    ax.hist(lag_d.clip(upper=30), bins=np.arange(0, 31, 1), color=S1, rwidth=0.85)
    med = lag_d.median()
    ax.axvline(med, color=INK, lw=1)
    ax.text(med + 0.6, ax.get_ylim()[1] * 0.9, f"median {med:.1f} d", color=INK, fontsize=7.5)
    ax.set_xlabel("days after the data date (capped at 30)")
    ax.set_ylabel("daily files")
    ax.set_title("Grid-India DSM rate file (n = %d)" % len(lag_d), loc="left")
    fig.tight_layout()
    save(fig, "fig3_publication_lags")


def fig_validation():
    s = pd.read_parquet(os.path.join(INTERIM, "npp_state_day.parquet"))
    x = pd.read_csv(os.path.join(RAW, "legacy_refused0", "IDP", "daily-power-generation.csv"),
                    usecols=["date", "state_name", "todays_gen_act"])
    x["date"] = pd.to_datetime(x.date)
    x["entity"] = x.state_name.map(E.entity_id)
    a = x.dropna(subset=["entity"]).groupby(["date", "entity"]).todays_gen_act.sum().rename("idp")
    j1 = pd.concat([a, s.set_index(["date", "entity"]).act_gwh_total.rename("ours")], axis=1, join="inner").reset_index()
    j1 = j1[j1.date <= BLIND_END]
    r = pd.read_parquet(os.path.join(INTERIM, "cea_re_state_day.parquet"))
    y = pd.read_csv(os.path.join(RAW, "legacy_refused0", "IDP", "daily-renewable-energy-generation.csv"))
    y["date"] = pd.to_datetime(y.date)
    y["entity"] = y.state_name.map(E.entity_id)
    y = y.dropna(subset=["entity"]).groupby(["date", "entity"]).total_renewable_energy.sum().rename("idp")
    j2 = r.set_index(["date", "entity"])[["all_total_re_gwh"]].join(y, how="inner").dropna().reset_index()
    j2 = j2[j2.date <= BLIND_END].rename(columns={"all_total_re_gwh": "ours"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, j, title in [(axes[0], j1, "Conventional generation (CEA NPP)"), (axes[1], j2, "RE generation (CEA daily RE)")]:
        v = j[(j.ours > 0.05) & (j.idp > 0.05)]
        hb = ax.hexbin(np.log10(v.idp), np.log10(v.ours), gridsize=60, cmap=SEQ, bins="log", mincnt=1, linewidths=0)
        lo, hi = -1.3, np.log10(max(v.idp.max(), v.ours.max())) + 0.1
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
        ticks = [0.1, 1, 10, 100, 1000]
        ax.set_xticks(np.log10(ticks), [f"{t:g}" for t in ticks])
        ax.set_yticks(np.log10(ticks), [f"{t:g}" for t in ticks])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        share = float(((j.ours - j.idp).abs() <= 0.01 * j.idp.abs().clip(lower=1)).mean())
        ax.set_title(title, loc="left")
        ax.text(0.03, 0.93, f"{len(j):,} State-days; {100 * share:.1f}% within 1%", transform=ax.transAxes, color=INK2,
                fontsize=7.5)
        ax.set_xlabel("pilot copy (India Data Portal), GWh")
        ax.set_ylabel("this dataset, GWh")
        ax.grid(False)
    cb = fig.colorbar(hb, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("State-days per cell (log)", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "fig4_validation")


def fig_example(entity="RJ"):
    P = pd.read_parquet(os.path.join(PROC, "refused8_state_day.parquet"))
    g = P[(P.entity == entity) & (P.date <= BLIND_END)].set_index("date").sort_index()
    wk = g.resample("W").mean(numeric_only=True)
    panels = [("dem_energy_met_gwh", "Energy met (GWh/day)"), ("dev_od_ud_gwh", "Over(+)/under(−) drawal (GWh/day)"),
              ("re_all_total_re_gwh", "RE generation, State + ISGS (GWh/day)"), ("prc_dam_acp", "DAM price, bid area (₹/MWh)"),
              ("co2_ci_gen_acc", "Carbon intensity of generation (t/MWh)"), ("wx_tmax_c", "Maximum temperature (°C)")]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.2), sharex=True)
    for ax, (c, t) in zip(axes.ravel(), panels):
        ax.plot(wk.index, wk[c], color=S1, lw=1.2)
        if c == "dev_od_ud_gwh":
            ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(t, loc="left", fontsize=8)
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=0)
    name = E.ENTITIES[entity][0]
    fig.suptitle(f"{name}: weekly means of selected panel columns, Apr 2018 – Mar 2025", x=0.01, ha="left",
                 color=INK, fontsize=9, weight="bold")
    fig.tight_layout()
    save(fig, "fig5_example_state")


if __name__ == "__main__":
    for fn in (fig_pipeline, fig_coverage, fig_lags, fig_validation, fig_example):
        fn()
        print(fn.__name__, "ok", flush=True)
