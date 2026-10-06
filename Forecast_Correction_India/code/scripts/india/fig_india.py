"""Figure of the pre-registered Indian hourly study: the best squared-error skill any gate design reached, for each
day-ahead forecast and series (test months 1 Mar - 13 Sep 2026). Reads results/india/confirm/analysis/IH5_strength.csv.
Output: figures/india/correction_skill_hourly.{png,pdf}; with --paper also the same chart without its title
(correction_skill_hourly_paper.pdf), for use under a caption
Usage:  python code/scripts/india/fig_india.py      (run on the H100: hpc/india_fig.pbs)
"""
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "results", "india", "confirm", "analysis", "IH5_strength.csv")
OUT = os.path.join(ROOT, "figures", "india")
NAMES = {"snaive168": "same hour last week", "lag48": "value 48 h earlier", "lgbm": "LightGBM", "dlinear": "DLinear",
         "tide": "TiDE", "patchtst": "PatchTST", "nhits": "N-HiTS", "chronos": "Chronos-Bolt"}
SERIES = [("demand", "Demand met"), ("net_demand", "Net demand (minus wind, solar)"), ("wind", "Wind"),
          ("solar", "Solar")]


def main():
    d = pd.read_csv(SRC)
    order = list(NAMES)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, (sid, title) in zip(axes, SERIES):
        g = d[d.sid == sid].set_index("baseline").reindex(order)
        y = range(len(order))[::-1]
        ax.barh(list(y), 100 * g.best_skill, color="#2a6fb0", height=0.62)
        for yi, v in zip(y, g.best_skill):
            ax.text(100 * v + 0.6, yi, f"{100 * v:.1f}", va="center", fontsize=8.5, color="#374151")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0, 40)
        ax.grid(axis="x", color="#e5e7eb")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("error removed (%)", fontsize=9)
    axes[0].set_yticks(list(range(len(order)))[::-1])
    axes[0].set_yticklabels([NAMES[b] for b in order], fontsize=9)
    os.makedirs(OUT, exist_ok=True)
    if "--paper" in sys.argv:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "correction_skill_hourly_paper.pdf"))
    fig.suptitle("All-India hourly forecasts, 1 Mar – 13 Sep 2026: squared error removed by the best gated correction",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"correction_skill_hourly.{ext}"), dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
