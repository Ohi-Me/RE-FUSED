"""Compute reference-baseline scores on the RE-FUSED test split.

Anchors the deep models against naive / seasonal-naive / moving-average
forecasters on the OBSERVABLE targets (LEAKAGE_AUDIT.md par.3 — synthetic
composites are excluded from forecast-skill claims).

Usage (Windows):  py -3.10 run_baselines.py
Writes: results/tables/baseline_metrics.csv
        results/outputs/baseline_summary.txt
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from refused_fixes.baselines import evaluate_baselines  # noqa: E402

OBSERVABLE_TARGETS = ["total_generation_mwh", "avg_market_price"]


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "RL_TRAIN_DATA.csv", parse_dates=["date"])
    test = pd.read_csv(ROOT / "data" / "RL_TEST_DATA.csv", parse_dates=["date"])

    table = evaluate_baselines(train, test, OBSERVABLE_TARGETS, m=7, ma_window=7)

    tables = ROOT / "results" / "tables"
    outputs = ROOT / "results" / "outputs"
    tables.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "baseline_metrics.csv", index=False)

    lines = [
        "=== Reference baselines on TEST 2024-2025 (per-state avg, m=7) ===",
        "Deep models must beat snaive (MASE < snaive's) to claim forecast skill.",
        "",
        table.to_string(index=False),
        "",
        f"saved: {tables / 'baseline_metrics.csv'}",
    ]
    text = "\n".join(lines)
    print(text)
    (outputs / "baseline_summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
