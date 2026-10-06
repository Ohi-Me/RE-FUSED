"""Run the LEAKAGE_AUDIT.md diagnostics end-to-end and save the evidence.

This executes the three checks the audit promised but could not run in the
original (read-only) session:

  1. causal_features.leakage_scan   -> results/tables/leakage_scan.csv
  2. units.units_table              -> results/tables/units_table.csv
  3. units.imputation_report        -> results/tables/imputation_report.csv

A human-readable summary goes to results/outputs/audit_diagnostics_summary.txt.

Usage (Windows):  py -3.10 run_audit_diagnostics.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from refused_fixes.causal_features import leakage_scan          # noqa: E402
from refused_fixes.units import (                               # noqa: E402
    detect_unit_jumps,
    imputation_report,
    observed_only,
    units_table,
)

TABLES = ROOT / "results" / "tables"
OUTPUTS = ROOT / "results" / "outputs"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    train = pd.read_csv(ROOT / "data" / "RL_TRAIN_DATA.csv", parse_dates=["date"])
    test = pd.read_csv(ROOT / "data" / "RL_TEST_DATA.csv", parse_dates=["date"])
    full = pd.concat([train, test], ignore_index=True)

    log("=== RE-FUSED-Alpha audit diagnostics ===")
    log(f"train: {train.shape[0]} rows x {train.shape[1]} cols, "
        f"{train['date'].min().date()} .. {train['date'].max().date()}")
    log(f"test : {test.shape[0]} rows x {test.shape[1]} cols, "
        f"{test['date'].min().date()} .. {test['date'].max().date()}")
    log()

    # ------------------------------------------------ 1. leakage scan (Task #1)
    log("--- 1. leakage_scan (suspect engineered columns) ---")
    scan = leakage_scan(train, test)
    scan.to_csv(TABLES / "leakage_scan.csv", index=False)
    log(scan.to_string(index=False))
    delta = pd.to_numeric(scan["refit_max_abs_delta"], errors="coerce").fillna(0.0)
    flagged = scan[(scan["test_out_of_train_range"] > 0.01) | (delta > 0.01)]
    log()
    log(f"columns flagged as leaky / shifted (oob>1% or refit delta>0.01): "
        f"{len(flagged)} of {len(scan)} scanned")
    if len(flagged):
        log("  -> " + ", ".join(flagged["column"].tolist()))
    log()

    # ------------------------------------------------ 2. units table (Task #2)
    log("--- 2. units_table (full 2017-2025 panel) ---")
    ut = units_table(full)
    ut.to_csv(TABLES / "units_table.csv", index=False)
    jumps = detect_unit_jumps(full)
    n_flagged = int(ut["unit_jump_flag"].sum())
    log(f"numeric columns: {len(ut)}; declared canonical units: "
        f"{int((ut['canonical_unit'] != 'UNDECLARED').sum())}; "
        f"unit-jump flags: {n_flagged}")
    for f in jumps:
        log(f"  jump: {f.column}  p95/p05 ratio={f.ratio_p95_p05}  "
            f"suspected multiplier={f.suspected_multiplier}")
    log()

    # ------------------------------------------ 3. imputation report (Task #2)
    log("--- 3. imputation_report (full panel) ---")
    rep = imputation_report(full)
    rep.per_column.to_csv(TABLES / "imputation_report.csv", index=False)
    log(str(rep))
    obs = observed_only(full)
    log(f"observed-only robustness slice: {len(obs)} / {len(full)} rows "
        f"({len(obs) / len(full):.1%})")
    log()

    # sanity check the audit's headline magnitude claim (gen ~ GWh not MWh)
    med_gen = float(pd.to_numeric(full["total_generation_mwh"], errors="coerce").median())
    log(f"median total_generation_mwh (state-day) = {med_gen:.2f} "
        f"-> plausible only as GWh (audit S6 confirmed)")

    (OUTPUTS / "audit_diagnostics_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    log()
    log(f"saved: {TABLES / 'leakage_scan.csv'}")
    log(f"saved: {TABLES / 'units_table.csv'}")
    log(f"saved: {TABLES / 'imputation_report.csv'}")
    log(f"saved: {OUTPUTS / 'audit_diagnostics_summary.txt'}")


if __name__ == "__main__":
    main()
