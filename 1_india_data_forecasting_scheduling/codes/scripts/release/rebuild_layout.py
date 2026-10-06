"""Copy the files of the RE-FUSED Zenodo record back into the project layout that the code expects.

The record is organised for reading (data, code, results, documentation). The scripts, however, find each other
and their data through fixed relative paths (codes/refused, codes/scripts/<group>, data/processed, results/...).
This script makes that layout in a new folder, so the driver and the scripts run unchanged.

Usage: python code/reproducibility/release/rebuild_layout.py <target folder>
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# (path in the record, path in the project layout)
MAP = [
    ("code/refused_library", "codes/refused"),
    ("code/preprocessing/acquire", "codes/scripts/acquire"),
    ("code/preprocessing/build", "codes/scripts/build"),
    ("code/forecasting/o2", "codes/scripts/o2"),
    ("code/forecasting/o3", "codes/scripts/o3"),
    ("code/forecasting/dev2", "codes/scripts/dev2"),
    ("code/analysis/o4", "codes/scripts/o4"),
    ("code/analysis/o5", "codes/scripts/o5"),
    ("code/analysis/confirm", "codes/scripts/confirm"),
    ("code/analysis/prereg", "codes/scripts/prereg"),
    ("code/analysis/paper", "codes/scripts/paper"),
    ("code/analysis/audit", "codes/scripts/audit"),
    ("code/reproducibility/release", "codes/scripts/release"),
    ("code/reproducibility/tests", "codes/tests"),
    ("code/reproducibility/design", "codes/design"),
    ("code/reproducibility/environment", "hpc"),   # the public record keeps only the environment recipe
    ("code/reproducibility/hpc", "hpc"),           # our complete local copy keeps every job file
    ("code/reproducibility/run_all.py", "run_all.py"),
    ("code/reproducibility/requirements-h100.txt", "requirements-h100.txt"),
    ("code/reproducibility/env.example", ".env.example"),
    ("data/processed_panel/refused_state_day.parquet", "data/processed/refused_state_day.parquet"),
    ("data/processed_panel/refused_entities.csv", "data/processed/refused_entities.csv"),
    ("data/processed_panel/CHECKSUM.txt", "data/processed/CHECKSUM.txt"),
    ("data/data_dictionary", "data/docs"),
    ("data/interim_tables", "data/interim"),
    ("results/frozen_results/prereg_choices", "results"),
    ("results/frozen_results/confirmatory", "results"),
    ("results/tables/latex", "results/paper/confirm/tables"),
    ("results/tables/numbers.tex", "results/paper/confirm/numbers.tex"),
    ("results/tables/breakdowns", "results/audit/strata"),
]


OPTIONAL = {"code/reproducibility/hpc", "code/reproducibility/environment", "code/reproducibility/env.example",
            "code/analysis/prereg"}


def copy(src, dst):
    if os.path.isdir(src):
        for base, _, files in os.walk(src):
            for f in files:
                s = os.path.join(base, f)
                copy(s, os.path.join(dst, os.path.relpath(s, src)))
    elif os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    elif os.path.relpath(src, RECORD).replace(os.sep, "/") not in OPTIONAL:
        print("not in this record, skipped:", os.path.relpath(src, RECORD))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = os.path.abspath(sys.argv[1])
    for a, b in MAP:
        copy(os.path.join(RECORD, *a.split("/")), os.path.join(target, *b.split("/")))
    print("project layout written to", target)
    print("next: cd", target, "and run  python run_all.py plan")
