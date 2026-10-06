"""Project paths (relative to the repository root; no absolute paths)."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "03_data", "raw")
INTERIM = os.path.join(ROOT, "03_data", "interim")
PROC = os.path.join(ROOT, "03_data", "processed")
DOCS = os.path.join(ROOT, "03_data", "docs")
RES = os.path.join(ROOT, "06_results")
FIG = os.path.join(ROOT, "07_figures")
TAB = os.path.join(ROOT, "08_tables")
LOGS = os.path.join(ROOT, "12_logs")
DESIGN = os.path.join(ROOT, "02_design")

for _d in (RAW, INTERIM, PROC, DOCS, RES, FIG, TAB, LOGS):
    os.makedirs(_d, exist_ok=True)
