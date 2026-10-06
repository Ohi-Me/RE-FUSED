"""Project paths for the RE-FUSED layout (relative to the RE-FUSED root; no absolute paths).

RE-FUSED/
  data/     raw (optional), interim, processed, docs, hf_cache
  codes/    refused package, scripts, tests, design (protocols and pre-registration)
  results/  <phase>/o2 … o5 outputs, tables, figures, paper
  logs/     run_all.py stage logs and PBS job logs

The root can be overridden with the environment variable REFUSED_ROOT, and the raw-data folder with REFUSED_RAW (the
panel rebuild check writes to results/rebuild while reading data/raw).
"""
import os

ROOT = os.environ.get("REFUSED_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
RAW = os.environ.get("REFUSED_RAW") or os.path.join(DATA, "raw")
INTERIM = os.path.join(DATA, "interim")
PROC = os.path.join(DATA, "processed")
DOCS = os.path.join(DATA, "docs")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
PAPER = os.path.join(RES, "paper")
LOGS = os.path.join(ROOT, "logs")
DESIGN = os.path.join(ROOT, "codes", "design")

for _d in (INTERIM, PROC, DOCS, RES, FIG, TAB, PAPER, LOGS):
    os.makedirs(_d, exist_ok=True)
