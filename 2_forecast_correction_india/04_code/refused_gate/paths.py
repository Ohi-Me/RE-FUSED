"""Folders of the gated-correction programme.

The programme holds two studies of the same method:
  * the Indian study (official Grid-India, CEA and IMD data), which uses the top-level folders;
  * the earlier study on non-Indian data (NYISO, OPSD/ENTSO-E, UCI, ETT, a semi-synthetic grid on OPSD, and the
    development on an aggregator-based India panel), kept in comparison_other_countries/ for comparison.
REFUSED_GATE_STUDY selects one: "foreign" (the default, so the scripts of the earlier study run unchanged) or
"india" (set by every script of the Indian study before it imports this module).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOREIGN = os.path.join(ROOT, "comparison_other_countries")
STUDY = os.environ.get("REFUSED_GATE_STUDY", "foreign")
BASE = ROOT if STUDY == "india" else FOREIGN
RAW = os.path.join(BASE, "03_data", "raw")
PROC = os.path.join(BASE, "03_data", "processed")
RES = os.path.join(BASE, "06_results")
FIG = os.path.join(BASE, "07_figures")
TAB = os.path.join(BASE, "08_tables")
LOGS = os.path.join(BASE, "12_logs")
DESIGN = os.path.join(BASE, "02_design")
PRIOR = os.path.join(FOREIGN, "90_prior_refused6")
# Small inputs carried over from the RE-FUSED-5 stage (an aggregator-based India daily panel and the public ETT
# files), used only by the earlier study's development runs.
_C5 = (os.path.abspath(os.path.join(ROOT, "..", "..", "..", "history", "preserved", "stage5_6_refused5",
                                    "data_small")),
       os.path.abspath(os.path.join(ROOT, "..", "REFUSED5", "data")))
REFUSED5_DATA = next((p for p in _C5 if os.path.isdir(p)), _C5[-1])
# the Indian data and forecasts come from the sibling India programme (official Grid-India, CEA and IMD data only)
INDIA = os.path.abspath(os.path.join(ROOT, "..", "1_india_data_forecasting_scheduling"))
INDIA_PROC = os.path.join(INDIA, "data", "processed")
INDIA_RES = os.path.join(INDIA, "results")
for _d in (PROC, RES, FIG, TAB, LOGS):
    os.makedirs(_d, exist_ok=True)
