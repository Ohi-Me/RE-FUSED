"""The semi-synthetic crossing-condition grid (x1_phase_diagram.py) on the Indian substrate: 34 State control areas,
daily drawal, seasonal-naive forecast. Runs the same grid code with the Indian study's folders.
Usage: python code/scripts/india/x1_india.py --part partition --source india_daily_dev --tag dev
"""
import os
import runpy
import sys

os.environ["REFUSED_GATE_STUDY"] = "india"
sys.argv[0] = os.path.join(os.path.dirname(__file__), "..", "dev", "x1_phase_diagram.py")
runpy.run_path(sys.argv[0], run_name="__main__")
