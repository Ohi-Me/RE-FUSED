"""Additional analysis, not pre-registered (delay study, Nov 2026): E4b, Chronos-2 with instant publication.

Runs dev2/d2_03_chronos.py (chronos2) unchanged, with every availability lag set to 0 in refused.o2data. Outputs are
redirected by REFUSED_DEV2_DIR=extra/delay/E4b (set in the PBS file), so nothing is written into the confirmatory
folders; the frozen cross-learning choice is read from results/dev2 as in the confirmatory run. GPU, inference only.
Usage: REFUSED_DEV2_DIR=extra/delay/E4b python codes/scripts/extra/x17_chronos_instant.py [T1 T2 ...]
"""
import os
import runpy
import sys

assert os.environ.get("REFUSED_DEV2_DIR", "").startswith("extra/"), "set REFUSED_DEV2_DIR=extra/delay/E4b"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from refused import o2data as O  # noqa: E402

REG = O.lags(False)
O.lags = lambda instant=False: {k: 0 for k in REG}      # instant publication for every column
tids = ",".join(sys.argv[1:] or list(O.TARGETS))
script = os.path.join(HERE, "..", "dev2", "d2_03_chronos.py")
sys.argv = [script, "chronos2", tids]
runpy.run_path(script, run_name="__main__")
