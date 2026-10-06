"""dev2 smoke test for the combination and evaluation steps: runs steps 5-9 on a tiny slice in results/dev2_smoke
(T5 for O2 and O3, 60 + 60 days for O5, 3000 rows for O4, a handful of bootstrap and placebo draws)."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, REFUSED_DEV2_DIR="dev2_smoke")
failed = []
for script in ("d2_05_ensemble.py", "d2_06_o2_eval.py", "d2_07_o3_eval.py", "d2_08_o4.py", "d2_09_o5.py"):
    t0 = time.time()
    print("=== smoke:", script, flush=True)
    rc = subprocess.run([sys.executable, "-u", os.path.join(HERE, script), "--smoke"], env=env).returncode
    print(f"=== smoke end rc={rc} {time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        failed.append(script)
print("SMOKE_FAILED " + ",".join(failed) if failed else "SMOKE_OK", flush=True)
sys.exit(1 if failed else 0)
