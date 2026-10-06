"""dev2 smoke test: run every new training script on a tiny slice (T5, one seed, a few steps or cutoffs) so that
code errors show up in minutes. Outputs go to results/dev2_smoke and are not used for anything else.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
env = dict(os.environ, REFUSED_DEV2_DIR="dev2_smoke", REFUSED_O3_OUT=os.path.join(ROOT, "results", "dev2_smoke", "o3"),
           REFUSED_O3_MAX_EPOCHS="1")
steps = [
    [os.path.join(HERE, "d2_01_xgb.py"), "T5", "--smoke"],
    [os.path.join(HERE, "d2_02_nf_models.py"), "nhits,tide,bitcn,nbeatsx", "T5", "--smoke"],
    [os.path.join(HERE, "d2_03_chronos.py"), "chronos2,chronos_base", "T5", "--smoke"],
    [os.path.join(HERE, "..", "o3", "o3_01_lamcag.py"), "T5", "mcag_instance_sd", "1"],
]
failed = []
for cmd in steps:
    t0 = time.time()
    print("=== smoke:", os.path.basename(cmd[0]), " ".join(cmd[1:]), flush=True)
    rc = subprocess.run([sys.executable, "-u"] + cmd, env=env).returncode
    print(f"=== smoke end rc={rc} {time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        failed.append(os.path.basename(cmd[0]))
print("SMOKE_FAILED " + ",".join(failed) if failed else "SMOKE_OK", flush=True)
sys.exit(1 if failed else 0)
