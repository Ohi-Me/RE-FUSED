"""Rehearsal before the freeze (development phase only, no reserved data).

Part 1 runs the round-2 scripts in frozen mode (REFUSED_USE_FROZEN=1) on small slices, so the code that reads the
development choices is checked before it is used for real. Outputs go to results/dev2_frozen_check.
Part 2 runs the hypothesis scoring script on the real development results, which gives a rehearsal table (the
development year has been seen, so this is a code check, not a test).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = os.path.join(HERE, "..", "dev2")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
frozen_env = dict(os.environ, REFUSED_USE_FROZEN="1", REFUSED_DEV2_DIR="dev2_frozen_check",
                  REFUSED_O3_OUT=os.path.join(ROOT, "results", "dev2_frozen_check", "o3"), REFUSED_O3_MAX_EPOCHS="1")
steps = [
    (frozen_env, [os.path.join(D2, "d2_01_xgb.py"), "T5", "--smoke"]),
    (frozen_env, [os.path.join(D2, "d2_02_nf_models.py"), "nhits,tide,bitcn,nbeatsx", "T5", "--smoke"]),
    (frozen_env, [os.path.join(D2, "d2_03_chronos.py"), "chronos2,chronos_base", "T5", "--smoke"]),
    (frozen_env, [os.path.join(HERE, "..", "o3", "o3_01_lamcag.py"), "T5", "mcag_instance_sd,fusion_fixed_sd", "1"]),
    (frozen_env, [os.path.join(D2, "d2_05_ensemble.py"), "T5", "--smoke"]),
    (frozen_env, [os.path.join(D2, "d2_06_o2_eval.py"), "T5", "--smoke"]),
    (frozen_env, [os.path.join(D2, "d2_07_o3_eval.py"), "T5", "--smoke"]),
    (frozen_env, [os.path.join(D2, "d2_08_o4.py"), "--smoke"]),
    (frozen_env, [os.path.join(D2, "d2_09_o5.py"), "--smoke"]),
    (frozen_env, [os.path.join(D2, "d2_11_ppo.py"), "--smoke"]),
    (dict(os.environ), [os.path.join(HERE, "c01_hypotheses.py")]),
]
failed = []
for env, cmd in steps:
    t0 = time.time()
    print("=== rehearsal:", os.path.basename(cmd[0]), " ".join(cmd[1:]), flush=True)
    rc = subprocess.run([sys.executable, "-u"] + cmd, env=env).returncode
    print(f"=== rehearsal end rc={rc} {time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        failed.append(os.path.basename(cmd[0]))
print("REHEARSAL_FAILED " + ",".join(failed) if failed else "REHEARSAL_OK", flush=True)
sys.exit(1 if failed else 0)
