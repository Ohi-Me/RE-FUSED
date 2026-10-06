"""Runs after the orchestrator: M5c (pre-registered parsimony test), then regenerates ALL_STATS."""
import ctypes, datetime, os, subprocess, time
ROOT = r"D:\\REFUSED5"; RES = os.path.join(ROOT, "results")
PY = r"C:\Users\Ohi\AppData\Local\Programs\Python\Python310\python.exe"
STATUS = os.path.join(RES, "refused6", "PIPELINE_STATUS.md")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4")
def log(m):
    open(STATUS, "a", encoding="utf-8").write(f"- `{datetime.datetime.now():%Y-%m-%d %H:%M:%S}` [post-chain] {m}\n")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
log("waiting for orchestrator to finish (ALL_STATS flag)")
while not os.path.exists(os.path.join(RES, "_allstats_done.txt")):
    time.sleep(60)
for attempt in (1, 2):
    log(f"M5c attempt {attempt} started")
    with open(os.path.join(RES, "M5c.log"), "w", encoding="utf-8") as f:
        rc = subprocess.run([PY, "-u", "src/refused6/m5c_parsimony_selection.py", "AB"], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc == 0 and os.path.exists(os.path.join(RES, "refused6", "M5c_parsimony.csv")):
        log("M5c complete"); break
    log(f"M5c attempt {attempt} FAILED (exit {rc})")
with open(os.path.join(RES, "final_stats.log"), "a", encoding="utf-8") as f:
    subprocess.run([PY, "-u", "src/refused6/final_stats.py"], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
open(os.path.join(RES, "_m5c_done.txt"), "w").write(str(datetime.datetime.now()))
log("ALL_STATS regenerated with M5c; post-chain finished")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
