"""Run M9b (tuned neural gate, H-M9b) then regenerate ALL_STATS."""
import ctypes, datetime, os, subprocess
ROOT = r"D:\\REFUSED5"; RES = os.path.join(ROOT, "results")
PY = r"C:\Users\Ohi\AppData\Local\Programs\Python\Python310\python.exe"
STATUS = os.path.join(RES, "refused6", "PIPELINE_STATUS.md")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4")
def log(m):
    open(STATUS, "a", encoding="utf-8").write("- `" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "` [post-chain-4] " + m + "\n")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
for attempt in (1, 2):
    log("M9b attempt %d started" % attempt)
    with open(os.path.join(RES, "M9b.log"), "w", encoding="utf-8") as f:
        rc = subprocess.run([PY, "-u", "src/refused6/m9b_neural_tuned.py"], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc == 0 and os.path.exists(os.path.join(RES, "refused6", "M9b_neural_tuned.csv")):
        log("M9b complete"); break
    log("M9b attempt %d FAILED (exit %d)" % (attempt, rc))
with open(os.path.join(RES, "final_stats.log"), "a", encoding="utf-8") as f:
    subprocess.run([PY, "-u", "src/refused6/final_stats.py"], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
open(os.path.join(RES, "_m9b_done.txt"), "w").write(str(datetime.datetime.now()))
log("ALL_STATS regenerated with M9b; post-chain-4 finished")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
