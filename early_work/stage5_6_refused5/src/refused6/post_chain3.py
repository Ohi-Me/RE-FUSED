"""After M7b: run M9 (neural gate, H-M9) and M10 (rolling origin, H-M10), then regenerate ALL_STATS."""
import ctypes, datetime, os, subprocess, time
ROOT = r"D:\\REFUSED5"; RES = os.path.join(ROOT, "results")
PY = r"C:\Users\Ohi\AppData\Local\Programs\Python\Python310\python.exe"
STATUS = os.path.join(RES, "refused6", "PIPELINE_STATUS.md")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4")
def log(m):
    open(STATUS, "a", encoding="utf-8").write("- `" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "` [post-chain-3] " + m + "\n")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
log("waiting for M7b post-chain to finish")
while not os.path.exists(os.path.join(RES, "_m7b_done.txt")):
    time.sleep(60)
for name, script, csv in [("M9", "m9_neural_gate.py", "M9_neural_gate.csv"), ("M10", "m10_rolling_origin.py", "M10_rolling_origin.csv")]:
    for attempt in (1, 2):
        log("%s attempt %d started" % (name, attempt))
        with open(os.path.join(RES, name + ".log"), "w", encoding="utf-8") as f:
            rc = subprocess.run([PY, "-u", "src/refused6/" + script], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0 and os.path.exists(os.path.join(RES, "refused6", csv)):
            log(name + " complete"); break
        log("%s attempt %d FAILED (exit %d)" % (name, attempt, rc))
    open(os.path.join(RES, "_%s_done.txt" % name.lower()), "w").write(str(datetime.datetime.now()))
with open(os.path.join(RES, "final_stats.log"), "a", encoding="utf-8") as f:
    subprocess.run([PY, "-u", "src/refused6/final_stats.py"], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
open(os.path.join(RES, "_m9m10_done.txt"), "w").write(str(datetime.datetime.now()))
log("ALL_STATS regenerated with M9/M10; post-chain-3 finished")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
