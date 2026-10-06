"""Hold Windows awake (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) until logs/_ALL_DONE exists or 48 h pass."""
import ctypes, os, time
FLAG = os.path.join(os.path.dirname(__file__), "..", "..", "Other_Countries", "logs", "_ALL_DONE")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
t0 = time.time()
while not os.path.exists(FLAG) and time.time() - t0 < 48 * 3600:
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
