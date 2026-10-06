"""RE-FUSED-6 unattended orchestrator.

Runs as an independent OS process, so it keeps going after the chat session ends.
  * holds a Windows keep-awake request (SetThreadExecutionState) for its whole lifetime;
    released automatically when it exits; changes no system settings
  * waits for the M5b full sweep already in progress (started by an earlier launcher), and re-runs it
    only if that process dies without producing output
  * then runs M8 -> M7 -> final statistics, each at most twice, skipping any stage already complete
  * writes a live status file and, at the end, a handoff note for the next session

Status:  results/refused6/PIPELINE_STATUS.md      Handoff: docs/HANDOFF_NEXT_SESSION.md
"""
import ctypes, datetime, json, os, subprocess, sys, time

ROOT = r"D:\\REFUSED5"
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "refused6")
PY = r"C:\Users\Ohi\AppData\Local\Programs\Python\Python310\python.exe"
STATUS = os.path.join(OUT, "PIPELINE_STATUS.md")
STATE = os.path.join(OUT, "pipeline_state.json")
HANDOFF = os.path.join(ROOT, "docs", "HANDOFF_NEXT_SESSION.md")

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4")

STAGES = [
    dict(name="M5b_full", script="src/refused6/m5b_selection_fixed.py", args=["full"],
         log="M5b_full.log", flag="_m5b_full_done.txt", output="refused6/M5b_selection_full.csv",
         external_marker="m5b_selection_fixed.py full"),
    dict(name="M8", script="src/refused6/m8_cross_domain.py", args=[],
         log="M8.log", flag="_m8_done.txt", output="refused6/M8_cross_domain.csv"),
    dict(name="M7", script="src/refused6/m7_risk_control.py", args=[],
         log="M7.log", flag="_m7_done.txt", output="refused6/M7_risk_control.csv"),
    dict(name="ALL_STATS", script="src/refused6/final_stats.py", args=[],
         log="final_stats.log", flag="_allstats_done.txt", output="refused6/ALL_STATS.md"),
]
state = {s["name"]: "pending" for s in STAGES}


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"- `{now()}` {msg}"
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    json.dump(dict(updated=now(), stages=state), open(STATE, "w"), indent=2)
    print(line, flush=True)


def done(st):
    return os.path.exists(os.path.join(RES, st["flag"])) and os.path.exists(os.path.join(RES, st["output"]))


def external_running(marker):
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=60).stdout
        return marker in out
    except Exception:
        return False


def run_stage(st):
    for attempt in (1, 2):
        log(f"**{st['name']}** attempt {attempt} started")
        state[st["name"]] = f"running (attempt {attempt})"
        with open(os.path.join(RES, st["log"]), "w", encoding="utf-8") as lf:
            rc = subprocess.run([PY, "-u", st["script"], *st["args"]], cwd=ROOT, env=ENV,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc == 0 and os.path.exists(os.path.join(RES, st["output"])):
            open(os.path.join(RES, st["flag"]), "w").write(now())
            state[st["name"]] = "complete"
            log(f"**{st['name']}** complete (exit 0)")
            return True
        tail = ""
        try:
            tail = open(os.path.join(RES, st["log"]), encoding="utf-8", errors="ignore").read()[-600:]
        except Exception:
            pass
        log(f"**{st['name']}** attempt {attempt} FAILED (exit {rc}). Log tail: `{tail[-300:].replace(chr(10), ' | ')}`")
    state[st["name"]] = "FAILED after 2 attempts"
    return False


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"\n# RE-FUSED-6 pipeline run started {now()}\n\n")
    log("orchestrator started; keep-awake held; queue = M5b_full -> M8 -> M7 -> ALL_STATS")
    t0 = time.time()
    for st in STAGES:
        if done(st):
            state[st["name"]] = "complete (already)"
            log(f"**{st['name']}** already complete, skipping")
            continue
        marker = st.get("external_marker")
        if marker and external_running(marker):
            log(f"**{st['name']}** is already running from an earlier launcher; waiting for it")
            state[st["name"]] = "running (external)"
            last_beat = 0
            while external_running(marker) and not done(st):
                if time.time() - last_beat > 600:
                    n = 0
                    try:
                        n = sum(1 for l in open(os.path.join(RES, st["log"]), encoding="utf-8", errors="ignore")
                                if l.startswith("["))
                    except Exception:
                        pass
                    log(f"heartbeat: {st['name']} still running, {n} configs logged")
                    last_beat = time.time()
                time.sleep(60)
            time.sleep(90)                      # allow the launcher to write its flag
            if os.path.exists(os.path.join(RES, st["output"])):
                if not os.path.exists(os.path.join(RES, st["flag"])):
                    open(os.path.join(RES, st["flag"]), "w").write(now())
                state[st["name"]] = "complete"
                log(f"**{st['name']}** complete (external launcher)")
                continue
            log(f"**{st['name']}** external process ended WITHOUT output; re-running it here")
        run_stage(st)

    hours = (time.time() - t0) / 3600
    ok = [k for k, v in state.items() if v.startswith("complete")]
    bad = [k for k, v in state.items() if not v.startswith("complete")]
    log(f"pipeline finished in {hours:.2f} h. complete={ok} failed={bad}")
    with open(HANDOFF, "w", encoding="utf-8") as f:
        f.write(f"# RE-FUSED-6 handoff for the next session\n\nWritten by the orchestrator at {now()} "
                f"after {hours:.2f} h.\n\n")
        f.write("## Compute stages\n\n| stage | status |\n|---|---|\n")
        for k, v in state.items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Where everything is\n\n"
                "* consolidated statistics: `results/refused6/ALL_STATS.md` (and `.json`)\n"
                "* live run log: `results/refused6/PIPELINE_STATUS.md`\n"
                "* defect and claim lineage: `docs/08_defect_lineage.md`\n"
                "* diagnosis and plan: `docs/07_refused6_diagnosis_and_plan.md`\n"
                "* frozen canonical M1 outputs: `results/refused6/frozen_v1/` (MD5 manifest)\n")
        f.write("\n## What still needs a model session (cannot run unattended)\n\n"
                "1. Audit M5b, M8 and M7 outputs; open D12+ for any new defect and re-run affected stages.\n"
                "2. Reviewer-quality gate on M3/M5/M5b.\n"
                "3. Theory stage: formal propositions, proofs, excess-risk and drift results.\n"
                "4. Novelty verification against the literature.\n"
                "5. Four-perspective reviewer audit and iteration.\n"
                "6. Final report.\n")
        if bad:
            f.write("\n## Failed stages - rerun command\n\n")
            for st in STAGES:
                if st["name"] in bad:
                    f.write(f"```\ncd /d D:\\\\REFUSED5\n\"{PY}\" -u {st['script']} {' '.join(st['args'])}\n```\n")
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    log("keep-awake released; orchestrator exiting")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"orchestrator crashed: {type(e).__name__}: {e}")
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        raise
