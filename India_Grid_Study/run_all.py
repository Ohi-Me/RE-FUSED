#!/usr/bin/env python3
"""RE-FUSED: one entry point that runs the whole RE-FUSED pipeline, from the official-data panel to the O2-O5 results.

Folders (all relative to this file):
  data/     raw official files, interim source tables, processed State-day panel, docs, Chronos weights (hf_cache)
  codes/    refused package, pipeline scripts, tests, design documents (protocols, pre-registration draft)
  results/  <phase>/o2, o3, o4, o5 outputs; tables/; figures/; paper/; rebuild/ (panel rebuild check); RUN_SUMMARY.md
  logs/     run_all.log, stages/<stage>.log, done/<stage>.json, pbs/ job scripts and PBS output

Commands
  python run_all.py plan                       list the stages, where they run (cpu/gpu) and whether they are done
  python run_all.py local [--only a,b] [--force]
                                               run every remaining stage here, strictly one after another
  python run_all.py stage NAME [NAME ...]      run the named stages (this is what the PBS jobs call)
  python run_all.py submit [--dry-run] [--force] [--jobs job1,job2] [--after JOBID]
                                               on the H100 login node: write PBS jobs and submit them with dependencies
  python run_all.py status                     stage state, and the user's PBS jobs when qstat exists
  python run_all.py pick-mig                   print a free MIG instance UUID (used inside the GPU job)
Laptop <-> cluster (SSH key authentication only: this script never reads, stores or sends a password)
  python run_all.py push [--no-raw] [--with-results]
                                               copy codes/, data/, hpc/ and this file to HPC_DIR on the cluster
  python run_all.py remote-setup               create the conda environment on the login node (hpc/setup_env.sh)
  python run_all.py remote ARGS...             run "python run_all.py ARGS" on the login node, e.g. remote submit
  python run_all.py pull                       copy results/ and logs/ back from the cluster into this folder

Every model script skips work that is already complete, so any stage or job can simply be run again after a stop.
Queue use: every PBS job runs on workq with one MIG instance (CPU-only work is stopped by the cluster's CPU audit,
even on cpuq), and nothing heavy runs on the login node.
"""
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(ROOT, "codes")
LOGS = os.path.join(ROOT, "logs")
DONE = os.path.join(LOGS, "done")
T = "T1,T2,T3,T4,T5"
PY = sys.executable


def S(path, *args):
    return [os.path.join(CODES, *path.split("/"))] + list(args)


# name: (queue, command, extra environment). A command is a script path with arguments, or "internal:<name>".
REBUILD_ENV = {"REFUSED_ROOT": os.path.join(ROOT, "results", "rebuild"), "REFUSED_RAW": os.path.join(ROOT, "data", "raw")}
STAGES = {
    "env_check":    ("cpu", "internal:env_check", {}),
    "leakage_test": ("cpu", S("tests/test_leakage.py"), {}),
    "o2_samples":   ("cpu", S("scripts/o2/o2_01_samples.py"), {}),
    "o2_lgbm":      ("cpu", S("scripts/o2/o2_02_lgbm.py", "T1", "T2", "T3", "T4", "T5"), {}),
    "o2_patchtst":  ("gpu", S("scripts/o2/o2_03_nf.py", "patchtst", T, "5"), {}),
    "o2_tft":       ("gpu", S("scripts/o2/o2_03_nf.py", "tft", T, "5"), {}),
    "o2_bilstm":    ("gpu", S("scripts/o2/o2_04_bilstm.py", T, "5"), {}),
    "o2_chronos":   ("gpu", S("scripts/o2/o2_05_chronos.py", T), {}),
    "o2_evaluate":  ("cpu", S("scripts/o2/o2_06_evaluate.py"), {}),
    "o4":           ("cpu", S("scripts/o4/o4_01_assessment.py"), {}),
    "o5_rules":     ("cpu", S("scripts/o5/o5_01_scheduling.py"), {}),
    "o5_ppo":       ("gpu", S("scripts/o5/o5_02_ppo.py"), {"REFUSED_PPO_DEVICE": "cuda"}),
    "o3_main":      ("gpu", S("scripts/o3/o3_01_lamcag.py", "T5,T4,T3,T2,T1",
                              "mcag_instance,fusion_fixed,mcag_fixed,mcag_regime", "5"), {}),
    "o3_ablation":  ("gpu", S("scripts/o3/o3_01_lamcag.py", "T1,T2,T5", "null_context,permuted_context,market_only", "3"), {}),
    "o3_ablation2": ("gpu", S("scripts/o3/o3_01_lamcag.py", "T1,T2,T3", "no_carbon", "3"), {}),
    "o3_evaluate":  ("cpu", S("scripts/o3/o3_02_evaluate.py"), {}),
    "prereg_fill":  ("cpu", S("scripts/prereg/fill_prereg.py"), {}),
    "data_paper_numbers": ("cpu", S("scripts/paper/make_numbers_data.py"), {}),
    "data_paper_figures": ("cpu", S("scripts/paper/make_figures_data.py"), {}),
    "build_b01_psp":  ("cpu", S("scripts/build/b01_psp.py"), REBUILD_ENV),
    "build_b02_dsm":  ("cpu", S("scripts/build/b02_dsm.py"), REBUILD_ENV),
    "build_b03_npp":  ("cpu", S("scripts/build/b03_npp.py"), REBUILD_ENV),
    "build_b04_re":   ("cpu", S("scripts/build/b04_cea_re.py"), REBUILD_ENV),
    "build_b05_co2":  ("cpu", S("scripts/build/b05_co2.py"), REBUILD_ENV),
    "build_b06_imd":  ("cpu", S("scripts/build/b06_imd.py"), REBUILD_ENV),
    "build_b07_panel": ("cpu", S("scripts/build/b07_panel.py"), REBUILD_ENV),
    "build_b08_validation": ("cpu", S("scripts/build/b08_validation.py"), REBUILD_ENV),
    "build_verify": ("cpu", "internal:build_verify", {}),
    "report":       ("cpu", "internal:report", {}),
    # development round 2 (codes/design/06_dev2_protocol.md)
    "d2_setup":     ("cpu", S("scripts/dev2/d2_00_setup.py"), {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}),
    "d2_smoke":     ("gpu", S("scripts/dev2/d2_smoke.py"), {}),
    "d2_xgb":       ("gpu", S("scripts/dev2/d2_01_xgb.py"), {}),
    "d2_nf":        ("gpu", S("scripts/dev2/d2_02_nf_models.py", "nhits,tide,bitcn,nbeatsx", T), {}),
    "d2_chronos":   ("gpu", S("scripts/dev2/d2_03_chronos.py", "chronos2,chronos_base", T), {}),
    "d2_o3_sd":     ("gpu", S("scripts/o3/o3_01_lamcag.py", "T5,T4,T3,T2,T1", "mcag_instance_sd,fusion_fixed_sd", "5"),
                     {"REFUSED_O3_OUT": os.path.join(ROOT, "results", "dev2", "o3")}),
    "d2_smoke_eval": ("gpu", S("scripts/dev2/d2_smoke_eval.py"), {}),
    "d2_ensemble":  ("gpu", S("scripts/dev2/d2_05_ensemble.py"), {}),
    "d2_o2_eval":   ("cpu", S("scripts/dev2/d2_06_o2_eval.py"), {}),
    "d2_o3_eval":   ("cpu", S("scripts/dev2/d2_07_o3_eval.py"), {}),
    "d2_o4":        ("gpu", S("scripts/dev2/d2_08_o4.py"), {}),
    "d2_o5":        ("gpu", S("scripts/dev2/d2_09_o5.py"), {}),
    "d2_evidence":  ("cpu", S("scripts/dev2/d2_10_evidence.py"), {}),
    "d2_ppo":       ("gpu", S("scripts/dev2/d2_11_ppo.py"), {"REFUSED_PPO_DEVICE": "cuda"}),
    # rehearsal before the freeze (development phase, frozen-mode code on small slices; no reserved data)
    "c_rehearsal":  ("gpu", S("scripts/confirm/c00_rehearsal.py"), {}),
}
# confirmatory phase (pre-registration v2): same scripts with REFUSED_PHASE=confirm; results go to results/confirm and
# results/confirm2. c_freeze hashes and tags the pre-registration first; every other stage needs it.
CONF = {"REFUSED_PHASE": "confirm", "REFUSED_CONFIRMATORY": "1"}
CONF_O3 = dict(CONF, REFUSED_O3_OUT=os.path.join(ROOT, "results", "confirm2", "o3"))
STAGES.update({
    "c_freeze":       ("cpu", "internal:freeze", {}),
    "c_env_check":    ("cpu", "internal:env_check", CONF),
    "c_leakage":      ("cpu", S("tests/test_leakage.py"), CONF),
    "c_o2_samples":   ("cpu", S("scripts/o2/o2_01_samples.py"), CONF),
    "c_o2_lgbm":      ("cpu", S("scripts/o2/o2_02_lgbm.py", "T1", "T2", "T3", "T4", "T5"), CONF),
    "c_o2_patchtst":  ("gpu", S("scripts/o2/o2_03_nf.py", "patchtst", T, "5"), CONF),
    "c_o2_tft":       ("gpu", S("scripts/o2/o2_03_nf.py", "tft", T, "5"), CONF),
    "c_o2_bilstm":    ("gpu", S("scripts/o2/o2_04_bilstm.py", T, "5"), CONF),
    "c_o2_chronos":   ("gpu", S("scripts/o2/o2_05_chronos.py", T), CONF),
    "c_d2_xgb":       ("gpu", S("scripts/dev2/d2_01_xgb.py"), CONF),
    "c_d2_nf":        ("gpu", S("scripts/dev2/d2_02_nf_models.py", "nhits,tide,bitcn,nbeatsx", T), CONF),
    "c_d2_chronos":   ("gpu", S("scripts/dev2/d2_03_chronos.py", "chronos2,chronos_base", T), CONF),
    "c_o3_main":      ("gpu", S("scripts/o3/o3_01_lamcag.py", "T5,T4,T3,T2,T1",
                                "mcag_instance,fusion_fixed,mcag_fixed,mcag_regime", "5"), CONF),
    "c_o3_ablation":  ("gpu", S("scripts/o3/o3_01_lamcag.py", "T1,T2,T5", "null_context,permuted_context,market_only", "3"), CONF),
    "c_o3_ablation2": ("gpu", S("scripts/o3/o3_01_lamcag.py", "T1,T2,T3", "no_carbon", "3"), CONF),
    "c_d2_o3_sd":     ("gpu", S("scripts/o3/o3_01_lamcag.py", "T5,T4,T3,T2,T1", "mcag_instance_sd,fusion_fixed_sd", "5"), CONF_O3),
    "c_o2_evaluate":  ("cpu", S("scripts/o2/o2_06_evaluate.py"), CONF),
    "c_o3_evaluate":  ("cpu", S("scripts/o3/o3_02_evaluate.py"), CONF),
    "c_d2_ensemble":  ("gpu", S("scripts/dev2/d2_05_ensemble.py"), CONF),
    "c_d2_o2_eval":   ("cpu", S("scripts/dev2/d2_06_o2_eval.py"), CONF),
    "c_d2_o3_eval":   ("cpu", S("scripts/dev2/d2_07_o3_eval.py"), CONF),
    "c_o4":           ("cpu", S("scripts/o4/o4_01_assessment.py"), CONF),
    "c_o5_rules":     ("cpu", S("scripts/o5/o5_01_scheduling.py"), CONF),
    "c_o5_ppo":       ("gpu", S("scripts/o5/o5_02_ppo.py"), dict(CONF, REFUSED_PPO_DEVICE="cuda")),
    "c_d2_o4":        ("gpu", S("scripts/dev2/d2_08_o4.py"), CONF),
    "c_d2_o5":        ("gpu", S("scripts/dev2/d2_09_o5.py"), CONF),
    "c_d2_ppo":       ("gpu", S("scripts/dev2/d2_11_ppo.py"), dict(CONF, REFUSED_PPO_DEVICE="cuda")),
    "c_evidence":     ("cpu", S("scripts/dev2/d2_10_evidence.py"), CONF),
    "c_hypotheses":   ("cpu", S("scripts/confirm/c01_hypotheses.py"), CONF),
    "c_paper":        ("cpu", S("scripts/paper/make_paper_outputs.py"), CONF),
    "paper_dev_check": ("cpu", S("scripts/paper/make_paper_outputs.py"), {}),
    # after the confirmatory run: the full tables for the papers and the supplement, descriptive breakdowns, and the
    # integrity check (frozen hashes, panel checksum, re-scoring the hypotheses from the stored results)
    "c_supp":         ("cpu", S("scripts/paper/make_supplement.py"), CONF),
    "c_strata":       ("cpu", S("scripts/audit/a01_strata.py"), CONF),
    "c_integrity":    ("cpu", S("scripts/audit/a02_integrity.py"), CONF),
    # additional analyses after the confirmatory run, not pre-registered and reported as such; they read the stored
    # results only and write to results/extra/
    "x_value_of_forecast":   ("gpu", S("scripts/extra/x01_value_of_forecast.py"), CONF),
    "x_settlement_quantile": ("gpu", S("scripts/extra/x02_settlement_quantile.py"), CONF),
    "x_settlement_extensions": ("gpu", S("scripts/extra/x04_settlement_extensions.py"), CONF),
    "x_paper_outputs":       ("cpu", S("scripts/extra/x03_paper_outputs.py"), CONF),
})
AUTOPILOT = ["c_supp", "c_strata", "c_integrity"]  # post-run reporting and checks, not part of the confirmatory run
CONFIRM = [s for s in STAGES if s.startswith("c_") and s not in ["c_rehearsal", "c_paper"] + AUTOPILOT]
BUILD = [s for s in STAGES if s.startswith("build_")]
DEV2 = [s for s in STAGES if s.startswith("d2_")]
# PBS jobs: (name, queue, stages, dependencies as (job, "afterok"|"afterany")). Every job runs on workq with one MIG
# instance (the cluster's CPU audit also stops CPU-only work on cpuq, and HPC asked that all computation goes through PBS).
# The two jobs are chained, so only one GPU instance is used at a time (manual: one GPU per user). Stages marked "cpu"
# run inside the GPU job's allocation with the GPU hidden from them; the driver keeps the job's CUDA context open.
JOBS = [
    ("refused_o2_gpu", "gpu", ["env_check", "leakage_test", "o2_samples", "o2_patchtst", "o2_tft", "o2_bilstm", "o2_chronos"], []),
    ("refused_rest_gpu", "gpu", ["o3_main", "o3_ablation", "o3_ablation2", "o2_lgbm", "o2_evaluate", "o4", "o5_rules", "o5_ppo",
                               "o3_evaluate", "data_paper_numbers", "data_paper_figures"] + BUILD + ["prereg_fill", "report"],
     [("refused_o2_gpu", "afterany")]),
    ("refused_dev2_setup", "gpu", ["d2_setup", "d2_smoke"], []),
    ("refused_dev2_train", "gpu", ["d2_xgb", "d2_nf", "d2_chronos", "d2_o3_sd"], [("refused_dev2_setup", "afterok")]),
    ("refused_dev2_eval", "gpu", ["d2_smoke_eval", "d2_ensemble", "d2_o2_eval", "d2_o3_eval", "d2_o4", "d2_o5", "d2_evidence"],
     [("refused_dev2_train", "afterany")]),
    ("refused_dev2_ppo", "gpu", ["d2_ppo"], [("refused_dev2_eval", "afterany")]),
    ("refused_rehearsal", "gpu", ["c_rehearsal"], []),
    ("refused_confirm", "gpu", CONFIRM, []),
    ("refused_confirm_post", "gpu", ["paper_dev_check", "c_paper"], [("refused_confirm", "afterany")]),
    ("refused_autopilot", "gpu", AUTOPILOT, [("refused_confirm_post", "afterany")]),
    ("refused_extra", "gpu", ["x_settlement_quantile", "x_value_of_forecast", "x_settlement_extensions",
                            "x_paper_outputs"],
     [("refused_autopilot", "afterany")]),
]
# a stage runs only if these stages are done (with --keep-going, independent stages continue after a failure)
O2_MODELS = ["o2_samples", "o2_lgbm", "o2_patchtst", "o2_tft", "o2_bilstm", "o2_chronos"]
REQUIRES = {
    "o2_lgbm": ["o2_samples"], "o2_patchtst": ["o2_samples"], "o2_tft": ["o2_samples"], "o2_bilstm": ["o2_samples"],
    "o2_chronos": ["o2_samples"], "o3_main": ["o2_samples"], "o3_ablation": ["o2_samples"], "o3_ablation2": ["o2_samples"],
    "o2_evaluate": O2_MODELS, "o4": ["o2_evaluate"], "o5_rules": ["o4"], "o5_ppo": ["o5_rules"],
    "o3_evaluate": ["o2_evaluate", "o3_main", "o3_ablation", "o3_ablation2"], "prereg_fill": ["o2_evaluate", "o4", "o5_rules"],
    "build_b07_panel": ["build_b01_psp", "build_b02_dsm", "build_b03_npp", "build_b04_re", "build_b05_co2", "build_b06_imd"],
    "build_b08_validation": ["build_b07_panel"], "build_verify": ["build_b07_panel"],
    "d2_smoke": ["d2_setup"], "d2_xgb": ["d2_smoke"], "d2_nf": ["d2_smoke"], "d2_chronos": ["d2_smoke"], "d2_o3_sd": ["d2_smoke"],
    "d2_o2_eval": ["d2_ensemble"],
    "d2_o4": ["d2_o2_eval"], "d2_o5": ["d2_o2_eval"], "d2_ppo": ["d2_o5"],
}
_c_models = ["c_o2_samples", "c_o2_lgbm", "c_o2_patchtst", "c_o2_tft", "c_o2_bilstm", "c_o2_chronos"]
REQUIRES.update({s: ["c_freeze"] for s in CONFIRM if s != "c_freeze"})
for s in CONFIRM:
    if s not in ("c_freeze", "c_env_check", "c_leakage", "c_o2_samples", "c_evidence", "c_hypotheses"):
        REQUIRES[s] = REQUIRES[s] + ["c_o2_samples"]
REQUIRES["c_o2_evaluate"] += _c_models
REQUIRES["c_o3_evaluate"] += ["c_o2_evaluate", "c_o3_main", "c_o3_ablation", "c_o3_ablation2"]
REQUIRES["c_d2_ensemble"] += _c_models + ["c_d2_xgb", "c_d2_nf", "c_d2_chronos", "c_o3_main", "c_d2_o3_sd"]
REQUIRES["c_d2_o2_eval"] += ["c_d2_ensemble", "c_o2_evaluate"]
REQUIRES["c_d2_o3_eval"] += ["c_o3_main", "c_d2_o3_sd"]
REQUIRES["c_o4"] += ["c_o2_evaluate"]
REQUIRES["c_o5_rules"] += ["c_o4"]
REQUIRES["c_o5_ppo"] += ["c_o5_rules"]
REQUIRES["c_d2_o4"] += ["c_d2_o2_eval", "c_o2_lgbm"]
REQUIRES["c_d2_o5"] += ["c_d2_o2_eval", "c_o2_lgbm"]
REQUIRES["c_d2_ppo"] += ["c_d2_o5"]
ORDER = [s for _, _, stages, _ in JOBS for s in stages]
assert sorted(ORDER) == sorted(STAGES), "every stage must belong to exactly one job"
LOCAL_ORDER = ORDER


# ----------------------------------------------------------------------------------------------------------- settings
def settings():
    """Non-secret settings from .env (laptop) and hpc/cluster.env (written by push). Password keys are ignored."""
    cfg = {"HPC_HOST": "<hpc-host>", "HPC_USER": "", "HPC_DIR": "", "HPC_CONDA_ENV": "refused_h100",
           "HPC_CONDA_SH": "/apps/compilers/anaconda3/etc/profile.d/conda.sh", "HPC_GPU_NCPUS": "8",
           "HPC_CPU_NCPUS": "8", "HPC_WALLTIME": "24:00:00", "HPC_GPU_WALLTIME": "48:00:00", "HPC_GPU_MEM": "32gb",
           "REFUSED_MIG_UUID": ""}
    for f in (os.path.join(ROOT, ".env"), os.path.join(ROOT, "hpc", "cluster.env")):
        if not os.path.exists(f):
            continue
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if "PASS" in k.upper() or "SECRET" in k.upper() or "TOKEN" in k.upper():
                continue
            if v:
                cfg[k] = v
    for k in cfg:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    if cfg["HPC_USER"] and not cfg["HPC_DIR"]:
        cfg["HPC_DIR"] = f"/home/{cfg['HPC_USER']}/RE-FUSED"
    return cfg


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def runlog(msg):
    os.makedirs(LOGS, exist_ok=True)
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "run_all.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stage_state(name):
    f = os.path.join(DONE, f"{name}.json")
    if not os.path.exists(f):
        return "todo"
    return json.load(open(f, encoding="utf-8")).get("state", "todo")


def child_env(extra):
    env = dict(os.environ)
    threads = os.environ.get("NCPUS") or os.environ.get("REFUSED_THREADS") or str(min(16, os.cpu_count() or 1))
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",  # PBS jobs export PYTHONNOUSERSITE=1
                "REFUSED_ROOT": ROOT, "REFUSED_THREADS": threads, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads, "HF_HOME": os.path.join(ROOT, "data", "hf_cache"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    env.update(extra)
    return env


# ------------------------------------------------------------------------------------------------------ run a stage
def run_stage(name, force=False):
    queue, cmd, extra = STAGES[name]
    if stage_state(name) == "done" and not force:
        runlog(f"skip {name} (done)")
        return True
    if name.startswith("build_") or name.startswith("data_paper"):
        if not os.path.isdir(os.path.join(ROOT, "data", "raw")):
            runlog(f"skip {name}: data/raw is not present (push without --no-raw to include it)")
            return True
    os.makedirs(DONE, exist_ok=True)
    os.makedirs(os.path.join(LOGS, "stages"), exist_ok=True)
    if isinstance(cmd, str):
        argv = [PY, "-u", os.path.abspath(__file__), "_internal", cmd.split(":", 1)[1]]
    else:
        argv = [PY, "-u"] + cmd
    runlog(f"start {name} [{queue}] {' '.join(os.path.relpath(a, ROOT) if os.path.isabs(a) else a for a in argv[2:])}")
    t0 = time.time()
    logf = os.path.join(LOGS, "stages", f"{name}.log")
    with open(logf, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {name} start {now()} host={socket.gethostname()} job={os.environ.get('PBS_JOBID', 'local')}\n")
        fh.flush()
        # CPU stages never see a GPU (e.g. Stable-Baselines3 would otherwise pick CUDA): one GPU per user, cpuq jobs
        env = child_env({**extra, "CUDA_VISIBLE_DEVICES": ""} if queue == "cpu" else extra)
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = p.wait()
        fh.write(f"===== {name} end {now()} rc={rc} seconds={time.time() - t0:.0f}\n")
    rec = {"stage": name, "state": "done" if rc == 0 else "failed", "returncode": rc, "queue": queue,
           "seconds": round(time.time() - t0, 1), "finished": now(), "host": socket.gethostname(),
           "pbs_job": os.environ.get("PBS_JOBID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "log": os.path.relpath(logf, ROOT)}
    json.dump(rec, open(os.path.join(DONE, f"{name}.json"), "w", encoding="utf-8"), indent=1)
    runlog(f"{'end' if rc == 0 else 'FAILED'} {name} rc={rc} {rec['seconds']:.0f}s")
    return rc == 0


def hold_gpu_handle():
    """Inside a PBS GPU job, keep one CUDA context open in the driver for the whole job, so that CPU-side data
    preparation between training steps is attributed to this job's single allocated GPU instance (manual v2.0, process
    audit). It uses the same one instance as the training processes; no additional GPU is touched."""
    if not os.environ.get("PBS_JOBID") or os.environ.get("REFUSED_HOLD_GPU", "1") == "0":
        return None
    import torch
    if not torch.cuda.is_available():
        runlog("ERROR: GPU stages requested but CUDA is not available (check CUDA_VISIBLE_DEVICES / MIG UUID)")
        sys.exit(3)
    if torch.cuda.device_count() != 1:
        runlog(f"ERROR: {torch.cuda.device_count()} GPU devices visible; exactly one MIG instance is allowed")
        sys.exit(3)
    keep = torch.zeros(1, device="cuda")
    runlog(f"GPU: {torch.cuda.get_device_name(0)} ({os.environ.get('CUDA_VISIBLE_DEVICES')}), torch {torch.__version__}")
    return keep


def cmd_stage(names, force=False, keep_going=False):
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        sys.exit(f"unknown stages: {unknown}; see 'python run_all.py plan'")
    keep = hold_gpu_handle() if os.environ.get("PBS_JOBID") or any(STAGES[n][0] == "gpu" for n in names) else None
    failed = []
    for n in names:
        missing = [r for r in REQUIRES.get(n, []) if stage_state(r) != "done"]
        if missing and not (n.startswith(("build_", "data_paper")) and not os.path.isdir(os.path.join(ROOT, "data", "raw"))):
            runlog(f"skip {n}: needs {', '.join(missing)}")
            failed.append(n)
            continue
        if not run_stage(n, force):
            failed.append(n)
            if not keep_going:
                runlog(f"stopping: stage {n} failed (see logs/stages/{n}.log); rerun the same command to resume")
                sys.exit(1)
    del keep
    if failed:
        runlog(f"FINISHED_WITH_PROBLEMS {','.join(failed)}")
        sys.exit(1)
    runlog(f"ALL_REQUESTED_STAGES_DONE {','.join(names)}")


def cmd_local(only=None, force=False):
    names = [s for s in LOCAL_ORDER if not only or s in only]
    cmd_stage(names, force, keep_going=True)
    runlog("ALL_LOCAL_DONE")


def cmd_plan():
    print(f"{'job':<22}{'queue':<6}{'stage':<22}state")
    for job, queue, stages, deps in JOBS:
        for s in stages:
            print(f"{job:<22}{STAGES[s][0]:<6}{s:<22}{stage_state(s)}")
    n = sum(stage_state(s) == "done" for s in STAGES)
    print(f"\n{n} of {len(STAGES)} stages done")


# --------------------------------------------------------------------------------------------------------- PBS side
def pbs_script(job, queue, stages, cfg):
    gpu = queue == "gpu"
    ncpus = cfg["HPC_GPU_NCPUS"] if gpu else cfg["HPC_CPU_NCPUS"]
    # workq: default memory is 1 GB and the maximum 32 GB, walltime up to 48 h; cpuq: walltime up to 24 h
    select = f"select=1:ncpus={ncpus}:ngpus=1:mem={cfg['HPC_GPU_MEM']}" if gpu else f"select=1:ncpus={ncpus}"
    walltime = cfg["HPC_GPU_WALLTIME"] if gpu else cfg["HPC_WALLTIME"]
    out = os.path.join(LOGS, "pbs", f"{job}.out")
    lines = ["#!/bin/bash", f"#PBS -N {job}", f"#PBS -q {'workq' if gpu else 'cpuq'}", f"#PBS -l {select}",
             f"#PBS -l walltime={walltime}", "#PBS -j oe", f"#PBS -o {out}", "",
             f"cd {shlex.quote(ROOT)}", "export PYTHONNOUSERSITE=1", f"export REFUSED_THREADS={ncpus}",
             f"source {cfg['HPC_CONDA_SH']}", f"conda activate {cfg['HPC_CONDA_ENV']}",
             'echo "job $PBS_JOBID on $(hostname) at $(date -Iseconds); python $(which python)"']
    if gpu:
        lines += ["# MIG setup (required for H100): a fixed UUID from hpc/cluster.env, else the first free instance",
                  f'MIG="{cfg["REFUSED_MIG_UUID"]}"',
                  'for try in $(seq 1 60); do [ -n "$MIG" ] && break; MIG=$(python run_all.py pick-mig 2>/dev/null) && break; '
                  'MIG=""; echo "no free MIG instance yet (try $try), waiting 60 s"; sleep 60; done',
                  '[ -z "$MIG" ] && { echo "no free MIG instance after 60 min"; exit 3; }',
                  'export CUDA_VISIBLE_DEVICES="$MIG"', 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"', "nvidia-smi -L"]
    lines += [f"python -u run_all.py stage --keep-going {' '.join(stages)}", ""]
    return "\n".join(lines)


def cmd_submit(dry_run=False, force=False, only_jobs=None, after=None):
    cfg = settings()
    os.makedirs(os.path.join(LOGS, "pbs"), exist_ok=True)
    if not dry_run and not shutil.which("qsub"):
        sys.exit("qsub not found: run 'submit' on the H100 login node (from the laptop: python run_all.py remote submit)")
    ids = {}
    for job, queue, stages, deps in JOBS:
        if only_jobs and job not in only_jobs:
            continue
        todo = [s for s in stages if force or stage_state(s) != "done"]
        if job == "refused_rebuild_paper" and not os.path.isdir(os.path.join(ROOT, "data", "raw")):
            todo = []
        if not todo:
            print(f"{job}: all stages done, not submitted")
            continue
        path = os.path.join(LOGS, "pbs", f"{job}.pbs")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(pbs_script(job, queue, todo, cfg))
        dep = [f"{kind}:{ids[j]}" for j, kind in deps if j in ids]
        if after and not dep and deps:
            dep = [f"afterany:{after}"]
        argv = ["qsub"] + (["-W", "depend=" + ",".join(dep)] if dep else []) + [path]
        if dry_run:
            ids[job] = f"<{job}>"
            print(" ".join(argv))
            continue
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"qsub failed for {job}: {r.stderr.strip()}")
        ids[job] = r.stdout.strip()
        runlog(f"submitted {job} -> {ids[job]} ({queue}; {', '.join(todo)})" + (f" depends {','.join(dep)}" if dep else ""))
    if ids and not dry_run:
        json.dump(ids, open(os.path.join(LOGS, "pbs", "last_submission.json"), "w"), indent=1)


def cmd_status():
    cmd_plan()
    if shutil.which("qstat"):
        user = os.environ.get("USER") or settings()["HPC_USER"]
        print(subprocess.run(["qstat", "-u", user], capture_output=True, text=True).stdout)
    tail = os.path.join(LOGS, "run_all.log")
    if os.path.exists(tail):
        print("last log lines:")
        print("".join(open(tail, encoding="utf-8").readlines()[-8:]))


def cmd_pick_mig():
    """Print the UUID of a MIG instance with no processes and (almost) no memory in use."""
    fixed = settings()["REFUSED_MIG_UUID"]
    if fixed:
        print(fixed)
        return
    L = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    full = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    devs, gpu = [], None
    for line in L.splitlines():
        m = re.match(r"\s*GPU (\d+):", line)
        if m:
            gpu = int(m.group(1))
            continue
        m = re.search(r"MIG\s+(\S+)\s+Device\s+(\d+):\s*\(UUID:\s*(MIG-[^)\s]+)\)", line)
        if m and gpu is not None:
            devs.append({"gpu": gpu, "profile": m.group(1), "dev": int(m.group(2)), "uuid": m.group(3)})
    mem, gici, busy = {}, {}, set()
    section = None
    for line in full.splitlines():
        if "MIG devices" in line:
            section = "mig"
        elif "Processes:" in line:
            section = "proc"
        if section == "mig":
            m = re.match(r"\|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\|\s+(\d+)MiB\s*/\s*(\d+)MiB", line)
            if m:
                g, gi, ci, d, used, total = map(int, m.groups())
                mem[(g, d)] = (used, total)
                gici[(g, gi, ci)] = d
        elif section == "proc":
            m = re.match(r"\|\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+(C|G|C\+G)\s", line)
            if m:
                g, gi, ci = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if (g, gi, ci) in gici:
                    busy.add((g, gici[(g, gi, ci)]))
    free = [d for d in devs if (d["gpu"], d["dev"]) not in busy and (d["gpu"], d["dev"]) in mem
            and mem[(d["gpu"], d["dev"])][0] < 200]
    if not free:
        sys.stderr.write("no free MIG instance found; set REFUSED_MIG_UUID in hpc/cluster.env\n" + L)
        sys.exit(3)
    free.sort(key=lambda d: -mem[(d["gpu"], d["dev"])][1])
    print(free[0]["uuid"])


# ---------------------------------------------------------------------------------------------------- laptop side
def ssh_base(cfg):
    if not cfg["HPC_USER"]:
        sys.exit("HPC_USER is not set in .env")
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=60",
            f"{cfg['HPC_USER']}@{cfg['HPC_HOST']}"]


def tar_exe():
    win = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe")
    return win if os.name == "nt" and os.path.exists(win) else "tar"


def check_ssh(cfg):
    r = subprocess.run(ssh_base(cfg) + ["echo ok"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("SSH key login is not set up (the password is never used by this script). Run once, typing your "
                 "password yourself when asked:\n  type %USERPROFILE%\\.ssh\\id_ed25519.pub | ssh "
                 f"{cfg['HPC_USER']}@{cfg['HPC_HOST']} \"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                 "cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys\"\n" + r.stderr)


def remote_shell(cfg, script):
    return subprocess.run(ssh_base(cfg) + ["bash -lc " + shlex.quote(script)]).returncode


def cmd_push(no_raw=False, with_results=False):
    cfg = settings()
    check_ssh(cfg)
    with open(os.path.join(ROOT, "hpc", "cluster.env"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# written by 'run_all.py push' (no secrets)\n")
        for k in ("HPC_USER", "HPC_DIR", "HPC_CONDA_ENV", "HPC_CONDA_SH", "HPC_GPU_NCPUS", "HPC_CPU_NCPUS",
                  "HPC_WALLTIME", "HPC_GPU_WALLTIME", "HPC_GPU_MEM", "REFUSED_MIG_UUID"):
            fh.write(f"{k}={cfg[k]}\n")
    members = ["run_all.py", "README.md", "requirements-h100.txt", ".env.example", "hpc", "codes", "data"]
    if with_results:
        members += ["results", "logs"]
    excl = ["--exclude=__pycache__", "--exclude=.env"] + (["--exclude=data/raw"] if no_raw else [])
    dest = shlex.quote(cfg["HPC_DIR"])
    runlog(f"push {', '.join(members)}{' (without data/raw)' if no_raw else ''} -> {cfg['HPC_HOST']}:{cfg['HPC_DIR']}")
    t0 = time.time()
    tar = subprocess.Popen([tar_exe(), "-cf", "-"] + excl + members, cwd=ROOT, stdout=subprocess.PIPE)
    ssh = subprocess.Popen(ssh_base(cfg) + [f"mkdir -p {dest} && tar -xf - -C {dest} && du -sh {dest}"], stdin=tar.stdout)
    tar.stdout.close()
    rc = ssh.wait()
    tar.wait()
    runlog(f"push {'done' if rc == 0 and tar.returncode == 0 else 'FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(rc or tar.returncode)


def cmd_pull():
    cfg = settings()
    check_ssh(cfg)
    runlog(f"pull results/ logs/ codes/design <- {cfg['HPC_HOST']}:{cfg['HPC_DIR']}")
    ssh = subprocess.Popen(ssh_base(cfg) + [f"tar -cf - -C {shlex.quote(cfg['HPC_DIR'])} results logs codes/design"],
                           stdout=subprocess.PIPE)
    tar = subprocess.Popen([tar_exe(), "-xf", "-"], cwd=ROOT, stdin=ssh.stdout)
    ssh.stdout.close()
    rc = tar.wait() or ssh.wait()
    runlog(f"pull {'done' if rc == 0 else 'FAILED'}")
    sys.exit(rc)


def cmd_remote_setup():
    cfg = settings()
    check_ssh(cfg)
    sys.exit(remote_shell(cfg, f"cd {shlex.quote(cfg['HPC_DIR'])} && mkdir -p logs && bash hpc/setup_env.sh {cfg['HPC_CONDA_ENV']} "
                               f"2>&1 | tee -a logs/setup_env.log"))


def cmd_remote(args):
    cfg = settings()
    check_ssh(cfg)
    script = (f"cd {shlex.quote(cfg['HPC_DIR'])} && export PYTHONNOUSERSITE=1 && source {cfg['HPC_CONDA_SH']} && "
              f"conda activate {cfg['HPC_CONDA_ENV']} && python run_all.py {' '.join(shlex.quote(a) for a in args)}")
    sys.exit(remote_shell(cfg, script))


# ------------------------------------------------------------------------------------------------- internal stages
def internal_env_check():
    import hashlib
    from importlib import metadata
    import pandas as pd
    info = {"time": now(), "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(), "pbs_job": os.environ.get("PBS_JOBID"), "ncpus": os.environ.get("NCPUS"),
            "packages": {}}
    for p in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "statsmodels", "lightgbm", "torch",
              "pytorch-lightning", "neuralforecast", "chronos-forecasting", "transformers", "stable_baselines3",
              "gymnasium", "matplotlib", "openpyxl", "xlrd", "pymupdf"):
        try:
            info["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info["packages"][p] = None
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_build"] = torch.version.cuda
    except ImportError:
        info["cuda_available"] = None
    missing = [p for p, v in info["packages"].items() if v is None]
    proc = os.path.join(ROOT, "data", "processed")
    ref = open(os.path.join(proc, "CHECKSUM.txt")).read().split("content_sha256=")[1].strip()
    P = pd.read_parquet(os.path.join(proc, "refused_state_day.parquet"))
    h = hashlib.sha256(pd.util.hash_pandas_object(P, index=False).values.tobytes()).hexdigest()
    info.update(panel_shape=list(P.shape), panel_sha256=h, panel_sha256_reference=ref, panel_checksum_ok=h == ref)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(info, open(os.path.join(ROOT, "results", "environment.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    if missing:
        sys.exit(f"missing packages: {missing}")
    if h != ref:
        sys.exit("panel checksum does not match data/processed/CHECKSUM.txt: stop, the data were changed")
    print("ENV_CHECK_OK")


def internal_freeze():
    """Freeze pre-registration v2: hash it, then commit the code and design files and tag the commit prereg-v1.
    Safe to run again: if the tag exists, it only checks that the file still has the recorded hash."""
    import hashlib
    design = os.path.join(ROOT, "codes", "design")
    pre = os.path.join(design, "02_preregistration.md")
    sha_file = os.path.join(design, "PREREG_SHA256.txt")
    text = open(pre, encoding="utf-8").read()
    if "DRAFT" in text.split("\n")[0] or "…" in text:
        sys.exit("pre-registration still has draft markers; not freezing")
    sha = hashlib.sha256(open(pre, "rb").read()).hexdigest()
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)  # noqa: E731
    if "prereg-v1" in git("tag", "--list", "prereg-v1").stdout:
        recorded = open(sha_file).read().split()[0]
        print("already frozen; recorded", recorded, "now", sha)
        sys.exit(0 if recorded == sha else "pre-registration changed after the freeze")
    open(sha_file, "w").write(f"{sha}  02_preregistration.md\n")
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        git("init", "-q")
    git("config", "user.name", os.environ.get("USER", "refused"))
    git("config", "user.email", f"{os.environ.get('USER', 'refused')}@hpc.local")
    wanted = [".gitignore", "run_all.py", "README.md", "NOVELTY_MAP.md", "requirements-h100.txt", "codes", "hpc"]
    steps = [("add", *[w for w in wanted if os.path.exists(os.path.join(ROOT, w))]),
             ("commit", "-q", "-m", f"Pre-registration v2 frozen before the confirmatory run (sha256 {sha})"),
             ("tag", "-a", "prereg-v1", "-m", f"RE-FUSED pre-registration v2, sha256 {sha}")]
    for step in steps:  # every git call must work, otherwise the freeze has not happened
        r = git(*step)
        if r.returncode != 0:
            sys.exit(f"git {step[0]} failed: {r.stderr.strip() or r.stdout.strip()}")
    commit = git("rev-parse", "HEAD").stdout.strip()
    if "prereg-v1" not in git("tag", "--list", "prereg-v1").stdout:
        sys.exit("tag prereg-v1 was not created")
    rec = dict(time=now(), sha256=sha, commit=commit, host=socket.gethostname(), pbs_job=os.environ.get("PBS_JOBID"))
    json.dump(rec, open(os.path.join(LOGS, "freeze.json"), "w"), indent=1)
    print("FROZEN", json.dumps(rec))


def internal_build_verify():
    ref = open(os.path.join(ROOT, "data", "processed", "CHECKSUM.txt")).read().strip()
    new_f = os.path.join(ROOT, "results", "rebuild", "data", "processed", "CHECKSUM.txt")
    new = open(new_f).read().strip() if os.path.exists(new_f) else "(rebuild produced no CHECKSUM.txt)"
    ok = ref == new
    text = (f"# Panel rebuild check ({now()})\n\nReference (data/processed/CHECKSUM.txt):\n\n    {ref}\n\n"
            f"Rebuilt from data/raw on {socket.gethostname()} (results/rebuild/data/processed/CHECKSUM.txt):\n\n"
            f"    {new}\n\nResult: **{'identical panel (content SHA-256 match)' if ok else 'MISMATCH'}**\n")
    open(os.path.join(ROOT, "results", "rebuild", "VERIFY.md"), "w", encoding="utf-8").write(text)
    print(text)
    if not ok:
        sys.exit(1)


def internal_report():
    import hashlib
    L = [f"# RE-FUSED run summary", "", f"Written {now()} on {socket.gethostname()}.", "",
         "| stage | queue | state | seconds | host | PBS job | log |", "|---|---|---|---|---|---|---|"]
    for s in ORDER:
        f = os.path.join(DONE, f"{s}.json")
        r = json.load(open(f)) if os.path.exists(f) else {}
        L.append(f"| {s} | {STAGES[s][0]} | {r.get('state', 'todo')} | {r.get('seconds', '')} | {r.get('host', '')} | "
                 f"{r.get('pbs_job') or ''} | {r.get('log', '')} |")
    env_f = os.path.join(ROOT, "results", "environment.json")
    if os.path.exists(env_f):
        e = json.load(open(env_f))
        L += ["", "## Environment (env_check)", "", f"* host {e['host']}, Python {e['python']}, CUDA available "
              f"{e.get('cuda_available')}, panel checksum ok: {e['panel_checksum_ok']}",
              "* " + ", ".join(f"{k} {v}" for k, v in e["packages"].items())]
    ver = os.path.join(ROOT, "results", "rebuild", "VERIFY.md")
    if os.path.exists(ver):
        L += ["", "## Panel rebuild", "", open(ver, encoding="utf-8").read().split("Result: ")[-1].strip()]
    L += ["", "## Result files", "", "| file | bytes | sha256 (first 16) |", "|---|---:|---|"]
    for dirpath, _, files in os.walk(os.path.join(ROOT, "results")):
        if os.path.join("results", "rebuild") in dirpath:
            continue
        for fn in sorted(files):
            if fn.endswith((".json", ".csv", ".md", ".tex")) and fn != "RUN_SUMMARY.md":
                p = os.path.join(dirpath, fn)
                L.append(f"| {os.path.relpath(p, ROOT).replace(os.sep, '/')} | {os.path.getsize(p)} | "
                         f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]} |")
    open(os.path.join(ROOT, "results", "RUN_SUMMARY.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:len(ORDER) + 6]))


# ---------------------------------------------------------------------------------------------------------- main
def main(argv):
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; logs contain τ, ₹ …
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    flags = {a for a in rest if a.startswith("--")}
    args = [a for a in rest if not a.startswith("--")]
    if cmd == "plan":
        cmd_plan()
    elif cmd == "local":
        only = None
        if "--only" in rest:
            only = set(rest[rest.index("--only") + 1].split(","))
        cmd_local(only, "--force" in flags)
    elif cmd == "stage":
        cmd_stage(args, "--force" in flags, "--keep-going" in flags)
    elif cmd == "submit":
        jobs = set(rest[rest.index("--jobs") + 1].split(",")) if "--jobs" in rest else None
        after = rest[rest.index("--after") + 1] if "--after" in rest else None
        cmd_submit("--dry-run" in flags, "--force" in flags, jobs, after)
    elif cmd == "status":
        cmd_status()
    elif cmd == "pick-mig":
        cmd_pick_mig()
    elif cmd == "push":
        cmd_push("--no-raw" in flags, "--with-results" in flags)
    elif cmd == "pull":
        cmd_pull()
    elif cmd == "remote-setup":
        cmd_remote_setup()
    elif cmd == "remote":
        cmd_remote(rest)
    elif cmd == "_internal":
        {"env_check": internal_env_check, "build_verify": internal_build_verify, "report": internal_report,
         "freeze": internal_freeze}[args[0]]()
    else:
        sys.exit(f"unknown command {cmd}; see 'python run_all.py help'")


if __name__ == "__main__":
    main(sys.argv[1:])
