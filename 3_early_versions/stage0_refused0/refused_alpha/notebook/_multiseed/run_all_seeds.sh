#!/usr/bin/env bash
set -uo pipefail
cd "D:\\REFUSED_Ready" || cd "/d//REFUSED_Ready"
PY="/c/Users/Ohi/AppData/Local/Programs/Python/Python310/python.exe"
LOG="notebook/_multiseed/run_all_seeds.log"
: > "$LOG"
SEEDS="123 456 789 2024"
for s in $SEEDS; do
  echo "===== SEED $s START $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$LOG"
  "$PY" -m nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    --output "executed_seed${s}.ipynb" \
    --output-dir "notebook/_multiseed" \
    "notebook/_multiseed/input_seed${s}.ipynb" >> "$LOG" 2>&1
  rc=$?
  echo "===== SEED $s END $(date '+%Y-%m-%d %H:%M:%S') rc=$rc =====" | tee -a "$LOG"
done
echo "ALL_SEEDS_DONE" | tee -a "$LOG"
