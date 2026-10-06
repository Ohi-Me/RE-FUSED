#!/usr/bin/env bash
# Confirmatory battery (pre-registered). Requires the frozen pre-registration (hash + git tag prereg-v1).
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 REFUSED_CONFIRMATORY=1
L=logs/confirm_chain.log
echo "$(date '+%F %T') start commit=$(git rev-parse HEAD) tag=$(git describe --tags --exact-match 2>/dev/null)" >> $L
py -3.10 -c "import sys; sys.path.insert(0,'code'); from refused_gate import guard; guard.require_confirmatory(); print('unlocked')" >> $L 2>&1 || { echo "BLOCKED" >> $L; exit 1; }
py -3.10 -u code/scripts/dev/make_learned_baselines.py --mode confirm --models lgbm,chronos,neural >> logs/confirm_learned_baselines.log 2>&1
echo "$(date '+%F %T') learned baselines exit=$?" >> $L
py -3.10 -u code/scripts/confirm/run_confirmatory.py >> logs/confirm_ladders.log 2>&1
echo "$(date '+%F %T') ladders exit=$?" >> $L
py -3.10 -u code/scripts/dev/x5_x8_prob_energy.py --mode confirm --seeds 3 >> logs/confirm_x5_x8.log 2>&1
echo "$(date '+%F %T') x5_x8 exit=$?" >> $L
py -3.10 -u code/scripts/dev/x1_phase_diagram.py --part partition --source opsd_confirm --seed_offset 900 --tag confirm >> logs/confirm_x1_partition.log 2>&1
echo "$(date '+%F %T') x1 partition exit=$?" >> $L
py -3.10 -u code/scripts/dev/x1_phase_diagram.py --part instance --source opsd_confirm --seed_offset 900 --tag confirm >> logs/confirm_x1_instance.log 2>&1
echo "$(date '+%F %T') x1 instance exit=$?" >> $L
py -3.10 -u code/scripts/confirm/analyze_confirmatory.py --map confirm >> logs/confirm_analysis.log 2>&1
echo "$(date '+%F %T') analysis exit=$?" >> $L
echo CONFIRM_CHAIN_DONE >> $L
