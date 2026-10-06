#!/usr/bin/env bash
# Dev chain 3: after learned baselines exist, add learned-baseline ladders; after Dev-2 ladders, offline PART evaluation.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=3 PYTHONIOENCODING=utf-8
while ! grep -q LEARNED_DONE 12_logs/dev_learned_baselines_v2.log 2>/dev/null; do sleep 60; done
while ! grep -q "dev2_ett exit" 12_logs/dev_chain.log 2>/dev/null; do sleep 60; done
py -3.10 -u 04_code/scripts/dev/run_dev_ladder.py dev2_nyiso --seeds 3 >> 12_logs/dev2_nyiso_learned.log 2>&1
echo "$(date '+%F %T') dev2_nyiso learned exit=$?" >> 12_logs/dev_chain3.log
py -3.10 -u 04_code/scripts/dev/eval_part_offline.py dev2_nyiso --instance >> 12_logs/part_offline_nyiso.log 2>&1
echo "$(date '+%F %T') offline nyiso exit=$?" >> 12_logs/dev_chain3.log
py -3.10 -u 04_code/scripts/dev/eval_part_offline.py dev2_ett >> 12_logs/part_offline_ett.log 2>&1
echo "$(date '+%F %T') offline ett exit=$?" >> 12_logs/dev_chain3.log
py -3.10 -u 04_code/scripts/dev/eval_part_offline.py dev2_india --instance >> 12_logs/part_offline_india_full.log 2>&1
echo "$(date '+%F %T') offline india exit=$?" >> 12_logs/dev_chain3.log
py -3.10 -u 04_code/scripts/dev/x3_capacity.py >> 12_logs/dev_x3.log 2>&1
echo "$(date '+%F %T') x3 exit=$?" >> 12_logs/dev_chain3.log
echo CHAIN_DEV3_DONE >> 12_logs/dev_chain3.log
