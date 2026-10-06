#!/usr/bin/env bash
# Development chain 2: learned baselines -> Dev-2 ladders (PART designs, PART-time) -> X5/X8 -> X1 phase diagrams
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
while ! grep -q CHAIN_DEV1_DONE 12_logs/dev_chain.log 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') chain2 start" >> 12_logs/dev_chain.log
py -3.10 -u 04_code/scripts/dev/make_learned_baselines.py --mode dev --models lgbm,chronos,neural >> 12_logs/dev_learned_baselines.log 2>&1
echo "$(date '+%F %T') learned baselines exit=$?" >> 12_logs/dev_chain.log
for e in dev2_nyiso dev2_india dev2_ett; do
  py -3.10 -u 04_code/scripts/dev/run_dev_ladder.py $e --seeds 3 >> 12_logs/$e.log 2>&1
  echo "$(date '+%F %T') $e exit=$?" >> 12_logs/dev_chain.log
done
py -3.10 -u 04_code/scripts/dev/x5_x8_prob_energy.py --seeds 3 >> 12_logs/dev_x5_x8.log 2>&1
echo "$(date '+%F %T') x5_x8 exit=$?" >> 12_logs/dev_chain.log
py -3.10 -u 04_code/scripts/dev/x1_phase_diagram.py --part partition --seeds 3 >> 12_logs/dev_x1_partition.log 2>&1
echo "$(date '+%F %T') x1 partition exit=$?" >> 12_logs/dev_chain.log
py -3.10 -u 04_code/scripts/dev/x1_phase_diagram.py --part instance --seeds 3 >> 12_logs/dev_x1_instance.log 2>&1
echo "$(date '+%F %T') x1 instance exit=$?" >> 12_logs/dev_chain.log
echo CHAIN_DEV2_DONE >> 12_logs/dev_chain.log
