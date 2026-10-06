#!/usr/bin/env bash
# Development chain 2: learned baselines -> Dev-2 ladders (PART designs, PART-time) -> X5/X8 -> X1 phase diagrams
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
while ! grep -q CHAIN_DEV1_DONE logs/dev_chain.log 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') chain2 start" >> logs/dev_chain.log
py -3.10 -u code/scripts/dev/make_learned_baselines.py --mode dev --models lgbm,chronos,neural >> logs/dev_learned_baselines.log 2>&1
echo "$(date '+%F %T') learned baselines exit=$?" >> logs/dev_chain.log
for e in dev2_nyiso dev2_india dev2_ett; do
  py -3.10 -u code/scripts/dev/run_dev_ladder.py $e --seeds 3 >> logs/$e.log 2>&1
  echo "$(date '+%F %T') $e exit=$?" >> logs/dev_chain.log
done
py -3.10 -u code/scripts/dev/x5_x8_prob_energy.py --seeds 3 >> logs/dev_x5_x8.log 2>&1
echo "$(date '+%F %T') x5_x8 exit=$?" >> logs/dev_chain.log
py -3.10 -u code/scripts/dev/x1_phase_diagram.py --part partition --seeds 3 >> logs/dev_x1_partition.log 2>&1
echo "$(date '+%F %T') x1 partition exit=$?" >> logs/dev_chain.log
py -3.10 -u code/scripts/dev/x1_phase_diagram.py --part instance --seeds 3 >> logs/dev_x1_instance.log 2>&1
echo "$(date '+%F %T') x1 instance exit=$?" >> logs/dev_chain.log
echo CHAIN_DEV2_DONE >> logs/dev_chain.log
