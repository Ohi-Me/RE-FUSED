#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
for e in nyiso_baselines nyiso_budget india ett; do
  py -3.10 -u 04_code/scripts/dev/run_dev_ladder.py $e --seeds 3 >> 12_logs/dev_$e.log 2>&1
done
echo CHAIN_DEV1_DONE >> 12_logs/dev_chain.log
