#!/usr/bin/env bash
# Deviation DC1b (sensitivity after unblinding): semi-synthetic grid with the OPSD plausibility filter, then H1 re-analysis.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 REFUSED_CONFIRMATORY=1
L=logs/sensitivity_dc1b.log
echo "$(date '+%F %T') start commit=$(git rev-parse --short HEAD)" >> $L
py -3.10 -u code/scripts/dev/x1_phase_diagram.py --part partition --source opsd_confirm --seed_offset 900 --tag confirm_dc1b --clean >> $L 2>&1
echo "$(date '+%F %T') x1 partition exit=$?" >> $L
py -3.10 -u code/scripts/dev/x1_phase_diagram.py --part instance --source opsd_confirm --seed_offset 900 --tag confirm_dc1b --clean >> $L 2>&1
echo "$(date '+%F %T') x1 instance exit=$?" >> $L
py -3.10 -u code/scripts/confirm/analyze_confirmatory.py --map sensitivity >> logs/sensitivity_dc1_analysis.log 2>&1
echo "$(date '+%F %T') analysis exit=$?" >> $L
echo DC1B_DONE >> $L
