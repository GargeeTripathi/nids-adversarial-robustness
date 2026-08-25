#!/usr/bin/env bash
# Wait for the phases 1-3 run to finish, then start phases 4 (poisoning) and 5 (defence).
#
# Chained rather than merged into one command so that phases 1-3 results are already on
# disk and inspectable while the long poisoning sweep runs. Phase 4 retrains every
# architecture once per (strategy, rate) combination - 12 retrains per model - so it is by
# far the most expensive phase.
#
# Completion is detected by the appearance of the phase 3 metrics file rather than by
# process name: process-name matching is unreliable under Git Bash on Windows, and the
# artefact is the thing we actually depend on.

set -u

PROJECT_DIR="/d/NIDS"
PYTHON="$PROJECT_DIR/.venv/Scripts/python.exe"
PHASE3_JSON="$PROJECT_DIR/outputs/metrics/nsl_kdd_phase3_transfer.json"
LOG="$PROJECT_DIR/outputs/logs/phase45_stdout.log"

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="$PROJECT_DIR/src"

echo "waiting for phases 1-3 to complete (watching $PHASE3_JSON)"
until [ -f "$PHASE3_JSON" ]; do
    sleep 30
done

echo "phases 1-3 complete; starting phases 4 and 5"
"$PYTHON" -u -m nids.experiments.run_all \
    --dataset nsl_kdd \
    --data-dir data/raw/nsl_kdd \
    --output-dir outputs \
    --epochs 20 \
    --batch-size 512 \
    --phases 4,5 > "$LOG" 2>&1

echo "phases 4 and 5 finished with exit code $?"
