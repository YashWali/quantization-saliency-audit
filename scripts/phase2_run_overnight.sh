#!/bin/zsh
# Self-healing overnight launcher for the Phase-2 GT run (phase2_03).
#
# phase2_03 is resumable (atomic checkpoints in results/pythia/gt_chunks etc.),
# so this loops it until it finalizes (exit 0) or hits MAXTRIES, resuming after
# any crash/OOM. A per-attempt memory watchdog kills the worker on a swap
# death-spiral (then the loop resumes from the last checkpoint). Launch detached:
#
#   nohup ./scripts/phase2_run_overnight.sh [run-name] >/dev/null 2>&1 &
#   run-name: pythia (default) | smollm2  -> selects config_<run> + results/<run>
#
# Monitor: tail -f results/pythia/gt_run.log   (progress)
#          tail -f results/pythia/gt_vitals.log (rss/swap)
# On completion the log shows "=== FINALIZED ===" and phase2_03 has written
# results/pythia/{ground_truth,gt_loo,gt_split_half}.parquet.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
# Run name selects the frozen config + output namespace; phase2_03 reads
# QSAL_CONFIG to pick the matching frozen config (default keeps Phase-2/Pythia).
RUN="${1:-pythia}"
export QSAL_CONFIG="config_${RUN}"
mkdir -p "results/$RUN"
LOG="results/$RUN/gt_run.log"
VIT="results/$RUN/gt_vitals.log"
MAXTRIES=24
SWAP_KILL_MB=13312   # ~13GB: death-spiral threshold (matches Phase-1 monitor)

n=0
while (( n < MAXTRIES )); do
  n=$((n+1))
  echo "=== attempt $n start $(date '+%F %T') ===" >> "$LOG"
  HF_HUB_OFFLINE=1 ./.venv/bin/python scripts/phase2_03_ground_truth.py >> "$LOG" 2>&1 &
  PYPID=$!

  # per-attempt memory watchdog
  ( while kill -0 $PYPID 2>/dev/null; do
      SWAP=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d 'M' | cut -d. -f1)
      RSS=$(ps -o rss= -p $PYPID 2>/dev/null | awk '{printf "%d",$1/1024}')
      echo "[$(date '+%T')] rss=${RSS}MB swap=${SWAP}MB | $(tail -1 "$LOG" | cut -c1-90)" >> "$VIT"
      if [ "${SWAP:-0}" -gt $SWAP_KILL_MB ]; then
        echo "[$(date '+%T')] KILL $PYPID: swap ${SWAP}MB > ${SWAP_KILL_MB}MB (death-spiral)" >> "$VIT"
        kill $PYPID; sleep 5; kill -9 $PYPID 2>/dev/null
      fi
      sleep 60
    done ) &
  MONPID=$!

  wait $PYPID
  rc=$?
  kill $MONPID 2>/dev/null

  echo "=== attempt $n exit rc=$rc $(date '+%F %T') ===" >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "=== FINALIZED (phase2_03 completed) ===" >> "$LOG"
    break
  fi
  echo "non-zero exit; resuming in 30s (checkpoints preserved)" >> "$LOG"
  sleep 30
done
