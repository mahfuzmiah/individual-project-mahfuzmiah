#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
LOGFILE="$(pwd)/lstm_epoch_runs.log"
echo "=== LSTM batch run started: $(date) ===" > "$LOGFILE"

# adjust path if your folder is Deep_learning vs Deep_Learning
for ep in 50, 100, 200, 500; do
  echo "[$(date '+%H:%M:%S')] Running epochs=$ep" | tee -a "$LOGFILE"
  python modeling/Deep_Learning/LSTM1.py --epochs "$ep" >> "$LOGFILE" 2>&1
done

echo "=== All LSTM runs complete: $(date) ===" >> "$LOGFILE"
