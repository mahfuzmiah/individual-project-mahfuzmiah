#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
LOGFILE="$(pwd)/model_batch_run.log"
echo "=== Model batch run started: $(date) ===" > "$LOGFILE"

# ───────────────────────────────────────────────
# 1) LSTM_3_Vectorised
echo ">>> Running LSTM_3_Vectorised.py" | tee -a "$LOGFILE"
for ep in 50 100 200 500; do
  echo "[$(date '+%H:%M:%S')] epochs=$ep" | tee -a "$LOGFILE"
  python modeling/Deep_Learning/LSTM_3_Vectorised.py \
    --rows 200 \
    --epochs "$ep" \
    >> "$LOGFILE" 2>&1
done

# ───────────────────────────────────────────────
# 2) LSTM_2_parralel
echo ">>> Running LSTM_2_parralel.py" | tee -a "$LOGFILE"
for ep in 50 100 200 500; do
  echo "[$(date '+%H:%M:%S')] epochs=$ep" | tee -a "$LOGFILE"
  python modeling/Deep_Learning/LSTM_2_parralel.py \
    --rows 200 \
    --epochs "$ep" \
    >> "$LOGFILE" 2>&1
done

# ───────────────────────────────────────────────
# 3) LSTM_4_Combined_Vector_Parrallel
echo ">>> Running LSTM_4_Combined_Vector_Parrallel.py" | tee -a "$LOGFILE"
for bs in 4 8 20 40; do
  for ep in 100 200 500; do
    echo "[$(date '+%H:%M:%S')] blocksizes=$bs  epochs=$ep" | tee -a "$LOGFILE"
    python3 modeling/Deep_Learning/LSTM_4_Combined_Vector_Parrallel.py \
      --rows 200 \
      --blocksizes "$bs" \
      --epochs "$ep" \
      >> "$LOGFILE" 2>&1
  done
done

# ───────────────────────────────────────────────
# 4) ARIMA
echo ">>> Running ArimaModel.py" | tee -a "$LOGFILE"
echo "[$(date '+%H:%M:%S')] ARIMA full run" | tee -a "$LOGFILE"
python modeling/Statistical_models/ArimaModel.py \
  >> "$LOGFILE" 2>&1

# ───────────────────────────────────────────────
# 5) LightGBM
echo ">>> Running LightGBM.py" | tee -a "$LOGFILE"
echo "[$(date '+%H:%M:%S')] LightGBM base model" | tee -a "$LOGFILE"
python modeling/Machine_leaning_models/LightGBM.py \
  >> "$LOGFILE" 2>&1

echo "[$(date '+%H:%M:%S')] LightGBM statistical tuning V4" | tee -a "$LOGFILE"
python modeling/Machine_leaning_models/LightGBM_statistical_tuning_V4.py \
  >> "$LOGFILE" 2>&1

echo "=== All model runs complete: $(date) ===" >> "$LOGFILE"