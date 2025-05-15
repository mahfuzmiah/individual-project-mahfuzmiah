'''RUN WIHT python modeling/Deep_Learning/LSTM_3_Vectorised.py.py --epochs 100 500 1000 --rows 200'''

#!/usr/bin/env python
import argparse
import time
import os
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

# ─── Determinism & Seeds ─────────────────────────────────────────────────────
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ─── Paths & Config ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import IMPUTED_RESULTS_DIR_TRAIN, IMPUTED_RESULTS_DIR_TEST, DIAGRAMS_DIR  # nopep8

TRAIN_CSV = IMPUTED_RESULTS_DIR_TRAIN / "knn.csv"
TEST_CSV = IMPUTED_RESULTS_DIR_TEST / "knn.csv"

PRED_DIR = REPO_ROOT / "predictions" / "LSTM"
RUNTIME_DIR = PRED_DIR / "runtimes"
PRED_DIR.mkdir(exist_ok=True)
RUNTIME_DIR.mkdir(exist_ok=True)

OUT_DIR = DIAGRAMS_DIR / "LSTM_Results_diagrams"
OUT_DIR.mkdir(exist_ok=True)

# ─── Utility Functions ────────────────────────────────────────────────────────


def wmape(y_true, y_pred):
    d = np.nansum(np.abs(y_true))
    return np.nan if d == 0 else np.nansum(np.abs(y_true-y_pred))/d


def split_sequence_vectorized(seq, n_steps):
    X, y = [], []
    for i in range(seq.shape[0]-n_steps):
        X.append(seq[i:i+n_steps])
        y.append(seq[i+n_steps])
    return np.array(X), np.array(y)

# ─── Main ─────────────────────────────────────────────────────────────────────


def main(epochs_list, max_rows=None):
    # 1) load & (optionally) truncate rows
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    if max_rows:
        train_df = train_df.iloc[:max_rows]
        test_df = test_df.iloc[:max_rows]
    # 2) extract just the time‐series columns
    train_data = train_df.iloc[:, 3:].values   # shape (R, T)
    test_data = test_df .iloc[:, 3:].values

    # 3) scale/transpose
    scaler = StandardScaler()
    train_T = scaler.fit_transform(train_data.T)
    test_T = scaler.transform(test_data .T)

    n_steps = 4
    # n_features = train_T.shape[1]   # = number of (possibly truncated) rows
    n_features = train_T.shape[1]  # = number of (possibly truncated) rows

    X_train, y_train = split_sequence_vectorized(train_T, n_steps)

    # loop over each epoch setting
    summary = []
    for ep in epochs_list:
        print(f"\n=== EPOCHS = {ep} ===")
        # 4) build & train
        model = Sequential([
            LSTM(100, return_sequences=True,
                 input_shape=(n_steps, n_features)),
            LSTM(50),
            Dense(n_features)
        ])
        model.compile('adam', 'mse')

        t0 = time.time()
        model.fit(X_train, y_train, epochs=ep, verbose=0)
        # 5) iterative forecast
        preds = []
        buf = list(test_T[:n_steps])
        H = test_T.shape[0] - n_steps
        for _ in range(H):
            x = np.array(buf[-n_steps:]).reshape((1, n_steps, n_features))
            p = model.predict(x, verbose=0)[0]
            preds.append(p)
            buf.append(p)
        total_sec = time.time() - t0

        # invert‐scale & reshape
        full = np.vstack([test_T[:n_steps], np.array(preds)])
        inv = scaler.inverse_transform(full)
        # extract the forecasted rows/time‐steps
        recs = []
        for row in range(n_features):
            actual = test_data[row, n_steps:]
            forecast = inv[n_steps:, row]
            for j, (a, f) in enumerate(zip(actual, forecast)):
                recs.append({
                    'row': row,
                    'horizon_idx': j,
                    'actual': a,
                    'forecast': f
                })
        # 6) dump this epoch’s predictions
        out_pred = PRED_DIR/f"vector_lstm_preds_{ep}epochs.csv"
        pd.DataFrame.from_records(recs).to_csv(out_pred, index=False)
        print(" Wrote:", out_pred.name)

        # 7) record runtime summary
        summary.append({
            'epochs': ep,
            'total_time_s': round(total_sec, 2),
            'avg_time_per_row_s': round(total_sec/n_features, 4)
        })

    # 8) write out the summary once
    pd.DataFrame(summary)\
      .to_csv(
        RUNTIME_DIR/"vector_lstm_summary.csv",
        index=False,
        columns=['epochs', 'total_time_s', 'avg_time_per_row_s']
    )
    print("Wrote summary to:", "vector_lstm_summary.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", nargs="+", type=int,
                   default=[100, 500, 1000],
                   help="Which epoch counts to try")
    p.add_argument("--rows", type=int, default=None,
                   help="If set, only use the first N rows of the data")
    args = p.parse_args()

    main(args.epochs, args.rows)
