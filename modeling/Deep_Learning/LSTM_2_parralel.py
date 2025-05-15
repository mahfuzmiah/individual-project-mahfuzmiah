
''' RUN program with python modeling/Deep_learning/LSTM_2_parralel.py --epochs 50 100 200 --rows 200'''
#!/usr/bin/env python
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import argparse
import sys
import time
import random
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf  # noqa
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ─── Paths & Config ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import IMPUTED_RESULTS_DIR_TRAIN, IMPUTED_RESULTS_DIR_TEST  # noqa

TRAIN_PATH = IMPUTED_RESULTS_DIR_TRAIN / "knn.csv"
TEST_PATH = IMPUTED_RESULTS_DIR_TEST / "knn.csv"
PRED_DIR = REPO_ROOT / "predictions" / "LSTM"
RUNTIME_DIR = PRED_DIR / "runtimes"
PRED_DIR.mkdir(exist_ok=True)
RUNTIME_DIR.mkdir(exist_ok=True)

# ─── Load Data Globally ───────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
TS_COLS_TRAIN = train.columns[3:]
TS_COLS_TEST = test.columns[3:]

# ─── Utility Functions ────────────────────────────────────────────────────────


def wmape(y_true, y_pred):
    d = np.nansum(np.abs(y_true))
    return np.nan if d == 0 else np.nansum(np.abs(y_true-y_pred))/d


def split_sequence(seq, n_steps):
    X, y = [], []
    for i in range(len(seq)-n_steps):
        X.append(seq[i:i+n_steps])
        y.append(seq[i+n_steps])
    return np.array(X), np.array(y)

# ─── Row Processor ────────────────────────────────────────────────────────────


def process_row(args):
    i, epochs = args
    # re-seed for reproducibility per row
    seed = 42 + i
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    series = train[TS_COLS_TRAIN].iloc[i].values
    n_steps, n_features = 4, 1
    # train
    if np.std(series) < 1e-6:
        constant = series[0]
        model = None
    else:
        X, y = split_sequence(series, n_steps)
        X = X.reshape((-1, n_steps, n_features))
        model = Sequential([LSTM(50, activation='relu',
                                 input_shape=(n_steps, n_features)),
                            Dense(1)])
        model.compile('adam', 'mse')
        model.fit(X, y, epochs=epochs, verbose=0)

    # forecast
    test_ser = test[TS_COLS_TEST].iloc[i].values
    buf = list(test_ser[:n_steps])
    preds = []
    for _ in range(len(test_ser)-n_steps):
        if model is None:
            p = constant
        else:
            x = np.array(buf[-n_steps:]).reshape((1, n_steps, n_features))
            p = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        buf.append(p)

    actual = test_ser[n_steps:]
    return i, preds, actual

# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", nargs="+", type=int, default=[200],
                   help="Epochs to try")
    p.add_argument("--rows",   type=int, default=200,
                   help="How many series (rows) to process")
    args = p.parse_args()

    summary = []
    for ep in args.epochs:
        print(f"\n=== epochs={ep} ===")
        # 1) time & forecast in parallel
        start = time.time()
        with ProcessPoolExecutor() as ex:
            futures = ex.map(process_row, [(i, ep) for i in range(args.rows)])
            results = list(futures)
        total = time.time() - start

        # unpack results
        per_row_times = [None]*args.rows
        preds_records = []
        for i, preds, actual in results:
            # here we don't capture per-row runtime; if needed, wrap process_row
            for j, (a, p) in enumerate(zip(actual, preds)):
                preds_records.append({"row": i, "horizon_idx": j,
                                      "actual": a, "forecast": p})

        # 2) save predictions
        pd.DataFrame(preds_records)\
          .to_csv(PRED_DIR/f"lstm2_predictions_{ep}epochs.csv",
                  index=False)
        print("  Wrote predictions for", ep, "epochs")

        # 3) record summary
        summary.append({
            "epochs":            ep,
            "total_time_s":      round(total, 2),
            "avg_time_per_row_s": round(total/args.rows, 4)
        })

    # 4) write summary CSV
    pd.DataFrame(summary)\
      .to_csv(RUNTIME_DIR/"lstm2_epochs_summary.csv",
              index=False,
              columns=["epochs", "total_time_s", "avg_time_per_row_s"])
    print("Wrote epoch summary to", RUNTIME_DIR/"lstm2_epochs_summary.csv")


if __name__ == "__main__":
    main()
