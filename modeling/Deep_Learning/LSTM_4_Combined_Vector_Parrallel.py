
''' RUN PROGRAM with python3 modeling/Deep_Learning/LSTM_4_Combined_Vector_Parrallel.py --rows 200 --blocksizes 4 8 20 40 --epochs 100 200 500 '''

#!/usr/bin/env python
import argparse
import time
import os
import random
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

TRAIN_CSV = IMPUTED_RESULTS_DIR_TRAIN/"knn.csv"
TEST_CSV = IMPUTED_RESULTS_DIR_TEST/"knn.csv"

PRED_DIR = REPO_ROOT/"predictions" / "LSTM"
RUNTIME_DIR = PRED_DIR/"runtimes"
PRED_DIR.mkdir(exist_ok=True)
RUNTIME_DIR.mkdir(exist_ok=True)
OUT_DIR = DIAGRAMS_DIR/"LSTM_Results_diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Utility Functions ────────────────────────────────────────────────────────


def wmape(y_true, y_pred):
    d = np.nansum(np.abs(y_true))
    return np.nan if d == 0 else np.nansum(np.abs(y_true-y_pred))/d


def split_sequence_vectorized(seq, n_steps):
    X, y = [], []
    T = seq.shape[0]
    for i in range(T-n_steps):
        X.append(seq[i:i+n_steps])
        y.append(seq[i+n_steps])
    return np.array(X), np.array(y)

# This will train & forecast one block (of block_size rows → n_features)


def process_block(args):
    block_idx, (blk_train, blk_test), n_steps, epochs, block_size = args
    # re-seed per-block for reproducibility
    seed = SEED + block_idx
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # prepare block data
    tr = blk_train.iloc[:, 3:].values.T   # shape (T, block_size)
    te = blk_test .iloc[:, 3:].values.T
    scaler = StandardScaler().fit(tr)
    tr = scaler.transform(tr)
    te = scaler.transform(te)

    n_features = tr.shape[1]
    X, y = split_sequence_vectorized(tr, n_steps)

    # build & train
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(n_steps, n_features)),
        LSTM(50),
        Dense(n_features)
    ])
    model.compile('adam', 'mse')
    model.fit(X, y, epochs=epochs, verbose=1)

    # forecast
    horizon = te.shape[0]-n_steps
    buf = list(te[:n_steps])
    preds = []
    for _ in range(horizon):
        x = np.array(buf[-n_steps:]).reshape((1, n_steps, n_features))
        p = model.predict(x, verbose=0)[0]
        preds.append(p)
        buf.append(p)

    # invert scale
    full = np.vstack([te[:n_steps], np.array(preds)])
    full_orig = scaler.inverse_transform(full)
    te_orig = scaler.inverse_transform(te)

    # flatten to per-row results
    rows = []
    for j in range(n_features):
        global_row = block_idx*block_size + j   # now block_size is defined

        actual = te_orig[n_steps:, j]
        forecast = full_orig[n_steps:, j]
        for h, (a, f) in enumerate(zip(actual, forecast)):
            rows.append({
                "row": global_row,
                "horizon_idx": h,
                "actual": a,
                "forecast": f
            })
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows",       type=int,
                   default=200, help="first N rows")
    p.add_argument("--blocksizes", nargs="+", type=int, default=[4, 8, 20, 40],
                   help="feature block sizes")
    p.add_argument("--epochs",     nargs="+", type=int, default=[100, 200, 500],
                   help="epoch counts to try")
    args = p.parse_args()

    # load & optionally truncate
    full_tr = pd.read_csv(TRAIN_CSV).iloc[:args.rows]
    full_te = pd.read_csv(TEST_CSV).iloc[:args.rows]
    n_steps = 4
    summary = []
    for block_size in args.blocksizes:
        # split into blocks
        train_blocks = [full_tr.iloc[i:i+block_size]
                        for i in range(0, args.rows, block_size)]
        test_blocks = [full_te.iloc[i:i+block_size]
                       for i in range(0, args.rows, block_size)]

        for ep in args.epochs:
            print(f"\n→ block_size={block_size}, epochs={ep}")
            jobs = [
                (idx,
                 (train_blocks[idx], test_blocks[idx]),
                 n_steps,
                 ep,
                 block_size)      # ← pass block_size here
                for idx in range(len(train_blocks))
            ]
            t0 = time.time()
            with ProcessPoolExecutor() as ex:
                all_rows = sum(ex.map(process_block, jobs), [])
            total_sec = time.time()-t0

            # dump per‐setting predictions
            out_pred = PRED_DIR/f"lstm_blocks_{block_size}feat_{ep}ep.csv"
            pd.DataFrame(all_rows).to_csv(out_pred, index=False)
            print("  wrote", out_pred.name)

            summary.append({
                "block_size": block_size,
                "epochs": ep,
                "total_time_s": round(total_sec, 2),
                "avg_time_per_row_s": round(total_sec/args.rows, 4)
            })

    # write master summary once
    pd.DataFrame(summary)\
      .to_csv(RUNTIME_DIR/"lstm_blocks_summary.csv",
              index=False,
              columns=["block_size", "epochs", "total_time_s", "avg_time_per_row_s"])
    print("\nwrote summary to", "lstm_blocks_summary.csv")
