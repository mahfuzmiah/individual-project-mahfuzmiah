# import os
# import time
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# import random
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import StandardScaler
# warnings.filterwarnings("ignore")

# os.environ['PYTHONHASHSEED'] = '42'
# # 2) Force TensorFlow to deterministic ops
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# import tensorflow as tf  # nopep8

# # 3) Seed everything
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
# # Import your config
# REPO_ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(REPO_ROOT))
# from config import DATASETS_DIR, DIAGRAMS_DIR, IMPUTED_RESULTS_DIR_TEST, IMPUTED_RESULTS_DIR_TRAIN  # nopep8

# TRAIN_PATH = IMPUTED_RESULTS_DIR_TRAIN / "knn.csv"
# TEST_PATH = IMPUTED_RESULTS_DIR_TEST / "knn.csv"
# OUT_DIR = DIAGRAMS_DIR / "LSTM_Results_diagrams"
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# # --- Utility Functions ---
# # Parameters:
# n_steps = 4       # number of time steps per input window
# block_size = 4    # number of rows to group together as features

# # --- Utility Functions ---

# scaler = StandardScaler()


# def wmape(y_true, y_pred):
#     return (np.abs(y_true - y_pred)).sum() / np.abs(y_true).sum()


# def calculate_mae(y_true, y_pred):
#     """
#     Compute the Mean Absolute Error (MAE) between actual and forecast values.

#     Parameters:
#         y_true (np.array): The actual values.
#         y_pred (np.array): The forecasted values.

#     Returns:
#         float: The MAE.
#     """
#     return np.mean(np.abs(y_true - y_pred))


# def calculate_smape(y_true, y_pred):
#     """
#     Compute the Symmetric Mean Absolute Percentage Error (SMAPE) between actual and forecast values.

#     SMAPE is defined as:
#         SMAPE = (100/n) * sum(2 * |y_true - y_pred| / (|y_true| + |y_pred|))

#     Parameters:
#         y_true (np.array): The actual values.
#         y_pred (np.array): The forecasted values.

#     Returns:
#         float: The SMAPE percentage.
#     """
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
#     # Avoid division by zero by setting any zeros to a small number
#     denominator[denominator == 0] = 1e-8
#     smape = np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100
#     return smape


# def plot_actual_vs_forecast(row_id, actual, forecast):
#     """
#     Plot the actual vs. forecast values for a given row.

#     Parameters:
#         row_id (int): The global row index.
#         actual (np.array): The actual time series values.
#         forecast (np.array): The forecasted time series values.
#     """
#     plt.figure(figsize=(10, 5))
#     plt.plot(actual, label='Actual', marker='o')
#     plt.plot(forecast, label='Forecast', marker='o')
#     plt.title(f'Row {row_id} Actual vs Forecast')
#     plt.xlabel('Time Step')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(
#         OUT_DIR / f"Row_{row_id}_Actual_vs_Forecast_Combined_Parrallel.png", dpi=300)
#     plt.show()


# def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
#     train = pd.read_csv(train_path)
#     test = pd.read_csv(test_path)
#     return train, test


# def seconds_to_hms(seconds):
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     secs = seconds % 60
#     return hours, minutes, secs


# def split_sequence_vectorized(sequence, n_steps):
#     """
#     Given a 2D array 'sequence' of shape (T, R) (time steps T, features R),
#     create overlapping windows of length n_steps and the next row as target.
#     """
#     X, y = [], []
#     T = sequence.shape[0]
#     for i in range(T - n_steps):
#         seq_x = sequence[i:i+n_steps, :]   # shape: (n_steps, R)
#         seq_y = sequence[i+n_steps, :]       # shape: (R,)
#         X.append(seq_x)
#         y.append(seq_y)
#     return np.array(X), np.array(y)

# # ---------- Processing a Block of Rows ----------


# def create_blocks(data, block_size):
#     """
#     Splits the 2D array 'data' (with rows representing time and columns representing features)
#     into a list of blocks, where each block consists of 'block_size' consecutive rows.
#     If the total number of rows is not an exact multiple of block_size, the final block
#     will contain the remaining rows (which may be fewer than block_size).

#     Parameters:
#         data (np.array): 2D array of shape (n_rows, n_columns)
#         block_size (int): Number of rows per block

#     Returns:
#         list of np.array: Each element in the list is a block with shape (block_size, n_columns)
#                           except possibly the last one if there aren't enough rows.
#     """
#     blocks = []
#     n_rows = data.shape[0]
#     for i in range(0, n_rows, block_size):
#         block = data.iloc[i:i + block_size, :]
#         blocks.append(block)
#     return blocks


# def process_block(Blocks):
#     block_index, (Block_train, Block_test) = Blocks
#     train_data = Block_train[Block_train.columns[3:]].values
#     train_data = train_data.T
#     train_data = scaler.fit_transform(train_data)
#     test_data = Block_test[Block_test.columns[3:]].values
#     test_data = test_data.T
#     test_data = scaler.transform(test_data)

#     n_steps = 4
#     n_features = train_data.shape[1]
#     X_train, y_train = split_sequence_vectorized(
#         train_data, n_steps)
#     model = Sequential()
#     model.add(LSTM(100, input_shape=(n_steps, n_features), return_sequences=True))
#     model.add(LSTM(50))
#     model.add(Dense(n_features))
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X_train, y_train, epochs=500, verbose=0)
#     # For iterative prediction on test data:
#     T_test = test_data.shape[0]   # total number of time steps in test data
#     n_predictions = T_test - n_steps  # number of predictions to generate

#     # Use the first n_steps rows of test_data_T as the initial window:
#     current_input = test_data[:n_steps, :]  # shape: (n_steps, n_features)
#     predictions = []
#     start_time = time.time()
#     for i in range(n_predictions):
#         x_input = current_input.reshape((1, n_steps, n_features))
#         yhat = model.predict(x_input, verbose=1)  # shape: (1, n_features)
#         predictions.append(yhat[0])

#         # Update window: remove the oldest row and append the new predicted row:
#         current_input = np.concatenate([current_input[1:], yhat], axis=0)

#     predictions = np.array(predictions)  # shape: (n_predictions, n_features)

#     # The full forecast: combine the initial seed with predictions (should match test_data_T shape)
#     full_forecast = np.concatenate(
#         [test_data[:n_steps, :], predictions], axis=0)
#     full_forecast_original = scaler.inverse_transform(full_forecast)
#     test_data_original = scaler.inverse_transform(test_data)
#     end_time = time.time()
#     results = []
#     # Loop over each original row (each column in full_forecast)
#     for i in range(n_features):
#         # Forecast for row i: skip the initial n_steps seed values
#         forecast = full_forecast_original[n_steps:, i]
#         global_row_index = block_index * block_size + i

#         actual = test_data_original[n_steps:, i]
#         results.append(
#             {'row': global_row_index, 'forecast': forecast, 'actual': actual})
#     return results


# if __name__ == '__main__':
#     # Load data
#     train, test = load_data()

#     df = pd.read_csv(TEST_PATH)

#     # Convert applicable columns to numeric types explicitly
#     df = df.infer_objects(copy=False)

#     # Select numerical columns separately
#     numeric_cols = df.select_dtypes(include=['number']).columns

#     # Create a copy of the original DataFrame to retain non-numeric data
#     df_filled = df.copy()

#     # Interpolate only numerical columns using linear interpolation along rows (axis=1)
#     df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(
#         method='linear',
#         axis=1,
#         limit_direction='both'
#     )

#     # Fill any remaining NaN values with 0
#     test = df_filled.fillna(0)
#   # Only take first 200 rows for test and train
#     train_sample = train.iloc[:5]
#     test_sample = test.iloc[:5]

#     train_blocks = create_blocks(train_sample, block_size)
#     test_blocks = create_blocks(test_sample, block_size)

#     # Combine matching blocks for processing
#     block_pairs = list(zip(train_blocks, test_blocks))

#     print(f"Processing {len(block_pairs)} blocks in parallel...")

#     NO_ROWS_PROCESSED = 200  # or total rows in your train_sample
#     start_time = time.time()

#     with ProcessPoolExecutor() as executor:
#         # Each block processes block_size rows. So we use enumerate(zip(...)) as before.
#         indexed_block_pairs = list(enumerate(zip(train_blocks, test_blocks)))
#         results = list(executor.map(process_block, indexed_block_pairs))

#     end_time = time.time()
#     total_time = end_time - start_time
#     average_time_per_row = total_time / NO_ROWS_PROCESSED
#     total_predicted_time = average_time_per_row * \
#         len(train)  # or total rows in your full dataset

#     print("Total time for subset: {:.2f} seconds".format(total_time))
#     print("Average time per row: {:.4f} seconds".format(average_time_per_row))
#     print("Predicted total time for full dataset: {:.2f} seconds".format(
#         total_predicted_time))
#     # Example: Flatten results and print WMAPE for each row
#     flat_results = [item for sublist in results for item in sublist]

#     # for res in flat_results:
#     #     # if res['row'] == 0:
#     #     #     print("First few actuals:", res['actual'][:10])
#     #     #     print("First few forecasts:", res['forecast'][:10])
#     #     #     break  # Just do it for one row
#     #     if len(res['actual']) == 0 or len(res['forecast']) == 0:
#     #         print(
#     #             f"Skipping row {res['row']} due to empty actual or forecast.")
#     #         continue
#     #     score = wmape(res['actual'], res['forecast'])
#     #     print(f"Row {res['row']} WMAPE: {score:.4f}")

#     for res in flat_results:
#         denom = np.abs(res['actual']).sum()
#         if denom < 1e-3:
#             print(
#                 f"Row {res['row']} has very small actual values; WMAPE may be inflated.")
#             continue
#         score = wmape(res['actual'], res['forecast'])
#         print(f"Row {res['row']} WMAPE: {score:.4f}")
#     rows_to_investigate = [40, 43, 128, 196]

#     for res in flat_results:
#         if res['row'] in rows_to_investigate:
#             mae = calculate_mae(res['actual'], res['forecast'])
#             smape = calculate_smape(res['actual'], res['forecast'])
#             print(f"Row {res['row']}: MAE = {mae:.4f}, SMAPE = {smape:.2f}%")
#             plot_actual_vs_forecast(res['row'], res['actual'], res['forecast'])
#     H = len(flat_results[0]['forecast'])
#     wmape_by_time = []
#     for j in range(H):
#         all_actual = np.array([res['actual'][j] for res in flat_results])
#         all_forecast = np.array([res['forecast'][j] for res in flat_results])
#         wmape_val = wmape(all_actual, all_forecast)
#         wmape_by_time.append(wmape_val)

#     for j, val in enumerate(wmape_by_time):
#         print(f"Time step {j}: WMAPE = {val}")
#     train, test = load_data()
#     test_time_series_cols = test.columns[3:]

#     date_labels = list(test_time_series_cols[n_steps:])

#     xvals = list(range(n_steps, n_steps + H))
#     plt.figure(figsize=(10, 5))
#     plt.plot(date_labels, wmape_by_time, marker='o')
#     plt.xlabel("Forecast Time Step")
#     plt.ylabel("WMAPE")
#     plt.title(f"WMAPE Over Time (LSTM Forecasts) Across Rows")
#     plt.grid(True)
#     plt.savefig(
#         OUT_DIR / "LSTM_Combined_Parrallel_WMAPE_over_time.png", dpi=300)

#     plt.show()
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
