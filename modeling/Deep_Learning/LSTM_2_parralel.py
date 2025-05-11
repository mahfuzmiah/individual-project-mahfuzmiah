
import sys
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


os.environ['PYTHONHASHSEED'] = '42'
# 2) Force TensorFlow to deterministic ops
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # disable GPU for full determinism

import tensorflow as tf  # nopep8

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# 3) Seed everything
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_PATH))
from config import REPO_ROOT, DATASETS_DIR, IMPUTED_RESULTS_DIR_TEST, IMPUTED_RESULTS_DIR_TRAIN, DIAGRAMS_DIR  # nopep8


TRAIN_PATH = IMPUTED_RESULTS_DIR_TRAIN / "knn.csv"
TEST_PATH = IMPUTED_RESULTS_DIR_TEST / "knn.csv"
OUT_DIR = DIAGRAMS_DIR / "LSTM_Results_diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# --- Utility Functions ---


def wmape(y_true, y_pred):
    denom = np.nansum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.nansum(np.abs(y_true - y_pred)) / denom


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix >= len(sequence):
            break
        seq_x = sequence[i:end_ix]
        seq_y = sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return hours, minutes, secs


# Load data globally so that child processes can pickle them
train, test = load_data()
# assume first 3 columns are metadata
train_time_series_cols = train.columns[3:]
test_time_series_cols = test.columns[3:]

n_steps = 4         # number of time steps per input
n_features = 1      # univariate data

# Define a function that processes a single row (index i)


def process_row(i):
    # For reproducibility, you can re-import necessary modules here if needed.
    # Training part:
    seed = SEED + i
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    train_series = train[train_time_series_cols].iloc[i].values
    # Check if the training series is constant
    if np.std(train_series) < 1e-6:
        constant_val = train_series[0]
        X_train, y_train = split_sequence(train_series.tolist(), n_steps)
        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], n_features))
        # Use constant value; no model training
        model_trained = None
    else:
        X_train, y_train = split_sequence(train_series.tolist(), n_steps)
        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], n_features))
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=200, verbose=0)
        model_trained = model

    # Prediction part:
    test_series = test[test_time_series_cols].iloc[i].values
    horizon = len(test_series)
    n_predictions = horizon - n_steps
    current_input = test_series[:n_steps].copy()
    predictions = []
    for j in range(n_predictions):
        if model_trained is None:
            pred = constant_val
        else:
            x_input = current_input.reshape((1, n_steps, n_features))
            yhat = model_trained.predict(x_input, verbose=0)
            pred = yhat[0][0]
        predictions.append(pred)
        current_input = np.concatenate([current_input[1:], [pred]])
    actual = test_series[n_steps:]
    return {'row': i, 'forecast': np.array(predictions), 'actual': np.array(actual)}


# Process rows in parallel (for example, for the first 200 rows)
if __name__ == '__main__':
    NO_ITERATIONS = 20
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_row, range(NO_ITERATIONS)))
    end_time = time.time()
    # Then process results and plot, etc.

    total_time = end_time - start_time
    average_time = total_time / NO_ITERATIONS
    total_predicted_time = average_time * len(train)
    hours, minutes, secs = seconds_to_hms(total_time)
    hours_avg, minutes_avg, secs_avg = seconds_to_hms(average_time)
    hours_predicted, minutes_predicted, secs_predicted = seconds_to_hms(
        total_predicted_time)

    print("Total time taken to process dataset: {} hours, {} minutes, {:.2f} seconds".format(
        hours, minutes, secs))
    print("Average time taken per row: {} hours, {} minutes, {:.2f} seconds".format(
        hours_avg, minutes_avg, secs_avg))
    print("Total time to predict entire test set: {} hours, {} minutes, {:.2f} seconds".format(
        hours_predicted, minutes_predicted, secs_predicted))

    # ---------------- Graphing ----------------

    # Graph 1: Forecast vs Actual for the first processed row (row 0)
    first_result = results[0]
    date_labels = list(test_time_series_cols[n_steps:])
   # time_index = list(range(n_steps, n_steps + len(first_result['forecast'])))
    plt.figure(figsize=(10, 5))
    plt.plot(date_labels, first_result['actual'], label='Actual', marker='o')
    plt.plot(date_labels, first_result['forecast'],
             label='Forecast (LSTM)', marker='x')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("LSTM Forecast vs Actual for Row 0")
    plt.legend()
    plt.savefig(OUT_DIR / "LSTM_Parrallel_forecast_row_0.png")

    plt.show()

    # Graph 2: WMAPE over time across the processed rows
    H = len(results[0]['forecast'])
    wmape_by_time = []
    for j in range(H):
        all_actual = np.array([res['actual'][j] for res in results])
        all_forecast = np.array([res['forecast'][j] for res in results])
        wmape_val = wmape(all_actual, all_forecast)
        wmape_by_time.append(wmape_val)

    for j, val in enumerate(wmape_by_time):
        print(f"Time step {j}: WMAPE = {val}")

    xvals = list(range(n_steps, n_steps + H))
    plt.figure(figsize=(10, 5))
    plt.plot(date_labels, wmape_by_time, marker='o')
    plt.xlabel("Forecast Time Step")
    plt.ylabel("WMAPE")
    plt.title(f"WMAPE Over Time (LSTM Forecasts) Across {NO_ITERATIONS} Rows")
    plt.grid(True)
    plt.savefig(OUT_DIR / "LSTM_Parrallel_WMAPE_over_time.png", dpi=300)
    plt.show()
