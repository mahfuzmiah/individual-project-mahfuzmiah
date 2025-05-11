
import pandas as pd  # ensure pandas is in scope
import os
import random
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


os.environ['PYTHONHASHSEED'] = '42'
# 2) Force TensorFlow to deterministic ops
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf  # nopep8

# 3) Seed everything
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# 1) Make Python’s hash seed fixed


# 4) Import your config
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import DATASETS_DIR, DIAGRAMS_DIR, IMPUTED_RESULTS_DIR_TEST, IMPUTED_RESULTS_DIR_TRAIN  # nopep8

TRAIN_PATH = IMPUTED_RESULTS_DIR_TRAIN / "knn.csv"
TEST_PATH = IMPUTED_RESULTS_DIR_TEST / "knn.csv"
OUT_DIR = DIAGRAMS_DIR / "LSTM_Results_diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# --- Utility Functions ---


def wmape(y_true, y_pred):
    # Use np.nansum to sum while ignoring NaNs
    denom = np.nansum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.nansum(np.abs(y_true - y_pred)) / denom


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# --- Define the splitting function for a univariate sequence ---


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


def main():

    # Load data
    train, test = load_data()

    # Time-series columns
    train_cols = train.columns[3:]
    test_cols = test.columns[3:]

    n_steps = 4
    n_features = 1
    results = []
    NO_ITERATIONS = 2

    start_time = time.time()
    for i in range(NO_ITERATIONS):
        # --- 1) Fit on row i of train ---
        train_series = train[train_cols].iloc[i].values
        if np.std(train_series) < 1e-6:
            constant_val = train_series[0]
            model_trained = None
            print(f"Row {i} constant; using {constant_val}")
        else:
            X_train, y_train = split_sequence(train_series.tolist(), n_steps)
            X_train = X_train.reshape((len(X_train), n_steps, n_features))
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=200, verbose=0)
            model_trained = model

        # --- 2) Warm-up window from end of training series ---
        current_input = train_series[-n_steps:].copy()

        # --- 3) Forecast full test horizon ---
        test_series = test[test_cols].iloc[i].values
        horizon = len(test_series)
        preds = []

        for _ in range(horizon):
            if model_trained is None:
                pred = constant_val
            else:
                x_in = current_input.reshape((1, n_steps, n_features))
                pred = model_trained.predict(x_in, verbose=0)[0][0]
            preds.append(pred)
            current_input = np.concatenate([current_input[1:], [pred]])

        results.append({
            'row':     i,
            'forecast': np.array(preds),
            'actual':   test_series
        })

    # End timer after processing all rows
    end_time = time.time()
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

    first = results[0]
    # build PeriodIndex + timestamps
    periods = pd.PeriodIndex(test_cols, freq='Q')
    # use 'start' so grid & points line up
    dates = periods.to_timestamp(how='start')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, first['actual'],   label='Actual',   marker='o')
    ax.plot(dates, first['forecast'], label='Forecast', marker='x')

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("LSTM Forecast vs Actual for Row 0")

    # major ticks every 4 quarters
    major_dates = dates[::4]
    major_periods = periods[::4]
    ax.set_xticks(major_dates)
    ax.set_xticklabels(
        [f"{p.year}-Q{p.quarter}" for p in major_periods],
        rotation=45, ha='center'
    )

    # minor ticks at every quarter (for gridlines)
    ax.set_xticks(dates, minor=True)

    # grid
    ax.grid(which='major', linewidth=1, alpha=0.8)
    ax.grid(which='minor', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR/"LSTM1_forecast_row_0_times.png", dpi=300)
    plt.show()

    # 1) compute wmape_by_time
    H = len(results[0]['forecast'])
    wmape_by_time = [
        wmape(
            np.array([r['actual'][j] for r in results]),
            np.array([r['forecast'][j] for r in results])
        )
        for j in range(H)
    ]

    # 2) build PeriodIndex + corresponding timestamps
    # e.g. Period('2020Q1')
    periods = pd.PeriodIndex(test_cols, freq='Q')
    # e.g. Timestamp('2020-03-31')
    dates = periods.to_timestamp()

    # 3) plot everything
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, wmape_by_time, marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("WMAPE")
    ax.set_title(
        f"WMAPE Over Time (LSTM Forecasts) Across {NO_ITERATIONS} Rows")

    # 4) ticks & labels
    # every 4th quarter
    major_dates = dates[::4]
    major_periods = periods[::4]
    ax.set_xticks(major_dates)
    ax.set_xticklabels(
        [f"{p.year}-Q{p.quarter}" for p in major_periods],
        ha='center'
    )

    # minor ticks at every single quarter for grid lines
    ax.set_xticks(dates, minor=True)

    # 5) grid on both major and minor
    ax.grid(which='major', linewidth=1, alpha=0.8)
    ax.grid(which='minor', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "LSTM1_WMAPE_over_time.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
