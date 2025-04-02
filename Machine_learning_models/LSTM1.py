

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

# --- Configuration ---
TRAIN_PATH = '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/imputed_linear.csv'
TEST_PATH = '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/TestingData.csv'

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

# -------------- Data Preparation and Model Training --------------


# Load data
train, test = load_data()

# Assume the first 3 columns are metadata; time-series data starts from the 4th column
train_time_series_cols = train.columns[3:]
test_time_series_cols = test.columns[3:]

n_steps = 4         # number of time steps per input
n_features = 1      # univariate data
results = []
NO_ITERATIONS = 200
# Start timer before processing the rows
start_time = time.time()
# for i in range(len(train)):
for i in range(NO_ITERATIONS):
    # ---- Training on row i from training set ----
    train_series = train[train_time_series_cols].iloc[i].values
    # Check if the training series is constant
    if np.std(train_series) < 1e-6:
        # If constant, no model training needed: forecast the constant value
        constant_val = train_series[0]
        X_train, y_train = split_sequence(train_series.tolist(), n_steps)
        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], n_features))
        # Print a warning message
        print(
            f"Row {i} training data is constant. Using constant value {constant_val} for predictions.")
        model_trained = None
    else:
        X_train, y_train = split_sequence(train_series.tolist(), n_steps)
        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], n_features))
        # Define and compile the LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # Fit the model on this row's training data
        model.fit(X_train, y_train, epochs=200, verbose=0)
        model_trained = model

    # ---- Iterative Prediction for the full test horizon for row i ----
    test_series = test[test_time_series_cols].iloc[i].values
    horizon = len(test_series)  # total number of time steps in test row
    n_predictions = horizon - n_steps  # number of predictions to generate

    # Use the first n_steps values from the test row as the initial window:
    current_input = test_series[:n_steps].copy()
    predictions = []
    for j in range(n_predictions):
        if model_trained is None:
            # If the training data was constant, simply use the constant value
            pred = constant_val
        else:
            x_input = current_input.reshape((1, n_steps, n_features))
            yhat = model_trained.predict(x_input, verbose=0)
            pred = yhat[0][0]
        predictions.append(pred)
        # Update the window: drop the first value and append the new prediction
        current_input = np.concatenate([current_input[1:], [pred]])

    # Store the results for this row:
    actual = test_series[n_steps:]
    results.append({'row': i, 'forecast': np.array(
        predictions), 'actual': np.array(actual)})


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


# ---------------- Graphing ----------------

# Graph 1: Forecast vs Actual for the first row (row 0)
first_result = results[0]
time_index = list(range(n_steps, n_steps + len(first_result['forecast'])))
plt.figure(figsize=(10, 5))
plt.plot(time_index, first_result['actual'], label='Actual', marker='o')
plt.plot(time_index, first_result['forecast'],
         label='Forecast (LSTM)', marker='x')
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("LSTM Forecast vs Actual for Row 0")
plt.legend()
plt.savefig(
    '/Users/mahfuz/Final_project/Final_repo/Diagrams/LSTM1_forecast_selected_series.png')
plt.show()

# Graph 2: WMAPE over time across the 200 rows
# Assuming all rows have the same forecast horizon
H = len(results[0]['forecast'])
wmape_by_time = []
for j in range(H):
    all_actual = np.array([res['actual'][j] for res in results])
    all_forecast = np.array([res['forecast'][j] for res in results])
    denom = abs(all_actual).sum()
    wmape_val = wmape(all_actual, all_forecast)
    wmape_by_time.append(wmape_val)


xvals = []
yvals = []
for idx, val in enumerate(wmape_by_time):
    if not np.isnan(val):
        xvals.append(idx)
        yvals.append(val)
plt.figure(figsize=(10, 5))
# plt.plot(list(range(n_steps, n_steps+H)), wmape_by_time, marker='o')
plt.plot(xvals, yvals, marker='o')

plt.xlabel("Forecast Time Step")
plt.ylabel("WMAPE")
plt.title(f"WMAPE Over Time (LSTM Forecasts) Across {NO_ITERATIONS} Rows")
plt.grid(True)
plt.savefig(
    '/Users/mahfuz/Final_project/Final_repo/Diagrams/LSTM1_WMAPE_over_time.png')
plt.show()
