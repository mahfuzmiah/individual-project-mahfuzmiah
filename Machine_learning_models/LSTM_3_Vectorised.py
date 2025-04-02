

import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- Configuration ---
TRAIN_PATH = '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/imputed_linear.csv'
TEST_PATH = '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/TestingData.csv'


# --- Utility Functions ---
scaler = StandardScaler()


def wmape(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / abs(y_true).sum()


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def split_sequence_vectorized(sequence, n_steps):
    """
    Given a 2D array 'sequence' of shape (T, R) where T is the number of time steps
    and R is the number of features (in our case, each column vector from the original data),
    this function creates overlapping windows of length n_steps and the following column as the target.
    """
    X, y = [], []
    T = sequence.shape[0]
    for i in range(T - n_steps):
        seq_x = sequence[i:i+n_steps, :]  # shape: (n_steps, R)
        seq_y = sequence[i+n_steps, :]      # shape: (R,)
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

df = pd.read_csv(TEST_PATH)

# Convert applicable columns to numeric types explicitly
df = df.infer_objects(copy=False)

# Select numerical columns separately
numeric_cols = df.select_dtypes(include=['number']).columns

# Create a copy of the original DataFrame to retain non-numeric data
df_filled = df.copy()

# Interpolate only numerical columns using linear interpolation along rows (axis=1)
df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(
    method='linear',
    axis=1,
    limit_direction='both'
)

# Fill any remaining NaN values with 0
test = df_filled.fillna(0)


# Assume the first 3 columns are metadata; time-series data starts from the 4th column.
# Original shape: (R, T) where each row is a time series.
# shape: (R, min(T, 200))

train_data = train[train.columns[3:]].values

# shape: (R, min(T, 200))
test_data = test[test.columns[3:]].values

# Transpose so that the time axis is vertical:
# New shape: (min(T, 200), R) where each row is a time step and each column is a feature (the entire vector downwards)
train_data_T = train_data.T
test_data_T = test_data.T
train_data_T = scaler.fit_transform(train_data_T)
test_data_T = scaler.transform(test_data_T)

# Define hyperparameters:
n_steps = 4  # number of time steps per input window
# now n_features is the number of rows in the original data
n_features = train_data_T.shape[1]
print(n_features)

# Build training dataset from transposed training data:
X_train, y_train = split_sequence_vectorized(train_data_T, n_steps)
# X_train shape: (samples, n_steps, n_features)
# y_train shape: (samples, n_features)

# Define and compile the LSTM model:
model = Sequential()
model.add(LSTM(100, input_shape=(
    n_steps, X_train.shape[2]), return_sequences=True))
model.add(LSTM(50))
# The final layer should have n_features units because we are predicting n_features values.
# output a vector of size n_features (predicts entire next column)
model.add(Dense(X_train.shape[2]))
model.compile(optimizer='adam', loss='mse')

# Train the model:
model.fit(X_train, y_train, epochs=500, verbose=1)

# For iterative prediction on test data:
T_test = test_data_T.shape[0]   # total number of time steps in test data
n_predictions = T_test - n_steps  # number of predictions to generate


# Use the first n_steps rows of test_data_T as the initial window:
current_input = test_data_T[:n_steps, :]  # shape: (n_steps, n_features)
predictions = []
test_time_series_cols = test.columns[3:]
start_time = time.time()
for i in range(n_predictions):
    x_input = current_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=1)  # shape: (1, n_features)
    predictions.append(yhat[0])
    # Update window: remove the oldest row and append the new predicted row:
    current_input = np.concatenate([current_input[1:], yhat], axis=0)

predictions = np.array(predictions)  # shape: (n_predictions, n_features)

# The full forecast: combine the initial seed with predictions (should match test_data_T shape)
full_forecast = np.concatenate([test_data_T[:n_steps, :], predictions], axis=0)
end_time = time.time()
results = []
# Loop over each original row (each column in full_forecast)
for i in range(n_features):
    # Forecast for row i: skip the initial n_steps seed values
    forecast = full_forecast[n_steps:, i]
    # Actual test series for row i comes from test_data.
    # Assuming test_data has shape (n_rows, T) with T columns,
    # use the same horizon: columns from n_steps onward.
    actual = test_data[i, n_steps:]
    results.append({'row': i, 'forecast': forecast, 'actual': actual})
# Assuming test_time_series_cols has 23 date labels, and n_steps=4:
# This will have 23 - 4 = 19 labels
date_labels = list(test_time_series_cols[n_steps:])


# For Graph 1 (for the first row):
first_actual = test_data[0, n_steps:]  # shape (19,)
first_forecast = predictions[:, 0]          # shape (19,)
second_actual = test_data[1, n_steps:]  # shape (19,)
second_forecast = predictions[:, 1]          # shape (19,)

print(f"First actual values: {first_actual}")
print(f"First forecast values: {first_forecast}")
print(f"Second actual values: {second_actual}")
print(f"Second forecast values: {second_forecast}")
plt.figure(figsize=(10, 5))
plt.plot(date_labels, first_actual, label='Actual Feature 0', marker='o')
plt.plot(date_labels, first_forecast, label='Forecast Feature 0', marker='x')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Forecast vs Actual for Feature 0")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# The full forecast (concatenating the seed with predictions)
full_forecast = np.concatenate([test_data_T[:n_steps, :], predictions], axis=0)
# Inverse transform the full forecast predictions to the original scale
full_forecast_original = scaler.inverse_transform(full_forecast)

# Build results using the inverse-transformed forecasts
results = []
for i in range(n_features):
    # Use inverse-transformed forecast for proper comparison
    forecast = full_forecast_original[n_steps:, i]
    actual = test_data[i, n_steps:]
    results.append({'row': i, 'forecast': forecast, 'actual': actual})

# Calculate WMAPE over forecast time steps
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
plt.title(f"WMAPE Over Time (LSTM Forecasts) Across Rows")
plt.grid(True)
# plt.savefig(
#     '/Users/mahfuz/Final_project/Final_repo/Diagrams/LSTM_parallel_WMAPE_over_time.png')
plt.show()
