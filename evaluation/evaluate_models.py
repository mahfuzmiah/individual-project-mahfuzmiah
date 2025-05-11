
# evaluation/evaluate_models.py

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
# (import your wmape, smape funcs here)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) /
                   (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100

if __name__ == "__main__":
    # Load your test-frame with y_true and y_pred columns
    df = pd.read_csv("imputed_results/metrics_test.csv")
    metrics = {
        "RMSE":  rmse(df.y_true, df.y_pred),
        "WMAPE": wmape(df.y_true, df.y_pred),
        "SMAPE": smape(df.y_true, df.y_pred),
    }
    print("Overall test metrics:", metrics)
