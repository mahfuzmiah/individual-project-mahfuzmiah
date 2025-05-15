

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def log_rmse(a, f, eps=1e-6):
    a_pos = np.maximum(a, eps)
    f_pos = np.maximum(f, eps)
    return np.sqrt(np.mean((np.log(a_pos) - np.log(f_pos))**2))


def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) /
                   (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df to have columns ['model','horizon','actual','forecast'].
    Returns a DataFrame with columns ['model','horizon','RMSE','Log-RMSE','SMAPE','WMAPE'].
    """
    records = []
    # groupby will only iterate over existing (model, horizon) pairs
    for (model, horizon), grp in df.groupby(['model', 'horizon'], sort=False):
        # sanity check—grp should never be empty, but just in case:
        if grp.shape[0] == 0:
            continue
        a = grp['actual'].to_numpy()
        f = grp['forecast'].to_numpy()
        records.append({
            'model':    model,
            'horizon':  horizon,
            'RMSE':     rmse(a, f),
            'Log-RMSE': log_rmse(a, f),
            'SMAPE':    smape(a, f),
            'WMAPE':    wmape(a, f),
        })
    return pd.DataFrame.from_records(records)
