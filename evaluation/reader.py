

import pandas as pd
import numpy as np

from pathlib import Path
import sys
# reproducibility
np.random.seed(42)

# make sure config.py is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from config import LONG_DATA_CSV, LONG_DATA_TEST_CSV, DIAGRAMS_DIR  # noqa


def _standardize(df: pd.DataFrame,
                 id_cols: list[str],
                 sep: str = "_") -> pd.DataFrame:
    # 0) Ensure unique_id exists and is a string
    if id_cols:
        df['unique_id'] = df[id_cols].astype(str).agg(sep.join, axis=1)
    df['unique_id'] = df['unique_id'].astype(str)          # <-- here

    # 1) Split back into key1–3
    parts = df['unique_id'].str.split(sep, expand=True)
    for i in range(3):
        df[f'key{i+1}'] = parts[i] if i in parts.columns else pd.NA

    # 2) Return uniform schema
    return df[['ds', 'unique_id', 'key1', 'key2', 'key3', 'actual', 'forecast']]


def read_arima_parallel(filepath: str) -> pd.DataFrame:
    """
    Read an ARIMA-parallel forecast file with columns:
      - ds (date string)
      - unique_id (series identifier)
      - actual (true value)
      - forecast (predicted value)

    Returns a DataFrame with:
      - ds as datetime
      - only the four columns [ds, unique_id, actual, forecast]
      - sorted by unique_id then ds
    """
    # Read CSV (assumes header row "ds,unique_id,actual,forecast")
    df = pd.read_csv(filepath)
    # Parse dates
    df['ds'] = pd.to_datetime(df['ds'])

    # Keep only the standard columns and sort
    df = df[['ds', 'unique_id', 'actual', 'forecast']]
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    return _standardize(df, id_cols=[])


def read_lightgbm_v4(filepath: Path) -> pd.DataFrame:
    """
    Read a LightGBM V4 forecast CSV with columns like:
      - L_REP_CTY, L_CP_COUNTRY, CBS_BASIS, period,
        (y_true or y or value), (y_pred or yhat or forecast)
    Returns a standardized DataFrame of:
      [ds, unique_id, key1, key2, key3, actual, forecast]
    """
    df = pd.read_csv(filepath)
    print(f"[read_lightgbm_v4] Raw columns: {df.columns.tolist()}")

    # 1) Parse ds
    df['ds'] = pd.to_datetime(df['period'].astype(str).apply(
        lambda q: pd.Period(q, freq='Q').to_timestamp(how='end')
    ))

    # 2) Build unique_id from these three
    id_cols = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
    df['unique_id'] = df[id_cols].astype(str).agg('_'.join, axis=1)

    # 3) Rename any possible true/pred columns to our standard names
    rename_map = {
        'y_true': 'actual',
        'y':      'actual',
        'value':  'actual',
        'exposure':      'actual',
        'y_pred': 'forecast',
        'yhat':   'forecast',
        'prediction':    'forecast',
        'exposure_pred': 'forecast',
    }
    # Only keep valid mappings
    valid_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_map)
    print(f"[read_lightgbm_v4] After rename: {df.columns.tolist()}")

    # 4) Sanity‐check
    if 'actual' not in df.columns or 'forecast' not in df.columns:
        raise KeyError(
            f"Could not find actual/forecast columns in {filepath}\n"
            f"After rename, columns are: {df.columns.tolist()}"
        )

    # 5) Subset to exactly what we need, then standardize
    df = df[['ds'] + id_cols + ['actual', 'forecast']]
    return _standardize(df, id_cols)


def read_lightgbm_iterative(filepath: str) -> pd.DataFrame:
    """
    Read a LightGBM iterative forecast CSV with columns:
      - L_REP_CTY, L_CP_COUNTRY, CBS_BASIS, period, exposure, exposure_pred
    Returns a DataFrame with standardized columns:
      - ds        (as Timestamp at end-of-quarter)
      - unique_id (concatenation of the three keys)
      - actual    (exposure)
      - forecast  (exposure_pred)
    """
    df = pd.read_csv(filepath)

    # parse quarter strings to end-of-quarter dates
    df['ds'] = df['period'].apply(
        lambda q: pd.Period(q, freq='Q').to_timestamp(how='end')
    )

    # original key columns
    id_cols = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']

    # build a unique_id
    df['unique_id'] = df[id_cols].agg('_'.join, axis=1)

    # rename and select
    df = df.rename(columns={'exposure': 'actual', 'exposure_pred': 'forecast'})
    df = df[['ds'] + id_cols + ['actual', 'forecast']]

    # standardize (this will recombine unique_id, then split back into key1/2/3)
    return _standardize(df, id_cols)


def read_lstm_predictions(filepath: str) -> pd.DataFrame:
    """
    Read an LSTM forecast CSV with columns:
      - row            (integer row index)
      - horizon_idx    (integer horizon index, can be ignored)
      - actual         (true value)
      - forecast       (predicted value)

    Returns a DataFrame with standardized columns:
      - ds        (datetime)
      - unique_id (series identifier)
      - actual    (true value)
      - forecast  (predicted value)
    """
    # load raw preds
    preds = pd.read_csv(filepath)

    # load test‐set metadata (map row -> ds, unique_id)
    test_meta = (
        pd.read_csv(LONG_DATA_TEST_CSV)[['ds', 'unique_id']]
        .reset_index()
        .rename(columns={'index': 'row'})
    )
    test_meta['ds'] = pd.to_datetime(test_meta['ds'])

    # merge to get ds & unique_id alongside actual/forecast
    df = preds.merge(test_meta, on='row', how='left')

    # pick & sort standard columns
    df = (
        df[['ds', 'unique_id', 'actual', 'forecast']]
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
    )
    return _standardize(df, id_cols=[])


def read_naive(filepath: str) -> pd.DataFrame:
    """
    Read a naive forecast CSV with columns:
      - unique_id (series identifier)
      - ds        (date string)
      - actual    (true value)
      - forecast  (predicted value)

    Returns a DataFrame with standardized columns:
      - ds        (as datetime)
      - unique_id
      - actual
      - forecast
    """
    df = pd.read_csv(filepath, parse_dates=['ds'])
    # detect and rename true-value column to 'actual'
    for col in ['actual', 'y_true', 'y', 'value', 'exposure']:
        if col in df.columns:
            df = df.rename(columns={col: 'actual'})
            break
    else:
        raise KeyError(
            f"No true-value column found in {filepath}, cols={list(df.columns)}")
    # detect and rename forecast column to 'forecast'
    for col in ['forecast', 'y_pred', 'yhat', 'prediction', 'exposure_pred']:
        if col in df.columns:
            df = df.rename(columns={col: 'forecast'})
            break
    else:
        raise KeyError(
            f"No forecast column found in {filepath}, cols={list(df.columns)}")
    # subset to standard
    df = df[['ds', 'unique_id', 'actual', 'forecast']]
    # ensure ds is datetime
    df['ds'] = pd.to_datetime(df['ds'])
    df = (
        df
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
    )
    return _standardize(df, id_cols=[])


def read_seasonal_naive(filepath: str) -> pd.DataFrame:
    """
    Read a seasonal naive forecast CSV with columns:
      - index
      - unique_id
      - ds           (date string)
      - SeasonalNaive   (forecast)
      - SeasonalNaive-lo-90
      - SeasonalNaive-hi-90
      - y            (actual)
    Returns a DataFrame with standardized columns:
      - ds
      - unique_id
      - actual
      - forecast
    """
    df = pd.read_csv(filepath)
    # parse the date column
    df['ds'] = pd.to_datetime(df['ds'])
    # rename for consistency
    df = df.rename(columns={'SeasonalNaive': 'forecast', 'y': 'actual'})
    # select & sort
    df = (
        df[['ds', 'unique_id', 'actual', 'forecast']]
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
    )
    return _standardize(df, id_cols=[])


def read_seasonal_wa(filepath: str) -> pd.DataFrame:
    """
    Read a seasonal window average forecast CSV with columns:
      - index
      - unique_id
      - ds           (date string)
      - SeasWA       (forecast)
      - SeasWA-lo-90 (lower 90% bound, unused)
      - SeasWA-hi-90 (upper 90% bound, unused)
      - y            (actual)
    Returns a DataFrame with standardized columns:
      - ds
      - unique_id
      - actual
      - forecast
    """
    df = pd.read_csv(filepath)
    # parse the date column
    df['ds'] = pd.to_datetime(df['ds'])
    # rename to match our standard
    df = df.rename(columns={'SeasWA': 'forecast', 'y': 'actual'})
    # select, sort, and reset index
    df = (
        df[['ds', 'unique_id', 'actual', 'forecast']]
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
    )
    return _standardize(df, id_cols=[])


READERS = {
    "arima":                 read_arima_parallel,
    "lightgbm":              read_lightgbm_v4,
    "lightgbm_iterative":    read_lightgbm_iterative,
    "lstm":                  read_lstm_predictions,
    "naive":                 read_naive,
    "seasonal_naive":        read_seasonal_naive,
    "seasonal_wa":           read_seasonal_wa,

}
