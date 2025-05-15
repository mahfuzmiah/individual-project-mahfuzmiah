# Data_preparation/feature_engineering.py

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────
ID_COLS = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
LAGS = [1, 2, 3, 4, 8]
ROLL_WIN = 8
# ────────────────────────────────────────────────────────────────────────────────


def melt_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn wide quarter columns into a long panel:
    e.g. columns 2005-Q1,2005-Q2,... → rows with 'quarter' and 'exposure'
    and a PeriodIndex column 'period'.
    """
    quarter_cols = [c for c in df.columns if c.endswith(
        tuple(f"-Q{i}" for i in range(1, 5)))]
    long = (
        df.set_index(ID_COLS)[quarter_cols]
          .stack()
          .reset_index()
          .rename(columns={'level_3': 'quarter', 0: 'exposure'})
    )
    long['period'] = pd.PeriodIndex(
        long['quarter'].str.replace('-Q', 'Q'), freq='Q')
    return long


def prepare_long(fp: str) -> pd.DataFrame:
    """
    Read CSV at fp, melt it, winsorize & clip outliers/infs,
    drop missing exposures, and log1p‐transform into 'y'.
    """
    df = pd.read_csv(fp)
    long = melt_panel(df)
    # winsorize at 99th percentile
    cap = long['exposure'].quantile(0.99)
    long['exposure'] = long['exposure'].clip(upper=cap)
    # clip infinities → NaN → drop, then log1p
    long['exposure'] = (
        long['exposure']
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=0)
    )
    long.dropna(subset=['exposure'], inplace=True)
    long['y'] = np.log1p(long['exposure'])
    return long


def add_features(long: pd.DataFrame):
    """
    Given a long‐format panel with 'exposure' and 'period', add:
      - lag_1, lag_2, ..., lag_8
      - rolling mean/std/skew/kurt over the prior 8 quarters
      - pct_gap_1, prop_gap_1
      - zero_flag, time_since_nonzero
      - counterparty‐quarter seasonality & anomaly
      - cyclical quarter features q_sin, q_cos
    """
    long.sort_values(ID_COLS + ['period'], inplace=True)
    grp = long.groupby(ID_COLS)['exposure']

    # Lags
    for lag in LAGS:
        long[f'lag_{lag}'] = grp.shift(lag)

    # Rolling stats
    long['roll_mean_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).mean()
    )
    long['roll_std_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).std()
    )
    long['roll_skew_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=ROLL_WIN//2).skew()
    )
    long['roll_kurt_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=ROLL_WIN//2).kurt()
    )

    # Percent‐change and proportional gap
    long['pct_gap_1'] = grp.pct_change(1)
    long['prop_gap_1'] = (long['exposure'] - long['lag_1']) / long['lag_1']

    # Zero‐flag and time since last non‐zero
    long['zero_flag'] = (long['exposure'] == 0).astype(int)
    long['time_since_nonzero'] = grp.transform(lambda x: x.ne(0).cumsum())

    # Counterparty‐quarter seasonality
    long['ctrpty_quarter_season'] = long.groupby(
        ['L_CP_COUNTRY', long['period'].dt.quarter]
    )['exposure'].transform('mean')
    long['ctrpty_quarter_anom'] = long['exposure'] - \
        long['ctrpty_quarter_season']

    # Cyclical quarter encoding
    long['qnum'] = long['period'].dt.quarter
    long['q_sin'] = np.sin(2 * np.pi * (long.qnum - 1) / 4)
    long['q_cos'] = np.cos(2 * np.pi * (long.qnum - 1) / 4)
    long.reset_index(drop=True, inplace=True)
