# tests/test_feature_engineering.py
import pytest
import numpy as np
import pandas as pd
from Data_preparation.feature_engineering import (
    melt_panel, prepare_long, add_features, ID_COLS, LAGS, ROLL_WIN
)
import sys
from pathlib import Path

# 1) Ensure repo root is on the import path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 2) Now import your feature‐engineering module


def synthetic_panel():
    return pd.DataFrame({
        'L_REP_CTY':    ['A', 'A', 'B', 'B'],
        'L_CP_COUNTRY': ['X', 'X', 'Y', 'Y'],
        'CBS_BASIS':    ['F', 'F', 'F', 'F'],
        '2005-Q1':      [1, 2, 3, 4],
        '2005-Q2':      [2, 3, 4, 5],
        '2005-Q3':      [3, 4, 5, 6],
        '2005-Q4':      [4, 5, 6, 7],
    })


def test_melt_and_prepare(tmp_path):
    df = synthetic_panel()
    long = melt_panel(df)
    assert len(long) == 16
    assert 'period' in long.columns

    f = tmp_path/'tmp.csv'
    df.to_csv(f, index=False)
    pl = prepare_long(str(f))
    assert 'y' in pl.columns
    assert (pl['y'] >= 0).all()


def test_add_features_lags_and_rolls():
    df = melt_panel(synthetic_panel())
    df['exposure'] = df['exposure'].astype(float)
    df['y'] = np.log1p(df['exposure'])

    add_features(df)

    for lag in LAGS:
        assert f'lag_{lag}' in df.columns

    for col in [
        'roll_mean_8', 'roll_std_8', 'roll_skew_8', 'roll_kurt_8',
        'pct_gap_1', 'prop_gap_1', 'zero_flag', 'time_since_nonzero',
        'ctrpty_quarter_season', 'ctrpty_quarter_anom', 'q_sin', 'q_cos'
    ]:
        assert col in df.columns

    assert np.isnan(df.loc[0, 'lag_1'])
    assert df.loc[1, 'lag_1'] == df.loc[0, 'exposure']
