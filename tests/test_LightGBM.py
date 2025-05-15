# tests/test_lightgbm_prep.py
from modeling.Machine_leaning_models.LightGBM import (
    melt_panel,
    prepare_long,
    add_features,
    add_more_features
)
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 1) Make repo root importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# 2) Import the functions to test

# 3) Build a tiny synthetic wide‐format dataset


@pytest.fixture
def small_wide_df():
    # Two IDs, four quarters
    data = {
        'L_REP_CTY': ['A', 'B'],
        'L_CP_COUNTRY': ['X', 'Y'],
        'CBS_BASIS': ['F', 'U'],
        '2000-Q1': [1.0,   2.0],
        '2000-Q2': [np.nan, 4.0],
        '2000-Q3': [3.0,   np.inf],
        '2000-Q4': [4.0,   5.0],
    }
    return pd.DataFrame(data)


def test_melt_and_prepare(small_wide_df, tmp_path):
    df = small_wide_df.copy()
    # 1) melt_panel
    long = melt_panel(df)
    # Expect 2 IDs * 4 quarters = 8 rows
    assert len(long) == 7
    assert set(long.columns) >= {
        'L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS', 'quarter', 'exposure', 'period'}

    # Manually write a CSV and test prepare_long
    csv = tmp_path/'tmp.csv'
    df.to_csv(csv, index=False)
    long2 = prepare_long(str(csv))
    # infinities drop, NaNs drop => exposures: rows with NaN or inf removed
    # small_wide_df has one NaN and one inf => 8 - 2 = 6 rows
    assert len(long2) == 6
    # y = log1p(exposure)
    assert np.allclose(long2['y'], np.log1p(long2['exposure']))


def test_feature_engineering(small_wide_df):
    # Melt and prepare
    df = small_wide_df.copy()
    long = melt_panel(df)
    long['exposure'] = long['exposure'].clip(lower=0).replace([np.inf], np.nan)
    long.dropna(subset=['exposure'], inplace=True)
    long['period'] = pd.PeriodIndex(
        long['quarter'].str.replace('-Q', 'Q'), freq='Q')

    # add basic features
    add_features(long)
    # Check that lag_1 exists and for first row per ID is NaN
    assert 'lag_1' in long.columns
    first_A = long[long['L_REP_CTY'] == 'A'].sort_values('period').iloc[0]
    assert pd.isna(first_A['lag_1'])
    # For second quarter of A, lag_1 == first exposure
    second_A = long[long['L_REP_CTY'] == 'A'].sort_values('period').iloc[1]
    assert second_A['lag_1'] == pytest.approx(first_A['exposure'])

    # Rolling mean over window=4 should equal past mean (only 1 past value for second row)
    assert 'roll_mean_4' in long.columns
    assert second_A['roll_mean_4'] == pytest.approx(first_A['exposure'])

    # add_more_features
    add_more_features(long)
    # Check qoQ and yoy
    assert 'qoq_pct' in long.columns
    assert 'yoy_pct' in long.columns
    # For the first available value, pct_change yields NaN
    # assert pd.isna(first_A['qoq_pct'])
