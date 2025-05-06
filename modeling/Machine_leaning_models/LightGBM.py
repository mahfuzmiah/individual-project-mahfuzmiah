#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
    median_absolute_error,
)
from pathlib import Path

# ───── CONFIG ────────────────────────────────────────────────────────────────
TRAIN_FILE = "imputed_results/train/knn.csv"   # 2005–2019 panel, already imputed
TEST_FILE = "imputed_results/test/knn.csv"    # 2020–2024 panel, already imputed
ID_COLS = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
LAGS = [1, 2, 3, 4]
ROLL_WIN = 4

# tuning split entirely within 2005–2019:
TRAIN_END = pd.Period("2018Q4", freq='Q')
VAL_START = pd.Period("2019Q1", freq='Q')
VAL_END = pd.Period("2019Q4", freq='Q')
TEST_START = pd.Period("2020Q1", freq='Q')
# ───────────────────────────────────────────────────────────────────────────────


def melt_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Wide panel → long with a PeriodIndex."""
    quarters = [
        c for c in df.columns
        if c.endswith(tuple(f"-Q{i}" for i in range(1, 5)))
    ]
    long = (
        df
        .set_index(ID_COLS)[quarters]
        .stack()
        .reset_index()
        .rename(columns={'level_3': 'quarter', 0: 'exposure'})
    )
    long['period'] = pd.PeriodIndex(
        long['quarter'].str.replace('-Q', 'Q', regex=False),
        freq='Q'
    )
    return long


def assert_no_empty_rows(df: pd.DataFrame, name: str):
    """Ensure no row is ALL-NaN across the quarter columns."""
    quarters = [
        c for c in df.columns
        if c.endswith(tuple(f"-Q{i}" for i in range(1, 5)))
    ]
    n_empty = df[quarters].isna().all(axis=1).sum()
    if n_empty > 0:
        raise RuntimeError(f"{n_empty} rows in {name!r} have ALL-NaN quarters")


def assert_finite(X: pd.DataFrame, name: str):
    """Ensure no NaNs or infs in X."""
    if not np.isfinite(X.values).all():
        raise RuntimeError(f"{name} contains NaN or infinite values")


def prepare_long(fp: str, label_col='y') -> pd.DataFrame:
    """
    1) sanity-check raw panel (no all-NaN).
    2) melt to long.
    3) clip infinities → NaN → drop.
    4) build log1p(target) column.
    """
    df = pd.read_csv(fp)
    assert_no_empty_rows(df, fp)
    long = melt_panel(df)
    long['exposure'] = (
        long['exposure']
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=0)
    )
    long.dropna(subset=['exposure'], inplace=True)
    long[label_col] = np.log1p(long['exposure'])
    return long


def add_features(long: pd.DataFrame):
    """Add lags, rolling‐mean/std, %‐changes, seasonal & cross‐sectional signals, cyclical quarter."""
    long.sort_values(ID_COLS + ['period'], inplace=True)
    grp = long.groupby(ID_COLS)['exposure']

    # 1) Lags + rolling‐mean
    for lag in LAGS:
        long[f'lag_{lag}'] = grp.shift(lag)
    long['roll_mean_4'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).mean()
    )

    # 2) Rolling volatility (std over past 4 quarters)
    long['roll_std_4'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).std()
    )

    # 3) % changes: quarter-on-quarter and year-on-year
    long['pct_change_1'] = grp.pct_change(1)
    long['pct_change_4'] = grp.pct_change(4)

    # 4) Seasonal anomaly: deviation from quarter‐of‐year mean
    quarter_avg = long.groupby('quarter')['exposure'].transform('mean')
    long['seasonal_anom'] = long['exposure'] - quarter_avg

    # 5) Cross‐sectional average & ratio
    ctrpty_avg = long.groupby(['L_CP_COUNTRY', 'period'])['exposure'] \
                     .transform('mean')
    long['ctrpty_avg'] = ctrpty_avg
    long['ratio_to_ctrpty_avg'] = long['exposure'] / ctrpty_avg

    # 6) Cyclical quarter‐of‐year
    long['qnum'] = long['period'].dt.quarter
    long['q_sin'] = np.sin(2*np.pi*(long.qnum - 1)/4)
    long['q_cos'] = np.cos(2*np.pi*(long.qnum - 1)/4)


def add_more_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """Additional signals: slopes, missing‐counts, deviation from period mean."""
    grp = df_long.groupby(ID_COLS)['exposure']

    # 1) Quarter‐over‐quarter % change
    df_long['qoq_pct'] = grp.pct_change(1)
    # 2) Year‐over‐year % change
    df_long['yoy_pct'] = grp.pct_change(4)
    # 3) Rolling volatility (std)
    df_long['roll_std_4'] = grp.transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).std()
    )

    # 4) Linear trend slope over last window
    def rolling_slope(x):
        if len(x) < ROLL_WIN:
            return np.nan
        idx = np.arange(len(x))
        y = x.to_numpy()[-ROLL_WIN:]
        m, _ = np.polyfit(idx[-ROLL_WIN:], y, 1)
        return m
    df_long['roll_slope_4'] = grp.transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=ROLL_WIN)
        .apply(rolling_slope, raw=False)
    )

    # 5) Count of missing before imputation (should be zero now but for safety)
    df_long['miss_count_4'] = grp.transform(
        lambda x: x.isna().rolling(ROLL_WIN, min_periods=1).sum()
    )

    # 6) Deviation from overall period mean
    period_mean = df_long.groupby('period')['exposure'].transform('mean')
    df_long['dev_from_period_mean'] = df_long['exposure'] - period_mean

    return df_long


def main():
    # 1) Load & melt
    train_long = prepare_long(TRAIN_FILE, label_col='y')
    test_long = prepare_long(TEST_FILE,  label_col='y')

    # 2) Feature engineering
    for df in (train_long, test_long):
        add_features(df)
        add_more_features(df)

    # 3) Build feature list
    feat_cols = [
        *[f'lag_{l}' for l in LAGS],
        'roll_mean_4', 'roll_std_4', 'pct_change_1', 'pct_change_4',
        'seasonal_anom', 'ctrpty_avg', 'ratio_to_ctrpty_avg',
        'q_sin', 'q_cos',
        'qoq_pct', 'yoy_pct', 'roll_slope_4',
        'miss_count_4', 'dev_from_period_mean'
    ]

    # 4) One‐shot clean
    def clean_inf_and_nan(df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=feat_cols + ['y'], inplace=True)

    clean_inf_and_nan(train_long)
    clean_inf_and_nan(test_long)
    assert_finite(train_long[feat_cols], "X_train")
    assert_finite(test_long[feat_cols],  "X_test")

    # 5) Split into train/val/test
    train_set = train_long[train_long.period <= TRAIN_END]
    val_set = train_long[(train_long.period >= VAL_START)
                         & (train_long.period <= VAL_END)]
    test_set = test_long[test_long.period >= TEST_START]

    X_tr,   y_tr_log = train_set[feat_cols], train_set['y']
    X_val,  y_val_log = val_set[feat_cols], val_set['y']
    X_te,   y_te_log = test_set[feat_cols], test_set['y']

    # 6) Train LightGBM on log‐scale target
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(
        X_tr, y_tr_log,
        eval_set=[(X_tr, y_tr_log), (X_val, y_val_log)],
        eval_names=['train', 'valid'],
    )

    # ─ 7) Predict & back‐transform
    y_pred_log = model.predict(X_te)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_te_log)

    # ─ 8) Compute metrics
    log_rmse = np.sqrt(mean_squared_error(y_te_log, y_pred_log))
    rmse = np.sqrt(mean_squared_error(y_true,    y_pred))
    mask_pos = y_true > 0
    mape = mean_absolute_percentage_error(
        y_true[mask_pos], y_pred[mask_pos]
    ) * 100

    # baseline: lag‐1 on original scale
    test_set['lag1_exposure'] = (
        test_set.groupby(ID_COLS)['exposure']
                .shift(1)
    )
    mask = test_set['lag1_exposure'].notna()
    baseline_rmse = np.sqrt(mean_squared_error(
        test_set.loc[mask, 'exposure'],
        test_set.loc[mask, 'lag1_exposure']
    ))
    baseline_rmse_log = np.sqrt(mean_squared_error(
        np.log1p(test_set.loc[mask, 'exposure']),
        np.log1p(test_set.loc[mask, 'lag1_exposure'])
    ))

    male = mean_absolute_error(y_te_log, y_pred_log)
    medae = median_absolute_error(y_true,    y_pred)
    cap = np.percentile(y_pred, 99)
    rmse_cap = np.sqrt(mean_squared_error(
        y_true, np.clip(y_pred, 0, cap)
    ))

    # ─ 9) Print summary
    print(f">>> log‐RMSE on test target:       {log_rmse:.4f}")
    print(f">>> Test RMSE (orig‐scale):         {rmse:,.1f}")
    print(f">>> Test MAPE (orig‐scale):         {mape:.2f}%")
    print(f">>> Lag‐1 baseline RMSE (orig):     {baseline_rmse:,.1f}")
    print(f">>> Lag‐1 baseline log‐RMSE:        {baseline_rmse_log:.4f}")
    print(f">>> Mean absolute log error (MALE): {male:.4f}")
    print(f">>> Median absolute error (orig):   {medae:.2f}")
    print(f">>> Winsorized RMSE @99th pct:       {rmse_cap:,.1f}")

    # ─ 10) Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model.booster_.save_model(str(model_dir/"lightgbm_model.txt"))
    print(f"\nModel saved to {model_dir/'lightgbm_model.txt'}")


if __name__ == "__main__":
    main()
