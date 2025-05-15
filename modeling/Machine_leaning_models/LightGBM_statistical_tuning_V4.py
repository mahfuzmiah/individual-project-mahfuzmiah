

import os
import sys
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from pathlib import Path
import time
# ───── CONFIG ────────────────────────────────────────────────────────────────
os.environ['PYTHONHASHSEED'] = '42'
os.environ['OMP_NUM_THREADS'] = '1'
random.seed(42)
np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import IMPUTED_RESULTS_DIR_TRAIN, IMPUTED_RESULTS_DIR_TEST, DIAGRAMS_DIR  # nopep8
#
TRAIN_PATH = IMPUTED_RESULTS_DIR_TRAIN / "knn.csv"
TEST_PATH = IMPUTED_RESULTS_DIR_TEST / "knn.csv"
OUT_DIR = DIAGRAMS_DIR / "LightGBM_Results_diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
LAGS = [1, 2, 3, 4, 8]
ROLL_WIN = 8

TRAIN_END = pd.Period("2018Q4", freq='Q')
VAL_END = pd.Period("2019Q4", freq='Q')
TEST_START = pd.Period("2020Q1", freq='Q')
TEST_END = pd.Period("2024Q3", freq='Q')
# ───────────────────────────────────────────────────────────────────────────────


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) /
                   (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def melt_panel(df):
    quarter_cols = [c for c in df.columns if c.endswith(
        tuple(f"-Q{i}" for i in range(1, 5)))]
    long = (df.set_index(ID_COLS)[quarter_cols]
              .stack()
              .reset_index()
              .rename(columns={'level_3': 'quarter', 0: 'exposure'}))
    long['period'] = pd.PeriodIndex(
        long['quarter'].str.replace('-Q', 'Q'), freq='Q')
    return long


def prepare_long(fp, n_rows=None):
    df = pd.read_csv(fp)
    if n_rows is not None:
        df = df.iloc[:n_rows]
    long = melt_panel(df)
    cap = long['exposure'].quantile(0.99)
    long['exposure'] = long['exposure'].clip(upper=cap)
    long['exposure'] = long['exposure'].replace(
        [np.inf, -np.inf], np.nan).clip(lower=0)
    long.dropna(subset=['exposure'], inplace=True)
    long['y'] = np.log1p(long['exposure'])
    return long


def add_static_stats(train_long, test_long):
    stats = (train_long
             .groupby(ID_COLS)['exposure']
             .agg(exp_mean='mean', exp_std='std', exp_min='min', exp_max='max')
             .reset_index())
    return train_long.merge(stats, on=ID_COLS), test_long.merge(stats, on=ID_COLS)


def add_features(long):
    long.sort_values(ID_COLS + ['period'], inplace=True)
    grp = long.groupby(ID_COLS)['exposure']
    for lag in LAGS:
        long[f'lag_{lag}'] = grp.shift(lag)
    long['roll_mean_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).mean())
    long['roll_std_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).std())
    long['roll_skew_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=ROLL_WIN//2).skew())
    long['roll_kurt_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=ROLL_WIN//2).kurt())
    long['pct_gap_1'] = grp.pct_change(1)
    long['prop_gap_1'] = (long['exposure'] - long['lag_1']) / long['lag_1']
    long['zero_flag'] = (long['exposure'] == 0).astype(int)
    long['time_since_nonzero'] = grp.transform(lambda x: x.ne(0).cumsum())
    long['ctrpty_quarter_season'] = long.groupby(
        ['L_CP_COUNTRY', long['period'].dt.quarter])['exposure'].transform('mean')
    long['ctrpty_quarter_anom'] = long['exposure'] - \
        long['ctrpty_quarter_season']
    long['qnum'] = long['period'].dt.quarter
    long['q_sin'] = np.sin(2*np.pi*(long.qnum-1)/4)
    long['q_cos'] = np.cos(2*np.pi*(long.qnum-1)/4)


def main():
    # Load & prep
    train_long = prepare_long(TRAIN_PATH, 200)
    test_long = prepare_long(TEST_PATH, 200)
    train_long, test_long = add_static_stats(train_long, test_long)

    # Feature engineering
    for df in (train_long, test_long):
        add_features(df)

    feat_cols = [c for c in train_long.columns if c.startswith(
                 ('lag_', 'roll_', 'pct_', 'prop_gap', 'time_since_nonzero', 'ctrpty_', 'q_sin', 'q_cos'))] \
        + ['exp_mean', 'exp_std', 'exp_min', 'exp_max']

    # Clean train
    train_long.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_long.dropna(subset=feat_cols + ['y'], inplace=True)

    # Clean test
    test_long.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_long.dropna(subset=['y'], inplace=True)
    test_long[feat_cols] = test_long[feat_cols].fillna(
        method='ffill').fillna(0)

    # Prepare PredefinedSplit for 2010–2019 expanding window
    tv = train_long[train_long['period'] <= VAL_END].copy()
    X_tv = tv[feat_cols]
    y_tv = tv['y']
    years = tv['period'].dt.year
    test_fold = years.apply(lambda y: y - 2010 if 2010 <=
                            y <= 2019 else -1).to_numpy()
    ps = PredefinedSplit(test_fold=test_fold)

    # Randomized search
    param_dist = {
        'learning_rate':    [0.01, 0.03, 0.05, 0.1],
        'num_leaves':       [31, 63, 127],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample':        [0.6, 0.8, 1.0],
        'subsample_freq':   [5],
        'reg_alpha':        [0, 1],
        'reg_lambda':       [0, 1]
    }
    base = lgb.LGBMRegressor(objective='regression', metric='rmse',
                             n_estimators=500, random_state=42)
    rand = RandomizedSearchCV(base, param_distributions=param_dist,
                              n_iter=50, cv=ps,
                              scoring='neg_root_mean_squared_error',
                              n_jobs=-1, verbose=1, random_state=42)
    tune_start = time.time()
    rand.fit(X_tv, y_tv)
    tune_time = time.time() - tune_start

    # Refit best on 2005–2018Q4
    best = rand.best_estimator_
    tr = train_long[train_long['period'] <= TRAIN_END]
    t0 = time.time()
    best.fit(tr[feat_cols], tr['y'])
    # Test eval
    tw = test_long[(test_long['period'] >= TEST_START) &
                   (test_long['period'] <= TEST_END)].copy()
    t0 = time.time()
    tw['y_pred'] = np.expm1(best.predict(tw[feat_cols]))
    tw['y_true'] = np.expm1(tw['y'])
    total_time = time.time() - t0
    n_points = len(tw)   # number of test‐rows you predicted

    # Quarterly metrics
    metrics_q = (tw.groupby('period').apply(lambda g: pd.Series({
        'WMAPE': wmape(g['y_true'], g['y_pred']),
        'SMAPE': smape(g['y_true'], g['y_pred']),
        'RMSE': rmse(g['y_true'], g['y_pred']),
    })).reset_index())

    print(metrics_q)
    PRED_DIR = REPO_ROOT / "predictions" / "lightgbm"
    RUNTIME_DIR = PRED_DIR / "runtimes"
    PRED_DIR.mkdir(exist_ok=True, parents=True)
    RUNTIME_DIR.mkdir(exist_ok=True, parents=True)

    # select the identifiers you want plus period, actual, pred
    preds_df = tw.reset_index()[[
        'L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS', 'period', 'y_true', 'y_pred'
    ]]
    preds_out = PRED_DIR / "lightgbm_V4_test_predictions.csv"
    preds_df.to_csv(preds_out, index=False)
    print(f"Wrote LightGBM predictions to {preds_out}")

    # ------------------------------------------------------------------------
    # 7) Save timing summary
    # ------------------------------------------------------------------------
    # 2) dump timing summary

    timing_rows = [
        {
            'model':            'lightgbm',
            'stage':            'hyperparameter_tuning',
            'total_time_s':     round(tune_time, 2),
            'avg_time_per_row_s': np.nan,
            'n_candidates':     len(rand.cv_results_['params'])
        },
        {
            'model':            'lightgbm',
            'stage':            'train_and_predict',
            'total_time_s':     round(total_time, 2),
            'avg_time_per_row_s': round(total_time / n_points, 4),
            'n_candidates':     np.nan
        }
    ]

    timing_df = pd.DataFrame(timing_rows)
    timing_out = RUNTIME_DIR/"lightgbm_timing_summary.csv"
    timing_df.to_csv(timing_out, index=False)
    print("Wrote timing summary to", timing_out)
    print(
        f"Overall nonzero RMSE: {rmse(tw['y_true'][tw['y_true'] > 0], tw['y_pred'][tw['y_true'] > 0]):.1f}")
    print(f"Overall SMAPE: {smape(tw['y_true'], tw['y_pred']):.2f}%")

    # Save model
    Path("models").mkdir(exist_ok=True)
    best.booster_.save_model("models/lightgbm_model.txt")


if __name__ == '__main__':
    main()
