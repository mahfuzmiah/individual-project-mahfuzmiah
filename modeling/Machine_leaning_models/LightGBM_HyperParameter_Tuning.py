

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scipy.stats import skew, kurtosis
from pathlib import Path
import time
# ───── CONFIG ────────────────────────────────────────────────────────────────
TRAIN_FILE = "imputed_results/train/knn.csv"
TEST_FILE = "imputed_results/test/knn.csv"
ID_COLS = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
LAGS = [1, 2, 3, 4, 8]
ROLL_WIN = 8

# tuning split entirely within 2005–2019:
TRAIN_END = pd.Period("2018Q4", freq='Q')
VAL_START = pd.Period("2019Q1", freq='Q')
VAL_END = pd.Period("2019Q4", freq='Q')
TEST_START = pd.Period("2020Q1", freq='Q')
TEST_END = pd.Period("2024Q3", freq='Q')
NFOLDS = 5
# ───────────────────────────────────────────────────────────────────────────────


def melt_panel(df):
    quarter_cols = [c for c in df.columns if c.endswith(
        tuple(f"-Q{i}" for i in range(1, 5)))]
    long = df.set_index(ID_COLS)[quarter_cols] \
             .stack() \
             .reset_index() \
             .rename(columns={'level_3': 'quarter', 0: 'exposure'})
    long['period'] = pd.PeriodIndex(
        long['quarter'].str.replace('-Q', 'Q'), freq='Q')
    return long


def prepare_long(fp):
    df = pd.read_csv(fp)
    long = melt_panel(df)
    long['exposure'] = long['exposure'].replace(
        [np.inf, -np.inf], np.nan).clip(lower=0)
    long.dropna(subset=['exposure'], inplace=True)
    long['y'] = np.log1p(long['exposure'])
    return long


def add_features(long: pd.DataFrame):
    long.sort_values(ID_COLS+['period'], inplace=True)
    grp = long.groupby(ID_COLS)['exposure']
    # lags and rolling stats
    for lag in LAGS:
        long[f'lag_{lag}'] = grp.shift(lag)
    long['roll_mean_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).mean())
    long['roll_min_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).min())
    long['roll_max_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).max())
    long['roll_std_8'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).std())
    long['roll_skew_8'] = grp.shift(1).transform(
        lambda s: s.rolling(ROLL_WIN, min_periods=ROLL_WIN//2).skew()
    )
    long['roll_kurt_8'] = grp.shift(1).transform(
        lambda s: s.rolling(ROLL_WIN, min_periods=ROLL_WIN//2).kurt()
    )
    # pct changes
    long['pct_gap_1'] = grp.pct_change(1)
    # proportional gap
    long['prop_gap_1'] = (long['exposure'] - long['lag_1']) / long['lag_1']
    # time since last nonzero
    long['zero_flag'] = (long['exposure'] == 0).astype(int)
    long['time_since_nonzero'] = grp.transform(lambda x: x.ne(0).cumsum())
    # group-level seasonality
    long['ctr_country_quarter_avg'] = long.groupby(['L_CP_COUNTRY', 'period'])[
        'exposure'].transform('mean')
    long['ctrpty_quarter_season'] = long.groupby(
        ['L_CP_COUNTRY', long['period'].dt.quarter])['exposure'].transform('mean')
    long['ctrpty_quarter_anom'] = long['exposure'] - \
        long['ctrpty_quarter_season']
    # cyclical qtr
    long['qnum'] = long['period'].dt.quarter
    long['q_sin'] = np.sin(2*np.pi*(long.qnum-1)/4)
    long['q_cos'] = np.cos(2*np.pi*(long.qnum-1)/4)


def main():
    # 1) Load & prep
    train_long = prepare_long(TRAIN_FILE)
    test_long = prepare_long(TEST_FILE)
    # 2) Feature engineering
    for df in (train_long, test_long):
        add_features(df)
    # 3) Feature list & cleanup
    feat_cols = [c for c in train_long.columns if c.startswith(
        ('lag_', 'roll_', 'pct_', 'prop_gap', 'time_since_nonzero', 'ctrpty_', 'q_sin', 'q_cos'))]
    for df in (train_long, test_long):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=feat_cols+['y'], inplace=True)
    # 4) Rolling K-Fold CV within train_long until VAL_END
    tv = train_long[train_long['period'] <= VAL_END]
    X = tv[feat_cols].values
    y = tv['y'].values
    tscv = TimeSeriesSplit(n_splits=NFOLDS)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        clf = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            bagging_freq=5,             # placeholder, overridden by grid
            random_state=42,
            n_estimators=1000,          # use early_stopping
        )
        t0 = time.time()

        clf.fit(X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])])
        single_fit_time = time.time() - t0
        print("single fit took", single_fit_time, "seconds")
        estimated_total_time = single_fit_time * (NFOLDS * 72)/4
        print(
            f"Estimated total time for {NFOLDS} folds: {estimated_total_time:.1f} seconds")

        yv = np.expm1(clf.predict(X[val_idx]))
        yt = np.expm1(y[val_idx])
        cv_scores.append(np.sqrt(mean_squared_error(yt, yv)))

    print(f"CV RMSEs: {cv_scores}, mean={np.mean(cv_scores):,.1f}")

    # ── 5) Hyperparameter grid search on train+val up to VAL_END ─────────
    tv = train_long[train_long['period'] <= VAL_END]
    X_tv, y_tv = tv[feat_cols], tv['y']

    # a) define a small grid
    param_grid = {
        # tree complexity vs. learning‐rate tradeoff
        # conservative → aggressive steps
        'learning_rate':    [0.01, 0.05, 0.1],
        # small → medium → large tree size
        'num_leaves':       [31, 63, 127],

        # sampling / regularization
        # fraction of features per tree
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample':        [0.6, 0.8, 1.0],      # fraction of rows per tree
        # perform row‐sampling every 5 iterations
        'subsample_freq':   [5],

        # L1/L2 penalties
        'reg_alpha':        [0, 1],               # L1 regularization
        'reg_lambda':       [0, 1],               # L2 regularization
    }

    tscv = TimeSeriesSplit(n_splits=NFOLDS)
    base = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=500,
        bagging_freq=5,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_root_mean_squared_error',
        verbose=2,
        n_jobs=-1,
    )
    t0 = time.time()

    grid.fit(X_tr, y_tr,
             eval_set=[(X_val, y_val)],
             early_stopping_rounds=50,
             eval_metric='rmse',
             verbose=False)
    print(f"\nGrid search took {time.time() - t0:.1f}s")
    print("Best params:", grid.best_params_)
    print("Best CV RMSE:", -grid.best_score_)

    print("🔍 Best params:", grid.best_params_)
    print("🔍 Best CV RMSE:", -grid.best_score_)

    # c) take the best estimator and re‐fit on full train_end
    best = grid.best_estimator_
    tr = train_long[train_long['period'] <= TRAIN_END]
    X_tr, y_tr = tr[feat_cols], tr['y']
    best.fit(X_tr, y_tr)

    # use `best` instead of `model` from now on
    model = best
    # 6) Evaluate on TEST_START→TEST_END
    test_window = test_long[(test_long['period'] >= TEST_START) & (
        test_long['period'] <= TEST_END)]
    X_te, y_te = test_window[feat_cols], test_window['y']

    y_pred = np.expm1(model.predict(X_te))
    y_true = np.expm1(y_te)
    # --- A) Metrics *excluding* zeros (mask them out) ---
    mask_nz = y_true > 0
    rmse_nz = np.sqrt(mean_squared_error(y_true[mask_nz],  y_pred[mask_nz]))
    mape_nz = mean_absolute_percentage_error(
        y_true[mask_nz], y_pred[mask_nz]) * 100

    print(f"→ RMSE (non-zero only): {rmse_nz:,.1f}")
    print(f"→ MAPE (non-zero only): {mape_nz:.2f}%")

    # --- B) Metrics *including* zeros ---
    # 1) plain RMSE
    rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))

    # 2) symmetric MAPE (SMAPE) to handle zeros gracefully
    #    SMAPE = 100% * mean( |F−A| / ((|A|+|F|)/2) )
    smape = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    )

    print(f"→ RMSE (all data):      {rmse_all:,.1f}")
    print(f"→ SMAPE (all data):     {smape:.2f}%")
    log_rmse = np.sqrt(mean_squared_error(y_te, np.log1p(y_pred)))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(
        y_true[y_true > 0], y_pred[y_true > 0])*100
    print(f">>> log-RMSE: {log_rmse:.4f}")
    print(f">>>    RMSE: {rmse:,.1f}")
    print(f">>>    MAPE: {mape:.2f}%")
    # 7) Save
    Path("models").mkdir(exist_ok=True)
    model.booster_.save_model("models/lightgbm_model.txt")


if __name__ == '__main__':
    main()
