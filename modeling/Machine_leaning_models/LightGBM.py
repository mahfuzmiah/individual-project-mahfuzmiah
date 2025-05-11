

import sys
import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ───────────────────────────────────────────────
# 1) determinism
os.environ['PYTHONHASHSEED'] = '42'
os.environ['OMP_NUM_THREADS'] = '1'
random.seed(42)
np.random.seed(42)

# ───────────────────────────────────────────────
# 2) paths & periods
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import IMPUTED_RESULTS_DIR_TRAIN, IMPUTED_RESULTS_DIR_TEST, DIAGRAMS_DIR  # nopep8

TRAIN_FP = IMPUTED_RESULTS_DIR_TRAIN / "knn.csv"
TEST_FP = IMPUTED_RESULTS_DIR_TEST / "knn.csv"
OUT_DIR = DIAGRAMS_DIR / "LightGBM_Results_diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Period("2018Q4", freq="Q")
FORECAST_START = pd.Period("2020Q1", freq="Q")
FORECAST_END = pd.Period("2024Q3", freq="Q")

ID_COLS = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
LAGS = [1, 2, 3, 4]
ROLL = 4

# ───────────────────────────────────────────────
# 3) data → long format


def melt_panel(df):
    quarters = [c for c in df.columns if c.endswith(
        tuple(f"-Q{i}" for i in (1, 2, 3, 4)))]
    long = (df.set_index(ID_COLS)[quarters]
            .stack().reset_index()
            .rename(columns={'level_3': 'quarter', 0: 'exposure'}))
    long['period'] = pd.PeriodIndex(
        long['quarter'].str.replace("-Q", "Q"), freq="Q")
    return long


def prepare_long(fp, n_rows=None):
    df = pd.read_csv(fp)
    if n_rows is not None:
        df = df.iloc[:n_rows]
    long = melt_panel(df)
    long['exposure'] = (long['exposure']
                        .clip(lower=0)
                        .replace([np.inf, -np.inf], np.nan))
    long.dropna(subset=['exposure'], inplace=True)
    long['y'] = np.log1p(long['exposure'])
    return long

# ───────────────────────────────────────────────
# 4) feature‐engineering


def add_features(df):
    df.sort_values(ID_COLS+['period'], inplace=True)
    grp = df.groupby(ID_COLS)['exposure']
    for lag in LAGS:
        df[f'lag_{lag}'] = grp.shift(lag)
    df['roll_mean_4'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL, 1).mean())
    df['roll_std_4'] = grp.shift(1).transform(
        lambda x: x.rolling(ROLL, 1).std())
    df['pct_change_1'] = grp.pct_change(1)
    df['pct_change_4'] = grp.pct_change(4)
    df['seasonal_anom'] = df['exposure'] - \
        df.groupby('quarter')['exposure'].transform('mean')
    ctrp = df.groupby(['L_CP_COUNTRY', 'period'])['exposure'].transform('mean')
    df['ctrpty_avg'] = ctrp
    df['ratio_to_ctrpty_avg'] = df['exposure']/ctrp
    df['qnum'] = df['period'].dt.quarter
    df['q_sin'] = np.sin(2*np.pi*(df.qnum-1)/4)
    df['q_cos'] = np.cos(2*np.pi*(df.qnum-1)/4)


def add_more_features(df):
    grp = df.groupby(ID_COLS)['exposure']
    df['qoq_pct'] = grp.pct_change(1)
    df['yoy_pct'] = grp.pct_change(4)
    df['roll_slope_4'] = grp.transform(
        lambda x: x.rolling(ROLL, ROLL).apply(
            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=False))
    df['miss_count_4'] = grp.transform(
        lambda x: x.isna().rolling(ROLL, 1).sum())
    df['dev_from_period_mean'] = df['exposure'] - \
        df.groupby('period')['exposure'].transform('mean')


# ───────────────────────────────────────────────
# 5) load + engineer both train & test up to 2019Q4
train_long = prepare_long(TRAIN_FP, 50)
test_long = prepare_long(TEST_FP, 50)

for df in (train_long, test_long):
    add_features(df)
    add_more_features(df)

feat_cols = [f'lag_{l}' for l in LAGS] + [
    'roll_mean_4', 'roll_std_4', 'pct_change_1', 'pct_change_4',
    'seasonal_anom', 'ctrpty_avg', 'ratio_to_ctrpty_avg',
    'q_sin', 'q_cos', 'qoq_pct', 'yoy_pct',
    'roll_slope_4', 'miss_count_4', 'dev_from_period_mean'
]

# drop NaNs from train
train_long.dropna(subset=feat_cols+['y'], inplace=True)

# train / val split
train_set = train_long[train_long.period <= TRAIN_END]
val_set = train_long[(train_long.period > TRAIN_END) & (
    train_long.period <= pd.Period("2019Q4", freq='Q'))]

X_tr, y_tr = train_set[feat_cols], train_set['y']
X_val, y_val = val_set[feat_cols], val_set['y']

# ───────────────────────────────────────────────
# 6) fit LightGBM
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
    deterministic=True,
    force_row_wise=True,
    bagging_seed=42,
    feature_fraction_seed=42,
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    eval_names=['train', 'valid'],
)

# ───────────────────────────────────────────────
# 7) iterative multi‐step forecast
# seed history through 2019Q4
history = pd.concat([
    train_long,
    test_long[test_long.period > TRAIN_END]
], ignore_index=True)

forecasts = []
for p in pd.period_range(FORECAST_START, FORECAST_END, freq='Q'):
    # recompute features on the *full* history
    df_h = history.copy()
    add_features(df_h)
    add_more_features(df_h)

    # extract features for period p
    Xp = df_h[df_h.period == p][feat_cols]
    if Xp.empty:
        continue

    # predict
    yp_log = model.predict(Xp)
    yp = np.expm1(yp_log)

    # attach into a DataFrame preserving ID_COLS + period + true exposure
    dfp = df_h[df_h.period == p][ID_COLS + ['period', 'exposure']].copy()
    dfp['exposure_pred'] = yp
    forecasts.append(dfp)

    # append predictions into history so next quarter’s lags use them
    history = pd.concat([history, dfp], ignore_index=True)

all_forecasts = pd.concat(forecasts, ignore_index=True)

# ───────────────────────────────────────────────
# 8) compute WMAPE by quarter
wmape_by_q = (
    all_forecasts
    .assign(
        abs_err=lambda d: (d.exposure - d.exposure_pred).abs(),
        abs_true=lambda d: d.exposure.abs()
    )
    .groupby('period')
    .agg({'abs_err': 'sum', 'abs_true': 'sum'})
    .assign(WMAPE=lambda d: d.abs_err/d.abs_true)
    .WMAPE
)

# plot & save
wmape_by_q.plot(marker='o')
plt.title("Yearly WMAPE on 2020–2024 (iterated forecast)")
plt.ylabel("WMAPE")
plt.savefig(OUT_DIR/"lightgbm_iterative_wmape.png", dpi=300)
plt.show()

print(wmape_by_q)
