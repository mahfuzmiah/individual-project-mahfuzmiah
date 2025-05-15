#!/usr/bin/env python
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

# reproducibility
np.random.seed(42)

# make sure config.py is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import LONG_DATA_CSV, LONG_DATA_TEST_CSV, DIAGRAMS_DIR  # noqa

# new: prediction output dir
PRED_STATS_DIR = REPO_ROOT / "predictions" / "statistics"
PRED_STATS_DIR.mkdir(parents=True, exist_ok=True)
# Make Diarams directory
DIAGRAMS_DIR = DIAGRAMS_DIR / "Statistics"


def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom


def rolling_validation(train, train_window, test_window):
    roll_preds = []
    start = train['ds'].min()
    end = train['ds'].max()

    while start + train_window + test_window <= end:
        train_roll = train[(train['ds'] >= start) &
                           (train['ds'] < start + train_window)]
        valid_roll = train[(train['ds'] >= start + train_window) &
                           (train['ds'] < start + train_window + test_window)]
        h = valid_roll['ds'].nunique()

        model = StatsForecast(
            models=[SeasonalNaive(season_length=4)],
            freq='QE', n_jobs=1
        ).fit(train_roll)
        p = model.predict(h=h, level=[90]).reset_index()

        merged = (p
                  .merge(valid_roll, on=['ds', 'unique_id'], how='left')
                  .assign(roll_start=start))
        roll_preds.append(merged)

        start += test_window

    return pd.concat(roll_preds, ignore_index=True)


def final_test_evaluation(train_path, test_path):
    train = pd.read_csv(train_path, parse_dates=['ds'])
    test = pd.read_csv(test_path, parse_dates=['ds'])
    h = test['ds'].nunique()

    model = StatsForecast(
        models=[SeasonalNaive(season_length=4)],
        freq='QE', n_jobs=1
    ).fit(train)
    p = model.predict(h=h, level=[90]).reset_index()
    final = p.merge(test, on=['ds', 'unique_id'],
                    how='left').dropna(subset=['y'])
    print("Final Test Overall WMAPE:", wmape(
        final['y'], final['SeasonalNaive']))
    return final


def main():
    # — Load & prepare —
    train = pd.read_csv(LONG_DATA_CSV, parse_dates=['ds'])
    test = pd.read_csv(LONG_DATA_TEST_CSV, parse_dates=['ds'])
    train['ds'] = train['ds'] + pd.offsets.QuarterEnd()

    # — Rolling Validation —
    train_window = pd.DateOffset(years=5)
    test_window = pd.DateOffset(years=1)
    t0 = time.time()
    roll = rolling_validation(train, train_window, test_window)
    roll = roll.dropna(subset=['y'])
    roll_time = time.time() - t0
    print("Overall Rolling WMAPE:", wmape(roll['y'], roll['SeasonalNaive']))

    # save rolling preds & actuals
    roll_out = PRED_STATS_DIR / "seasonalnaive_rolling_preds.csv"
    roll.to_csv(roll_out, index=False)
    print("Wrote rolling predictions to:", roll_out)

    # — Final Test Evaluation —
    t1 = time.time()
    final = final_test_evaluation(LONG_DATA_CSV, LONG_DATA_TEST_CSV)
    final_time = time.time() - t1

    # save final preds & actuals
    final_out = PRED_STATS_DIR / "seasonalnaive_final_preds.csv"
    final.to_csv(final_out, index=False)
    print("Wrote final predictions to:", final_out)

    # — Timing summary —
    summary = pd.DataFrame([{
        'stage': 'rolling_validation',
        'total_time_s': round(roll_time, 2),
        'avg_time_per_forecast_s': round(roll_time / len(roll), 4)
    }, {
        'stage': 'final_test',
        'total_time_s': round(final_time, 2),
        'avg_time_per_forecast_s': round(final_time / len(final), 4)
    }])
    summary_out = PRED_STATS_DIR / "seasonalnaive_timing_summary.csv"
    summary.to_csv(summary_out, index=False)
    print("Wrote timing summary to:", summary_out)
    # plot rolling WMAPE over time
    wmape_by_date = roll.groupby('ds').apply(
        lambda g: wmape(g['y'], g['SeasonalNaive'])
    )
    out_dir = DIAGRAMS_DIR / "SeasonalNaive_Results_diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.title("Seasonal Naïve WMAPE Over Time (Rolling Validation)")
    plt.xlabel("Date")
    plt.ylabel("WMAPE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir /
                "Rolling_WMAPE_over_time.png", dpi=300)
    plt.close()

    # — Final Test Evaluation —
    final = final_test_evaluation(LONG_DATA_CSV, LONG_DATA_TEST_CSV)

    # plot one or two example series
    for series_id in ["AT_AL_Q", "US_ZW_U"]:
        df_s = final[final['unique_id'] == series_id]
        plt.figure(figsize=(10, 5))
        plt.plot(df_s['ds'], df_s['y'],
                 label='Actual', marker='o')
        plt.plot(df_s['ds'], df_s['SeasonalNaive'],
                 label='Forecast', marker='x')
        plt.fill_between(
            df_s['ds'],
            df_s['SeasonalNaive-lo-90'],
            df_s['SeasonalNaive-hi-90'],
            color='gray', alpha=0.2, label='90% CI'
        )
        plt.title(f"Seasonal Naïve Forecast vs Actual: {series_id}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / f"SeasonalNaiveForecast_{series_id}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
