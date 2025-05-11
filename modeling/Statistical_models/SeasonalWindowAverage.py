
# from statsforecast import StatsForecast
# from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage, WindowAverage
# from statsforecast.utils import ConformalIntervals
# import pandas as pd
# import matplotlib.pyplot as plt


# def wmape(y_true, y_pred):
#     return (abs(y_true - y_pred)).sum() / abs(y_true).sum()


# def rolling_validation(data, train_window, test_window):
#     """
#     Perform rolling validation on the given data.
#     The data must have columns ['ds', 'unique_id', 'y'] with ds as quarter-end dates.
#     For each rolling window, fit the SeasonalWindowAverage model and forecast for the test period.
#     Returns a DataFrame with merged forecasts and actual values.
#     """
#     roll_preds = []
#     start_date = data['ds'].min()
#     end_date = data['ds'].max()

#     while start_date + train_window + test_window <= end_date:
#         # Define the training and validation (test) windows for this rolling iteration
#         train_roll = data[(data['ds'] >= start_date) & (
#             data['ds'] < start_date + train_window)]
#         valid_roll = data[(data['ds'] >= start_date + train_window) &
#                           (data['ds'] < start_date + train_window + test_window)]

#         # Forecast horizon: number of unique dates in the validation window
#         h = valid_roll['ds'].nunique()

#         # Create conformal intervals configuration for prediction intervals
#         conf_int = ConformalIntervals()

#         # Instantiate and fit the SeasonalWindowAverage model for this rolling window
#         model = StatsForecast(
#             models=[SeasonalWindowAverage(
#                 season_length=4, window_size=3, prediction_intervals=conf_int)],
#             freq='QE', n_jobs=1
#         )
#         model.fit(train_roll)

#         # Forecast for h periods
#         p = model.predict(h=h, level=[90])
#         p_reset = p.reset_index()

#         # Merge forecasts with the actual values from the validation window
#         merged = p_reset.merge(valid_roll, on=['ds', 'unique_id'], how='left')
#         merged['roll_start'] = start_date  # Tag which rolling window this is
#         roll_preds.append(merged)

#         # Move the rolling window forward by the test window length
#         start_date += test_window

#     return pd.concat(roll_preds)


# def final_test_evaluation(train_path, test_path):
#     """
#     Fit the model on the entire training set and generate forecasts for the final test set.
#     Then merge the forecasts with the test data and compute overall WMAPE.
#     """
#     # Load training and test data
#     train = pd.read_csv(train_path, parse_dates=['ds'])
#     test = pd.read_csv(test_path, parse_dates=['ds'])

#     # Forecast horizon is the number of unique dates in the test set
#     h = test['ds'].nunique()

#     conf_int = ConformalIntervals()

#     # Instantiate and fit the model on the full training data
#     model = StatsForecast(
#         models=[SeasonalWindowAverage(
#             season_length=4, window_size=3, prediction_intervals=conf_int)],
#         freq='QE', n_jobs=-1
#     )
#     model.fit(train)

#     # Forecast for the final test set horizon
#     p = model.predict(h=h, level=[90])
#     p_reset = p.reset_index()

#     # Filter test set to include only forecast dates
#     test_filtered = test[test['ds'].isin(p_reset['ds'].unique())]

#     # Merge forecasts with test data and drop rows with missing actual values
#     final_results = p_reset.merge(
#         test_filtered, on=['ds', 'unique_id'], how='left').dropna(subset=['y'])
#     overall_final_wmape = wmape(final_results['y'], final_results['SeasWA'])
#     print("Final Test Overall WMAPE:", overall_final_wmape)

#     return final_results


# def main():
#     # File paths for your cleaned long-format data
#     train_path = '/Users/mahfuz/Final_project/Final_repo/long_data.csv'
#     test_path = '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv'

#     # --- Rolling Validation ---
#     # Load training data and convert ds to quarter-end dates
#     train = pd.read_csv(train_path, parse_dates=['ds'])
#     train['ds'] = train['ds'] + pd.offsets.QuarterEnd()

#     # Define rolling validation windows, e.g., 5-year training window and 1-year test window
#     train_window = pd.DateOffset(years=5)
#     test_window = pd.DateOffset(years=1)

#     roll_results = rolling_validation(train, train_window, test_window)
#     # Drop any rows where actual y is missing
#     roll_results = roll_results.dropna(subset=['y'])

#     overall_roll_wmape = wmape(roll_results['y'], roll_results['SeasWA'])
#     print("Overall Rolling WMAPE:", overall_roll_wmape)

#     # Optionally, plot WMAPE over time from rolling validation
#     wmape_by_date = roll_results.groupby('ds')[['y', 'SeasWA']].apply(
#         lambda x: wmape(x['y'], x['SeasWA']))
#     plt.figure(figsize=(10, 5))
#     plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
#     plt.xlabel('Forecast Date')
#     plt.ylabel('WMAPE')
#     plt.title('WMAPE Over Time (Rolling Validation)')
#     plt.grid(True)
#     plt.show()

#     # --- Final Test Evaluation ---
#     final_results = final_test_evaluation(train_path, test_path)

#     # Optionally, plot forecasts vs. actuals for a specific series from the final test
#     series_id = 'AT_AL_Q'
#     series_data = final_results[final_results['unique_id'] == series_id]
#     plt.figure(figsize=(10, 5))
#     plt.plot(series_data['ds'], series_data['y'], label='Actual', marker='o')
#     plt.plot(series_data['ds'], series_data['SeasWA'],
#              label='Forecast (Seasonal Window Average)', marker='x')
#     if 'SeasWA-lo-90' in series_data.columns and 'SeasWA-hi-90' in series_data.columns:
#         plt.fill_between(series_data['ds'],
#                          series_data['SeasWA-lo-90'],
#                          series_data['SeasWA-hi-90'],
#                          color='gray', alpha=0.2, label='90% CI')
#     plt.title(
#         f"Seasonal Window Average Forecast vs Actual for {series_id} (Final Test)")
#     plt.xlabel("Date")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.show()


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import SeasonalWindowAverage
from statsforecast.utils import ConformalIntervals

# make config.py importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import LONG_DATA_CSV, LONG_DATA_TEST_CSV, DIAGRAMS_DIR  # nopep8

# reproducible “randomness”
np.random.seed(0)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE, returns NaN when sum(abs(y_true)) == 0."""
    denom = np.abs(y_true).sum()
    return np.nan if denom == 0 else np.abs(y_true - y_pred).sum() / denom


def rolling_validation(
    data: pd.DataFrame,
    train_window: pd.DateOffset,
    test_window: pd.DateOffset
) -> pd.DataFrame:
    """
    Sliding‐window validation:
      • start at min(ds)
      • fit SeasonalWindowAverage on [start, start+train_window)
      • forecast next h periods where h = number of unique ds in test slice
      • merge forecasts with truth
      • advance start by test_window
      • repeat until you run out of data
    """
    results = []
    start, last = data['ds'].min(), data['ds'].max()

    while start + train_window + test_window <= last:
        train_slice = data[(data['ds'] >= start) &
                           (data['ds'] < start + train_window)]
        valid_slice = data[(data['ds'] >= start + train_window) &
                           (data['ds'] < start + train_window + test_window)]
        h = valid_slice['ds'].nunique()

        # set up conformal intervals for PI
        conf = ConformalIntervals()

        model = StatsForecast(
            models=[SeasonalWindowAverage(
                season_length=4,
                window_size=3,
                prediction_intervals=conf
            )],
            freq='QE',
            n_jobs=1,
        )
        model.fit(train_slice)
        preds = model.predict(h=h, level=[90]).reset_index()

        merged = preds.merge(valid_slice, on=['ds', 'unique_id'], how='left')
        merged['roll_start'] = start
        results.append(merged)

        start += test_window

    return pd.concat(results, ignore_index=True)


def final_test_evaluation(train_path: Path, test_path: Path) -> pd.DataFrame:
    """
    Fit SeasonalWindowAverage on the entire training set and
    forecast the final test period, then return merged results.
    """
    train = pd.read_csv(train_path, parse_dates=['ds'])
    test = pd.read_csv(test_path,  parse_dates=['ds'])
    h = test['ds'].nunique()

    conf = ConformalIntervals()
    model = StatsForecast(
        models=[SeasonalWindowAverage(
            season_length=4, window_size=3, prediction_intervals=conf
        )],
        freq='QE',
        n_jobs=1,
    )
    model.fit(train)
    preds = model.predict(h=h, level=[90]).reset_index()

    final = preds.merge(test, on=['ds', 'unique_id'],
                        how='left').dropna(subset=['y'])
    print("Final Test Overall WMAPE:", wmape(final['y'], final['SeasWA']))
    return final


def main():
    # — Load & align data —
    train = pd.read_csv(LONG_DATA_CSV, parse_dates=['ds'])
    train['ds'] = train['ds'] + pd.offsets.QuarterEnd()
    test = pd.read_csv(LONG_DATA_TEST_CSV, parse_dates=['ds'])
    test['ds'] = test['ds'] + pd.offsets.QuarterEnd()

    # — Rolling validation —
    train_window = pd.DateOffset(years=5)
    test_window = pd.DateOffset(years=1)

    roll = rolling_validation(train, train_window, test_window)
    roll = roll.dropna(subset=['y'])
    overall_roll = wmape(roll['y'], roll['SeasWA'])
    print("Overall Rolling WMAPE:", overall_roll)

    # plot WMAPE over time
    wmape_by_date = roll.groupby('ds').apply(
        lambda g: wmape(g['y'], g['SeasWA'])
    )
    out_dir = DIAGRAMS_DIR / "SeasonalWA_Results"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.title("Seasonal Window Average WMAPE (Rolling Validation)")
    plt.xlabel("Date")
    plt.ylabel("WMAPE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "Rolling_WMAPE_over_time.png", dpi=300)
    plt.close()

    # — Final test evaluation —
    final = final_test_evaluation(LONG_DATA_CSV, LONG_DATA_TEST_CSV)

    # example series plots
    for sid in ["AT_AL_Q", "US_ZW_U"]:
        df_s = final[final['unique_id'] == sid]
        plt.figure(figsize=(10, 5))
        plt.plot(df_s['ds'], df_s['y'],           label='Actual', marker='o')
        plt.plot(df_s['ds'], df_s['SeasWA'],      label='Forecast', marker='x')
        if 'SeasWA-lo-90' in df_s.columns:
            plt.fill_between(
                df_s['ds'],
                df_s['SeasWA-lo-90'], df_s['SeasWA-hi-90'],
                color='gray', alpha=0.2, label='90% CI'
            )
        plt.title(f"Seasonal WA Forecast vs Actual: {sid}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"SeasonalWAForecast_{sid}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
