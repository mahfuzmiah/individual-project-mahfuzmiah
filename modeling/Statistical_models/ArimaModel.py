
#!/usr/bin/env python
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import os
warnings.filterwarnings("ignore")
np.random.seed(42)


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_PATH))
from config import REPO_ROOT, DATASETS_DIR, DIAGRAMS_DIR, LONG_DATA_CSV, LONG_DATA_TEST_CSV  # nopep8

# --- Data Loading ---
TRAIN_PATH = LONG_DATA_CSV
TEST_PATH = LONG_DATA_TEST_CSV
ARIMA_DIAGRAMS_DIR = DIAGRAMS_DIR / "Arima_diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
PRED_ARIMA_DIR = REPO_ROOT_PATH / "predictions" / "arima"
PRED_ARIMA_DIR.mkdir(parents=True, exist_ok=True)


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train = pd.read_csv(train_path, parse_dates=['ds'])
    test = pd.read_csv(test_path, parse_dates=['ds'])
    return train, test

# --- Utility Functions ---


# def wmape(y_true, y_pred):
#     return (abs(y_true - y_pred)).sum() / abs(y_true).sum()

def wmape(y_true, y_pred):
    # mask to only keep finite pairs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y, ŷ = y_true[mask], y_pred[mask]
    denom = np.abs(y).sum()
    if denom == 0:
        return np.nan
    return np.abs(y - ŷ).sum() / denom

# --- Function to convert seconds ---


def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return hours, minutes, secs


# --- Worker Function for Parallel Processing ---

def process_series(uid, group, test):
    try:
        group = group.sort_values('ds').set_index('ds')
        fit_start = time.time()
        stepwise_fit = auto_arima(
            group['y'],
            start_p=1, start_q=1,
            max_p=3, max_q=3, m=4,
            seasonal=True,
            d=None, D=1,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            random_state=42  # reproducible search

        )
        fit_end = time.time()
        fit_time = fit_end - fit_start

        test_series = test[test['unique_id'] == uid].sort_values('ds')
        if test_series.empty:
            print(f"No test data for series {uid}")
            return None, uid, fit_time, 0.0

        test_series = test_series.set_index('ds')['y']
        n_periods = len(test_series)
        forecast_start = time.time()
        forecast = stepwise_fit.predict(n_periods=n_periods)
        forecast_end = time.time()
        forecast_time = forecast_end - forecast_start

        forecast_index = test_series.index[:n_periods]
        forecast_series = pd.Series(
            forecast, index=forecast_index, name='forecast')
        merged = pd.concat([test_series, forecast_series], axis=1)
        merged['unique_id'] = uid

        error = wmape(merged['y'], merged['forecast'])
        print(
            f"UID={uid}, WMAPE={error:.4f}, ARIMA Model={stepwise_fit.order}, Seasonal={stepwise_fit.seasonal_order}")
        return merged, uid, fit_time, forecast_time

    except Exception as e:
        print(f"Could not fit model for series {uid}: {e}")
        return None, uid, 0.0, 0.0

# --- ARIMA Forecasting Functions ---


def arima_test_method_in_series(subset_length=None, train=None, test=None):
    # Load data if not provided
    if train is None or test is None:
        train, test = load_data()

    # Limit the dataset to the first subset_length unique IDs
    if subset_length is None:
        subset_length = len(train['unique_id'].unique())
    subset_uids = train['unique_id'].unique()[:subset_length]
    train = train[train['unique_id'].isin(subset_uids)]
    test = test[test['unique_id'].isin(subset_uids)]
    df = train.sort_values('ds')

    results_list = []
    total_fit_time = 0.0
    total_forecast_time = 0.0

    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds').set_index('ds')
        try:
            fit_start = time.time()
            stepwise_fit = auto_arima(
                group['y'],
                start_p=1, start_q=1,
                max_p=3, max_q=3, m=4,
                seasonal=True,
                trace=False,
                d=None, D=1,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            fit_end = time.time()
            total_fit_time += (fit_end - fit_start)
        except Exception as e:
            print(f"Could not fit model for series {uid}: {e}")
            continue

        test_series = test[test['unique_id'] == uid].sort_values('ds')
        if test_series.empty:
            print(f"No test data for series {uid}")
            continue
        test_series = test_series.set_index('ds')['y']
        n_periods = len(test_series)
        forecast_start = time.time()
        forecast = stepwise_fit.predict(n_periods=n_periods)
        forecast_end = time.time()
        total_forecast_time += (forecast_end - forecast_start)
        forecast_index = test_series.index[:n_periods]
        forecast_series = pd.Series(
            forecast, index=forecast_index, name='forecast')
        merged = pd.concat([test_series, forecast_series], axis=1)
        merged['unique_id'] = uid
        results_list.append(merged)
        error = wmape(merged['y'], merged['forecast'])
        print(
            f"UID={uid}, WMAPE={error:.4f}, ARIMA Model={stepwise_fit.order}, Seasonal={stepwise_fit.seasonal_order}")

    if not results_list:
        print("No forecast results available.")
        return

    results = pd.concat(results_list)
    preds = results.reset_index().rename(
        columns={'y': 'actual', 'forecast': 'forecast'})
    preds_out = PRED_ARIMA_DIR / "arima_parallel_predictions.csv"
    preds.to_csv(preds_out, index=False)
    print(f"Wrote ARIMA predictions to {preds_out}")
    return total_fit_time, total_forecast_time

# Estimate total time taken to process dataset in series


def estimate_time_taken_for_series(subset_length=None, train=None, test=None):
    if train is None or test is None:
        train, test = load_data()
    if subset_length is None:
        subset_length = len(train['unique_id'].unique())
    total_fit_time, total_forecast_time = arima_test_method_in_series(
        subset_length, train, test)
    estimated_total = (total_fit_time / subset_length) * \
        len(train['unique_id'].unique())
    average_time = (total_fit_time / subset_length)
    hours, minutes, seconds = seconds_to_hms(estimated_total)
    print(f"{hours} hours, {minutes} minutes, {seconds:.0f} seconds")
    print(f"Average time per series: {average_time:.2f} seconds")


def arima_test_method_parallel(subset_length=None, train=None, test=None):
    # Load data if not provided
    if train is None or test is None:
        train, test = load_data()

    if subset_length is None:
        subset_length = len(train['unique_id'].unique())
    subset_uids = train['unique_id'].unique()[:subset_length]

    train = train[train['unique_id'].isin(subset_uids)]
    test = test[test['unique_id'].isin(subset_uids)]
    df = train.sort_values('ds')

    results_list = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=os.cpu_count()
                             ) as executor:
        futures = []
        start_time = time.time()
        for uid, group in df.groupby('unique_id'):
            futures.append(executor.submit(process_series, uid, group, test))
        for future in as_completed(futures):
            merged, uid, fit_time, forecast_time = future.result()
            if merged is not None:
                results_list.append(merged)
        end_time = time.time()
    total_time = time.time() - t0
    n_series = len(subset_uids)
    avg_time = total_time / n_series
    # 2) dump timing summary
    timing_df = pd.DataFrame([{
        'model': 'arima_parallel',
        'total_time_s': round(total_time, 2),
        'avg_time_per_series_s': round(avg_time, 4)
    }])
    timing_out = PRED_ARIMA_DIR/"arima_timing_summary.csv"
    timing_df.to_csv(timing_out, index=False)
    print(f"Wrote timing summary to {timing_out}")

    if not results_list:
        print("No forecast results available.")
        return

    results = pd.concat(results_list)

    # wmape_by_date = results.groupby(results.index).apply(
    #     lambda x: wmape(x['y'], x['forecast']))
    # 1) concatenate all per‐series DataFrames

    # 2) write out actual vs forecast for every (series, date)
    preds = (
        results
        .reset_index()                     # bring ds back as a column
        .rename(columns={'y': 'actual'})    # y -> actual
        [['ds', 'unique_id', 'actual', 'forecast']]
    )
    preds_out = PRED_ARIMA_DIR/"arima_parallel_predictions.csv"
    preds.to_csv(preds_out, index=False)
    print(f"Wrote ARIMA parallel predictions to {preds_out}")
    wmape_by_date = results.groupby('ds')[['y', 'forecast']] \
        .apply(lambda df: wmape(df['y'].values,
                                df['forecast'].values))

    total_hours, total_minutes, total_secs = seconds_to_hms(total_time)
    average_time = total_time / subset_length
    avg_hours, avg_minutes, avg_secs = seconds_to_hms(average_time)
    print("Total time: {} hours, {} minutes, {:.2f} seconds".format(
        total_hours, total_minutes, total_secs))
    print("Average time per series: {:.2f} seconds ({} hours, {} minutes, {:.2f} seconds)".format(
        average_time, avg_hours, avg_minutes, avg_secs))
    # Example Plot: Forecast vs. Actual for the first series
    selected_series = results[results['unique_id'] == subset_uids[0]]
    plt.figure(figsize=(10, 5))
    plt.plot(selected_series.index,
             selected_series['y'], label='Actual', marker='o')
    plt.plot(selected_series.index,
             selected_series['forecast'], label='Forecast (ARIMA)', marker='x')
    plt.title(f"ARIMA Forecast vs Actual for {subset_uids[0]}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(ARIMA_DIAGRAMS_DIR /
                f"ArimaForecast_{subset_uids[0]}.png", dpi=300)

    # plt.savefig(
    #     '/Users/mahfuz/Final_project/Final_repo/Diagrams/Arima_diagrams/Arima_forecast_selected_series.png')
    plt.show()
    plt.close()

    # Example Plot: WMAPE over time
    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.xlabel('Forecast Date')
    plt.ylabel('WMAPE')
    plt.title('WMAPE Over Time (ARIMA Forecasts)')
    plt.grid(True)
    plt.savefig(ARIMA_DIAGRAMS_DIR /
                "Arima_WMAPE_over_time.png", dpi=300)
    # plt.savefig(
    #     '/Users/mahfuz/Final_project/Final_repo/Diagrams/Arima_diagrams/Arima_WMAPE_over_time.png')
    plt.show()
    plt.close()


def calculate_composite_variability(train, cutoff_date=pd.Timestamp('2014-01-01'), min_points=8, period=4):
    """
    Calculate the composite variability score for each series in the training data.

    Parameters:
        train (DataFrame): Training data with at least 'unique_id', 'ds', and 'y' columns.
        cutoff_date (Timestamp): Only consider data on or before this date for variability calculation.
        min_points (int): Minimum number of data points required in the early period.
        period (int): The seasonal period to use in seasonal_decompose.

    Returns:
        dict: A dictionary mapping each unique_id to its composite variability score.
    """
    df = train.sort_values('ds')
    composite_variability = {}

    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds')
        # Filter to data at or before the cutoff_date
        group_early = group[group['ds'] <= cutoff_date].copy()
        if len(group_early) < min_points:
            continue

        group_early = group_early.set_index('ds')
        try:
            result = seasonal_decompose(
                group_early['y'], model='additive', period=period)
            trend_std = np.nanstd(result.trend)
            resid_std = np.nanstd(result.resid)
            composite_score = trend_std + resid_std
            composite_variability[uid] = composite_score
        except Exception as e:
            print(f"Could not decompose series {uid}: {e}")

    return composite_variability


def save_best_seasonal_decomposition(train, composite_variability, period=4):
    """
    Identify the best series (highest composite variability) and save its seasonal decomposition diagram.

    Parameters:
        train (DataFrame): Training data with at least 'unique_id', 'ds', and 'y' columns.
        composite_variability (dict): A dictionary mapping unique_id to composite variability scores.
        output_path (str): File path where the diagram will be saved.
        period (int): The seasonal period to use in seasonal_decompose.
    """
    if composite_variability:
        best_uid = max(composite_variability, key=composite_variability.get)
        print("Series with highest (trend + residual) variability in the early data:", best_uid)

        df = train.sort_values('ds')
        best_series = df[df['unique_id'] == best_uid].sort_values('ds')
        best_series = best_series.set_index('ds')

        # Perform seasonal decomposition on the full series
        result_full = seasonal_decompose(
            best_series['y'], model='additive', period=period)
        fig = result_full.plot()
        axes = fig.axes
        axes[0].set_title("")      # Clear default title on the observed plot
        axes[0].set_ylabel("Observed")
        fig.suptitle(f'Seasonal Decomposition for {best_uid}')
        plt.savefig(ARIMA_DIAGRAMS_DIR /
                    f"SeasonalDecomposition_{best_uid}.png", dpi=300)
        # plt.savefig(
        #     "/Users/mahfuz/Final_project/Final_repo/Diagrams/Arima_diagrams/SeasonalDecomposition.png")
        plt.close()
    else:
        print("No series qualified under the early-data filter.")


def main():
    train, test = load_data()

    # comp_variability = calculate_composite_variability(train)
    # save_best_seasonal_decomposition(train, comp_variability)
    # estimate_time_taken_for_series(5, train, test)
    arima_test_method_parallel(10, train, test)


if __name__ == '__main__':
    main()
