
# Importing required libraries
import time
import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage, WindowAverage, ARIMA
from statsforecast.utils import ConformalIntervals
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")


def wmape(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / abs(y_true).sum()


def arima_test_method(subset_length):

    train = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data.csv', parse_dates=['ds'])
    test = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv', parse_dates=['ds'])
    if subset_length != 5:
        subset_length = len(train['unique_id'].unique())
    subset_uids = train['unique_id'].unique()[:subset_length]
    train = train[train['unique_id'].isin(subset_uids)]
    test = test[test['unique_id'].isin(subset_uids)]

    # Sort training data by date
    df = train.sort_values('ds')
    results_list = []

    # ---- ARIMA Forecasting with auto_arima ----
    total_fit_time = 0.0
    total_forecast_time = 0.0

    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds').set_index('ds')
        try:
            fit_start = time.time()
            stepwise_fit = auto_arima(group['y'], start_p=1, start_q=1,
                                      max_p=3, max_q=3, m=4,
                                      seasonal=True,
                                      d=None, D=1, trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)
            fit_end = time.time()
            total_fit_time += (fit_end - fit_start)
        except Exception as e:
            print(f"Could not fit model for series {uid}: {e}")
            continue

       # print(f"Summary for series {uid}:")
       # print(stepwise_fit.summary())
        group = df[df['unique_id'] == 'AT_AD_F'].sort_values('ds')
        # Select corresponding test series
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

        # Align forecast with test index
        forecast_index = test_series.index[:n_periods]
        forecast_series = pd.Series(
            forecast, index=forecast_index, name='forecast')

        merged = pd.concat([test_series, forecast_series], axis=1)
        merged['unique_id'] = uid  # record which series this is
        merged = merged.dropna(subset=['y'])  # drop if no actual value

        # Store merged results
        results_list.append(merged)
        error = wmape(merged['y'], merged['forecast'])
        print(
            f"UID={uid}, WMAPE={error:.4f}, ARIMA Model={stepwise_fit.order}, Seasonal={stepwise_fit.seasonal_order}")
        # Concatenate all forecast results into a single DataFrame
    if not results_list:
        print("No forecast results available.")
        return
    results = pd.concat(results_list)

    wmape_by_date = results.groupby(results.index).apply(
        lambda x: wmape(x['y'], x['forecast']))
    print("Total fit time: {:.2f} seconds".format(total_fit_time))
    print("Total forecast time: {:.2f} seconds".format(total_forecast_time))
    # --- Plotting ---
    # (A) Plot forecasts vs. actuals for a selected series (example: first series in our subset)
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
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Diagrams/Arima_diagrams/Arima_forecast_selected_series.png')
    plt.show()
    plt.close()

    # (B) Plot WMAPE over time (each forecast date)
    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.xlabel('Forecast Date')
    plt.ylabel('WMAPE')
    plt.title('WMAPE Over Time (ARIMA Forecasts)')
    plt.grid(True)
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Diagrams/Arima_diagrams/Arima_WMAPE_over_time.png')
    plt.show()
    plt.close()


def composite_variability():
    train = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data.csv', parse_dates=['ds'])
    test = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv', parse_dates=['ds'])
    # train_ts = train.set_index('ds')

    # Decompose the time series. For quarterly data, period=4.
    # Use additive or multiplicative model depending on your data's characteristics.
    # Ensure the series are sorted by date
    df = train.sort_values('ds')

    # Group by unique_id

    composite_variability = {}

    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds')

        # 1. Focus on the earlier portion (e.g., before 2014-01-01).
        #    If everything is imputed early on, you might set a different cutoff or skip.
        group_early = group[group['ds'] <= pd.Timestamp('2014-01-01')].copy()

        # 2. Ensure enough data points remain in the early subset.
        #    For example, skip if fewer than 8 data points remain.
        if len(group_early) < 8:
            continue

        # 3. Set ds as the index for seasonal decomposition.
        group_early = group_early.set_index('ds')

        try:
            # 4. Perform seasonal decomposition on the early portion
            #    Adjust period=4 for quarterly data or your actual season length
            result = seasonal_decompose(
                group_early['y'], model='additive', period=4)

            # Compute standard deviations for trend & residual
            trend_std = np.nanstd(result.trend)
            resid_std = np.nanstd(result.resid)

            # Composite score: sum of trend & residual variability
            composite_score = trend_std + resid_std
            composite_variability[uid] = composite_score
        except Exception as e:
            print(f"Could not decompose series {uid}: {e}")

    # 5. Pick the series with the highest variability in the early portion
    if composite_variability:
        best_uid = max(composite_variability, key=composite_variability.get)
        print("Series with highest (trend + residual) variability in the early data:", best_uid)

        # Optional: Re-decompose the entire series for final plotting
        best_series = df[df['unique_id'] == best_uid].sort_values('ds')
        best_series = best_series.set_index('ds')
        # Decompose the full series
        result_full = seasonal_decompose(
            best_series['y'], model='additive', period=4)
        fig = result_full.plot()
        axes = fig.axes
        axes[0].set_title("")      # Remove the default title ("y")
        axes[0].set_ylabel("Observed")  # or "y" or whatever label you want

        fig.suptitle(
            f'Seasonal Decomposition for {best_uid}')
        plt.savefig(
            '/Users/mahfuz/Final_project/Final_repo/Diagrams/Arima_diagrams/SeasonalDecomposition.png')
    else:
        print("No series qualified under the early-data filter.")


def main():
    # composite_variability()
    arima_test_method(5)


if __name__ == '__main__':
    main()
