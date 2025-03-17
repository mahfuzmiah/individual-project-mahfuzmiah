
# Importing required libraries
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


def arima_test_method():
    train = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data.csv', parse_dates=['ds'])
    test = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv', parse_dates=['ds'])

    subset_uids = train['unique_id'].unique()[:1]
    train = train[train['unique_id'].isin(subset_uids)]
    test = test[test['unique_id'].isin(subset_uids)]

    # Sort training data by date
    df = train.sort_values('ds')

    # ---- ARIMA Forecasting with auto_arima ----
    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds').set_index('ds')
        try:
            stepwise_fit = auto_arima(group['y'], start_p=1, start_q=1,
                                      max_p=3, max_q=3, m=4,
                                      seasonal=True,
                                      d=None, D=1, trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)
        except Exception as e:
            print(f"Could not fit model for series {uid}: {e}")
            continue

        print(f"Summary for series {uid}:")
        print(stepwise_fit.summary())
        group = df[df['unique_id'] == 'AT_AD_F'].sort_values('ds')
        print(group['y'].describe())
        # Select corresponding test series
        test_series = test[test['unique_id'] == uid].sort_values('ds')
        if test_series.empty:
            print(f"No test data for series {uid}")
            continue
        test_series = test_series.set_index('ds')['y']
        n_periods = len(test_series)
        forecast = stepwise_fit.predict(n_periods=n_periods)
        print(forecast)
        # Align forecast with test index
        forecast_index = test_series.index[:n_periods]
        forecast_series = pd.Series(
            forecast, index=forecast_index, name='forecast')
        print(f"forecast_series= {forecast_series}")

        merged = pd.concat([test_series, forecast_series], axis=1).dropna()
        error = wmape(merged['y'], merged['forecast'])
        print(
            f"UID={uid}, WMAPE={error:.4f}, ARIMA Model={stepwise_fit.order}, Seasonal={stepwise_fit.seasonal_order}")


def main():
    # --- Load Cleaned Data ---
    # Replace with your file paths
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

    # for uid, group in df.groupby('unique_id'):
    #     group = group.sort_values('ds')
    #     group = group.set_index('ds')
    #     result = seasonal_decompose(
    #         group['y'], model='additive', period=4)
    #     # Fit auto_arima function to AirPassengers dataset
    #     try:
    #         stepwise_fit = auto_arima(group['y'], start_p=1, start_q=1,
    #                                   max_p=3, max_q=3, m=4,
    #                                   seasonal=True,
    #                                   d=None, D=1, trace=True,
    #                                   error_action='ignore',   # we don't want to know if an order does not work
    #                                   suppress_warnings=True,  # we don't want convergence warnings
    #                                   stepwise=True)           # set to stepwise
    #     except Exception as e:
    #         print(f"Could not fit model for series {uid}: {e}")
    #         continue
    #     # To print the summary
    #     print(f"Summary for series {uid}:")
    #     # Generate forecasts for the length of the test set
    #     print(stepwise_fit.summary())
    #     test_series = test[test['unique_id'] == uid].sort_values('ds')
    #     if test_series.empty:
    #         print(f"No test data for series {uid}")
    #         continue
    #     test_series = test_series.set_index('ds')['y']
    #     n_periods = len(test_series)
    #     forecast = stepwise_fit.predict(n_periods=n_periods)

    #     # Align the forecast with the test index
    #     forecast_index = test_series.index[:n_periods]  # same dates as test
    #     forecast_series = pd.Series(
    #         forecast, index=forecast_index, name='forecast')

    #     # Merge with the actual test data for evaluation
    #     merged = pd.concat([test_series, forecast_series], axis=1).dropna()
    #     # Compute WMAPE
    #     error = wmape(merged['y'], merged['forecast'])
    #     print(
    #         f"UID={uid}, WMAPE={error:.4f}, ARIMA Model={stepwise_fit.order}, Seasonal={stepwise_fit.seasonal_order}")

    # For testing, select a small subset of unique_id groups (e.g., first 3 unique ids)


if __name__ == '__main__':
    arima_test_method()
    # main()
