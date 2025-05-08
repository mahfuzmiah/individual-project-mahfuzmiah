
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage, WindowAverage
from statsforecast.utils import ConformalIntervals
import pandas as pd
import matplotlib.pyplot as plt


def wmape(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / abs(y_true).sum()


def rolling_validation(data, train_window, test_window):
    """
    Perform rolling validation on the given data.
    The data must have columns ['ds', 'unique_id', 'y'] with ds as quarter-end dates.
    For each rolling window, fit the SeasonalWindowAverage model and forecast for the test period.
    Returns a DataFrame with merged forecasts and actual values.
    """
    roll_preds = []
    start_date = data['ds'].min()
    end_date = data['ds'].max()

    while start_date + train_window + test_window <= end_date:
        # Define the training and validation (test) windows for this rolling iteration
        train_roll = data[(data['ds'] >= start_date) & (
            data['ds'] < start_date + train_window)]
        valid_roll = data[(data['ds'] >= start_date + train_window) &
                          (data['ds'] < start_date + train_window + test_window)]

        # Forecast horizon: number of unique dates in the validation window
        h = valid_roll['ds'].nunique()

        # Create conformal intervals configuration for prediction intervals
        conf_int = ConformalIntervals()

        # Instantiate and fit the SeasonalWindowAverage model for this rolling window
        model = StatsForecast(
            models=[SeasonalWindowAverage(
                season_length=4, window_size=3, prediction_intervals=conf_int)],
            freq='QE', n_jobs=1
        )
        model.fit(train_roll)

        # Forecast for h periods
        p = model.predict(h=h, level=[90])
        p_reset = p.reset_index()

        # Merge forecasts with the actual values from the validation window
        merged = p_reset.merge(valid_roll, on=['ds', 'unique_id'], how='left')
        merged['roll_start'] = start_date  # Tag which rolling window this is
        roll_preds.append(merged)

        # Move the rolling window forward by the test window length
        start_date += test_window

    return pd.concat(roll_preds)


def final_test_evaluation(train_path, test_path):
    """
    Fit the model on the entire training set and generate forecasts for the final test set.
    Then merge the forecasts with the test data and compute overall WMAPE.
    """
    # Load training and test data
    train = pd.read_csv(train_path, parse_dates=['ds'])
    test = pd.read_csv(test_path, parse_dates=['ds'])

    # Forecast horizon is the number of unique dates in the test set
    h = test['ds'].nunique()

    conf_int = ConformalIntervals()

    # Instantiate and fit the model on the full training data
    model = StatsForecast(
        models=[SeasonalWindowAverage(
            season_length=4, window_size=3, prediction_intervals=conf_int)],
        freq='QE', n_jobs=-1
    )
    model.fit(train)

    # Forecast for the final test set horizon
    p = model.predict(h=h, level=[90])
    p_reset = p.reset_index()

    # Filter test set to include only forecast dates
    test_filtered = test[test['ds'].isin(p_reset['ds'].unique())]

    # Merge forecasts with test data and drop rows with missing actual values
    final_results = p_reset.merge(
        test_filtered, on=['ds', 'unique_id'], how='left').dropna(subset=['y'])
    overall_final_wmape = wmape(final_results['y'], final_results['SeasWA'])
    print("Final Test Overall WMAPE:", overall_final_wmape)

    return final_results


def main():
    # File paths for your cleaned long-format data
    train_path = '/Users/mahfuz/Final_project/Final_repo/long_data.csv'
    test_path = '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv'

    # --- Rolling Validation ---
    # Load training data and convert ds to quarter-end dates
    train = pd.read_csv(train_path, parse_dates=['ds'])
    train['ds'] = train['ds'] + pd.offsets.QuarterEnd()

    # Define rolling validation windows, e.g., 5-year training window and 1-year test window
    train_window = pd.DateOffset(years=5)
    test_window = pd.DateOffset(years=1)

    roll_results = rolling_validation(train, train_window, test_window)
    # Drop any rows where actual y is missing
    roll_results = roll_results.dropna(subset=['y'])

    overall_roll_wmape = wmape(roll_results['y'], roll_results['SeasWA'])
    print("Overall Rolling WMAPE:", overall_roll_wmape)

    # Optionally, plot WMAPE over time from rolling validation
    wmape_by_date = roll_results.groupby('ds')[['y', 'SeasWA']].apply(
        lambda x: wmape(x['y'], x['SeasWA']))
    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.xlabel('Forecast Date')
    plt.ylabel('WMAPE')
    plt.title('WMAPE Over Time (Rolling Validation)')
    plt.grid(True)
    plt.show()

    # --- Final Test Evaluation ---
    final_results = final_test_evaluation(train_path, test_path)

    # Optionally, plot forecasts vs. actuals for a specific series from the final test
    series_id = 'AT_AL_Q'
    series_data = final_results[final_results['unique_id'] == series_id]
    plt.figure(figsize=(10, 5))
    plt.plot(series_data['ds'], series_data['y'], label='Actual', marker='o')
    plt.plot(series_data['ds'], series_data['SeasWA'],
             label='Forecast (Seasonal Window Average)', marker='x')
    if 'SeasWA-lo-90' in series_data.columns and 'SeasWA-hi-90' in series_data.columns:
        plt.fill_between(series_data['ds'],
                         series_data['SeasWA-lo-90'],
                         series_data['SeasWA-hi-90'],
                         color='gray', alpha=0.2, label='90% CI')
    plt.title(
        f"Seasonal Window Average Forecast vs Actual for {series_id} (Final Test)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
