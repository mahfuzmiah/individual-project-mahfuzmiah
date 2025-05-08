
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage
import pandas as pd
import matplotlib.pyplot as plt


def wmape(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / abs(y_true).sum()


def rolling_validation(train, train_window, test_window):
    """
    Perform rolling validation on a training DataFrame that uses quarter-end dates.
    """
    roll_preds = []
    start_date = train['ds'].min()
    end_date = train['ds'].max()

    while start_date + train_window + test_window <= end_date:
        # Define training and validation windows
        train_roll = train[(train['ds'] >= start_date) & (
            train['ds'] < start_date + train_window)]
        valid_roll = train[(train['ds'] >= start_date + train_window) &
                           (train['ds'] < start_date + train_window + test_window)]

        h = valid_roll['ds'].nunique()

        # Instantiate and fit the model for this rolling window
        model = StatsForecast(models=[SeasonalNaive(
            season_length=4)], freq='QE', n_jobs=1)
        model.fit(train_roll)

        # Forecast for h periods
        p = model.predict(h=h, level=[90])
        p_reset = p.reset_index()

        # Merge forecasts with the validation data
        merged = p_reset.merge(valid_roll, on=['ds', 'unique_id'], how='left')
        merged['roll_start'] = start_date  # Record the start of the window
        roll_preds.append(merged)

        # Move the rolling window forward by the test window length
        start_date += test_window

    return pd.concat(roll_preds)


def final_test_evaluation(train_path, test_path):
    """
    Fit the model on the entire training set and forecast for the final test set.
    Then merge the forecasts with the test data and compute overall WMAPE.
    """
    # Load training and test data
    train = pd.read_csv(train_path, parse_dates=['ds'])
    test = pd.read_csv(test_path, parse_dates=['ds'])

    # Forecast horizon is number of unique test dates
    h = test['ds'].nunique()

    # Fit model on all training data
    model = StatsForecast(models=[SeasonalNaive(
        season_length=4)], freq='QE', n_jobs=-1)
    model.fit(train)

    # Generate forecasts for the final test set
    p = model.predict(h=h, level=[90])
    p_reset = p.reset_index()

    # Merge forecasts with test data
    final_results = p_reset.merge(test, on=['ds', 'unique_id'], how='left')
    final_results = final_results.dropna(subset=['y'])

    overall_final_wmape = wmape(
        final_results['y'], final_results['SeasonalNaive'])
    print("Final Test Overall WMAPE:", overall_final_wmape)

    return final_results


def main():
    # File paths for your cleaned data in long format
    train_path = '/Users/mahfuz/Final_project/Final_repo/long_data.csv'
    test_path = '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv'

    # --- Rolling Validation ---
    # Load training data and convert ds to quarter-end
    train = pd.read_csv(train_path, parse_dates=['ds'])
    train['ds'] = train['ds'] + pd.offsets.QuarterEnd()

    # Define rolling window lengths (e.g., 5 years training and 1 year testing)
    train_window = pd.DateOffset(years=5)
    test_window = pd.DateOffset(years=3)

    roll_results = rolling_validation(train, train_window, test_window)
    roll_results = roll_results.dropna(subset=['y'])

    overall_roll_wmape = wmape(
        roll_results['y'], roll_results['SeasonalNaive'])
    print("Overall Rolling WMAPE:", overall_roll_wmape)

    # Optionally, plot WMAPE over time for rolling windows
    wmape_by_date = roll_results.groupby('ds')[['y', 'SeasonalNaive']].apply(
        lambda x: wmape(x['y'], x['SeasonalNaive']))
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
    plt.plot(series_data['ds'], series_data['SeasonalNaive'],
             label='Forecast (Seasonal Naïve)', marker='x')
    if 'SeasonalNaive-lo-90' in series_data.columns and 'SeasonalNaive-hi-90' in series_data.columns:
        plt.fill_between(series_data['ds'],
                         series_data['SeasonalNaive-lo-90'],
                         series_data['SeasonalNaive-hi-90'],
                         color='gray', alpha=0.2, label='90% CI')
    plt.title(
        f"Seasonal Naïve Forecast vs Actual for {series_id} (Final Test)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
