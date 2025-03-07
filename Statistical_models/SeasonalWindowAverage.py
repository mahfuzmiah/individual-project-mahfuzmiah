
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage, WindowAverage
from statsforecast.utils import ConformalIntervals
import pandas as pd
import matplotlib.pyplot as plt


def wmape(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / abs(y_true).sum()


def main():
    # --- Load Cleaned Data ---
    train = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data.csv', parse_dates=['ds'])
    test = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv', parse_dates=['ds'])

    # Define forecast horizon h based on the testing set
    h = test['ds'].nunique()
    conf_int = ConformalIntervals()

    # Instantiate the model using SeasonalWindowAverage with proper prediction_intervals.
    model = StatsForecast(
        models=[SeasonalWindowAverage(
            season_length=4, window_size=3, prediction_intervals=conf_int)],
        freq='QE',  # Quarterly (end) frequency
        n_jobs=-1
    )
    model.fit(train)

    # Generate forecasts for the testing horizon
    p = model.predict(h=h, level=[90])

    # Reset index to get 'ds' as a column and extract forecast dates
    p_reset = p.reset_index()
    forecast_dates = p_reset['ds'].unique()

    # Filter test set to only include dates that match the forecast dates
    test_filtered = test[test['ds'].isin(forecast_dates)]

    # Merge forecasts with actual test data for evaluation and drop rows with NaNs in actual y
    results = p_reset.merge(test_filtered, on=['ds', 'unique_id'], how='left')
    results = results.dropna(subset=['y'])

    # Overall WMAPE across all series (using Seasonal Naïve forecast)
    overall_wmape = wmape(results['y'], results['SeasWA'])
    print("Overall WMAPE:", overall_wmape)

    # Compute WMAPE per unique_id
    wmape_by_series = results.groupby('unique_id')[['y', 'SeasWA']].apply(
        lambda x: wmape(x['y'], x['SeasWA'])
    )

    # Plot Seasonal Naïve Forecast vs Actual for a specific series
    series_id = 'AT_AL_Q'
    series_data = results[results['unique_id'] == series_id]

    plt.figure(figsize=(10, 5))
    plt.plot(series_data['ds'], series_data['y'], label='Actual', marker='o')
    plt.plot(series_data['ds'], series_data['SeasWA'],
             label='Forecast (Seasonal Naïve)', marker='x')
    plt.fill_between(series_data['ds'],
                     series_data['SeasWA-lo-90'],
                     series_data['SeasWA-hi-90'],
                     color='gray', alpha=0.2, label='90% CI')
    plt.title(f"Seasonal Naïve Forecast vs Actual for {series_id}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Diagrams/SeasWa_Forecast_Diagrams/SeasonalNaiveForecast_AT_AL_Q.png')
    plt.show()

    # Plot WMAPE over time
    wmape_by_date = results.groupby('ds')[['y', 'SeasWA']].apply(
        lambda x: wmape(x['y'], x['SeasWA'])
    )
    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.xlabel('Forecast Date')
    plt.ylabel('WMAPE')
    plt.title('WMAPE Over Time')
    plt.grid(True)
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Diagrams/SeasWa_Forecast_Diagrams/WMAPE_over_time.png')
    plt.show()

    # Plot for series 'US_ZW_U'
    series_id = 'US_ZW_U'
    series_data = results[results['unique_id'] == series_id]
    plt.figure(figsize=(10, 5))
    plt.plot(series_data['ds'], series_data['y'], label='Actual', marker='o')
    plt.plot(series_data['ds'], series_data['SeasWA'],
             label='Forecast (Seasonal Naïve)', marker='x')
    plt.fill_between(series_data['ds'],
                     series_data['SeasWA-lo-90'],
                     series_data['SeasWA-hi-90'],
                     color='gray', alpha=0.2, label='90% CI')
    plt.title(f"Seasonal Naïve Forecast vs Actual for {series_id}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Diagrams/SeasWa_Forecast_Diagrams/SeasonalNaiveForecast_US_ZW_U.png')
    plt.show()


if __name__ == '__main__':
    main()
