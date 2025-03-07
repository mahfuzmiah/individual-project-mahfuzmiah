
from statsforecast import StatsForecast
from statsforecast.models import Naive
import pandas as pd
import matplotlib.pyplot as plt


def wmape(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / abs(y_true).sum()


def main():
    # --- Process Training Data ---
    path_train = '/Users/mahfuz/Final_project/Final_repo/DatasetsCBS/imputed_linear.csv'
    data_training = pd.read_csv(path_train)
    # Reshape from wide to long format
    data_long_train = data_training.melt(
        id_vars=['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS'],
        var_name='ds',
        value_name='y'
    )
    # Create unique_id
    data_long_train['unique_id'] = (
        data_long_train['L_REP_CTY'] + '_' +
        data_long_train['L_CP_COUNTRY'] + '_' +
        data_long_train['CBS_BASIS']
    )
    # Drop unnecessary columns
    data_long_train = data_long_train.drop(
        columns=['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
    )
    # Convert 'ds' from 'YYYY-QX' to datetime
    data_long_train['ds'] = pd.to_datetime(
        data_long_train['ds'].str[:4] + '-' +
        data_long_train['ds'].str[-2:].replace({
            'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'
        }) + '-01'
    )
    # Reorder columns
    data_long_train = data_long_train[['ds', 'unique_id', 'y']]
    # Save cleaned training data (optional)
    data_long_train.to_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data.csv', index=False)

    # --- Process Test Data ---
    path_test = '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/TestingData.csv'
    data_testing = pd.read_csv(path_test)
    data_long_test = data_testing.melt(
        id_vars=['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS'],
        var_name='ds',
        value_name='y'
    )
    data_long_test['unique_id'] = (
        data_long_test['L_REP_CTY'] + '_' +
        data_long_test['L_CP_COUNTRY'] + '_' +
        data_long_test['CBS_BASIS']
    )
    data_long_test = data_long_test.drop(
        columns=['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
    )
    data_long_test['ds'] = pd.to_datetime(
        data_long_test['ds'].str[:4] + '-' +
        data_long_test['ds'].str[-2:].replace({
            'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'
        }) + '-01'
    )
    # **Convert test dates to quarter-end dates**
    data_long_test['ds'] = data_long_test['ds'] + pd.offsets.QuarterEnd()
    data_long_test = data_long_test[['ds', 'unique_id', 'y']]
    # Save cleaned test data (optional)
    data_long_test.to_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv', index=False)

    # --- Load Cleaned Data ---
    train = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data.csv', parse_dates=['ds'])
    test = pd.read_csv(
        '/Users/mahfuz/Final_project/Final_repo/long_data_test.csv', parse_dates=['ds'])

    # Define forecast horizon h based on the testing set
    h = test['ds'].nunique()

    # Instantiate and fit the model on the full training data
    model = StatsForecast(models=[Naive()], freq='QE', n_jobs=1)
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
    # Overall WMAPE across all series
    overall_wmape = wmape(results['y'], results['Naive'])
    print("Overall WMAPE:", overall_wmape)

    # Or compute per unique_id
    wmape_by_series = results.groupby('unique_id')[['y', 'Naive']].apply(
        lambda x: wmape(x['y'], x['Naive'])
    )

    series_id = 'AT_AL_Q'
    series_data = results[results['unique_id'] == series_id]

    # Group results by forecast date 'ds' and compute WMAPE for each period
    wmape_by_date = results.groupby('ds')[['y', 'Naive']].apply(
        lambda x: wmape(x['y'], x['Naive'])
    )
    plt.figure(figsize=(10, 5))
    plt.plot(series_data['ds'], series_data['y'], label='Actual', marker='o')
    plt.plot(series_data['ds'], series_data['Naive'],
             label='Forecast (Naive)', marker='x')
    plt.fill_between(series_data['ds'],
                     series_data['Naive-lo-90'],
                     series_data['Naive-hi-90'],
                     color='gray', alpha=0.2, label='90% CI')
    plt.title(f"Naive Forecast vs Actual for {series_id}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Diagrams/Naive_Results_diagrams/NaiveForecast_AT_AL_Q.png')
    plt.show()  # Show the plot after saving

    # Plot WMAPE over time
    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.xlabel('Forecast Date')
    plt.ylabel('WMAPE')
    plt.title('WMAPE Over Time')
    plt.grid(True)
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Diagrams/Naive_Results_diagrams/WMAPE_over_time.png')
    plt.show()

    # Plot for series 'US_ZW_U'
    series_id = 'US_ZW_U'
    series_data = results[results['unique_id'] == series_id]
    plt.figure(figsize=(10, 5))
    plt.plot(series_data['ds'], series_data['y'], label='Actual', marker='o')
    plt.plot(series_data['ds'], series_data['Naive'],
             label='Forecast (Naive)', marker='x')
    plt.fill_between(series_data['ds'],
                     series_data['Naive-lo-90'],
                     series_data['Naive-hi-90'],
                     color='gray', alpha=0.2, label='90% CI')
    plt.title(f"Naive Forecast vs Actual for {series_id}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Diagrams/Naive_Results_diagrams/NaiveForecast_US_ZW_U.png')
    plt.show()


if __name__ == '__main__':
    main()
