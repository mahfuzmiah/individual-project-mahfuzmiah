from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


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
    data_long_train['ds'] = data_long_train['ds'] + pd.offsets.QuarterEnd()

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
