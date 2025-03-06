import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from fancyimpute import IterativeSVD

# 1. Baseline: Fill with zeros


def impute_with_zeros(filepath):
    df = pd.read_csv(filepath)
    df_filled = df.fillna(0)
    return df_filled

# 2. Forward Fill (and Backward Fill as backup)


def impute_forward_fill(filepath):
    df = pd.read_csv(filepath)

    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Create a copy to avoid modifying original DataFrame
    df_filled = df.copy()

    # Fill missing values only in numerical columns
    df_filled[numeric_cols] = df_filled[numeric_cols].ffill(axis=1)
    df_filled[numeric_cols] = df_filled[numeric_cols].bfill(axis=1)

    return df_filled

# 3. Linear Interpolation


def impute_linear_interpolation(filepath):
    df = pd.read_csv(filepath)

    # Convert applicable columns to numeric types explicitly
    df = df.infer_objects(copy=False)

    # Select numerical columns separately
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Create a copy of the original DataFrame to retain non-numeric data
    df_filled = df.copy()

    # Interpolate only numerical columns using linear interpolation along rows (axis=1)
    df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(
        method='linear',
        axis=1,
        limit_direction='both'
    )

    # Fill any remaining NaN values with 0
    df_filled = df_filled.fillna(0)

    return df_filled


# 4. Polynomial Interpolation


def impute_polynomial_interpolation(filepath, order=3):
  # Read in the data
    df = pd.read_csv(filepath)

    # Identify the quarterly columns
    quarterly_columns = [col for col in df.columns if 'Q' in col]

    # Ensure quarterly columns are numeric
    df[quarterly_columns] = df[quarterly_columns].apply(
        pd.to_numeric, errors='coerce')

    for i in range(len(df)):
        # Extract the row values for the quarterly columns
        row_values = df.loc[i, quarterly_columns]
        # Convert objects to numeric dtypes
        row_values = row_values.infer_objects()

        nan_mask = row_values.isna()
        # Proceed only if there are missing values, but not if all are missing
        if nan_mask.sum() > 0 and nan_mask.sum() < len(row_values):
            # Create a numeric x-axis for the quarters (0, 1, 2, ...)
            x_all = np.arange(len(row_values)).astype(float)
            # Extract valid indices and values, ensuring they are floats
            valid_x = x_all[~nan_mask]
            valid_y = np.array(row_values[~nan_mask], dtype=float)

            # If we have enough points to fit the chosen polynomial order, use polynomial interpolation
            if len(valid_y) >= order + 1:
                try:
                    poly_coeffs = np.polyfit(valid_x, valid_y, order)
                    poly_func = np.poly1d(poly_coeffs)
                    missing_x = x_all[nan_mask]
                    row_values[nan_mask] = poly_func(missing_x)
                except np.linalg.LinAlgError:
                    # Fallback to linear interpolation if polynomial fitting fails
                    row_values = row_values.interpolate(
                        method="linear", limit_direction='both')
            else:
                # Fallback to linear interpolation if not enough data points
                row_values = row_values.interpolate(
                    method="linear", limit_direction='both')

        # Assign the interpolated values back to the DataFrame
        df.loc[i, quarterly_columns] = row_values

    return df


# 5. KNN Imputation using scikit-learn


def impute_knn(filepath, n_neighbors=5):
    df = pd.read_csv(filepath)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

# 6. Iterative Imputer (ML-based, similar to MICE)


def impute_iterative(filepath):
    df = pd.read_csv(filepath)
    imputer = IterativeImputer(random_state=0)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

# 7. SVD-based Imputation (a form of matrix factorization, available via fancyimpute)


def impute_iterative_svd(filepath, rank=10):
    df = pd.read_csv(filepath)
    # IterativeSVD expects a numpy array and returns an imputed array.
    imputer = IterativeSVD(rank=rank)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

# 8. (Placeholder) Deep Learning-Based Imputation (Autoencoder/GAN)
# This is more complex and typically requires TensorFlow or PyTorch.
# Here is a simple placeholder function.


def impute_deep_learning(filepath):
    df = pd.read_csv(filepath)
    # You would implement a deep learning model here (e.g., autoencoder) to impute missing values.
    # For now, we simply return the original dataframe.
    # TODO: Implement deep learning imputation.
    return df


# Example usage:
if __name__ == '__main__':
    filepath = '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/TrainingData.csv'
    # df0 = impute_with_zeros(filepath)
    # df_ff = impute_forward_fill(filepath)
    df_lin = impute_linear_interpolation(filepath)
    # df_poly = impute_polynomial_interpolation(filepath, order=3)
    # df_knn = impute_knn(filepath, n_neighbors=5)
    # df_iter = impute_iterative(filepath)
    # df_svd = impute_iterative_svd(filepath, rank=10)
    # df_dl = impute_deep_learning(filepath)

    # Save imputed datasets for inspection:
    # df0.to_csv('imputed_zeros.csv', index=False)
    # df_ff.to_csv(
    #     '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/imputed_forward_fill.csv', index=False)
    df_lin.to_csv(
        '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/imputed_linear.csv', index=False)
    # df_poly.to_csv(
    # '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/imputed_poly.csv', index=False)
    # df_knn.to_csv('imputed_knn.csv', index=False)
    # df_iter.to_csv('imputed_iterative.csv', index=False)
    # df_svd.to_csv('imputed_svd.csv', index=False)
    # df_dl.to_csv('imputed_deep_learning.csv', index=False)
