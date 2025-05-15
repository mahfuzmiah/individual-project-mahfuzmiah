import pandas as pd
from pathlib import Path
import argparse
import re
from reader import READERS
from config import PREDICTION_FILES, MODEL_TO_READER, LONG_DATA_TEST_CSV
from metrics import compute_metrics
from gather_runtimes import gather_runtimes
from plots import (
    plot_metric_vs_horizon,
    plot_accuracy_speed,
    plot_lstm_training_times,
    plot_lstm_epoch_runtimes,
    plot_lstm_epoch_accuracy,
    plot_lightgbm_accuracy_runtime,
    plot_lstm_variant_tradeoff,
    plot_lstm_hybrid_by_block,
    plot_all_hybrid_metrics
)


def main(pred_dir: Path):
    all_preds = {}
    for model_name, rel_path in PREDICTION_FILES.items():
        filepath = pred_dir / rel_path.relative_to(pred_dir)

        # Explicit reader lookup
        reader_key = MODEL_TO_READER.get(model_name)
        if reader_key is None:
            raise KeyError(f"No reader mapping for model '{model_name}'")

        df = READERS[reader_key](filepath)
        all_preds[model_name] = df

    # 2. Optionally inspect
    for name, df in all_preds.items():
        print(f"{name}: {df.shape[0]} rows, columns={list(df.columns)}")

    # 3. Stack into one big DataFrame with a 'model' column
    big_df = pd.concat(
        [df.assign(model=name) for name, df in all_preds.items()],
        ignore_index=True
    )

    # 3.5 Ensure rows are sorted by series & time
    test_meta = pd.read_csv(LONG_DATA_TEST_CSV, parse_dates=['ds'])
    cutoff = test_meta['ds'].min()

    # restrict to forecasts (i.e. rows on or after your first test date)
    forecast_df = big_df[big_df['ds'] >= cutoff].copy()

    # now horizon=1 means 1-quarter ahead, 2 means 2-ahead, etc.
    forecast_df['horizon'] = (
        forecast_df.groupby('unique_id')
                   .cumcount() + 1
    )

    # compute metrics only on those filtered horizons
    metrics_df = compute_metrics(forecast_df)
    metrics_df['step'] = metrics_df['horizon']

    # 5. Gather runtimes from your `predictions/…/runtimes` files
    runtime_df = gather_runtimes(pred_dir)
    runtime_df['total_time_s'] = pd.to_numeric(
        runtime_df['total_time_s'], errors='coerce')

    # # 6. Produce your standard suite of plots
    plot_metric_vs_horizon(metrics_df, 'RMSE', log_scale=True)
    plot_metric_vs_horizon(metrics_df, 'SMAPE')
    plot_lstm_epoch_runtimes(runtime_df)
    plot_lstm_epoch_accuracy(metrics_df, metric='WMAPE')
    plot_lightgbm_accuracy_runtime(metrics_df, runtime_df,
                                   metrics=['RMSE', 'SMAPE', 'WMAPE'])
    plot_lstm_variant_tradeoff(metrics_df, runtime_df, metric='WMAPE')
    plot_accuracy_speed(runtime_df, metrics_df)
    plot_lstm_training_times(runtime_df)
    for m in ("RMSE", "SMAPE", "WMAPE", "Log-RMSE"):
        plot_lstm_hybrid_by_block(metrics_df, metric=m)
    plot_all_hybrid_metrics(metrics_df)
    # 7. Summarise mean performance per model
    summary = (
        metrics_df
        .groupby('model')
        .agg(
            RMSE_mean=('RMSE',    'mean'),
            SMAPE_mean=('SMAPE',   'mean'),
            WMAPE_mean=('WMAPE',   'mean'),
            LogRMSE_mean=('Log-RMSE', 'mean'),
        )
        .reset_index()
    )
    summary.to_csv(
        pred_dir / 'summary.csv',)

    # 1) define a categorisation function
    def categorize(model):
        if model == 'arima':
            return 'ARIMA'
        if model == 'lightgbm':
            return 'LightGBM_Global'
        if model == 'lightgbm_iterative':
            return 'LightGBM_Iterative'
        if model == 'naive':
            return 'Naïve'
        if model == 'seasonal_naive':
            return 'Seasonal_Naive'
        if model == 'seasonal_window_average':
            return 'Seasonal_Window_Avg'
        if model.startswith('lstm1_'):
            return 'LSTM_Series1'
        if model.startswith('lstm2_'):
            return 'LSTM_Series2'
        if model.startswith('vector_lstm_'):
            return 'LSTM_Vectorized'
        m = re.match(r'lstm_blocks_(\d+)feat', model)
        if m:
            return f"LSTM_Hybrid_{m.group(1)}feat"
        return 'Other'

    # 2) annotate categories
    summary['Category'] = summary['model'].apply(categorize)

    # 3) for each metric, pick the best model per category
    metrics = ['RMSE_mean', 'SMAPE_mean', 'WMAPE_mean', 'LogRMSE_mean']
    best_per_cat = {}
    for met in metrics:
        best = summary.loc[summary.groupby('Category')[met].idxmin(), [
            'Category', 'model', met]]
        best_per_cat[met] = best

    # 4) show it
    print("🏆  Best model in each category per metric:")
    for met, df in best_per_cat.items():
        print(f"\n  — {met} —")
        print(df.to_string(index=False))

    # 5) save the full summary for later
    summary.to_csv('all_models_summary.csv', index=False)
    print("\nFull summary written to all_models_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred-dir',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'predictions',
        help='Root folder where all model sub‐dirs live'
    )
    args = parser.parse_args()
    main(args.pred_dir)
