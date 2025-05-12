import sys
from pathlib import Path
from statsforecast import StatsForecast
from statsforecast.models import Naive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(42)
# Insert repo root onto sys.path so that config can be found:
REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_PATH))

from config import (
    LONG_DATA_CSV,
    LONG_DATA_TEST_CSV,
    DIAGRAMS_DIR,
)  # nopep8


PRED_STATS_DIR = REPO_ROOT_PATH / "predictions" / "statistics"
PRED_STATS_DIR.mkdir(parents=True, exist_ok=True)
out_dir = DIAGRAMS_DIR / "Naive_Results_diagrams"
out_dir.mkdir(parents=True, exist_ok=True)


def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    if denom == 0:
        # no true volume → define WMAPE as NaN (or 0, if you prefer)
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom


def main():

    # --- Load Cleaned Data ---
    train = pd.read_csv(
        LONG_DATA_CSV, parse_dates=['ds'])
    test = pd.read_csv(
        LONG_DATA_TEST_CSV, parse_dates=['ds'])

    # Define forecast horizon h based on the testing set
    h = test['ds'].nunique()

    # Instantiate and fit the model on the full training data
    t0 = time.time()

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
    total_sec = time.time() - t0
    n_series = results['unique_id'].nunique()
    avg_sec = total_sec / n_series

    # 2) write a small runtime CSV
    runtime_df = pd.DataFrame([{
        'model': 'Naive',
        'total_time_s': round(total_sec, 2),
        'avg_time_per_series_s': round(avg_sec, 4)
    }])
    runtime_out = PRED_STATS_DIR/"naive_runtime.csv"
    runtime_df.to_csv(runtime_out, index=False)
    print(f"Wrote runtime summary to {runtime_out}")

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

    plt.savefig(out_dir / f"NaiveForecast_{series_id}.png", dpi=300)
    plt.show()  # Show the plot after saving

    # Plot WMAPE over time
    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.xlabel('Forecast Date')
    plt.ylabel('WMAPE')
    plt.title('WMAPE Over Time')
    plt.grid(True)
    plt.savefig(out_dir / "WMAPE_over_time.png", dpi=300)
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
        out_dir / f"NaiveForecast_{series_id}.png", dpi=300)
    plt.show()
    preds_df = (
        results
        .rename(columns={'Naive': 'forecast', 'y': 'actual'})
        [['unique_id', 'ds', 'actual', 'forecast']]
    )
    preds_out = PRED_STATS_DIR / "naive_predictions.csv"
    preds_df.to_csv(preds_out, index=False)
    print(f"Wrote predictions to {preds_out}")

    # 2) write overall + per-series summary
    summary_rows = []

    # overall
    summary_rows.append({
        'unique_id': 'ALL',
        'wmape': overall_wmape
    })

    # per-series
    for uid, group in results.groupby('unique_id'):
        summary_rows.append({
            'unique_id': uid,
            'wmape': wmape(group['y'], group['Naive'])
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_out = PRED_STATS_DIR / "naive_wmape_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Wrote WMAPE summary to {summary_out}")


if __name__ == '__main__':
    np.random.seed(42)
    main()
