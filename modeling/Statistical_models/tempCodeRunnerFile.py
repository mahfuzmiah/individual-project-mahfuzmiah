
#!/usr/bin/env python

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import SeasonalWindowAverage
from statsforecast.utils import ConformalIntervals

# make config.py importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import LONG_DATA_CSV, LONG_DATA_TEST_CSV, DIAGRAMS_DIR  # nopep8


# new: prediction output dir
PRED_STATS_DIR = REPO_ROOT / "predictions" / \
    "statistics" / "seasonal_window_average"
PRED_STATS_DIR.mkdir(parents=True, exist_ok=True)
# Make Diarams directory

# reproducible “randomness”
np.random.seed(42)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE, returns NaN when sum(abs(y_true)) == 0."""
    denom = np.abs(y_true).sum()
    return np.nan if denom == 0 else np.abs(y_true - y_pred).sum() / denom


def rolling_validation(
    data: pd.DataFrame,
    train_window: pd.DateOffset,
    test_window: pd.DateOffset
) -> pd.DataFrame:
    """
    Sliding‐window validation:
      • start at min(ds)
      • fit SeasonalWindowAverage on [start, start+train_window)
      • forecast next h periods where h = number of unique ds in test slice
      • merge forecasts with truth
      • advance start by test_window
      • repeat until you run out of data
    """
    results = []
    start, last = data['ds'].min(), data['ds'].max()

    while start + train_window + test_window <= last:
        train_slice = data[(data['ds'] >= start) &
                           (data['ds'] < start + train_window)]
        valid_slice = data[(data['ds'] >= start + train_window) &
                           (data['ds'] < start + train_window + test_window)]
        h = valid_slice['ds'].nunique()

        # set up conformal intervals for PI
        conf = ConformalIntervals()

        model = StatsForecast(
            models=[SeasonalWindowAverage(
                season_length=4,
                window_size=3,
                prediction_intervals=conf
            )],
            freq='QE',
            n_jobs=1,
        )
        model.fit(train_slice)
        preds = model.predict(h=h, level=[90]).reset_index()

        merged = preds.merge(valid_slice, on=['ds', 'unique_id'], how='left')
        merged['roll_start'] = start
        results.append(merged)

        start += test_window

    return pd.concat(results, ignore_index=True)


def final_test_evaluation(train_path: Path, test_path: Path) -> pd.DataFrame:
    """
    Fit SeasonalWindowAverage on the entire training set and
    forecast the final test period, then return merged results.
    """
    train = pd.read_csv(train_path, parse_dates=['ds'])
    test = pd.read_csv(test_path,  parse_dates=['ds'])
    h = test['ds'].nunique()

    conf = ConformalIntervals()
    model = StatsForecast(
        models=[SeasonalWindowAverage(
            season_length=4, window_size=3, prediction_intervals=conf
        )],
        freq='QE',
        n_jobs=1,
    )
    model.fit(train)
    preds = model.predict(h=h, level=[90]).reset_index()

    final = preds.merge(test, on=['ds', 'unique_id'],
                        how='left').dropna(subset=['y'])
    print("Final Test Overall WMAPE:", wmape(final['y'], final['SeasWA']))
    return final


def main():
    # — Load & align data —
    train = pd.read_csv(LONG_DATA_CSV, parse_dates=['ds'])
    train['ds'] += pd.offsets.QuarterEnd()
    test = pd.read_csv(LONG_DATA_TEST_CSV, parse_dates=['ds'])
    test['ds'] += pd.offsets.QuarterEnd()

    # — Rolling validation —
    t0 = time.time()
    roll = rolling_validation(train,
                              train_window=pd.DateOffset(years=5),
                              test_window=pd.DateOffset(years=1))
    roll_time = time.time() - t0

    roll = roll.dropna(subset=['y'])
    roll_out = PRED_STATS_DIR / "seasonalwa_rolling_preds.csv"
    roll.to_csv(roll_out, index=False)
    print(f"Wrote rolling preds to {roll_out} (took {roll_time:.2f}s)")

    # — Final test evaluation —
    t1 = time.time()
    final = final_test_evaluation(LONG_DATA_CSV, LONG_DATA_TEST_CSV)
    final_time = time.time() - t1

    final_out = PRED_STATS_DIR / "seasonalwa_final_preds.csv"
    final.to_csv(final_out, index=False)
    print(f"Wrote final   preds to {final_out} (took {final_time:.2f}s)")

    # — Timing summary —
    timings = pd.DataFrame([
        {'stage': 'rolling_validation', 'total_time_s': round(roll_time, 2)},
        {'stage': 'final_test',         'total_time_s': round(final_time, 2)},
    ])
    timing_out = PRED_STATS_DIR / "seasonalwa_timing_summary.csv"
    timings.to_csv(timing_out, index=False)
    print(f"Wrote timing summary to {timing_out}")

    # plot WMAPE over time
    wmape_by_date = roll.groupby('ds').apply(
        lambda g: wmape(g['y'], g['SeasWA'])
    )
    out_dir = DIAGRAMS_DIR / "SeasonalWA_Results"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.title("Seasonal Window Average WMAPE (Rolling Validation)")
    plt.xlabel("Date")
    plt.ylabel("WMAPE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "Rolling_WMAPE_over_time.png", dpi=300)
    plt.close()

    # — Final test evaluation —
    final = final_test_evaluation(LONG_DATA_CSV, LONG_DATA_TEST_CSV)

    # example series plots
    for sid in ["AT_AL_Q", "US_ZW_U"]:
        df_s = final[final['unique_id'] == sid]
        plt.figure(figsize=(10, 5))
        plt.plot(df_s['ds'], df_s['y'],           label='Actual', marker='o')
        plt.plot(df_s['ds'], df_s['SeasWA'],      label='Forecast', marker='x')
        if 'SeasWA-lo-90' in df_s.columns:
            plt.fill_between(
                df_s['ds'],
                df_s['SeasWA-lo-90'], df_s['SeasWA-hi-90'],
                color='gray', alpha=0.2, label='90% CI'
            )
        plt.title(f"Seasonal WA Forecast vs Actual: {sid}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"SeasonalWAForecast_{sid}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
