from pathlib import Path

# Project root directory
REPO_ROOT = Path(__file__).resolve().parent

# Directory containing your raw datasets
DATASETS_DIR = REPO_ROOT / "DataSetsCBS"

# Directories for your imputed train/test CSVs
IMPUTED_RESULTS_DIR_TRAIN = REPO_ROOT / "imputed_results" / "train"
IMPUTED_RESULTS_DIR_TEST = REPO_ROOT / "imputed_results" / "test"

# Where to save or load diagrams
DIAGRAMS_DIR = REPO_ROOT / "Diagrams"

LONG_DATA_CSV = DATASETS_DIR / "Train_long.csv"
LONG_DATA_TEST_CSV = DATASETS_DIR / "Test_long.csv"

PREDICTION_DIR = REPO_ROOT / "predictions"
ARIMA_DIR = PREDICTION_DIR / "arima"
LIGHTGBM_DIR = PREDICTION_DIR / "lightgbm"
LIGHTGBM_ITER_DIR = PREDICTION_DIR / "lightgbm_iterative"
LSTM_DIR = PREDICTION_DIR / "lstm"
STATS_DIR = PREDICTION_DIR / "statistics"
SeasonalWindows_DIR = STATS_DIR / "seasonal_window_average"
PREDICTION_FILES = {
    # ARIMA
    "arima":                  ARIMA_DIR / "arima_parallel_predictions.csv",

    # LightGBM
    "lightgbm":               LIGHTGBM_DIR / "lightgbm_V4_test_predictions.csv",
    "lightgbm_iterative":     LIGHTGBM_ITER_DIR / "iterative_forecasts.csv",

    # Naïve / statistical
    "naive":                  STATS_DIR / "naive_predictions.csv",
    "seasonal_naive":         STATS_DIR / "seasonalnaive_final_preds.csv",
    "seasonal_window_average": SeasonalWindows_DIR / "seasonalwa_final_preds.csv",
    # LSTM1 variants
    "lstm1_2epochs":          LSTM_DIR / "lstm1_predictions_2epochs.csv",
    "lstm1_3epochs":          LSTM_DIR / "lstm1_predictions_3epochs.csv",
    "lstm1_5epochs":          LSTM_DIR / "lstm1_predictions_5epochs.csv",
    "lstm1_50epochs":         LSTM_DIR / "lstm1_predictions_50epochs.csv",
    "lstm1_100epochs":        LSTM_DIR / "lstm1_predictions_100epochs.csv",
    "lstm1_200epochs":        LSTM_DIR / "lstm1_predictions_200epochs.csv",
    "lstm1_500epochs":        LSTM_DIR / "lstm1_predictions_500epochs.csv",

    # LSTM2 variants
    "lstm2_50epochs":         LSTM_DIR / "lstm2_predictions_50epochs.csv",
    "lstm2_100epochs":        LSTM_DIR / "lstm2_predictions_100epochs.csv",
    "lstm2_200epochs":        LSTM_DIR / "lstm2_predictions_200epochs.csv",
    "lstm2_500epochs":        LSTM_DIR / "lstm2_predictions_500epochs.csv",

    # Vectorized LSTM
    "vector_lstm_50epochs":   LSTM_DIR / "vector_lstm_preds_50epochs.csv",
    "vector_lstm_100epochs":  LSTM_DIR / "vector_lstm_preds_100epochs.csv",
    "vector_lstm_200epochs":  LSTM_DIR / "vector_lstm_preds_200epochs.csv",
    "vector_lstm_500epochs":  LSTM_DIR / "vector_lstm_preds_500epochs.csv",

    # Blocked LSTMs (4, 8, 20, 40 features × 100/200/500 epochs)
    "lstm_blocks_4feat_100ep":  LSTM_DIR / "lstm_blocks_4feat_100ep.csv",
    "lstm_blocks_4feat_200ep":  LSTM_DIR / "lstm_blocks_4feat_200ep.csv",
    "lstm_blocks_4feat_500ep":  LSTM_DIR / "lstm_blocks_4feat_500ep.csv",
    "lstm_blocks_8feat_100ep":  LSTM_DIR / "lstm_blocks_8feat_100ep.csv",
    "lstm_blocks_8feat_200ep":  LSTM_DIR / "lstm_blocks_8feat_200ep.csv",
    "lstm_blocks_8feat_500ep":  LSTM_DIR / "lstm_blocks_8feat_500ep.csv",
    "lstm_blocks_20feat_100ep": LSTM_DIR / "lstm_blocks_20feat_100ep.csv",
    "lstm_blocks_20feat_200ep": LSTM_DIR / "lstm_blocks_20feat_200ep.csv",
    "lstm_blocks_20feat_500ep": LSTM_DIR / "lstm_blocks_20feat_500ep.csv",
    "lstm_blocks_40feat_100ep": LSTM_DIR / "lstm_blocks_40feat_100ep.csv",
    "lstm_blocks_40feat_200ep": LSTM_DIR / "lstm_blocks_40feat_200ep.csv",
    "lstm_blocks_40feat_500ep": LSTM_DIR / "lstm_blocks_40feat_500ep.csv",
}

MODEL_TO_READER = {
    # must match READERS keys exactly:
    "arima":                   "arima",
    "lightgbm":                "lightgbm",
    "lightgbm_iterative":      "lightgbm_iterative",
    "naive":                   "naive",
    "seasonal_naive":          "seasonal_naive",
    "seasonal_window_average": "seasonal_wa",

    # all the various LSTM variants should point to the single 'lstm' reader
    **{key: "lstm" for key in [
        "lstm1_2epochs", "lstm1_3epochs", "lstm1_5epochs",
        "lstm1_50epochs", "lstm1_100epochs", "lstm1_200epochs", "lstm1_500epochs",
        "lstm2_50epochs", "lstm2_100epochs", "lstm2_200epochs", "lstm2_500epochs",
        "vector_lstm_50epochs", "vector_lstm_100epochs", "vector_lstm_200epochs", "vector_lstm_500epochs",
        "lstm_blocks_4feat_100ep", "lstm_blocks_4feat_200ep", "lstm_blocks_4feat_500ep",
        "lstm_blocks_8feat_100ep", "lstm_blocks_8feat_200ep", "lstm_blocks_8feat_500ep",
        "lstm_blocks_20feat_100ep", "lstm_blocks_20feat_200ep", "lstm_blocks_20feat_500ep",
        "lstm_blocks_40feat_100ep", "lstm_blocks_40feat_200ep", "lstm_blocks_40feat_500ep"
    ]}
}
