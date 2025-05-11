# config.py (placed alongside requirements.txt)

from pathlib import Path


# Root of the repository
REPO_ROOT = Path(__file__).resolve().parent

# Top-level directories
DATASETS_DIR = REPO_ROOT / "DataSetsCBS"
DATA_PREP_DIR = REPO_ROOT / "Data_preparation"
DATA_UNDERSTAND_DIR = REPO_ROOT / "Data_understanding"
DIAGRAMS_DIR = REPO_ROOT / "Diagrams"
EVALUATION_DIR = REPO_ROOT / "evaluation"
GRAPHS_DIR = REPO_ROOT / "graphs"
IMPUTED_RESULTS_DIR_TRAIN = REPO_ROOT / "imputed_results" / "train"
IMPUTED_RESULTS_DIR_TEST = REPO_ROOT / "imputed_results" / "test"
MODELING_DIR = REPO_ROOT / "modeling"

# Modeling subdirectories
DEEP_LEARNING_DIR = MODELING_DIR / "Deep_Learning"
MACHINE_LEARNING_DIR = MODELING_DIR / "Machine_leaning_models"
STATISTICAL_MODELS_DIR = MODELING_DIR / "Statistical_models"
MODELS_DIR = REPO_ROOT / "models"

# Top-level CSV files
LONG_DATA_CSV = DATASETS_DIR / "long_data.csv"
LONG_DATA_TEST_CSV = DATASETS_DIR / "long_data_test.csv"
METRICS_TRAIN_CSV = REPO_ROOT / "metrics_train.csv"
METRICS_TEST_CSV = REPO_ROOT / "metrics_test.csv"
IMPUTE_EVAL_TRAIN_CSV = REPO_ROOT / "imputation_evaluation_metrics_train.csv"
IMPUTE_EVAL_TEST_CSV = REPO_ROOT / "imputation_evaluation_metrics_test.csv"

# DataSetsCBS files
RAW_CSV = DATASETS_DIR / "WS_CBS_PUB_csv_col.csv"
CLEANED_CSV = DATASETS_DIR / "CleanedCBSDataSet.csv"
TRAINING_DATA_CSV = DATASETS_DIR / "TrainingData.csv"
TESTING_DATA_CSV = DATASETS_DIR / "TestingData.csv"
UNIQUE_COUNTS_CSV = DATASETS_DIR / "UniqueColumnSummary_with_counts.csv"
UNIQUE_COUNTS_CLEAN_CSV = DATASETS_DIR / \
    "UniqueColumnSummary_with_counts_cleaned.csv"
COUNTRY_COUNTS_CSV = DATASETS_DIR / "country_counts_comparison.csv"
COUNTRY_COUNTS_CLEAN_CSV = DATASETS_DIR / \
    "country_counts_comparison_cleaned.csv"

# Example: LightGBM model artifact
LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.txt"

# Diagram subdirectories
ARIMA_DIAGRAMS_DIR = DIAGRAMS_DIR / "Arima_diagrams"
LSTM_DIAGRAMS_DIR = DIAGRAMS_DIR / \
    "LSTM_Parrallel_forecast_selected_series.png"  # or Dir
# add other specific diagram dirs as needed
