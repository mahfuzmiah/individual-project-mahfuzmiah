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
