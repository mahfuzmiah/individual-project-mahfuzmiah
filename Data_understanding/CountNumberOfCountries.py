import pandas as pd
import sys
from pathlib import Path
REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT_PATH))
from config import REPO_ROOT, DATASETS_DIR, IMPUTED_RESULTS_DIR_TEST, IMPUTED_RESULTS_DIR_TRAIN  # nopep8

# -- CONFIG ────────────────────────────────────────────────────────────────
ORIGINAL_FILE = DATASETS_DIR / "WS_CBS_PUB_csv_col.csv"   # Original dataset
OUTPUT_FILE = DATASETS_DIR / "country_counts_comparison.csv"  # Output file

# Load dataset
original_data = pd.read_csv(ORIGINAL_FILE)

# Ensure required columns exist
required_columns = ['Reporting country', 'Counterparty country']
missing_columns = [
    col for col in required_columns if col not in original_data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Drop missing values in required columns to avoid errors
original_data = original_data.dropna(subset=required_columns)

# Count occurrences of each unique reporting country
reporting_counts = original_data['Reporting country'].value_counts(
).reset_index()
reporting_counts.columns = ['Country', 'Reporting Count']

# Count occurrences of each unique counterparty country
counterparty_counts = original_data['Counterparty country'].value_counts(
).reset_index()
counterparty_counts.columns = ['Country', 'Counterparty Count']

# Merge both counts into a single DataFrame for easy comparison
country_counts = pd.merge(
    reporting_counts, counterparty_counts, on="Country", how="outer").fillna(0)

# Convert counts to integer type
country_counts[['Reporting Count', 'Counterparty Count']] = country_counts[[
    'Reporting Count', 'Counterparty Count']].astype(int)


country_counts.to_csv(OUTPUT_FILE, index=False)
print(f"Country counts have been saved to: {OUTPUT_FILE}")
