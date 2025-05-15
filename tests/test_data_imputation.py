# tests/test_data_imputation.py
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# **lowercase** data_imputation folder:
from Data_preparation.Data_Imputation.DataImputation import evaluate_on  # nopep8

# <-- Notice the capital “Data_Imputation” -->


@pytest.fixture
def make_synthetic(tmp_path):
    cols = ["L_REP_CTY", "L_CP_COUNTRY", "CBS_BASIS"] + \
        [f"2005-Q{i}" for i in range(1, 5)]
    data = [
        ["X", "Y", "Z", 0.0, 1.0, 2.0, 3.0],
        ["X", "Y", "Z", 1.0, np.nan, 3.0, 4.0],
        ["X", "Y", "Z", 2.0, 3.0, np.nan, 5.0],
        ["X", "Y", "Z", 3.0, 4.0, 5.0, np.nan],
        ["X", "Y", "Z", np.nan, 5.0, 6.0, 7.0],
    ]
    df = pd.DataFrame(data, columns=cols)
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_evaluate_imputers(tmp_path, make_synthetic):
    csv = make_synthetic
    out_dir = tmp_path / "outputs"
    metrics = tmp_path / "metrics.csv"

    # Run your imputation evaluation
    evaluate_on(csv, metrics, out_dir)

    base_methods = ["zeros", "ffill", "linear", "poly"]
    # pick up all knn_*.csv files
    knn_methods = sorted(p.stem for p in out_dir.glob("knn_*.csv"))
    assert knn_methods, "No knn_<k>.csv files were generated"
    for method in base_methods + knn_methods:
        assert (out_dir / f"{method}.csv").is_file()

    # Metrics file lists all methods
    dfm = pd.read_csv(metrics)
    expected = set(base_methods + knn_methods)
    assert set(dfm["method"]) == expected    # Numeric metrics non-negative
    num_cols = ["MAE", "RMSE", "KS", "Wasserstein", "time_sec"]
    assert (dfm[num_cols] >= 0).all().all()
