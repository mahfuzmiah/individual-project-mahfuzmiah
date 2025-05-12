# tests/test_clean_and_split.py
from Data_preparation.CleaningCBSData import main
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def create_synthetic_csv(tmp_path):
    cols = [
        "Reporting country", "Counterparty country",
        "L_REP_CTY", "L_CP_COUNTRY", "CBS_BASIS",
        "L_MEASURE", "Type of instruments", "Remaining maturity",
        "Currency type of booking location", "Counterparty sector",
        "Balance sheet position", "CBS_BANK_TYPE",
        "2005-Q1", "2005-Q2", "2020-Q1", "2020-Q2"
    ]
    data = [
        ["A", "B", "A", "B", "F", "", "", "", "", "All sectors",
            "Total claims", "4B", 10, 20, 30, 40],
        ["A", "B", "A", "B", "F", "", "", "", "",
            "All sectors", "Total claims", "4R", 1, 2, 3, 4],
        ["A", "B", "A", "B", "F", "", "", "", "",
            "All sectors", "Total claims", "4C", 5, 5, 5, 5],
        ["A", "B", "A", "B", "F", "", "", "", "",
            "All sectors", "Total claims", "4O", 0, 0, 0, 0],
        ["All reporting countries", "B", "ALL", "B", "F", "", "",
            "", "", "All sectors", "Total claims", "4B", 1, 1, 1, 1],
        ["A", "B", "A", "B", "F", "", "", "", "", "All sectors",
            "Total claims", "4B", np.nan, np.nan, np.nan, np.nan]
    ]
    df = pd.DataFrame(data, columns=cols)
    path = tmp_path / "WS_CBS_PUB_csv_col.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def patch_paths(monkeypatch, tmp_path):
    csv = create_synthetic_csv(tmp_path)
    import Data_preparation.CleaningCBSData as mod
    monkeypatch.setattr(mod, "input_file", csv)
    monkeypatch.setattr(mod, "output_file", tmp_path / "CleanedCBSDataSet.csv")
    monkeypatch.setattr(mod, "training_file", tmp_path / "TrainingData.csv")
    monkeypatch.setattr(mod, "testing_file", tmp_path / "TestingData.csv")
    return tmp_path


def test_clean_and_split(patch_paths):
    tmp = patch_paths
    main()

    cleaned = pd.read_csv(tmp / "CleanedCBSDataSet.csv")
    train = pd.read_csv(tmp / "TrainingData.csv")
    test = pd.read_csv(tmp / "TestingData.csv")

    # 1) filtered out the unwanted row & all-NaN row => 4 left
    assert len(cleaned) == 4

    # 2) consolidation appears in train split: exactly one A→B row with 2005-Q1 = (10-1)+(5+0) = 14
    assert len(train) == 1
    val = train["2005-Q1"].iloc[0]
    assert val == 14

    # 3) split is correct: train has only 2005-era columns, test only 2020-era
    assert all(col.startswith("200") for col in train.columns if "-Q" in col)
    assert all(col.startswith("2020") for col in test.columns if "-Q" in col)
