# tests/test_data_loading.py
import pandas as pd
from Data_preparation.CleaningCBSData import load_dataset


def test_load_dataset_not_empty():
    df = load_dataset("DataSetsCBS/TrainingData.csv")
    assert not df.empty
    assert "unique_id" in df.columns
