# tests/test_data_loading.py

from Data_preparation.CleaningCBSData import load_dataset
import pandas as pd
import sys
import os
# ─── Add project root to sys.path ───────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ───────────────────────────────────────────────────────


def test_load_dataset_not_empty():
    df = load_dataset("TrainingData.csv")
    assert not df.empty
    # should have at least these identifier columns
    for col in ("L_REP_CTY", "L_CP_COUNTRY", "CBS_BASIS"):
        assert col in df.columns
    # and at least one quarterly column
    # (e.g. anything matching YYYY-Qn)
    qt_cols = [c for c in df.columns if c.endswith(
        tuple(f"-Q{q}" for q in range(1, 5)))]
    assert len(qt_cols) > 0
