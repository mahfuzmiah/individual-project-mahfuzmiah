# tests/test_arima_functions.py

from modeling.Statistical_models.ArimaModel import (
    load_data,
    wmape,
    seconds_to_hms,
    arima_test_method_in_series,
    process_series
)
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# 1) Make repo root importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# 2) Import functions to test


def test_wmape_basic():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 0.5, 2.5, np.nan])
    # masked pairs: indices 0,1,2
    # errors = [0, 0.5, 0.5], denom = |0+1+2|=3
    assert pytest.approx(wmape(y_true, y_pred),
                         rel=1e-3) == (0 + 0.5 + 0.5) / 3


@pytest.mark.parametrize("sec,exp", [
    (0, (0, 0, 0)),
    (3600, (1, 0, 0)),
    (3661, (1, 1, 1)),
    (7322, (2, 2, 2)),
])
def test_seconds_to_hms(sec, exp):
    assert seconds_to_hms(sec) == exp


def make_panel_df():
    """
    Create a tiny panel DataFrame:
    unique_id, ds, y
    Two series, each length=4 quarters
    """
    quarters = pd.date_range("2020-01-01", periods=4, freq="Q")
    rows = []
    for uid in [1, 2]:
        for i, ds in enumerate(quarters):
            rows.append({
                "unique_id": uid,
                "ds": ds,
                "y": float(uid * 10 + i)  # predictable sequence
            })
    return pd.DataFrame(rows)


class DummyAR:
    def __init__(self, order=None):
        # match the interface your code expects
        self.order = order or (1, 0, 0)
        self.seasonal_order = (0, 1, 0, 4)

    def fit(self, series):
        return self

    def forecast(self, n_periods):
        # return constant forecasts
        return np.ones(n_periods, dtype=float)

    # alias predict for methods that call .predict
    predict = forecast


@pytest.fixture(autouse=True)
def patch_auto_arima(monkeypatch):
    """
    Replace pmdarima.auto_arima with a dummy that returns DummyAR.
    """
    import modeling.Statistical_models.ArimaModel as mod
    monkeypatch.setattr(mod, "auto_arima", lambda *args, **kw: DummyAR())


def test_process_series_and_series_method(tmp_path):
    # Build train/test
    panel = make_panel_df()
    # split train/test by unique_id
    train = panel.copy()
    test = panel.copy()
    # test process_series
    uid = 1
    group = train[train["unique_id"] == uid]
    merged, out_uid, fit_t, pred_t = process_series(uid, group, test)
    assert out_uid == uid
    assert merged["y"].tolist() == [10.0, 11.0, 12.0, 13.0]
    # forecast comes from DummyAR.forecast->ones
    assert merged["forecast"].tolist() == [1.0, 1.0, 1.0, 1.0]

    # test arima_test_method_in_series on both series
    fit_time, fore_time = arima_test_method_in_series(
        subset_length=2, train=train, test=test
    )
    # should have returned two floats
    assert isinstance(fit_time, float)
    assert isinstance(fore_time, float)
