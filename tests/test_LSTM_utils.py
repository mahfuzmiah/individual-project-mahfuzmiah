# tests/test_lstm_utils.py
import numpy as np
import pytest
from modeling.Deep_Learning.LSTM1 import split_sequence, seconds_to_hms, wmape


def test_split_sequence_basic():
    seq = [10, 20, 30, 40, 50]
    # with n_steps=2 we get windows [10,20]->30, [20,30]->40, [30,40]->50
    X, y = split_sequence(seq, n_steps=2)
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    np.testing.assert_array_equal(X[0], [10, 20])
    assert y[0] == 30


def test_seconds_to_hms_various():
    assert seconds_to_hms(0) == (0, 0, 0)
    assert seconds_to_hms(3661) == (1, 1, 1)
    assert seconds_to_hms(7322) == (2, 2, 2)


def test_wmape_computation():
    y_true = np.array([0.0, 1.0, 2.0, np.nan])
    y_pred = np.array([0.0, 2.0, 1.0, 5.0])
    # mask the nan pair, denom = |0|+|1|+|2| = 3
    # abs errors = [0,1,1] → sum=2  → 2/3
    assert pytest.approx(wmape(y_true, y_pred), rel=1e-6) == 2/3
