# tests/test_lstm_main.py
import modeling.Deep_Learning.LSTM1 as lstm_mod
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# 1) ensure repo root is importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture(autouse=True)
def patch_keras_and_load_data(monkeypatch):
    # build one synthetic wide row
    quarters = [f"{2000+q//4}-Q{q % 4+1}" for q in range(6)]
    ids = dict(
        L_REP_CTY=["A"],
        L_CP_COUNTRY=["B"],
        CBS_BASIS=["F"]
    )
    values = [1, 2, 3, 4, 5, 6]
    wide = pd.DataFrame({**ids, **{q: [v] for q, v in zip(quarters, values)}})

    # Tile it 200 times to satisfy NO_ITERATIONS
    train = pd.concat([wide]*200, ignore_index=True)
    test = train.copy()

    # Patch load_data
    monkeypatch.setattr(lstm_mod, "load_data", lambda *a, **k: (train, test))

    # Stub out keras.Sequential
    class DummyModel:
        def __init__(self, layers): pass
        def compile(self, **kw): pass
        def fit(self, X, y, epochs, verbose): return

        def predict(self, X, verbose):
            return np.full((X.shape[0], 1), 99.0)
    monkeypatch.setattr(lstm_mod, "Sequential", DummyModel)

    yield


def test_main_constant_path_and_output():
    # run main for 1 epoch
    total_time, per_row_times, results = lstm_mod.main(epochs=1)

    assert isinstance(total_time, float)
    assert len(per_row_times) == 200

    first = results[0]
    # horizon = 6 quarters - 4 warmup = 2
    assert first['forecast'].shape == first['actual'].shape
    # and our dummy model always predicts 99, so each forecast == 99
    assert np.all(first['forecast'] == 99.0)
    # actual should match the last 6 values we seeded: [1,2,3,4,5,6]
    assert np.all(first['actual'] == np.array([1, 2, 3, 4, 5, 6]))
