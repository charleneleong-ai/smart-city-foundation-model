import numpy as np

from sctwin.forecast.baselines import (
    DegreeDayRegressor,
    GBMForecaster,
    PersistenceForecaster,
)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def test_degree_day_recovers_linear_signal():
    rng = np.random.default_rng(0)
    hdd = rng.uniform(0, 20, size=500)
    cdd = rng.uniform(0, 10, size=500)
    y = 3.0 * hdd + 1.5 * cdd + rng.normal(0, 0.1, size=500)
    x = np.column_stack([hdd, cdd])
    pred = DegreeDayRegressor().fit(x, y).predict(x)
    assert _mae(pred, y) < 0.5
    assert _mae(pred, y) < _mae(np.full_like(y, y.mean()), y)  # beats mean


def test_gbm_beats_persistence_on_nonlinear():
    rng = np.random.default_rng(1)
    n = 800
    lag = rng.uniform(0, 10, size=n)
    feat = rng.uniform(0, 10, size=n)
    y = np.sin(feat) * 5 + lag * 0.2 + rng.normal(0, 0.1, size=n)
    x = np.column_stack([feat, lag])
    cut = 600
    gbm_mae = _mae(GBMForecaster().fit(x[:cut], y[:cut]).predict(x[cut:]), y[cut:])
    pers_mae = _mae(
        PersistenceForecaster().fit(x[:cut, 1:2], y[:cut]).predict(x[cut:, 1:2]), y[cut:]
    )
    assert gbm_mae < pers_mae


def test_persistence_returns_lag_column():
    x = np.array([[2.0], [5.0], [9.0]])
    pred = PersistenceForecaster().fit(x, np.zeros(3)).predict(x)
    assert np.allclose(pred, [2.0, 5.0, 9.0])
