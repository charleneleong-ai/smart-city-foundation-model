from typing import Protocol, Self, runtime_checkable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


@runtime_checkable
class Forecaster(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Forecaster": ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class SklearnForecaster:
    def __init__(self, estimator: RegressorMixin) -> None:
        self._m = estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        self._m.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._m.predict(x)


def DegreeDayRegressor() -> SklearnForecaster:
    """Weather-normalized linear baseline over [HDD, CDD]."""
    return SklearnForecaster(LinearRegression())


def GBMForecaster(**kwargs: object) -> SklearnForecaster:
    return SklearnForecaster(HistGradientBoostingRegressor(**kwargs))


class PersistenceForecaster:
    """Predict the lag column passed as the single feature (naive baseline)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0]
