from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalCalibrator:
    quantile: float

    @classmethod
    def fit(
        cls, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1
    ) -> "ConformalCalibrator":
        resid = np.abs(np.asarray(y_pred) - np.asarray(y_true))
        n = len(resid)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # finite-sample correction
        return cls(quantile=float(np.quantile(resid, level, method="higher")))

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_pred)
        return y - self.quantile, y + self.quantile


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y_true)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))
