from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np
import polars as pl

from sctwin.forecast.baselines import Forecaster
from sctwin.forecast.features import to_xy


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict[str, float]
    predictions: np.ndarray
    split_time: datetime


def temporal_split(
    frame: pl.DataFrame, test_frac: float = 0.25
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ordered = frame.sort("time")
    cut = int(round(ordered.height * (1 - test_frac)))
    return ordered[:cut], ordered[cut:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    out = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2)))}
    nz = y_true != 0
    out["mape"] = float(np.mean(np.abs(err[nz] / y_true[nz]))) if nz.any() else float("nan")
    return out


def backtest(
    model: Forecaster, frame: pl.DataFrame, feature_cols: list[str], test_frac: float = 0.25
) -> BacktestResult:
    train, test = temporal_split(frame, test_frac)
    xtr, ytr = to_xy(train, feature_cols)
    xte, yte = to_xy(test, feature_cols)
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    return BacktestResult(metrics(yte, pred), pred, cast(datetime, test["time"].min()))
