import numpy as np
import polars as pl

from sctwin.forecast.backtest import temporal_split
from sctwin.forecast.baselines import Forecaster
from sctwin.forecast.features import to_xy
from sctwin.verify.conformal import ConformalCalibrator

RESULT_FIELDS = ["y_true", "y_pred", "abs_error", "lo", "hi", "covered"]


def verification_frame(
    model: Forecaster,
    frame: pl.DataFrame,
    feature_cols: list[str],
    *,
    calib_frac: float = 0.3,
    test_frac: float = 0.25,
    alpha: float = 0.1,
) -> pl.DataFrame:
    """3-way temporal split (train -> calibrate -> test): fit the model, calibrate
    split-conformal intervals on the calib residuals, evaluate on test. Returns a wide
    frame keyed on (cell, time) with predicted-vs-actual, interval, and covered."""
    fit_part, test = temporal_split(frame, test_frac)
    train, calib = temporal_split(fit_part, calib_frac)

    xtr, ytr = to_xy(train, feature_cols)
    model.fit(xtr, ytr)

    xca, yca = to_xy(calib, feature_cols)
    cal = ConformalCalibrator.fit(yca, model.predict(xca), alpha=alpha)

    xte, yte = to_xy(test, feature_cols)
    pred = model.predict(xte)
    lo, hi = cal.interval(pred)
    return test.select("cell", "time").with_columns(
        pl.Series("y_true", yte),
        pl.Series("y_pred", pred),
        pl.Series("abs_error", np.abs(pred - yte)),
        pl.Series("lo", lo),
        pl.Series("hi", hi),
        pl.Series("covered", (yte >= lo) & (yte <= hi)),
    )


def as_layer(results: pl.DataFrame, field: str) -> pl.DataFrame:
    """Project one result field to the canonical (cell, time, layer, value) schema so the
    SP8 H3 renderer can draw it (error / coverage / drift maps)."""
    if field not in RESULT_FIELDS:
        raise ValueError(f"unknown field {field!r}; expected one of {RESULT_FIELDS}")
    return results.select(
        "cell", "time", pl.lit(field).alias("layer"), pl.col(field).alias("value")
    )
