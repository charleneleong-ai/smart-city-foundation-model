"""Chronos-2 — a pretrained time-series foundation model as the demand forecaster.

Swaps the GBM box for amazon/chronos-2: per-cell target history + weather covariates → a
calibrated quantile forecast over the test horizon. Chronos-2 is probabilistic, so its 0.1/0.9
quantiles *are* the interval (no separate conformal needed) — `verify` returns the same wide
frame as `verification_frame`, so it flows into `skill()` and the reasoning baseline unchanged.

Heavy deps (`chronos`, torch, pandas) are lazy-loaded behind the `tsfm` extra + a model
download; the orchestration here is pure polars, and the one `predict_df` call is isolated in
`_forecast` so tests can override it without a GPU or the model.
"""

from typing import Any

import polars as pl

_QUANTILES = (0.1, 0.5, 0.9)  # 0.5 = point forecast, 0.1/0.9 = the calibrated interval
_COVARIATES = ["t2m", "hdd", "cdd", "hour", "dow", "month"]  # weather + calendar (Chronos models the lags itself)


def _time_split(frame: pl.DataFrame, test_frac: float) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Split by timestamp so every cell shares one contiguous test window — series forecasting
    needs this, unlike the row-count tabular split used for the GBM."""
    times = frame.select("time").unique().sort("time")["time"].to_list()
    cut = int(round(len(times) * (1 - test_frac)))
    split, horizon = times[cut], len(times) - cut
    return frame.filter(pl.col("time") < split), frame.filter(pl.col("time") >= split), horizon


class ChronosForecaster:
    def __init__(self, *, model: str = "amazon/chronos-2", device: str = "cpu", pipeline: Any = None) -> None:
        self._model, self._device, self._pipeline = model, device, pipeline

    def _forecast(
        self, history: pl.DataFrame, future: pl.DataFrame, horizon: int, target: str, covariates: list[str]
    ) -> pl.DataFrame:
        """Chronos-2 `predict_df` at the pandas boundary → (cell, time, y_pred, lo, hi). Isolated
        so tests can stub it without torch / chronos / pandas / a model download."""
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(self._model, device_map=self._device)
        ctx = history.select("cell", "time", target, *covariates).to_pandas()
        fut = future.select("cell", "time", *covariates).to_pandas()
        pred = self._pipeline.predict_df(
            ctx, future_df=fut, prediction_length=horizon, quantile_levels=list(_QUANTILES),
            id_column="cell", timestamp_column="time", target=target,
        )
        return pl.from_pandas(pred).rename({"0.5": "y_pred", "0.1": "lo", "0.9": "hi"}).select(
            "cell", "time", "y_pred", "lo", "hi"
        )

    def verify(
        self, frame: pl.DataFrame, *, target: str = "y", covariates: list[str] | None = None, test_frac: float = 0.25
    ) -> pl.DataFrame:
        """Forecast the last `test_frac` of timestamps from the preceding history + covariates;
        return the wide (cell, time, y_true, y_pred, error, abs_error, lo, hi, covered) frame —
        the same schema `verification_frame` produces."""
        history, test, horizon = _time_split(frame, test_frac)
        preds = self._forecast(history, test, horizon, target, covariates or _COVARIATES)
        out = test.select("cell", "time", pl.col(target).alias("y_true")).join(preds, on=["cell", "time"])
        return out.with_columns(
            (pl.col("y_pred") - pl.col("y_true")).alias("error"),
            (pl.col("y_pred") - pl.col("y_true")).abs().alias("abs_error"),
            ((pl.col("y_true") >= pl.col("lo")) & (pl.col("y_true") <= pl.col("hi"))).alias("covered"),
        )
