from datetime import datetime, timedelta, timezone

import polars as pl

from sctwin.forecast.chronos import ChronosForecaster, _time_split


def _frame(days: int = 2, cells: tuple[str, ...] = ("a", "b")) -> pl.DataFrame:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        [
            {"cell": c, "time": t0 + timedelta(hours=h), "y": 50.0 + h, "t2m": 8.0, "hdd": 10.0,
             "cdd": 0.0, "hour": h % 24, "dow": 1, "month": 1}
            for c in cells for h in range(24 * days)
        ]
    )


class _Stub(ChronosForecaster):
    """Stub Chronos: y_pred = y_true + offset, interval ±band — exercises the assembly without
    torch / the model."""

    def __init__(self, *, offset: float, band: float) -> None:
        super().__init__()
        self._offset, self._band = offset, band

    def _forecast(self, history, future, horizon, target, covariates):
        return future.select(
            "cell", "time",
            (pl.col(target) + self._offset).alias("y_pred"),
            (pl.col(target) + self._offset - self._band).alias("lo"),
            (pl.col(target) + self._offset + self._band).alias("hi"),
        )


def test_time_split_shares_one_contiguous_test_window_across_cells():
    hist, test, horizon = _time_split(_frame(days=2), 0.25)
    assert test.select("time").n_unique() == horizon  # horizon counts distinct test timestamps
    assert hist.select("time").max().item() < test.select("time").min().item()  # clean past/future cut
    a = set(test.filter(pl.col("cell") == "a")["time"].to_list())
    b = set(test.filter(pl.col("cell") == "b")["time"].to_list())
    assert a == b and len(a) == horizon  # every cell forecasts the same window


def test_verify_assembles_wide_frame_with_error_and_coverage():
    out = _Stub(offset=3.0, band=5.0).verify(_frame(days=2))  # biased +3, wide band -> covered
    assert set(out.columns) == {"cell", "time", "y_true", "y_pred", "lo", "hi", "error", "abs_error", "covered"}
    assert (out["error"] == 3.0).all() and (out["abs_error"] == 3.0).all()
    assert out["covered"].all()  # |3| < band 5


def test_verify_marks_misses_outside_the_interval():
    out = _Stub(offset=10.0, band=2.0).verify(_frame(days=2))  # biased +10, band ±2 -> y_true escapes
    assert not out["covered"].any()
