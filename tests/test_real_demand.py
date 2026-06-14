from datetime import datetime, timedelta

import polars as pl

from sctwin.demand import electricity_to_long
from sctwin.forecast.features import CALENDAR_COLS, build_calendar_supervised


def _raw() -> pl.DataFrame:
    t0 = datetime(2013, 1, 1)
    times = [t0 + timedelta(hours=h) for h in range(72)]
    return pl.DataFrame(
        {"id": ["T0", "T1", "T2"], "timestamp": [times, times, times],
         "target": [[float(h) for h in range(72)]] * 3}
    )


def test_electricity_to_long_reshapes_and_windows():
    out = electricity_to_long(_raw(), start=datetime(2013, 1, 1, 6), end=datetime(2013, 1, 1, 12), n_meters=2)
    assert out.columns == ["cell", "time", "layer", "value"]
    assert set(out["cell"].unique().to_list()) == {"T0", "T1"}  # only the first n_meters
    assert out["time"].min() >= datetime(2013, 1, 1, 6) and out["time"].max() <= datetime(2013, 1, 1, 12)
    assert out["layer"].unique().to_list() == ["load"]


def test_build_calendar_supervised_adds_features_and_drops_lag_warmup():
    demand = electricity_to_long(_raw(), start=datetime(2013, 1, 1), end=datetime(2013, 1, 4), n_meters=1)
    sup = build_calendar_supervised(demand)
    assert set(CALENDAR_COLS) <= set(sup.columns) and "y" in sup.columns
    assert "t2m" not in sup.columns  # no weather
    assert sup.height == 72 - 24  # all 72 h, 24 dropped for the y_lag_24 warmup (single meter)
    assert sup.filter(pl.col("y") == 24.0)["y_lag_24"].item() == 0.0  # lag aligns to 24 h earlier
