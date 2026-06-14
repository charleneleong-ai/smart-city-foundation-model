from datetime import datetime, timedelta, timezone

import h3
import polars as pl

from sctwin.demand import ev_charging_load


def _weather(temp: float, hours: range, res: int = 7) -> pl.DataFrame:
    cells = list(h3.grid_disk(h3.latlng_to_cell(51.5, -0.12, res), 1))  # a small London patch
    t0 = datetime(2020, 1, 15, tzinfo=timezone.utc)
    return pl.DataFrame(
        [{"cell": c, "time": t0 + timedelta(hours=h), "value": temp} for c in cells for h in hours]
    )


def test_charging_peaks_in_the_evening_not_overnight():
    ld = ev_charging_load(_weather(8.0, range(24)), res=7)
    by_hour = ld.with_columns(pl.col("time").dt.hour().alias("h")).group_by("h").agg(pl.col("value").mean())
    peak = by_hour.sort("value", descending=True).row(0, named=True)["h"]
    overnight = by_hour.filter(pl.col("h") == 4).row(0, named=True)["value"]
    assert 17 <= peak <= 21  # evening plug-in, not 04:00
    assert by_hour.filter(pl.col("h") == peak).row(0, named=True)["value"] > 5 * overnight


def test_cold_amplifies_charging_and_demand_is_non_negative():
    evening = range(19, 20)
    cold = ev_charging_load(_weather(-2.0, evening), res=7)["value"].mean()
    mild = ev_charging_load(_weather(18.0, evening), res=7)["value"].mean()
    assert cold > mild  # heating-degree amplification
    assert ev_charging_load(_weather(25.0, range(24)), res=7)["value"].min() >= 0.0
