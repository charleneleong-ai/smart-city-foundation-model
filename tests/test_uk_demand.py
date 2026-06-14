from datetime import datetime, timedelta, timezone

import polars as pl

from sctwin.demand import london_smart_meters_to_long
from sctwin.geo import cell_of


def _raw() -> pl.DataFrame:
    # two meters, half-hourly readings at HH:00:01 / HH:30:01 over 6 hours
    t0 = datetime(2013, 1, 7)
    times = [t0 + timedelta(minutes=30 * i, seconds=1) for i in range(12)]
    return pl.DataFrame(
        {"id": ["M0", "M1"], "timestamp": [times, times], "target": [[0.2] * 12, [0.5] * 12]}
    )


def test_averages_half_hours_to_the_hour_and_maps_meters_to_cells():
    cells = [cell_of(51.50, -0.12, res=7), cell_of(51.52, -0.10, res=7)]
    out = london_smart_meters_to_long(_raw(), cells, start=datetime(2013, 1, 7, tzinfo=timezone.utc),
                                      end=datetime(2013, 1, 7, 5, tzinfo=timezone.utc))
    assert out.columns == ["cell", "time", "layer", "value"]
    assert set(out["cell"].unique().to_list()) == {cells[0].h3, cells[1].h3}  # meters -> distinct cells
    assert out.filter(pl.col("cell") == cells[0].h3).height == 6  # 12 half-hours -> 6 hourly
    assert out["time"].dt.minute().max() == 0  # snapped to the hour
    assert abs(out.filter(pl.col("cell") == cells[1].h3)["value"].mean() - 0.5) < 1e-9  # M1 averages to 0.5
