from datetime import datetime, timezone

import numpy as np
import pytest

xr = pytest.importorskip("xarray")  # only runs with the `gridded` extra installed

from sctwin.adapters.era5 import _sample  # noqa: E402
from sctwin.geo import cell_of  # noqa: E402


def _grid() -> "xr.Dataset":
    lats = [52.0, 51.0, 50.0]  # descending, like ERA5
    lons = [-1.0, 0.0, 1.0]
    times = [np.datetime64("2020-01-15T00:00"), np.datetime64("2020-01-15T01:00")]
    data = np.empty((len(times), len(lats), len(lons)))
    for i in range(len(lats)):
        for j in range(len(lons)):
            data[:, i, j] = 273.15 + (10 * i + j)  # Kelvin, distinct per grid point
    return xr.Dataset(
        {"t2m": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def test_sample_picks_nearest_grid_point_and_converts_kelvin():
    # cell centroid near (51, 0) -> nearest grid (i=1, j=1) -> 273.15 + 11 K -> 11 C
    cell = cell_of(51.02, 0.01, 6)
    df = _sample(_grid(), [cell])
    assert df.columns == ["cell", "time", "layer", "value"]
    assert df.height == 2  # two timesteps
    assert df["cell"].unique().to_list() == [cell.h3]
    assert all(abs(v - 11.0) < 1e-6 for v in df["value"].to_list())
    assert df["time"][0] == datetime(2020, 1, 15, 0, tzinfo=timezone.utc)


def test_sample_maps_each_cell_to_its_own_nearest_value():
    nw = cell_of(51.98, -0.98, 6)  # near (52, -1) -> 10*0 + 0 = 0 -> 0 C
    se = cell_of(50.02, 0.98, 6)  # near (50, 1) -> 10*2 + 2 = 22 -> 22 C
    df = _sample(_grid(), [nw, se])
    by_cell = {c: v for c, v in zip(df["cell"].to_list(), df["value"].to_list(), strict=False)}
    assert abs(by_cell[nw.h3] - 0.0) < 1e-6
    assert abs(by_cell[se.h3] - 22.0) < 1e-6
