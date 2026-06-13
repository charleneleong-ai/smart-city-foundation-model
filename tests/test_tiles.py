from datetime import timedelta

import polars as pl

from sctwin.app.tiles import tile_records
from sctwin.geo import Cell, center_of, haversine_km
from sctwin.registry import Registry

LAT, LON = 51.505, -0.12


class _FakeWeather:
    name = "weather.t2m"

    def fetch(self, cells, start, end) -> pl.DataFrame:
        rows = [(c.h3, start + timedelta(hours=h), "t2m", float(h)) for c in cells for h in range(24)]
        return pl.DataFrame(
            {
                "cell": [r[0] for r in rows],
                "time": [r[1] for r in rows],
                "layer": [r[2] for r in rows],
                "value": [r[3] for r in rows],
            }
        )


def _reg() -> Registry:
    reg = Registry()
    reg.register(_FakeWeather())
    return reg


def test_tile_records_are_deckgl_hexes_within_radius():
    recs = tile_records(_reg(), "weather.t2m", lat=LAT, lon=LON, radius_km=6, res=8, date="2020-01-15")
    assert len(recs) > 0
    for r in recs:
        assert {"cell", "value", "color", "polygon"} <= set(r)
        assert r["polygon"][0] == r["polygon"][-1]  # closed ring
        clat, clon = center_of(Cell(r["cell"], 8))
        assert haversine_km(LAT, LON, clat, clon) <= 6 + 0.5  # inside the radius (+ hex slack)


def test_hour_selects_that_timestamps_value():
    recs = tile_records(_reg(), "weather.t2m", lat=LAT, lon=LON, radius_km=4, res=8,
                        date="2020-01-15", hour=7)
    assert {r["value"] for r in recs} == {7.0}  # the fake returns value == hour


def test_caps_to_nearest_max_cells():
    recs = tile_records(_reg(), "weather.t2m", lat=LAT, lon=LON, radius_km=30, res=8,
                        date="2020-01-15", max_cells=15)
    assert len(recs) == 15
