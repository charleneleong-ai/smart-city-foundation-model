from datetime import datetime, timezone

import polars as pl

from sctwin.adapters.cache import CachingAdapter
from sctwin.geo import Cell, cell_of

_DAY = datetime(2020, 1, 1, tzinfo=timezone.utc)


class _CountingAdapter:
    name = "weather.t2m"

    def __init__(self) -> None:
        self.calls = 0
        self.last: list[Cell] = []

    def fetch(self, cells, start, end) -> pl.DataFrame:
        self.calls += 1
        self.last = list(cells)
        return pl.DataFrame(
            {
                "cell": [c.h3 for c in cells],
                "time": [start] * len(cells),
                "layer": ["t2m"] * len(cells),
                "value": [1.0] * len(cells),
            }
        )


def test_second_fetch_is_served_from_cache(tmp_path):
    inner = _CountingAdapter()
    cad = CachingAdapter(inner, tmp_path)
    cells = [cell_of(51.5, -0.1, 7), cell_of(48.8, 2.3, 7)]

    df1 = cad.fetch(cells, _DAY, _DAY)
    assert inner.calls == 1 and df1.height == 2

    df2 = cad.fetch(cells, _DAY, _DAY)  # identical request -> no inner call
    assert inner.calls == 1
    assert set(df2["cell"].to_list()) == {c.h3 for c in cells}


def test_only_missing_cells_are_fetched(tmp_path):
    inner = _CountingAdapter()
    cad = CachingAdapter(inner, tmp_path)
    a, b = cell_of(51.5, -0.1, 7), cell_of(48.8, 2.3, 7)

    cad.fetch([a], _DAY, _DAY)  # caches a
    cad.fetch([a, b], _DAY, _DAY)  # only b is missing
    assert inner.calls == 2
    assert inner.last == [b]  # second inner call requested only the uncached cell


class _MultiLayerAdapter:
    name = "weather"

    def fetch(self, cells, start, end) -> pl.DataFrame:
        layers = ["t2m", "wind_dir", "precip"]  # >1 row per (cell, time)
        return pl.DataFrame(
            {
                "cell": [c.h3 for c in cells for _ in layers],
                "time": [start for _ in cells for _ in layers],
                "layer": [layer for _ in cells for layer in layers],
                "value": [1.0 for _ in cells for _ in layers],
            }
        )


def test_cache_preserves_every_layer_per_cell(tmp_path):
    # regression: dedup must key on (cell, time, layer), not (cell, time), or multi-layer
    # adapters collapse to a single layer through the cache.
    cad = CachingAdapter(_MultiLayerAdapter(), tmp_path)
    cell = cell_of(34.05, -118.24, 8)
    out = cad.fetch([cell], _DAY, _DAY)  # writes the parquet, then re-reads it
    assert set(out["layer"].to_list()) == {"t2m", "wind_dir", "precip"}
