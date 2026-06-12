from datetime import datetime, timezone

import polars as pl
import pytest

from sctwin.adapters.base import LayerAdapter
from sctwin.geo import cell_of
from sctwin.registry import Registry


class _FakeAdapter:
    name = "weather.t2m"

    def fetch(self, cells, start, end) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "cell": [c.h3 for c in cells],
                "time": [datetime(2020, 1, 1, tzinfo=timezone.utc)] * len(cells),
                "layer": ["t2m"] * len(cells),
                "value": [1.0] * len(cells),
            }
        )


def test_fake_adapter_satisfies_protocol():
    assert isinstance(_FakeAdapter(), LayerAdapter)


def test_get_routes_to_registered_adapter():
    reg = Registry()
    reg.register(_FakeAdapter())
    cell = cell_of(51.5, -0.1, res=7)
    df = reg.get("weather.t2m", [cell], datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert df["cell"].to_list() == [cell.h3]


def test_get_unknown_layer_raises():
    reg = Registry()
    with pytest.raises(KeyError, match="no adapter"):
        reg.get("missing", [], datetime(2020, 1, 1), datetime(2020, 1, 1))


def test_register_duplicate_raises():
    reg = Registry()
    reg.register(_FakeAdapter())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_FakeAdapter())
