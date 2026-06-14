from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from sctwin.adapters.base import LayerAdapter
from sctwin.adapters.demand import ElectricityMeterAdapter, LondonSmartMeterAdapter, demand_source
from sctwin.geo import cell_of

_CELLS = [cell_of(51.50, -0.12, res=7), cell_of(51.52, -0.10, res=7)]
_S = datetime(2013, 1, 1, tzinfo=timezone.utc)
_E = datetime(2013, 1, 8, tzinfo=timezone.utc)


def _london_raw() -> pl.DataFrame:
    times = [datetime(2013, 1, 7) + timedelta(minutes=30 * i, seconds=1) for i in range(12)]
    return pl.DataFrame({"id": ["M0", "M1"], "timestamp": [times, times], "target": [[0.2] * 12, [0.5] * 12]})


def _elec_raw() -> pl.DataFrame:
    times = [datetime(2013, 1, 1) + timedelta(hours=h) for h in range(48)]
    return pl.DataFrame({"id": ["E0", "E1"], "timestamp": [times, times], "target": [[float(h) for h in range(48)]] * 2})


class _StubLondon(LondonSmartMeterAdapter):
    def _read(self, n: int) -> pl.DataFrame:
        return _london_raw().head(n)


class _StubElec(ElectricityMeterAdapter):
    def _read(self) -> pl.DataFrame:
        return _elec_raw()


@pytest.mark.parametrize("adapter", [_StubLondon(), _StubElec()])
def test_demand_adapters_conform_key_on_cells_and_emit_canonical_utc(adapter):
    assert isinstance(adapter, LayerAdapter) and adapter.name == "demand.load"  # same contract as weather
    out = adapter.fetch(_CELLS, _S, _E)
    assert out.columns == ["cell", "time", "layer", "value"]
    assert set(out["cell"].unique().to_list()) <= {c.h3 for c in _CELLS}  # mapped onto the requested cells
    assert out["layer"].unique().to_list() == ["load"]
    assert out["time"].dtype == pl.Datetime("us", "UTC")  # uniform tz so demand joins the weather frame


def test_demand_source_factory_selects_and_rejects_unknown():
    assert isinstance(demand_source("london"), LondonSmartMeterAdapter)
    assert isinstance(demand_source("electricity"), ElectricityMeterAdapter)
    with pytest.raises(ValueError):
        demand_source("nope")
