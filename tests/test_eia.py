from datetime import datetime, timezone

import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.adapters.demand import EIADemandAdapter, demand_source
from sctwin.demand import eia_load_to_long
from sctwin.geo import cell_of


def _rows(hours: range) -> list[dict]:
    return [{"period": f"2023-01-01T{h:02d}", "respondent": "CISO", "type": "D",
             "value": 24000 + h, "value-units": "megawatthours"} for h in hours]


def test_eia_load_to_long_parses_to_canonical_utc():
    out = eia_load_to_long(_rows(range(3)), cell="abc")
    assert out.columns == ["cell", "time", "layer", "value"]
    assert out["time"].dtype == pl.Datetime("us", "UTC")
    assert out["value"].to_list() == [24000.0, 24001.0, 24002.0]
    assert out["time"].to_list()[1] == datetime(2023, 1, 1, 1, tzinfo=timezone.utc)  # 'YYYY-MM-DDT01' -> 01:00 UTC
    assert out["layer"].unique().to_list() == ["load"] and out["cell"].unique().to_list() == ["abc"]


def test_eia_load_to_long_drops_null_values():
    rows = _rows(range(2)) + [{"period": "2023-01-01T02", "respondent": "CISO", "type": "D", "value": None}]
    assert eia_load_to_long(rows, cell="abc").height == 2  # EIA reports gaps as null — dropped, not zero


class _StubEIA(EIADemandAdapter):
    def __init__(self, rows: list[dict], **kw: object) -> None:
        super().__init__(**kw)
        self._rows = rows

    def _read(self, start: datetime, end: datetime) -> list[dict]:
        return self._rows


def test_eia_adapter_conforms_and_windows_to_request():
    adapter = _StubEIA(_rows(range(24)), respondent="CISO")
    assert isinstance(adapter, LayerAdapter) and adapter.name == "demand.load"
    out = adapter.fetch([cell_of(34.05, -118.24, 7)],
                        datetime(2023, 1, 1, 2, tzinfo=timezone.utc), datetime(2023, 1, 1, 5, tzinfo=timezone.utc))
    assert out["value"].to_list() == [24002.0, 24003.0, 24004.0, 24005.0]  # clipped to [02:00, 05:00]
    assert isinstance(demand_source("eia"), EIADemandAdapter)
