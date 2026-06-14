from datetime import datetime, timezone

import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.adapters.demand import ENTSOEAdapter, _entsoe_windows, demand_source
from sctwin.demand import entsoe_load_to_long
from sctwin.geo import cell_of

_XML = """<?xml version="1.0"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries><Period>
    <timeInterval><start>2023-01-01T00:00Z</start><end>2023-01-01T03:00Z</end></timeInterval>
    <resolution>PT60M</resolution>
    <Point><position>1</position><quantity>26000</quantity></Point>
    <Point><position>2</position><quantity>25800</quantity></Point>
    <Point><position>3</position><quantity>25500</quantity></Point>
  </Period></TimeSeries>
</GL_MarketDocument>"""


def test_entsoe_load_to_long_parses_points_to_utc_timestamps():
    out = entsoe_load_to_long(_XML, cell="GB").sort("time")
    assert out.columns == ["cell", "time", "layer", "value"]
    assert out["time"].dtype == pl.Datetime("us", "UTC")
    assert out["value"].to_list() == [26000.0, 25800.0, 25500.0]
    assert out["time"].to_list()[1] == datetime(2023, 1, 1, 1, tzinfo=timezone.utc)  # position 2 -> start + 1 h
    assert out["layer"].unique().to_list() == ["load"] and out["cell"].unique().to_list() == ["GB"]


def test_entsoe_windows_split_into_at_most_one_year_chunks():
    w = _entsoe_windows(datetime(2019, 1, 1, tzinfo=timezone.utc), datetime(2021, 7, 1, tzinfo=timezone.utc))
    assert len(w) == 3 and w[0][0] == "201901010000"  # 5-year history -> ~5 one-year requests, looped


class _StubENTSOE(ENTSOEAdapter):
    def _read(self, period_start: str, period_end: str) -> str:
        return _XML


def test_entsoe_adapter_conforms_and_windows_the_request():
    adapter = _StubENTSOE(zone="GB")
    assert isinstance(adapter, LayerAdapter) and adapter.name == "demand.load"
    out = adapter.fetch([cell_of(51.5, -0.12, 7)],
                        datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
    assert out["value"].to_list() == [26000.0, 25800.0, 25500.0]
    assert isinstance(demand_source("entsoe"), ENTSOEAdapter)
