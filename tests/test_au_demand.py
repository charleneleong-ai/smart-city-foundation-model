from datetime import datetime, timezone

import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.adapters.demand import AEMODemandAdapter, _months, demand_source
from sctwin.demand import aemo_to_long
from sctwin.geo import cell_of


def _raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "REGION": ["NSW1"] * 3,
            "SETTLEMENTDATE": ["2023/01/01 10:05:00", "2023/01/01 10:35:00", "2023/01/01 11:05:00"],
            "TOTALDEMAND": [6000.0, 6200.0, 7000.0],
            "RRP": [100.0, 110.0, 120.0],
            "PERIODTYPE": ["TRADE"] * 3,
        }
    )


def test_aemo_to_long_converts_aest_to_utc_and_means_to_the_hour():
    out = aemo_to_long(_raw(), cell="abc", start=datetime(2023, 1, 1, tzinfo=timezone.utc),
                       end=datetime(2023, 1, 1, 23, tzinfo=timezone.utc))
    assert out.columns == ["cell", "time", "layer", "value"]
    assert out["cell"].unique().to_list() == ["abc"] and out["layer"].unique().to_list() == ["load"]
    row0 = out.sort("time").row(0, named=True)
    assert row0["time"] == datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)  # AEST 10:00 -> UTC 00:00
    assert abs(row0["value"] - 6100.0) < 1e-9  # mean of the two 5-min readings in that hour


def test_months_span_is_inclusive_and_crosses_year_boundary():
    assert _months(datetime(2022, 11, 15), datetime(2023, 2, 3)) == ["202211", "202212", "202301", "202302"]


class _StubAEMO(AEMODemandAdapter):
    def _read(self, ym: str) -> pl.DataFrame:
        return _raw()


def test_aemo_adapter_conforms_and_pins_to_one_cell():
    adapter = _StubAEMO()
    assert isinstance(adapter, LayerAdapter) and adapter.name == "demand.load"
    out = adapter.fetch([cell_of(-33.87, 151.21, res=7)],
                        datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
    assert out["cell"].n_unique() == 1 and out["layer"].unique().to_list() == ["load"]
    assert isinstance(demand_source("aemo"), AEMODemandAdapter)
