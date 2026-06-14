from datetime import datetime, timezone

import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.adapters.demand import ElectricityMapsAdapter, demand_source
from sctwin.demand import el_maps_to_long
from sctwin.geo import cell_of

_S = datetime(2026, 6, 13, tzinfo=timezone.utc)
_E = datetime(2026, 6, 14, tzinfo=timezone.utc)


def _history(hours: range) -> list[dict]:
    base = datetime(2026, 6, 13, tzinfo=timezone.utc)
    return [{"datetime": base.replace(hour=h).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
             "value": 1000.0 + h, "unit": "MW"} for h in hours]


def test_el_maps_to_long_parses_to_canonical_utc():
    out = el_maps_to_long(_history(range(3)), cell="abc", start=_S, end=_E).sort("time")
    assert out.columns == ["cell", "time", "layer", "value"]
    assert out["cell"].unique().to_list() == ["abc"] and out["layer"].unique().to_list() == ["load"]
    assert out["time"].dtype == pl.Datetime("us", "UTC")
    assert out["value"].to_list()[:2] == [1000.0, 1001.0]


class _StubEM(ElectricityMapsAdapter):
    def __init__(self, history: list[dict], **kw: object) -> None:
        super().__init__(**kw)
        self._history = history

    def _read(self, cell: object) -> list[dict]:
        return self._history


def test_adapter_accumulates_across_polls(tmp_path):
    cell = cell_of(51.5, -0.12, 7)
    _StubEM(_history(range(0, 12)), zone="GB", cache_dir=str(tmp_path)).fetch([cell], _S, _E)  # first poll: 0..11
    out = _StubEM(_history(range(6, 18)), zone="GB", cache_dir=str(tmp_path)).fetch([cell], _S, _E)  # overlap 6..17
    assert out["time"].n_unique() == 18  # 0..17 accumulated across polls, the overlap deduped
    assert isinstance(demand_source("electricitymaps"), ElectricityMapsAdapter)
    assert isinstance(demand_source("electricitymaps"), LayerAdapter)
