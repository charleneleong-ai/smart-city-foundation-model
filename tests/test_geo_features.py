from datetime import datetime, timedelta, timezone

import h3
import polars as pl

from sctwin.demand import ev_charging_load
from sctwin.geo_features import population_by_cell
from sctwin.geo import cell_of


def test_population_by_cell_sums_children_into_the_target_resolution():
    parent = h3.latlng_to_cell(51.5, -0.12, 2)
    children = list(h3.cell_to_children(parent, 8))[:3]
    out = population_by_cell(pl.DataFrame({"h3": children, "population": [100.0, 50.0, 25.0]}), 2)
    assert out[parent] == 175.0  # the three Kontur hexes aggregate into their res-2 ancestor


def _weather(cells: list, temp: float = 5.0, hours: int = 24) -> pl.DataFrame:
    t0 = datetime(2020, 1, 15, tzinfo=timezone.utc)
    return pl.DataFrame(
        [{"cell": c.h3, "time": t0 + timedelta(hours=h), "value": temp} for c in cells for h in range(hours)]
    )


def test_ev_charging_demand_scales_with_real_population():
    dense, sparse = cell_of(51.50, -0.12, 7), cell_of(51.52, -0.10, 7)
    population = {dense.h3: 1000.0, sparse.h3: 100.0}  # 10x more people in `dense`
    ld = ev_charging_load(_weather([dense, sparse]), res=7, population=population)
    by_cell = ld.group_by("cell").agg(pl.col("value").mean())
    d = by_cell.filter(pl.col("cell") == dense.h3)["value"].item()
    s = by_cell.filter(pl.col("cell") == sparse.h3)["value"].item()
    assert d > 3 * s  # charging demand tracks the population, not a synthetic gradient
