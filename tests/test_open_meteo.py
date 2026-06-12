from datetime import datetime, timezone

import httpx
import respx

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.geo import cell_of

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@respx.mock
def test_fetch_maps_response_to_canonical_frame():
    respx.get(ARCHIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "hourly": {
                    "time": ["2020-01-01T00:00", "2020-01-01T01:00"],
                    "temperature_2m": [4.5, 4.1],
                }
            },
        )
    )
    cell = cell_of(51.5074, -0.1278, res=7)
    df = OpenMeteoWeatherAdapter().fetch([cell], datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert df.columns == ["cell", "time", "layer", "value"]
    assert df.height == 2
    assert df["layer"].unique().to_list() == ["t2m"]
    assert df["cell"].unique().to_list() == [cell.h3]
    assert df["value"].to_list() == [4.5, 4.1]
    assert df["time"][0] == datetime(2020, 1, 1, 0, tzinfo=timezone.utc)


@respx.mock
def test_fetch_multiple_cells_concats():
    respx.get(ARCHIVE).mock(
        return_value=httpx.Response(
            200,
            json={"hourly": {"time": ["2020-01-01T00:00"], "temperature_2m": [3.0]}},
        )
    )
    cells = [cell_of(51.5, -0.1, res=7), cell_of(48.85, 2.35, res=7)]
    df = OpenMeteoWeatherAdapter().fetch(cells, datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert df.height == 2
    assert set(df["cell"].to_list()) == {c.h3 for c in cells}
