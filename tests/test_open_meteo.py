from datetime import datetime, timezone

import httpx
import respx

from sctwin.adapters.open_meteo import OpenMeteoForecastAdapter, OpenMeteoWeatherAdapter
from sctwin.geo import cell_of

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"


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
def test_fetch_batches_multiple_cells_in_one_request():
    # >1 location -> Open-Meteo returns a JSON array, one object per coordinate, in one call
    route = respx.get(ARCHIVE).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"hourly": {"time": ["2020-01-01T00:00"], "temperature_2m": [3.0]}},
                {"hourly": {"time": ["2020-01-01T00:00"], "temperature_2m": [9.0]}},
            ],
        )
    )
    cells = [cell_of(51.5, -0.1, res=7), cell_of(48.85, 2.35, res=7)]
    df = OpenMeteoWeatherAdapter().fetch(cells, datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert route.call_count == 1  # both cells fetched in a single batched request
    assert df.height == 2
    assert dict(zip(df["cell"].to_list(), df["value"].to_list())) == {cells[0].h3: 3.0, cells[1].h3: 9.0}


@respx.mock
def test_forecast_adapter_hits_the_forecast_endpoint_not_the_archive():
    fc = respx.get(FORECAST).mock(
        return_value=httpx.Response(200, json={"hourly": {"time": ["2026-06-15T00:00"], "temperature_2m": [12.0]}})
    )
    arch = respx.get(ARCHIVE).mock(return_value=httpx.Response(200, json={"hourly": {"time": [], "temperature_2m": []}}))
    cell = cell_of(51.5, -0.1, res=7)
    df = OpenMeteoForecastAdapter().fetch([cell], datetime(2026, 6, 15), datetime(2026, 6, 15))
    assert fc.call_count == 1 and arch.call_count == 0  # forecast NWP, not reanalysis
    assert df["value"].to_list() == [12.0]
