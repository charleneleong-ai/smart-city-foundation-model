from datetime import datetime, timezone

import httpx
import polars as pl
import respx

from sctwin.adapters.open_meteo import (
    WEATHER_VARS,
    OpenMeteoForecastAdapter,
    OpenMeteoWeatherAdapter,
)
from sctwin.geo import cell_of

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
LA = (34.05, -118.24)  # Los Angeles — the fire-weather use case


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


def test_name_reflects_variable_set():
    assert OpenMeteoWeatherAdapter().name == "weather.t2m"  # default single var, back-compat
    assert OpenMeteoWeatherAdapter(variables=WEATHER_VARS).name == "weather"


@respx.mock
def test_fetch_emits_one_layer_per_requested_variable():
    respx.get(ARCHIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "hourly": {
                    "time": ["2025-01-07T00:00", "2025-01-07T01:00"],
                    "temperature_2m": [12.0, 13.0],
                    "wind_speed_10m": [40.0, 55.0],
                    "wind_direction_10m": [70.0, 65.0],  # NE Santa Ana flow
                    "precipitation": [0.0, 0.0],
                    "relative_humidity_2m": [15, 12],  # API may return ints -> must still concat
                }
            },
        )
    )
    cell = cell_of(*LA, res=7)
    df = OpenMeteoWeatherAdapter(variables=WEATHER_VARS).fetch([cell], datetime(2025, 1, 7), datetime(2025, 1, 7))
    assert set(df["layer"].unique().to_list()) == set(WEATHER_VARS)
    assert df.height == 2 * len(WEATHER_VARS)  # 2 timesteps x 5 variables
    wind_dir = df.filter(pl.col("layer") == "wind_dir").sort("time")["value"].to_list()
    assert wind_dir == [70.0, 65.0]
    assert df["value"].dtype == pl.Float64  # int humidity coerced, no concat dtype clash


@respx.mock
def test_request_asks_for_all_hourly_variables():
    route = respx.get(ARCHIVE).mock(
        return_value=httpx.Response(
            200,
            json={"hourly": {"time": ["2025-01-07T00:00"], **{v: [0.0] for v in WEATHER_VARS.values()}}},
        )
    )
    OpenMeteoWeatherAdapter(variables=WEATHER_VARS).fetch([cell_of(*LA, res=7)], datetime(2025, 1, 7), datetime(2025, 1, 7))
    hourly = route.calls.last.request.url.params["hourly"]
    assert all(omv in hourly for omv in WEATHER_VARS.values())


@respx.mock
def test_forecast_adapter_carries_fire_variables():
    fc = respx.get(FORECAST).mock(
        return_value=httpx.Response(200, json={"hourly": {"time": ["2026-06-20T00:00"], "wind_direction_10m": [90.0]}})
    )
    df = OpenMeteoForecastAdapter(variables={"wind_dir": "wind_direction_10m"}).fetch(
        [cell_of(*LA, res=7)], datetime(2026, 6, 20), datetime(2026, 6, 20)
    )
    assert fc.call_count == 1
    assert df["layer"].unique().to_list() == ["wind_dir"]
    assert df["value"].to_list() == [90.0]
