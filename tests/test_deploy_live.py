from datetime import datetime

import httpx
import respx

from sctwin.adapters.open_meteo import OpenMeteoForecastAdapter, WEATHER_VARS
from sctwin.deploy.hazard import FireScenario
from sctwin.geo import cell_of

FORECAST = "https://api.open-meteo.com/v1/forecast"


@respx.mock
def test_from_live_pulls_wind_and_temp_from_adapter():
    respx.get(FORECAST).mock(
        return_value=httpx.Response(
            200,
            json={
                "hourly": {
                    "time": ["2026-06-20T12:00"],
                    "temperature_2m": [36.0],
                    "wind_speed_10m": [11.0],
                    "wind_direction_10m": [70.0],
                    "precipitation": [0.0],
                    "relative_humidity_2m": [18.0],
                }
            },
        )
    )
    cell = cell_of(34.05, -118.24, res=7)
    scn = FireScenario.from_live(
        cell.h3, res=7, fire_type="grass", size=4.0, duration_min=180.0, pm25=120.0,
        when=datetime(2026, 6, 20), adapter=OpenMeteoForecastAdapter(variables=WEATHER_VARS),
    )
    assert scn.temp_c == 36.0
    assert scn.wind_speed == 11.0
    assert scn.wind_dir == 70.0
    assert scn.cell == cell.h3 and scn.pm25 == 120.0
