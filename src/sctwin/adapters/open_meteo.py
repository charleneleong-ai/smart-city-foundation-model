import time
from datetime import datetime

import httpx
import polars as pl

from sctwin.geo import Cell, center_of
from sctwin.schema import empty_frame, validate_frame

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"  # reanalysis: past, decades back
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"  # real NWP: recent past + up to 16 days ahead
_BATCH = 100  # Open-Meteo accepts many comma-separated coordinates per request
_RETRIES = 6  # the free tier rate-limits by location count (429) — back off and retry
_BACKOFF_CAP = 60.0  # cap a single backoff so a sustained 429 can't sleep unboundedly


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying a 429: honour the server's Retry-After header when present,
    else exponential backoff (1, 2, 4, 8, 16). Both are capped at _BACKOFF_CAP (binds only on a
    large Retry-After; plain backoff tops out at 16 s within _RETRIES)."""
    after = resp.headers.get("Retry-After", "")
    if after.replace(".", "", 1).isdigit():  # numeric seconds form (Open-Meteo doesn't send HTTP-date)
        return min(float(after), _BACKOFF_CAP)
    return min(2.0**attempt, _BACKOFF_CAP)

# canonical layer -> Open-Meteo `hourly` variable. The fire-weather covariates a spread / Fire
# Weather Index model needs: temperature, 10 m wind speed + direction, precipitation, humidity.
# NB `wind_dir` is degrees [0, 360) — a circular quantity; consumers must not linearly average it.
WEATHER_VARS: dict[str, str] = {
    "t2m": "temperature_2m",
    "wind_speed": "wind_speed_10m",
    "wind_dir": "wind_direction_10m",
    "precip": "precipitation",
    "rh": "relative_humidity_2m",
}
_DEFAULT_VARS: dict[str, str] = {"t2m": "temperature_2m"}


class OpenMeteoWeatherAdapter:
    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        url: str = ARCHIVE_URL,
        variables: dict[str, str] | None = None,
    ) -> None:
        self._client = client or httpx.Client(timeout=60.0)
        self._url = url
        self._variables = variables or _DEFAULT_VARS
        # one var -> "weather.<layer>" (back-compat with the t2m-only adapter); many -> "weather"
        self.name = f"weather.{next(iter(self._variables))}" if len(self._variables) == 1 else "weather"

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        frames = [
            self._fetch_batch(cells[i : i + _BATCH], start, end)
            for i in range(0, len(cells), _BATCH)
        ]
        return validate_frame(pl.concat(frames)) if frames else empty_frame()

    def _fetch_batch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        centers = [center_of(c) for c in cells]
        params = {
            "latitude": ",".join(f"{lat:.4f}" for lat, _ in centers),
            "longitude": ",".join(f"{lon:.4f}" for _, lon in centers),
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": ",".join(self._variables.values()),
        }
        for attempt in range(_RETRIES):
            resp = self._client.get(self._url, params=params)
            if resp.status_code == 429 and attempt < _RETRIES - 1:
                time.sleep(_retry_delay(resp, attempt))
                continue
            resp.raise_for_status()
            break
        payload = resp.json()
        results = payload if isinstance(payload, list) else [payload]  # array iff >1 location
        frames = [
            pl.DataFrame(
                {
                    "cell": cell.h3,
                    "time": pl.Series(r["hourly"]["time"]).str.to_datetime(time_zone="UTC"),
                    "layer": layer,
                    # cast to canonical Float64 so a var the API returns as int (e.g. humidity)
                    # doesn't break the concat against float vars like temperature
                    "value": pl.Series(r["hourly"][omv], dtype=pl.Float64),
                }
            )
            for cell, r in zip(cells, results, strict=True)
            for layer, omv in self._variables.items()
        ]
        return pl.concat(frames)


def OpenMeteoForecastAdapter(
    client: httpx.Client | None = None, *, variables: dict[str, str] | None = None
) -> OpenMeteoWeatherAdapter:
    """Open-Meteo *forecast* endpoint — real NWP (recent past + up to 16 days ahead), the
    future-weather covariate for the demand cascade and the fire-spread model. Same shape as
    the archive adapter; only valid for near-now dates (the archive serves arbitrary history)."""
    return OpenMeteoWeatherAdapter(client, url=FORECAST_URL, variables=variables)
