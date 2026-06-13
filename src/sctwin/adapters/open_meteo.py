from datetime import datetime

import httpx
import polars as pl

from sctwin.geo import Cell, center_of
from sctwin.schema import empty_frame, validate_frame

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoWeatherAdapter:
    name = "weather.t2m"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0)

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        frames = [self._fetch_one(cell, start, end) for cell in cells]
        return validate_frame(pl.concat(frames)) if frames else empty_frame()

    def _fetch_one(self, cell: Cell, start: datetime, end: datetime) -> pl.DataFrame:
        lat, lon = center_of(cell)
        resp = self._client.get(
            ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "hourly": "temperature_2m",
            },
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        return pl.DataFrame(
            {
                "cell": cell.h3,
                "time": pl.Series(hourly["time"]).str.to_datetime(time_zone="UTC"),
                "layer": "t2m",
                "value": hourly["temperature_2m"],
            }
        )
