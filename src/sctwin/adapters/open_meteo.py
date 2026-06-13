import time
from datetime import datetime

import httpx
import polars as pl

from sctwin.geo import Cell, center_of
from sctwin.schema import empty_frame, validate_frame

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_BATCH = 100  # Open-Meteo accepts many comma-separated coordinates per request
_RETRIES = 4  # the free tier rate-limits by location count (429) — back off and retry


class OpenMeteoWeatherAdapter:
    name = "weather.t2m"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=60.0)

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
            "hourly": "temperature_2m",
        }
        for attempt in range(_RETRIES):
            resp = self._client.get(ARCHIVE_URL, params=params)
            if resp.status_code == 429 and attempt < _RETRIES - 1:
                time.sleep(2**attempt)  # 1, 2, 4 s backoff
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
                    "layer": "t2m",
                    "value": r["hourly"]["temperature_2m"],
                }
            )
            for cell, r in zip(cells, results, strict=True)
        ]
        return pl.concat(frames)
