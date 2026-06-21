"""Per-cell ground elevation from the Open-Meteo elevation API (free, no key) — the DEM the
fire CA's slope term needs. Elevation is static (no time axis), so this is a plain dict keyed
by H3 id, not a time-series LayerAdapter."""

import json
import time
from pathlib import Path

import httpx

from sctwin.geo import Cell, center_of

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
_BATCH = 100  # Open-Meteo accepts up to 100 comma-separated coordinates per request
_RETRIES = 4  # free tier rate-limits by location count (429) — back off and retry
_CACHE_FILE = Path(".cache/elevation.json")  # per-cell disk cache; elevation is static so this never staleness-expires
_ELEV: dict[str, float] | None = None  # lazy-loaded in-memory mirror of the disk cache


def _cache() -> dict[str, float]:
    global _ELEV
    if _ELEV is None:
        try:
            _ELEV = json.loads(_CACHE_FILE.read_text())
        except (OSError, ValueError):
            _ELEV = {}
    return _ELEV


def _get(client: httpx.Client, params: dict) -> httpx.Response:
    for attempt in range(_RETRIES):
        resp = client.get(ELEVATION_URL, params=params)
        if resp.status_code == 429 and attempt < _RETRIES - 1:
            time.sleep(2**attempt)  # 1, 2, 4 s backoff
            continue
        resp.raise_for_status()
        return resp
    return resp


def fetch_elevation(cells: list[Cell], client: httpx.Client | None = None) -> dict[str, float]:
    """Map each cell's centroid to its ground elevation in metres. Per-cell disk cache (.cache/
    elevation.json) — only ever fetches a cell once, so restarts and overlapping bboxes never re-hit
    the rate-limited API."""
    cache = _cache()
    missing = [c for c in cells if c.h3 not in cache]
    if missing:
        client = client or httpx.Client(timeout=60.0)
        for i in range(0, len(missing), _BATCH):
            batch = missing[i : i + _BATCH]
            centers = [center_of(c) for c in batch]
            resp = _get(client, {
                "latitude": ",".join(f"{lat:.4f}" for lat, _ in centers),
                "longitude": ",".join(f"{lon:.4f}" for _, lon in centers),
            })
            cache.update({c.h3: float(e) for c, e in zip(batch, resp.json()["elevation"], strict=True)})
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache))
    return {c.h3: cache[c.h3] for c in cells}
