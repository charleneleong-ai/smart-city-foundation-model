"""Per-cell ground elevation from the Open-Meteo elevation API (free, no key) — the DEM the
fire CA's slope term needs. Elevation is static (no time axis), so this is a plain dict keyed
by H3 id, not a time-series LayerAdapter."""

import time

import httpx

from sctwin.geo import Cell, center_of

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
_BATCH = 100  # Open-Meteo accepts up to 100 comma-separated coordinates per request
_RETRIES = 4  # free tier rate-limits by location count (429) — back off and retry
_CACHE: dict[frozenset[str], dict[str, float]] = {}  # elevation is static — memo per cell set


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
    """Map each cell's centroid to its ground elevation in metres. Cached per cell set so repeated
    calls (e.g. the live feed server recomputing under different fire params) don't re-hit the API."""
    key = frozenset(c.h3 for c in cells)
    if key in _CACHE:
        return _CACHE[key]
    client = client or httpx.Client(timeout=60.0)
    out: dict[str, float] = {}
    for i in range(0, len(cells), _BATCH):
        batch = cells[i : i + _BATCH]
        centers = [center_of(c) for c in batch]
        resp = _get(client, {
            "latitude": ",".join(f"{lat:.4f}" for lat, _ in centers),
            "longitude": ",".join(f"{lon:.4f}" for _, lon in centers),
        })
        out.update({c.h3: float(e) for c, e in zip(batch, resp.json()["elevation"], strict=True)})
    _CACHE[key] = out
    return out
