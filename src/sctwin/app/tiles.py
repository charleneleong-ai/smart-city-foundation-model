import math
from datetime import datetime, timezone

from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.geo import boundary_ring, center_of, haversine_km, validate_res
from sctwin.registry import Registry


def _bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def tile_records(
    registry: Registry,
    layer: str,
    *,
    lat: float,
    lon: float,
    radius_km: float,
    res: int,
    date: str,
    hour: int = 12,
    max_cells: int = 1000,
) -> list[dict]:
    """Records for the cells within radius_km of (lat, lon) at one timestamp — each a
    deck.gl-ready hexagon {cell, polygon, value, color}. Fetched through the registry
    (cached adapter), so panning the centre reuses prior downloads. Caps to the nearest
    `max_cells` so a single request stays bounded."""
    validate_res(res)
    south, west, north, east = _bbox(lat, lon, radius_km)
    cells = [
        c for c in cells_in_bbox(south, west, north, east, res)
        if haversine_km(lat, lon, *center_of(c)) <= radius_km
    ]
    if not cells:
        return []
    if len(cells) > max_cells:
        cells = sorted(cells, key=lambda c: haversine_km(lat, lon, *center_of(c)))[:max_cells]

    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    frame = registry.get(layer, cells, day, day)
    recs = h3_layer_records(frame, at=day.replace(hour=hour))
    by_h3 = {c.h3: c for c in cells}
    return [{**r, "polygon": boundary_ring(by_h3[r["cell"]])} for r in recs]
