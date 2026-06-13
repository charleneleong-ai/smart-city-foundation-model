import math
from dataclasses import dataclass

import h3


@dataclass(frozen=True)
class Cell:
    h3: str
    res: int


def validate_res(res: int) -> None:
    if not 0 <= res <= 15:
        raise ValueError(f"H3 resolution must be 0..15, got {res}")


def cell_of(lat: float, lon: float, res: int) -> Cell:
    validate_res(res)
    return Cell(h3=h3.latlng_to_cell(lat, lon, res), res=res)


def center_of(cell: Cell) -> tuple[float, float]:
    return h3.cell_to_latlng(cell.h3)


def boundary_ring(cell: Cell) -> list[list[float]]:
    """Closed [lng, lat] ring of the cell's hexagon (for GeoJSON / deck.gl polygons)."""
    ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(cell.h3)]
    ring.append(ring[0])
    return ring


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))
